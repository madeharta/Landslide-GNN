"""Training, evaluation and inference loops."""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import __version__
from .config import Config
from .data import (Prepared, SnapshotDataset, apply_normalizer, assign_splits, collate_snapshots,
                   feature_normalizer, prepare)
from .decision import decide_sequence
from .graph import edge_normalizer
from .losses import FocalLoss, class_weights
from .metrics import classification_metrics, early_warning_metrics
from .models import LandslideGNN, count_parameters
from .schema import CLASS_NAMES, FEATURE_COLUMNS

log = logging.getLogger("landslide_gnn")


# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _to(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _loader(ds, cfg: Config, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle,
                      collate_fn=collate_snapshots, num_workers=cfg.train.num_workers)


def _split(events, name):
    return [e for e in events if e.split == name and len(e.times) >= 1]


# ---------------------------------------------------------------------------
# Inference over a list of events -> long predictions table
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_events(model, events, graphs, edge_norm, cfg: Config, device, attention: bool = False):
    """Run the model on every window (stride 1) of ``events``.

    Returns a long table with one row per (event, time, station) holding the
    class probabilities, the raw argmax, and the decision-layer alert level.
    If ``attention`` is True, also returns the mean attention per graph edge.
    """
    model.eval()
    ds = SnapshotDataset(events, graphs, edge_norm, cfg.data.window, stride=1,
                         min_obs_frac=cfg.data.min_obs_frac, require_label=False)
    site_list = sorted(graphs)
    site_pos = {s: i for i, s in enumerate(site_list)}
    att_sum = {s: None for s in site_list}
    att_cnt = {s: None for s in site_list}
    chunks = []
    for batch in _loader(ds, cfg, shuffle=False):
        b = _to(batch, device)
        logits, attn = model(b["x"], b["edge_index"], b["edge_attr"], return_attention=True)
        prob = torch.softmax(logits, -1).cpu().numpy()
        n_tot = prob.shape[0]
        ev_idx = np.empty(n_tot, dtype=np.int64)
        t_idx = np.empty(n_tot, dtype=np.int64)
        st_idx = np.empty(n_tot, dtype=np.int64)
        for ev_i, t, off, n in batch["meta"]:
            ev_idx[off:off + n], t_idx[off:off + n], st_idx[off:off + n] = ev_i, t, np.arange(n)
        chunks.append((ev_idx, t_idx, st_idx, batch["y"].numpy(), batch["active"].numpy(), prob))
        if attention and attn and attn[0] is not None:
            node_site = np.array([site_pos[events[e].site_id] for e in ev_idx])
            dst = batch["edge_index"][1].numpy()
            eid = batch["edge_id"].numpy()
            esite = node_site[dst]
            for li, a in enumerate(attn):
                a = a.mean(-1).cpu().numpy()          # average over heads
                for si in np.unique(esite):
                    s = site_list[si]
                    if att_sum[s] is None:
                        E = graphs[s].edge_index.shape[1]
                        att_sum[s] = np.zeros((len(attn), E))
                        att_cnt[s] = np.zeros((len(attn), E))
                    m = esite == si
                    np.add.at(att_sum[s][li], eid[m], a[m])
                    np.add.at(att_cnt[s][li], eid[m], 1)
    ev_idx, t_idx, st_idx, y, act, prob = (np.concatenate(c) for c in zip(*chunks))
    pcols = [f"p_{c.lower()}" for c in CLASS_NAMES]
    pred = pd.DataFrame({
        "site_id": [events[e].site_id for e in ev_idx],
        "event_id": [events[e].event_id for e in ev_idx],
        "t_s": [float(events[e].times[t]) for e, t in zip(ev_idx, t_idx)],
        "station_id": [graphs[events[e].site_id].station_ids[k] for e, k in zip(ev_idx, st_idx)],
        "label": y, "active": act,
    })
    for c, col in enumerate(pcols):
        pred[col] = prob[:, c]
    pred["pred_argmax"] = prob.argmax(1)
    pred = pred.sort_values(["site_id", "event_id", "station_id", "t_s"]).reset_index(drop=True)
    alerts = np.zeros(len(pred), dtype=np.int64)
    P = pred[pcols].to_numpy()
    for _, idx in pred.groupby(["site_id", "event_id", "station_id"]).indices.items():
        alerts[idx] = decide_sequence(P[idx], cfg.decision)
    pred["alert"] = alerts
    att = None
    if attention and any(v is not None for v in att_sum.values()):
        recs = []
        for s in site_list:
            if att_sum[s] is None:
                continue
            g = graphs[s]
            for li in range(att_sum[s].shape[0]):
                for e_id in np.nonzero(att_cnt[s][li])[0]:
                    s_i, d_i = g.edge_index[:, e_id]
                    recs.append((s, li + 1, g.station_ids[s_i], g.station_ids[d_i], int(g.edge_attr[e_id, 4] > 0.5),
                                 float(g.edge_attr[e_id, 0]), att_sum[s][li, e_id] / att_cnt[s][li, e_id]))
        att = pd.DataFrame(recs, columns=["site_id", "layer", "source", "target", "is_flow", "distance_m", "mean_alpha"])
    return pred, att


def summarize(pred: pd.DataFrame, cfg: Config) -> dict:
    y = pred["label"].to_numpy()
    return {
        "raw_argmax": classification_metrics(y, pred["pred_argmax"].to_numpy()),
        "decision_layer": classification_metrics(y, pred["alert"].to_numpy()),
        "early_warning": early_warning_metrics(pred, cfg.decision.horizon_s, "alert"),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_model(cfg: Config) -> LandslideGNN:
    return LandslideGNN(cfg.model)


def train(cfg: Config) -> dict:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    cfg.save(out / "config.yaml")

    prep: Prepared = prepare(cfg)
    for w in prep.warnings:
        log.warning(w)
    assign_splits(prep.events, cfg.data.split_mode, cfg.data.split_ratios, cfg.train.seed)
    pd.DataFrame([(e.site_id, e.event_id, e.split, len(e.times)) for e in prep.events],
                 columns=["site_id", "event_id", "split", "timesteps"]).to_csv(out / "splits.csv", index=False)
    fnorm = feature_normalizer(prep.events)
    apply_normalizer(prep.events, fnorm)
    enorm = edge_normalizer(list(prep.graphs.values()))
    for g in prep.graphs.values():
        log.info("graph %s", g.summary())

    tr_ev, va_ev, te_ev = (_split(prep.events, s) for s in ("train", "val", "test"))
    if not tr_ev:
        raise RuntimeError("no training events -- check the 'split' column or split_ratios")
    ds_tr = SnapshotDataset(tr_ev, prep.graphs, enorm, cfg.data.window, cfg.data.stride, cfg.data.min_obs_frac)
    if len(ds_tr) == 0:
        raise RuntimeError("no labelled training windows (events shorter than data.window?)")
    tr_labels = np.concatenate([e.y[cfg.data.window - 1:].ravel() for e in tr_ev])
    w = class_weights(tr_labels, cfg.loss.class_weighting, cfg.loss.beta)
    log.info("training windows=%d | class counts=%s | weights=%s", len(ds_tr),
             np.bincount(tr_labels[tr_labels >= 0], minlength=4).tolist(), np.round(w, 3).tolist())

    model = build_model(cfg).to(device)
    log.info("model=%s parameters=%d device=%s", cfg.model.type, count_parameters(model), device)
    crit = FocalLoss(torch.tensor(w), cfg.loss.focal_gamma).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    meta = {"version": __version__, "feature_columns": FEATURE_COLUMNS, "class_names": CLASS_NAMES,
            "feature_norm": fnorm, "edge_norm": enorm, "class_weights": w.tolist(), "config": cfg.to_dict()}
    best, best_ep, wait, history = -1.0, -1, 0, []
    loader = _loader(ds_tr, cfg, shuffle=True)
    for ep in range(1, cfg.train.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch in loader:
            b = _to(batch, device)
            logits = model(b["x"], b["edge_index"], b["edge_attr"])
            loss = crit(logits, b["y"], b["active"])
            opt.zero_grad()
            loss.backward()
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "seconds": time.time() - t0}
        sel_events = va_ev or tr_ev
        pred, _ = predict_events(model, sel_events, prep.graphs, enorm, cfg, device)
        m = classification_metrics(pred["label"].to_numpy(), pred["pred_argmax"].to_numpy())
        rec.update({"val_macro_f1": m.get("macro_f1", 0.0), "val_balanced_acc": m.get("balanced_accuracy", 0.0)})
        history.append(rec)
        log.info("epoch %3d  loss %.4f  val macroF1 %.4f  balAcc %.4f  (%.1fs)", ep, rec["train_loss"],
                 rec["val_macro_f1"], rec["val_balanced_acc"], rec["seconds"])
        if rec["val_macro_f1"] > best:
            best, best_ep, wait = rec["val_macro_f1"], ep, 0
            torch.save({"model_state": model.state_dict(), **meta, "epoch": ep}, out / "best.pt")
        else:
            wait += 1
            if wait >= cfg.train.patience:
                log.info("early stopping at epoch %d (best %d)", ep, best_ep)
                break
    pd.DataFrame(history).to_csv(out / "train_log.csv", index=False)
    torch.save({"model_state": model.state_dict(), **meta, "epoch": ep}, out / "last.pt")

    result = {"best_epoch": best_ep, "best_val_macro_f1": best}
    if te_ev:
        ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        result["test"] = evaluate_events(model, te_ev, prep.graphs, enorm, cfg, device, out, tag="test")
    with open(out / "summary.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def evaluate_events(model, events, graphs, enorm, cfg, device, out: Path, tag: str, attention=True) -> dict:
    pred, att = predict_events(model, events, graphs, enorm, cfg, device, attention=attention)
    pred.to_csv(out / f"predictions_{tag}.csv", index=False)
    if att is not None:
        att.to_csv(out / f"attention_{tag}.csv", index=False)
    summ = summarize(pred, cfg)
    cm = summ["decision_layer"].get("confusion_matrix")
    if cm:
        pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_NAMES],
                     columns=[f"pred_{c}" for c in CLASS_NAMES]).to_csv(out / f"confusion_{tag}.csv")
    with open(out / f"metrics_{tag}.json", "w") as f:
        json.dump(summ, f, indent=2)
    return summ


# ---------------------------------------------------------------------------
# Evaluation / prediction from a checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint(path: str, device=None):
    device = device or torch.device("cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ck["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    return model, cfg, ck


def evaluate_checkpoint(checkpoint: str, stations_csv: str, observations_csv: str, out_dir: str,
                        split: str | None = "test", device: str = "auto", overrides=None) -> dict:
    dev = resolve_device(device)
    model, cfg, ck = load_checkpoint(checkpoint, dev)
    cfg.data.stations_csv, cfg.data.observations_csv = stations_csv, observations_csv
    cfg.override(overrides)
    prep = prepare(cfg)
    for w in prep.warnings:
        log.warning(w)
    if split and all(e.split for e in prep.events):
        events = [e for e in prep.events if e.split == split]
    else:
        events = prep.events          # no split column -> evaluate every event
    if not events:
        raise RuntimeError(f"no events with split='{split}'")
    apply_normalizer(events, ck["feature_norm"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return evaluate_events(model, events, prep.graphs, ck["edge_norm"], cfg, dev, out, tag=split or "all")


def predict_checkpoint(checkpoint: str, stations_csv: str, observations_csv: str, out_csv: str,
                       device: str = "auto") -> pd.DataFrame:
    dev = resolve_device(device)
    model, cfg, ck = load_checkpoint(checkpoint, dev)
    cfg.data.stations_csv, cfg.data.observations_csv = stations_csv, observations_csv
    prep = prepare(cfg, require_labels=False)
    apply_normalizer(prep.events, ck["feature_norm"])
    pred, _ = predict_events(model, prep.events, prep.graphs, ck["edge_norm"], cfg, dev)
    pred["alert_name"] = [CLASS_NAMES[a] for a in pred["alert"]]
    pred["inferred_from_neighbors"] = ~pred["active"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_csv, index=False)
    return pred
