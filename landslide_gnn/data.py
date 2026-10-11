"""CSV loading, validation, windowing, splitting and batching."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import schema
from .config import Config
from .graph import SiteGraph, build_site_graph, normalize_edges


class DataFormatError(ValueError):
    """Raised when a CSV file does not follow docs/DATA_FORMAT.md."""


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def _add_optional(df: pd.DataFrame, optional: dict) -> pd.DataFrame:
    for col, default in optional.items():
        if col not in df.columns:
            df[col] = default
    return df


def load_stations(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station_id": str, "site_id": str, "geo_unit": str})
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in schema.STATION_REQUIRED if c not in df.columns]
    if missing:
        raise DataFormatError(f"{path}: missing required column(s) {missing}")
    df = _add_optional(df, schema.STATION_OPTIONAL)
    df["site_id"] = df["site_id"].fillna("site_0").astype(str)
    df["station_id"] = df["station_id"].astype(str)
    for c in ["x_m", "y_m", "z_m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["x_m", "y_m", "z_m"]].isna().any().any():
        raise DataFormatError(f"{path}: x_m, y_m, z_m must be numeric and non-empty")
    dup = df.duplicated(["site_id", "station_id"])
    if dup.any():
        raise DataFormatError(f"{path}: duplicated (site_id, station_id): "
                              f"{df.loc[dup, ['site_id', 'station_id']].values.tolist()[:5]}")
    df["downslope_az_deg"] = pd.to_numeric(df["downslope_az_deg"], errors="coerce")
    return df


def _parse_time(col: pd.Series) -> np.ndarray:
    """Return timestamps as float seconds (ISO-8601 strings or numeric seconds)."""
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().all():
        return num.to_numpy(dtype=float)
    ts = pd.to_datetime(col, errors="coerce", utc=True)
    if ts.isna().any():
        bad = col[ts.isna()].iloc[:3].tolist()
        raise DataFormatError(f"unparseable timestamp(s), e.g. {bad}")
    return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def load_observations(path: str, require_labels: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station_id": str, "site_id": str, "event_id": str, "split": str})
    df.columns = [c.strip() for c in df.columns]
    need = schema.KEY_COLUMNS + schema.FEATURE_COLUMNS + ([schema.LABEL_COLUMN] if require_labels else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise DataFormatError(f"{path}: missing required column(s) {missing}")
    df = _add_optional(df, schema.OBS_OPTIONAL)
    df["site_id"] = df["site_id"].fillna("site_0").astype(str)
    df["station_id"] = df["station_id"].astype(str)
    df["event_id"] = df["event_id"].astype(str)
    df["t_s"] = _parse_time(df["timestamp"])
    for c in schema.FEATURE_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    lab = pd.to_numeric(df[schema.LABEL_COLUMN], errors="coerce").fillna(schema.UNLABELED).astype(int)
    bad = ~lab.isin([schema.UNLABELED] + list(range(schema.NUM_CLASSES)))
    if bad.any():
        raise DataFormatError(f"{path}: label must be 0..3 or empty/-1, found {sorted(lab[bad].unique())[:5]}")
    df[schema.LABEL_COLUMN] = lab
    dup = df.duplicated(["site_id", "event_id", "station_id", "t_s"])
    if dup.any():
        raise DataFormatError(f"{path}: {int(dup.sum())} duplicated (site, event, station, timestamp) rows")
    if df["split"].notna().any():
        s = df["split"].dropna().str.lower().str.strip()
        if not set(s.unique()) <= schema.VALID_SPLITS:
            raise DataFormatError(f"{path}: split must be one of {sorted(schema.VALID_SPLITS)}")
        df.loc[df["split"].notna(), "split"] = s
        per_event = df.dropna(subset=["split"]).groupby(["site_id", "event_id"])["split"].nunique()
        if (per_event > 1).any():
            raise DataFormatError(f"{path}: split must be constant within each event")
    return df


def check_consistency(stations: pd.DataFrame, obs: pd.DataFrame) -> list[str]:
    """Cross-file checks. Returns warnings; raises on hard errors."""
    warnings = []
    known = set(zip(stations.site_id, stations.station_id))
    used = set(zip(obs.site_id, obs.station_id))
    unknown = used - known
    if unknown:
        raise DataFormatError(f"observations reference stations not in stations.csv: {sorted(unknown)[:5]}")
    unused = known - used
    if unused:
        warnings.append(f"{len(unused)} station(s) have no observations (they stay inactive): {sorted(unused)[:5]}")
    nan_frac = obs[schema.FEATURE_COLUMNS].isna().mean()
    for c, v in nan_frac.items():
        if v > 0.2:
            warnings.append(f"feature '{c}' is missing in {v:.0%} of rows")
    return warnings


# ---------------------------------------------------------------------------
# Event tensors
# ---------------------------------------------------------------------------
@dataclass
class EventData:
    site_id: str
    event_id: str
    split: str | None
    times: np.ndarray        # [Tt] seconds
    x: np.ndarray            # [Tt, N, F] raw features (NaN where missing)
    avail: np.ndarray        # [Tt, N] bool, observed (after short-gap filling)
    y: np.ndarray            # [Tt, N] int labels (-1 unknown)
    xn: np.ndarray | None = field(default=None, repr=False)  # normalised, NaN -> 0


def build_events(obs: pd.DataFrame, graphs: dict[str, SiteGraph], max_gap: int) -> list[EventData]:
    events = []
    F = len(schema.FEATURE_COLUMNS)
    for (site, ev), g in obs.groupby(["site_id", "event_id"], sort=True):
        graph = graphs[site]
        sidx = {s: i for i, s in enumerate(graph.station_ids)}
        times = np.sort(g["t_s"].unique())
        tidx = np.searchsorted(times, g["t_s"].to_numpy())
        nidx = g["station_id"].map(sidx).to_numpy()
        Tt, N = len(times), graph.num_nodes
        x = np.full((Tt, N, F), np.nan, dtype=np.float32)
        avail = np.zeros((Tt, N), dtype=bool)
        y = np.full((Tt, N), schema.UNLABELED, dtype=np.int64)
        x[tidx, nidx] = g[schema.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        avail[tidx, nidx] = True
        y[tidx, nidx] = g[schema.LABEL_COLUMN].to_numpy()
        # forward-fill short gaps (features only; labels of filled steps stay unknown)
        if max_gap > 0:
            ar = np.arange(Tt)[:, None]
            last = np.maximum.accumulate(np.where(avail, ar, -1), axis=0)   # [Tt, N]
            fill = (~avail) & (last >= 0) & (ar - last <= max_gap)
            tt, nn = np.nonzero(fill)
            x[tt, nn] = x[last[tt, nn], nn]
            avail |= fill
        split = g["split"].dropna().iloc[0] if g["split"].notna().any() else None
        events.append(EventData(site, ev, split, times, x, avail, y))
    return events


def assign_splits(events: list[EventData], mode: str, ratios, seed: int) -> None:
    """Fill ``event.split`` (train / val / test) at the event level."""
    if mode == "auto":
        mode = "column" if all(e.split for e in events) else "random"
    if mode == "column":
        if not all(e.split for e in events):
            raise DataFormatError("split_mode=column but some events have no 'split' value")
        return
    if mode == "all":
        for e in events:
            e.split = "test"
        return
    if mode != "random":
        raise ValueError(f"unknown split_mode '{mode}'")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(events))
    r = np.asarray(ratios, dtype=float)
    r = r / r.sum()
    n = len(events)
    n_tr = max(1, int(round(r[0] * n)))
    n_va = max(1 if n >= 3 else 0, int(round(r[1] * n)))
    for rank, i in enumerate(order):
        events[i].split = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")


def feature_normalizer(events: list[EventData]) -> dict:
    rows = [e.x[e.avail] for e in events if e.split == "train"]
    rows = np.concatenate(rows) if rows else np.concatenate([e.x[e.avail] for e in events])
    mean = np.nanmean(rows, axis=0)
    std = np.nanstd(rows, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist()}


def apply_normalizer(events: list[EventData], norm: dict) -> None:
    mean = np.asarray(norm["mean"], np.float32)
    std = np.asarray(norm["std"], np.float32)
    for e in events:
        xn = (e.x - mean) / std
        xn[~e.avail] = 0.0
        e.xn = np.nan_to_num(xn, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Snapshot dataset
# ---------------------------------------------------------------------------
class SnapshotDataset(Dataset):
    """One item = one graph snapshot (all stations of a site at decision time t)."""

    def __init__(self, events: list[EventData], graphs: dict[str, SiteGraph], edge_norm: dict,
                 window: int, stride: int = 1, min_obs_frac: float = 0.5,
                 require_label: bool = True):
        self.events = events
        self.window = window
        self.min_obs_frac = min_obs_frac
        self.graph_tensors = {
            s: (torch.as_tensor(g.edge_index, dtype=torch.long),
                torch.as_tensor(normalize_edges(g.edge_attr, edge_norm)))
            for s, g in graphs.items()
        }
        self.index = []
        for ei, e in enumerate(events):
            for t in range(window - 1, len(e.times), stride):
                if require_label and not (e.y[t] >= 0).any():
                    continue
                self.index.append((ei, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> dict:
        ei, t = self.index[k]
        e = self.events[ei]
        sl = slice(t - self.window + 1, t + 1)
        x = torch.from_numpy(np.ascontiguousarray(e.xn[sl].transpose(1, 2, 0)))  # [N, F, T]
        frac = e.avail[sl].mean(axis=0)
        active = torch.from_numpy(e.avail[t] & (frac >= self.min_obs_frac))
        y = torch.from_numpy(e.y[t].copy())
        ei_t, ea_t = self.graph_tensors[e.site_id]
        return {"x": x, "y": y, "active": active, "edge_index": ei_t, "edge_attr": ea_t,
                "event": ei, "t": t}


def collate_snapshots(items: list[dict]) -> dict:
    """Merge snapshots into one block-diagonal graph.

    Edges whose source station is inactive are removed, so an offline station
    sends no messages (it can still receive them).
    """
    xs, ys, acts, eis, eas, eids, batch, meta = [], [], [], [], [], [], [], []
    off = 0
    for b, it in enumerate(items):
        n = it["x"].shape[0]
        ei = it["edge_index"]
        keep = it["active"][ei[0]]
        eis.append(ei[:, keep] + off)
        eas.append(it["edge_attr"][keep])
        eids.append(torch.nonzero(keep).squeeze(1))
        xs.append(it["x"])
        ys.append(it["y"])
        acts.append(it["active"])
        batch.append(torch.full((n,), b, dtype=torch.long))
        meta.append((it["event"], it["t"], off, n))
        off += n
    return {
        "x": torch.cat(xs), "y": torch.cat(ys), "active": torch.cat(acts),
        "edge_index": torch.cat(eis, dim=1), "edge_attr": torch.cat(eas),
        "edge_id": torch.cat(eids),
        "batch": torch.cat(batch), "meta": meta,
    }


# ---------------------------------------------------------------------------
# One-call preparation
# ---------------------------------------------------------------------------
@dataclass
class Prepared:
    stations: pd.DataFrame
    graphs: dict
    events: list
    warnings: list


def prepare(cfg: Config, require_labels: bool = True) -> Prepared:
    stations = load_stations(cfg.data.stations_csv)
    obs = load_observations(cfg.data.observations_csv, require_labels=require_labels)
    warnings = check_consistency(stations, obs)
    graphs = {site: build_site_graph(g, cfg.graph) for site, g in stations.groupby("site_id")}
    events = build_events(obs, graphs, cfg.data.max_gap)
    short = [e.event_id for e in events if len(e.times) < cfg.data.window]
    if short:
        warnings.append(f"{len(short)} event(s) shorter than the window ({cfg.data.window}) are skipped: {short[:5]}")
    return Prepared(stations, graphs, events, warnings)
