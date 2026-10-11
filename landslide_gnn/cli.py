"""Command-line interface:  python -m landslide_gnn <command> [options]

Commands
  validate   check that the CSV files follow docs/DATA_FORMAT.md
  train      train, validate (early stopping) and test a model
  evaluate   evaluate a saved checkpoint on labelled data
  predict    produce alerts for (unlabelled) data with a saved checkpoint
  graph      export the graph built from stations.csv (for inspection)
  synth      write a synthetic dataset for smoke-testing the pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config


def _common_data(p):
    p.add_argument("--stations", help="path to stations.csv (overrides config)")
    p.add_argument("--observations", help="path to observations.csv (overrides config)")


def _cfg_from_args(a) -> Config:
    cfg = Config.load(a.config)
    if getattr(a, "stations", None):
        cfg.data.stations_csv = a.stations
    if getattr(a, "observations", None):
        cfg.data.observations_csv = a.observations
    if getattr(a, "output", None):
        cfg.output_dir = a.output
    cfg.override(getattr(a, "set", None))
    return cfg


def cmd_validate(a) -> int:
    from .data import DataFormatError, build_events, prepare

    cfg = _cfg_from_args(a)
    try:
        prep = prepare(cfg, require_labels=not a.unlabeled)
    except DataFormatError as e:
        print(f"INVALID: {e}")
        return 1
    print("OK: files follow the expected format.")
    for g in prep.graphs.values():
        print("  graph:", g.summary())
    ev = prep.events
    print(f"  events: {len(ev)} | timesteps: {sum(len(e.times) for e in ev)}")
    import numpy as np
    labs = np.concatenate([e.y.ravel() for e in ev])
    print("  label counts (Normal, Warning, Evacuate, Landslide):",
          np.bincount(labs[labs >= 0], minlength=4).tolist(), f"| unknown: {(labs < 0).sum()}")
    splits = {}
    for e in ev:
        splits[e.split] = splits.get(e.split, 0) + 1
    print("  events per split:", splits)
    for w in prep.warnings:
        print("  WARNING:", w)
    return 0


def cmd_train(a) -> int:
    from .engine import train

    cfg = _cfg_from_args(a)
    res = train(cfg)
    print(json.dumps({k: v for k, v in res.items() if k != "test"}, indent=2))
    if "test" in res:
        t = res["test"]
        print("test (decision layer):", json.dumps({k: t["decision_layer"].get(k) for k in
                                                     ("macro_f1", "balanced_accuracy", "mcc", "under_escalation_rate")}))
        print("test early warning:", json.dumps(t["early_warning"]))
    print(f"outputs written to {cfg.output_dir}")
    return 0


def cmd_evaluate(a) -> int:
    from .engine import evaluate_checkpoint

    split = None if a.split == "all" else a.split
    res = evaluate_checkpoint(a.checkpoint, a.stations, a.observations, a.output, split=split,
                              device=a.device, overrides=a.set)
    print(json.dumps({"decision_layer": {k: res["decision_layer"].get(k) for k in
                                         ("macro_f1", "balanced_accuracy", "mcc", "under_escalation_rate")},
                      "early_warning": res["early_warning"]}, indent=2))
    print(f"outputs written to {a.output}")
    return 0


def cmd_predict(a) -> int:
    from .engine import predict_checkpoint

    pred = predict_checkpoint(a.checkpoint, a.stations, a.observations, a.output, device=a.device)
    print(f"{len(pred)} predictions written to {a.output}")
    print(pred["alert_name"].value_counts().to_string())
    return 0


def cmd_graph(a) -> int:
    import pandas as pd

    from .data import load_stations
    from .graph import EDGE_FEATURES, build_site_graph

    cfg = _cfg_from_args(a)
    st = load_stations(cfg.data.stations_csv)
    rows = []
    for site, g in st.groupby("site_id"):
        G = build_site_graph(g, cfg.graph)
        print(G.summary())
        for e in range(G.edge_index.shape[1]):
            s, d = G.edge_index[:, e]
            rows.append([site, G.station_ids[s], G.station_ids[d], *G.edge_attr[e].tolist()])
    pd.DataFrame(rows, columns=["site_id", "source", "target", *EDGE_FEATURES]).to_csv(a.output, index=False)
    print(f"edge list written to {a.output}")
    return 0


def cmd_synth(a) -> int:
    from .synthetic import generate

    sp, op = generate(a.output, n_sites=a.sites, n_events=a.events, hours=a.hours,
                      dt_min=a.dt_min, seed=a.seed)
    print(f"wrote {sp} and {op}  (SYNTHETIC -- for pipeline testing only)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="landslide_gnn", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="check CSV files")
    p.add_argument("--config", default=None)
    _common_data(p)
    p.add_argument("--unlabeled", action="store_true", help="do not require the label column")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.window=30")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("train", help="train + validate + test")
    p.add_argument("--config", default="configs/default.yaml")
    _common_data(p)
    p.add_argument("--output", help="output directory (overrides config)")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. model.type=gcn train.epochs=50")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stations", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--split", default="test", help="train | val | test | all")
    p.add_argument("--output", default="runs/eval")
    p.add_argument("--device", default="auto")
    p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. decision.tau=0.2")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("predict", help="predict alerts with a checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stations", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--output", default="runs/predictions.csv")
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("graph", help="export the station graph as an edge list")
    p.add_argument("--config", default=None)
    p.add_argument("--stations", required=True)
    p.add_argument("--output", default="graph_edges.csv")
    p.add_argument("--set", nargs="*", default=[])
    p.set_defaults(fn=cmd_graph)

    p = sub.add_parser("synth", help="generate a synthetic dataset (testing only)")
    p.add_argument("--output", default="data/synthetic")
    p.add_argument("--sites", type=int, default=2)
    p.add_argument("--events", type=int, default=40)
    p.add_argument("--hours", type=float, default=72.0)
    p.add_argument("--dt-min", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_synth)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
