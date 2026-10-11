"""Collect metrics_test.json files into one table (mean ± std over seeds).

usage: python scripts/summarize_runs.py runs/exp [--out results.csv]
Expected layout: <root>/<experiment>/<seed>/metrics_test.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

KEYS = [
    ("decision_layer", "macro_f1"), ("decision_layer", "balanced_accuracy"), ("decision_layer", "mcc"),
    ("decision_layer", "under_escalation_rate"),
    ("early_warning", "detection_rate"), ("early_warning", "lead_time_median_h"),
    ("early_warning", "false_alarms_per_station_month"), ("early_warning", "alert_switches_per_station_day"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for f in sorted(Path(a.root).glob("*/*/metrics_test.json")):
        m = json.loads(f.read_text())
        r = {"experiment": f.parent.parent.name, "run": f.parent.name}
        for sec, k in KEYS:
            r[k] = m.get(sec, {}).get(k)
        r["evacuate_recall"] = m["decision_layer"]["per_class"]["Evacuate"]["recall"]
        rows.append(r)
    if not rows:
        raise SystemExit(f"no metrics_test.json under {a.root}")
    df = pd.DataFrame(rows)
    num = df.drop(columns=["run"])
    for c in num.columns[1:]:
        num[c] = pd.to_numeric(num[c], errors="coerce")
    g = num.groupby("experiment")
    mean, std = g.mean(numeric_only=True), g.std(numeric_only=True)
    fmt = lambda m, s: "-" if pd.isna(m) else (f"{m:.3f}" if pd.isna(s) else f"{m:.3f} ± {s:.3f}")
    table = pd.DataFrame({c: [fmt(m, s) for m, s in zip(mean[c], std[c])] for c in mean.columns}, index=mean.index)
    table.insert(0, "n_runs", g.size())
    pd.set_option("display.width", 200)
    print(table.to_string())
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"per-run values written to {a.out}")


if __name__ == "__main__":
    main()
