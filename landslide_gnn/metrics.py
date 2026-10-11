"""Classification and early-warning metrics (Section VI-F)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             matthews_corrcoef, precision_recall_fscore_support)

from .schema import CLASS_NAMES, NUM_CLASSES


def _quiet(fn, *args) -> float:
    with warnings.catch_warnings():          # single-class subsets are legitimate here
        warnings.simplefilter("ignore")
        return float(fn(*args))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = y_true >= 0
    y_true, y_pred = y_true[m], y_pred[m]
    labels = list(range(NUM_CLASSES))
    if len(y_true) == 0:
        return {"n": 0}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = s > 0
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _quiet(balanced_accuracy_score, y_true, y_pred),
        "macro_precision": float(p[present].mean()),
        "macro_recall": float(r[present].mean()),
        "macro_f1": float(f[present].mean()),
        "mcc": _quiet(matthews_corrcoef, y_true, y_pred),
        "under_escalation_rate": float((y_pred < y_true).mean()),
        "over_escalation_rate": float((y_pred > y_true).mean()),
        "per_class": {
            CLASS_NAMES[c]: {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]), "support": int(s[c])}
            for c in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def early_warning_metrics(pred: pd.DataFrame, horizon_s: float, alert_col: str = "alert") -> dict:
    """Event-level metrics from a predictions table.

    ``pred`` needs columns: site_id, event_id, t_s, station_id, label, <alert_col>.
    Failure onset of an event = first time any station is labelled Landslide (3).
    An event is detected when the site-level alert (max over stations) reaches
    Evacuate or higher within ``horizon_s`` before onset; lead time is the
    onset minus the first such alert. Alerts raised only after onset count as
    missed (reported separately as late). A false alarm is a rise of the
    site-level alert to >= Evacuate that is not followed by an onset within
    ``horizon_s``.
    """
    leads, detected, missed, late, false_alarms = [], 0, 0, 0, 0
    station_days, switches = 0.0, 0
    for (site, ev), g in pred.groupby(["site_id", "event_id"]):
        g = g.sort_values("t_s")
        site_true = g.groupby("t_s")["label"].max()
        site_alert = g.groupby("t_s")[alert_col].max()
        t = site_alert.index.to_numpy(dtype=float)
        a = site_alert.to_numpy()
        onset_t = site_true.index[site_true.to_numpy() == 3]
        onset = float(onset_t[0]) if len(onset_t) else None
        if onset is not None:
            win = (t >= onset - horizon_s) & (t <= onset) & (a >= 2)
            if win.any():
                detected += 1
                leads.append(onset - float(t[win][0]))
            else:
                missed += 1
                if ((t > onset) & (a >= 2)).any():
                    late += 1
        rises = np.nonzero((a[1:] >= 2) & (a[:-1] < 2))[0] + 1
        if len(a) and a[0] >= 2:
            rises = np.r_[0, rises]
        for r in rises:
            # a rise inside the warning horizon or after onset is not a false alarm
            if onset is None or t[r] < onset - horizon_s:
                false_alarms += 1
        dur_days = max(t[-1] - t[0], 0.0) / 86400.0 if len(t) else 0.0
        n_st = g["station_id"].nunique()
        station_days += dur_days * n_st
        for _, s in g.groupby("station_id"):
            v = s[alert_col].to_numpy()
            switches += int((v[1:] != v[:-1]).sum())
    n_events = detected + missed
    return {
        "failure_events": n_events,
        "detected": detected,
        "missed": missed,
        "missed_but_alerted_after_onset": late,
        "detection_rate": detected / n_events if n_events else None,
        "lead_time_median_h": float(np.median(leads)) / 3600 if leads else None,
        "lead_time_iqr_h": [float(np.percentile(leads, 25)) / 3600, float(np.percentile(leads, 75)) / 3600] if leads else None,
        "false_alarms": false_alarms,
        "false_alarms_per_station_month": false_alarms / (station_days / 30.0) if station_days > 0 else None,
        "alert_switches_per_station_day": switches / station_days if station_days > 0 else None,
    }
