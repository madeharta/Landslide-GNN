"""Synthetic dataset generator -- FOR PIPELINE TESTING ONLY.

Produces ``stations.csv`` and ``observations.csv`` in the documented format so
the code can be exercised end to end before real data exist. The physics is a
deliberately simple caricature:

* stations sit on a grid on a planar slope (row 0 = crest);
* rainfall comes in random storms;
* in a *failure* event a block fails first near the crest and the failure
  propagates downslope row by row; each failing station shows accelerating
  (inverse-velocity-like) tilt and displacement in the hours before its own
  failure time;
* non-failure events contain rain-driven creep, sensor noise, packet loss and
  occasional single-station disturbances (e.g. an animal knocking a node),
  which must NOT be classified as hazardous.

Numbers produced here say nothing about real slopes. Do not report them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import FEATURE_COLUMNS


def _site_stations(site: str, rows: int, cols: int, rng) -> pd.DataFrame:
    recs = []
    slope = np.deg2rad(rng.uniform(25, 40))
    for r in range(rows):
        for c in range(cols):
            x = c * 25.0 + rng.normal(0, 2)
            y = -r * 20.0 + rng.normal(0, 2)
            z = 200.0 - r * 20.0 * np.tan(slope) + rng.normal(0, 0.5)
            recs.append({"site_id": site, "station_id": f"{site}_S{r}{c}", "x_m": round(x, 2),
                         "y_m": round(y, 2), "z_m": round(z, 2),
                         "geo_unit": "colluvium" if r < rows - 1 else "residual_soil",
                         "row": r, "col": c})
    return pd.DataFrame(recs)


def _rain(n, dt_h, rng):
    rate = np.zeros(n)
    for _ in range(rng.integers(1, 4)):
        s = rng.integers(0, n)
        d = int(rng.uniform(2, 14) / dt_h)
        rate[s:s + d] += rng.uniform(5, 35) * np.abs(np.sin(np.linspace(0, np.pi, len(rate[s:s + d]))))
    return rate


def _accum(rate, dt_h, hours, prior):
    k = int(hours / dt_h)
    c = np.convolve(rate * dt_h, np.ones(k))[: len(rate)]
    return c + prior * np.clip(1 - np.arange(len(rate)) / k, 0, 1)


def generate(out_dir: str, n_sites: int = 2, rows: int = 4, cols: int = 3, n_events: int = 40,
             hours: float = 72.0, dt_min: float = 10.0, failure_frac: float = 0.5,
             loss_rate: float = 0.03, seed: int = 0, split: bool = True,
             start: str = "2026-01-01T00:00:00Z") -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stations = pd.concat([_site_stations(f"site{s}", rows, cols, rng) for s in range(n_sites)], ignore_index=True)
    dt_h = dt_min / 60.0
    n = int(hours / dt_h)
    t_h = np.arange(n) * dt_h
    H_E, H_W, T_PRE = 3.0, 12.0, 24.0
    frames = []
    ev_ids = [f"E{e:03d}" for e in range(n_events)]
    order = rng.permutation(n_events)
    splits = {}
    for rank, i in enumerate(order):
        splits[ev_ids[i]] = "train" if rank < 0.7 * n_events else ("val" if rank < 0.85 * n_events else "test")
    t0 = pd.Timestamp(start)
    for e, ev in enumerate(ev_ids):
        site = f"site{e % n_sites}"
        st = stations[stations.site_id == site].reset_index(drop=True)
        rate = _rain(n, dt_h, rng)
        prior72 = rng.uniform(0, 80)
        r24 = _accum(rate, dt_h, 24, prior72 / 3)
        r72 = _accum(rate, dt_h, 72, prior72)
        temp = 22 + 4 * np.sin(2 * np.pi * (t_h - 9) / 24) - 0.08 * rate + rng.normal(0, 0.3, n)
        rh = np.clip(75 - 12 * np.sin(2 * np.pi * (t_h - 9) / 24) + 1.2 * rate + rng.normal(0, 2, n), 30, 100)
        failure = rng.random() < failure_frac
        tf = np.full(len(st), np.inf)
        if failure:
            tf0 = rng.uniform(0.45, 0.85) * hours
            fail_cols = set(rng.choice(cols, size=rng.integers(1, cols + 1), replace=False).tolist())
            delay = rng.uniform(0.8, 2.0)
            for i, s in st.iterrows():
                if s.col in fail_cols:
                    tf[i] = tf0 + s.row * delay + rng.normal(0, 0.2)
        ev_start = t0 + pd.Timedelta(days=4 * e)
        for i, s in st.iterrows():
            # --- kinematics -------------------------------------------------
            creep = 0.002 * np.cumsum(rate * dt_h) / 50.0                     # m, rain-driven
            disp = creep.copy()
            if np.isfinite(tf[i]):
                pre = (t_h > tf[i] - T_PRE) & (t_h < tf[i])
                disp[pre] += 0.08 * (1.0 / (tf[i] - t_h[pre] + 0.5) - 1.0 / (T_PRE + 0.5))
                post = t_h >= tf[i]
                disp[post] += 0.08 * (1 / 0.5 - 1 / (T_PRE + 0.5)) + 0.5 * (t_h[post] - tf[i])
            tilt_gain = rng.uniform(30, 60)                                   # deg per m
            roll = rng.normal(0, 1.5) + tilt_gain * disp * 0.4 + rng.normal(0, 0.05, n)
            pitch = rng.normal(0, 1.5) + tilt_gain * disp + rng.normal(0, 0.05, n)
            yaw = rng.uniform(-180, 180) + np.cumsum(rng.normal(0, 0.01, n)) + 5 * disp
            if not failure and rng.random() < 0.3:                          # nuisance disturbance
                k = rng.integers(n // 4, n)
                L = int(rng.uniform(0.3, 1.0) / dt_h) + 1
                bump = rng.uniform(2, 6)
                roll[k:k + L] += bump
                pitch[k:k + L] += bump * rng.uniform(-1, 1)
            gps_noise = 0.01
            dirv = np.array([0.0, -np.cos(0.6), -np.sin(0.6)])                # downslope, down
            gps = disp[:, None] * dirv[None, :] + rng.normal(0, gps_noise, (n, 3))
            smooth = pd.Series(disp).rolling(6, min_periods=1).mean().to_numpy()
            vel = np.r_[0, np.diff(smooth)] / dt_h * 1000.0                   # mm/h
            # --- labels -----------------------------------------------------
            lab = np.zeros(n, dtype=int)
            if np.isfinite(tf[i]):
                lab[(t_h >= tf[i] - H_W) & (t_h < tf[i] - H_E)] = 1
                lab[(t_h >= tf[i] - H_E) & (t_h < tf[i])] = 2
                lab[t_h >= tf[i]] = 3
            lab[(lab == 0) & (r72 > 150)] = 1                                # very wet antecedent
            feats = np.c_[roll, pitch, yaw,
                          np.r_[0, np.diff(roll)], np.r_[0, np.diff(pitch)], np.r_[0, np.diff(yaw)],
                          gps, vel, rate, r24, r72, temp, rh]
            keep = rng.random(n) > loss_rate
            df = pd.DataFrame(np.round(feats[keep], 5), columns=FEATURE_COLUMNS)
            df.insert(0, "station_id", s.station_id)
            df.insert(0, "timestamp", (ev_start + pd.to_timedelta(t_h[keep], unit="h")).strftime("%Y-%m-%dT%H:%M:%SZ"))
            df.insert(0, "event_id", ev)
            df.insert(0, "site_id", site)
            df["label"] = lab[keep]
            if split:
                df["split"] = splits[ev]
            df["source"] = "synthetic"
            frames.append(df)
    obs = pd.concat(frames, ignore_index=True)
    sp = out / "stations.csv"
    op = out / "observations.csv"
    stations.drop(columns=["row", "col"]).to_csv(sp, index=False)
    obs.to_csv(op, index=False)
    return sp, op
