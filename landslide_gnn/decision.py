"""Cost-sensitive alert decision with smoothing and hysteresis (Algorithm 2)."""

from __future__ import annotations

import numpy as np

from .config import DecisionConfig

HIGH = (2, 3)   # evacuation-level classes
LOW = (0, 1)


def tau_from_costs(cost_false_alarm: float, cost_miss: float) -> float:
    """Eq. (17): tau = c_f / (c_f + c_m)."""
    return cost_false_alarm / (cost_false_alarm + cost_miss)


def candidate_class(p: np.ndarray, tau: float) -> np.ndarray:
    """Cost-sensitive candidate for probability rows ``p`` [..., 4]."""
    high = p[..., 2] + p[..., 3] > tau
    c_high = np.where(p[..., 3] > p[..., 2], 3, 2)
    c_low = np.where(p[..., 1] > p[..., 0], 1, 0)
    return np.where(high, c_high, c_low)


def decide_sequence(probs: np.ndarray, cfg: DecisionConfig, valid: np.ndarray | None = None) -> np.ndarray:
    """Apply the decision layer to one station's time-ordered probabilities.

    probs : [T, 4] class probabilities at consecutive decision times
    valid : [T] bool; steps without a prediction keep the previous alert level
    returns [T] alert level per step
    """
    T = len(probs)
    out = np.zeros(T, dtype=np.int64)
    if valid is None:
        valid = np.ones(T, dtype=bool)
    ema = None
    level, n_up, n_down = 0, 0, 0
    for t in range(T):
        if not valid[t]:
            out[t] = level
            continue
        ema = probs[t] if ema is None else cfg.ema * probs[t] + (1 - cfg.ema) * ema
        c = int(candidate_class(ema, cfg.tau))
        if c > level:
            n_up, n_down = n_up + 1, 0
            need = cfg.k_up_landslide if c == 3 else cfg.k_up
            if n_up >= need:
                level, n_up = c, 0
        elif c < level:
            n_down, n_up = n_down + 1, 0
            if n_down >= cfg.k_down:
                level, n_down = c, 0
        else:
            n_up = n_down = 0
        out[t] = level
    return out
