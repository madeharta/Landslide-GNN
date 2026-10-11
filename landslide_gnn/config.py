"""Experiment configuration (YAML <-> nested dataclasses)."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    stations_csv: str = "data/stations.csv"
    observations_csv: str = "data/observations.csv"
    window: int = 60                 # T, samples per window
    stride: int = 1                  # step between consecutive training windows
    max_gap: int = 3                 # forward-fill gaps up to this many samples
    min_obs_frac: float = 0.5        # station inactive if less of its window observed
    split_mode: str = "auto"         # auto | column | random | all
    split_ratios: list = field(default_factory=lambda: [0.70, 0.15, 0.15])


@dataclass
class GraphConfig:
    k: int = 3                       # k nearest neighbours (symmetrised)
    use_knn_edges: bool = True
    use_flow_edges: bool = True
    flow_radius_m: float = 60.0      # r_max
    flow_min_drop_m: float = 1.0     # Delta z_min
    flow_max_angle_deg: float = 45.0 # psi_max
    slope_fit_neighbors: int = 6     # stations used for the local plane fit


@dataclass
class ModelConfig:
    type: str = "gat"                # gat | gatv2 | gcn | sage | none (per-node CNN)
    conv_channels: list = field(default_factory=lambda: [32, 64])
    conv_kernels: list = field(default_factory=lambda: [5, 3])
    hidden: int = 64
    pool: str = "mean_last"          # mean (paper Eq. 24) | mean_last (recommended) | mean_max
    heads: int = 4
    layers: int = 2
    use_edge_features: bool = True
    head_hidden: int = 32
    dropout: float = 0.2
    attn_dropout: float = 0.2
    negative_slope: float = 0.2


@dataclass
class LossConfig:
    focal_gamma: float = 2.0
    class_weighting: str = "effective"   # effective | inverse | none
    beta: float = 0.999                  # varsigma in Eq. (29)


@dataclass
class TrainConfig:
    epochs: int = 200
    patience: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0


@dataclass
class DecisionConfig:
    ema: float = 0.5                 # eta
    tau: float = 0.3                 # escalation threshold, Eq. (17)
    k_up: int = 2
    k_down: int = 5
    k_up_landslide: int = 1
    horizon_s: float = 24 * 3600.0   # H_max for event-level metrics


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    output_dir: str = "runs/default"

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Config":
        return _fill(cls(), d or {})

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def override(self, assignments: list[str] | None) -> "Config":
        """Apply ``section.key=value`` strings (value parsed as YAML)."""
        for a in assignments or []:
            if "=" not in a:
                raise ValueError(f"override must look like section.key=value, got '{a}'")
            key, val = a.split("=", 1)
            obj = self
            parts = key.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            if not hasattr(obj, parts[-1]):
                raise KeyError(f"unknown config key '{key}'")
            setattr(obj, parts[-1], yaml.safe_load(val))
        return self


def _fill(obj: Any, d: dict) -> Any:
    names = {f.name for f in fields(obj)}
    for k, v in d.items():
        if k not in names:
            raise KeyError(f"unknown config key '{k}' in section {type(obj).__name__}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _fill(cur, v)
        else:
            setattr(obj, k, copy.deepcopy(v))
    return obj
