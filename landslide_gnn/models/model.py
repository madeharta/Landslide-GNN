"""Landslide severity GNN: 1D-CNN temporal encoder -> graph layers -> classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..graph import EDGE_DIM
from ..schema import FEATURE_COLUMNS, NUM_CLASSES
from .layers import EdgeGATLayer, GCNLayer, SAGELayer


class TemporalEncoder(nn.Module):
    """Stage A, Eqs. (22)-(24): shared Conv1D stack + global average pooling."""

    def __init__(self, in_features: int, channels: list, kernels: list, hidden: int, dropout: float,
                 pool: str = "mean"):
        super().__init__()
        if pool not in ("mean", "mean_last", "mean_max"):
            raise ValueError(f"unknown model.pool '{pool}' (mean | mean_last | mean_max)")
        self.pool = pool
        layers, c_in = [], in_features
        for c, k in zip(channels, kernels):
            layers += [nn.Conv1d(c_in, c, k, padding=k // 2), nn.ReLU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        mult = 1 if pool == "mean" else 2
        self.proj = nn.Sequential(nn.Linear(mult * c_in, hidden), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):            # x: [N, F, T]
        u = self.conv(x)
        z = u.mean(dim=-1)
        if self.pool == "mean_last":   # keep the most recent feature map column as well
            z = torch.cat([z, u[..., -1]], dim=-1)
        elif self.pool == "mean_max":
            z = torch.cat([z, u.amax(dim=-1)], dim=-1)
        return self.proj(z)


class LandslideGNN(nn.Module):
    def __init__(self, cfg: ModelConfig, in_features: int = len(FEATURE_COLUMNS),
                 edge_dim: int = EDGE_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.cfg = cfg
        self.encoder = TemporalEncoder(in_features, cfg.conv_channels, cfg.conv_kernels, cfg.hidden,
                                       cfg.dropout, cfg.pool)
        t = cfg.type.lower()
        n_layers = 0 if t == "none" else cfg.layers
        layers = []
        for _ in range(n_layers):
            if t in ("gat", "gatv2"):
                layers.append(EdgeGATLayer(cfg.hidden, cfg.heads, edge_dim, cfg.use_edge_features,
                                           v2=(t == "gatv2"), negative_slope=cfg.negative_slope,
                                           attn_dropout=cfg.attn_dropout, dropout=cfg.dropout))
            elif t == "gcn":
                layers.append(GCNLayer(cfg.hidden, cfg.dropout))
            elif t == "sage":
                layers.append(SAGELayer(cfg.hidden, cfg.dropout))
            else:
                raise ValueError(f"unknown model.type '{cfg.type}' (gat | gatv2 | gcn | sage | none)")
        self.gnn = nn.ModuleList(layers)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.head_hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, return_attention: bool = False):
        """Return per-node logits [N, C] (and per-layer attention [E, H] if asked)."""
        h = self.encoder(x)
        attn = []
        for layer in self.gnn:
            h, a = layer(h, edge_index, edge_attr)
            attn.append(a)
        logits = self.head(h)
        return (logits, attn) if return_attention else logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
