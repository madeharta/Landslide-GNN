"""Graph layers implemented in plain PyTorch (no PyTorch Geometric needed).

Conventions: ``edge_index[0]`` = source j, ``edge_index[1]`` = target i.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax of ``scores`` [E, H] grouped by target node ``index`` [E]."""
    idx = index.unsqueeze(-1).expand_as(scores)
    smax = torch.full((num_nodes, scores.size(1)), float("-inf"), device=scores.device, dtype=scores.dtype)
    smax = smax.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - smax[index])
    den = torch.zeros((num_nodes, scores.size(1)), device=scores.device, dtype=scores.dtype)
    den = den.index_add(0, index, ex)
    return ex / (den[index] + 1e-16)


class EdgeGATLayer(nn.Module):
    """Edge-aware multi-head graph attention with a residual connection.

    Implements Eqs. (25)-(27) of the paper. With ``v2=True`` the score follows
    GATv2 (nonlinearity applied before the attention vector).
    """

    def __init__(self, dim: int, heads: int, edge_dim: int, use_edge_features: bool = True,
                 v2: bool = False, negative_slope: float = 0.2, attn_dropout: float = 0.0,
                 dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"hidden size {dim} must be divisible by heads {heads}")
        self.h, self.dh = heads, dim // heads
        self.v2 = v2
        self.use_edge = use_edge_features
        self.W = nn.Linear(dim, dim, bias=False)
        self.We = nn.Linear(edge_dim, dim, bias=False) if use_edge_features else None
        n_parts = 1 if v2 else (3 if use_edge_features else 2)
        self.a = nn.Parameter(torch.empty(heads, n_parts * self.dh))
        self.negative_slope = negative_slope
        self.attn_drop = nn.Dropout(attn_dropout)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
        if self.We is not None:
            nn.init.xavier_uniform_(self.We.weight)

    def forward(self, h, edge_index, edge_attr):
        N = h.size(0)
        src, dst = edge_index
        wh = self.W(h).view(N, self.h, self.dh)
        parts_i, parts_j = wh[dst], wh[src]
        we = self.We(edge_attr).view(-1, self.h, self.dh) if self.use_edge else None
        if self.v2:
            z = parts_i + parts_j + (we if we is not None else 0)
            s = (F.leaky_relu(z, self.negative_slope) * self.a.unsqueeze(0)).sum(-1)
        else:
            cat = [parts_i, parts_j] + ([we] if we is not None else [])
            s = F.leaky_relu((torch.cat(cat, dim=-1) * self.a.unsqueeze(0)).sum(-1), self.negative_slope)
        alpha = segment_softmax(s, dst, N)                               # [E, H]
        msg = self.attn_drop(alpha).unsqueeze(-1) * parts_j                # [E, H, dh]
        agg = torch.zeros(N, self.h, self.dh, device=h.device, dtype=h.dtype).index_add(0, dst, msg)
        return h + self.drop(F.elu(agg.reshape(N, -1))), alpha


class GCNLayer(nn.Module):
    """Kipf-Welling propagation with symmetric normalisation (ignores edge features)."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr=None):
        N = h.size(0)
        s, d = edge_index   # self-loops are already part of the graph
        deg = torch.zeros(N, device=h.device).index_add(0, d, torch.ones_like(d, dtype=h.dtype)).clamp(min=1)
        w = deg[s].rsqrt() * deg[d].rsqrt()
        agg = torch.zeros_like(h).index_add(0, d, w.unsqueeze(-1) * self.W(h)[s])
        return h + self.drop(F.elu(agg)), None


class SAGELayer(nn.Module):
    """GraphSAGE (mean aggregator) over the directed in-neighbourhood."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.nb_lin = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr=None):
        N = h.size(0)
        src, dst = edge_index
        keep = src != dst
        src, dst = src[keep], dst[keep]
        cnt = torch.zeros(N, device=h.device).index_add(0, dst, torch.ones_like(dst, dtype=h.dtype)).clamp(min=1)
        mean = torch.zeros_like(h).index_add(0, dst, h[src]) / cnt.unsqueeze(-1)
        return h + self.drop(F.elu(self.self_lin(h) + self.nb_lin(mean))), None
