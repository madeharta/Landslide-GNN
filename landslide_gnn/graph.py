"""Construction of the directed slope graph (Section V-A of the paper).

Edges ``(j -> i)`` are stored as ``edge_index[:, e] = [j, i]`` (source, target).
The edge set is the union of

* symmetric k-nearest-neighbour edges (3-D distance), Eq. (19),
* directed upslope-to-downslope *flow* edges, Eq. (20),
* one self-loop per station.

Each edge carries ``[d_ij, z_j - z_i, g_ij, sigma_ij, 1_flow]``, Eq. (21).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import GraphConfig

EDGE_FEATURES = ["distance_m", "drop_m", "gradient", "geo_similarity", "is_flow"]
EDGE_DIM = len(EDGE_FEATURES)
_CONTINUOUS = [0, 1, 2]  # standardised; similarity and indicator stay in [0, 1]


@dataclass
class SiteGraph:
    site_id: str
    station_ids: list
    pos: np.ndarray          # [N, 3] east, north, up (m)
    edge_index: np.ndarray   # [2, E] int64, rows = (source, target)
    edge_attr: np.ndarray    # [E, EDGE_DIM] float32 (raw, not standardised)
    downslope: np.ndarray    # [N, 2] unit horizontal steepest-descent vectors

    @property
    def num_nodes(self) -> int:
        return len(self.station_ids)

    def summary(self) -> dict:
        is_self = self.edge_index[0] == self.edge_index[1]
        return {
            "site_id": self.site_id,
            "stations": self.num_nodes,
            "edges_total": int(self.edge_index.shape[1]),
            "self_loops": int(is_self.sum()),
            "flow_edges": int((self.edge_attr[:, 4] > 0.5).sum()),
        }


# ---------------------------------------------------------------------------
def _downslope_directions(pos: np.ndarray, az_deg: np.ndarray | None, n_fit: int) -> np.ndarray:
    """Unit horizontal steepest-descent vector per station.

    Uses ``downslope_az_deg`` when given (azimuth clockwise from north);
    otherwise fits a local plane z = a x + b y + c to the ``n_fit`` nearest
    stations and returns -(a, b) normalised.
    """
    n = len(pos)
    out = np.zeros((n, 2))
    d = np.linalg.norm(pos[:, None, :2] - pos[None, :, :2], axis=-1)
    for i in range(n):
        if az_deg is not None and np.isfinite(az_deg[i]):
            a = np.deg2rad(az_deg[i])
            out[i] = [np.sin(a), np.cos(a)]  # east, north
            continue
        idx = np.argsort(d[i])[: max(3, min(n_fit, n))]
        if len(idx) < 3:
            continue
        A = np.c_[pos[idx, 0], pos[idx, 1], np.ones(len(idx))]
        coef, *_ = np.linalg.lstsq(A, pos[idx, 2], rcond=None)
        g = -coef[:2]
        nrm = np.linalg.norm(g)
        if nrm > 1e-9:
            out[i] = g / nrm
    return out


def build_site_graph(stations: pd.DataFrame, cfg: GraphConfig) -> SiteGraph:
    """Build the graph of one site from its rows of ``stations.csv``."""
    st = stations.reset_index(drop=True)
    pos = st[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    n = len(pos)
    az = st["downslope_az_deg"].to_numpy(dtype=float) if "downslope_az_deg" in st else None
    geo = st["geo_unit"].astype(str).to_numpy() if "geo_unit" in st and st["geo_unit"].notna().any() else None
    down = _downslope_directions(pos, az, cfg.slope_fit_neighbors)

    diff = pos[None, :, :] - pos[:, None, :]      # diff[i, j] = p_j - p_i
    dist = np.linalg.norm(diff, axis=-1)
    edges: dict[tuple[int, int], bool] = {}        # (src, dst) -> is_flow

    # --- symmetric kNN edges --------------------------------------------
    if cfg.use_knn_edges and n > 1:
        k = min(cfg.k, n - 1)
        for i in range(n):
            order = [j for j in np.argsort(dist[i]) if j != i][:k]
            for j in order:
                edges.setdefault((int(j), i), False)
                edges.setdefault((i, int(j)), False)

    # --- directed flow edges (upslope j -> downslope i) ------------------
    if cfg.use_flow_edges and n > 1:
        cos_max = np.cos(np.deg2rad(cfg.flow_max_angle_deg))
        for j in range(n):
            if not np.any(down[j]):
                continue
            for i in range(n):
                if i == j or dist[j, i] > cfg.flow_radius_m:
                    continue
                if pos[j, 2] - pos[i, 2] < cfg.flow_min_drop_m:
                    continue
                h = pos[i, :2] - pos[j, :2]
                hn = np.linalg.norm(h)
                if hn < 1e-9 or float(h @ down[j]) / hn < cos_max:
                    continue
                edges[(j, i)] = True

    # --- self loops ------------------------------------------------------
    for i in range(n):
        edges[(i, i)] = False

    keys = sorted(edges, key=lambda e: (e[1], e[0]))
    src = np.array([e[0] for e in keys], dtype=np.int64)
    dst = np.array([e[1] for e in keys], dtype=np.int64)
    d_ij = dist[dst, src]
    drop = pos[src, 2] - pos[dst, 2]                      # z_j - z_i
    d_h = np.linalg.norm(pos[src, :2] - pos[dst, :2], axis=-1)
    grad = np.where(d_h > 1e-6, drop / np.maximum(d_h, 1e-6), 0.0)
    if geo is not None:
        sim = (geo[src] == geo[dst]).astype(float)
    else:
        sim = np.ones(len(src))
    flow = np.array([1.0 if edges[e] else 0.0 for e in keys])
    attr = np.stack([d_ij, drop, grad, sim, flow], axis=1).astype(np.float32)

    return SiteGraph(
        site_id=str(st["site_id"].iloc[0]) if "site_id" in st else "site_0",
        station_ids=st["station_id"].astype(str).tolist(),
        pos=pos,
        edge_index=np.stack([src, dst]),
        edge_attr=attr,
        downslope=down,
    )


def edge_normalizer(graphs: list[SiteGraph]) -> dict:
    """Mean/std of the continuous edge features over non-self-loop edges."""
    rows = [g.edge_attr[g.edge_index[0] != g.edge_index[1]] for g in graphs]
    rows = [r for r in rows if len(r)]
    mean = np.zeros(EDGE_DIM, dtype=np.float32)
    std = np.ones(EDGE_DIM, dtype=np.float32)
    if rows:
        allr = np.concatenate(rows)
        mean[_CONTINUOUS] = allr[:, _CONTINUOUS].mean(0)
        std[_CONTINUOUS] = allr[:, _CONTINUOUS].std(0) + 1e-6
    return {"mean": mean.tolist(), "std": std.tolist()}


def normalize_edges(attr: np.ndarray, norm: dict) -> np.ndarray:
    return ((attr - np.asarray(norm["mean"], np.float32)) / np.asarray(norm["std"], np.float32)).astype(np.float32)
