"""Unit tests for graph construction, model, loss, decision layer and metrics."""

import numpy as np
import pandas as pd
import pytest
import torch

from landslide_gnn.config import Config, GraphConfig, ModelConfig
from landslide_gnn.decision import candidate_class, decide_sequence, tau_from_costs
from landslide_gnn.graph import build_site_graph
from landslide_gnn.losses import FocalLoss, class_weights
from landslide_gnn.metrics import early_warning_metrics
from landslide_gnn.models import LandslideGNN, segment_softmax


def _slope_stations():
    # 3 rows down a slope facing south (row 0 = crest), 2 columns
    rows = []
    for r in range(3):
        for c in range(2):
            rows.append({"site_id": "s", "station_id": f"S{r}{c}", "x_m": 20.0 * c,
                         "y_m": -20.0 * r, "z_m": 100.0 - 12.0 * r})
    return pd.DataFrame(rows)


def test_flow_edges_point_downslope():
    g = build_site_graph(_slope_stations(), GraphConfig(k=2, flow_radius_m=50))
    src, dst = g.edge_index
    flow = g.edge_attr[:, 4] > 0.5
    assert flow.any()
    # every flow edge goes from higher to lower elevation
    assert np.all(g.pos[src[flow], 2] > g.pos[dst[flow], 2])
    # downslope direction estimated from the plane fit points south
    assert np.allclose(g.downslope[:, 1], -1.0, atol=1e-6)
    # one self-loop per station
    assert int((src == dst).sum()) == g.num_nodes


def test_no_flow_edges_when_disabled():
    g = build_site_graph(_slope_stations(), GraphConfig(use_flow_edges=False))
    assert not (g.edge_attr[:, 4] > 0.5).any()


def test_segment_softmax_sums_to_one():
    scores = torch.randn(7, 3)
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    a = segment_softmax(scores, idx, 3)
    sums = torch.zeros(3, 3).index_add(0, idx, a)
    assert torch.allclose(sums, torch.ones(3, 3), atol=1e-6)


@pytest.mark.parametrize("mtype", ["gat", "gatv2", "gcn", "sage", "none"])
def test_model_forward_shapes(mtype):
    g = build_site_graph(_slope_stations(), GraphConfig())
    cfg = ModelConfig(type=mtype)
    m = LandslideGNN(cfg)
    x = torch.randn(g.num_nodes, 15, 30)
    logits, attn = m(x, torch.as_tensor(g.edge_index), torch.as_tensor(g.edge_attr), return_attention=True)
    assert logits.shape == (g.num_nodes, 4)
    loss = logits.sum()
    loss.backward()


def test_message_passing_reaches_neighbours():
    """Changing an upslope station's input must change a downslope station's output."""
    torch.manual_seed(0)
    g = build_site_graph(_slope_stations(), GraphConfig())
    m = LandslideGNN(ModelConfig(dropout=0.0, attn_dropout=0.0)).eval()
    ei, ea = torch.as_tensor(g.edge_index), torch.as_tensor(g.edge_attr)
    x = torch.randn(g.num_nodes, 15, 30)
    base = m(x, ei, ea)
    x2 = x.clone()
    x2[0] += 5.0                                   # perturb the crest station
    out = m(x2, ei, ea)
    toe = g.station_ids.index("S20")
    assert not torch.allclose(base[toe], out[toe])  # info travelled 2 hops downslope
    # the per-node baseline must NOT propagate it
    m0 = LandslideGNN(ModelConfig(type="none", dropout=0.0)).eval()
    assert torch.allclose(m0(x, ei, ea)[toe], m0(x2, ei, ea)[toe])


def test_focal_reduces_to_ce():
    logits = torch.randn(10, 4)
    y = torch.randint(0, 4, (10,))
    fl = FocalLoss(torch.ones(4), gamma=0.0)(logits, y)
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(fl, ce, atol=1e-6)


def test_focal_ignores_unlabeled_and_masked():
    logits = torch.randn(4, 4)
    y = torch.tensor([0, -1, 2, 3])
    mask = torch.tensor([True, True, False, True])
    a = FocalLoss(gamma=2.0)(logits, y, mask)
    b = FocalLoss(gamma=2.0)(logits[[0, 3]], y[[0, 3]])
    assert torch.allclose(a, b)


def test_class_weights_rare_class_heavier():
    labels = np.array([0] * 1000 + [1] * 100 + [2] * 10 + [3] * 50)
    w = class_weights(labels, "effective", beta=0.9999)
    assert w[2] > w[3] > w[1] > w[0]
    assert np.isclose(w.sum(), 4)


def test_tau_from_costs():
    assert np.isclose(tau_from_costs(3, 7), 0.3)


def test_candidate_class_escalates_below_argmax():
    p = np.array([0.1, 0.55, 0.25, 0.10])  # Warning most likely, but P(E)+P(L)=0.35 > 0.3
    assert candidate_class(p, 0.3) == 2
    assert candidate_class(p, 0.5) == 1


def test_hysteresis():
    from landslide_gnn.config import DecisionConfig
    cfg = DecisionConfig(ema=1.0, tau=0.5, k_up=2, k_down=3, k_up_landslide=1)
    P = np.array([[1, 0, 0, 0]] * 2 + [[0, 0, 1, 0]] * 3 + [[1, 0, 0, 0]] * 4, dtype=float)
    out = decide_sequence(P, cfg)
    assert out.tolist() == [0, 0, 0, 2, 2, 2, 2, 0, 0]
    P = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    assert decide_sequence(P, cfg).tolist() == [0, 3]   # Landslide escalates at once


def test_early_warning_metrics():
    t = np.arange(10) * 600.0
    rows = []
    lab = [0, 0, 1, 2, 2, 3, 3, 3, 3, 3]
    al = [0, 0, 0, 0, 2, 2, 3, 3, 3, 3]
    for k in range(10):
        rows.append(("s", "E1", t[k], "A", lab[k], al[k]))
    # second event without failure but with a false alarm
    for k in range(10):
        rows.append(("s", "E2", t[k], "A", 0, 2 if k in (4, 5) else 0))
    df = pd.DataFrame(rows, columns=["site_id", "event_id", "t_s", "station_id", "label", "alert"])
    m = early_warning_metrics(df, horizon_s=3600)
    assert m["detected"] == 1 and m["missed"] == 0
    assert np.isclose(m["lead_time_median_h"], 600 / 3600)
    assert m["false_alarms"] == 1


def test_config_override():
    cfg = Config().override(["model.type=gcn", "train.lr=0.01", "graph.k=5"])
    assert cfg.model.type == "gcn" and cfg.train.lr == 0.01 and cfg.graph.k == 5
    with pytest.raises(KeyError):
        Config().override(["model.nope=1"])
