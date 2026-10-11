"""End-to-end tests: CSV validation, training, evaluation and prediction."""

import json

import pandas as pd
import pytest

from landslide_gnn.cli import main
from landslide_gnn.config import Config
from landslide_gnn.data import DataFormatError, load_observations, load_stations, prepare
from landslide_gnn.synthetic import generate


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    sp, op = generate(str(d), n_sites=1, rows=3, cols=2, n_events=8, hours=12, dt_min=20, seed=1)
    return d, sp, op


def test_validate_ok(tiny):
    _, sp, op = tiny
    assert main(["validate", "--stations", str(sp), "--observations", str(op), "--set", "data.window=12"]) == 0


def test_missing_column_is_reported(tiny, tmp_path):
    _, sp, op = tiny
    bad = pd.read_csv(op).drop(columns=["rain_72h_mm"])
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(DataFormatError, match="rain_72h_mm"):
        load_observations(str(p))


def test_bad_label_is_reported(tiny, tmp_path):
    _, sp, op = tiny
    bad = pd.read_csv(op)
    bad.loc[0, "label"] = 7
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(DataFormatError, match="label"):
        load_observations(str(p))


def test_unknown_station_is_reported(tiny, tmp_path):
    _, sp, op = tiny
    st = load_stations(str(sp)).iloc[1:]
    p = tmp_path / "st.csv"
    st.to_csv(p, index=False)
    cfg = Config()
    cfg.data.stations_csv, cfg.data.observations_csv = str(p), str(op)
    with pytest.raises(DataFormatError, match="not in stations.csv"):
        prepare(cfg)


def test_train_evaluate_predict(tiny, tmp_path):
    _, sp, op = tiny
    out = tmp_path / "run"
    common = ["--stations", str(sp), "--observations", str(op)]
    rc = main(["train", "--config", "configs/default.yaml", *common, "--output", str(out),
               "--set", "train.epochs=2", "data.window=12", "train.batch_size=8"])
    assert rc == 0
    for f in ["best.pt", "last.pt", "config.yaml", "train_log.csv", "splits.csv", "summary.json",
              "predictions_test.csv", "metrics_test.json", "confusion_test.csv", "attention_test.csv"]:
        assert (out / f).exists(), f
    m = json.loads((out / "metrics_test.json").read_text())
    assert "macro_f1" in m["decision_layer"] and "detection_rate" in m["early_warning"]

    ev = tmp_path / "eval"
    assert main(["evaluate", "--checkpoint", str(out / "best.pt"), *common, "--output", str(ev),
                 "--split", "all"]) == 0
    assert (ev / "metrics_all.json").exists()

    # prediction on data without labels
    unl = pd.read_csv(op).drop(columns=["label", "split"])
    up = tmp_path / "unlabeled.csv"
    unl.to_csv(up, index=False)
    pp = tmp_path / "pred.csv"
    assert main(["predict", "--checkpoint", str(out / "best.pt"), "--stations", str(sp),
                 "--observations", str(up), "--output", str(pp)]) == 0
    pred = pd.read_csv(pp)
    assert {"alert", "alert_name", "p_evacuate"} <= set(pred.columns)
