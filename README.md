# landslide-gnn

Reference implementation of the graph neural network described in

> M. H. Dwijaksara, Y. Wicaksana, D. Sarah, and A. Nasikun,
> **"A Low-Cost Landslide Monitoring System Using Graph Neural Network with LoRa-Connected IMU and GNSS Sensors,"** 2026 (manuscript in preparation).

Each sensing station on a slope (six-axis IMU + GNSS, plus the nearest weather
station) is a node of a directed graph. A shared 1-D CNN summarises each
station's recent history, edge-aware graph attention lets stations exchange
information along proximity and upslope→downslope *flow* edges, and a per-node
classifier assigns one of four severity levels:

| 0 | 1 | 2 | 3 |
|---|---|---|---|
| Normal | Warning | Evacuate | Landslide |

Alerts are issued through a cost-sensitive rule, escalate when
`P(Evacuate) + P(Landslide) > τ` with `τ = c_false / (c_false + c_miss)`,
followed by exponential smoothing and hysteresis.

```
window per station ──► Conv1D(15→32,k5) ─► Conv1D(32→64,k3) ─► pool ─► Dense 64     (shared weights)
  [15 features × T]                                                       │
                                                                          ▼
                  graph (kNN + flow edges, edge features) ─► edge-aware GAT × L (K heads, residual)
                                                                          │
                                                                          ▼
                                Dense 64→32 ─► Dense 32→4 ─► softmax ─► decision layer ─► alert
```

The implementation uses plain PyTorch, so PyTorch Geometric is **not** required.

---

## Installation

```bash
git clone https://github.com/<your-account>/landslide-gnn.git
cd landslide-gnn
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch            # or the CUDA build for your GPU, see pytorch.org
pip install -e ".[dev]"
pytest -q                    # 22 tests, well under a minute on a laptop CPU
```

Python ≥ 3.9 and PyTorch ≥ 2.1. A GPU is optional; the default model has
about 28 k parameters.

## Data

The dataset consists of **two CSV files**, `stations.csv` (one row per station)
and `observations.csv` (one row per station per timestamp, with 15 features, a
label and an optional split). The full specification is in
**[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)**; empty templates are in
`examples/`.

```bash
python -m landslide_gnn validate --stations data/stations.csv --observations data/observations.csv
```

`validate` checks columns, types, labels, splits and station references, and
prints the graph sizes, label counts and events per split.

## Train, validate and test

```bash
python -m landslide_gnn train --config configs/default.yaml \
    --stations data/stations.csv --observations data/observations.csv \
    --output runs/proposed
```

* **train** events are used to fit the model, and feature normalisation and class weights are computed from them only;
* **val** events drive early stopping on macro-F1 and best-checkpoint selection;
* **test** events are evaluated once with the best checkpoint at the end.

Splits come from the `split` column, or are drawn at random per event (70/15/15)
when the column is absent. Any hyperparameter can be overridden:

```bash
python -m landslide_gnn train ... --set model.type=gcn train.epochs=100 data.window=30 decision.tau=0.2
```

Outputs in `--output`:

| File | Content |
|---|---|
| `best.pt`, `last.pt` | checkpoints (weights + config + normalisation statistics) |
| `config.yaml` | the exact configuration used |
| `splits.csv` | event → split assignment |
| `train_log.csv` | loss and validation metrics per epoch |
| `metrics_test.json` | classification metrics (raw argmax and after the decision layer) and early-warning metrics |
| `confusion_test.csv` | confusion matrix after the decision layer |
| `predictions_test.csv` | per station and time: probabilities, argmax, alert level |
| `attention_test.csv` | mean attention weight per edge and layer (interpretability) |

## Evaluate a checkpoint / predict alerts

```bash
# labelled data, e.g. a new field campaign (no split column -> all events)
python -m landslide_gnn evaluate --checkpoint runs/proposed/best.pt \
    --stations data/stations.csv --observations data/field.csv --split all --output runs/field_eval

# unlabelled data -> alerts
python -m landslide_gnn predict --checkpoint runs/proposed/best.pt \
    --stations data/stations.csv --observations data/live.csv --output runs/alerts.csv
```

`evaluate` accepts `--set decision.tau=...` to trace detection-rate / false-alarm
curves without retraining. In `predict`, stations that are offline get a
prediction inferred from their neighbours and are flagged
`inferred_from_neighbors=True`.

## Baselines and ablations (paper Section VI-E)

```bash
bash scripts/run_experiments.sh data/stations.csv data/observations.csv runs/exp "1 2 3 4 5"
python scripts/summarize_runs.py runs/exp --out runs/exp/all_runs.csv
```

| `--set` option | Meaning |
|---|---|
| `model.type=gat` | proposed edge-aware GAT (default) |
| `model.type=gatv2` | GATv2 attention score |
| `model.type=gcn` / `sage` | GCN / GraphSAGE baselines on the same graph |
| `model.type=none` | per-station 1D-CNN, no message passing |
| `graph.use_flow_edges=false` | kNN edges only |
| `model.use_edge_features=false` | attention without edge features |
| `loss.focal_gamma=0` | weighted cross-entropy instead of focal loss |
| `loss.class_weighting=none\|inverse\|effective` | class weighting scheme |
| `model.pool=mean\|mean_last\|mean_max` | temporal pooling (see below) |
| `model.layers`, `model.heads`, `graph.k`, `data.window` | depth, heads, neighbours, window length |

Threshold rules, gradient-boosted trees and the earlier CNN-DQN design are not
part of this package.

## Notes and deviations from the manuscript

1. **Temporal pooling.** The manuscript, Eq. (24), averages the last convolutional
   feature map over the whole window. Averaging over a 10-hour window dilutes the
   final samples, where pre-failure acceleration shows up. The default here is
   `model.pool=mean_last`, which concatenates the average with the most recent
   column. On the synthetic smoke test below, this was the difference between
   alerting only after failure and alerting hours before it. Use `model.pool=mean`
   to reproduce the manuscript exactly, and update Eq. (24) if you keep the default.
2. **Parameter count.** For the default configuration the counts are: encoder
   16 896 (12 800 with `pool=mean`), each attention layer 4 608, classifier
   2 212, for a total of **28 324** (24 228 with `pool=mean`).
3. **Class weighting.** With `loss.beta=0.999`, effective-number weights become
   almost uniform once a class has more than a few thousand samples. Check the
   weights printed at the start of training. If Evacuate is not clearly
   up-weighted, try `loss.beta=0.9999` or `0.99999`, or `loss.class_weighting=inverse`,
   and select on the validation split.

## Synthetic data (pipeline testing only)

```bash
python -m landslide_gnn synth --output data/synthetic        # 2 sites x 12 stations, 40 events
python -m landslide_gnn train --stations data/synthetic/stations.csv \
    --observations data/synthetic/observations.csv --output runs/synthetic
```

The generator writes CSVs in the documented format. Its slope physics is a
deliberate caricature: accelerating precursors that propagate downslope, rain
storms, packet loss and nuisance disturbances. It exists only to check that the
code runs and that the graph can carry upslope information. **Numbers obtained
on it say nothing about real slopes and must not be reported.** For reference,
one run per setting (25 epochs, `data.stride=3`, 6 test events of which only 2
contain a failure) gave:

| pooling | macro-F1 | Evacuate recall | failures detected before onset | median lead |
|---|---|---|---|---|
| `mean` (manuscript) | 0.85 | 0.36 | 0 / 2 | — |
| `mean_last` (default) | 0.85 | 0.61 | 2 / 2 | 7 h |

## Repository layout

```
landslide_gnn/
  schema.py        CSV column definitions and class names
  config.py        configuration dataclasses (YAML)
  data.py          loading, validation, windowing, splitting, batching
  graph.py         kNN + flow-edge graph and edge features
  models/          temporal encoder, edge-aware GAT, GCN/SAGE baselines
  losses.py        class-weighted focal loss
  decision.py      cost-sensitive decision layer with hysteresis
  metrics.py       classification and early-warning metrics
  engine.py        train / evaluate / predict loops
  synthetic.py     synthetic data generator (testing only)
  cli.py           command-line interface
configs/default.yaml   default hyperparameters
docs/DATA_FORMAT.md    CSV specification
examples/              CSV templates and a small synthetic sample
scripts/               experiment runner and results table
tests/                 unit and end-to-end tests
```

## Citation

See `CITATION.cff`. Please cite the paper if you use this code.

## Acknowledgment

Supported by the Indonesian Collaborative Research Program (Program Riset
Kolaborasi Indonesia, RKI) 2026.

## License

MIT, see `LICENSE`.
