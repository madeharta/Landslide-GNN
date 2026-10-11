#!/usr/bin/env bash
# Baselines and ablations of Section VI-E, each with several seeds.
#   usage: bash scripts/run_experiments.sh data/stations.csv data/observations.csv [runs/exp] [seeds]
# Then summarise with:  python scripts/summarize_runs.py runs/exp
set -euo pipefail
ST=${1:?stations.csv}; OB=${2:?observations.csv}; OUT=${3:-runs/exp}; SEEDS=${4:-"1 2 3 4 5"}

declare -A EXP=(
  [proposed]=""
  [base_none_cnn]="model.type=none"
  [base_gcn]="model.type=gcn"
  [base_sage]="model.type=sage"
  [abl_gatv2]="model.type=gatv2"
  [abl_no_flow_edges]="graph.use_flow_edges=false"
  [abl_no_edge_features]="model.use_edge_features=false"
  [abl_no_focal]="loss.focal_gamma=0"
  [abl_no_class_weights]="loss.class_weighting=none"
  [abl_pool_mean_paper]="model.pool=mean"
  [abl_L1]="model.layers=1"
  [abl_L3]="model.layers=3"
  [abl_K1]="model.heads=1"
  [abl_K8]="model.heads=8"
)

for name in "${!EXP[@]}"; do
  for s in $SEEDS; do
    dir="$OUT/$name/seed$s"
    [ -f "$dir/summary.json" ] && { echo "skip $dir"; continue; }
    echo ">>> $name seed $s"
    python -m landslide_gnn train --config configs/default.yaml --stations "$ST" --observations "$OB" \
      --output "$dir" --set train.seed=$s ${EXP[$name]}
  done
done
