#!/usr/bin/env bash

set -u

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR" || exit 1

DEFAULT_ID_DATASETS=("mnist" "fashion_mnist" "kmnist" "emnist")
DEFAULT_OOD_DATASETS=("mnist" "fashion_mnist" "kmnist" "emnist" "omniglot" "not_mnist")

if [[ $# -gt 0 ]]; then
  ID_DATASETS=("$@")
else
  ID_DATASETS=("${DEFAULT_ID_DATASETS[@]}")
fi

source .venv/bin/activate

# Set TOP_K>0 to enable goodness-based top-k aggregation in PatternOOD.
# Example: TOP_K=3 bash launch_ood_nonconv_optimized.sh mnist
TOP_K="${TOP_K:-0}"

python3 scripts/run_ood_nonconv.py \
  --experiments_dir "experiments/train" \
  --output_root "experiments/OOD/nonconv" \
  --id_datasets "kmnist" \
  --ood_datasets "fashion_mnist" \
  --latest_only \
  --include_ann \
  --include_snn \
  --device "cuda:0" \
  --batch_size "512" \
  --resize "28" \
  --total_samples "5000" \
  --inverse_base "400" \
  --zero_scale "0.0" \
  --bounds "0.2" \
  --latent_depth "1" \
  --distance "euclidean" \
  --top_k "$TOP_K" \
  --verbose "1"
