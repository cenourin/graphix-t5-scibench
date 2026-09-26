#!/usr/bin/env bash
# Optuna search for t5-small + RGAT on ScienceBenchmark (early stopping + pruning).
# Usage: scripts/run_optuna.sh [extra args for seq2seq/run_optuna_search.py]
#   e.g. scripts/run_optuna.sh --n-trials 12 --timeout-hours 48
# Runs detached in a container named optuna-search; follow with `docker logs -f optuna-search`.
# The study lives in optuna_studies/*.db, so re-running resumes it.
set -euo pipefail
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p -m 777 "$BASE_DIR/optuna_studies" "$BASE_DIR/train_db_id"
SB=data_all_in/data/sciencebenchmark/output

docker run -d --name optuna-search --gpus all \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env DGLBACKEND=pytorch --env CUDA_VISIBLE_DEVICES=0 \
  --env GRAPHIX_TRAIN_GRAPH_PEDIA_PATH=$SB/graph_pedia_train.bin \
  --env GRAPHIX_EVAL_GRAPH_PEDIA_PATH=$SB/graph_pedia_dev.bin \
  --env GRAPHIX_TRAIN_DATASET_PATH=$SB/seq2seq_train_dataset.json \
  --env GRAPHIX_EVAL_DATASET_PATH=$SB/seq2seq_dev_dataset.json \
  --env GRAPHIX_MAX_GRAPH_NODES=1400 \
  --mount type=bind,source="$BASE_DIR/train_db_id",target=/train_db_id \
  --mount type=bind,source="$BASE_DIR/transformers_cache",target=/transformers_cache \
  --mount type=bind,source="$BASE_DIR/optuna_studies",target=/optuna_studies \
  --mount type=bind,source="$BASE_DIR/configs",target=/app/configs \
  --mount type=bind,source="$BASE_DIR/seq2seq",target=/app/seq2seq \
  --mount type=bind,source="$BASE_DIR/data_all_in",target=/app/data_all_in \
  eyuansu62/graphix-text-to-sql:v2 \
  /bin/bash -c "[ -d /optuna_studies/.pkgs/optuna ] || pip install --quiet --target /optuna_studies/.pkgs 'optuna<4' 'sqlalchemy<2' 'alembic<1.13'; rm -rf /optuna_studies/.pkgs/typing_extensions*; python seq2seq/run_optuna_search.py $*"
