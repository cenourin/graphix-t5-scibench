#!/usr/bin/env bash
# t5-base study, arms rgat and plain (Optuna search -> final training -> dev scoring), on Spider and
# ScienceBenchmark. See seq2seq/run_t5base_study.py for the stages.
# Usage: scripts/run_t5base_study.sh [extra args for seq2seq/run_t5base_study.py]
# Runs detached in a container named t5base-study; progress in optuna_studies/t5base_study.jsonl.
# Re-running resumes: finished stages are skipped, the Optuna studies live in optuna_studies/*.db.
#
# The code and configs are copied to a snapshot at launch (RISCOS.md R0.2): editing the
# repo while the study runs must not change what later trials execute.
set -euo pipefail
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${T5BASE_CONTAINER:-t5base-study}"   # T5BASE_CONTAINER=t5base-smoke for the --smoke check
SNAP="$BASE_DIR/train_db_id/${NAME//-/_}_code"
mkdir -p -m 777 "$BASE_DIR/optuna_studies" "$BASE_DIR/train_db_id/optuna"
if [ ! -d "$SNAP" ]; then
  mkdir -p "$SNAP"
  cp -r "$BASE_DIR/seq2seq" "$BASE_DIR/configs" "$SNAP/"
  git -C "$BASE_DIR" rev-parse HEAD > "$SNAP/COMMIT"
  git -C "$BASE_DIR" diff HEAD -- seq2seq configs > "$SNAP/uncommitted.diff"
fi

docker run -d --name "$NAME" --gpus all \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env DGLBACKEND=pytorch --env CUDA_VISIBLE_DEVICES=0 \
  --env GRAPHIX_CODE_COMMIT="$(cat "$SNAP/COMMIT")$( [ -s "$SNAP/uncommitted.diff" ] && echo +uncommitted )" \
  --mount type=bind,source="$BASE_DIR/train_db_id",target=/train_db_id \
  --mount type=bind,source="$BASE_DIR/transformers_cache",target=/transformers_cache \
  --mount type=bind,source="$BASE_DIR/optuna_studies",target=/optuna_studies \
  --mount type=bind,source="$SNAP/configs",target=/app/configs \
  --mount type=bind,source="$SNAP/seq2seq",target=/app/seq2seq \
  --mount type=bind,source="$BASE_DIR/data_all_in",target=/app/data_all_in \
  eyuansu62/graphix-text-to-sql:v2 \
  /bin/bash -c "[ -d /optuna_studies/.pkgs/optuna ] || pip install --quiet --target /optuna_studies/.pkgs 'optuna<4' 'sqlalchemy<2' 'alembic<1.13'; rm -rf /optuna_studies/.pkgs/typing_extensions*; python seq2seq/run_t5base_study.py $*"
