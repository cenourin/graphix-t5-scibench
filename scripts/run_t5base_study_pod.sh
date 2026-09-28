#!/usr/bin/env bash
# Same study as scripts/run_t5base_study.sh, for a cloud pod (e.g. RunPod) whose container
# IS the training image eyuansu62/graphix-text-to-sql:v2, so there is no docker-in-docker.
# See docs/RUNPOD.md for GPU choice, data transfer and the volume layout.
# Usage (inside the pod, from the repo root on the persistent volume):
#   scripts/run_t5base_study_pod.sh [--smoke] [extra args for seq2seq/run_t5base_study.py]
# Runs in the background with nohup; progress in $VOL/optuna_studies/t5base_study.jsonl.
# Re-running resumes, as locally.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VOL="${VOL:-/workspace}"                       # RunPod's persistent volume
export GRAPHIX_STUDY_DIR="$VOL/optuna_studies"
export GRAPHIX_RUNS_DIR="$VOL/runs"
export GRAPHIX_OPTUNA_PKGS="$GRAPHIX_STUDY_DIR/.pkgs"
export DGLBACKEND=pytorch HOME="${HOME:-/tmp}"
mkdir -p "$GRAPHIX_STUDY_DIR" "$GRAPHIX_RUNS_DIR/optuna" /transformers_cache

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA in this pod"
cap = torch.cuda.get_device_capability(0)
print("GPU:", torch.cuda.get_device_name(0), "capability", cap)
# torch 1.9 + CUDA 11.1 ships kernels up to sm_86 (Ampere). Newer GPUs are untested here.
if cap > (8, 6):
    print("WARNING: GPU newer than Ampere; run --smoke first and check it actually trains.")
EOF

[ -f "$REPO/data_all_in/data/splits/spider_train.json" ] || { echo "data missing: see docs/RUNPOD.md"; exit 1; }
[ -d "$GRAPHIX_OPTUNA_PKGS/optuna" ] || pip install --quiet --target "$GRAPHIX_OPTUNA_PKGS" 'optuna<4' 'sqlalchemy<2' 'alembic<1.13'
rm -rf "$GRAPHIX_OPTUNA_PKGS"/typing_extensions*

# Same snapshot rule as the local launcher (RISCOS.md R0.2): run a frozen copy of the code.
SNAP="$VOL/study_code"
if [ ! -d "$SNAP" ]; then
  mkdir -p "$SNAP"
  cp -r "$REPO/seq2seq" "$REPO/configs" "$SNAP/"
  git -C "$REPO" rev-parse HEAD > "$SNAP/COMMIT"
  git -C "$REPO" diff HEAD -- seq2seq configs > "$SNAP/uncommitted.diff"
fi
# The code expects to run from the repo root (relative data_all_in/ paths), so run from a
# directory that holds the snapshot code next to a link to the real data.
RUN="$VOL/study_run"
mkdir -p "$RUN"
ln -sfn "$SNAP/seq2seq" "$RUN/seq2seq"
ln -sfn "$SNAP/configs" "$RUN/configs"
ln -sfn "$REPO/data_all_in" "$RUN/data_all_in"
cd "$RUN"
nohup python seq2seq/run_t5base_study.py "$@" > "$GRAPHIX_STUDY_DIR/pipeline.out" 2>&1 &
echo "started pid $!; follow: tail -f $GRAPHIX_STUDY_DIR/t5base_study.jsonl"
