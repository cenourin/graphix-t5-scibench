#!/usr/bin/env bash
# Same study as scripts/run_t5base_study.sh, for a cloud pod (e.g. RunPod) whose container
# IS the training image eyuansu62/graphix-text-to-sql:v2, so there is no docker-in-docker.
# See docs/RUNPOD.md for GPU choice, data transfer and the volume layout.
# Usage (inside the pod, from the repo root on the persistent volume):
#   scripts/run_t5base_study_pod.sh [--smoke | --probe] [extra args for seq2seq/run_t5base_study.py]
#   GRAPHIX_PRICE_PER_HOUR=0.74 BUDGET_USD=15 [STOP_POD_WHEN_DONE=1] scripts/run_t5base_study_pod.sh
# Order (docs/RUNPOD.md §4): --smoke, then --probe (cost per cell), then the real study.
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
# torch 1.9 + CUDA 11.1 ships SASS up to sm_86 and DGL 0.8.2 up to sm_80. By CUDA's binary
# compatibility they run on any 8.x GPU (RTX 4090 = 8.9), but not on 9.x (H100).
if cap[0] != 8 and cap > (8, 6):
    raise SystemExit("GPU compute capability %s.%s is not supported by this image's kernels" % cap)
if cap > (8, 6):
    print("NOTE: GPU newer than the image's kernels (8.6); run --smoke first.")
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
export GRAPHIX_CODE_COMMIT="$(cat "$SNAP/COMMIT")$( [ -s "$SNAP/uncommitted.diff" ] && echo +uncommitted )"

# Session cap: MAX_HOURS, or BUDGET_USD / GRAPHIX_PRICE_PER_HOUR. At the cap the whole
# process group gets SIGTERM: a running trial becomes FAIL on the next launch, a final
# training resumes from its last epoch checkpoint. Only this process is capped; the pod
# itself keeps billing until stopped (STOP_POD_WHEN_DONE=1 does it if runpodctl exists).
LIMIT_H="${MAX_HOURS:-}"
if [ -z "$LIMIT_H" ] && [ -n "${BUDGET_USD:-}" ]; then
  [ -n "${GRAPHIX_PRICE_PER_HOUR:-}" ] || { echo "BUDGET_USD needs GRAPHIX_PRICE_PER_HOUR"; exit 1; }
  LIMIT_H=$(python -c "print(round($BUDGET_USD / $GRAPHIX_PRICE_PER_HOUR, 2))")
fi
echo "session cap: ${LIMIT_H:-none} h"
nohup bash -c '
  if [ -n "$0" ]; then timeout --signal=TERM --kill-after=2m "${0}h" python seq2seq/run_t5base_study.py "$@"
  else python seq2seq/run_t5base_study.py "$@"; fi
  rc=$?
  [ "$rc" = 124 ] && echo "{\"time\": \"$(date -u "+%Y-%m-%d %H:%M:%S")\", \"event\": \"session_cap_reached\", \"hours\": \"$0\"}" >> "$GRAPHIX_STUDY_DIR/t5base_study.jsonl"
  if [ "${STOP_POD_WHEN_DONE:-0}" = 1 ]; then
    if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then runpodctl stop pod "$RUNPOD_POD_ID"
    else echo "STOP_POD_WHEN_DONE: runpodctl or RUNPOD_POD_ID missing; stop the pod by hand"; fi
  fi' "$LIMIT_H" "$@" > "$GRAPHIX_STUDY_DIR/pipeline.out" 2>&1 &
echo "started pid $!; follow: tail -f $GRAPHIX_STUDY_DIR/t5base_study.jsonl"
