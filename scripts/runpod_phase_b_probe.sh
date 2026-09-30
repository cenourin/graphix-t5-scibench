#!/usr/bin/env bash
# Phase-B first session (docs/RUNPOD_PHASE_B.md): FP32 strict baseline, no study.
#   bash /workspace/project/graphix-t5-scibench/scripts/runpod_phase_b_probe.sh [all|smoke|steps50|probe|report]
# all (default) = preflight -> smoke (train, checkpoint, resume, eval with generation) ->
#   50 steps (T10 replayed) -> probe (4 cells, PROBE_STEPS each, profiling off) -> profile
#   (RGAT cells, GRAPHIX_PROFILE=1) -> dev generation speed -> report -> prune weights ->
#   "STOP THE POD NOW".
# Never starts Optuna or a final training; never enables TF32/BF16/compile (entry.py fails a
# run if any of them is on). One session at a time (flock on $WS/benchmarks). Every stage
# has a timeout; any failure stops the session with exit != 0.
# Env: SESSION (phase_b_s1), REPORT_NAME (4090_baseline_fp32), PROBE_STEPS (250),
#   PROFILE_STEPS (40), DEVGEN_N (32), PROBE_CELLS ("spider_rgat spider_plain
#   sciencebenchmark_rgat sciencebenchmark_plain"), STAGE_TIMEOUT_SCALE (1; e.g. 4 on a GTX
#   1070), SESSION_MAX_MIN (180), KEEP_WEIGHTS=1 keeps the run weights, RUNPOD_GPU_USD_PER_HOUR (0.74).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runpod/common.sh"
WHAT=${1:-all}
PROBE_CELLS=${PROBE_CELLS:-"spider_rgat spider_plain sciencebenchmark_rgat sciencebenchmark_plain"}
SCALE=${STAGE_TIMEOUT_SCALE:-1}
DEADLINE=$(( $(date +%s) + ${SESSION_MAX_MIN:-180} * 60 ))
export GRAPHIX_IMAGE_DIGEST=${GRAPHIX_IMAGE_DIGEST:-$IMAGE_DIGEST}

take_lock
if [ "$WHAT" = all ] && [ -e "$SOUT/stages.jsonl" ]; then
  die "$SOUT already has a session; use a new SESSION=... (outputs are never mixed)"
fi
mkdir -p "$SOUT" "$SCKPT"
cd "$PROJ"

finish() {
  local rc=$?
  stop_monitors
  echo
  if [ $rc -eq 0 ]; then echo "SESSION $SESSION FINISHED OK. Report: $BENCH/$REPORT_NAME.md"
  else echo "SESSION $SESSION FAILED (rc=$rc). Logs: $SOUT/logs"; fi
  echo "=================================================================="
  echo "  STOP THE POD NOW  (outputs are on the network volume: $WS)"
  echo "=================================================================="
}
trap finish EXIT
trap 'exit 130' INT TERM

# st NAME MINUTES CMD... : stage with timeout scaled and capped by the session deadline
st() {
  local name=$1 min=$(( $2 * SCALE )); shift 2
  local left=$(( (DEADLINE - $(date +%s)) / 60 ))
  [ "$left" -gt 0 ] || die "session time limit (${SESSION_MAX_MIN:-180} min) reached before $name"
  [ "$min" -le "$left" ] || min=$left
  stage "$name" "$min" "$@"
}

start_monitors
if [ "$WHAT" = all ]; then
  st preflight 20 env PREFLIGHT_UNDER_LAUNCHER=1 bash scripts/runpod_preflight.sh
fi
if [ "$WHAT" = all ] || [ "$WHAT" = smoke ]; then
  st smoke_train 20 python scripts/runpod/runs.py smoke_train
  st smoke_resume 20 python scripts/runpod/runs.py smoke_resume
  st smoke_eval 20 python scripts/runpod/runs.py smoke_eval
fi
if [ "$WHAT" = all ] || [ "$WHAT" = steps50 ]; then
  st steps50 20 python scripts/runpod/runs.py steps50
fi
if [ "$WHAT" = all ] || [ "$WHAT" = probe ]; then
  for cell in $PROBE_CELLS; do
    st "probe_$cell" 45 python scripts/runpod/runs.py "probe_$cell"
    case $cell in
      *_rgat) b=${cell%_rgat}
              st "profile_${b}_rgat" 20 python scripts/runpod/runs.py "profile_${b}_rgat"
              st "devgen_$b" 20 python scripts/runpod/runs.py "devgen_$b" ;;
    esac
  done
fi
st report 10 python scripts/runpod/report.py
if [ "${KEEP_WEIGHTS:-0}" != 1 ]; then
  find "$SCKPT" \( -name '*.safetensors' -o -name 'optimizer.pt' -o -name 'scheduler.pt' -o -name 'rng_state*.pth' \) -delete
  echo "pruned model/optimizer weights under $SCKPT (configs, logs, trainer_state and results kept)"
fi
