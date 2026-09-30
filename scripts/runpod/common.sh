#!/usr/bin/env bash
# Shared settings for the phase-B pod session scripts (docs/RUNPOD_PHASE_B.md). Sourced, not run.
# Everything the session produces goes to the persistent network volume (WS, default /workspace):
#   $WS/project/graphix-t5-scibench[-<sha>]   git clone; data_all_in/data -> $WS/data,
#                                     data_all_in/t5-base-st -> $WS/data/t5-base-st (symlinks)
#   $WS/data         contents of data_all_in/data (scripts/prepare_runpod_data.sh) + t5-base-st,
#                    manifests/ (DATA_MANIFEST.sha256, DATA_VERIFIED), port_tests/T10 (references)
#   $WS/checkpoints  <session>/ run directories (weights pruned at the end of the session)
#   $WS/outputs      <session>/ logs, configs, monitoring, stage timings, results
#   $WS/benchmarks   reports (<name>.json / .md)
#   $WS/cache        HF datasets/modules cache (persists across sessions)
set -euo pipefail

IMAGE_DIGEST=sha256:371f61af7521069e62bb8973c2c42584377659440b5902d025876fb94d859037
IMAGE_REF=silveirabruno/graphix-modern@$IMAGE_DIGEST
PHASE_A_TAG=port-phase-a-final

WS=${WS:-/workspace}
# the repository these scripts live in (a per-commit clone on the volume, see docs/RUNPOD_PHASE_B.md §3)
PROJ=${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DATA=$WS/data
DATA_MANIFEST_SHA256=5f40d54754a93f6718568bcc24afd5249a6eb76ca2494cc3ad5af95cf24f149a  # 450 files, 2026-09-29
CKPT=$WS/checkpoints
OUTS=$WS/outputs
BENCH=$WS/benchmarks
CACHE=$WS/cache
SESSION=${SESSION:-phase_b_s1}
REPORT_NAME=${REPORT_NAME:-4090_baseline_fp32}
SOUT=$OUTS/$SESSION        # this session's logs/configs/monitoring
SCKPT=$CKPT/$SESSION       # this session's run directories

# Runtime environment of every run: fp32 strict, caches on the volume, no silent CPU fallback
export DGLBACKEND=pytorch
export HF_HOME=$CACHE/hf HF_DATASETS_CACHE=$CACHE/hf/datasets HF_MODULES_CACHE=$CACHE/hf/modules
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   # everything is local
export PYTHONPATH=$PROJ PYTHONUNBUFFERED=1
unset GRAPHIX_ALLOW_TF32 GRAPHIX_ALLOW_CPU_RGAT GRAPHIX_ALLOW_OOM_SKIP GRAPHIX_PROFILE || true
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# allocator setting only (no computation changes); used by the phase-A modern runs on the 1070
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

die() { echo "FATAL: $*" >&2; exit 1; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# stage NAME TIMEOUT_MIN CMD... : run with a hard timeout, log, record start/end/rc in stages.jsonl
stage() {
  local name=$1 tmin=$2; shift 2
  mkdir -p "$SOUT/logs"
  local t0; t0=$(date +%s.%N)
  echo "{\"stage\": \"$name\", \"event\": \"start\", \"time\": \"$(now)\", \"epoch_s\": $t0}" >> "$SOUT/stages.jsonl"
  echo "=== [$(now)] $name (timeout ${tmin} min)"
  set +e
  timeout --signal=TERM --kill-after=60 "${tmin}m" "$@" > "$SOUT/logs/$name.log" 2>&1
  local rc=$?
  set -e
  local t1; t1=$(date +%s.%N)
  echo "{\"stage\": \"$name\", \"event\": \"end\", \"time\": \"$(now)\", \"epoch_s\": $t1, \"rc\": $rc}" >> "$SOUT/stages.jsonl"
  if [ $rc -ne 0 ]; then
    tail -40 "$SOUT/logs/$name.log" >&2
    [ $rc -eq 124 ] && die "stage $name timed out after ${tmin} min (log: $SOUT/logs/$name.log)"
    die "stage $name failed rc=$rc (log: $SOUT/logs/$name.log)"
  fi
  echo "    ok ($(python -c "print(round($t1-$t0,1))") s)"
}

# One session at a time on this volume (flock is released automatically if the shell dies)
take_lock() {
  mkdir -p "$BENCH"
  exec 9> "$BENCH/.phase_b_session.lock"
  flock -n 9 || die "another phase-B session holds $BENCH/.phase_b_session.lock (pid $(cat "$BENCH/.phase_b_session.pid" 2>/dev/null))"
  echo $$ > "$BENCH/.phase_b_session.pid"
}

# Background monitors (GPU every 1 s, CPU/RAM every 1 s); stopped by stop_monitors / on exit
start_monitors() {
  mkdir -p "$SOUT/monitor"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
    --format=csv,nounits -l 1 > "$SOUT/monitor/gpu.csv" 2> "$SOUT/monitor/gpu.err" &
  echo $! > "$SOUT/monitor/gpu.pid"
  nvidia-smi dmon -s pucm -o T > "$SOUT/monitor/dmon.log" 2> /dev/null &
  echo $! > "$SOUT/monitor/dmon.pid"
  python "$PROJ/scripts/runpod/sampler.py" "$SOUT/monitor/cpu.csv" > /dev/null 2>&1 &
  echo $! > "$SOUT/monitor/cpu.pid"
}
stop_monitors() {
  for p in gpu dmon cpu; do
    [ -f "$SOUT/monitor/$p.pid" ] && kill "$(cat "$SOUT/monitor/$p.pid")" 2>/dev/null || true
    rm -f "$SOUT/monitor/$p.pid"
  done
}
