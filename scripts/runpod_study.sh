#!/usr/bin/env bash
# t5-base study on the pod (docs/RUNPOD_STUDY.md): graphix_modern/study.py, strict FP32, in the
# background, surviving the web terminal. Everything persists on the volume under the study root
# ($WS/study, or $WS/study_smoke with --smoke):
#   launcher/pid             PID of the detached runner (its process group holds the study)
#   launcher/state.json      starting | running | finished | failed | stopped, rc, times, commit
#   launcher/orchestrator.log  orchestrator output (each run also logs to its own train.log)
#   launcher/gpu.csv, cpu.csv  monitors (10 s / 1 s)
#   events.jsonl, optuna/, trials/, final/  the orchestrator's own outputs
#
#   EXPECT_COMMIT=<sha> bash scripts/runpod_study.sh start [--smoke] [study.py args...]
#   bash scripts/runpod_study.sh status [--smoke]     read-only summary
#   bash scripts/runpod_study.sh logs   [--smoke]     follow the orchestrator log (Ctrl-C only stops tail)
#   bash scripts/runpod_study.sh logs   [--smoke] run follow the log of the run in progress
#   bash scripts/runpod_study.sh stop   [--smoke]     SIGTERM to the study's process group
#
# start: refuses if a study is already running on that root; checks that the orchestrator and
# training files are byte-identical to the ones validated by TE/TS (docs/port/orchestrator/
# tested_files.sha256); runs the full preflight (scripts/runpod_preflight.sh: volume, commit ==
# EXPECT_COMMIT, data hashes, GPU, versions, image fingerprint, strict FP32); then detaches
# (setsid + nohup, stdin closed) and returns. Re-running start after a stop or a crash resumes
# with the validated semantics of graphix_modern/study.py: a trial killed while running is
# marked FAIL and counts; the final run resumes from its last complete checkpoint (incomplete
# ones are moved aside); the early-stopping counter restarts on resume, as in the legacy code.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runpod/common.sh"
CMD=${1:-status}; shift || true
SMOKE=0
if [ "${1:-}" = --smoke ]; then SMOKE=1; shift; fi
ROOT=$WS/study$([ $SMOKE = 1 ] && echo _smoke || true)
LD=$ROOT/launcher
export GRAPHIX_IMAGE_DIGEST=${GRAPHIX_IMAGE_DIGEST:-$IMAGE_DIGEST}
cd "$PROJ"

alive() {  # the runner recorded in launcher/pid is alive
  [ -f "$LD/pid" ] || return 1
  local p; p=$(cat "$LD/pid")
  [ -d "/proc/$p" ] && grep -q "runpod_study.sh" "/proc/$p/cmdline" 2>/dev/null
}

state() {  # state STATUS [RC]
  python3 - "$LD/state.json" "$1" "${2:-}" "$ROOT" <<'EOF'
import json, os, sys, time
path, status, rc, root = sys.argv[1:5]
s = json.load(open(path)) if os.path.exists(path) else {}
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
s.update(status=status, updated=now, root=root)
if status == "running":
    s.update(started=now, pid=os.getppid(), rc=None, ended=None)
if rc != "":
    s.update(rc=int(rc), ended=now)
json.dump(s, open(path + ".tmp", "w"), indent=1)
os.replace(path + ".tmp", path)
EOF
}

case "$CMD" in
start)
  mkdir -p "$LD"
  alive && die "a study is already running on $ROOT (pid $(cat "$LD/pid")); see: bash $0 status$([ $SMOKE = 1 ] && echo ' --smoke' || true)"
  [ -n "${EXPECT_COMMIT:-}" ] || die "set EXPECT_COMMIT=<sha> (the commit of this clone, checked by the preflight)"
  echo "== code validated by TE/TS (docs/port/orchestrator/tested_files.sha256)"
  grep -v '^#' docs/port/orchestrator/tested_files.sha256 | sha256sum -c - || die "orchestrator/training files differ from the validated ones"
  echo "== preflight"
  SESSION=$(basename "$ROOT")_preflight MIN_FREE_GB=${MIN_FREE_GB:-30} bash scripts/runpod_preflight.sh \
    || die "preflight failed: the study was not started"
  python3 -c "import optuna" || die "optuna not importable"
  git_commit=$(python3 scripts/runpod/gitinfo.py .)
  python3 - "$LD/launch.json" "$git_commit" "$SMOKE" "$@" <<'EOF'
import json, sys, time
path, commit, smoke, *args = sys.argv[1:]
rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": commit,
       "smoke": smoke == "1", "study_args": args}
open(path.replace("launch.json", "launches.jsonl"), "a").write(json.dumps(rec) + "\n")
json.dump(rec, open(path, "w"), indent=1)
EOF
  state starting
  rm -f "$LD/pid"
  setsid nohup bash "$0" _run $([ $SMOKE = 1 ] && echo --smoke || true) "$@" </dev/null >>"$LD/orchestrator.log" 2>&1 &
  for _ in $(seq 1 30); do alive && break; sleep 0.5; done
  alive || die "the runner did not start; see $LD/orchestrator.log"
  echo
  echo "STUDY STARTED in the background (pid $(cat "$LD/pid"), root $ROOT)."
  echo "It keeps running if you close this terminal. Follow it with:"
  echo "  bash $0 status$([ $SMOKE = 1 ] && echo ' --smoke' || true)"
  echo "  bash $0 logs$([ $SMOKE = 1 ] && echo ' --smoke' || true)        (Ctrl-C stops only the tail)"
  ;;

_run)  # the detached runner (setsid: own session and process group)
  echo $$ > "$LD/pid"
  state running
  echo "=== $(now) runner $$ started: study.py --root $WS/study $*"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,nounits -l 10 >> "$LD/gpu.csv" 2>/dev/null &
  mon1=$!
  python "$PROJ/scripts/runpod/sampler.py" "$LD/cpu.csv" >/dev/null 2>&1 &
  mon2=$!
  stopped=0
  trap 'stopped=1' TERM INT
  set +e
  python graphix_modern/study.py --root "$WS/study" "$@" &
  child=$!
  wait $child; rc=$?
  if [ $stopped = 1 ]; then kill -TERM $child 2>/dev/null; wait $child 2>/dev/null; fi
  set -e
  kill $mon1 $mon2 2>/dev/null || true
  if [ $stopped = 1 ]; then st=stopped; elif [ $rc = 0 ]; then st=finished; else st=failed; fi
  state $st $rc
  rm -f "$LD/pid"
  echo "=== $(now) study $st (rc=$rc)"
  echo "=================================================================="
  echo "  STUDY $st. STOP THE POD NOW (outputs are on the volume: $ROOT)"
  echo "=================================================================="
  ;;

status)
  python3 scripts/runpod/study_status.py "$ROOT"
  if alive; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader 2>/dev/null \
      | sed 's/^/GPU now: util, mem used, mem total, power = /' || true
  fi
  ;;

logs)
  [ -f "$LD/orchestrator.log" ] || die "no study log at $LD/orchestrator.log"
  if [ "${1:-}" = run ]; then
    f=$(find "$ROOT/trials" "$ROOT/final" \( -name train.log -o -name eval.log \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    [ -n "$f" ] || die "no run log yet"
    echo "== following $f (Ctrl-C stops only the tail)"
    exec tail -n 40 -F "$f"
  fi
  exec tail -n 40 -F "$LD/orchestrator.log"
  ;;

stop)
  alive || die "no study running on $ROOT"
  p=$(cat "$LD/pid")
  echo "sending SIGTERM to the study's process group $p (resume later with start: the running trial"
  echo "will be marked FAIL; a final run resumes from its last complete checkpoint)"
  kill -TERM -- "-$p"
  ;;

*)
  die "usage: bash $0 start|status|logs|stop [--smoke] ..."
  ;;
esac
