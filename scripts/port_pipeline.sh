#!/usr/bin/env bash
# Phase-A equivalence pipeline (PORTABILIDADE.md §5): runs every test implemented so far,
# in order, each in its environment, and stops at the first failure.
#   scripts/port_pipeline.sh            # T2 (skipped if already passed), T1, T3, ...
#   scripts/port_pipeline.sh --force-t2 # also redo the 15-min A0 export + T2
# Summary: data_all_in/data/port_tests/pipeline_summary.json (commit, images, results).
# New tests (T4..T10) are added as steps below when their port step lands.
set -uo pipefail
cd "$(dirname "$0")/.."
LEGACY=eyuansu62/graphix-text-to-sql:v2
MODERN=silveirabruno/graphix-modern@sha256:7f9513344460f65cda11f29236ea724025ae0786c86bf441efc5d6c14d276a75
OUT=data_all_in/data/port_tests
mkdir -p "$OUT"
FORCE_T2=0; [ "${1:-}" = "--force-t2" ] && FORCE_T2=1

COMMIT="$(git rev-parse HEAD)$(git diff --quiet HEAD -- seq2seq graphix_modern scripts tests configs || echo +uncommitted)"
U=(--user "$(id -u):$(id -g)" -e HOME=/tmp -e DGLBACKEND=pytorch -e GRAPHIX_CODE_COMMIT="$COMMIT")
M=(-v "$PWD/seq2seq:/app/seq2seq:ro" -v "$PWD/graphix_modern:/app/graphix_modern:ro"
   -v "$PWD/scripts:/app/scripts:ro" -v "$PWD/tests:/app/tests:ro" -v "$PWD/configs:/app/configs:ro"
   -v "$PWD/data_all_in:/app/data_all_in" -w /app)
# legacy|modern [docker options] -- command...
in_image() {
  local image=$1; shift; local opts=()
  while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
  docker run --rm "${U[@]}" "${M[@]}" "${opts[@]}" "$image" "$@"
}
legacy() { in_image "$LEGACY" "$@"; }
modern() { in_image "$MODERN" "$@"; }
declare -a RESULTS=()
finish() {
  python3 - "$COMMIT" "$1" "${RESULTS[@]}" <<'EOF'
import json, sys, time
commit, status, results = sys.argv[1], sys.argv[2], sys.argv[3:]
summary = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": commit,
           "legacy_image": "eyuansu62/graphix-text-to-sql:v2",
           "modern_image": "silveirabruno/graphix-modern@sha256:7f9513344460f65cda11f29236ea724025ae0786c86bf441efc5d6c14d276a75",
           "status": status, "steps": [dict(zip(("test", "result"), r.split("=", 1))) for r in results]}
json.dump(summary, open("data_all_in/data/port_tests/pipeline_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
EOF
  [ "$1" = passed ]
}
step() {  # step NAME CMD... : run, record, stop the pipeline on failure
  local name=$1; shift
  echo "=== $name"
  if "$@"; then RESULTS+=("$name=passed"); else RESULTS+=("$name=FAILED"); finish "failed at $name"; exit 1; fi
}

# --- T2: A0 graph export reproduces every legacy graph (legacy image) ---------------
t2() {
  if [ $FORCE_T2 = 0 ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('data_all_in/data/graph_export/T2_report.json'))['passed'] else 1)" 2>/dev/null; then
    echo "T2 already passed (data_all_in/data/graph_export/T2_report.json); use --force-t2 to redo"; return 0
  fi
  legacy -- bash -c "python scripts/export_graph_pedia.py && python scripts/validate_graph_export.py"
}
# --- T1: tokenizer, all 14 642 examples ------------------------------------------------
t1() {
  legacy -- python scripts/port_t1_tokenizer.py dump legacy >/dev/null 2>&1 &&
  modern -- python scripts/port_t1_tokenizer.py dump modern >/dev/null 2>&1 &&
  python3 scripts/port_t1_tokenizer.py compare
}
# --- T3: RGAT layer, same weights and inputs, GPU reference ---------------------------
t3() {
  { [ -f "$OUT/T3/inputs.npz" ] || legacy -- python scripts/port_t3_rgat.py inputs; } &&
  legacy --gpus all -- python scripts/port_t3_rgat.py run legacy cuda >/dev/null 2>&1 &&
  modern --gpus all -- python scripts/port_t3_rgat.py run modern cuda >/dev/null 2>&1 &&
  modern -- python scripts/port_t3_rgat.py compare cuda
}

step T2 t2
step T1 t1
step T3 t3
# (T4..T10 are appended here as the port advances)
finish passed
