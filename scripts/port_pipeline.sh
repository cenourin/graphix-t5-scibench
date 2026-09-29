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
MODERN=silveirabruno/graphix-modern@sha256:371f61af7521069e62bb8973c2c42584377659440b5902d025876fb94d859037
OUT=data_all_in/data/port_tests
mkdir -p "$OUT"
FORCE_T2=0; [ "${1:-}" = "--force-t2" ] && FORCE_T2=1

COMMIT="$(git rev-parse HEAD)$(git diff --quiet HEAD -- seq2seq graphix_modern scripts tests configs || echo +uncommitted)"
U=(--user "$(id -u):$(id -g)" -e HOME=/tmp -e DGLBACKEND=pytorch -e GRAPHIX_CODE_COMMIT="$COMMIT"
   -e HF_HOME=/tmp/hf -e HF_MODULES_CACHE=/tmp/hf/modules -e HF_DATASETS_CACHE=/tmp/hf/datasets)
M=(-v "$PWD/seq2seq:/app/seq2seq:ro" -v "$PWD/graphix_modern:/app/graphix_modern:ro"
   -v "$PWD/scripts:/app/scripts:ro" -v "$PWD/tests:/app/tests:ro" -v "$PWD/configs:/app/configs:ro"
   -v "$PWD/third_party:/app/third_party:ro" -v "$PWD/data_all_in:/app/data_all_in" -w /app)
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
           "modern_image": "silveirabruno/graphix-modern@sha256:371f61af7521069e62bb8973c2c42584377659440b5902d025876fb94d859037",
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

# --- A4 build check: the ported modeling_t5.py is exactly what the builder generates ---
a4_build() {
  { [ -f data_all_in/t5-base-st/manifest.json ] || modern -- python scripts/port_convert_t5_safetensors.py; } &&
  cp graphix_modern/modeling_t5.py /tmp/modeling_t5.committed.py &&
  docker run --rm "${U[@]}" -v "$PWD:/app" -w /app "$MODERN" python scripts/port_build_modeling_t5.py >/dev/null 2>&1 &&
  cmp -s graphix_modern/modeling_t5.py /tmp/modeling_t5.committed.py
}
# --- T4: structure and legacy checkpoint loading (strict) ------------------------------
t4() {
  legacy -- python scripts/port_t4_structure.py dump legacy >/dev/null 2>&1 &&
  modern -- python scripts/port_t4_structure.py dump modern >/dev/null 2>&1 &&
  modern -- python scripts/port_t4_structure.py compare
}
# --- T5: training forward (use_cache=False), same weights/inputs, GPU reference -------
t5() {
  { [ -f "$OUT/T5/inputs.npz" ] || legacy -- python scripts/port_t5_forward.py inputs; } &&
  legacy --gpus all -- python scripts/port_t5_forward.py run legacy >/dev/null 2>&1 &&
  modern --gpus all -- python scripts/port_t5_forward.py run modern >/dev/null 2>&1 &&
  modern -- python scripts/port_t5_forward.py compare
}

step T2 t2
step T1 t1
step T3 t3
step A4_build a4_build
step T4 t4
# --- T6: gradients (R1 spider no-ckpt, R2 spider ckpt, R3 scibench ckpt), GPU reference -
t6() {
  legacy --gpus all -- python scripts/port_t6_gradients.py run legacy >/dev/null 2>&1 &&
  modern --gpus all -- python scripts/port_t6_gradients.py run modern >/dev/null 2>&1 &&
  modern -- python scripts/port_t6_gradients.py compare
}
# --- A6.2a TD: data stage (filters, dev examples, alignment, schemas); run sequentially ---
td() {
  for b in spider sciencebenchmark; do
    legacy -- python scripts/port_td_data.py dump legacy $b >/dev/null 2>&1 &&
    modern -- python scripts/port_td_data.py dump modern $b >/dev/null 2>&1 || return 1
  done
  modern -- python scripts/port_td_data.py compare
}
# --- A6.2b TW: model wrappers (graph_batch, loss), GPU ---------------------------------
tw() {
  legacy --gpus all -- python scripts/port_tw_wrapper.py run legacy >/dev/null 2>&1 &&
  modern --gpus all -- python scripts/port_tw_wrapper.py run modern >/dev/null 2>&1 &&
  modern -- python scripts/port_tw_wrapper.py compare
}
# --- A6.2c TRACE: training-loop semantics vs the legacy trace (N=10, GA=4, 3 epochs) ---
trace() {
  legacy -- python scripts/port_trace_epochs.py legacy stock >/dev/null 2>&1 &&
  modern -- python scripts/port_trace_epochs.py modern ported >/dev/null 2>&1 &&
  python3 - <<'PY'
import json, sys
D = "data_all_in/data/port_tests/TRACE/"
a, b = json.load(open(D + "legacy_stock.json")), json.load(open(D + "modern_ported.json"))
keys = ["max_steps", "global_step", "optimizer_steps", "scheduler_steps", "scheduler_num_training_steps",
        "order", "processed_after_last_update", "updates"]
bad = [k for k in keys if a[k] != b[k]]
print("TRACE", "PASSED" if not bad else "FAILED: %s" % bad)
sys.exit(1 if bad else 0)
PY
}
step T5 t5
step T6 t6
step TD td
step TW tw
step TRACE trace
# (T7..T10 are appended here as the port advances)
finish passed
