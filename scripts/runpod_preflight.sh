#!/usr/bin/env bash
# Pre-flight of a phase-B pod session (docs/RUNPOD_PHASE_B.md). Read-only; exits != 0 at the
# first failed precondition. Run inside the pod (image silveirabruno/graphix-modern@sha256:
# 371f61af...), from anywhere:
#   bash /workspace/project/graphix-t5-scibench/scripts/runpod_preflight.sh
# Env: EXPECT_COMMIT=<sha prefix> (recommended), EXPECT_GPU (default "RTX 4090"),
#      EXPECT_CAPABILITY (8.9), EXPECT_MIN_VRAM_GB (23), MIN_FREE_GB (20),
#      PREFLIGHT_FULL_HASH=1 re-hashes the 15 GB of SQLite files too (default: sizes only for
#      *.sqlite, sha256 for every other file of the manifest).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runpod/common.sh"
[ "${PREFLIGHT_UNDER_LAUNCHER:-0}" = 1 ] || take_lock
mkdir -p "$SOUT"
ok() { printf "  %-44s ok   %s\n" "$1" "${2:-}"; }

echo "== volume and layout"
mountpoint -q "$WS" || die "$WS is not a mount point (network volume not attached?)"
ok "$WS mounted"
touch "$WS/.preflight_write_test" && rm -f "$WS/.preflight_write_test" || die "$WS not writable"
ok "$WS writable"
free_gb=$(df -BG --output=avail "$WS" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "${MIN_FREE_GB:-20}" ] || die "only ${free_gb} GB free on $WS (need ${MIN_FREE_GB:-20})"
ok "free space on $WS" "${free_gb} GB"
for d in "$CKPT" "$OUTS" "$BENCH" "$CACHE"; do mkdir -p "$d"; done
ok "checkpoints/ outputs/ benchmarks/ cache/"

echo "== code"
[ -d "$PROJ/graphix_modern" ] || die "no repository at $PROJ"
cd "$PROJ"
commit=$(python3 scripts/runpod/gitinfo.py .) || die "cannot read the commit of $PROJ"
ok "commit" "$commit (branch $(python3 -c 'import sys;sys.path.insert(0,"scripts/runpod");import gitinfo;print(gitinfo.branch("."))'))"
if [ -n "${EXPECT_COMMIT:-}" ]; then
  case "$commit" in "$EXPECT_COMMIT"*) ok "commit matches EXPECT_COMMIT" ;; *) die "commit $commit != EXPECT_COMMIT $EXPECT_COMMIT" ;; esac
fi
grep -q "port-phase-a-final" .git/packed-refs 2>/dev/null || [ -f .git/refs/tags/port-phase-a-final ] \
  || die "tag port-phase-a-final not present in the clone (fetch tags)"
ok "tag port-phase-a-final present"
for f in scripts/runpod/entry.py scripts/runpod/runs.py scripts/port_entry_harness.py graphix_modern/train.py; do
  echo "$(sha256sum "$f")"
done > "$SOUT/code_hashes.txt"
ok "hashes of the session's entry scripts" "$SOUT/code_hashes.txt"

echo "== data"
for l in data_all_in/data data_all_in/t5-base-st; do
  [ -e "$l/." ] || die "$PROJ/$l missing: ln -sfn $DATA ${PROJ}/data_all_in/data; ln -sfn $DATA/t5-base-st ${PROJ}/data_all_in/t5-base-st"
done
[ "$(readlink -f data_all_in/data)" = "$(readlink -f "$DATA")" ] || die "data_all_in/data does not resolve to $DATA"
ok "data_all_in/data -> $DATA, t5-base-st present"
M=$DATA/manifests/DATA_MANIFEST.sha256
[ -f "$M" ] || die "$M missing"
[ "$(sha256sum < "$M" | cut -d' ' -f1)" = "$DATA_MANIFEST_SHA256" ] || die "manifest hash differs from the uploaded one"
[ "$(wc -l < "$M")" = 450 ] || die "manifest has $(wc -l < "$M") lines, expected 450"
ok "manifest" "450 files, sha256 ${DATA_MANIFEST_SHA256:0:12}..."
[ -f "$DATA/manifests/DATA_VERIFIED" ] || die "no $DATA/manifests/DATA_VERIFIED: run the full sha256sum -c validation first"
ok "full validation recorded" "$(cat "$DATA/manifests/DATA_VERIFIED")"
( cd "$DATA"
  if [ "${PREFLIGHT_FULL_HASH:-0}" = 1 ]; then
    sha256sum --quiet -c manifests/DATA_MANIFEST.sha256
  else
    grep -v '\.sqlite$' manifests/DATA_MANIFEST.sha256 | sha256sum --quiet -c -
    for f in $(grep '\.sqlite$' manifests/DATA_MANIFEST.sha256 | awk '{print $2}'); do [ -s "$f" ] || { echo "empty/missing $f"; exit 1; }; done
  fi ) > "$SOUT/preflight_data_check.log" 2>&1 || { cat "$SOUT/preflight_data_check.log"; die "data check failed"; }
ok "data re-check" "$([ "${PREFLIGHT_FULL_HASH:-0}" = 1 ] && echo 'all files hashed' || echo 'sha256 of non-SQLite files, SQLite present')"

echo "== GPU and environment"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no GPU runtime in this container"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | sed 's/^/  /'
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ge 1 ] || die "no GPU visible"
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" = 0 ] || die "$busy process(es) already using the GPU (another probe running?)"
ok "GPU idle"
export GRAPHIX_IMAGE_DIGEST=${GRAPHIX_IMAGE_DIGEST:-$IMAGE_DIGEST}
python3 scripts/runpod/env_check.py check "$SOUT/preflight.json" || die "environment check failed (see above)"
echo "PREFLIGHT PASSED (commit $commit)"
