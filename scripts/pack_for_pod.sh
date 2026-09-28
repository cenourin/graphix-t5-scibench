#!/usr/bin/env bash
# Pack the gitignored data the t5-base study reads into one tar (paths relative to the repo
# root), with a sha256 manifest. Extract on the pod from the repo root:
#   tar -xf graphix_study_data.tar && sha256sum -c pod_data.sha256
# Usage: scripts/pack_for_pod.sh [output_dir]      (~31 GB; see docs/RUNPOD.md)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$REPO}"
cd "$REPO"
SB=data_all_in/data/sciencebenchmark
FILES=(
  data_all_in/t5-base
  data_all_in/data/train_syntax.json data_all_in/data/dev_syntax.json   # Spider HF loader
  data_all_in/data/spider                                               # Spider DBs + jsons
  data_all_in/data/output/graph_pedia_total.bin
  data_all_in/data/output/seq2seq_train_dataset.json
  data_all_in/data/output/seq2seq_dev_dataset.json
  data_all_in/data/splits
  $SB/output/train_syntax.json $SB/output/dev_syntax.json $SB/output/tables_merged.json
  $SB/output/seq2seq_train_dataset.json $SB/output/seq2seq_dev_dataset.json
  $SB/output/graph_pedia_train.bin.dat $SB/output/graph_pedia_train.bin.dir
  $SB/output/graph_pedia_dev.bin
  $SB/database
)
for f in "${FILES[@]}"; do [ -e "$f" ] || { echo "missing: $f"; exit 1; }; done
find "${FILES[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > pod_data.sha256
tar -cf "$OUT/graphix_study_data.tar" pod_data.sha256 "${FILES[@]}"
ls -lh "$OUT/graphix_study_data.tar"
