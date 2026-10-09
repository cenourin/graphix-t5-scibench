#!/usr/bin/env bash
# Offline scoring and pre-registered analysis of the t5-base study run on the RTX 4090
# (2026-10-06), as PROTOCOLO.md §6-7 and its 2026-09-28 deviations prescribe: the same offline
# scorer for the 4 cells (scripts/score_predictions.py: ElementAI Spider exact-match and
# test-suite execution evaluators vendored in third_party/, see third_party/PROVENANCE.md),
# after training, on an idle machine, inside the image the scorer documents, one cell at a time;
# then scripts/analyze_study.py unchanged (EM/EX with bootstrap CIs, exact McNemar, Holm over 4
# tests, ScienceBenchmark without leaked examples, per database).
#   bash docs/study_4090/score.sh            (from the repo root)
set -euo pipefail
STUDY=train_db_id/study_4090_2026-10-06            # copy of /workspace/study (sha256 manifest next to it)
OUT=train_db_id/rescore_study_4090                 # per-example scores (new directory)
DOC=docs/study_4090
IMAGE=eyuansu62/graphix-text-to-sql@sha256:1fb86bb618456dbc0dba73d13fd7af7219eda4bad22ad9d3a476254dd66d6614
mkdir -p "$OUT"

{ echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_commit: $(git rev-parse HEAD)"
  echo "scorer_image: $IMAGE"
  echo "study_copy_manifest_sha256: $(sha256sum ${STUDY}.sha256 | cut -d' ' -f1)"
  docker run --rm --entrypoint python "$IMAGE" -c "import sys,sqlite3;print('python:',sys.version.split()[0]);print('sqlite:',sqlite3.sqlite_version)"
  echo "file hashes:"
  sha256sum scripts/score_predictions.py scripts/analyze_study.py seq2seq/metrics/spider/*.py \
    third_party/spider/evaluation.py third_party/spider/process_sql.py \
    third_party/test_suite/evaluation.py third_party/test_suite/exec_eval.py \
    third_party/test_suite/parse.py third_party/test_suite/process_sql.py | sed 's/^/  /'
  echo "predictions:"
  for c in spider-rgat spider-plain sciencebenchmark-rgat sciencebenchmark-plain; do
    sha256sum $STUDY/final/study-t5base-$c/dev_eval/run/predictions_eval_None.json | sed 's/^/  /'
  done
} > $DOC/scoring_environment.txt

for c in spider-rgat spider-plain sciencebenchmark-rgat sciencebenchmark-plain; do
  echo "=== $(date -u +%H:%M:%S) scoring $c"
  docker run --rm --cpus 1 --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/app -w /app --entrypoint python "$IMAGE" \
    scripts/score_predictions.py $STUDY/final/study-t5base-$c/dev_eval/run/predictions_eval_None.json \
    $OUT/study-t5base-$c > $OUT/study-t5base-$c.log 2>&1
  tail -12 $OUT/study-t5base-$c.log
done

python3 scripts/analyze_study.py --scores-dir $OUT --out $DOC/RESULTADOS.md
cp $OUT/study-t5base-*.summary.json $OUT/study-t5base-*.jsonl $DOC/
sha256sum $DOC/study-t5base-*.jsonl $DOC/study-t5base-*.summary.json $DOC/RESULTADOS.md > $DOC/outputs.sha256
