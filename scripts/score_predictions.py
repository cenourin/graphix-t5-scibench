#!/usr/bin/env python
"""Per-example EM/EX scoring of a predictions_eval_*.json file, offline.

Uses the same scorers as the training-time metric (seq2seq/metrics/spider/: Spider
exact-match Evaluator and the test-suite execution Evaluator with the same progress
deadline), but keeps each example's outcome, which paired tests (McNemar, paired
bootstrap; PROTOCOLO.md §7) need and eval_results.json does not store.

  --realign-with REF  Pair each prediction with the gold of REF's row instead of the
                      file's own metas. REF is an aligned predictions file for the same
                      dev set and filters (e.g. one produced after the 2026-09-26 fix).
                      Every row is checked: the prediction's input question must match
                      REF's question, or the script aborts. Used to re-score runs from
                      before the fix, whose golds were shifted after the first filtered
                      example.

Output: <out>.jsonl (one line per example: idx, db_id, question, gold, pred, em, ex;
em/ex None = not scorable) and <out>.summary.json (aggregates, as eval_results.json).
Run inside the training image from the repo root (needs /app/third_party):
  docker run --rm --cpus 1 -v "$PWD":/app -w /app eyuansu62/graphix-text-to-sql:v2 \
      python scripts/score_predictions.py PRED.json OUT_PREFIX [--realign-with REF.json]
"""
import argparse
import json
import re
import sys

sys.path.insert(0, ".")
from seq2seq.metrics.spider import spider_test_suite  # noqa: F401  (installs the exec deadline)
from seq2seq.metrics.spider.spider_test_suite import EXEC_SKIP_DB_IDS
from third_party.spider import evaluation as spider_evaluation
from third_party.test_suite import evaluation as test_suite_evaluation


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fk_maps(refs, module):
    maps = {}
    for r in refs:
        if r["db_id"] not in maps:
            maps[r["db_id"]] = module.build_foreign_key_map({
                "table_names_original": r["db_table_names"],
                "column_names_original": list(zip(r["db_column_names"]["table_id"],
                                                  r["db_column_names"]["column_name"])),
                "foreign_keys": list(zip(r["db_foreign_keys"]["column_id"],
                                         r["db_foreign_keys"]["other_column_id"])),
            })
    return maps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions")
    p.add_argument("out_prefix")
    p.add_argument("--realign-with")
    a = p.parse_args()

    rows = json.load(open(a.predictions))
    if a.realign_with:
        ref = json.load(open(a.realign_with))
        if len(ref) != len(rows):
            sys.exit(f"cannot realign: {len(rows)} predictions vs {len(ref)} reference rows")
        bad = [i for i, (r, g) in enumerate(zip(rows, ref))
               if not norm(r["context"]).startswith(norm(g["question"])[:25])]
        if bad:
            sys.exit(f"cannot realign: input question differs from reference at rows {bad[:10]}")
        refs = [{k: v for k, v in g.items() if k not in ("prediction", "context", "label")} for g in ref]
    else:
        refs = rows
    # Same post-processing as SpiderTrainer._compute_metrics (target_with_db_id).
    preds = [r["prediction"].split("|", 1)[-1].strip() for r in rows]

    em_eval = spider_evaluation.Evaluator(refs[0]["db_path"], fk_maps(refs, spider_evaluation), "match")
    ex_eval = test_suite_evaluation.Evaluator(
        db_dir=refs[0]["db_path"], kmaps=fk_maps(refs, test_suite_evaluation), etype="exec",
        plug_value=False, keep_distinct=False, progress_bar_for_each_datapoint=False)
    turn_scores = {"exec": [], "exact": []}
    out = []
    for i, (pred, g) in enumerate(zip(preds, refs)):
        try:
            em = int(em_eval.evaluate_one(g["db_id"], g["query"], pred)["exact"])
        except Exception:
            em = None  # gold unparseable by process_sql, as in spider_exact_match.py
        ex = None
        if g["db_id"] not in EXEC_SKIP_DB_IDS:
            try:
                n = len(turn_scores["exec"])
                ex_eval.evaluate_one(g["db_id"], g["query"], pred, turn_scores, idx=0)
                ex = turn_scores["exec"][-1] if len(turn_scores["exec"]) > n else None
            except Exception:
                ex = None
        out.append({"idx": i, "db_id": g["db_id"], "question": g["question"],
                    "gold": g["query"], "pred": pred, "em": em, "ex": ex})
    em_eval.finalize()
    ex_eval.finalize()

    with open(a.out_prefix + ".jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    em_s = [r["em"] for r in out if r["em"] is not None]
    ex_s = [r["ex"] for r in out if r["ex"] is not None]
    summary = {
        "source": a.predictions, "realigned_with": a.realign_with, "n": len(out),
        "exact_match": em_eval.scores["all"]["exact"], "exact_match_scored_examples": len(em_s),
        "exec": ex_eval.scores["all"]["exec"], "exec_scored_examples": len(ex_s),
        "exact_match_check": sum(em_s) / len(em_s) if em_s else None,
        "exec_check": sum(ex_s) / len(ex_s) if ex_s else None,
    }
    json.dump(summary, open(a.out_prefix + ".summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
