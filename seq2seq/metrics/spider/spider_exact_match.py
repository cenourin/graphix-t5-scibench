"""Spider exact match metric."""

import logging
from typing import Dict, Any
from third_party.spider import evaluation as spider_evaluation

logger = logging.getLogger(__name__)


def compute_exact_match_metric(predictions, references) -> Dict[str, Any]:
    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = spider_evaluation.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )
    evaluator = spider_evaluation.Evaluator(references[0]["db_path"], foreign_key_maps, "match")
    # third_party/spider/process_sql.py implements a restricted SQL grammar calibrated on
    # Spider's own gold queries -- it has no coverage for, e.g., parenthesized boolean
    # groups in a WHERE clause (raises ParenthesesInConditionError), which Spider's gold
    # SQL never uses but ScienceBenchmark's does. A single unparseable gold query used to
    # crash this whole metric, discarding exact_match for every other example too. Skip
    # just that pair instead, logged by db_id for the same transparency reasons as the
    # token/graph-node filter in run_seq2seq_eval.py.
    skipped_by_db = {}
    n_scored = 0
    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        try:
            _ = evaluator.evaluate_one(reference["db_id"], reference["query"], prediction)
            n_scored += 1
        except Exception as e:
            db_id = reference.get("db_id")
            skipped_by_db[db_id] = skipped_by_db.get(db_id, 0) + 1
            logger.warning(
                "Skipped exact_match scoring for db_id=%s (gold query unparseable by "
                "third_party/spider/process_sql.py: %s: %s)", db_id, type(e).__name__, e,
            )
    if skipped_by_db:
        logger.warning(
            "exact_match: skipped %d/%d examples with unparseable gold SQL -- by db_id: %s",
            sum(skipped_by_db.values()), len(references), skipped_by_db,
        )
    evaluator.finalize()
    # NOTE: this dict flows into transformers.Trainer.log_metrics(), which formats
    # every value as a number -- confirmed via a real crash ("TypeError: unsupported
    # format string passed to dict.__format__") when a dict-valued field
    # (exact_match_skipped_by_db) was included here. Keep only numeric fields in the
    # returned dict; the per-db breakdown is still fully available via the
    # logger.warning() call above (and in any standalone script that calls this
    # function directly and doesn't go through Trainer.log_metrics).
    return {
        "exact_match": evaluator.scores["all"]["exact"],
        "exact_match_scored_examples": n_scored,
    }
