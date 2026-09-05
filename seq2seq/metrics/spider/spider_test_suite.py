"""Spider Test Suite Execution Accuracy metric."""
import logging
import os
import time
from typing import Optional, Dict, Any
from third_party.test_suite import evaluation as test_suite_evaluation
from third_party.test_suite import exec_eval as _exec_eval_module

logger = logging.getLogger(__name__)

# --- Real fix for the asyncio-timeout-doesn't-work hang (see EXEC_SKIP_DB_IDS below
# for the still-kept belt-and-suspenders skip layer) ---
#
# third_party/test_suite/exec_eval.py wraps each SQL execution in
# asyncio.wait_for(exec_on_db_(...), timeout=60), but exec_on_db_ does a *blocking*
# sqlite3 cursor.execute() with no internal await point. asyncio.wait_for can only
# cancel at an await; a long-running C-level query never yields one, so the timer
# never actually fires -- confirmed empirically: a query against sdss's 76.8M-row
# neighbors table hung an eval for 10+ hours. signal.alarm() has the same underlying
# problem for the same reason (CPython only processes pending signals between
# bytecodes; a blocking C call that never returns control never lets the handler run
# either) -- and additionally only works on the main thread, which would silently
# break under threaded execution.
#
# sqlite3.Connection.set_progress_handler(callback, n) is the mechanism actually
# designed for this: SQLite calls it periodically *during* query execution, from
# inside the same blocking call -- returning non-zero aborts the query in place with
# an OperationalError, which the existing try/except already handles like any other
# execution error. No watchdog thread, no signal handling, no race conditions.
#
# exec_eval.py lives in the Docker image, not on the host, so it can't be edited
# directly (an edit there would be invisible and lost on the next image change).
# Monkeypatching the module's exec_on_db_ from here achieves the same effect at the
# call boundary and survives image rebuilds since this file is what's on the host.
_ORIGINAL_exec_on_db_ = _exec_eval_module.exec_on_db_


async def _exec_on_db_with_progress_deadline(sqlite_path: str, query: str):
    query = _exec_eval_module.replace_cur_year(query)
    cursor = _exec_eval_module.get_cursor_from_path(sqlite_path)
    deadline = time.monotonic() + _exec_eval_module.TIMEOUT
    cursor.connection.set_progress_handler(lambda: time.monotonic() > deadline, 1000)
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        cursor.connection.close()
        return "result", result
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        return "exception", e


_exec_eval_module.exec_on_db_ = _exec_on_db_with_progress_deadline

# Empty by default now: the progress-handler fix above was validated against all 94
# real sdss dev-set gold queries (not just a synthetic worst case) -- 0 uncaught
# errors, every query either completed normally or was cleanly interrupted at the
# ~60s deadline. Kept as an env-var escape hatch (GRAPHIX_EXEC_SKIP_DB_IDS) in case a
# future db_id/query pattern needs the same treatment without another code change.
EXEC_SKIP_DB_IDS = set(
    db_id.strip() for db_id in os.environ.get("GRAPHIX_EXEC_SKIP_DB_IDS", "").split(",")
    if db_id.strip()
)


def compute_test_suite_metric(predictions, references, db_dir: Optional[str] = None) -> Dict[str, Any]:
    if db_dir is None:
        references[0]["db_path"]

    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = test_suite_evaluation.build_foreign_key_map(
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

    evaluator = test_suite_evaluation.Evaluator(
        db_dir=db_dir if db_dir is not None else references[0]["db_path"],
        kmaps=foreign_key_maps,
        etype="exec",
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
    )
    # Only used for Sparc/CoSQL
    turn_scores = {"exec": [], "exact": []}
    skipped_by_db = {}
    n_scored = 0
    n_skipped_hang_risk = 0
    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        if reference.get("db_id") in EXEC_SKIP_DB_IDS:
            n_skipped_hang_risk += 1
            continue
        try:
            _ = evaluator.evaluate_one(
                reference["db_id"],
                reference["query"],
                prediction,
                turn_scores,
                idx=turn_idx,
            )
            n_scored += 1
        except Exception as e:
            # Was `except AssertionError` only. third_party/test_suite's own
            # eval_exec_match() already turns a failed/timed-out *gold* query into a
            # deliberate AssertionError (caught fine before), but it calls
            # remove_distinct(g_str) on the gold query with no try/except at all --
            # any sqlparse failure there (plausible on ScienceBenchmark's less
            # Spider-like SQL, e.g. the WHERE-parentheses pattern that already broke
            # exact_match) would raise uncaught and crash exec for every example, not
            # just this one. Broadened to catch and skip any single-pair failure,
            # same defensive pattern as compute_exact_match_metric.
            db_id = reference.get("db_id")
            skipped_by_db[db_id] = skipped_by_db.get(db_id, 0) + 1
            logger.warning(f"unexpected evaluation error ({type(e).__name__}) for db_id={db_id}: {e}")
    if skipped_by_db:
        logger.warning(
            "exec: skipped %d/%d examples due to evaluation errors -- by db_id: %s",
            sum(skipped_by_db.values()), len(references), skipped_by_db,
        )
    if n_skipped_hang_risk:
        logger.warning(
            "exec: skipped %d/%d examples entirely (db_id in EXEC_SKIP_DB_IDS=%s -- "
            "known asyncio-timeout-doesn't-work hang risk, not scored at all)",
            n_skipped_hang_risk, len(references), sorted(EXEC_SKIP_DB_IDS),
        )
    evaluator.finalize()
    # NOTE: this dict flows into transformers.Trainer.log_metrics(), which formats
    # every value as a number -- confirmed via a real crash ("TypeError: unsupported
    # format string passed to dict.__format__") when dict/list-valued fields
    # (exec_skipped_by_db, exec_skipped_hang_risk_db_ids) were included here. Keep
    # only numeric fields in the returned dict; the per-db breakdown is still fully
    # available via the logger.warning() calls above (and in any standalone script
    # that calls this function directly and doesn't go through Trainer.log_metrics).
    return {
        "exec": evaluator.scores["all"]["exec"],
        "exec_scored_examples": n_scored,
        "exec_skipped_hang_risk_count": n_skipped_hang_risk,
    }
