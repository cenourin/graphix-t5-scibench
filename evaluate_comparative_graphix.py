"""Re-score saved Graphix SQL with the SchemaGraphSQL execution protocol.

The input predictions remain immutable. Output JSONL is resumable; the summary
uses paired questions only and reports an exploratory paired comparison.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import random
import time

import evaluate_schemagraphsql_metrics as metrics
import schemagraphsql_sqlite as sg


GRAPHIX = sg.HERE / '../graphix-t5-scibench-3b/eval/sciencebenchmark/predictions_eval_None.json'
SCHEMA_PREDICTIONS = sg.HERE / 'results/schemagraphsql_sciencebenchmark/predictions.jsonl'
SCHEMA_METRICS = sg.HERE / 'results/schemagraphsql_sciencebenchmark/paper_metrics/per_question.jsonl'
DEFAULT_OUTPUT = sg.HERE / 'results/comparative_graphix_common'
DB_NAMES = {'cordis_temporary': 'cordis', 'oncomx_v1_0_25_small': 'oncomx',
            'skyserver_dr16_2020_11_30': 'sdss'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired_summary(rows, seed=2026, repetitions=10000):
    rows = sorted(rows, key=lambda row: (row['dataset'], row['index']))
    n = len(rows)
    both = sum(row['schema_ex'] and row['graphix_execution']['ex'] for row in rows)
    schema_only = sum(row['schema_ex'] and not row['graphix_execution']['ex'] for row in rows)
    graphix_only = sum(not row['schema_ex'] and row['graphix_execution']['ex'] for row in rows)
    neither = n - both - schema_only - graphix_only
    discordant = schema_only + graphix_only
    p_value = (min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(schema_only, graphix_only) + 1))
                   / 2 ** discordant) if discordant else 1.0)
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset']].append(row['schema_ex'] - row['graphix_execution']['ex'])
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        total = sum(rng.choice(group) for group in groups.values() for _ in range(len(group)))
        samples.append(total / n)
    samples.sort()
    by_db = {}
    for name in sg.DATABASES:
        subset = [row for row in rows if row['dataset'] == name]
        by_db[name] = {'n': len(subset), 'schema_matches': sum(row['schema_ex'] for row in subset),
                       'graphix_matches': sum(row['graphix_execution']['ex'] for row in subset),
                       'graphix_status': dict(Counter(row['graphix_execution']['status'] for row in subset))}
    return {'n': n, 'schema_matches': both + schema_only, 'graphix_matches': both + graphix_only,
            'schema_ex_percent': 100 * (both + schema_only) / n,
            'graphix_ex_percent': 100 * (both + graphix_only) / n,
            'difference_percentage_points': 100 * (schema_only - graphix_only) / n,
            'difference_ci95_percentage_points': [100 * samples[int(.025 * repetitions)],
                                                   100 * samples[min(repetitions - 1, int(.975 * repetitions))]],
            'paired_counts': {'both_correct': both, 'schema_only': schema_only,
                              'graphix_only': graphix_only, 'both_incorrect': neither},
            'mcnemar_exact_p_exploratory': p_value, 'bootstrap_seed': seed,
            'bootstrap_repetitions': repetitions, 'by_db': by_db}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--pair-timeout', type=float, default=120)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.pair_timeout <= 0 or args.workers <= 0:
        parser.error('timeout and workers must be positive')

    graphix = json.loads(GRAPHIX.read_text())
    schema_predictions = read_jsonl(SCHEMA_PREDICTIONS)
    schema_metrics = read_jsonl(SCHEMA_METRICS)
    lookup = {(p['dataset'], p['question'], p['reference_sql']): p['index'] for p in schema_predictions}
    if len(lookup) != len(schema_predictions):
        raise ValueError('SchemaGraphSQL question keys are not unique')
    schema_scores = {(r['dataset'], r['index']): r['execution'] for r in schema_metrics}
    if len(schema_scores) != len(schema_metrics):
        raise ValueError('SchemaGraphSQL metric keys are not unique')

    matched = []
    for item in graphix:
        db_id = item['db_id']
        name = DB_NAMES[db_id]
        index = lookup[(name, item['question'], item['query'])]
        prefix = db_id + ' | '
        if not item['prediction'].startswith(prefix):
            raise ValueError(f'Missing Graphix prefix for {name}:{index}')
        sql = item['prediction'][len(prefix):]
        matched.append((name, index, sql, item['query']))
    keys = [(name, index) for name, index, _, _ in matched]
    if len(set(keys)) != len(keys):
        raise ValueError('Graphix question keys are not unique')
    if set(keys) - set(schema_scores):
        raise ValueError('Missing SchemaGraphSQL scores for paired questions')

    databases = {}
    fingerprints = {}
    for name in sg.DATABASES:
        db, meta_path, db_id = sg.sb_paths(name)
        schema = sg.add_tables_json(sg.introspect(db, db_id), meta_path, db_id)
        databases[name] = db, schema.tables
        fingerprints[name] = {'size': db.stat().st_size, 'mtime_ns': db.stat().st_mtime_ns}
    config = {'graphix_sha256': sha256(GRAPHIX), 'schema_predictions_sha256': sha256(SCHEMA_PREDICTIONS),
              'schema_metrics_sha256': sha256(SCHEMA_METRICS),
              'evaluator_sha256': sha256(Path(metrics.__file__)),
              'script_sha256': sha256(Path(__file__)), 'databases': fingerprints,
              'pair_timeout_seconds': args.pair_timeout, 'workers': args.workers,
              'protocol': 'full SQLite row-set equality; errors and timeouts score zero',
              'graphix_transform': 'remove exact db_id | prefix only', 'paired_n': len(matched)}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit('Changed configuration: use a new output directory')
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    output_path = args.output / 'per_question.jsonl'
    results = read_jsonl(output_path) if output_path.exists() else []
    done = {(row['dataset'], row['index']) for row in results}
    if len(done) != len(results) or not done <= set(keys):
        raise ValueError('Invalid existing output records')

    def evaluate(item):
        name, index, sql, reference = item
        db, tables = databases[name]
        start = time.monotonic()
        execution = metrics.execute_pair(db, sql, reference, tables, args.pair_timeout)
        return {'dataset': name, 'index': index, 'schema_ex': schema_scores[(name, index)]['ex'],
                'schema_status': schema_scores[(name, index)]['status'],
                'graphix_execution': execution, 'seconds': round(time.monotonic() - start, 3)}

    with output_path.open('a', buffering=1) as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, item) for item in matched if item[:2] not in done]
        for future in as_completed(futures):
            row = future.result()
            stream.write(json.dumps(row) + '\n')
            results.append(row)
            print(f"{len(results)}/{len(matched)} {row['dataset']}:{row['index']} "
                  f"{row['graphix_execution']['status']}", flush=True)
    if len(results) == len(matched):
        summary = paired_summary(results)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
