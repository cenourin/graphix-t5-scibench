"""Offline table-linking metrics and BIRD-style full-result execution accuracy.

No model calls. Input predictions are immutable. Exact disk-backed row sets avoid
truncation and unbounded RAM. SQLite references are executed without rewriting.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import time

import schemagraphsql_sqlite as sg


def gold_tables(tree, names):
    result = set()
    def visit(node):
        if isinstance(node, dict):
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            if len(node) == 2 and node[0] == 'table_unit' and isinstance(node[1], int):
                if not 0 <= node[1] < len(names):
                    raise ValueError('Invalid gold table index')
                result.add(names[node[1]].casefold())
            else:
                for value in node:
                    visit(value)
    visit(tree)
    if not result:
        raise ValueError('Gold query has no table annotation')
    return result


def scores(predicted, gold):
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold)
    return dict(precision=p, recall=r,
                f1=2*p*r/(p+r) if p+r else 0.0,
                f6=37*p*r/(36*p+r) if 36*p+r else 0.0,
                emr=float(predicted == gold), tp=tp,
                predicted_count=len(predicted), gold_count=len(gold))


def row_key(row):
    # Match Python tuple equality used by BIRD, including 1 == 1.0,
    # while distinguishing strings, blobs and NULL. No lossy hashing.
    def cell(v):
        if v is None:
            return ['null']
        if isinstance(v, (int, float)):
            if isinstance(v, float) and not math.isfinite(v):
                return ['number', str(v)]
            n, d = (v, 1) if isinstance(v, int) else v.as_integer_ratio()
            return ['number', str(n), str(d)]
        if isinstance(v, bytes):
            return ['blob', v.hex()]
        return ['text', v]
    return json.dumps([cell(v) for v in row], ensure_ascii=True, separators=(',', ':')).encode()


class EvaluationTimeout(Exception):
    pass


def execute_pair(database, predicted_sql, reference_sql, allowed, seconds):
    deadline = time.monotonic() + seconds
    stats = {}
    def check():
        if time.monotonic() >= deadline:
            raise EvaluationTimeout(f'Pair exceeded {seconds}s')
    # Ephemeral files contain result keys; original databases remain read-only.
    with tempfile.TemporaryDirectory(prefix='schemagraph-ex-') as tmp:
        store = sqlite3.connect(str(Path(tmp) / 'sets.sqlite'))
        source = sg.ro_connect(database)
        try:
            store.execute('PRAGMA journal_mode=OFF')
            store.execute('PRAGMA synchronous=OFF')
            store.execute('PRAGMA cache_size=-32768')
            store.execute('CREATE TABLE p (row BLOB PRIMARY KEY) WITHOUT ROWID')
            store.execute('CREATE TABLE g (row BLOB PRIMARY KEY) WITHOUT ROWID')
            source.set_authorizer(sg.authorizer(allowed))
            source.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10000)
            for stage, table, sql in [('prediction', 'p', predicted_sql), ('reference', 'g', reference_sql)]:
                try:
                    check()
                    cursor = source.execute(sg.plain_sql(sql))
                    count = 0
                    while True:
                        check()
                        batch = cursor.fetchmany(1000)
                        if not batch:
                            break
                        store.executemany(f'INSERT OR IGNORE INTO {table} VALUES (?)',
                                          ((row_key(row),) for row in batch))
                        count += len(batch)
                    stats[stage + '_rows'] = count
                    stats[stage + '_distinct_rows'] = store.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                except (sqlite3.Error, sg.PipelineError, EvaluationTimeout) as exc:
                    timed_out = isinstance(exc, EvaluationTimeout) or time.monotonic() >= deadline
                    return dict(stats, status=stage + ('_timeout' if timed_out else '_error'),
                                error=str(exc), ex=0)
            store.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10000)
            try:
                check()
                different = store.execute('SELECT 1 FROM (SELECT row FROM p EXCEPT SELECT row FROM g) LIMIT 1').fetchone()
                if not different:
                    different = store.execute('SELECT 1 FROM (SELECT row FROM g EXCEPT SELECT row FROM p) LIMIT 1').fetchone()
                return dict(stats, status='mismatch' if different else 'match', ex=int(not different))
            except (sqlite3.Error, EvaluationTimeout) as exc:
                return dict(stats, status='comparison_timeout', error=str(exc), ex=0)
        finally:
            source.close()
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=sg.HERE / 'results/schemagraphsql_sciencebenchmark')
    parser.add_argument('--output', type=Path, default=sg.HERE / 'results/schemagraphsql_sciencebenchmark/paper_metrics')
    parser.add_argument('--pair-timeout', type=float, default=120)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.input / 'predictions.jsonl'
    predictions = [json.loads(line) for line in records_path.read_text().splitlines()]
    cache, expected = {}, set()
    fingerprints = {}
    for name in sg.DATABASES:
        db, meta_path, dbid = sg.sb_paths(name)
        meta = json.loads(meta_path.read_text())
        if isinstance(meta, list):
            meta = next(m for m in meta if m['db_id'] == dbid)
        dev_path = sg.DATA_ROOT / name / 'dev.json'
        items = json.loads(dev_path.read_text())
        schema = sg.add_tables_json(sg.introspect(db, dbid), meta_path, dbid)
        cache[name] = db, meta, items, schema
        expected.update((name, i) for i in range(len(items)))
        fingerprints[name] = {'dev': hashlib.sha256(dev_path.read_bytes()).hexdigest(),
                              'metadata': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                              'db_size': db.stat().st_size, 'db_mtime_ns': db.stat().st_mtime_ns}
    assert len(predictions) == len(expected)
    assert {(p['dataset'], p['index']) for p in predictions} == expected
    config = {'predictions_sha256': hashlib.sha256(records_path.read_bytes()).hexdigest(),
              'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inputs': fingerprints, 'pair_timeout_seconds': args.pair_timeout,
              'workers': args.workers, 'row_limit': None,
              'protocol': 'BIRD-style set equality of complete rows; errors score zero; SQLite adaptation',
              'source': 'https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py'}
    config_path = args.output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit('Changed configuration: use a new output directory')
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    output_path = args.output / 'per_question.jsonl'
    results = [json.loads(line) for line in output_path.read_text().splitlines()] if output_path.exists() else []
    done = {(r['dataset'], r['index']) for r in results}

    def evaluate(p):
        start = time.monotonic()
        name, index = p['dataset'], p['index']
        db, meta, items, schema = cache[name]
        item = items[index]
        assert item['question'] == p['question'] and item['query'] == p['reference_sql']
        gold = gold_tables(item['sql'], meta['table_names_original'])
        saved = p.get('prediction')
        selected = {t.casefold() for t in saved['filtered_tables']} if saved is not None else set()
        assert selected <= {t.casefold() for t in schema.tables}
        result = {'dataset': name, 'index': index, 'gold_tables': sorted(gold),
                  'predicted_tables': sorted(selected) if saved is not None else None,
                  'schema_observed': saved is not None, 'schema_scores': scores(selected, gold)}
        if saved is None:
            result['execution'] = {'status': 'missing_prediction', 'ex': 0}
        else:
            result['execution'] = execute_pair(db, saved['sql'], item['query'], schema.tables, args.pair_timeout)
        result['seconds'] = round(time.monotonic() - start, 3)
        return result

    def summarize():
        summary = {'expected': len(expected), 'completed': len(results), 'datasets': {}}
        from collections import Counter
        for name in [*sg.DATABASES, 'all']:
            subset = [r for r in results if name == 'all' or r['dataset'] == name]
            observed = [r for r in subset if r['schema_observed']]
            def aggregate(rows):
                if not rows:
                    return None
                fields = ['precision', 'recall', 'f1', 'f6', 'emr']
                macro = {f: 100*sum(r['schema_scores'][f] for r in rows)/len(rows) for f in fields}
                tp = sum(r['schema_scores']['tp'] for r in rows)
                np = sum(r['schema_scores']['predicted_count'] for r in rows)
                ng = sum(r['schema_scores']['gold_count'] for r in rows)
                precision, recall = tp/np if np else 0, tp/ng if ng else 0
                return {'n': len(rows), 'macro_percent': macro,
                        'micro_percent': {'precision': 100*precision, 'recall': 100*recall,
                        'f1': 100*2*precision*recall/(precision+recall) if precision+recall else 0,
                        'f6': 100*37*precision*recall/(36*precision+recall) if precision+recall else 0}}
            summary['datasets'][name] = {
                'n': len(subset), 'schema_observed': len(observed),
                'schema_missing': len(subset)-len(observed),
                'schema_zero_for_missing': aggregate(subset), 'schema_observed_only': aggregate(observed),
                'execution_matches': sum(r['execution']['ex'] for r in subset),
                'execution_accuracy_percent': 100*sum(r['execution']['ex'] for r in subset)/len(subset) if subset else None,
                'execution_status': dict(Counter(r['execution']['status'] for r in subset))}
        tmp = args.output / 'summary.json.tmp'
        tmp.write_text(json.dumps(summary, indent=2) + '\n')
        tmp.replace(args.output / 'summary.json')
        return summary

    with output_path.open('a', buffering=1) as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, p) for p in predictions if (p['dataset'], p['index']) not in done]
        for future in as_completed(futures):
            r = future.result()
            stream.write(json.dumps(r) + '\n')
            results.append(r)
            summarize()
            print(f"{len(results)}/{len(expected)} {r['dataset']}:{r['index']} {r['execution']['status']}", flush=True)
    summarize()


if __name__ == '__main__':
    main()
