"""Run the local ScienceBenchmark dev split; save resumable per-question results.

Execution agreement is a local SQLite check, not the official benchmark metric.
Gold SQL is executed unchanged and is never passed to the model. Results above
the row limit are excluded from comparison. Column order and duplicates matter;
row order matters only when the reference parse tree has a top-level ORDER BY.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import schemagraphsql_sqlite as sg


def compare_rows(predicted, reference, ordered):
    if predicted['truncated'] or reference['truncated']:
        return None
    if len(predicted['columns']) != len(reference['columns']):
        return False
    a = [tuple(row) for row in predicted['rows']]
    b = [tuple(row) for row in reference['rows']]
    return a == b if ordered else Counter(a) == Counter(b)


def compact(result):
    return {**result, 'returned_rows': len(result['rows']), 'rows': result['rows'][:5]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', default=str(sg.HERE / '.openwebui-key'))
    parser.add_argument('--url', default='http://home2.scanuto.com:8081')
    parser.add_argument('--model', default='qwen3-coder-next:latest')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', type=Path, default=sg.HERE / 'results/schemagraphsql_sciencebenchmark')
    args = parser.parse_args()
    key = Path(args.key_file).read_text().strip()
    if not key:
        raise SystemExit('Empty API key file')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'model': args.model, 'url': args.url, 'mode': 'paper',
              'api_timeout': 120, 'query_timeout': 30, 'max_rows': 10000,
              'pipeline_sha256': hashlib.sha256(Path(sg.__file__).read_bytes()).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'datasets': {}}
    jobs = []
    for name in sg.DATABASES:
        database, metadata, db_id = sg.sb_paths(name)
        schema = sg.add_tables_json(sg.introspect(database, db_id), metadata, db_id)
        source = sg.DATA_ROOT / name / 'dev.json'
        config['datasets'][name] = hashlib.sha256(source.read_bytes()).hexdigest()
        for index, item in enumerate(json.loads(source.read_text())):
            jobs.append((name, index, item, database, schema))
    config_path = args.output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit('Configuration changed; select a new --output directory')
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    records_path = args.output / 'predictions.jsonl'
    records = []
    if records_path.exists():
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    done = {(r['dataset'], r['index']) for r in records}

    def evaluate(job):
        name, index, item, database, schema = job
        start = time.monotonic()
        record = {'dataset': name, 'index': index, 'question': item['question'],
                  'reference_sql': item['query'], 'execution_agreement': None}
        prediction = reference = None
        try:
            client = sg.OpenWebUI(args.url, key, args.model, timeout=120)
            output = sg.run_paper(schema, item['question'], '', client)
            record['prediction'] = output
            prediction = sg.validate_sql(database, output['sql'], output['filtered_tables'],
                                         execute=True, max_rows=10000, timeout=30)
            record['prediction_execution'] = compact(prediction)
        except Exception as exc:
            record['prediction_error'] = str(exc).replace(key, '[REDACTED]')
        try:
            reference = sg.validate_sql(database, item['query'], schema.tables,
                                        execute=True, max_rows=10000, timeout=30)
            record['reference_execution'] = compact(reference)
        except Exception as exc:
            record['reference_error'] = str(exc).replace(key, '[REDACTED]')
        if prediction is not None and reference is not None:
            record['execution_agreement'] = compare_rows(
                prediction, reference, bool(item.get('sql', {}).get('orderBy')))
        record['seconds'] = round(time.monotonic() - start, 3)
        return record

    def summarize():
        summary = {'expected': len(jobs), 'completed': len(records), 'datasets': {},
                   'note': 'Local SQLite result agreement; not official ScienceBenchmark accuracy. '
                           'Only successful, untruncated pairs are comparable; references are unmodified.'}
        for name in [*sg.DATABASES, 'all']:
            subset = [r for r in records if name == 'all' or r['dataset'] == name]
            comparable = [r for r in subset if r['execution_agreement'] is not None]
            matches = sum(r['execution_agreement'] is True for r in comparable)
            summary['datasets'][name] = {
                'completed': len(subset),
                'prediction_executed': sum('prediction_execution' in r for r in subset),
                'prediction_errors': sum('prediction_error' in r for r in subset),
                'reference_errors': sum('reference_error' in r for r in subset),
                'comparable': len(comparable), 'matching': matches,
                'agreement_percent_on_comparable': round(100 * matches / len(comparable), 2) if comparable else None}
        temp = args.output / 'summary.json.tmp'
        temp.write_text(json.dumps(summary, indent=2) + '\n')
        temp.replace(args.output / 'summary.json')
        return summary

    with records_path.open('a', buffering=1) as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(evaluate, job) for job in jobs if (job[0], job[1]) not in done]
        for future in as_completed(pending):
            record = future.result()
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            records.append(record)
            summarize()
            print(f"{len(records)}/{len(jobs)} {record['dataset']}:{record['index']} "
                  f"executed={'prediction_execution' in record} "
                  f"agreement={record['execution_agreement']} seconds={record['seconds']}", flush=True)
    print(json.dumps(summarize(), indent=2), flush=True)


if __name__ == '__main__':
    main()
