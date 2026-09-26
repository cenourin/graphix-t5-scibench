import sqlite3
import tempfile
import unittest
from pathlib import Path

from evaluate_schemagraphsql_metrics import execute_pair, gold_tables, row_key, scores


class MetricTests(unittest.TestCase):
    def test_nested_gold_tables(self):
        tree = {'from': {'table_units': [['table_unit', 0]]},
                'where': [{'from': {'table_units': [['table_unit', 1]]}}],
                'union': {'from': {'table_units': [['table_unit', 0], ['table_unit', 2]]}}}
        self.assertEqual(gold_tables(tree, ['A', 'B', 'C']), {'a', 'b', 'c'})

    def test_precision_recall_f6_and_exact_match(self):
        s = scores({'a', 'b'}, {'a'})
        self.assertEqual(s['precision'], .5)
        self.assertEqual(s['recall'], 1)
        self.assertAlmostEqual(s['f6'], 18.5/19)
        self.assertEqual(s['emr'], 0)
        self.assertEqual(scores({'a'}, {'a'})['emr'], 1)
        self.assertEqual(scores(set(), {'a'})['f6'], 0)

    def test_numeric_equality_without_text_or_blob_collisions(self):
        self.assertEqual(row_key((1, -0.0)), row_key((1.0, 0)))
        self.assertNotEqual(row_key((b'ab',)), row_key(('6162',)))
        self.assertNotEqual(row_key((None,)), row_key(('null',)))
        self.assertNotEqual(row_key((9007199254740993,)), row_key((float(9007199254740993),)))


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / 'db.sqlite'
        with sqlite3.connect(self.db) as conn:
            conn.execute('CREATE TABLE t(x INTEGER)')
            conn.executemany('INSERT INTO t VALUES (?)', [(i,) for i in range(10002)] + [(1,)])
        self.initial = self.db.read_bytes()

    def tearDown(self):
        self.assertEqual(self.db.read_bytes(), self.initial)
        self.tmp.cleanup()

    def pair(self, pred, gold, timeout=10):
        return execute_pair(self.db, pred, gold, {'t'}, timeout)

    def test_full_results_ignore_duplicates_and_order(self):
        r = self.pair('SELECT x FROM t ORDER BY x DESC', 'SELECT DISTINCT x FROM t ORDER BY x')
        self.assertEqual(r['status'], 'match')
        self.assertEqual(r['prediction_rows'], 10003)
        self.assertEqual(r['prediction_distinct_rows'], 10002)

    def test_difference_beyond_old_row_limit(self):
        self.assertEqual(self.pair('SELECT x FROM t WHERE x < 10001', 'SELECT x FROM t')['ex'], 0)

    def test_invalid_reference_and_writes_are_separate_failures(self):
        self.assertEqual(self.pair('SELECT x FROM t', 'SELECT missing FROM t')['status'], 'reference_error')
        self.assertEqual(self.pair('DELETE FROM t', 'SELECT x FROM t')['status'], 'prediction_error')

    def test_timeout_is_not_a_mismatch(self):
        self.assertEqual(self.pair('SELECT x FROM t', 'SELECT x FROM t', -1)['status'], 'prediction_timeout')


if __name__ == '__main__':
    unittest.main()
