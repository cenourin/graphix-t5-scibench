import sqlite3
import tempfile
import unittest
from pathlib import Path

import schemagraphsql_sqlite as sg


class GraphTests(unittest.TestCase):
    def test_force_union_keeps_all_tied_shortest_paths(self):
        schema = sg.Schema(
            {
                "a": ["id"],
                "b": ["id"],
                "c": ["id"],
                "d": ["id"],
                "long": ["id"],
            },
            edges=[
                sg.Edge("a", "b", (("id", "id"),)),
                sg.Edge("b", "d", (("id", "id"),)),
                sg.Edge("a", "c", (("id", "id"),)),
                sg.Edge("c", "d", (("id", "id"),)),
                sg.Edge("a", "long", (("id", "id"),)),
                sg.Edge("long", "c", (("id", "id"),)),
            ],
        )
        backbone = sg.force_union(sg.Graph.build(schema, schema.edges), ["a"], ["d"])
        self.assertEqual(backbone.tables, frozenset({"a", "b", "c", "d"}))
        self.assertEqual(len(backbone.edges), 4)

    def test_disconnected_pair_is_an_error(self):
        schema = sg.Schema({"a": ["id"], "b": ["id"]})
        with self.assertRaises(sg.UnreachableError):
            sg.force_union(sg.Graph.build(schema, []), ["a"], ["b"])

    def test_zero_hop_without_source(self):
        schema = sg.Schema({"a": ["id"]})
        backbone = sg.force_union(sg.Graph.build(schema, []), [], ["a"])
        self.assertEqual(backbone.tables, frozenset({"a"}))
        self.assertFalse(backbone.edges)


class SQLiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "sample.sqlite"
        connection = sqlite3.connect(self.path)
        connection.executescript(
            """
            CREATE TABLE parent (
                a INTEGER,
                b INTEGER,
                label TEXT UNIQUE,
                PRIMARY KEY (a, b)
            );
            CREATE TABLE child (
                id INTEGER PRIMARY KEY,
                x INTEGER,
                y INTEGER,
                FOREIGN KEY (x, y) REFERENCES parent (a, b)
            );
            CREATE TABLE isolated (id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO parent VALUES (1, 2, 'ok');
            INSERT INTO child VALUES (10, 1, 2);
            """
        )
        connection.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_introspection_preserves_composite_fk(self):
        schema = sg.introspect(self.path, "sample")
        self.assertEqual(set(schema.tables), {"parent", "child", "isolated"})
        self.assertEqual(schema.primary_keys["parent"], ("a", "b"))
        self.assertEqual(len(schema.edges), 1)
        self.assertEqual(schema.edges[0].pairs, (("x", "a"), ("y", "b")))

    def test_validation_and_execution_are_read_only(self):
        before = self.path.stat()
        result = sg.validate_sql(
            self.path,
            "SELECT child.id FROM child JOIN parent "
            "ON child.x = parent.a AND child.y = parent.b;",
            {"child", "parent"},
            execute=True,
            max_rows=5,
        )
        after = self.path.stat()
        self.assertTrue(result["validated"])
        self.assertEqual(result["rows"], [[10]])
        self.assertEqual((before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns))

    def test_table_outside_backbone_is_denied(self):
        with self.assertRaises(sg.PipelineError):
            sg.validate_sql(self.path, "SELECT * FROM isolated", {"parent"})

    def test_non_select_is_rejected(self):
        with self.assertRaises(sg.PipelineError):
            sg.validate_sql(self.path, "DELETE FROM child", {"child"})


class ScienceBenchmarkSmokeTests(unittest.TestCase):
    def test_sdss_official_metadata_supplies_six_edges(self):
        database, metadata, db_id = sg.sb_paths("sdss")
        self.assertIsNotNone(metadata)
        schema = sg.add_tables_json(sg.introspect(database, db_id), metadata, db_id)
        self.assertEqual(len(schema.tables), 6)
        self.assertEqual(len(schema.edges), 6)
        for edge in schema.edges:
            schema.validate_edge(edge)


if __name__ == "__main__":
    unittest.main()
