"""Run SchemaGraphSQL against the local ScienceBenchmark SQLite databases.

The default paper mode uses two focused LLM calls and deterministic local
graph search. The single-call mode uses a consolidated system prompt and is an
approximation. SQLite files are always opened read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Sequence

import requests


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data_all_in/data/sciencebenchmark"
DATABASES = {
    "cordis": ("cordis_temporary", "cordis"),
    "oncomx": ("oncomx_v1_0_25_small", "oncomx"),
    "sdss": ("skyserver_dr16_2020_11_30", "sdss"),
}
DEFAULT_SINGLE_PROMPT = HERE / "schemagraphsql_single_call_prompt.txt"


class PipelineError(RuntimeError):
    pass


class UnreachableError(PipelineError):
    def __init__(self, pairs):
        self.pairs = tuple(pairs)
        super().__init__("unreachable graph pair(s): " + ", ".join(f"{a}->{b}" for a, b in pairs))


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class Edge:
    source_table: str
    target_table: str
    pairs: tuple[tuple[str, str], ...]
    provenance: str = "sqlite_fk"

    def key(self):
        forward = (self.source_table, self.target_table, self.pairs)
        reverse = (self.target_table, self.source_table, tuple((b, a) for a, b in self.pairs))
        return min(forward, reverse)

    def render(self):
        predicates = " AND ".join(
            f"{q(self.source_table)}.{q(a)} = {q(self.target_table)}.{q(b)}"
            for a, b in self.pairs
        )
        return f"{predicates} [{self.provenance}]"


@dataclass
class Schema:
    tables: dict[str, list[str]]
    types: dict[tuple[str, str], str] = field(default_factory=dict)
    primary_keys: dict[str, tuple[str, ...]] = field(default_factory=dict)
    unique_keys: dict[str, list[tuple[str, ...]]] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    db_id: str | None = None

    def validate_edge(self, edge: Edge):
        if edge.source_table not in self.tables or edge.target_table not in self.tables:
            raise PipelineError(f"edge references unknown table: {edge}")
        if not edge.pairs:
            raise PipelineError(f"edge has no columns: {edge}")
        for a, b in edge.pairs:
            if a not in self.tables[edge.source_table] or b not in self.tables[edge.target_table]:
                raise PipelineError(f"edge references unknown column: {edge}")

    def render(self):
        lines = []
        for table, columns in self.tables.items():
            lines.append(f"TABLE {q(table)}")
            pk = set(self.primary_keys.get(table, ()))
            for column in columns:
                suffix = f" {self.types.get((table, column), '')}".rstrip()
                if column in pk:
                    suffix += " PRIMARY KEY"
                lines.append(f"  COLUMN {q(column)}{suffix}")
        if self.edges:
            lines.append("RELATIONSHIPS")
            lines.extend("  " + edge.render() for edge in dedupe_edges(self.edges))
        else:
            lines.append("RELATIONSHIPS: NONE DECLARED")
        return "\n".join(lines)

    def subset(self, selected: Iterable[str], edges: Iterable[Edge]):
        chosen = set(selected)
        return Schema(
            {t: c for t, c in self.tables.items() if t in chosen},
            {k: v for k, v in self.types.items() if k[0] in chosen},
            {t: v for t, v in self.primary_keys.items() if t in chosen},
            {t: v for t, v in self.unique_keys.items() if t in chosen},
            [e for e in edges if e.source_table in chosen and e.target_table in chosen],
            self.db_id,
        )

    def summary(self):
        return {
            "db_id": self.db_id,
            "tables": len(self.tables),
            "columns": sum(map(len, self.tables.values())),
            "foreign_keys": len(self.edges),
            "table_names": list(self.tables),
        }


def dedupe_edges(edges: Iterable[Edge]) -> list[Edge]:
    result = {}
    for edge in edges:
        result.setdefault(edge.key(), edge)
    return sorted(result.values(), key=lambda edge: edge.key())


def ro_connect(path: str | Path):
    database = Path(path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.enable_load_extension(False)
    return connection


def pragma(connection, name, target):
    allowed = {"table_xinfo", "table_info", "foreign_key_list", "index_list", "index_xinfo"}
    if name not in allowed:
        raise ValueError(name)
    return connection.execute(f"PRAGMA {name}({q(target)})").fetchall()


def introspect(path: str | Path, db_id: str | None = None) -> Schema:
    connection = ro_connect(path)
    try:
        names = [
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        tables, types, primary, unique = {}, {}, {}, {}
        for table in names:
            try:
                rows = pragma(connection, "table_xinfo", table)
            except sqlite3.OperationalError:
                rows = pragma(connection, "table_info", table)
            tables[table] = [row[1] for row in rows]
            for row in rows:
                types[(table, row[1])] = row[2] or ""
            pk = tuple(row[1] for row in sorted(rows, key=lambda x: x[5]) if row[5])
            if pk:
                primary[table] = pk
            keys = []
            for index in pragma(connection, "index_list", table):
                if not index[2] or (len(index) > 4 and index[4]):
                    continue
                cols = [
                    row[2] for row in pragma(connection, "index_xinfo", index[1])
                    if row[2] is not None and (len(row) < 6 or row[5])
                ]
                if cols:
                    keys.append(tuple(cols))
            if keys:
                unique[table] = sorted(set(keys))

        edges = []
        for table in names:
            groups = defaultdict(list)
            for row in pragma(connection, "foreign_key_list", table):
                groups[row[0]].append(row)
            for rows in groups.values():
                rows.sort(key=lambda x: x[1])
                target = rows[0][2]
                target_pk = primary.get(target, ())
                pairs = []
                for row in rows:
                    target_column = row[4]
                    if target_column is None:
                        if row[1] >= len(target_pk):
                            raise PipelineError(f"cannot resolve implicit FK from {table}.{row[3]}")
                        target_column = target_pk[row[1]]
                    pairs.append((row[3], target_column))
                edges.append(Edge(table, target, tuple(pairs)))
    finally:
        connection.close()

    schema = Schema(tables, types, primary, unique, dedupe_edges(edges), db_id or Path(path).stem)
    if not schema.tables:
        raise PipelineError("database has no user tables")
    for edge in schema.edges:
        schema.validate_edge(edge)
    return schema


def add_tables_json(schema: Schema, path: str | Path, db_id: str | None = None) -> Schema:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload if isinstance(payload, list) else [payload]
    wanted = db_id or schema.db_id
    matches = [entry for entry in entries if entry.get("db_id") == wanted]
    if not matches and len(entries) == 1 and db_id is None:
        matches = entries
    if len(matches) != 1:
        raise PipelineError(f"expected one tables.json entry for {wanted!r}")
    meta = matches[0]
    table_names = meta["table_names_original"]
    columns = meta["column_names_original"]

    def resolve(index):
        if not isinstance(index, int) or not 0 <= index < len(columns):
            raise PipelineError(f"invalid metadata column index {index!r}")
        table_index, column = columns[index]
        if not 0 <= table_index < len(table_names):
            raise PipelineError(f"invalid table index for column {index}")
        table = table_names[table_index]
        if table not in schema.tables or column not in schema.tables[table]:
            raise PipelineError(f"metadata conflicts with SQLite: {table}.{column}")
        return table, column

    for table in table_names:
        if table not in schema.tables:
            raise PipelineError(f"metadata table missing from SQLite: {table}")

    # Expose exactly the benchmark schema to the model, not helper columns
    # that may exist only in the converted SQLite image.
    official_tables = {table: [] for table in table_names}
    for index, (table_index, _column) in enumerate(columns):
        if table_index >= 0:
            table, column = resolve(index)
            official_tables[table].append(column)
    official_types = {
        (table, column): schema.types.get((table, column), "")
        for table, table_columns in official_tables.items()
        for column in table_columns
    }
    official_primary = {
        table: tuple(column for column in key if column in official_tables[table])
        for table, key in schema.primary_keys.items()
        if table in official_tables and any(column in official_tables[table] for column in key)
    }
    official_unique = {
        table: [key for key in keys if all(column in official_tables[table] for column in key)]
        for table, keys in schema.unique_keys.items()
        if table in official_tables
    }
    physical_edges = [
        edge for edge in schema.edges
        if edge.source_table in official_tables
        and edge.target_table in official_tables
        and all(
            source in official_tables[edge.source_table]
            and target in official_tables[edge.target_table]
            for source, target in edge.pairs
        )
    ]

    # Keep physical SQLite primary keys. Some converted benchmark metadata
    # contains repeated/partial PK indices; official FK pairs are still useful.
    metadata_edges = []
    for pair in meta.get("foreign_keys", []):
        if not isinstance(pair, list) or len(pair) != 2:
            raise PipelineError(f"invalid metadata FK {pair!r}")
        st, sc = resolve(pair[0])
        tt, tc = resolve(pair[1])
        metadata_edges.append(Edge(st, tt, ((sc, tc),), "benchmark_metadata"))
    enriched = Schema(
        official_tables, official_types, official_primary, official_unique,
        dedupe_edges([*physical_edges, *metadata_edges]), meta.get("db_id") or schema.db_id,
    )
    for edge in enriched.edges:
        enriched.validate_edge(edge)
    return enriched


class OpenWebUI:
    def __init__(self, url=None, api_key=None, model=None, timeout=120):
        self.url = (url or os.getenv("OPENWEBUI_URL", "")).rstrip("/")
        self.key = api_key or os.getenv("OPENWEBUI_API_KEY", "") or os.getenv("OPENWEBUI_JWT", "")
        self.model = model or os.getenv("OPENWEBUI_MODEL", "")
        self.timeout = timeout
        if not self.url or not self.key or not self.model:
            raise PipelineError(
                "set OPENWEBUI_URL, OPENWEBUI_API_KEY and OPENWEBUI_MODEL"
            )

    def chat(self, system, user, temperature=0.2):
        try:
            response = requests.post(
                self.url + "/api/chat/completions",
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "temperature": temperature,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except requests.Timeout as exc:
            raise PipelineError(f"Open WebUI timeout after {self.timeout}s") from exc
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "unknown"
            raise PipelineError(f"Open WebUI HTTP {code}") from exc
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            raise PipelineError(f"invalid Open WebUI response: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise PipelineError("Open WebUI returned empty content")
        return content.strip()


LINK_PROMPT = """Identify SchemaGraphSQL source and destination tables.
Sources contain WHERE/HAVING filter columns. Destinations supply requested
SELECT values or aggregates. A table may be both; sources may be empty when
there is no filter. Use only exact table names from SCHEMA.
Return only: {"sources":["table"],"destinations":["table"]}"""

FK_PROMPT = """Infer only well-supported missing join relationships.
Both endpoint identifiers must occur exactly in SCHEMA. Require compatible
types and strong key/table naming evidence. Never connect generic columns just
to make the graph connected. Return one edge per line exactly as:
source_table.source_column -> target_table.target_column
Return exactly NONE when no edge is well supported. No markdown or explanation."""

SQL_PROMPT = """Generate exactly one read-only SQLite SELECT query from the
filtered schema and force-union shortest-path backbone. Use only exact supplied
identifiers and explicit JOIN ... ON predicates from ALLOWED EDGES. The filtered
schema is allowed context, not a requirement to join every table. Never use
NATURAL JOIN, comma joins, CROSS JOIN, ILIKE, DATE_TRUNC, EXTRACT, INTERVAL,
CONCAT, TOP, QUALIFY, PostgreSQL :: casts, RIGHT JOIN, or FULL OUTER JOIN.
Return only SQL, without markdown, comments, labels or alternatives. If a query
requires an invented identifier or relationship, return:
SELECT NULL AS schema_graphsql_error WHERE 0;"""


def parse_endpoints(raw, schema):
    if chr(96) * 3 in raw:
        raise PipelineError("endpoint response contains markdown")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"invalid endpoint JSON: {raw!r}") from exc
    if set(data) != {"sources", "destinations"}:
        raise PipelineError("endpoint JSON must have exactly sources and destinations")
    sources, destinations = data["sources"], data["destinations"]
    if not isinstance(sources, list) or not isinstance(destinations, list):
        raise PipelineError("endpoint values must be arrays")
    if not all(isinstance(x, str) for x in sources + destinations):
        raise PipelineError("endpoint names must be strings")
    unknown = (set(sources) | set(destinations)) - set(schema.tables)
    if unknown:
        raise PipelineError(f"model invented table(s): {sorted(unknown)}")
    if not destinations:
        raise PipelineError("no destination table returned")
    return list(dict.fromkeys(sources)), list(dict.fromkeys(destinations))


def affinity(value):
    value = value.upper()
    if not value:
        return "UNKNOWN"
    if "INT" in value:
        return "NUMBER"
    if any(x in value for x in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return "NUMBER"
    if any(x in value for x in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if "BLOB" in value:
        return "BLOB"
    return "OTHER"


def plausible(schema, edge):
    if edge.source_table == edge.target_table or len(edge.pairs) != 1:
        return False
    a, b = edge.pairs[0]
    ta = affinity(schema.types.get((edge.source_table, a), ""))
    tb = affinity(schema.types.get((edge.target_table, b), ""))
    if ta != "UNKNOWN" and tb != "UNKNOWN" and ta != tb:
        return False
    norm = lambda x: re.sub("[^a-z0-9]", "", x.lower())
    na, nb = norm(a), norm(b)
    generic = {"id", "name", "type", "status", "date", "value", "code"}
    score = 2 if na == nb and na not in generic else 0
    if min(len(na), len(nb)) >= 4 and (na.endswith(nb) or nb.endswith(na)):
        score += 1
    sta = norm(edge.source_table).rstrip("s")
    stb = norm(edge.target_table).rstrip("s")
    if stb and stb in na:
        score += 2
    if sta and sta in nb:
        score += 2
    keys_a = set(schema.primary_keys.get(edge.source_table, ()))
    keys_b = set(schema.primary_keys.get(edge.target_table, ()))
    if (a in keys_a and nb.endswith("id")) or (b in keys_b and na.endswith("id")):
        score += 1
    return score >= 2


def infer_edges(schema, llm):
    raw = llm.chat(FK_PROMPT, schema.render())
    if raw == "NONE":
        return []
    if chr(96) * 3 in raw:
        raise PipelineError("FK response contains markdown")
    edges = []
    for line in raw.splitlines():
        match = re.fullmatch(r"\s*(.+)\.([^.]+)\s*->\s*(.+)\.([^.]+)\s*", line)
        if not match:
            raise PipelineError(f"invalid inferred FK: {line!r}")
        st, sc, tt, tc = (x.strip() for x in match.groups())
        edge = Edge(st, tt, ((sc, tc),), "llm_inferred")
        schema.validate_edge(edge)
        if plausible(schema, edge):
            edges.append(edge)
    return dedupe_edges(edges)


@dataclass
class Graph:
    nodes: set[str]
    edges: list[Edge]
    adjacency: dict[str, set[str]]

    @classmethod
    def build(cls, schema, edges):
        adjacency = {table: set() for table in schema.tables}
        edge_list = dedupe_edges(edges)
        for edge in edge_list:
            schema.validate_edge(edge)
            adjacency[edge.source_table].add(edge.target_table)
            adjacency[edge.target_table].add(edge.source_table)
        return cls(set(schema.tables), edge_list, adjacency)


def distances(graph, start):
    result, queue = {start: 0}, deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(graph.adjacency[current]):
            if neighbor not in result:
                result[neighbor] = result[current] + 1
                queue.append(neighbor)
    return result


@dataclass(frozen=True)
class Backbone:
    tables: frozenset[str]
    edges: tuple[Edge, ...]
    pairs: tuple[tuple[str, str], ...]


def force_union(graph, sources: Sequence[str], destinations: Sequence[str]):
    unknown = (set(sources) | set(destinations)) - graph.nodes
    if unknown:
        raise PipelineError(f"unknown endpoints: {sorted(unknown)}")
    if not destinations:
        raise PipelineError("at least one destination is required")
    endpoint_pairs = (
        list(product(sources, destinations))
        if sources else list(combinations(destinations, 2))
    )
    chosen = set(sources) | set(destinations)
    chosen_edges, cache, unreachable = set(), {}, []

    def d(node):
        if node not in cache:
            cache[node] = distances(graph, node)
        return cache[node]

    for source, destination in endpoint_pairs:
        if source == destination:
            continue
        ds = d(source)
        if destination not in ds:
            unreachable.append((source, destination))
            continue
        dd, length = d(destination), ds[destination]
        chosen.update(
            node for node in graph.nodes
            if node in ds and node in dd and ds[node] + dd[node] == length
        )
        for edge in graph.edges:
            a, b = edge.source_table, edge.target_table
            if (
                (a in ds and b in dd and ds[a] + 1 + dd[b] == length)
                or (b in ds and a in dd and ds[b] + 1 + dd[a] == length)
            ):
                chosen_edges.add(edge.key())
    if unreachable:
        raise UnreachableError(unreachable)
    return Backbone(
        frozenset(chosen),
        tuple(edge for edge in graph.edges if edge.key() in chosen_edges),
        tuple(endpoint_pairs),
    )


def plain_sql(raw):
    sql = raw.strip()
    if not sql or chr(96) * 3 in sql or "--" in sql or "/*" in sql or "*/" in sql:
        raise PipelineError("model output is not plain uncommented SQL")
    first = re.match("[A-Za-z]+", sql)
    if not first or first.group().upper() not in {"SELECT", "WITH"}:
        raise PipelineError("SQL must start with SELECT or WITH")
    return sql


def run_paper(schema, question, evidence, llm, manual_sources=None, manual_destinations=None):
    if manual_destinations is None:
        raw = llm.chat(
            LINK_PROMPT,
            f"SCHEMA:\n{schema.render()}\n\nQUESTION:\n{question}\n\nEVIDENCE:\n{evidence or 'NONE'}",
        )
        sources, destinations = parse_endpoints(raw, schema)
    else:
        sources, destinations = list(manual_sources or []), list(manual_destinations)
        unknown = (set(sources) | set(destinations)) - set(schema.tables)
        if unknown or not destinations:
            raise PipelineError(f"invalid manual endpoints: {sorted(unknown)}")

    inferred = []
    edges = list(schema.edges)
    if not edges:
        inferred = infer_edges(schema, llm)
        edges = dedupe_edges([*edges, *inferred])
    try:
        backbone = force_union(Graph.build(schema, edges), sources, destinations)
    except UnreachableError:
        temp_schema = Schema(
            schema.tables, schema.types, schema.primary_keys, schema.unique_keys, edges, schema.db_id
        )
        extra = infer_edges(temp_schema, llm)
        inferred = dedupe_edges([*inferred, *extra])
        edges = dedupe_edges([*edges, *extra])
        backbone = force_union(Graph.build(schema, edges), sources, destinations)

    filtered = schema.subset(backbone.tables, backbone.edges)
    edge_text = "\n".join("- " + edge.render() for edge in backbone.edges) or "NONE (zero-hop)"
    user = (
        f"FILTERED SCHEMA:\n{filtered.render()}\n\n"
        f"ALLOWED TABLES:\n{', '.join(sorted(backbone.tables))}\n\n"
        f"ALLOWED EDGES:\n{edge_text}\n\nQUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{evidence or 'NONE'}"
    )
    sql = plain_sql(llm.chat(SQL_PROMPT, user))
    return {
        "mode": "paper",
        "sources": sources,
        "destinations": destinations,
        "filtered_tables": sorted(backbone.tables),
        "endpoint_pairs": [list(pair) for pair in backbone.pairs],
        "backbone_edges": [edge.render() for edge in backbone.edges],
        "inferred_edges": [edge.render() for edge in inferred],
        "sql": sql,
    }


def run_single(schema, question, evidence, llm, prompt_path):
    prompt = Path(prompt_path)
    if not prompt.is_file():
        raise FileNotFoundError(prompt)
    raw = llm.chat(
        prompt.read_text(encoding="utf-8"),
        f"SCHEMA:\n{schema.render()}\n\nQUESTION:\n{question}\n\nEVIDENCE:\n{evidence or 'NONE'}",
    )
    return {"mode": "single-call-approximation", "sql": plain_sql(raw)}


def authorizer(allowed_tables):
    denied = {
        getattr(sqlite3, name) for name in (
            "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_CREATE_INDEX",
            "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TEMP_TABLE", "SQLITE_CREATE_TRIGGER",
            "SQLITE_CREATE_VIEW", "SQLITE_DROP_INDEX", "SQLITE_DROP_TABLE",
            "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VIEW", "SQLITE_ALTER_TABLE",
            "SQLITE_REINDEX", "SQLITE_ANALYZE", "SQLITE_PRAGMA", "SQLITE_ATTACH",
            "SQLITE_DETACH", "SQLITE_TRANSACTION", "SQLITE_SAVEPOINT",
        ) if hasattr(sqlite3, name)
    }
    allowed = set(allowed_tables)

    def check(action, arg1, arg2, database_name, trigger_name):
        del database_name, trigger_name
        if action in denied:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and arg1 not in allowed:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").lower() == "load_extension":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return check


def validate_sql(database, sql, allowed_tables, execute=False, max_rows=100, timeout=30):
    if max_rows < 1 or timeout <= 0:
        raise PipelineError("max_rows and timeout must be positive")
    sql = plain_sql(sql)
    connection = ro_connect(database)
    connection.set_authorizer(authorizer(allowed_tables))
    deadline = time.monotonic() + timeout
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10_000)
    try:
        try:
            connection.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        except sqlite3.Error as exc:
            raise PipelineError(f"SQLite validation failed: {exc}") from exc
        result = {"validated": True, "executed": False}
        if execute:
            try:
                cursor = connection.execute(sql)
                columns = [item[0] for item in cursor.description or []]
                rows = cursor.fetchmany(max_rows + 1)
            except sqlite3.Error as exc:
                raise PipelineError(f"SQLite execution failed: {exc}") from exc
            result.update({
                "executed": True,
                "columns": columns,
                "rows": [[value.hex() if isinstance(value, bytes) else value for value in row]
                         for row in rows[:max_rows]],
                "truncated": len(rows) > max_rows,
            })
        return result
    finally:
        connection.close()


def sb_paths(name):
    db_id, folder = DATABASES[name]
    database = DATA_ROOT / "database" / db_id / f"{db_id}.sqlite"
    metadata = DATA_ROOT / folder / "tables.json"
    return database, metadata if metadata.is_file() else None, db_id


def split_names(value):
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def make_parser():
    result = argparse.ArgumentParser(description="SchemaGraphSQL + ScienceBenchmark SQLite")
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--sciencebenchmark", choices=sorted(DATABASES))
    source.add_argument("--database")
    result.add_argument("--tables-json")
    result.add_argument("--db-id")
    result.add_argument("--question")
    result.add_argument("--evidence", default="")
    result.add_argument("--mode", choices=("paper", "single-call"), default="paper")
    result.add_argument("--source-tables")
    result.add_argument("--destination-tables")
    result.add_argument("--single-call-prompt", default=str(DEFAULT_SINGLE_PROMPT))
    result.add_argument("--inspect-schema", action="store_true")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--max-rows", type=int, default=100)
    result.add_argument("--query-timeout", type=float, default=30)
    result.add_argument("--sql-only", action="store_true")
    result.add_argument("--openwebui-url")
    result.add_argument("--api-key")
    result.add_argument("--model")
    result.add_argument("--api-timeout", type=float, default=120)
    return result


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        if args.sciencebenchmark:
            database, auto_meta, db_id = sb_paths(args.sciencebenchmark)
        else:
            database, auto_meta, db_id = Path(args.database), None, args.db_id
        schema = introspect(database, db_id)
        metadata = Path(args.tables_json) if args.tables_json else auto_meta
        if metadata:
            schema = add_tables_json(schema, metadata, db_id)

        if args.inspect_schema:
            print(json.dumps(schema.summary(), ensure_ascii=False, indent=2))
            return 0
        if not args.question:
            raise PipelineError("--question is required")
        llm = OpenWebUI(args.openwebui_url, args.api_key, args.model, args.api_timeout)
        if args.mode == "paper":
            manual_destinations = (
                split_names(args.destination_tables)
                if args.destination_tables is not None else None
            )
            output = run_paper(
                schema, args.question, args.evidence, llm,
                split_names(args.source_tables), manual_destinations,
            )
            allowed = output["filtered_tables"]
        else:
            if args.source_tables or args.destination_tables:
                raise PipelineError("manual endpoints are only valid in paper mode")
            output = run_single(
                schema, args.question, args.evidence, llm, args.single_call_prompt
            )
            allowed = schema.tables

        output["database"] = str(Path(database).resolve())
        output["schema"] = schema.summary()
        output["validation"] = validate_sql(
            database, output["sql"], allowed, args.execute, args.max_rows, args.query_timeout
        )
        print(output["sql"] if args.sql_only else json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (PipelineError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
