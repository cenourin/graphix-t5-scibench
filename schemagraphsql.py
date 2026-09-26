"""
SchemaGraphSQL — training-free schema linking + Text-to-SQL pipeline.

Implements the full pipeline from:
  Safdarian et al., "SchemaGraphSQL: Efficient Schema Linking with
  Pathfinding Graph Algorithms for Text-to-SQL on Large-Scale Databases"
  (Findings of ACL: EACL 2026)

Pipeline (force-union configuration, the paper's primary/recall-oriented setting):
  1. Build a table graph: from declared foreign keys, or (when FKs are
     missing/unreliable) via LLM-guided joinability discovery (Prompt 6).
  2. Identify source/destination tables with a single LLM call (Prompt 2, n-n).
  3. Compute the shortest join path for every (source, destination) pair
     with Dijkstra and take the UNION of all paths -> filtered sub-schema.
  4. Generate the final SQLite query from the filtered schema (Prompt 4).

All LLM calls are routed through an Open WebUI instance's OpenAI-compatible
chat completions endpoint, authenticated with a JWT bearer token.

Env vars:
  OPENWEBUI_URL     e.g. http://localhost:3000   (no trailing slash)
  OPENWEBUI_JWT     JWT token (Settings > Account > API Keys, or browser devtools)
  OPENWEBUI_MODEL   model id as registered in Open WebUI, e.g. "gpt-4o-mini"

Usage:
  python schemagraphsql.py --schema schema.json --question "Which store sold product 1?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable

import networkx as nx
import requests

# --------------------------------------------------------------------------
# Open WebUI client (JWT-authenticated, OpenAI-compatible chat endpoint)
# --------------------------------------------------------------------------

class OpenWebUIClient:
    def __init__(self, base_url: str | None = None, jwt: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.environ.get("OPENWEBUI_URL", "")).rstrip("/")
        self.jwt = jwt or os.environ.get("OPENWEBUI_JWT", "")
        self.model = model or os.environ.get("OPENWEBUI_MODEL", "")
        if not self.base_url or not self.jwt or not self.model:
            raise RuntimeError(
                "Missing Open WebUI config. Set OPENWEBUI_URL, OPENWEBUI_JWT, "
                "OPENWEBUI_MODEL (env vars) or pass them explicitly."
            )

    def chat(self, system_prompt: str, user_content: str, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/api/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.jwt}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------
# Schema model
# --------------------------------------------------------------------------

@dataclass
class Schema:
    tables: dict[str, list[str]]                 # table -> [column, ...]
    foreign_keys: list[tuple[str, str, str, str]] = field(default_factory=list)
    # each FK: (src_table, src_col, dst_table, dst_col)

    def has_declared_fks(self) -> bool:
        return len(self.foreign_keys) > 0

    def render_ddl_like(self) -> str:
        lines = []
        for table, cols in self.tables.items():
            lines.append(f"TABLE {table} ({', '.join(cols)})")
        for s_t, s_c, d_t, d_c in self.foreign_keys:
            lines.append(f"FOREIGN KEY {s_t}.{s_c} -> {d_t}.{d_c}")
        return "\n".join(lines)

    def subset(self, tables: Iterable[str]) -> "Schema":
        tables = set(tables)
        return Schema(
            tables={t: cols for t, cols in self.tables.items() if t in tables},
            foreign_keys=[fk for fk in self.foreign_keys if fk[0] in tables and fk[2] in tables],
        )


# --------------------------------------------------------------------------
# Step 1 — Build the schema graph
# --------------------------------------------------------------------------

FK_INFERENCE_PROMPT = """ROLE & OBJECTIVE
You are a database schema analyzer. Your job is to infer likely foreign key
relationships from SQL DDL statements that do NOT explicitly define foreign keys.
Infer relationships purely from table and column names (e.g., 'user_id' likely
references 'users.id').

INSTRUCTIONS
- Respond with one inferred foreign key per line in the exact format:
  source_table.source_column -> target_table.target_column
- Use lowercase and uppercase table and column names letters exactly as they
  appear in the DDL.
- Do NOT add explanations, comments, or extra text.
- If no foreign keys can be reasonably inferred, respond with exactly: NONE
"""


def build_schema_graph(schema: Schema, llm: OpenWebUIClient) -> nx.Graph:
    """Table-level undirected graph; edges = joinable table pairs."""
    graph = nx.Graph()
    graph.add_nodes_from(schema.tables.keys())

    if schema.has_declared_fks():
        for s_t, _s_c, d_t, _d_c in schema.foreign_keys:
            if s_t in schema.tables and d_t in schema.tables:
                graph.add_edge(s_t, d_t)
        return graph

    # LLM-guided joinability discovery (Section 2, Prompt 6)
    raw = llm.chat(FK_INFERENCE_PROMPT, schema.render_ddl_like())
    if raw.strip().upper() == "NONE":
        return graph

    for line in raw.splitlines():
        line = line.strip()
        if "->" not in line:
            continue
        left, right = (p.strip() for p in line.split("->", 1))
        if "." not in left or "." not in right:
            continue
        s_t, s_c = left.rsplit(".", 1)
        d_t, d_c = right.rsplit(".", 1)
        # plausibility filter: tables/columns must actually exist
        if s_t in schema.tables and d_t in schema.tables \
                and s_c in schema.tables[s_t] and d_c in schema.tables[d_t]:
            graph.add_edge(s_t, d_t)
    return graph


# --------------------------------------------------------------------------
# Step 2 — Identify source(s) and destination(s)  (Prompt 2, n-n / force-union)
# --------------------------------------------------------------------------

SRC_DST_PROMPT = """ROLE & OBJECTIVE
You are a senior data engineer who analyses SQL schemas and maps user questions
precisely to source tables (filtering/conditions) and destination tables
(final result columns).

TASK
Identify:
- Source table(s) (src): contain columns used in filters/conditions.
- Destination table(s) (dst): contain columns returned in the answer.

INSTRUCTIONS
1. Internally inspect every table to determine
   - which tables participate in filtering, and
   - which tables supply the requested output columns.
   Briefly justify your choice internally but do NOT include that
   justification in the final answer.
2. Output exactly one line in the following format:
   src=TableA,TableB, dst=TableC,TableD
"""


def identify_source_destination(schema: Schema, question: str, llm: OpenWebUIClient,
                                 evidence: str = "") -> tuple[list[str], list[str]]:
    user_content = f"Schema:\n{schema.render_ddl_like()}\n\nQuestion: {question}"
    if evidence:
        user_content += f"\nEvidence: {evidence}"

    raw = llm.chat(SRC_DST_PROMPT, user_content)
    src_part, dst_part = "", ""
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk.lower().startswith("src="):
            src_part = chunk[4:].strip()
        elif chunk.lower().startswith("dst="):
            dst_part = chunk[4:].strip()
        elif src_part and not dst_part:
            src_part += "," + chunk
        elif dst_part:
            dst_part += "," + chunk

    sources = [t.strip() for t in src_part.split(",") if t.strip() in schema.tables]
    destinations = [t.strip() for t in dst_part.split(",") if t.strip() in schema.tables]

    if not sources or not destinations:
        raise ValueError(f"Could not parse src/dst from LLM output: {raw!r}")
    return sources, destinations


# --------------------------------------------------------------------------
# Step 3 — Shortest-path backbone (force-union configuration)
# --------------------------------------------------------------------------

def shortest_path_backbone(graph: nx.Graph, sources: list[str], destinations: list[str]) -> set[str]:
    """Union of shortest join paths across all (source, destination) pairs."""
    filtered_tables: set[str] = set(sources) | set(destinations)
    for src, dst in product(sources, destinations):
        if src == dst:
            continue
        try:
            path = nx.shortest_path(graph, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        filtered_tables.update(path)
    return filtered_tables


# --------------------------------------------------------------------------
# Step 4 — SQL generation after schema linking (Prompt 4)
# --------------------------------------------------------------------------

SQL_GEN_PROMPT = """ROLE & OBJECTIVE
You are an expert in SQLite query generation. Your task is to generate a valid
query to answer a user question based on the given schema and join path.

INSTRUCTIONS
1. Use the provided schema and join path to construct a valid SQLite query.
2. Ensure the query correctly answers the user's question.
3. Format the query clearly and confirm it adheres to SQLite syntax.
Return ONLY the SQL query, nothing else.
"""


def generate_sql(filtered_schema: Schema, join_tables: set[str], question: str,
                  llm: OpenWebUIClient, evidence: str = "") -> str:
    user_content = (
        f"Schema:\n{filtered_schema.render_ddl_like()}\n\n"
        f"Join Path (tables): {', '.join(sorted(join_tables))}\n\n"
        f"Question Context: {question}"
    )
    if evidence:
        user_content += f"\nEvidence: {evidence}"
    return llm.chat(SQL_GEN_PROMPT, user_content)


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------

def schemagraphsql(schema: Schema, question: str, llm: OpenWebUIClient,
                    evidence: str = "") -> dict:
    graph = build_schema_graph(schema, llm)
    sources, destinations = identify_source_destination(schema, question, llm, evidence)
    join_tables = shortest_path_backbone(graph, sources, destinations)
    filtered_schema = schema.subset(join_tables)
    sql = generate_sql(filtered_schema, join_tables, question, llm, evidence)
    return {
        "sources": sources,
        "destinations": destinations,
        "filtered_tables": sorted(join_tables),
        "sql": sql,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_schema(path: str) -> Schema:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    fks = [tuple(fk) for fk in data.get("foreign_keys", [])]
    return Schema(tables=data["tables"], foreign_keys=fks)


def main():
    parser = argparse.ArgumentParser(description="SchemaGraphSQL over Open WebUI")
    parser.add_argument("--schema", required=True, help="Path to schema JSON "
                         '({"tables": {"t": ["col", ...]}, "foreign_keys": [[s_t,s_c,d_t,d_c], ...]})')
    parser.add_argument("--question", required=True)
    parser.add_argument("--evidence", default="")
    parser.add_argument("--openwebui-url", default=None)
    parser.add_argument("--openwebui-jwt", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    llm = OpenWebUIClient(base_url=args.openwebui_url, jwt=args.openwebui_jwt, model=args.model)
    result = schemagraphsql(schema, args.question, llm, args.evidence)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
