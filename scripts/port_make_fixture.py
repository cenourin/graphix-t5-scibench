#!/usr/bin/env python
"""Fixed example set shared by the phase-A equivalence tests T3-T10 (PORTABILIDADE.md §5).

Chosen once by explicit criteria over the A0 graph export and frozen into
tests/port/fixture.json (versioned), so every test and both environments use exactly
the same examples. 8 Spider + 8 ScienceBenchmark:
  Spider: smallest graph in dev and in train; most distinct relation types; largest
          graph; then train/dev examples at fixed positions.
  ScienceBenchmark: largest oncomx graph (train); largest graph per remaining db
          (cordis, sdss); most distinct relation types; one dev example per db; one
          fixed train position.
Run anywhere with numpy (e.g. the graphix-modern image): python scripts/port_make_fixture.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from seq2seq.utils.graph_export import GraphExport

EXPORT = Path("data_all_in/data/graph_export")
SPIDER, SCIB = "data_all_in/data/output", "data_all_in/data/sciencebenchmark/output"
OUT = Path("tests/port/fixture.json")


def examples(dataset, export_name):
    ex = GraphExport(EXPORT / export_name)
    rows = []
    for pos, e in enumerate(json.load(open(dataset))):
        k = int(e["graph_idx"])
        i = ex._pos[k]
        lo, hi = int(ex.edge_ptr[i]), int(ex.edge_ptr[i + 1])
        rows.append({"bench": "spider" if dataset.startswith(SPIDER) else "sciencebenchmark",
                     "dataset": dataset, "position": pos, "export": export_name, "graph_idx": k,
                     "db_id": e["db_id"], "num_nodes": int(ex.num_nodes[i]), "num_edges": hi - lo,
                     "relation_types": int(len(np.unique(ex.rel[lo:hi])))})
    return rows


def main():
    sp_tr = examples(SPIDER + "/seq2seq_train_dataset.json", "spider_total")
    sp_dev = examples(SPIDER + "/seq2seq_dev_dataset.json", "spider_total")
    sb_tr = examples(SCIB + "/seq2seq_train_dataset.json", "sciencebenchmark_train")
    sb_dev = examples(SCIB + "/seq2seq_dev_dataset.json", "sciencebenchmark_dev")
    chosen, seen = [], set()

    def take(row, reason):
        key = (row["dataset"], row["position"])
        if key not in seen:
            seen.add(key)
            chosen.append(dict(row, reason=reason))

    take(min(sp_dev, key=lambda r: r["num_nodes"]), "spider: smallest graph (dev)")
    take(min(sp_tr, key=lambda r: r["num_nodes"]), "spider: smallest graph (train)")
    take(max(sp_tr, key=lambda r: (r["relation_types"], r["num_edges"])), "spider: most relation types")
    take(max(sp_tr + sp_dev, key=lambda r: r["num_nodes"]), "spider: largest graph")
    for pos in (0, 4000, 8000):
        take(sp_tr[pos], "spider: fixed train position %d" % pos)
    take(sp_dev[500], "spider: fixed dev position 500")

    dbs = sorted({r["db_id"] for r in sb_tr})
    onc = [d for d in dbs if d.startswith("oncomx")][0]
    take(max((r for r in sb_tr if r["db_id"] == onc), key=lambda r: r["num_nodes"]), "scibench: largest oncomx graph")
    for db in dbs:
        if db != onc:
            take(max((r for r in sb_tr if r["db_id"] == db), key=lambda r: r["num_nodes"]),
                 "scibench: largest graph of " + db)
    take(max(sb_tr, key=lambda r: (r["relation_types"], r["num_edges"])), "scibench: most relation types")
    for db in dbs:
        take(next(r for r in sb_dev if r["db_id"] == db), "scibench: first dev example of " + db)
    take(sb_tr[2000], "scibench: fixed train position 2000")

    assert sum(r["bench"] == "spider" for r in chosen) == 8 and sum(r["bench"] == "sciencebenchmark" for r in chosen) == 8, \
        [r["reason"] for r in chosen]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"source": "data_all_in/data/graph_export (A0)", "examples": chosen}, open(OUT, "w"), indent=1)
    for r in chosen:
        print("%-45s %-28s nodes=%5d edges=%6d rel_types=%2d" % (r["reason"], r["db_id"], r["num_nodes"], r["num_edges"], r["relation_types"]))


if __name__ == "__main__":
    main()
