#!/usr/bin/env python
"""Test T2 (PORTABILIDADE.md, step A0): the DGL-independent export reproduces every legacy
graph exactly. Runs in the LEGACY_REFERENCE image, next to the original DGLGraphs.

For every graph of every export, rebuilt with dgl.graph((src, dst), num_nodes, int32)
from the neutral arrays and compared with the original pickled/shelved entry:
  - number of nodes and of edges;
  - src and dst, element by element, in edge-id order (so edge order is checked);
  - relation of every edge (against the original 'edges' list);
  - every metadata field, with dict key types, order and defaultdict type;
  - per-graph content hash and the key set (no graph missing or extra).
And for every dataset example: its graph_idx exists, its new_struct_in equals the
graph's, and its db_id is the database named in that schema. Artifacts are checked
against the SHA-256 in manifest.json first. Exit status 0 only if everything passes;
the report goes to data_all_in/data/graph_export/T2_report.json.

Same docker invocation as scripts/export_graph_pedia.py, with this script instead.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from seq2seq.utils.graph_export import GraphExport, file_sha256, graph_sha256

OUT = Path("data_all_in/data/graph_export")


def same_value(a, b):
    """Equality that also requires the same dict kind, key types and key order."""
    # A numpy integer in the original is exported as a plain int (encode_value).
    a = int(a) if isinstance(a, np.integer) else a
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict)
                and isinstance(a, defaultdict) == isinstance(b, defaultdict)
                and [(type(k), k) for k in a] == [(type(k), k) for k in b]
                and all(same_value(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(same_value(x, y) for x, y in zip(a, b)))
    return type(a) == type(b) and a == b


def check_export(name, info, failures):
    import dgl
    import torch
    sys.path.insert(0, "scripts")
    from export_graph_pedia import open_store

    d = OUT / name
    for f, meta in info["artifacts"].items():
        if file_sha256(d / f) != meta["sha256"]:
            failures.append((name, "artifact sha256", f))
    for p, h in info["source"]["files"].items():
        if file_sha256(p) != h:
            failures.append((name, "source sha256", p))

    ex = GraphExport(d)
    store, _, _ = open_store(info["source"]["path"])
    str_keys = info["source"]["key_type"] == ["str"]
    original_keys = sorted(int(k) for k in store.keys())
    if original_keys != [int(k) for k in ex.keys.tolist()]:
        failures.append((name, "key set", "export keys differ from the original store"))

    checked = 0
    for i, k in enumerate(ex.keys.tolist()):
        orig = store[str(k)] if str_keys else store[k]
        g0 = orig["graph"]
        n, src, dst, rel = ex.structure(k)
        g1 = dgl.graph((torch.from_numpy(src.astype(np.int64)), torch.from_numpy(dst.astype(np.int64))),
                       num_nodes=n, idtype=torch.int32)
        s0, t0 = g0.edges(order="eid")
        s1, t1 = g1.edges(order="eid")
        errs = []
        if g1.number_of_nodes() != g0.number_of_nodes():
            errs.append("num_nodes")
        if g1.number_of_edges() != g0.number_of_edges():
            errs.append("num_edges")
        if not (torch.equal(s0.long(), s1.long()) and torch.equal(t0.long(), t1.long())):
            errs.append("src/dst or edge order")
        entry = ex.entry(k)
        if [tuple(e) for e in orig["edges"]] != entry["edges"]:
            errs.append("edges list (src, dst, relation)")
        line = ex.meta_lines()[i]
        if line["graph_idx"] != k:
            errs.append("meta line order")
        meta_fields = [f for f in orig if f not in ("graph", "edges")]
        if sorted(meta_fields) != sorted(f for f in entry if f not in ("edges", "num_nodes")):
            errs.append("metadata field set")
        for f in meta_fields:
            if not same_value(orig[f], entry[f]):
                errs.append("field " + f)
        encoded = {f: v for f, v in line.items() if f not in ("graph_idx", "sha256")}
        if graph_sha256(n, src, dst, rel, encoded) != line["sha256"]:
            errs.append("graph sha256")
        if errs:
            failures.append((name, k, errs))
        checked += 1
    if hasattr(store, "close"):
        store.close()

    n_examples = 0
    ds_cache = {}
    with open(d / "examples.jsonl", encoding="utf-8") as f:
        for row in (json.loads(l) for l in f):
            n_examples += 1
            k = row["graph_idx"]
            if k not in ex:
                failures.append((name, "example without graph", row["dataset"], row["position"]))
                continue
            if row["dataset"] not in ds_cache:
                ds_cache[row["dataset"]] = json.load(open(row["dataset"]))
            example = ds_cache[row["dataset"]][row["position"]]
            struct = ex.entry(k)["new_struct_in"]
            if int(example["graph_idx"]) != k or example["new_struct_in"] != struct:
                failures.append((name, "example/graph mismatch", row["dataset"], row["position"]))
            elif not struct.startswith("schema: | %s |" % example["db_id"]):
                failures.append((name, "db_id not in schema", row["dataset"], row["position"]))
    return {"graphs_checked": checked, "graphs_in_export": len(ex), "examples_checked": n_examples}


def main():
    t0 = time.time()
    manifest = json.load(open(OUT / "manifest.json"))
    failures, per_export = [], {}
    for name, info in manifest["exports"].items():
        print("T2:", name, flush=True)
        per_export[name] = check_export(name, info, failures)
        print("  ", per_export[name], flush=True)
    passed = not failures and all(v["graphs_checked"] == v["graphs_in_export"] for v in per_export.values())
    report = {"test": "T2", "passed": passed, "per_export": per_export,
              "failures": [list(map(str, f)) for f in failures[:200]], "n_failures": len(failures),
              "seconds": round(time.time() - t0, 1),
              "manifest_sha256": file_sha256(OUT / "manifest.json")}
    json.dump(report, open(OUT / "T2_report.json", "w"), indent=2)
    print("T2", "PASSED" if passed else "FAILED (%d failures)" % len(failures))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
