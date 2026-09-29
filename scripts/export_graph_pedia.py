#!/usr/bin/env python
"""A0 (PORTABILIDADE.md): export the legacy graph_pedia stores to a DGL-independent format.

Runs in the LEGACY_REFERENCE image (needs DGL 0.8.2 to unpickle the stored DGLGraphs):
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e DGLBACKEND=pytorch \
    -e GRAPHIX_CODE_COMMIT="$(git rev-parse HEAD)" \
    -v "$PWD/seq2seq":/app/seq2seq:ro -v "$PWD/scripts":/app/scripts:ro \
    -v "$PWD/data_all_in":/app/data_all_in -w /app eyuansu62/graphix-text-to-sql:v2 \
    python scripts/export_graph_pedia.py
Writes data_all_in/data/graph_export/<name>/ (format: seq2seq/utils/graph_export.py) and
data_all_in/data/graph_export/manifest.json with SHA-256 of sources and artifacts.
Validate with scripts/validate_graph_export.py (test T2) before using the export.
"""
import json
import os
import pickle
import platform
import shelve
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from seq2seq.models.graphix.constants import GRAPHIX_RELATIONS
from seq2seq.utils.graph_export import ARRAYS, encode_value, file_sha256, graph_sha256

OUT = Path("data_all_in/data/graph_export")
SPIDER, SCIB = "data_all_in/data/output", "data_all_in/data/sciencebenchmark/output"
# name -> (graph_pedia source, datasets whose examples point into it by graph_idx)
EXPORTS = {
    "spider_total": (f"{SPIDER}/graph_pedia_total.bin",
                     [f"{SPIDER}/seq2seq_train_dataset.json", f"{SPIDER}/seq2seq_dev_dataset.json"]),
    "sciencebenchmark_train": (f"{SCIB}/graph_pedia_train.bin", [f"{SCIB}/seq2seq_train_dataset.json"]),
    "sciencebenchmark_dev": (f"{SCIB}/graph_pedia_dev.bin", [f"{SCIB}/seq2seq_dev_dataset.json"]),
}
STRUCTURAL = ("graph", "edges")


def open_store(path):
    """(store, container kind, source files). Shelve stores keep str keys on disk."""
    if os.path.exists(path + ".dat"):
        return shelve.open(path, flag="r"), "shelve(dbm.dumb)", [path + ".dat", path + ".dir"]
    with open(path, "rb") as f:
        return pickle.load(f), "pickle", [path]


def export(name, source, datasets):
    store, kind, source_files = open_store(source)
    raw_keys = list(store.keys())
    keys = sorted(int(k) for k in raw_keys)
    key_type = sorted({type(k).__name__ for k in raw_keys})
    rel_id = {r: i for i, r in enumerate(GRAPHIX_RELATIONS)}
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)

    num_nodes, edge_ptr, srcs, dsts, rels = [], [0], [], [], []
    fields = None
    with open(d / "meta.jsonl", "w", encoding="utf-8") as meta_f:
        for k in keys:
            e = store[str(k)] if key_type == ["str"] else store[k]
            g = e["graph"]
            assert g.is_homogeneous and not g.ndata.keys() and not g.edata.keys(), (name, k)
            assert str(g.idtype) == "torch.int32", (name, k, g.idtype)
            s, t = g.edges(order="eid")
            s, t = s.numpy().astype("<i4"), t.numpy().astype("<i4")
            listed = e["edges"]
            # The model reads relation names from the 'edges' list and message-passes on
            # the DGLGraph; both must describe the same edges in the same order.
            assert len(listed) == g.number_of_edges() == len(s), (name, k)
            assert all(a == int(x) and b == int(y) for (a, b, _), x, y in zip(listed, s, t)), (name, k)
            r = np.array([rel_id[rel] for _, _, rel in listed], dtype="<i2")
            meta = {f: encode_value(v) for f, v in e.items() if f not in STRUCTURAL}
            if fields is None:
                fields = sorted(meta)
            assert sorted(meta) == fields, (name, k, sorted(meta))
            n = g.number_of_nodes()
            line = dict(meta, graph_idx=k, sha256=graph_sha256(n, s, t, r, meta))
            meta_f.write(json.dumps(line, ensure_ascii=False) + "\n")
            num_nodes.append(n)
            srcs.append(s)
            dsts.append(t)
            rels.append(r)
            edge_ptr.append(edge_ptr[-1] + len(s))

    arrays = {
        "keys": np.array(keys, dtype="<i8"), "num_nodes": np.array(num_nodes, dtype="<i8"),
        "edge_ptr": np.array(edge_ptr, dtype="<i8"),
        "src": np.concatenate(srcs), "dst": np.concatenate(dsts), "rel": np.concatenate(rels),
    }
    for a in ARRAYS:
        np.save(str(d / (a + ".npy")), arrays[a], allow_pickle=False)
    json.dump(GRAPHIX_RELATIONS, open(d / "relations.json", "w"), indent=0)

    # Example <-> graph association, identified by dataset file and position.
    with open(d / "examples.jsonl", "w", encoding="utf-8") as ex_f:
        for ds in datasets:
            for pos, ex in enumerate(json.load(open(ds))):
                ex_f.write(json.dumps({
                    "dataset": ds, "position": pos, "graph_idx": int(ex["graph_idx"]),
                    "db_id": ex["db_id"], "question": " ".join(ex["raw_question_toks"]),
                }, ensure_ascii=False) + "\n")

    if hasattr(store, "close"):
        store.close()
    files = sorted(p.name for p in d.iterdir())
    return {
        "source": {"path": source, "container": kind, "key_type": key_type,
                   "files": {p: file_sha256(p) for p in source_files}},
        "datasets": {p: file_sha256(p) for p in datasets},
        "n_graphs": len(keys), "n_edges": int(edge_ptr[-1]),
        "key_range": [keys[0], keys[-1]], "graph_idtype": "int32", "meta_fields": fields,
        "artifacts": {f: {"sha256": file_sha256(d / f), "bytes": (d / f).stat().st_size} for f in files},
    }


def main():
    import dgl
    import torch

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "step": "A0", "format": "seq2seq/utils/graph_export.py",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": os.environ.get("GRAPHIX_CODE_COMMIT"),
        "environment": {"python": platform.python_version(), "dgl": dgl.__version__,
                        "torch": torch.__version__, "numpy": np.__version__},
        "relations": GRAPHIX_RELATIONS, "exports": {},
    }
    for name, (source, datasets) in EXPORTS.items():
        print("exporting", name, "from", source, flush=True)
        manifest["exports"][name] = export(name, source, datasets)
        print("  ", manifest["exports"][name]["n_graphs"], "graphs,",
              manifest["exports"][name]["n_edges"], "edges", flush=True)
    manifest["seconds"] = round(time.time() - t0, 1)
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
