"""Reader for the DGL-independent graph_pedia export (PORTABILIDADE.md, step A0).

The legacy graph_pedia_*.bin files pickle DGLGraph objects from DGL 0.8.2 / Python 3.7,
which newer DGL versions are not guaranteed to load. scripts/export_graph_pedia.py
rewrites each one, in the legacy image, into plain arrays and JSON that any Python 3.7+
with numpy can read. This module needs only numpy and the standard library, so the same
reader serves the legacy validator (scripts/validate_graph_export.py) and the modern stack.

Layout of one export directory (all arrays little-endian .npy, allow_pickle=False):
  keys.npy        int64 [N]   graph_idx of each graph, ascending
  num_nodes.npy   int64 [N]
  edge_ptr.npy    int64 [N+1] edges of graph i are [edge_ptr[i], edge_ptr[i+1])
  src.npy, dst.npy int32 [E]  in the original DGL edge-id order
  rel.npy         int16 [E]   index into relations.json
  relations.json              relation vocabulary (GRAPHIX_RELATIONS order)
  meta.jsonl                  one line per graph, same order as keys.npy: every
                              non-graph field of the original entry (see encode_value),
                              plus "graph_idx" and "sha256" (see graph_sha256)
  examples.jsonl              example <-> graph association for the datasets using it
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARRAYS = ("keys", "num_nodes", "edge_ptr", "src", "dst", "rel")


def encode_value(v):
    """JSON-safe encoding that keeps what plain JSON would lose: dict key types (int)
    and order, and the defaultdict type (defaultdict(None) in the legacy data)."""
    if isinstance(v, dict):
        kind = "defaultdict" if isinstance(v, defaultdict) else "dict"
        for k in v:
            if not isinstance(k, int):
                raise TypeError("only int dict keys are supported, got %r" % (k,))
        return {"__type__": kind, "items": [[k, encode_value(x)] for k, x in v.items()]}
    if isinstance(v, (list, tuple)):
        return [encode_value(x) for x in v]
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, np.integer):
        return int(v)
    raise TypeError("unsupported type in graph_pedia entry: %s" % type(v).__name__)


def decode_value(v):
    if isinstance(v, dict) and "__type__" in v:
        items = [(k, decode_value(x)) for k, x in v["items"]]
        return defaultdict(None, items) if v["__type__"] == "defaultdict" else dict(items)
    if isinstance(v, list):
        return [decode_value(x) for x in v]
    return v


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def graph_sha256(num_nodes, src, dst, rel, meta_encoded):
    """Content hash of one graph: structure, relations and all metadata fields."""
    h = hashlib.sha256()
    h.update(np.int64(num_nodes).tobytes())
    h.update(np.ascontiguousarray(src, dtype="<i4").tobytes())
    h.update(np.ascontiguousarray(dst, dtype="<i4").tobytes())
    h.update(np.ascontiguousarray(rel, dtype="<i2").tobytes())
    h.update(canonical_json(meta_encoded).encode("utf-8"))
    return h.hexdigest()


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class GraphExport(object):
    """Read-only view of one export directory, indexed by graph_idx."""

    def __init__(self, directory):
        self.dir = Path(directory)
        a = {name: np.load(str(self.dir / (name + ".npy")), allow_pickle=False) for name in ARRAYS}
        self.keys, self.num_nodes, self.edge_ptr = a["keys"], a["num_nodes"], a["edge_ptr"]
        self.src, self.dst, self.rel = a["src"], a["dst"], a["rel"]
        self.relations = json.load(open(str(self.dir / "relations.json")))
        self._pos = {int(k): i for i, k in enumerate(self.keys.tolist())}
        self._meta = None

    def __len__(self):
        return len(self.keys)

    def __contains__(self, graph_idx):
        return int(graph_idx) in self._pos

    def meta_lines(self):
        if self._meta is None:
            with open(str(self.dir / "meta.jsonl"), encoding="utf-8") as f:
                self._meta = [json.loads(line) for line in f]
        return self._meta

    def structure(self, graph_idx):
        """(num_nodes, src, dst, rel_ids) of one graph, as numpy arrays."""
        i = self._pos[int(graph_idx)]
        lo, hi = int(self.edge_ptr[i]), int(self.edge_ptr[i + 1])
        return int(self.num_nodes[i]), self.src[lo:hi], self.dst[lo:hi], self.rel[lo:hi]

    def entry(self, graph_idx):
        """Everything the legacy entry held except the DGLGraph itself: the original
        'edges' list of (src, dst, relation name) and every metadata field."""
        i = self._pos[int(graph_idx)]
        line = self.meta_lines()[i]
        n, src, dst, rel = self.structure(graph_idx)
        out = {k: decode_value(v) for k, v in line.items() if k not in ("graph_idx", "sha256")}
        out["edges"] = [(int(s), int(d), self.relations[int(r)]) for s, d, r in zip(src, dst, rel)]
        out["num_nodes"] = n
        return out
