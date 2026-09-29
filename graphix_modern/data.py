"""Data stage of training/evaluation on the modern stack (step A6.2a).

Ports the data part of seq2seq/run_seq2seq_train.py:
  - graph_pedia: the legacy pickled/shelved DGLGraph stores are replaced by the A0 export
    (seq2seq/utils/graph_export.py). GraphStore serves entries with the fields legacy code
    reads ('graph', 'new_struct_in', ...), the graph rebuilt with DGL 2.4 from src/dst in
    the original edge order (proven identical by T2). Relations come as ids ('rel_ids'),
    which the export stores, instead of the legacy (src, dst, name) list that was mapped
    name -> id on every training step: same values, checked by the A6.2b test.
  - the two example filters (graph size; token count == node count), unchanged;
  - eval-example alignment after filtering (the 2026-09-26 fix), unchanged;
  - the legacy HF-datasets loader (seq2seq/utils/dataset_loader.py) reused as is, with
    trust_remote_code=True, which datasets 2.21 requires for its local loading scripts
    and metric scripts (they are this repository's own code).
"""
import functools
import json
import logging
import os
import re
from pathlib import Path

import numpy as np

from seq2seq.utils.graph_export import GraphExport

logger = logging.getLogger(__name__)
EXPORT_ROOT = Path("data_all_in/data/graph_export")


class GraphStore(object):
    """graph_pedia replacement over an A0 export, keyed by graph_idx (int or str)."""

    def __init__(self, directory):
        import dgl
        import torch
        self._dgl, self._torch = dgl, torch
        self.export = GraphExport(directory)
        self._cache = {}

    def __len__(self):
        return len(self.export)

    def keys(self):
        return [int(k) for k in self.export.keys.tolist()]

    def __getitem__(self, graph_idx):
        k = int(graph_idx)
        entry = self._cache.get(k)
        if entry is None:
            n, src, dst, rel = self.export.structure(k)
            i = self.export._pos[k]
            meta = {f: v for f, v in self.export.meta_lines()[i].items() if f not in ("graph_idx", "sha256")}
            t = self._torch
            entry = {
                "graph": self._dgl.graph((t.from_numpy(src.astype(np.int64)), t.from_numpy(dst.astype(np.int64))),
                                         num_nodes=n, idtype=t.int32),
                "rel_ids": rel.astype(np.int64),
                "new_struct_in": meta["new_struct_in"],
            }
            self._cache[k] = entry
        return entry


def resolve_export(path):
    """Accept an export directory, or a legacy graph_pedia_*.bin path (the values the
    study's GRAPHIX_*_GRAPH_PEDIA_PATH variables hold), mapped through the A0 manifest."""
    p = Path(path)
    if (p / "keys.npy").exists():
        return p
    manifest = json.load(open(EXPORT_ROOT / "manifest.json"))
    for name, info in manifest["exports"].items():
        if os.path.normpath(info["source"]["path"]) == os.path.normpath(str(p)):
            return EXPORT_ROOT / name
    raise FileNotFoundError("no A0 export for graph_pedia %s (see scripts/export_graph_pedia.py)" % path)


def filter_by_graph_size(dataset, store, max_nodes):
    kept = [item for item in dataset if store[item["graph_idx"]]["graph"].num_nodes() <= max_nodes]
    dropped = len(dataset) - len(kept)
    if dropped:
        logger.warning("Dropped %d/%d examples with RGAT graph > %d nodes (VRAM accommodation)",
                       dropped, len(dataset), max_nodes)
    return kept


def filter_by_token_node_match(dataset, store, tokenizer, max_source_length):
    def _match(raw_item):
        question_in = " ".join(raw_item["raw_question_toks"])
        struct_in_norm = re.sub("  +", " ", store[raw_item["graph_idx"]]["new_struct_in"])
        seq_in = "{} ; {}".format(question_in, struct_in_norm)
        tokenized = tokenizer(seq_in, max_length=max_source_length, truncation=True)
        n_tokens = len([a for a in tokenized.input_ids if a > 1])
        return n_tokens == store[raw_item["graph_idx"]]["graph"].num_nodes()

    matches = [_match(item) for item in dataset]
    kept = [item for item, ok in zip(dataset, matches) if ok]
    dropped = len(dataset) - len(kept)
    if dropped:
        from collections import Counter
        dropped_by_db = Counter(item.get("db_id") for item, ok in zip(dataset, matches) if not ok)
        logger.warning("Dropped %d/%d examples with a token/graph-node count mismatch (by db_id: %s)",
                       dropped, len(dataset), dict(dropped_by_db))
    return kept


def tag_eval_positions(dataset):
    for pos, item in enumerate(dataset):
        item["_eval_pos"] = pos
    return dataset


def align_eval_examples(eval_split, seq2seq_eval_dataset, max_val_samples, scored_eval):
    """Same rule as run_seq2seq_train.py: the metric pairs the i-th prediction with the i-th
    HF-loaded example, so drop from those examples the positions the filters dropped."""
    if scored_eval and max_val_samples is not None:
        seq2seq_eval_dataset = [it for it in seq2seq_eval_dataset if it["_eval_pos"] < max_val_samples]
    if scored_eval and len(eval_split.examples) != len(seq2seq_eval_dataset):
        kept = [item["_eval_pos"] for item in seq2seq_eval_dataset]
        assert len(eval_split.examples) > max(kept), "eval dataset file and HF dev split differ"
        eval_split.examples = eval_split.examples.select(kept)
        logger.warning("Aligned eval examples to the %d filtered eval items", len(kept))
    return seq2seq_eval_dataset


def load_dataset_splits(data_args, model_args, data_training_args, training_args, tokenizer):
    """seq2seq.utils.dataset_loader.load_dataset, unchanged, with trust_remote_code=True
    passed to datasets' load_dataset/load_metric (required by datasets 2.21 for the
    repository's own loading and metric scripts)."""
    import datasets.load
    from seq2seq.utils import dataset_loader
    orig_ds, orig_metric = datasets.load.load_dataset, datasets.load.load_metric
    datasets.load.load_dataset = functools.partial(orig_ds, trust_remote_code=True)
    datasets.load.load_metric = functools.partial(orig_metric, trust_remote_code=True)
    try:
        return dataset_loader.load_dataset(data_args=data_args, model_args=model_args,
                                           data_training_args=data_training_args,
                                           training_args=training_args, tokenizer=tokenizer)
    finally:
        datasets.load.load_dataset, datasets.load.load_metric = orig_ds, orig_metric
