#!/usr/bin/env python
"""Test TD (PORTABILIDADE.md, step A6.2a): the data stage of training/evaluation selects and
prepares exactly the same examples on the modern stack. Python 3.7 compatible.

Legacy side: the real legacy code (seq2seq/run_seq2seq_train.py filters on the pickled
graph_pedia, seq2seq/utils/dataset_loader.load_dataset). Modern side: graphix_modern.data
(GraphStore over the A0 export, same filters, same loader with trust_remote_code).
For each benchmark (study config), compares:
  F  positions kept by the two filters (graph size, token/node count) for the train and
     validation slices (splits/) and the official dev;
  E  every HF dev example (all fields) and the aligned eval examples after filtering;
  P  the HF preprocessing of the dev split (input_ids/labels, including the serialized
     schema with database content matched through rapidfuzz);
  S  schemas; and the size of the HF train split.
  dump ENV BENCH  |  compare
"""
import hashlib
import io
import contextlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
OUT = Path("data_all_in/data/port_tests/TD")
SPIDER, SCIB, SPLITS = "data_all_in/data/output", "data_all_in/data/sciencebenchmark/output", "data_all_in/data/splits"
BENCH = {
    "spider": {"config": "configs/study_t5base_spider.json", "max_nodes": 512, "roles": {
        "train": (SPLITS + "/spider_train.json", SPIDER + "/graph_pedia_total.bin"),
        "val": (SPLITS + "/spider_val.json", SPIDER + "/graph_pedia_total.bin"),
        "dev": (SPIDER + "/seq2seq_dev_dataset.json", SPIDER + "/graph_pedia_total.bin")}},
    "sciencebenchmark": {"config": "configs/study_t5base_sciencebenchmark.json", "max_nodes": 1400, "roles": {
        "train": (SPLITS + "/sciencebenchmark_train.json", SCIB + "/graph_pedia_train.bin"),
        "val": (SPLITS + "/sciencebenchmark_val.json", SCIB + "/graph_pedia_train.bin"),
        "dev": (SCIB + "/seq2seq_dev_dataset.json", SCIB + "/graph_pedia_dev.bin")}},
}


def h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def build_args(cfg_path):
    from transformers import Seq2SeqTrainingArguments
    from seq2seq.utils.args import ModelArguments
    from seq2seq.utils.dataset import DataArguments, DataTrainingArguments
    from dataclasses import fields
    cfg = json.load(open(cfg_path))

    def make(cls, extra=None):
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in cfg.items() if k in names}
        kw.update(extra or {})
        return cls(**kw)

    model_args = make(ModelArguments, {"cache_dir": "/tmp/hf_cache"})  # fresh cache per run
    data_args = make(DataArguments)
    dta = make(DataTrainingArguments)
    targs = Seq2SeqTrainingArguments(output_dir="/tmp/td_out", do_train=True, do_eval=True, report_to=[])
    return model_args, data_args, dta, targs


def dump(env, bench):
    from port_t1_tokenizer import load_tokenizer
    spec = BENCH[bench]
    model_args, data_args, dta, targs = build_args(spec["config"])
    tok = load_tokenizer()
    out = {"env": env, "bench": bench, "filters": {}}
    if env == "legacy":
        # The legacy module loads its graph_pedia stores at import. Point its "train" store at
        # this bench's dev store (reused below, no second copy) and its "eval" store at the
        # smallest file, so a 1.4 GB pickled store is held in memory once, not three times.
        os.environ.setdefault("GRAPHIX_TRAIN_GRAPH_PEDIA_PATH", spec["roles"]["dev"][1])
        os.environ.setdefault("GRAPHIX_EVAL_GRAPH_PEDIA_PATH", SCIB + "/graph_pedia_dev.bin")
        os.environ.setdefault("GRAPHIX_TRAIN_DATASET_PATH", spec["roles"]["dev"][0])
        os.environ.setdefault("GRAPHIX_EVAL_DATASET_PATH", SCIB + "/seq2seq_dev_dataset.json")  # matches its eval store
        sys.path.insert(0, "seq2seq")
        with contextlib.redirect_stdout(io.StringIO()):
            import run_seq2seq_train as legacy_train  # module-level load is harmless here
        stores = {os.environ["GRAPHIX_TRAIN_GRAPH_PEDIA_PATH"]: legacy_train.graph_pedia_train}

        def load_store(p):
            if p not in stores:
                stores[p] = legacy_train._load_graph_pedia(p)
            return stores[p]
        size_filter = lambda d, s: legacy_train._filter_by_graph_size(d, s, spec["max_nodes"])
        token_filter = lambda d, s: legacy_train._filter_by_token_node_match(d, s, tok, dta.max_source_length)
        from seq2seq.utils.dataset_loader import load_dataset
        load = lambda: load_dataset(data_args=data_args, model_args=model_args, data_training_args=dta,
                                    training_args=targs, tokenizer=tok)
    else:
        from graphix_modern import data as D
        stores = {}
        load_store = lambda p: stores.setdefault(p, D.GraphStore(D.resolve_export(p)))
        size_filter = lambda d, s: D.filter_by_graph_size(d, s, spec["max_nodes"])
        token_filter = lambda d, s: D.filter_by_token_node_match(d, s, tok, dta.max_source_length)
        load = lambda: D.load_dataset_splits(data_args, model_args, dta, targs, tok)

    dev_kept = None
    for role, (ds_path, gp_path) in spec["roles"].items():
        data = json.load(open(ds_path))
        for pos, item in enumerate(data):
            item["_pos"] = pos
        store = load_store(gp_path)
        after_size = size_filter(data, store)
        kept = token_filter(after_size, store)
        out["filters"][role] = {"n": len(data), "after_size": len(after_size), "kept": [it["_pos"] for it in kept]}
        if role == "dev":
            dev_kept = [it["_pos"] for it in kept]
        print(env, bench, role, len(data), "->", len(after_size), "->", len(kept), flush=True)

    metric, splits = load()
    ex = splits.eval_split.examples
    out["eval_examples"] = [h(ex[i]) for i in range(len(ex))]
    out["eval_examples_aligned"] = [h(ex[i]) for i in dev_kept] if len(ex) > max(dev_kept) else None
    ev = splits.eval_split.dataset
    out["eval_preprocessed"] = [h({"input_ids": ev[i]["input_ids"], "labels": ev[i]["labels"]}) for i in range(len(ev))]
    out["schemas"] = {k: h(v) for k, v in sorted(splits.schemas.items())}
    out["train_split_len"] = len(splits.train_split.dataset) if splits.train_split is not None else None
    out["metric"] = type(metric).__name__
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(str(OUT / ("%s_%s.json" % (env, bench))), "w"))
    print(env, bench, "eval examples", len(ex), "preprocessed", len(ev), "schemas", len(out["schemas"]),
          "train split", out["train_split_len"], flush=True)


def compare():
    report, ok = {"test": "TD", "benches": {}}, True
    for bench in BENCH:
        a = json.load(open(str(OUT / ("legacy_%s.json" % bench))))
        b = json.load(open(str(OUT / ("modern_%s.json" % bench))))
        r = {"filters": {role: a["filters"][role] == b["filters"][role] for role in a["filters"]}}
        for key in ("eval_examples", "eval_examples_aligned", "eval_preprocessed"):
            x, y = a[key], b[key]
            r[key] = {"n": [len(x) if x else 0, len(y) if y else 0],
                      "mismatches": None if x is None or y is None or len(x) != len(y) else
                      [i for i, (p, q) in enumerate(zip(x, y)) if p != q][:20]}
            r[key]["equal"] = x == y
        r["schemas_equal"] = a["schemas"] == b["schemas"]
        r["train_split_len"] = [a["train_split_len"], b["train_split_len"]]
        r["counts"] = {role: [a["filters"][role]["n"], a["filters"][role]["after_size"], len(a["filters"][role]["kept"])]
                       for role in a["filters"]}
        r["passed"] = (all(r["filters"].values()) and r["eval_examples"]["equal"] and r["eval_examples_aligned"]["equal"]
                       and r["eval_preprocessed"]["equal"] and r["schemas_equal"]
                       and a["train_split_len"] == b["train_split_len"])
        ok = ok and r["passed"]
        report["benches"][bench] = r
        print(bench, json.dumps(r))
    report["passed"] = ok
    json.dump(report, open(str(OUT / "TD_report.json"), "w"), indent=1)
    print("TD", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3])
    else:
        sys.exit(0 if compare() else 1)
