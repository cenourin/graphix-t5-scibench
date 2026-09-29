#!/usr/bin/env python
"""Test T1 (PORTABILIDADE.md, step A2): the modern stack tokenizes every example exactly
like the legacy one.

Same script in both images (Python 3.7 compatible). The tokenizer is loaded as in
run_seq2seq_train.py (AutoTokenizer, fast, plus the added tokens " <=" and " <"), and
each example goes through the project's own TokenizedDataset.__getitem__ (input text,
alias normalization of the target, truncation), with the schema text read from the A0
graph export instead of the legacy DGL store.

  dump:    python scripts/port_t1_tokenizer.py dump <legacy|modern>
  compare: python scripts/port_t1_tokenizer.py compare       (any Python 3.7+)
Output in data_all_in/data/port_tests/T1/.
"""
import gzip
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, ".")
OUT = Path("data_all_in/data/port_tests/T1")
EXPORT = Path("data_all_in/data/graph_export")
SPIDER, SCIB = "data_all_in/data/output", "data_all_in/data/sciencebenchmark/output"
# dataset file -> (graph export, study config giving max_source_length / max_target_length)
DATASETS = {
    SPIDER + "/seq2seq_train_dataset.json": ("spider_total", "configs/study_t5base_spider.json"),
    SPIDER + "/seq2seq_dev_dataset.json": ("spider_total", "configs/study_t5base_spider.json"),
    SCIB + "/seq2seq_train_dataset.json": ("sciencebenchmark_train", "configs/study_t5base_sciencebenchmark.json"),
    SCIB + "/seq2seq_dev_dataset.json": ("sciencebenchmark_dev", "configs/study_t5base_sciencebenchmark.json"),
}


class _Nodes(object):
    """Stands in for the DGLGraph: TokenizedDataset only asks it for its node count."""

    def __init__(self, n):
        self.n = n

    def number_of_nodes(self):
        return self.n


class _ExportPedia(object):
    def __init__(self, export):
        self.export = export

    def __getitem__(self, graph_idx):
        i = self.export._pos[int(graph_idx)]
        return {"new_struct_in": self.export.meta_lines()[i]["new_struct_in"],
                "graph": _Nodes(int(self.export.num_nodes[i]))}


class _Args(object):
    def __init__(self, max_source_length, max_target_length):
        self.max_source_length, self.max_target_length = max_source_length, max_target_length


def load_tokenizer():
    from tokenizers import AddedToken
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("data_all_in/t5-base", use_fast=True)
    tok.add_tokens([AddedToken(" <="), AddedToken(" <")])
    return tok


def dump(env_name):
    import io
    import contextlib
    import tokenizers
    import transformers
    from seq2seq.utils.dataset_graph import TokenizedDataset
    from seq2seq.utils.graph_export import GraphExport

    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()
    info = {"env": env_name, "python": platform.python_version(),
            "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
            "tokenizer_class": type(tok).__name__, "len_tokenizer": len(tok),
            "added_tokens": {t: tok.convert_tokens_to_ids(t) for t in (" <=", " <")},
            "special_tokens": tok.special_tokens_map,
            "probe": tok("a <= b < c ; x").input_ids}
    exports = {}
    with gzip.open(str(OUT / ("%s.jsonl.gz" % env_name)), "wt", encoding="utf-8") as f:
        f.write(json.dumps({"info": info}) + "\n")
        for ds_path, (export_name, cfg_path) in DATASETS.items():
            if export_name not in exports:
                exports[export_name] = GraphExport(EXPORT / export_name)
            cfg = json.load(open(cfg_path))
            args = _Args(cfg["max_source_length"], cfg.get("max_target_length", 1024))
            data = json.load(open(ds_path))
            td = TokenizedDataset(args, None, tok, data, _ExportPedia(exports[export_name]))
            for pos in range(len(data)):
                with contextlib.redirect_stdout(io.StringIO()):  # its node-count mismatch prints
                    item = td[pos]
                f.write(json.dumps({"dataset": ds_path, "position": pos,
                                    "input_ids": item["input_ids"], "labels": item["labels"],
                                    "attention_mask": item["attention_mask"]}) + "\n")
            print(env_name, ds_path, len(data), "examples", flush=True)
    print(json.dumps(info))


def compare():
    def rows(name):
        with gzip.open(str(OUT / ("%s.jsonl.gz" % name)), "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    legacy, modern = rows("legacy"), rows("modern")
    info_l, info_m = next(legacy)["info"], next(modern)["info"]
    report = {"test": "T1", "legacy": info_l, "modern": info_m, "per_dataset": {}, "mismatches": []}
    header_ok = all(info_l[k] == info_m[k] for k in ("len_tokenizer", "added_tokens", "special_tokens", "probe"))
    n = 0
    for a, b in zip(legacy, modern):
        n += 1
        key = (a["dataset"], a["position"])
        d = report["per_dataset"].setdefault(a["dataset"], {"examples": 0, "mismatched": 0})
        d["examples"] += 1
        if (b["dataset"], b["position"]) != key:
            report["mismatches"].append({"example": key, "what": "row order"})
            d["mismatched"] += 1
            continue
        bad = [f for f in ("input_ids", "labels", "attention_mask") if a[f] != b[f]]
        if bad:
            d["mismatched"] += 1
            if len(report["mismatches"]) < 50:
                first = next(i for i, (x, y) in enumerate(zip(a[bad[0]] + [None], b[bad[0]] + [None])) if x != y)
                report["mismatches"].append({"example": key, "fields": bad, "first_diff_at": first,
                                             "legacy": a[bad[0]][max(0, first - 3):first + 3],
                                             "modern": b[bad[0]][max(0, first - 3):first + 3]})
    leftover = sum(1 for _ in legacy) + sum(1 for _ in modern)
    total_bad = sum(d["mismatched"] for d in report["per_dataset"].values())
    report.update(examples=n, header_identical=header_ok, extra_rows=leftover,
                  passed=header_ok and leftover == 0 and total_bad == 0 and n == 14642)
    json.dump(report, open(str(OUT / "T1_report.json"), "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("passed", "examples", "header_identical", "extra_rows")}))
    print(json.dumps(report["per_dataset"], indent=1))
    for m in report["mismatches"][:5]:
        print(m)
    if not header_ok:
        print({k: (info_l[k], info_m[k]) for k in ("len_tokenizer", "added_tokens", "probe") if info_l[k] != info_m[k]})
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    dump(sys.argv[2]) if sys.argv[1] == "dump" else compare()
