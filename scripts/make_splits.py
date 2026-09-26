#!/usr/bin/env python3
"""Carve a validation split out of each benchmark's (already preprocessed) train set.

Protocol (see RISCOS.md 1.1-1.3): hyperparameter search selects on this validation split;
the official dev set is used only for final reporting.

- spider: database-disjoint. Whole train_spider DBs are held out (train_others DBs always
  stay in train), mirroring Spider's cross-domain dev/test setup.
- sciencebenchmark: train and dev share the same 3 DBs, so splitting by DB is impossible.
  Instead split by SQL *skeleton* (literals masked) within each DB: synth contains many
  examples per skeleton, and a random split would put near-duplicates on both sides.

Output: splits/<bench>.json (indices into the preprocessed train file; versioned) and
data_all_in/data/splits/<bench>_{train,val}.json (sliced datasets; regenerable, gitignored).
Examples keep their original graph_idx, so validation runs must point
GRAPHIX_EVAL_GRAPH_PEDIA_PATH at the *train* graph_pedia file.
"""
import argparse, json, random, re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCHES = {
    "spider": REPO / "data_all_in/data/output/seq2seq_train_dataset.json",
    "sciencebenchmark": REPO / "data_all_in/data/sciencebenchmark/output/seq2seq_train_dataset.json",
}


def skeleton(sql):
    s = sql.lower()
    s = re.sub(r"'[^']*'|\"[^\"]*\"", "V", s)
    s = re.sub(r"\b\d+(\.\d+)?\b", "N", s)
    return re.sub(r"\s+", " ", s).strip()


def pick_groups(groups, target, rng):
    """Shuffle group keys and take whole groups until `target` examples are covered."""
    keys = sorted(groups)
    rng.shuffle(keys)
    chosen, n = [], 0
    for k in keys:
        if n >= target:
            break
        chosen.append(k)
        n += len(groups[k])
    return chosen


def split(bench, data, frac, seed):
    rng = random.Random(seed)
    if bench == "spider":
        spider_dbs = {ex["db_id"] for ex in json.load(open(REPO / "data_all_in/data/spider/train_spider.json"))}
        groups = defaultdict(list)
        for i, ex in enumerate(data):
            if ex["db_id"] in spider_dbs:
                groups[ex["db_id"]].append(i)
        val_keys = pick_groups(groups, round(frac * len(data)), rng)
        val = sorted(i for k in val_keys for i in groups[k])
        meta = {"unit": "db_id", "val_groups": sorted(val_keys)}
    else:
        val, val_keys = [], {}
        by_db = defaultdict(lambda: defaultdict(list))
        for i, ex in enumerate(data):
            by_db[ex["db_id"]][skeleton(ex["query"])].append(i)
        for db in sorted(by_db):
            n_db = sum(len(v) for v in by_db[db].values())
            keys = pick_groups(by_db[db], round(frac * n_db), rng)
            val_keys[db] = len(keys)
            val += [i for k in keys for i in by_db[db][k]]
        val = sorted(val)
        meta = {"unit": "sql_skeleton_within_db", "val_groups_per_db": val_keys}
    return val, meta


def check(bench, data, train_idx, val_idx):
    tr, va = [data[i] for i in train_idx], [data[i] for i in val_idx]
    assert not set(train_idx) & set(val_idx) and len(train_idx) + len(val_idx) == len(data)
    exact = {ex["query"].strip().lower() for ex in tr} & {ex["query"].strip().lower() for ex in va}
    out = {"exact_sql_overlap": len(exact)}
    if bench == "spider":
        out["db_overlap"] = len({e["db_id"] for e in tr} & {e["db_id"] for e in va})
        assert out["db_overlap"] == 0
    else:
        out["skeleton_overlap"] = len({(e["db_id"], skeleton(e["query"])) for e in tr}
                                      & {(e["db_id"], skeleton(e["query"])) for e in va})
        assert out["skeleton_overlap"] == 0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benches", nargs="+", default=list(BENCHES))
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    (REPO / "splits").mkdir(exist_ok=True)
    out_dir = REPO / "data_all_in/data/splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for bench in a.benches:
        data = json.load(open(BENCHES[bench]))
        val_idx, meta = split(bench, data, a.val_frac, a.seed)
        vset = set(val_idx)
        train_idx = [i for i in range(len(data)) if i not in vset]
        checks = check(bench, data, train_idx, val_idx)
        per_db = defaultdict(lambda: [0, 0])
        for i, ex in enumerate(data):
            per_db[ex["db_id"]][i in vset] += 1
        record = {"bench": bench, "source": str(BENCHES[bench].relative_to(REPO)), "seed": a.seed,
                  "val_frac_target": a.val_frac, "n_total": len(data), "n_train": len(train_idx),
                  "n_val": len(val_idx), **meta, "checks": checks,
                  "train_idx": train_idx, "val_idx": val_idx}
        if bench == "sciencebenchmark":
            record["per_db_train_val"] = {k: v for k, v in sorted(per_db.items())}
        json.dump(record, open(REPO / f"splits/{bench}.json", "w"), indent=1)
        for name, idx in (("train", train_idx), ("val", val_idx)):
            json.dump([data[i] for i in idx], open(out_dir / f"{bench}_{name}.json", "w"))
        print(f"{bench}: total={len(data)} train={len(train_idx)} val={len(val_idx)} "
              f"({len(val_idx) / len(data):.1%}) checks={checks}")


if __name__ == "__main__":
    main()
