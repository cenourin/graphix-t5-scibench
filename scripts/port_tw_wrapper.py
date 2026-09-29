#!/usr/bin/env python
"""Test TW (PORTABILIDADE.md, step A6.2b): the model wrappers (RGAT and plain arms) build the
same graph batch and return the same loss as the legacy wrappers. Python 3.7 compatible.

Legacy: seq2seq.models.graphix.rgat.Model / plain.Model over the pickled/shelved
graph_pedia. Modern: graphix_modern.models.RGATModel / PlainModel over the A0 GraphStore.
Weights: RGAT arm, T4's legacy state_dict (strict); plain arm, the stock t5-base in both.
Inputs: the T5 fixture examples, as the Trainer would pass them (input_ids, attention_mask,
labels, graph_idx). eval() mode, GPU (the legacy RGAT cannot run on CPU).
Gates: identical graph_batch (graph src/dst/num_nodes, edge relation ids) for the RGAT arm;
|loss difference| <= 1e-5 for both arms.
  run ENV  |  compare
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
OUT = Path("data_all_in/data/port_tests/TW")
T4 = Path("data_all_in/data/port_tests/T4/legacy_state_dict.npz")
T5 = Path("data_all_in/data/port_tests/T5")
FIXTURE = Path("tests/port/fixture.json")
LEGACY_PEDIA = {"spider_total": "data_all_in/data/output/graph_pedia_total.bin",
                "sciencebenchmark_train": "data_all_in/data/sciencebenchmark/output/graph_pedia_train.bin",
                "sciencebenchmark_dev": "data_all_in/data/sciencebenchmark/output/graph_pedia_dev.bin"}
TOL_LOSS = 1e-5


class _MA(object):
    def __init__(self, path):
        self.model_name_or_path, self.cache_dir = path, "/tmp/hf_cache"
        self.model_revision, self.use_auth_token = "main", False


def run(env):
    import torch
    from transformers import AutoConfig
    from port_t1_tokenizer import load_tokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = load_tokenizer()
    fx = json.load(open(str(FIXTURE)))["examples"]
    meta = json.load(open(str(T5 / "inputs.json")))
    inp = np.load(str(T5 / "inputs.npz"))
    path = "data_all_in/t5-base" if env == "legacy" else "data_all_in/t5-base-st"
    stores = {}
    if env == "legacy":
        import pickle, shelve, os
        def store(name):
            p = LEGACY_PEDIA[name]
            if name not in stores:
                stores[name] = shelve.open(p, flag="r") if os.path.exists(p + ".dat") else pickle.load(open(p, "rb"))
            return stores[name]
    else:
        from graphix_modern.data import GraphStore
        def store(name):
            if name not in stores:
                stores[name] = GraphStore(Path("data_all_in/data/graph_export") / name)
            return stores[name]
    out = {"env": env, "torch": torch.__version__, "arms": {}}
    arrays = {}
    for arm in ("rgat", "plain"):
        cfg = AutoConfig.from_pretrained(path)
        if env == "legacy":
            if arm == "rgat":
                from seq2seq.models.graphix.rgat import Model
            else:
                from seq2seq.models.graphix.plain import Model
            model = Model(tok, lambda c: c, _MA(path), cfg, None, None)
        else:
            from graphix_modern.models import PlainModel, RGATModel
            model = (RGATModel if arm == "rgat" else PlainModel)(tok, _MA(path), cfg, None, None)
        if arm == "rgat":
            w = np.load(str(T4))
            res = model.pretrain_model.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files}, strict=True)
            assert not res.missing_keys and not res.unexpected_keys
        model = model.cuda().eval()
        losses = {}
        for ex, m in zip(fx, meta):
            if not m["usable"]:
                continue
            i = m["i"]
            s = store(ex["export"])
            model.graph_pedia = model.graph_pedia_eval = s
            batch = {"input_ids": torch.from_numpy(inp["input_ids/%d" % i]).unsqueeze(0).cuda(),
                     "labels": torch.from_numpy(inp["labels/%d" % i]).unsqueeze(0).cuda(),
                     "graph_idx": torch.tensor([ex["graph_idx"]]).cuda()}
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
            if arm == "rgat":
                gb = model.graph_factory({"graph_idx": batch["graph_idx"]})
                g, e = gb[0]["graph"], gb[0]["edges"]
                src, dst = g.edges(order="eid")
                arrays["%d/src" % i] = src.cpu().numpy().astype(np.int64)
                arrays["%d/dst" % i] = dst.cpu().numpy().astype(np.int64)
                arrays["%d/edges" % i] = e.cpu().numpy().astype(np.int64)
                arrays["%d/num_nodes" % i] = np.array([g.number_of_nodes() if env == "legacy" else g.num_nodes()])
            with torch.no_grad():
                losses[str(i)] = float(model(**batch)["loss"])
            print(env, arm, i, ex["reason"], "loss %.6f" % losses[str(i)], flush=True)
        out["arms"][arm] = losses
        del model
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(str(OUT / ("%s_graph_batch.npz" % env)), **arrays)
    json.dump(out, open(str(OUT / ("%s.json" % env)), "w"), indent=1)


def compare():
    L, M = json.load(open(str(OUT / "legacy.json"))), json.load(open(str(OUT / "modern.json")))
    a, b = np.load(str(OUT / "legacy_graph_batch.npz")), np.load(str(OUT / "modern_graph_batch.npz"))
    graph_ok = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    report = {"test": "TW", "graph_batch_identical": graph_ok, "graph_batch_arrays": len(a.files), "arms": {}}
    ok = graph_ok
    for arm in ("rgat", "plain"):
        diffs = {i: abs(L["arms"][arm][i] - M["arms"][arm][i]) for i in L["arms"][arm]}
        passed = set(L["arms"][arm]) == set(M["arms"][arm]) and max(diffs.values()) <= TOL_LOSS
        ok = ok and passed and len(diffs) > 0
        report["arms"][arm] = {"examples": len(diffs), "max_abs_loss_diff": max(diffs.values()), "passed": passed}
        print(arm, report["arms"][arm])
    report["passed"] = ok
    print("graph_batch identical:", graph_ok, "(%d arrays)" % len(a.files))
    json.dump(report, open(str(OUT / "TW_report.json"), "w"), indent=1)
    print("TW", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        sys.exit(0 if compare() else 1)
