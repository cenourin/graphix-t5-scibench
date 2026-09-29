#!/usr/bin/env python
"""Test T8 (PORTABILIDADE.md, step A6.3): greedy generation (the evaluation path) gives the
same SQL on both stacks. Python 3.7 compatible; GPU (GTX 1070), both images.

Generation exactly as the trainers' prediction_step does it: wrapper.generate with
max_length = val_max_target_length (512), num_beams 1, no_repeat_ngram_size 0,
synced_gpus False, use_cache True (the modern stack's new Cache path), graph_idx; config
built as the entrypoints build it; decode with skip_special_tokens and the legacy
clean_up_tokenization_spaces=True. Both arms, two weight sets:
  t4       T4's legacy state_dict (stock t5-base + legacy-initialized Graphix layers)
  trained  the entrypoint smoke's trained model (data_all_in/data/port_tests/TE/run),
           converted once to .npz so the legacy image (no safetensors) can load it
(the plain arm uses the stock t5-base in both sets' place, it has no Graphix layers).
Gate: generated token ids identical for every usable fixture example (-> identical SQL).
  prepare (modern) | run ENV | compare
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
OUT = Path("data_all_in/data/port_tests/T8")
T4 = Path("data_all_in/data/port_tests/T4/legacy_state_dict.npz")
T5 = Path("data_all_in/data/port_tests/T5")
TE = Path("data_all_in/data/port_tests/TE/run/model.safetensors")
FIXTURE = Path("tests/port/fixture.json")
LEGACY_PEDIA = {"spider_total": "data_all_in/data/output/graph_pedia_total.bin",
                "sciencebenchmark_train": "data_all_in/data/sciencebenchmark/output/graph_pedia_train.bin",
                "sciencebenchmark_dev": "data_all_in/data/sciencebenchmark/output/graph_pedia_dev.bin"}
MAX_LEN = 512


def prepare():
    """TE's safetensors -> npz of the inner T5 (keys without 'pretrain_model.'), tied keys filled."""
    from safetensors.torch import load_file
    s = load_file(str(TE))
    shared = s["pretrain_model.shared.weight"]
    out = {k[len("pretrain_model."):]: v.numpy() for k, v in s.items()}
    for k in ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"):
        out.setdefault(k, shared.numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(str(OUT / "trained_state_dict.npz"), **out)
    print("wrote", len(out), "tensors")


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

    def store(name):
        if name not in stores:
            if env == "legacy":
                import os, pickle, shelve
                p = LEGACY_PEDIA[name]
                stores[name] = shelve.open(p, flag="r") if os.path.exists(p + ".dat") else pickle.load(open(p, "rb"))
            else:
                from graphix_modern.data import GraphStore
                stores[name] = GraphStore(Path("data_all_in/data/graph_export") / name)
        return stores[name]

    out = {"env": env, "runs": {}}
    for arm, weights in (("rgat", "t4"), ("rgat", "trained"), ("plain", "stock")):
        # config as the entrypoints build it (max_target_length default 1024; the study
        # configs set no num_beams/no_repeat_ngram_size, so DataTrainingArguments defaults)
        cfg = AutoConfig.from_pretrained(path, max_length=1024, num_beams=1, num_beam_groups=1,
                                         diversity_penalty=None, no_repeat_ngram_size=None,
                                         gradient_checkpointing=False, use_cache=True)
        if env == "legacy":
            if arm == "rgat":
                from seq2seq.models.graphix.rgat import Model
            else:
                from seq2seq.models.graphix.plain import Model
            model = Model(tok, lambda c: c, _MA(path), cfg, None, None)
        else:
            from graphix_modern.models import PlainModel, RGATModel
            model = (RGATModel if arm == "rgat" else PlainModel)(tok, _MA(path), cfg, None, None)
        if weights != "stock":
            w = np.load(str(T4 if weights == "t4" else OUT / "trained_state_dict.npz"))
            res = model.pretrain_model.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files}, strict=True)
            assert not res.missing_keys and not res.unexpected_keys
        model = model.cuda().eval()
        rows = {}
        for ex, m in zip(fx, meta):
            if not m["usable"]:
                continue
            i = m["i"]
            model.graph_pedia = model.graph_pedia_eval = store(ex["export"])
            ids = torch.from_numpy(inp["input_ids/%d" % i]).unsqueeze(0).cuda()
            with torch.no_grad():
                gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_length=MAX_LEN, num_beams=1,
                                     synced_gpus=False, no_repeat_ngram_size=0,
                                     graph_idx=torch.tensor([ex["graph_idx"]]).cuda())
            g = gen[0].tolist()
            rows[str(i)] = {"ids": g, "text": tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)}
        key = "%s/%s" % (arm, weights)
        out["runs"][key] = rows
        print(env, key, "generated", len(rows), "| e.g.", list(rows.values())[0]["text"][:80], flush=True)
        del model
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(str(OUT / ("%s.json" % env)), "w"))


def compare():
    a, b = json.load(open(str(OUT / "legacy.json"))), json.load(open(str(OUT / "modern.json")))
    report, ok = {"test": "T8", "runs": {}}, True
    for key in a["runs"]:
        x, y = a["runs"][key], b["runs"].get(key, {})
        diff = [i for i in x if x[i]["ids"] != y.get(i, {}).get("ids")]
        text_diff = [i for i in x if x[i]["text"] != y.get(i, {}).get("text")]
        lens = [len(x[i]["ids"]) for i in x]
        r = {"examples": len(x), "ids_differ": diff, "text_differ": text_diff,
             "generated_len_min_max": [min(lens), max(lens)], "passed": not diff and len(x) > 0}
        if diff:
            i = diff[0]
            first = next(k for k, (p, q) in enumerate(zip(x[i]["ids"] + [None], y[i]["ids"] + [None])) if p != q)
            r["first_divergence"] = {"example": i, "position": first,
                                     "legacy": x[i]["text"][:160], "modern": y[i]["text"][:160]}
        ok = ok and r["passed"]
        report["runs"][key] = r
        print(key, json.dumps(r)[:400])
    report["passed"] = ok
    json.dump(report, open(str(OUT / "T8_report.json"), "w"), indent=1)
    print("T8", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare()
    elif cmd == "run":
        run(sys.argv[2])
    else:
        sys.exit(0 if compare() else 1)
