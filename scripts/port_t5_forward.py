#!/usr/bin/env python
"""Test T5 (PORTABILIDADE.md, step A4): the ported Graphix-T5 training forward equals the
legacy one. Python 3.7 compatible; same script in both images, same GPU (GTX 1070, no TF32).

Same weights (T4's legacy state_dict, loaded with strict=True in both), same token ids and
graphs, eval() (dropout off), use_cache=False (cache/generation are T8's), fp32 strict.

  inputs        (legacy)  input_ids/labels of the fixture examples, tokenized as in training
                          (same code as T1); examples the training filters would drop (graph
                          node count != non-special token count) are listed and skipped.
  run ENV                 forward on cuda with hooks at: shared embedding (encoder call),
                          every encoder block, every block's RGAT, encoder final, every
                          decoder block, lm_head; plus the loss.
  compare                 per example and checkpoint: shape, dtype, max abs, max relative diff
                          (max|a-b| / max|a|); first checkpoint over tolerance localizes a
                          divergence. Gate: logits relative <= 1e-4 and |loss diff| <= 1e-5.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
OUT = Path("data_all_in/data/port_tests/T5")
T4 = Path("data_all_in/data/port_tests/T4/legacy_state_dict.npz")
FIXTURE = Path("tests/port/fixture.json")
EXPORT = Path("data_all_in/data/graph_export")
TOL_LOGITS_REL, TOL_LOSS = 1e-4, 1e-5
CONFIGS = {"spider": "configs/study_t5base_spider.json", "sciencebenchmark": "configs/study_t5base_sciencebenchmark.json"}


def fixture():
    return json.load(open(str(FIXTURE)))["examples"]


def make_inputs():
    import contextlib
    import io
    from port_t1_tokenizer import _Args, _ExportPedia, load_tokenizer
    from seq2seq.utils.dataset_graph import TokenizedDataset
    from seq2seq.utils.graph_export import GraphExport
    tok = load_tokenizer()
    arrays, meta, exports, data_cache = {}, [], {}, {}
    for i, ex in enumerate(fixture()):
        if ex["export"] not in exports:
            exports[ex["export"]] = GraphExport(EXPORT / ex["export"])
        if ex["dataset"] not in data_cache:
            data_cache[ex["dataset"]] = json.load(open(ex["dataset"]))
        cfg = json.load(open(CONFIGS[ex["bench"]]))
        td = TokenizedDataset(_Args(cfg["max_source_length"], cfg.get("max_target_length", 1024)), None, tok,
                              data_cache[ex["dataset"]], _ExportPedia(exports[ex["export"]]))
        with contextlib.redirect_stdout(io.StringIO()):
            item = td[ex["position"]]
        n_tokens = len([t for t in item["input_ids"] if t > 1])
        usable = n_tokens == ex["num_nodes"]
        meta.append({"i": i, "reason": ex["reason"], "usable": usable, "n_tokens": n_tokens,
                     "num_nodes": ex["num_nodes"], "input_len": len(item["input_ids"]), "label_len": len(item["labels"])})
        if usable:
            arrays["input_ids/%d" % i] = np.array(item["input_ids"], dtype=np.int64)
            arrays["labels/%d" % i] = np.array(item["labels"], dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(str(OUT / "inputs.npz"), **arrays)
    json.dump(meta, open(str(OUT / "inputs.json"), "w"), indent=1)
    for m in meta:
        print(m)


def run(env):
    import dgl
    import torch
    from transformers import AutoConfig
    from seq2seq.utils.graph_export import GraphExport
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if env == "legacy":
        from seq2seq.models.modeling_t5 import T5ForConditionalGeneration
        path = "data_all_in/t5-base"
    else:
        from graphix_modern.modeling_t5 import T5ForConditionalGeneration
        path = "data_all_in/t5-base-st"
    cfg = AutoConfig.from_pretrained(path)
    model = T5ForConditionalGeneration(cfg)
    w = np.load(str(T4))
    res = model.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files}, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    model = model.cuda().eval()

    captured = {}

    def hook(name):
        def f(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            t = getattr(t, "last_hidden_state", t)
            if name not in captured:  # the shared embedding is called by encoder first, then decoder
                captured[name] = t.detach().float().cpu().numpy()
        return f

    points = [("embed_encoder", model.shared)]
    for b, block in enumerate(model.encoder.block):
        points += [("enc_rgat_%02d" % b, block.rgat_layer), ("enc_block_%02d" % b, block)]
    points.append(("enc_final_norm", model.encoder.final_layer_norm))
    for b, block in enumerate(model.decoder.block):
        points.append(("dec_block_%02d" % b, block))
    points += [("dec_final_norm", model.decoder.final_layer_norm), ("lm_head", model.lm_head)]
    handles = [m.register_forward_hook(hook(n)) for n, m in points]

    inp = np.load(str(OUT / "inputs.npz"))
    meta = json.load(open(str(OUT / "inputs.json")))
    exports, losses = {}, {}
    for ex, m in zip(fixture(), meta):
        if not m["usable"]:
            continue
        i = m["i"]
        if ex["export"] not in exports:
            exports[ex["export"]] = GraphExport(EXPORT / ex["export"])
        n, src, dst, rel = exports[ex["export"]].structure(ex["graph_idx"])
        g = dgl.graph((torch.from_numpy(src.astype(np.int64)), torch.from_numpy(dst.astype(np.int64))),
                      num_nodes=n, idtype=torch.int32)
        graph_batch = [{"graph": g, "edges": torch.from_numpy(rel.astype(np.int64)).cuda()}]
        input_ids = torch.from_numpy(inp["input_ids/%d" % i]).unsqueeze(0).cuda()
        labels = torch.from_numpy(inp["labels/%d" % i]).unsqueeze(0).cuda()
        captured.clear()
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels,
                        use_cache=False, graph_batch=graph_batch)
        losses[str(i)] = float(out.loss)
        arrays = dict(captured)
        arrays["logits"] = out.logits.detach().float().cpu().numpy()
        np.savez(str(OUT / ("%s_%02d.npz" % (env, i))), **arrays)
        print(env, i, ex["reason"], "loss %.6f" % losses[str(i)], flush=True)
    for h in handles:
        h.remove()
    json.dump({"env": env, "torch": torch.__version__, "dgl": dgl.__version__,
               "gpu": torch.cuda.get_device_name(0), "losses": losses},
              open(str(OUT / ("%s.json" % env)), "w"), indent=1)


def order_key(name):
    return (0 if name.startswith("embed") else 1 if name.startswith("enc") else 2 if name.startswith("dec") else 3, name)


def compare():
    L, M = json.load(open(str(OUT / "legacy.json"))), json.load(open(str(OUT / "modern.json")))
    meta = json.load(open(str(OUT / "inputs.json")))
    report = {"test": "T5", "tolerances": {"logits_rel": TOL_LOGITS_REL, "loss_abs": TOL_LOSS},
              "legacy": {k: L[k] for k in ("torch", "dgl", "gpu")}, "modern": {k: M[k] for k in ("torch", "dgl", "gpu")},
              "skipped": [m for m in meta if not m["usable"]], "examples": []}
    ok = True
    for m in meta:
        if not m["usable"]:
            continue
        i = m["i"]
        a, b = np.load(str(OUT / ("legacy_%02d.npz" % i))), np.load(str(OUT / ("modern_%02d.npz" % i)))
        names = sorted(set(a.files) | set(b.files), key=order_key)
        rows, first = {}, None
        for nm in names:
            if nm not in a.files or nm not in b.files:
                rows[nm] = {"present_in_both": False}
                first = first or nm
                continue
            x, y = a[nm], b[nm]
            same = x.shape == y.shape and x.dtype == y.dtype
            d = np.abs(x.astype(np.float64) - y.astype(np.float64)) if same else None
            rows[nm] = {"shape": list(x.shape), "shape_dtype_equal": same,
                        "max_abs": float(d.max()) if same else None,
                        "max_rel": float(d.max() / max(np.abs(x).max(), 1e-30)) if same else None}
            if first is None and (not same or rows[nm]["max_rel"] > TOL_LOGITS_REL):
                first = nm
        dloss = abs(L["losses"][str(i)] - M["losses"][str(i)])
        passed = rows["logits"].get("shape_dtype_equal", False) and rows["logits"]["max_rel"] <= TOL_LOGITS_REL and dloss <= TOL_LOSS
        ok = ok and passed
        report["examples"].append({"i": i, "reason": m["reason"], "loss_legacy": L["losses"][str(i)],
                                   "loss_modern": M["losses"][str(i)], "loss_abs_diff": dloss,
                                   "first_checkpoint_rel_over_1e-4": first, "passed": passed, "checkpoints": rows})
        print("%2d %-48s loss %.6f/%.6f |d|=%.1e logits rel=%.1e %s%s" % (
            i, m["reason"], L["losses"][str(i)], M["losses"][str(i)], dloss, rows["logits"]["max_rel"],
            "PASS" if passed else "FAIL", "" if first is None else "  (first > 1e-4: %s)" % first))
    report["examples_compared"] = len(report["examples"])
    report["passed"] = ok and report["examples_compared"] > 0
    json.dump(report, open(str(OUT / "T5_report.json"), "w"), indent=1)
    print("skipped (dropped by the training filters):", [(s["i"], s["reason"]) for s in report["skipped"]])
    print("T5", "PASSED" if report["passed"] else "FAILED")
    return report["passed"]


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inputs":
        make_inputs()
    elif cmd == "run":
        run(sys.argv[2])
    else:
        sys.exit(0 if compare() else 1)
