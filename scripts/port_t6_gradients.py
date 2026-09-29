#!/usr/bin/env python
"""Test T6 (PORTABILIDADE.md, step A5): the ported Graphix-T5 produces the same gradients as
the legacy one. Python 3.7 compatible; same script in both images, same GPU (GTX 1070).

Same weights (T4), same inputs (T5); every nn.Dropout replaced by x * 1.0 (see build()) and
the model in train() mode, so gradient checkpointing is active where requested while
nothing is random.
Gradients are accumulated over the examples of a run (as gradient accumulation does).
Runs, mirroring real use:
  R1 spider, no gradient checkpointing   (the Spider study config)
  R2 spider, gradient checkpointing      (same examples: isolates the checkpointing path)
  R3 sciencebenchmark, gradient checkpointing (the ScienceBenchmark study config)
Gates per run, legacy vs modern: cosine >= 0.9999 for every parameter tensor; identical set
of parameters that receive a gradient (and of those that do not, e.g. the unused
rgat_layer.filter and decoder.relation_emb). Also reported: max relative difference per
tensor, and within each environment R1 vs R2 (checkpointing must not change gradients).
  run ENV  |  compare
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
OUT = Path("data_all_in/data/port_tests/T6")
T4 = Path("data_all_in/data/port_tests/T4/legacy_state_dict.npz")
T5 = Path("data_all_in/data/port_tests/T5")
FIXTURE = Path("tests/port/fixture.json")
EXPORT = Path("data_all_in/data/graph_export")
RUNS = {"R1": ("spider", False), "R2": ("spider", True), "R3": ("sciencebenchmark", True)}
COS_MIN = 0.9999


def build(env):
    import torch
    from transformers import AutoConfig
    if env == "legacy":
        from seq2seq.models.modeling_t5 import T5ForConditionalGeneration
        path = "data_all_in/t5-base"
    else:
        from graphix_modern.modeling_t5 import T5ForConditionalGeneration
        path = "data_all_in/t5-base-st"
    model = T5ForConditionalGeneration(AutoConfig.from_pretrained(path))
    w = np.load(str(T4))
    res = model.load_state_dict({k: torch.from_numpy(w[k]) for k in w.files}, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    # Dropout off, but as an identity that returns a NEW tensor (x * 1.0), not x itself:
    # T5LayerRGAT.graph_caption writes the RGAT output in place into the slice the RGAT
    # just consumed. In training the RGAT's q/k/v Linears see feat_dropout's fresh output,
    # so autograd never saved that slice; with a pass-through dropout (p=0 or eval()) they
    # save the slice itself and backward fails on the in-place write, in both legacy and
    # modern alike. x * 1.0 keeps the training-time autograd structure without randomness.
    # T5Attention applies dropout functionally (nn.functional.dropout(p=self.dropout,
    # training=self.training)), not through an nn.Dropout module: set its rate to 0 too.
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.forward = lambda x: x * 1.0
        elif type(m).__name__ == "T5Attention":
            m.dropout = 0.0
    return model.cuda().train()


def run(env):
    import dgl
    import torch
    from seq2seq.utils.graph_export import GraphExport
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    fx = json.load(open(str(FIXTURE)))["examples"]
    meta = json.load(open(str(T5 / "inputs.json")))
    inp = np.load(str(T5 / "inputs.npz"))
    exports, info = {}, {"env": env, "torch": torch.__version__, "dgl": dgl.__version__,
                         "gpu": torch.cuda.get_device_name(0), "runs": {}}
    for run_name, (bench, ckpt) in RUNS.items():
        model = build(env)
        if ckpt:
            if env == "legacy":
                model.gradient_checkpointing_enable()  # transformers 4.17: reentrant torch checkpoint
            else:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.zero_grad()
        losses = {}
        for ex, m in zip(fx, meta):
            if not m["usable"] or ex["bench"] != bench:
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
            out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels,
                        use_cache=False, graph_batch=graph_batch)
            out.loss.backward()
            losses[str(i)] = float(out.loss)
        grads = {n: p.grad.detach().cpu().numpy() for n, p in model.named_parameters() if p.grad is not None}
        no_grad = [n for n, p in model.named_parameters() if p.grad is None]
        np.savez(str(OUT / ("%s_%s.npz" % (env, run_name))), **grads)
        info["runs"][run_name] = {"bench": bench, "checkpointing": ckpt, "losses": losses, "no_grad": no_grad,
                                  "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        print(env, run_name, bench, "ckpt" if ckpt else "no-ckpt", "examples", len(losses),
              "params with grad", len(grads), "without", len(no_grad), flush=True)
        del model, grads
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    json.dump(info, open(str(OUT / ("%s.json" % env)), "w"), indent=1)


def tensor_cmp(x, y):
    x64, y64 = x.astype(np.float64).ravel(), y.astype(np.float64).ravel()
    nx, ny = np.linalg.norm(x64), np.linalg.norm(y64)
    cos = 1.0 if nx == 0 and ny == 0 else (float(x64 @ y64 / (nx * ny)) if nx and ny else 0.0)
    d = np.abs(x64 - y64).max()
    return cos, float(d / max(np.abs(x64).max(), 1e-30))


def compare_pair(a_path, b_path):
    a, b = np.load(str(a_path)), np.load(str(b_path))
    rows = {}
    for k in sorted(set(a.files) | set(b.files)):
        if k not in a.files or k not in b.files:
            rows[k] = {"in_both": False}
            continue
        if a[k].shape != b[k].shape:
            rows[k] = {"in_both": True, "shape_equal": False}
            continue
        cos, rel = tensor_cmp(a[k], b[k])
        rows[k] = {"in_both": True, "shape_equal": True, "cosine": cos, "max_rel": rel}
    return rows


def compare():
    L, M = json.load(open(str(OUT / "legacy.json"))), json.load(open(str(OUT / "modern.json")))
    report = {"test": "T6", "cosine_min": COS_MIN, "legacy": {k: L[k] for k in ("torch", "dgl", "gpu")},
              "modern": {k: M[k] for k in ("torch", "dgl", "gpu")}, "runs": {}}
    ok = True
    for r in RUNS:
        rows = compare_pair(OUT / ("legacy_%s.npz" % r), OUT / ("modern_%s.npz" % r))
        bad = {k: v for k, v in rows.items() if not v.get("in_both") or not v.get("shape_equal") or v["cosine"] < COS_MIN}
        same_no_grad = sorted(L["runs"][r]["no_grad"]) == sorted(M["runs"][r]["no_grad"])
        cos_min = min(v["cosine"] for v in rows.values() if "cosine" in v)
        rel_max = max(v["max_rel"] for v in rows.values() if "max_rel" in v)
        worst = sorted(((v["cosine"], k) for k, v in rows.items() if "cosine" in v))[:3]
        graphix = [k for k in rows if "rgat_layer" in k or "relation_emb" in k]
        passed = not bad and same_no_grad and len(rows) > 0
        ok = ok and passed
        report["runs"][r] = {
            "bench": RUNS[r][0], "checkpointing": RUNS[r][1], "passed": passed,
            "params_with_grad": len(rows), "graphix_params_with_grad": len(graphix),
            "no_grad_legacy": L["runs"][r]["no_grad"], "no_grad_modern": M["runs"][r]["no_grad"],
            "same_no_grad_set": same_no_grad, "min_cosine": cos_min, "max_rel_diff": rel_max,
            "worst_cosine": worst, "failing": dict(list(bad.items())[:20]),
            "loss_legacy": L["runs"][r]["losses"], "loss_modern": M["runs"][r]["losses"],
            "peak_vram_gb": {"legacy": L["runs"][r]["peak_vram_gb"], "modern": M["runs"][r]["peak_vram_gb"]},
        }
        print("%s %-16s %-7s grads=%d (graphix %d) no_grad legacy/modern=%d/%d same=%s min_cos=%.8f max_rel=%.1e %s" % (
            r, RUNS[r][0], "ckpt" if RUNS[r][1] else "no-ckpt", len(rows), len(graphix),
            len(L["runs"][r]["no_grad"]), len(M["runs"][r]["no_grad"]), same_no_grad, cos_min, rel_max,
            "PASS" if passed else "FAIL"))
    # within each environment, checkpointing must not change the gradients (R1 vs R2)
    report["checkpointing_invariance"] = {}
    for env in ("legacy", "modern"):
        rows = compare_pair(OUT / ("%s_R1.npz" % env), OUT / ("%s_R2.npz" % env))
        cmin = min(v["cosine"] for v in rows.values() if "cosine" in v)
        rmax = max(v["max_rel"] for v in rows.values() if "max_rel" in v)
        report["checkpointing_invariance"][env] = {"min_cosine": cmin, "max_rel_diff": rmax}
        print("checkpointing invariance %-6s R1 vs R2: min_cos=%.8f max_rel=%.1e" % (env, cmin, rmax))
    report["passed"] = ok
    json.dump(report, open(str(OUT / "T6_report.json"), "w"), indent=1)
    print("T6", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        sys.exit(0 if compare() else 1)
