#!/usr/bin/env python
"""Test T3 (PORTABILIDADE.md, step A3): the ported RGAT_Layer (graphix_modern, DGL 2.4)
gives the same output as the legacy one (seq2seq, DGL 0.8.2) for the same weights and
inputs. Python 3.7 compatible; the same script runs in both images.

  inputs   (legacy image)  RGAT_Layer(768, 768, heads=1) and nn.Embedding(25, 768)
                           initialized under a fixed seed, plus random node features per
                           fixture example -> inputs.npz. Both environments load these.
  run ENV DEVICE           ENV in {legacy, modern}, DEVICE in {cpu, cuda}: eval() (dropout
                           off), graphs rebuilt from the A0 export, every sub-operation
                           recorded: relation_emb -> q/k/v -> score -> exp -> message
                           passing (wv) -> normalizer (z) -> normalization (o) -> output
                           projection -> residual + layernorm -> FFN. The staged final
                           output must equal the layer's own forward() bit for bit.
  compare DEVICE           shape, dtype, max abs and max relative difference per stage;
                           pass iff the final output differs by <= 1e-5 (CPU gate).
Output in data_all_in/data/port_tests/T3/.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
OUT = Path("data_all_in/data/port_tests/T3")
FIXTURE = Path("tests/port/fixture.json")
EXPORT = Path("data_all_in/data/graph_export")
D_MODEL, N_REL, SEED, TOL = 768, 25, 0, 1e-5
STAGES = ["edge_feats_by_relation", "q", "k", "v", "score", "score_exp", "wv", "z", "o",
          "attn_out", "residual_ln", "final"]


def fixture():
    return json.load(open(str(FIXTURE)))["examples"]


def make_inputs():
    import torch
    from seq2seq.models.graphix.rgat_tuning import RGAT_Layer
    torch.manual_seed(SEED)
    layer = RGAT_Layer(D_MODEL, D_MODEL, num_heads=1, feat_drop=0.2)
    rel_emb = torch.nn.Embedding(N_REL, D_MODEL)
    arrays = {"param/" + k: v.detach().numpy() for k, v in layer.state_dict().items()}
    arrays["relation_emb"] = rel_emb.weight.detach().numpy()
    rng = np.random.RandomState(SEED)
    for i, ex in enumerate(fixture()):
        arrays["x/%d" % i] = rng.standard_normal((ex["num_nodes"], D_MODEL)).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(str(OUT / "inputs.npz"), **arrays)
    print("wrote inputs for", len(fixture()), "examples;", len([k for k in arrays if k.startswith("param/")]), "parameters")


def run(env, device):
    import dgl
    import dgl.function as fn
    import torch
    from seq2seq.utils.graph_export import GraphExport
    if env == "legacy":
        from seq2seq.models.graphix import functions as F
        from seq2seq.models.graphix.rgat_tuning import RGAT_Layer
        copy_e = fn.copy_edge
    else:
        from graphix_modern import functions as F
        from graphix_modern.rgat_tuning import RGAT_Layer
        copy_e = fn.copy_e
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # DGL 0.8.2's CPU SpMM through libxsmm fails to generate a kernel for some graphs
    # ("Failed to generate libxsmm kernel for the SpMM operation"); the legacy layer then
    # silently skipped the RGAT. Turning libxsmm off only changes which CPU kernel sums
    # the messages, not the math; done in both environments for symmetry.
    if device == "cpu" and hasattr(dgl, "use_libxsmm"):
        dgl.use_libxsmm(False)
    meta_libxsmm = getattr(dgl, "is_libxsmm_enabled", lambda: None)()

    inp = np.load(str(OUT / "inputs.npz"))
    layer = RGAT_Layer(D_MODEL, D_MODEL, num_heads=1, feat_drop=0.2)
    layer.load_state_dict({k[len("param/"):]: torch.from_numpy(inp[k]) for k in inp.files if k.startswith("param/")}, strict=True)
    layer = layer.to(device).eval()
    rel_emb = torch.nn.Embedding(N_REL, D_MODEL)
    rel_emb.weight.data.copy_(torch.from_numpy(inp["relation_emb"]))
    rel_emb = rel_emb.to(device)
    exports, out, meta = {}, {}, {"env": env, "device": device, "torch": torch.__version__, "dgl": dgl.__version__,
                                  "libxsmm_enabled": meta_libxsmm, "self_consistent": {}, "unavailable": {}}
    with torch.no_grad():
        for i, ex in enumerate(fixture()):
            try:
                result = run_one(i, ex, layer, rel_emb, inp, exports, device, F, fn, copy_e)
            except (dgl.DGLError, RuntimeError) as err:
                # the legacy layer now raises RuntimeError from the DGLError (docs/incidentes.md)
                if env == "legacy" and device == "cpu":
                    meta["unavailable"][str(i)] = str(err).splitlines()[0][:300]
                    print(env, device, i, ex["reason"], "UNAVAILABLE:", meta["unavailable"][str(i)], flush=True)
                    continue
                raise
            s, consistent = result
            meta["self_consistent"][str(i)] = consistent
            for st in STAGES:
                out["%d/%s" % (i, st)] = s[st].detach().cpu().numpy()
            print(env, device, i, ex["reason"], "self-consistent:", consistent, flush=True)
    np.savez(str(OUT / ("%s_%s.npz" % (env, device))), **out)
    json.dump(meta, open(str(OUT / ("%s_%s.json" % (env, device))), "w"), indent=1)


def run_one(i, ex, layer, rel_emb, inp, exports, device, F, fn, copy_e):
    import dgl
    import torch
    from seq2seq.utils.graph_export import GraphExport
    H, dk = layer.num_heads, layer.d_k
    if ex["export"] not in exports:
        exports[ex["export"]] = GraphExport(EXPORT / ex["export"])
    n, src, dst, rel = exports[ex["export"]].structure(ex["graph_idx"])
    g = dgl.graph((torch.from_numpy(src.astype(np.int64)), torch.from_numpy(dst.astype(np.int64))),
                  num_nodes=n, idtype=torch.int32)
    x = torch.from_numpy(inp["x/%d" % i]).to(device)
    rel_t = torch.from_numpy(rel.astype(np.int64)).to(device)
    lgx = rel_emb(rel_t)
    forward_out, _ = layer(x, lgx, g)  # the real code path

    gd = g.to(device)
    s = {}
    first = np.unique(rel, return_index=True)[1]
    s["edge_feats_by_relation"] = lgx[torch.from_numpy(first).to(device)]
    q, k, v = layer.affine_q(layer.feat_dropout(x)), layer.affine_k(layer.feat_dropout(x)), layer.affine_v(layer.feat_dropout(x))
    s["q"], s["k"], s["v"] = q, k, v
    e = lgx.view(-1, H, dk) if lgx.size(-1) == q.size(-1) else lgx.unsqueeze(1).expand(-1, H, -1)
    with gd.local_scope():
        gd.ndata['q'], gd.ndata['k'] = q.view(-1, H, dk).float(), k.view(-1, H, dk).float()
        gd.ndata['v'] = v.view(-1, H, dk).float()
        gd.edata['e'] = e.float()
        gd.apply_edges(F.src_sum_edge_mul_dst('k', 'q', 'e', 'score'))
        s["score"] = gd.edata['score'].clone()
        gd.apply_edges(F.scaled_exp('score', math.sqrt(dk)))
        s["score_exp"] = gd.edata['score'].clone()
        gd.update_all(F.src_sum_edge_mul_edge('v', 'e', 'score', 'v'), fn.sum('v', 'wv'))
        s["wv"] = gd.ndata['wv'].clone()
        # The layer fuses these two in update_all(copy, sum, div_by_z). DGL 2.x no
        # longer keeps the reduced 'z' in ndata when an apply function is given, so
        # the harness splits it to record z; self_consistent proves the split
        # reproduces the fused forward() bit for bit.
        gd.update_all(copy_e('score', 'score'), fn.sum('score', 'z'))
        s["z"] = gd.ndata['z'].clone()
        gd.apply_nodes(F.div_by_z('wv', 'z', 'o'))
        o = gd.ndata['o'].clone()
    s["o"] = o
    o = o.to(x.dtype)
    s["attn_out"] = layer.affine_o(o.view(-1, H * dk))
    s["residual_ln"] = layer.layernorm(x + s["attn_out"])
    s["final"] = layer.ffn(s["residual_ln"])
    edge_ok = bool(torch.equal(lgx, rel_emb.weight[rel_t]))
    return s, bool(torch.equal(s["final"], forward_out)) and edge_ok


def compare(device):
    a, b = np.load(str(OUT / ("legacy_%s.npz" % device))), np.load(str(OUT / ("modern_%s.npz" % device)))
    ma, mb = json.load(open(str(OUT / ("legacy_%s.json" % device)))), json.load(open(str(OUT / ("modern_%s.json" % device))))
    report = {"test": "T3", "device": device, "tolerance_final_max_abs": TOL,
              "legacy": {k: ma[k] for k in ("torch", "dgl")}, "modern": {k: mb[k] for k in ("torch", "dgl")},
              "examples": []}
    ok = all(ma["self_consistent"].values()) and all(mb["self_consistent"].values())
    unavailable = ma.get("unavailable", {})
    report["legacy_unavailable"] = unavailable
    if device != "cpu" and unavailable:
        ok = False
    for i, ex in enumerate(fixture()):
        if str(i) in unavailable:
            print("%2d %-50s legacy unavailable on %s: %s" % (i, ex["reason"], device, unavailable[str(i)][:80]))
            continue
        row = {"i": i, "reason": ex["reason"], "num_nodes": ex["num_nodes"], "num_edges": ex["num_edges"], "stages": {}}
        first_over = None
        for st in STAGES:
            x, y = a["%d/%s" % (i, st)], b["%d/%s" % (i, st)]
            same_meta = x.shape == y.shape and x.dtype == y.dtype
            d = np.abs(x.astype(np.float64) - y.astype(np.float64)) if same_meta else None
            row["stages"][st] = {
                "shape": list(x.shape), "dtype": str(x.dtype), "shape_dtype_equal": same_meta,
                "max_abs": float(d.max()) if same_meta and d.size else None,
                "max_rel": float(d.max() / max(np.abs(x).max(), 1e-30)) if same_meta and d.size else None,
            }
            if first_over is None and (not same_meta or row["stages"][st]["max_abs"] > TOL):
                first_over = st
        row["first_stage_over_tolerance"] = first_over
        row["passed"] = row["stages"]["final"]["shape_dtype_equal"] and row["stages"]["final"]["max_abs"] <= TOL
        ok = ok and row["passed"]
        report["examples"].append(row)
        f = row["stages"]["final"]
        print("%2d %-50s final max_abs=%.2e max_rel=%.2e %s%s" % (
            i, ex["reason"], f["max_abs"], f["max_rel"], "PASS" if row["passed"] else "FAIL",
            "" if first_over is None else "  (first stage > tol: %s)" % first_over))
    report["self_consistent"] = {"legacy": ma["self_consistent"], "modern": mb["self_consistent"]}
    # A comparison over zero examples proves nothing: never a pass.
    report["examples_compared"] = len(report["examples"])
    ok = ok and report["examples_compared"] > 0
    report["passed"] = ok
    json.dump(report, open(str(OUT / ("T3_report_%s.json" % device)), "w"), indent=1)
    print("T3", device, "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inputs":
        make_inputs()
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        sys.exit(0 if compare(sys.argv[2]) else 1)
