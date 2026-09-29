#!/usr/bin/env python
"""Test T4 (PORTABILIDADE.md, step A4): the ported Graphix-T5 has exactly the legacy
structure and loads a legacy checkpoint strictly. Python 3.7 compatible.

  dump legacy   seq2seq.models.modeling_t5 (transformers 4.17) from data_all_in/t5-base;
                writes its full state_dict as .npz (pickle-free) plus structure.
  dump modern   graphix_modern.modeling_t5 (transformers 4.57.6) from data_all_in/t5-base-st;
                structure, then loads the legacy state_dict with strict=True.
  compare       the gates below; report in data_all_in/data/port_tests/T4/.
Gates:
  G1 load_state_dict(legacy state_dict, strict=True): zero missing, zero unexpected keys,
     and every tensor equal bit for bit after loading;
  G2 ordered named_parameters(): same names, shapes, dtypes, requires_grad (includes the
     unused rgat_layer.filter and decoder.relation_emb kept for parity);
  G3 same buffers; G4 same config values that define the model; G5 same weight tying;
  G6 same set of keys missing from the stock t5-base (= the Graphix-added parameters);
  G7 same initialization scheme for those parameters: each environment's tensors match the
     scheme's theoretical distribution (range; mean and std within 5 standard errors).
     Bitwise-equal initial draws are reported, not required (RNG streams differ by version).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
OUT = Path("data_all_in/data/port_tests/T4")
CONFIG_KEYS = ["d_model", "d_ff", "d_kv", "num_layers", "num_decoder_layers", "num_heads",
               "relative_attention_num_buckets", "relative_attention_max_distance", "dropout_rate",
               "layer_norm_epsilon", "initializer_factor", "feed_forward_proj", "vocab_size",
               "tie_word_embeddings", "is_encoder_decoder", "decoder_start_token_id", "pad_token_id",
               "eos_token_id"]
SEED = 0


def build(env):
    import torch
    from transformers import AutoConfig
    torch.manual_seed(SEED)
    if env == "legacy":
        from seq2seq.models.modeling_t5 import T5ForConditionalGeneration
        path = "data_all_in/t5-base"
    else:
        from graphix_modern.modeling_t5 import T5ForConditionalGeneration
        path = "data_all_in/t5-base-st"
    cfg = AutoConfig.from_pretrained(path)
    model, info = T5ForConditionalGeneration.from_pretrained(path, config=cfg, output_loading_info=True)
    return model, info, cfg


def effective_config(cfg):
    """The values each implementation actually uses. transformers 4.17's T5Config has no
    relative_attention_max_distance, dense_act_fn or is_gated_act: the legacy code hardcodes
    max_distance=128 (_relative_position_bucket default, never overridden by its caller) and
    derives the FF block from feed_forward_proj ("relu" -> T5DenseReluDense, not gated)."""
    c = {k: getattr(cfg, k, None) for k in CONFIG_KEYS}
    fp = getattr(cfg, "feed_forward_proj", "relu")
    c["relative_attention_max_distance"] = getattr(cfg, "relative_attention_max_distance", 128)
    c["dense_act_fn"] = getattr(cfg, "dense_act_fn", fp.split("-")[-1])
    c["is_gated_act"] = getattr(cfg, "is_gated_act", fp.startswith("gated-"))
    return c


def init_stats(t):
    t = t.detach().double()
    return {"min": float(t.min()), "max": float(t.max()), "mean": float(t.mean()),
            "std": float(t.std()) if t.numel() > 1 else 0.0}


def structure(model, info, cfg):
    sd = model.state_dict()
    return {
        "named_parameters": [[n, list(p.shape), str(p.dtype), bool(p.requires_grad)] for n, p in model.named_parameters()],
        "buffers": [[n, list(b.shape), str(b.dtype)] for n, b in model.named_buffers()],
        "state_dict_keys": list(sd.keys()),
        "config": effective_config(cfg),
        "tying": {
            "lm_head_is_shared": model.lm_head.weight.data_ptr() == model.shared.weight.data_ptr(),
            "encoder_embed_is_shared": model.encoder.embed_tokens.weight.data_ptr() == model.shared.weight.data_ptr(),
            "decoder_embed_is_shared": model.decoder.embed_tokens.weight.data_ptr() == model.shared.weight.data_ptr(),
        },
        "missing_from_t5_base": sorted(info["missing_keys"]),
        "unexpected_in_t5_base": sorted(info["unexpected_keys"]),
        "init": {k: init_stats(sd[k]) for k in sorted(info["missing_keys"])},
    }


def dump(env):
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    model, info, cfg = build(env)
    s = structure(model, info, cfg)
    s["env"] = {"torch": torch.__version__, "transformers": __import__("transformers").__version__}
    if env == "legacy":
        sd = model.state_dict()
        np.savez(str(OUT / "legacy_state_dict.npz"), **{k: v.detach().cpu().numpy() for k, v in sd.items()})
        s["init_values_sha"] = {k: __import__("hashlib").sha256(sd[k].detach().numpy().tobytes()).hexdigest()
                                for k in s["missing_from_t5_base"]}
    else:
        legacy = np.load(str(OUT / "legacy_state_dict.npz"))
        state = {k: torch.from_numpy(legacy[k]) for k in legacy.files}
        result = model.load_state_dict(state, strict=True)
        after = model.state_dict()
        s["strict_load"] = {
            "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
            "not_bit_equal_after_load": [k for k in state if not torch.equal(after[k], state[k])],
        }
        s["init_values_sha"] = {}  # filled from the pre-load model below
        model2, info2, _ = build(env)
        sd2 = model2.state_dict()
        s["init_values_sha"] = {k: __import__("hashlib").sha256(sd2[k].detach().numpy().tobytes()).hexdigest()
                                for k in s["missing_from_t5_base"]}
    json.dump(s, open(str(OUT / ("%s.json" % env)), "w"), indent=1)
    print(env, "params", len(s["named_parameters"]), "missing from t5-base", len(s["missing_from_t5_base"]))


def init_ok(kind, st, n, fan_in):
    """One environment's initial tensor against the theoretical distribution of its scheme,
    within 5 standard errors (n = number of elements)."""
    if kind == "embedding":  # N(0, 1)
        return abs(st["mean"]) < 5 / np.sqrt(n) and abs(st["std"] - 1) < 5 * np.sqrt(0.5 / n)
    bound = 1.0 / np.sqrt(fan_in)  # uniform(-bound, bound): mean 0, std bound/sqrt(3)
    sd = bound / np.sqrt(3)
    return (max(abs(st["min"]), abs(st["max"])) <= bound * (1 + 1e-6)
            and abs(st["mean"]) < 5 * sd / np.sqrt(n)
            and abs(st["std"] - sd) < 5 * sd * np.sqrt(0.2 / n))  # uniform kurtosis 1.8: var(s)~sd^2*0.8/(4n)


def init_kind(name):
    if name.endswith("relation_emb.weight"):
        return "embedding"
    if "layer_norm" in name or "layernorm" in name:
        return "norm"
    return "linear"


def compare():
    L = json.load(open(str(OUT / "legacy.json")))
    M = json.load(open(str(OUT / "modern.json")))
    g = {}
    sl = M["strict_load"]
    g["G1_strict_load"] = not sl["missing_keys"] and not sl["unexpected_keys"] and not sl["not_bit_equal_after_load"]
    g["G2_named_parameters"] = L["named_parameters"] == M["named_parameters"]
    g["G3_buffers"] = L["buffers"] == M["buffers"]
    g["G4_config"] = L["config"] == M["config"]
    g["G5_tying"] = L["tying"] == M["tying"] and all(L["tying"].values())
    g["G6_missing_from_t5_base"] = L["missing_from_t5_base"] == M["missing_from_t5_base"] and not M["unexpected_in_t5_base"]
    # G7: same initialization scheme, per tensor
    shape = {n: s for n, s, _, _ in L["named_parameters"]}
    bad_init = []
    for k in L["missing_from_t5_base"]:
        a, b, kind = L["init"][k], M["init"][k], init_kind(k)
        n = int(np.prod(shape[k])) if k in shape else 1
        if kind == "norm":  # deterministic: ones (weights) / zeros (biases)
            ok = a == b
        else:  # each environment against the theoretical distribution
            fan_in = shape[k.replace(".bias", ".weight")][-1]
            ok = init_ok(kind, a, n, fan_in) and init_ok(kind, b, n, fan_in)
        if not ok:
            bad_init.append({"key": k, "legacy": a, "modern": b})
    g["G7_init_scheme"] = not bad_init
    same_draws = sum(L["init_values_sha"][k] == M["init_values_sha"][k] for k in L["missing_from_t5_base"])
    report = {"test": "T4", "gates": g, "passed": all(g.values()),
              "legacy_env": L["env"], "modern_env": M["env"], "strict_load": sl,
              "n_named_parameters": [len(L["named_parameters"]), len(M["named_parameters"])],
              "n_missing_from_t5_base": len(L["missing_from_t5_base"]), "init_mismatches": bad_init[:20],
              "info_bitwise_equal_initial_draws": "%d/%d" % (same_draws, len(L["missing_from_t5_base"]))}
    if not g["G2_named_parameters"]:
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(L["named_parameters"], M["named_parameters"])) if a != b][:10]
        report["named_parameters_first_diffs"] = diff
    json.dump(report, open(str(OUT / "T4_report.json"), "w"), indent=1)
    print(json.dumps({k: report[k] for k in ("gates", "passed", "n_named_parameters", "n_missing_from_t5_base",
                                            "info_bitwise_equal_initial_draws")}, indent=1))
    if not report["passed"]:
        print(json.dumps({k: report.get(k) for k in ("strict_load", "init_mismatches", "named_parameters_first_diffs")}, indent=1)[:3000])
    return report["passed"]


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        sys.exit(0 if compare() else 1)
