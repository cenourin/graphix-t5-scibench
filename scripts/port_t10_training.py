#!/usr/bin/env python
"""Test T10 (PORTABILIDADE.md, step A7): a short real training run gives the same dynamics
on both stacks, across an epoch boundary with leftover micro-batches.

Both real entrypoints (legacy seq2seq/run_seq2seq_train.py, modern graphix_modern/train.py)
through scripts/port_entry_harness.py, GTX 1070, fp32, TF32 off, dropout neutralized
(x * 1.0), same initial weights (T4's legacy state_dict, as a wrapper .bin via
GRAPHIX_INIT_STATE_DICT). Spider, the first 206 filtered training examples, GA 8, 2 epochs:
floor(206/8) = 25 updates per epoch with 6 leftover micro-batches that must join the first
update of epoch 2 (4.17 semantics), max_steps = 50; logging every step.
Compares: global_step, number of optimizer steps, LR per step (exact), the order of the
training examples fetched (exact), and the loss per step against each stack's run-to-run
noise envelope (3 runs per stack; runs 2 and 3 from scripts/port_t10_noise.sh; see compare()).
  prepare (legacy image) | run ENV | compare
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
OUT = Path("data_all_in/data/port_tests/T10")
T4 = Path("data_all_in/data/port_tests/T4/legacy_state_dict.npz")
N, GA, EPOCHS = 206, 8, 2


def math_sign(x):
    return (x > 0) - (x < 0)


def config(env):
    c = json.load(open("configs/study_t5base_spider.json"))
    c.update({"model_name_or_path": "data_all_in/t5-base" if env == "legacy" else "data_all_in/t5-base-st",
              "cache_dir": "/tmp/hf_cache", "output_dir": str(OUT / env / "run"),
              "max_train_samples": N, "gradient_accumulation_steps": GA, "num_train_epochs": EPOCHS,
              "learning_rate": 1e-4, "warmup_ratio": 0.1, "weight_decay": 0.01,
              "do_train": True, "do_eval": False, "evaluation_strategy": "no", "save_strategy": "no",
              "load_best_model_at_end": False, "predict_with_generate": False, "overwrite_output_dir": True,
              "logging_strategy": "steps", "logging_steps": 1, "logging_first_step": True, "report_to": []})
    for k in ("eval_steps", "save_steps", "metric_for_best_model", "greater_is_better"):
        c.pop(k, None)
    return c


def prepare():
    import numpy as np
    import torch
    w = np.load(str(T4))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"pretrain_model." + k: torch.from_numpy(w[k]) for k in w.files}, str(OUT / "init_state_dict.bin"))
    for env in ("legacy", "modern"):
        (OUT / env).mkdir(parents=True, exist_ok=True)
        json.dump(config(env), open(str(OUT / env / "config.json"), "w"), indent=1)
    print("wrote", OUT / "init_state_dict.bin", "and configs")


def compare():
    res, ok = {"test": "T10"}, True
    st = {e: json.load(open(str(OUT / e / "run" / "trainer_state.json"))) for e in ("legacy", "modern")}
    logs = {e: {h["step"]: h for h in st[e]["log_history"] if "loss" in h} for e in st}
    order = {e: json.load(open(str(OUT / e / "order.json"))) for e in st}
    res["global_step"] = [st["legacy"]["global_step"], st["modern"]["global_step"]]
    res["max_steps"] = [st["legacy"]["max_steps"], st["modern"]["max_steps"]]
    res["logged_steps"] = [len(logs["legacy"]), len(logs["modern"])]
    steps = sorted(logs["legacy"])
    lr_diff = [abs(logs["legacy"][s]["learning_rate"] - logs["modern"][s]["learning_rate"]) for s in steps if s in logs["modern"]]
    loss_rel = [abs(logs["legacy"][s]["loss"] - logs["modern"][s]["loss"]) / max(abs(logs["legacy"][s]["loss"]), 1e-12)
                for s in steps if s in logs["modern"]]
    res["lr_max_abs_diff"] = max(lr_diff) if lr_diff else None
    res["loss_max_rel_diff"] = max(loss_rel) if loss_rel else None
    res["loss_rel_diff_by_step"] = {s: round(r, 6) for s, r in zip(steps, loss_rel)}
    res["order_identical"] = order["legacy"] == order["modern"]
    res["order_lengths"] = [len(order["legacy"]), len(order["modern"])]
    res["epoch_boundary"] = {"micro_batches_epoch1": N, "leftover": N % GA,
                             "first_order_diff": next((i for i, (p, q) in enumerate(zip(order["legacy"], order["modern"])) if p != q), None)}
    # Loss criterion, fixed on 2026-09-29 BEFORE the run it judges (PORTABILIDADE.md, T10).
    # GPU training here is not bitwise reproducible (atomic scatter-sums in DGL, cuBLAS), and
    # 50 steps amplify 1e-6 perturbations to ~1e-2: nominally identical runs of the SAME image
    # already diverge that much. The unit of analysis is the RUN (steps of a loss curve are
    # autocorrelated, not independent samples): 3 runs per stack give 3 legacy-legacy and
    # 3 modern-modern pairs (6 same-stack pairs, the envelope) and 9 legacy-modern pairs.
    # Per pair, over the 50 steps: mean and max relative |dloss|, RMSE of the curves, first
    # step with relative |dloss| > 1e-4. Passes if, for mean_rel, max_rel and RMSE, the
    # MEDIAN of the 9 cross-stack pairs is <= the largest same-stack value, and there is no
    # consistent sign bias: it fails if every legacy run's mean loss is on the same side of
    # every modern run's AND the gap between the two stacks' average run means exceeds the
    # largest same-stack gap. (Median, not "all 9": with 15 exchangeable draws the largest
    # falls among the 9 cross pairs 60% of the time even with no stack effect.)
    # first_divergence_step is DIAGNOSTIC, not gated (decided before the runs): it detects
    # the first rounding-level difference, not its size, and the two stacks' kernels are
    # known to differ at ~1e-6 (T5/T6), so cross pairs are expected to leave the 4th decimal
    # 1-2 steps earlier without any change in training dynamics. The perturbed run is also
    # a diagnostic only. This is an empirical equivalence within the
    # intrinsic GPU variability, not a formal significance test (3 runs per stack).
    import itertools
    import statistics
    runs = {"L": ["legacy", "legacy2", "legacy3"], "M": ["modern", "modern2", "modern3"]}
    curves = {r: {h["step"]: h["loss"] for h in json.load(open(str(OUT / r / "run" / "trainer_state.json")))["log_history"]
                  if "loss" in h} for r in runs["L"] + runs["M"] + ["perturbed"]
              if (OUT / r / "run" / "trainer_state.json").exists()}
    missing = [r for r in runs["L"] + runs["M"] if r not in curves]

    def pair(a, b):
        x, y = curves[a], curves[b]
        rel = [abs(x[s] - y[s]) / abs(x[s]) for s in sorted(x)]
        return {"mean_rel": statistics.mean(rel), "max_rel": max(rel),
                "rmse": statistics.mean([(x[s] - y[s]) ** 2 for s in sorted(x)]) ** 0.5,
                "first_divergence_step": next((i + 1 for i, r in enumerate(rel) if r > 1e-4), len(rel) + 1)}
    envelope_ok, bias = False, None
    if not missing:
        same = {a + "~" + b: pair(a, b) for k in "LM" for a, b in itertools.combinations(runs[k], 2)}
        cross = {a + "~" + b: pair(a, b) for a in runs["L"] for b in runs["M"]}
        gate = {}
        for m in ("mean_rel", "max_rel", "rmse", "first_divergence_step"):
            sv, cv = [p[m] for p in same.values()], [p[m] for p in cross.values()]
            med = statistics.median(cv)
            ok = None if m == "first_divergence_step" else med <= max(sv)  # None = diagnostic
            gate[m] = {"same_stack_range": [min(sv), max(sv)], "cross_median": med, "cross_range": [min(cv), max(cv)],
                       "cross_pairs_outside_envelope": sum(1 for c in cv if (c < min(sv) if m == "first_divergence_step" else c > max(sv))),
                       "passed": ok}
        run_mean = {r: statistics.mean(curves[r].values()) for r in runs["L"] + runs["M"]}
        signs = {math_sign(run_mean[a] - run_mean[b]) for a in runs["L"] for b in runs["M"]}
        stack_gap = abs(statistics.mean(run_mean[r] for r in runs["L"]) - statistics.mean(run_mean[r] for r in runs["M"]))
        same_gap = max(abs(run_mean[a] - run_mean[b]) for k in "LM" for a, b in itertools.combinations(runs[k], 2))
        bias = {"run_mean_loss": run_mean, "all_cross_same_sign": len(signs) == 1, "stack_gap": stack_gap,
                "max_same_stack_gap": same_gap, "consistent_bias": len(signs) == 1 and stack_gap > same_gap}
        envelope_ok = all(g["passed"] for g in gate.values() if g["passed"] is not None) and not bias["consistent_bias"]
        res["pairs_same_stack"], res["pairs_cross_stack"], res["loss_gate"] = same, cross, gate
        if "perturbed" in curves:
            res["diagnostic_perturbed_1e-6"] = {r + "~perturbed": pair(r, "perturbed") for r in runs["M"]}
    res["missing_runs"] = missing
    # Mechanics must also be exactly equal across ALL six runs (not only the first pair).
    mech = {}
    for r in runs["L"] + runs["M"]:
        if r in curves and (OUT / r / "order.json").exists():
            st_r = json.load(open(str(OUT / r / "run" / "trainer_state.json")))
            mech[r] = (st_r["global_step"], st_r["max_steps"],
                       tuple(h["learning_rate"] for h in st_r["log_history"] if "loss" in h),
                       tuple(json.load(open(str(OUT / r / "order.json")))))
    res["mechanics_identical_all_runs"] = len(mech) == 6 and len(set(mech.values())) == 1
    res["sign_bias"] = bias
    res["loss_within_noise_envelope"] = envelope_ok
    res["passed"] = (res["global_step"][0] == res["global_step"][1] == 50 and res["max_steps"][0] == res["max_steps"][1]
                     and res["logged_steps"][0] == res["logged_steps"][1] and res["lr_max_abs_diff"] == 0.0
                     and res["order_identical"] and res["mechanics_identical_all_runs"] and envelope_ok)
    json.dump(res, open(str(OUT / "T10_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("loss_rel_diff_by_step", "pairs_same_stack", "pairs_cross_stack", "diagnostic_perturbed_1e-6")}, indent=1))
    print("loss rel diff at steps 1, 25, 26, 50:", {s: res["loss_rel_diff_by_step"].get(s) for s in (1, 25, 26, 50)})
    print("T10", "PASSED" if res["passed"] else "FAILED")
    return res["passed"]


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        sys.exit(0 if compare() else 1)
