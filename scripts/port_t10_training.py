#!/usr/bin/env python
"""Test T10 (PORTABILIDADE.md, step A7): a short real training run gives the same dynamics
on both stacks, across an epoch boundary with leftover micro-batches.

Both real entrypoints (legacy seq2seq/run_seq2seq_train.py, modern graphix_modern/train.py)
through scripts/port_entry_harness.py, GTX 1070, fp32, TF32 off, dropout neutralized
(x * 1.0), same initial weights (T4's legacy state_dict, as a wrapper .bin via
GRAPHIX_INIT_STATE_DICT). Spider, the first 206 filtered training examples, GA 8, 2 epochs:
floor(206/8) = 25 updates per epoch with 6 leftover micro-batches that must join the first
update of epoch 2 (4.17 semantics), max_steps = 50; logging every step.
Compares: global_step, number of optimizer steps, LR per step (exact), loss per step,
grad norm where both log it, and the order of the training examples fetched.
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
LOSS_REL_TOL = 1e-3


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
    res["passed"] = (res["global_step"][0] == res["global_step"][1] == 50 and res["max_steps"][0] == res["max_steps"][1]
                     and res["logged_steps"][0] == res["logged_steps"][1] and res["lr_max_abs_diff"] == 0.0
                     and res["order_identical"] and res["loss_max_rel_diff"] is not None
                     and res["loss_max_rel_diff"] <= LOSS_REL_TOL)
    json.dump(res, open(str(OUT / "T10_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "loss_rel_diff_by_step"}, indent=1))
    print("loss rel diff at steps 1, 25, 26, 50:", {s: res["loss_rel_diff_by_step"].get(s) for s in (1, 25, 26, 50)})
    print("T10", "PASSED" if res["passed"] else "FAILED")
    return res["passed"]


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        sys.exit(0 if compare() else 1)
