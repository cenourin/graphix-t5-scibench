#!/usr/bin/env python
"""Test T9 (PORTABILIDADE.md, step A6.3): the modern trainer builds the same optimizer and
LR schedule as the legacy one. Python 3.7 compatible; CPU only.

Uses the real study configs (configs/study_t5base_*.json) with representative Optuna
hyperparameters, a tiny stand-in model (optimizer/scheduler construction does not depend on
the architecture beyond parameter names, and weight_decay grouping is by name), and the
legacy vs modern trainer classes. Compares: optimizer class; every param-group
hyperparameter; the parameter grouping (which names get weight decay); scheduler class; and
the learning rate at every one of max_steps steps (4.17 semantics: max_steps =
ceil(epochs * floor(N / GA))).
  run ENV  |  compare
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")
OUT = Path("data_all_in/data/port_tests/T9")
CASES = [  # (config, lr, warmup_ratio, grad_accum, weight_decay, n_micro_batches, epochs)
    ("configs/study_t5base_spider.json", 1.02e-4, 0.072, 16, 0.1, 7345, 15),
    ("configs/study_t5base_sciencebenchmark.json", 3.4e-4, 0.0, 8, 0.0, 4233, 15),
    ("configs/study_t5base_sciencebenchmark.json", 5e-5, 0.1, 64, 0.01, 4233, 3),
]


def run(env):
    import torch
    from torch import nn
    from transformers import Seq2SeqTrainingArguments

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Embedding(4, 3)
            self.layer_norm = nn.LayerNorm(3)
            self.dense = nn.Linear(3, 3)

        def forward(self, **kw):
            return {"loss": self.dense.weight.sum()}

    if env == "legacy":
        from transformers import Seq2SeqTrainer as T
    else:
        from graphix_modern.trainer import LegacyLoopTrainer as T
    out = []
    for cfg_path, lr, wu, ga, wd, n_micro, epochs in CASES:
        cfg = json.load(open(cfg_path))
        kw = dict(output_dir="/tmp/t9", learning_rate=lr, warmup_ratio=wu, gradient_accumulation_steps=ga,
                  weight_decay=wd, num_train_epochs=epochs, adafactor=cfg["adafactor"], adam_epsilon=cfg["adam_eps"],
                  per_device_train_batch_size=1, report_to=[], seed=cfg["seed"])
        if env == "modern":
            kw["optim"] = "adafactor" if cfg["adafactor"] else "adamw_torch"
        args = Seq2SeqTrainingArguments(**kw)
        model = Tiny()
        trainer = T(model=model, args=args)
        max_steps = math.ceil(epochs * max(n_micro // ga, 1))  # 4.17 semantics, as the ported loop computes
        trainer.create_optimizer_and_scheduler(num_training_steps=max_steps)
        opt, sch = trainer.optimizer, trainer.lr_scheduler
        inner = getattr(opt, "optimizer", opt)
        names = {id(p): n for n, p in model.named_parameters()}
        groups = [{"params": sorted(names[id(p)] for p in g["params"]),
                   "hyper": {k: v for k, v in g.items() if k != "params"}} for g in inner.param_groups]
        lrs = []
        for _ in range(max_steps):
            lrs.append(inner.param_groups[0]["lr"])
            inner.step()
            sch.step()
        out.append({"config": cfg_path, "lr": lr, "warmup_ratio": wu, "grad_accum": ga, "weight_decay": wd,
                    "max_steps": max_steps, "optimizer": type(inner).__name__, "scheduler": type(sch).__name__,
                    "groups": groups, "lrs": lrs})
        print(env, cfg_path, "optimizer", type(inner).__name__, "steps", max_steps, "lr[0..2]", lrs[:3], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"env": env, "cases": out}, open(str(OUT / ("%s.json" % env)), "w"), indent=1, default=str)


def compare():
    a, b = json.load(open(str(OUT / "legacy.json"))), json.load(open(str(OUT / "modern.json")))
    ok, rows = True, []
    for x, y in zip(a["cases"], b["cases"]):
        r = {"config": x["config"], "grad_accum": x["grad_accum"],
             "optimizer_same": x["optimizer"] == y["optimizer"], "scheduler_same": x["scheduler"] == y["scheduler"],
             "groups_same": x["groups"] == y["groups"], "max_steps": [x["max_steps"], y["max_steps"]],
             "lr_max_abs_diff": max(abs(p - q) for p, q in zip(x["lrs"], y["lrs"])) if len(x["lrs"]) == len(y["lrs"]) else None}
        if not r["groups_same"]:
            r["groups"] = [x["groups"], y["groups"]]
        r["passed"] = (r["optimizer_same"] and r["scheduler_same"] and r["groups_same"]
                       and x["max_steps"] == y["max_steps"] and r["lr_max_abs_diff"] == 0.0)
        ok = ok and r["passed"]
        rows.append(r)
        print(json.dumps(r, default=str)[:800])
    json.dump({"test": "T9", "passed": ok, "cases": rows}, open(str(OUT / "T9_report.json"), "w"), indent=1, default=str)
    print("T9", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        sys.exit(0 if compare() else 1)
