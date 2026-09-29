#!/usr/bin/env python
"""Empirical trace of the training loop's accumulation/epoch semantics (step A6.2c).
Python 3.7 compatible; same script in both images.

Toy setup: N=10 micro-batches (batch size 1), gradient_accumulation_steps=4, 3 epochs, a
one-parameter model whose loss per micro-batch is w * 1, so after backward w.grad equals
(micro-batches accumulated) / (loss divisor). The trainer's optimizer.step and
scheduler.step are wrapped to record, per optimizer update:
  - which example ids (in processing order) contributed since the previous update,
  - global_step, learning rate used, and w.grad (reveals the loss normalization);
plus the full example order per epoch, scheduler calls, max_steps, and the ids processed
after the last update (never stepped).
  python scripts/port_trace_epochs.py ENV TRAINER   (TRAINER: stock | ported)
Output: data_all_in/data/port_tests/TRACE/<ENV>_<TRAINER>.json
"""
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, ".")
OUT = Path("data_all_in/data/port_tests/TRACE")
N, GA, EPOCHS, SEED = 10, 4, 3, 1


class Toy(nn.Module):
    def __init__(self, log):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.log = log

    def forward(self, ids=None, labels=None, **kw):
        for i in ids.tolist():
            self.log["pending"].append(int(i))
            self.log["order"].append(int(i))
        return {"loss": (self.w * torch.ones(len(ids))).sum()}


class DS(torch.utils.data.Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return {"ids": torch.tensor(i), "labels": torch.tensor(0)}


def collate(items):
    return {"ids": torch.stack([x["ids"] for x in items]), "labels": torch.stack([x["labels"] for x in items])}


def main(env, which):
    from transformers import Seq2SeqTrainingArguments, set_seed
    if which == "stock":
        from transformers import Trainer as Base
    else:
        from graphix_modern.trainer import LegacyLoopTrainer as Base

    log = {"pending": [], "order": [], "updates": [], "scheduler_steps": 0}

    class Traced(Base):
        def create_optimizer(self):
            opt = super().create_optimizer()
            trainer = self
            base = type(opt)

            def step(opt_self, *a, **k):
                w = trainer.model.w
                log["updates"].append({"global_step_before": int(trainer.state.global_step),
                                       "ids": list(log["pending"]), "n_micro": len(log["pending"]),
                                       "lr": float(opt_self.param_groups[0]["lr"]),
                                       "w_grad": float(w.grad) if w.grad is not None else None})
                log["pending"] = []
                return base.step(opt_self, *a, **k)
            # a real subclass method: torch's LR scheduler wraps optimizer.step and needs it bound
            opt.__class__ = type("Traced" + base.__name__, (base,), {"step": step})
            return opt

        def create_scheduler(self, num_training_steps, optimizer=None):
            sch = super().create_scheduler(num_training_steps, optimizer)
            base = type(sch)

            def step(sch_self, *a, **k):
                log["scheduler_steps"] += 1
                return base.step(sch_self, *a, **k)
            sch.__class__ = type("Traced" + base.__name__, (base,), {"step": step})
            log["scheduler_num_training_steps"] = int(num_training_steps)
            return sch

    set_seed(SEED)
    args = Seq2SeqTrainingArguments(output_dir="/tmp/trace_out", per_device_train_batch_size=1,
                                    gradient_accumulation_steps=GA, num_train_epochs=EPOCHS, learning_rate=1.0,
                                    warmup_steps=0, lr_scheduler_type="linear", max_grad_norm=0.0,
                                    logging_steps=10 ** 6, save_strategy="no", report_to=[], seed=SEED,
                                    optim="adamw_torch",
                                    dataloader_drop_last=False, remove_unused_columns=False, disable_tqdm=True)
    model = Toy(log)
    if which == "ported":
        model.accepts_loss_kwargs = False
    trainer = Traced(model=model, args=args, train_dataset=DS(), data_collator=collate)
    trainer.train()
    import transformers
    result = {"env": env, "trainer": which, "transformers": transformers.__version__, "N": N, "GA": GA,
              "epochs": EPOCHS, "max_steps": int(trainer.state.max_steps), "global_step": int(trainer.state.global_step),
              "optimizer_steps": len(log["updates"]), "scheduler_steps": log["scheduler_steps"],
              "scheduler_num_training_steps": log.get("scheduler_num_training_steps"),
              "updates": log["updates"], "order": log["order"],
              "order_per_epoch": [log["order"][e * N:(e + 1) * N] for e in range(EPOCHS)],
              "processed_after_last_update": log["pending"]}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(str(OUT / ("%s_%s.json" % (env, which))), "w"), indent=1)
    print(json.dumps({k: result[k] for k in ("transformers", "max_steps", "global_step", "optimizer_steps",
                                            "scheduler_steps", "processed_after_last_update")}))
    for u in result["updates"]:
        print("step_before=%d n_micro=%d ids=%s lr=%.4f w.grad=%s" % (u["global_step_before"], u["n_micro"], u["ids"],
                                                                     u["lr"], u["w_grad"]))
    print("order per epoch:", result["order_per_epoch"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
