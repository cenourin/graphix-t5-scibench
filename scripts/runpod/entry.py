#!/usr/bin/env python
"""Pod entrypoint wrapper: graphix_modern/train.py unchanged, run through the phase-A test
harness (scripts/port_entry_harness.py: HARNESS_NO_DROPOUT / HARNESS_ORDER_LOG /
HARNESS_STOP_AT_STEP), plus guards and timing that do not change any computation:

  - fp32 guard: a forward pre-hook on the model that fails the run if, at any forward,
    autocast is enabled, TF32 is allowed for matmul or cuDNN, the fp32 matmul precision is
    not "highest", or the model is not on CUDA (no silent CPU fallback). Checks Python-side
    flags only: no GPU synchronization.
  - timing: wall-clock marks at process start, train begin/end, every optimizer step end
    and evaluation end (no synchronization; averaged over many steps this is the real
    throughput), written to <output_dir>/run_timing.json.
  - run_env.json in <output_dir>: GPU, versions, backend flags, commit, image digest, and the
    GRAPHIX_*/HARNESS_* variables of the run.

  python scripts/runpod/entry.py CONFIG.json
"""
import json
import os
import runpy
import sys
import time

T0 = time.time()
sys.path.insert(0, os.getcwd())
config_path = sys.argv[1]
cfg = json.load(open(config_path))
out_dir = cfg["output_dir"]
for k in ("fp16", "bf16", "tf32", "torch_compile", "fp16_full_eval", "bf16_full_eval"):
    if cfg.get(k):
        raise SystemExit("entry.py: %s is set in %s; phase-B baseline is strict fp32" % (k, config_path))

import torch  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

from graphix_modern import trainer as T  # noqa: E402

marks = {"process_start": T0, "step_end": [], "evaluate_end": []}



def _commit():
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "runpod"))
    try:
        import gitinfo
        return gitinfo.commit(".")
    except Exception as e:
        return "unknown (%s)" % e

def _fp32_guard(module, inputs):
    bad = []
    if torch.is_autocast_enabled():
        bad.append("autocast enabled")
    if torch.backends.cuda.matmul.allow_tf32:
        bad.append("matmul TF32 allowed")
    if torch.backends.cudnn.allow_tf32:
        bad.append("cuDNN TF32 allowed")
    if torch.get_float32_matmul_precision() != "highest":
        bad.append("float32 matmul precision %s" % torch.get_float32_matmul_precision())
    p = next(module.parameters())
    if p.device.type != "cuda":
        bad.append("model on %s" % p.device)
    if p.dtype != torch.float32:
        bad.append("parameters in %s" % p.dtype)
    if bad:
        raise RuntimeError("fp32 guard: " + "; ".join(bad))


class _Timing(TrainerCallback):
    def on_train_begin(self, args, state, control, **kw):
        marks["train_begin"] = time.time()

    def on_step_end(self, args, state, control, **kw):
        marks["step_end"].append([state.global_step, time.time()])

    def on_train_end(self, args, state, control, **kw):
        marks["train_end"] = time.time()

    def on_evaluate(self, args, state, control, **kw):
        marks["evaluate_end"].append(time.time())

    def on_save(self, args, state, control, **kw):
        marks.setdefault("save_end", []).append([state.global_step, time.time()])


_orig_init = T.LegacyLoopTrainer.__init__


def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    self.add_callback(_Timing())
    self.model.register_forward_pre_hook(_fp32_guard)
    marks["model_ready"] = time.time()
    marks["n_train"] = len(self.train_dataset) if self.train_dataset is not None else None
    marks["n_eval"] = len(self.eval_dataset) if self.eval_dataset is not None else None


T.LegacyLoopTrainer.__init__ = _init


def _write():
    os.makedirs(out_dir, exist_ok=True)
    marks["process_end"] = time.time()
    json.dump(marks, open(os.path.join(out_dir, "run_timing.json"), "w"))
    env = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
           "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
           "float32_matmul_precision": torch.get_float32_matmul_precision(),
           "commit": _commit(),
           "image_digest": os.environ.get("GRAPHIX_IMAGE_DIGEST"),
           "vars": {k: v for k, v in os.environ.items() if k.startswith(("GRAPHIX_", "HARNESS_", "PYTORCH_CUDA"))},
           "config": config_path}
    json.dump(env, open(os.path.join(out_dir, "run_env.json"), "w"), indent=1)


import atexit  # noqa: E402

atexit.register(_write)
sys.argv = ["port_entry_harness.py", "modern", config_path]
runpy.run_path("scripts/port_entry_harness.py", run_name="__main__")
