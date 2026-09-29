#!/usr/bin/env python
"""Test T7 (PORTABILIDADE.md, step A6.3): a modern run interrupted and resumed from its
checkpoint ends where the uninterrupted run ends. Modern image only (the legacy has no
resume path of its own to compare).

Same setup as T10 (modern config, dropout neutralized, common initial weights, 50 steps).
  A   uninterrupted run
  A'  the same again: measures run-to-run noise (DGL's atomic scatter-sums on GPU)
  B   stopped after step 30 with a checkpoint (HARNESS_STOP_AT_STEP), then resumed from it
      (overwrite_output_dir false -> last checkpoint) to step 50
Gate: B's final global_step is 50; LR identical to A at every step 31-50; the loss gap
B vs A over steps 31-50 and the final-weights gap B vs A are no larger than A' vs A (plus a
small floor). The resumed run exercises checkpoint save/load of model (safetensors),
optimizer and scheduler (loaded with weights_only, graphix_modern/trainer.py), RNG state,
and the 4.17 data-skip logic.
  python scripts/port_t7_resume.py compare   (runs are driven from the host, see pipeline)
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("data_all_in/data/port_tests/T7")
FLOOR = 1e-6


def weights(run_dir):
    from safetensors.numpy import load_file
    return load_file(str(Path(run_dir) / "model.safetensors"))


def logs(run_dir):
    st = json.load(open(str(Path(run_dir) / "trainer_state.json")))
    return st, {h["step"]: h for h in st["log_history"] if "loss" in h}


def max_weight_gap(a, b):
    return max(float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()) for k in a)


def compare():
    stA, la = logs(OUT / "A" / "run")
    stA2, la2 = logs(OUT / "A2" / "run")
    stB, lb = logs(OUT / "B" / "run")
    steps = [s for s in range(31, 51)]
    gap = lambda x, y: max(abs(x[s]["loss"] - y[s]["loss"]) for s in steps)
    lr_same = all(la[s]["learning_rate"] == lb[s]["learning_rate"] for s in steps)
    wA, wA2, wB = weights(OUT / "A" / "run"), weights(OUT / "A2" / "run"), weights(OUT / "B" / "run")
    res = {"test": "T7",
           "global_step": {"A": stA["global_step"], "A2": stA2["global_step"], "B": stB["global_step"]},
           "resumed_from": 30, "lr_identical_steps_31_50": lr_same,
           "loss_gap_31_50": {"B_vs_A": gap(lb, la), "A2_vs_A (noise)": gap(la2, la)},
           "final_weight_gap": {"B_vs_A": max_weight_gap(wB, wA), "A2_vs_A (noise)": max_weight_gap(wA2, wA)}}
    res["passed"] = (stB["global_step"] == stA["global_step"] == 50 and lr_same
                     and res["loss_gap_31_50"]["B_vs_A"] <= res["loss_gap_31_50"]["A2_vs_A (noise)"] + FLOOR
                     and res["final_weight_gap"]["B_vs_A"] <= res["final_weight_gap"]["A2_vs_A (noise)"] + FLOOR)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(str(OUT / "T7_report.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("T7", "PASSED" if res["passed"] else "FAILED")
    return res["passed"]


if __name__ == "__main__":
    sys.exit(0 if compare() else 1)
