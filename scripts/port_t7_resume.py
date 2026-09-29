#!/usr/bin/env python
"""Test T7 (PORTABILIDADE.md, step A6.3): resuming from a checkpoint restores the training
state exactly (model, optimizer, scheduler, data position). Modern image only.

GPU training is not bitwise reproducible (T10), so a resumed run cannot be compared with a
separate uninterrupted run step by step: they have already drifted apart before the
checkpoint. The resume is isolated instead, with every compared run starting from the SAME
checkpoint-30 (runs from scripts/port_t10_noise.sh and scripts/port_t7_resume2.sh):
  A  uninterrupted run to 50            B  stopped at 30 with checkpoint-30, resumed to 50
  D  a second resume from B's checkpoint-30, logging the item order
  E  resume from checkpoint-30 with the Adafactor state emptied (negative control)
Gates:
  G1 B and D reach global_step 50, with the LR of A at every step 31-50 (scheduler state);
  G2 the items trained at steps 31-50 by D are exactly A's (data position, sampler, RNG);
  G3 step 31 (same weights, same batch): B and D agree within the loss log's rounding
     (4 decimals), so model weights and data are restored;
  G4 over steps 32-35, B vs D stays far below B vs E: the optimizer state is restored and
     matters (without it the updates change measurably).
"""
import json
import sys
from pathlib import Path

OUT = Path("data_all_in/data/port_tests/T7")
ROUND = 1.0e-4  # trainer logs round the loss to 4 decimals


def logs(run):
    st = json.load(open(str(OUT / run / "run" / "trainer_state.json")))
    return st, {h["step"]: h for h in st["log_history"] if "loss" in h}


def compare():
    (stA, A), (stB, B), (stD, D), (stE, E) = logs("A"), logs("B"), logs("D"), logs("E")
    orderA = json.load(open("data_all_in/data/port_tests/T10/modern/order.json"))  # A = the T10 modern run
    orderD = json.load(open(str(OUT / "D" / "order.json")))
    trained_after_30 = 20 * 8  # 20 updates x GA 8, all inside epoch 2
    res = {"test": "T7"}
    res["G1"] = {"global_step": {"B": stB["global_step"], "D": stD["global_step"]},
                 "lr_equal_A_31_50": all(A[s]["learning_rate"] == B[s]["learning_rate"] == D[s]["learning_rate"]
                                         for s in range(31, 51))}
    res["G1"]["passed"] = stB["global_step"] == stD["global_step"] == 50 and res["G1"]["lr_equal_A_31_50"]
    res["G2"] = {"items_31_50_equal_A": orderA[-trained_after_30:] == orderD[-trained_after_30:],
                 "order_log_lengths": {"A": len(orderA), "D": len(orderD)}}
    res["G2"]["passed"] = res["G2"]["items_31_50_equal_A"]
    d31 = abs(B[31]["loss"] - D[31]["loss"])
    res["G3"] = {"step31_loss": {"B": B[31]["loss"], "D": D[31]["loss"], "E": E[31]["loss"]}, "B_vs_D_abs": d31,
                 "passed": d31 <= ROUND}
    bd = [abs(B[s]["loss"] - D[s]["loss"]) for s in range(32, 36)]
    be = [abs(B[s]["loss"] - E[s]["loss"]) for s in range(32, 36)]
    res["G4"] = {"steps": [32, 33, 34, 35], "B_vs_D": bd, "B_vs_E_control": be,
                 "passed": sum(bd) * 10 <= sum(be) and sum(be) > 10 * ROUND}
    res["post_resume_drift_31_50"] = {"B_vs_D_max_abs": max(abs(B[s]["loss"] - D[s]["loss"]) for s in range(31, 51)),
                                      "B_vs_A_max_abs": max(abs(B[s]["loss"] - A[s]["loss"]) for s in range(31, 51))}
    res["passed"] = all(res[g]["passed"] for g in ("G1", "G2", "G3", "G4"))
    json.dump(res, open(str(OUT / "T7_report.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("T7", "PASSED" if res["passed"] else "FAILED")
    return res["passed"]


if __name__ == "__main__":
    sys.exit(0 if compare() else 1)
