#!/usr/bin/env python
"""Cost estimate of the t5-base study (PROTOCOLO.md §4-5) from a probe session's measurements.

  python scripts/runpod/estimate_cost.py RESULTS_DIR [--price USD_PER_HOUR]
  (price default: $RUNPOD_GPU_USD_PER_HOUR, else 0.74)

Measured per cell (probe_<bench>_<arm>, dropout on, profiling off, GA 8):
  s/sample   steady-state wall time per training micro-batch (batch 1): the time between
             optimizer steps, after the first 10 steps, divided by GA;
  n_train    training examples after the study's filters (the probe's own dataset);
  val_eval   the study's per-epoch validation (loss only, full val split), measured once;
  startup    process start -> first training step (data, graph store, model load), with the
             HF datasets cache already built (the lowest startup measured for the cell); the
             first run of each benchmark on a fresh volume also builds that cache once
             (reported as one_time_hf_cache_build_s, not multiplied by the runs).
Measured once: checkpoint save time (smoke, checkpoint-3); dev generation s/example
(devgen_<bench>, RGAT arm, with the undertrained probe model: generation length may differ
from a trained model's, so this term is rough). Dev sizes are the full dev sets.

Study structure (PROTOCOLO.md): per cell 6 trials x 3 epochs (search; each epoch = train +
val eval + checkpoint save), a final run of 4..15 epochs (early stopping, patience 3), and
one dev evaluation with generation. Trials are counted at their full 3 epochs (pruning and
early stopping can only make them shorter), so the trial figures are upper bounds; the
final run is given as a range. GA changes the number of optimizer steps per epoch, not the
number of micro-batches, so epoch time is taken as GA-independent (optimizer-step cost is
in the per-sample time at GA 8 and is small; the profile reports it). This is an estimate
from a few hundred steps, not a measurement of the study.
"""
import argparse
import json
import math
import os
from pathlib import Path

CELLS = [("spider", "rgat"), ("spider", "plain"), ("sciencebenchmark", "rgat"), ("sciencebenchmark", "plain")]
GA_CHOICES = [8, 16, 32, 64]
N_DEV = {"spider": 1034, "sciencebenchmark": 299}
TRIALS, SEARCH_EPOCHS, FINAL_EPOCHS_MIN, FINAL_EPOCHS_MAX = 6, 3, 4, 15
WARMUP_SKIP = 10


def per_sample_seconds(timing, ga=8):
    se = sorted(timing["step_end"])
    se = [t for t in se if t[0] > WARMUP_SKIP] or se
    if len(se) < 2:
        return None
    (s0, t0), (s1, t1) = se[0], se[-1]
    return (t1 - t0) / (s1 - s0) / ga


def estimate(results_dir, price):
    R = Path(results_dir)

    def load(name):
        p = R / (name + ".json")
        return json.load(open(p)) if p.exists() else None

    smoke = load("smoke_train")
    save_s = smoke.get("checkpoint_save_s") if smoke else None

    def startups(bench, arm):
        xs = []
        for kind in ("probe", "profile"):
            r = load("%s_%s_%s" % (kind, bench, arm))
            if r and "train_begin" in r["timing"]:
                xs.append(r["timing"]["train_begin"] - r["timing"]["process_start"])
        return xs
    out = {"price_usd_per_hour": price, "checkpoint_save_s": save_s, "cells": {}, "assumptions": __doc__.strip(),
           "one_time_hf_cache_build_s": {}}
    tot = {"lower_h": 0.0, "upper_h": 0.0}
    for bench, arm in CELLS:
        pr = load("probe_%s_%s" % (bench, arm))
        if pr is None:
            out["cells"]["%s_%s" % (bench, arm)] = {"missing": True}
            continue
        tim = pr["timing"]
        sps = per_sample_seconds(tim)
        n_train, n_val = tim.get("n_train"), tim.get("n_eval")
        val_s = pr["eval"].get("eval_runtime")
        su = startups(bench, arm)
        startup = min(su)  # warm HF cache: what every study run pays
        if len(su) > 1 or arm == "rgat":
            cold = max(startups(bench, "rgat") + startups(bench, "plain"))
            out["one_time_hf_cache_build_s"][bench] = max(0.0, cold - startup)
        dg = load("devgen_%s" % bench)
        gen_s = dg["eval"]["eval_runtime"] / dg["eval"]["eval_samples"] if dg else None
        epoch_train = n_train * sps
        epoch_total = epoch_train + (val_s or 0) + (save_s or 0)
        trial = startup + SEARCH_EPOCHS * epoch_total
        final_lo = startup + FINAL_EPOCHS_MIN * epoch_total
        final_hi = startup + FINAL_EPOCHS_MAX * epoch_total
        dev = startup + N_DEV[bench] * gen_s if gen_s else None
        # lower bound: every trial stopped after epoch 1 (pruning starts at epoch 1), 4-epoch final
        lo = TRIALS * (startup + epoch_total) + final_lo + (dev or 0)
        hi = TRIALS * trial + final_hi + (dev or 0)
        h = lambda s: s / 3600.0
        cell = {
            "n_train_after_filters": n_train, "n_val": n_val, "n_dev": N_DEV[bench],
            "seconds_per_sample": sps, "seconds_per_step_at_ga": {ga: sps * ga for ga in GA_CHOICES},
            "steps_per_epoch_at_ga": {ga: n_train // ga for ga in GA_CHOICES},
            "epoch_minutes": {"train": epoch_train / 60, "val_eval": (val_s or 0) / 60,
                              "checkpoint_save": (save_s or 0) / 60, "total": epoch_total / 60},
            "startup_minutes": startup / 60,
            "usd_per_epoch": h(epoch_total) * price,
            "trial_3_epochs": {"hours": h(trial), "usd": h(trial) * price, "note": "upper bound (no pruning)"},
            "six_trials": {"hours": h(TRIALS * trial), "usd": h(TRIALS * trial) * price},
            "final_training": {"hours": [h(final_lo), h(final_hi)], "usd": [h(final_lo) * price, h(final_hi) * price],
                               "note": "4..15 epochs (early stopping, patience 3)"},
            "dev_eval": {"hours": h(dev) if dev else None, "usd": h(dev) * price if dev else None,
                         "gen_seconds_per_example": gen_s,
                         "note": "RGAT-arm generation speed used for both arms" if arm == "plain" else ""},
            "cell_total": {"hours": [h(lo), h(hi)], "usd": [h(lo) * price, h(hi) * price]},
        }
        out["cells"]["%s_%s" % (bench, arm)] = cell
        tot["lower_h"] += h(lo)
        tot["upper_h"] += h(hi)
    complete = all("missing" not in c for c in out["cells"].values())
    out["four_cells"] = ({"hours": [tot["lower_h"], tot["upper_h"]],
                          "usd": [tot["lower_h"] * price, tot["upper_h"] * price]} if complete else None)
    return out


def fmt_table(est):
    rows = ["| cell | s/sample | steps/epoch (GA 8/16/32/64) | min/epoch | US$/epoch | trial (3 ep) | 6 trials | final | dev eval | cell total |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in est["cells"].items():
        if c.get("missing"):
            rows.append("| %s | not measured | | | | | | | | |" % name)
            continue
        spe = "/".join(str(c["steps_per_epoch_at_ga"][g]) for g in GA_CHOICES)
        rows.append("| %s | %.3f | %s | %.1f | %.3f | %.2f h / $%.2f | %.1f h / $%.2f | %.1f–%.1f h | %s | %.1f–%.1f h / $%.2f–%.2f |" % (
            name, c["seconds_per_sample"], spe, c["epoch_minutes"]["total"], c["usd_per_epoch"],
            c["trial_3_epochs"]["hours"], c["trial_3_epochs"]["usd"], c["six_trials"]["hours"], c["six_trials"]["usd"],
            c["final_training"]["hours"][0], c["final_training"]["hours"][1],
            ("%.2f h" % c["dev_eval"]["hours"]) if c["dev_eval"]["hours"] else "n/a",
            c["cell_total"]["hours"][0], c["cell_total"]["hours"][1], c["cell_total"]["usd"][0], c["cell_total"]["usd"][1]))
    if est.get("four_cells"):
        f = est["four_cells"]
        rows.append("")
        rows.append("**4 cells: %.1f–%.1f GPU-hours, US$ %.2f–%.2f at US$ %.2f/h** (upper bound: no pruning, 15-epoch finals)."
                    % (f["hours"][0], f["hours"][1], f["usd"][0], f["usd"][1], est["price_usd_per_hour"]))
    return "\n".join(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--price", type=float, default=float(os.environ.get("RUNPOD_GPU_USD_PER_HOUR", "0.74")))
    a = ap.parse_args()
    est = estimate(a.results_dir, a.price)
    print(json.dumps(est, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else str(x))[:4000])
    print()
    print(fmt_table(est))
