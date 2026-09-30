#!/usr/bin/env python
"""Runs of a phase-B pod session (docs/RUNPOD_PHASE_B.md). Each run writes its config, runs
graphix_modern/train.py through scripts/runpod/entry.py (fp32 guard, timing), then checks
post-conditions and exits != 0 if any fails.

  python scripts/runpod/runs.py RUN            (cwd = project root; env from common.sh)

RUN is one of:
  smoke_train   Spider/RGAT, 48 examples, GA 8 (6 optimizer steps planned), stopped after
                step 3 with a checkpoint (harness HARNESS_STOP_AT_STEP, as in T7)
  smoke_resume  the same run resumed from checkpoint-3 to step 6 (weights, Adafactor,
                scheduler, RNG, data position), final model saved
  smoke_eval    dev evaluation of the smoke model: 16 Spider dev examples, generation +
                exact match + execution (test-suite evaluator), as the study's dev_eval
  steps50       T10's modern run replayed exactly (same config, initial weights, dropout
                neutralized): 50 steps; mechanics must equal the 1070 reference exactly
                (steps, LR per step, order of the 406 items); the loss is compared with the
                T10 same-stack envelope (reported, not a gate: GPU runs are not bitwise
                reproducible and this is a different GPU)
  probe_<bench>_<arm>    real training (dropout on) on the study's train split, GA 8,
                PROBE_STEPS optimizer steps (default 250), then the study's per-epoch
                validation (loss only, full val split). Throughput runs: profiling OFF.
  profile_<bench>_rgat   PROFILE_STEPS (default 40) steps with GRAPHIX_PROFILE=1 (CUDA sync
                at every section boundary): per-section time shares, not throughput
  devgen_<bench>         dev generation speed: DEVGEN_N (default 32) dev examples with the
                probe model (full metric), for the dev-eval part of the cost estimate
bench = spider | sciencebenchmark, arm = rgat | plain.

Study semantics are untouched: base configs are configs/study_t5base_*.json (only the run
length, GA 8, logging and output paths are set here), data/filters/eval exactly as
seq2seq/run_t5base_study.py builds them.
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

WS = Path(os.environ.get("WS", "/workspace"))
SESSION = os.environ.get("SESSION", "phase_b_s1")
SCKPT = WS / "checkpoints" / SESSION
SOUT = WS / "outputs" / SESSION
REF1070 = Path("data_all_in/data/port_tests/T10")   # uploaded T10 references (1070)
T10_INIT = REF1070 / "init_state_dict.bin"
PROBE_STEPS = int(os.environ.get("PROBE_STEPS", "250"))
PROFILE_STEPS = int(os.environ.get("PROFILE_STEPS", "40"))
DEVGEN_N = int(os.environ.get("DEVGEN_N", "32"))

SPIDER = "data_all_in/data/output"
SCIB = "data_all_in/data/sciencebenchmark/output"
SPLITS = "data_all_in/data/splits"
# identical to seq2seq/run_t5base_study.py BENCHES
BENCHES = {
    "spider": {"base_config": "configs/study_t5base_spider.json", "max_nodes": "512",
               "train": (f"{SPLITS}/spider_train.json", f"{SPIDER}/graph_pedia_total.bin"),
               "val": (f"{SPLITS}/spider_val.json", f"{SPIDER}/graph_pedia_total.bin"),
               "dev": (f"{SPIDER}/seq2seq_dev_dataset.json", f"{SPIDER}/graph_pedia_total.bin")},
    "sciencebenchmark": {"base_config": "configs/study_t5base_sciencebenchmark.json", "max_nodes": "1400",
                         "train": (f"{SPLITS}/sciencebenchmark_train.json", f"{SCIB}/graph_pedia_train.bin"),
                         "val": (f"{SPLITS}/sciencebenchmark_val.json", f"{SCIB}/graph_pedia_train.bin"),
                         "dev": (f"{SCIB}/seq2seq_dev_dataset.json", f"{SCIB}/graph_pedia_dev.bin")},
}


class CheckFailed(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise CheckFailed(msg)


def data_env(bench, eval_key, arm):
    spec = BENCHES[bench]
    (tr_ds, tr_gp), (ev_ds, ev_gp) = spec["train"], spec[eval_key]
    return {"GRAPHIX_TRAIN_DATASET_PATH": tr_ds, "GRAPHIX_TRAIN_GRAPH_PEDIA_PATH": tr_gp,
            "GRAPHIX_EVAL_DATASET_PATH": ev_ds, "GRAPHIX_EVAL_GRAPH_PEDIA_PATH": ev_gp,
            "GRAPHIX_MAX_GRAPH_NODES": spec["max_nodes"], "GRAPHIX_MODEL_VARIANT": arm}


def base_config(bench, run_dir):
    c = json.load(open(BENCHES[bench]["base_config"]))
    for k in ("eval_steps", "save_steps"):
        c.pop(k, None)
    c.update({"model_name_or_path": "data_all_in/t5-base-st",  # t5-base as safetensors (PORTABILIDADE.md)
              "cache_dir": str(WS / "cache" / "hf" / "transformers"), "output_dir": str(run_dir),
              "overwrite_output_dir": True, "report_to": [], "load_best_model_at_end": False,
              "evaluation_strategy": "no", "save_strategy": "no", "fp16": False})
    return c


def launch(name, cfg, env_extra):
    run_dir = Path(cfg["output_dir"])
    cfg_dir = SOUT / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / (name + ".json")
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    env = dict(os.environ)
    env.update(env_extra)
    json.dump(env_extra, open(cfg_dir / (name + ".env.json"), "w"), indent=1)
    log_path = SOUT / "logs" / (name + ".train.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("run %s -> %s (log %s)" % (name, run_dir, log_path), flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.call([sys.executable, "scripts/runpod/entry.py", str(cfg_path)], env=env,
                             stdout=log, stderr=subprocess.STDOUT)
    print("  rc=%d in %.1f s" % (rc, time.time() - t0), flush=True)
    if rc != 0:
        os.system("tail -30 %s" % log_path)
        raise CheckFailed("%s exited with rc=%d" % (name, rc))
    return run_dir


def jload(p):
    return json.load(open(str(p)))


def losses(run_dir):
    st = jload(Path(run_dir) / "trainer_state.json")
    return st, {h["step"]: h for h in st["log_history"] if "loss" in h}


def check_finite(run_dir):
    st, L = losses(run_dir)
    need(L, "no loss logged in %s" % run_dir)
    need(all(math.isfinite(h["loss"]) for h in L.values()), "non-finite loss in %s" % run_dir)
    return st, L


# --- smoke ---------------------------------------------------------------------------------
def smoke_train():
    d = SCKPT / "smoke" / "run"
    (SOUT / "results").mkdir(parents=True, exist_ok=True)
    c = base_config("spider", d)
    c.update({"max_train_samples": 48, "gradient_accumulation_steps": 8, "num_train_epochs": 1,
              "logging_steps": 1, "save_strategy": "steps", "save_steps": 3, "save_total_limit": 2,
              "do_eval": False, "predict_with_generate": False})
    launch("smoke_train", c, {**data_env("spider", "val", "rgat"), "HARNESS_STOP_AT_STEP": "3"})
    ck = d / "checkpoint-3"
    for f in ("optimizer.pt", "scheduler.pt", "trainer_state.json", "model.safetensors"):
        need((ck / f).exists(), "smoke: %s missing in checkpoint-3" % f)
    need(any(p.name.startswith("rng_state") for p in ck.iterdir()), "smoke: rng_state missing")
    st = jload(ck / "trainer_state.json")
    need(st["global_step"] == 3, "smoke: checkpoint global_step %s != 3" % st["global_step"])
    check_finite(d)
    t = jload(d / "run_timing.json")
    step3 = [ts for s, ts in t["step_end"] if s == 3]
    saves = [ts for s, ts in t.get("save_end", []) if s == 3]
    need(step3 and saves, "smoke: no step-3 / save timing recorded")
    json.dump(t, open(SOUT / "results" / "smoke_train_timing.json", "w"))
    return {"checkpoint": str(ck), "checkpoint_bytes": sum(p.stat().st_size for p in ck.iterdir()),
            "checkpoint_save_s": saves[0] - step3[0]}


def smoke_resume():
    d = SCKPT / "smoke" / "run"
    c = jload(SOUT / "configs" / "smoke_train.json")
    c["overwrite_output_dir"] = False
    launch("smoke_resume", c, data_env("spider", "val", "rgat"))
    st, L = check_finite(d)
    need(st["global_step"] == 6, "smoke resume: global_step %s != 6" % st["global_step"])
    need(set(range(4, 7)) <= set(L), "smoke resume: steps 4-6 not logged")
    # proof of resuming: this process executed only steps 4-6 (its own step timings), not 1-6
    steps_here = [s for s, _ in jload(d / "run_timing.json")["step_end"]]
    need(steps_here and steps_here[0] == 4 and steps_here[-1] == 6,
         "smoke resume: steps executed by the resumed process were %s, expected 4..6" % steps_here)
    need((d / "model.safetensors").exists(), "smoke resume: final model not saved")
    tr = jload(d / "train_results.json")
    return {"global_step": st["global_step"], "steps_executed_after_resume": steps_here,
            "train_peak_vram_gb": tr.get("train_peak_vram_gb"),
            "loss_steps_4_6": [L[s]["loss"] for s in (4, 5, 6)]}


def smoke_eval():
    src = SCKPT / "smoke" / "run" / "model.safetensors"
    d = SCKPT / "smoke_eval" / "run"
    c = base_config("spider", d)
    c.update({"do_train": False, "do_eval": True, "predict_with_generate": True, "max_val_samples": 16})
    launch("smoke_eval", c, {**data_env("spider", "dev", "rgat"), "GRAPHIX_INIT_STATE_DICT": str(src)})
    ev = jload(d / "eval_results.json")
    need(ev.get("eval_samples") == 16, "smoke eval: eval_samples %s != 16" % ev.get("eval_samples"))
    metric_keys = [k for k in ev if "exact" in k or "exec" in k]
    need(metric_keys, "smoke eval: no exact/exec metric in eval_results.json: %s" % sorted(ev))
    need((d / "predictions_eval_None.json").exists() or any(d.glob("predictions*")),
         "smoke eval: no predictions file")
    return {k: ev[k] for k in sorted(ev)}


# --- 50 steps: T10 replayed ------------------------------------------------------------------
def steps50():
    d = SCKPT / "steps50" / "run"
    c = jload(REF1070 / "modern" / "config.json")
    c.update({"output_dir": str(d), "cache_dir": str(WS / "cache" / "hf" / "transformers")})
    order_log = SCKPT / "steps50" / "order.json"
    env = {"GRAPHIX_TRAIN_DATASET_PATH": f"{SPLITS}/spider_train.json",
           "GRAPHIX_TRAIN_GRAPH_PEDIA_PATH": f"{SPIDER}/graph_pedia_total.bin",
           "GRAPHIX_EVAL_DATASET_PATH": f"{SPLITS}/spider_val.json",
           "GRAPHIX_EVAL_GRAPH_PEDIA_PATH": f"{SPIDER}/graph_pedia_total.bin",
           "GRAPHIX_MAX_GRAPH_NODES": "512", "GRAPHIX_INIT_STATE_DICT": str(T10_INIT),
           "HARNESS_NO_DROPOUT": "1", "HARNESS_ORDER_LOG": str(order_log)}
    launch("steps50", c, env)
    st, L = check_finite(d)
    ref_st = jload(REF1070 / "modern" / "run" / "trainer_state.json")
    ref_L = {h["step"]: h for h in ref_st["log_history"] if "loss" in h}
    need(st["global_step"] == 50 == ref_st["global_step"], "steps50: global_step %s" % st["global_step"])
    need(sorted(L) == sorted(ref_L), "steps50: logged steps differ from the 1070 reference")
    lr_diff = max(abs(L[s]["learning_rate"] - ref_L[s]["learning_rate"]) for s in L)
    need(lr_diff == 0.0, "steps50: LR differs from the 1070 reference (max %g)" % lr_diff)
    order = jload(order_log)
    need(order == jload(REF1070 / "modern" / "order.json"), "steps50: item order differs from the 1070 reference")
    # loss vs the T10 envelope (report only)
    runs = ["legacy", "legacy2", "legacy3", "modern", "modern2", "modern3"]
    ref = {r: {h["step"]: h["loss"] for h in jload(REF1070 / r / "run" / "trainer_state.json")["log_history"]
               if "loss" in h} for r in runs}
    mine = {s: L[s]["loss"] for s in L}

    def pair(x, y):
        rel = [abs(x[s] - y[s]) / abs(x[s]) for s in sorted(x)]
        return {"mean_rel": sum(rel) / len(rel), "max_rel": max(rel),
                "rmse": (sum((x[s] - y[s]) ** 2 for s in x) / len(x)) ** 0.5,
                "first_divergence_step": next((i + 1 for i, r in enumerate(rel) if r > 1e-4), len(rel) + 1)}
    t10 = jload(REF1070 / "T10_report.json")
    env_max = {m: t10["loss_gate"][m]["same_stack_range"][1] for m in ("mean_rel", "max_rel", "rmse")}
    pairs = {r: pair(ref[r], mine) for r in runs}
    med = {m: sorted(p[m] for p in pairs.values())[len(pairs) // 2 - 1: len(pairs) // 2 + 1] for m in env_max}
    med = {m: sum(v) / 2 for m, v in med.items()}
    tr = jload(d / "train_results.json")
    return {"mechanics_identical_to_1070": True, "lr_max_abs_diff": lr_diff,
            "loss_vs_1070_runs": pairs, "median_vs_1070": med, "t10_same_stack_envelope_max": env_max,
            "loss_within_t10_envelope": all(med[m] <= env_max[m] for m in env_max),
            "train_runtime_s": tr.get("train_runtime"), "train_peak_vram_gb": tr.get("train_peak_vram_gb")}


# --- probe / profile / devgen ------------------------------------------------------------------
def probe(bench, arm):
    name = "probe_%s_%s" % (bench, arm)
    d = SCKPT / name / "run"
    c = base_config(bench, d)
    c.update({"gradient_accumulation_steps": 8, "max_steps": PROBE_STEPS, "warmup_ratio": 0.1,
              "logging_steps": 10, "logging_first_step": True, "do_train": True, "do_eval": True,
              "predict_with_generate": False})
    launch(name, c, {**data_env(bench, "val", arm), "GRAPHIX_LOSS_ONLY_EVAL": "1"})
    st, L = check_finite(d)
    need(st["global_step"] == PROBE_STEPS, "%s: global_step %s != %d" % (name, st["global_step"], PROBE_STEPS))
    ev = jload(d / "eval_results.json")
    need(math.isfinite(ev.get("eval_loss", float("nan"))), "%s: eval_loss missing/non-finite" % name)
    return {"train": jload(d / "train_results.json"), "eval": ev, "timing": jload(d / "run_timing.json")}


def profile(bench):
    name = "profile_%s_rgat" % bench
    d = SCKPT / name / "run"
    c = base_config(bench, d)
    c.update({"gradient_accumulation_steps": 8, "max_steps": PROFILE_STEPS, "warmup_ratio": 0.1,
              "logging_steps": 10, "do_train": True, "do_eval": False, "predict_with_generate": False})
    launch(name, c, {**data_env(bench, "val", "rgat"), "GRAPHIX_PROFILE": "1"})
    check_finite(d)
    need((d / "profile.json").exists(), "%s: profile.json missing" % name)
    return {"profile": jload(d / "profile.json"), "timing": jload(d / "run_timing.json")}


def devgen(bench):
    name = "devgen_%s" % bench
    src = SCKPT / ("probe_%s_rgat" % bench) / "run" / "model.safetensors"
    d = SCKPT / name / "run"
    c = base_config(bench, d)
    c.update({"do_train": False, "do_eval": True, "predict_with_generate": True, "max_val_samples": DEVGEN_N})
    launch(name, c, {**data_env(bench, "dev", "rgat"), "GRAPHIX_INIT_STATE_DICT": str(src)})
    ev = jload(d / "eval_results.json")
    need(ev.get("eval_samples") == DEVGEN_N, "%s: eval_samples %s" % (name, ev.get("eval_samples")))
    return {"eval": ev, "timing": jload(d / "run_timing.json")}


def main():
    run = sys.argv[1]
    if run in ("smoke_train", "smoke_resume", "smoke_eval", "steps50"):
        res = globals()[run]()
    elif run.startswith("probe_"):
        bench, arm = run[len("probe_"):].rsplit("_", 1)
        res = probe(bench, arm)
    elif run.startswith("profile_"):
        res = profile(run[len("profile_"):].rsplit("_", 1)[0])
    elif run.startswith("devgen_"):
        res = devgen(run[len("devgen_"):])
    else:
        raise SystemExit("unknown run %s" % run)
    (SOUT / "results").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(SOUT / "results" / (run + ".json"), "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:3000])
    print(run, "OK")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as e:
        print("CHECK FAILED:", e, flush=True)
        sys.exit(1)
