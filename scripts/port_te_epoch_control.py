#!/usr/bin/env python
"""Test TE: epoch-level training control of graphix_modern/train.py on the modern stack, as
the study orchestrator (graphix_modern/study.py) uses it. Phase A validated the step-level
loop (TRACE, T7, T9, T10); these controls run on transformers 4.57's own code around the
ported 4.17 loop and were not covered:

  evaluation at every epoch end; checkpoint at every epoch end with save_total_limit
  rotation (the best checkpoint kept); load_best_model_at_end (best = lowest eval_loss);
  EarlyStoppingCallback (GRAPHIX_EARLY_STOPPING_PATIENCE); Optuna MedianPruner through
  seq2seq/utils/optuna_callback.py (pruned and not pruned); GRAPHIX_STOP_AFTER_EPOCHS with
  the LR schedule of the full run (a trial is a prefix of the final run); resume of an
  interrupted final-style run from its last checkpoint, with best-model selection across
  the interruption.

Each case runs the real entrypoint (scripts/runpod/entry.py -> graphix_modern/train.py,
strict fp32 guard) on 24 Spider training examples (GA 8 -> 3 optimizer steps per epoch)
and 8 validation examples (loss only). The eval_loss each decision sees is fixed per epoch
by HARNESS_FAKE_EVAL_LOSS (scripts/port_entry_harness.py), so the expected decisions are
exact; the real loss is kept as eval_real_loss and is used to prove the loaded weights
are the best checkpoint's.

  python scripts/port_te_epoch_control.py        (inside the a5 image, cwd = project root)
Writes $WS/outputs/port_tests/TE/TE_report.json; exit != 0 on any failed check.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "runpod"))
import runs as R  # noqa: E402  (BENCHES, data_env, base_config of the phase-B session)

WS = Path(os.environ.get("WS", "/workspace"))
ROOT = WS / "checkpoints" / "port_tests" / "TE"
OUT = WS / "outputs" / "port_tests" / "TE"
N_TRAIN, GA, N_VAL = 24, 8, 8
SPE = N_TRAIN // GA  # optimizer steps per epoch (4.17 semantics: floor)
results, failures = {}, []


def check(case, cond, msg):
    results.setdefault(case, {}).setdefault("checks", []).append({"ok": bool(cond), "check": msg})
    print("  [%s] %s" % ("ok" if cond else "FAIL", msg), flush=True)
    if not cond:
        failures.append("%s: %s" % (case, msg))


def config(name, **over):
    d = ROOT / name / "run"
    c = R.base_config("spider", d)
    c.update({"max_train_samples": N_TRAIN, "max_val_samples": N_VAL, "gradient_accumulation_steps": GA,
              "learning_rate": 1e-4, "warmup_ratio": 0.0, "logging_steps": 1, "logging_first_step": True,
              "do_train": True, "do_eval": True, "predict_with_generate": False,
              "evaluation_strategy": "epoch", "save_strategy": "epoch", "save_total_limit": 2,
              "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False})
    c.update(over)
    return c


def launch(name, cfg, env_extra):
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = OUT / (name + ".json")
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    env = dict(os.environ)
    env.update(R.data_env("spider", "val", "rgat"))
    env.update({"GRAPHIX_LOSS_ONLY_EVAL": "1"})
    env.update(env_extra)
    t0 = time.time()
    with open(OUT / (name + ".log"), "w") as log:
        rc = subprocess.call([sys.executable, "scripts/runpod/entry.py", str(cfg_path)], env=env,
                             stdout=log, stderr=subprocess.STDOUT)
    print("run %s: rc=%d in %.0f s" % (name, rc, time.time() - t0), flush=True)
    return rc


def state(run_dir):
    return json.load(open(Path(run_dir) / "trainer_state.json"))


def evals(st):
    return [h for h in st["log_history"] if "eval_loss" in h]


def tensors(path):
    from safetensors.torch import load_file
    return load_file(str(path))


def same_weights(a, b):
    ta, tb = tensors(a), tensors(b)
    if set(ta) != set(tb):
        return False, "key sets differ (%d vs %d)" % (len(ta), len(tb))
    diff = [k for k in ta if not (ta[k].shape == tb[k].shape and (ta[k] == tb[k]).all())]
    return not diff, "%d of %d tensors differ" % (len(diff), len(ta))


def ckpts(run_dir):
    return sorted(int(p.name.split("-")[1]) for p in Path(run_dir).glob("checkpoint-*"))


def final_eval(run_dir):
    return json.load(open(Path(run_dir) / "eval_results.json"))


# ---- C1: eval/save per epoch, rotation, early stopping, load best ----------------------------
def case_best_early_stop():
    case = "C1_best_earlystop"
    print("== " + case)
    # epoch losses 3,1,2,4,5 with patience 2: best = epoch 2; epochs 3,4 do not improve ->
    # stop after epoch 4 (of 5). Best checkpoint = step 2*SPE, kept through rotation.
    d = ROOT / case / "run"
    rc = launch(case, config(case, num_train_epochs=5),
                {"HARNESS_FAKE_EVAL_LOSS": "3,1,2,4,5", "GRAPHIX_EARLY_STOPPING_PATIENCE": "2"})
    check(case, rc == 0, "run exits 0")
    if rc:
        return
    st = state(d)
    ev = evals(st)
    steps = [h["step"] for h in ev]
    check(case, steps[:4] == [SPE, 2 * SPE, 3 * SPE, 4 * SPE],
          "evaluated at the end of epochs 1-4 (steps %s)" % steps[:4])
    check(case, st["global_step"] == 4 * SPE, "early stopping after epoch 4: global_step %d == %d"
          % (st["global_step"], 4 * SPE))
    check(case, st["best_metric"] == 1.0, "best_metric == 1.0 (epoch 2)")
    check(case, st["best_model_checkpoint"].endswith("checkpoint-%d" % (2 * SPE)),
          "best_model_checkpoint is checkpoint-%d (%s)" % (2 * SPE, st["best_model_checkpoint"]))
    check(case, ckpts(d) == [2 * SPE, 4 * SPE], "rotation keeps best + last: %s" % ckpts(d))
    ok, why = same_weights(d / "model.safetensors", d / ("checkpoint-%d" % (2 * SPE)) / "model.safetensors")
    check(case, ok, "final model == best checkpoint weights (%s)" % why)
    ok2, why2 = same_weights(d / "model.safetensors", d / ("checkpoint-%d" % (4 * SPE)) / "model.safetensors")
    check(case, not ok2, "final model != last checkpoint weights (best was really loaded; %s)" % why2)
    real2 = ev[1]["eval_real_loss"]
    fe = final_eval(d)
    rel = abs(fe["eval_real_loss"] - real2) / real2
    check(case, rel < 1e-5, "post-training real eval loss %.6f == epoch-2 real loss %.6f (rel %.1e)"
          % (fe["eval_real_loss"], real2, rel))
    results[case].update({"eval_steps": steps, "global_step": st["global_step"], "checkpoints": ckpts(d),
                          "real_losses": [h["eval_real_loss"] for h in ev], "final_real_eval_loss": fe["eval_real_loss"]})


# ---- C2: a trial (pruning callback attached, not pruned), stop after 2 of 15 epochs -------
def make_study(name, values):
    import optuna
    storage = "sqlite:///%s" % (ROOT / (name + ".db"))
    if (ROOT / (name + ".db")).exists():
        (ROOT / (name + ".db")).unlink()
    study = optuna.create_study(study_name=name, storage=storage, direction="minimize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0))
    for v in values:  # finished trials reporting one value per epoch
        t = study.ask()
        for i, x in enumerate(v):
            t.report(x, i)
        study.tell(t, min(v))
    trial = study.ask()
    return storage, trial._trial_id, trial.number


def trial_env(case, storage, trial_id):
    marker = ROOT / case / "PRUNED"
    if marker.exists():
        marker.unlink()
    return marker, {"GRAPHIX_OPTUNA_STORAGE": storage, "GRAPHIX_OPTUNA_STUDY": case,
                    "GRAPHIX_OPTUNA_TRIAL_ID": str(trial_id), "GRAPHIX_OPTUNA_PRUNED_MARKER": str(marker),
                    "GRAPHIX_OPTUNA_STARTUP_TRIALS": "2", "GRAPHIX_OPTUNA_WARMUP_STEPS": "0"}


def expected_lr(lr, warmup_ratio, total, step):
    warm = math.ceil(warmup_ratio * total)
    f = step / max(1, warm) if step < warm else max(0.0, (total - step) / max(1, total - warm))
    return lr * f


def case_trial_not_pruned():
    case = "C2_trial_stop_after"
    print("== " + case)
    (ROOT / case).mkdir(parents=True, exist_ok=True)
    storage, tid, _ = make_study(case, [[2.0, 1.9], [2.2, 2.1]])
    marker, env = trial_env(case, storage, tid)
    env.update({"HARNESS_FAKE_EVAL_LOSS": "1.0,0.9", "GRAPHIX_EARLY_STOPPING_PATIENCE": "1",
                "GRAPHIX_STOP_AFTER_EPOCHS": "2"})
    d = ROOT / case / "run"
    lr, wr, epochs = 3e-4, 0.1, 15
    rc = launch(case, config(case, num_train_epochs=epochs, learning_rate=lr, warmup_ratio=wr), env)
    check(case, rc == 0, "run exits 0")
    if rc:
        return
    st = state(d)
    check(case, st["global_step"] == 2 * SPE, "stopped after epoch 2 of %d: global_step %d" % (epochs, st["global_step"]))
    check(case, st["max_steps"] == epochs * SPE, "schedule length is the full run's: max_steps %d == %d"
          % (st["max_steps"], epochs * SPE))
    lrs = {h["step"]: h["learning_rate"] for h in st["log_history"] if "learning_rate" in h and "loss" in h}
    exp = {s: expected_lr(lr, wr, epochs * SPE, s) for s in lrs}
    worst = max(abs(lrs[s] - exp[s]) / lr for s in lrs)
    check(case, worst < 1e-9, "LR per step = linear warmup/decay over %d steps (max rel err %.1e)" % (epochs * SPE, worst))
    check(case, not marker.exists(), "not pruned (values below the median)")
    import optuna
    t = [t for t in optuna.load_study(study_name=case, storage=storage).trials if t._trial_id == tid][0]
    check(case, t.intermediate_values == {0: 1.0, 1: 0.9},
          "reported one value per epoch to the study: %s" % t.intermediate_values)
    results[case].update({"global_step": st["global_step"], "lr": lrs, "intermediate": t.intermediate_values})


# ---- C3: a trial pruned at its first evaluation --------------------------------------------
def case_trial_pruned():
    case = "C3_trial_pruned"
    print("== " + case)
    (ROOT / case).mkdir(parents=True, exist_ok=True)
    storage, tid, _ = make_study(case, [[2.0, 1.9], [2.2, 2.1]])
    marker, env = trial_env(case, storage, tid)
    env.update({"HARNESS_FAKE_EVAL_LOSS": "9,9,9", "GRAPHIX_EARLY_STOPPING_PATIENCE": "1",
                "GRAPHIX_STOP_AFTER_EPOCHS": "3"})
    d = ROOT / case / "run"
    rc = launch(case, config(case, num_train_epochs=15), env)
    check(case, rc == 0, "run exits 0 (pruning stops training cleanly)")
    if rc:
        return
    st = state(d)
    check(case, marker.exists(), "pruned marker written")
    check(case, marker.exists() and json.load(open(marker))["step"] == 1, "pruned at the first report (epoch 1)")
    check(case, st["global_step"] == SPE, "training stopped after epoch 1: global_step %d" % st["global_step"])
    # the driver's side (as in seq2seq/run_optuna_search.py): marker -> tell PRUNED
    import optuna
    study = optuna.load_study(study_name=case, storage=storage)
    trial = [t for t in study.trials if t._trial_id == tid][0]
    if marker.exists():
        study.tell(trial.number, state=optuna.trial.TrialState.PRUNED)
    trial = [t for t in optuna.load_study(study_name=case, storage=storage).trials if t._trial_id == tid][0]
    check(case, trial.state == optuna.trial.TrialState.PRUNED, "trial state in the study is %s" % trial.state.name)
    check(case, trial.intermediate_values == {0: 9.0}, "one value reported before pruning: %s" % trial.intermediate_values)
    results[case].update({"global_step": st["global_step"], "state": trial.state.name})


# ---- C5: final-style run that never stops early ends exactly at epoch 15 ---------------------
def case_full_15():
    case = "C5_full_15_epochs"
    print("== " + case)
    epochs, lr, wr = 15, 3e-4, 0.1
    d = ROOT / case / "run"
    fake = ",".join("%.1f" % (20 - e) for e in range(1, epochs + 1))  # strictly improving
    rc = launch(case, config(case, num_train_epochs=epochs, learning_rate=lr, warmup_ratio=wr),
                {"HARNESS_FAKE_EVAL_LOSS": fake, "GRAPHIX_EARLY_STOPPING_PATIENCE": "3"})
    check(case, rc == 0, "run exits 0")
    if rc:
        return
    st = state(d)
    check(case, st["global_step"] == epochs * SPE == st["max_steps"],
          "ended exactly at epoch %d: global_step %d == max_steps %d" % (epochs, st["global_step"], st["max_steps"]))
    check(case, abs(st["epoch"] - epochs) < 1e-9, "state.epoch == %s" % st["epoch"])
    ev_steps = [h["step"] for h in evals(st)]
    check(case, ev_steps[:epochs] == [SPE * e for e in range(1, epochs + 1)], "one evaluation per epoch, 15 in all")
    lrs = {h["step"]: h["learning_rate"] for h in st["log_history"] if "learning_rate" in h and "loss" in h}
    exp = {s: expected_lr(lr, wr, epochs * SPE, s) for s in lrs}
    worst = max(abs(lrs[s] - exp[s]) / lr for s in lrs)
    check(case, worst < 1e-9 and lrs[epochs * SPE] == 0.0,
          "LR schedule exact (max rel err %.1e), LR at the last step %g" % (worst, lrs[epochs * SPE]))
    check(case, st["best_model_checkpoint"].endswith("checkpoint-%d" % (epochs * SPE)), "best = last checkpoint")
    check(case, ckpts(d) == [epochs * SPE] or ckpts(d)[-1] == epochs * SPE, "checkpoints kept: %s" % ckpts(d))
    check(case, len(ckpts(d)) <= 2, "save_total_limit 2 respected over 15 saves")
    ok, why = same_weights(d / "model.safetensors", d / ("checkpoint-%d" % (epochs * SPE)) / "model.safetensors")
    check(case, ok, "final model == best checkpoint weights (%s)" % why)
    results[case].update({"global_step": st["global_step"], "eval_steps": ev_steps, "checkpoints": ckpts(d)})


# ---- C4: final-style run interrupted after epoch 2, resumed -----------------------------------
def case_resume():
    case = "C4_resume"
    print("== " + case)
    d = ROOT / case / "run"
    cfg = config(case, num_train_epochs=4)
    fake = {"HARNESS_FAKE_EVAL_LOSS": "3,1,2,4", "GRAPHIX_EARLY_STOPPING_PATIENCE": "3"}
    rc = launch(case + "_a", cfg, {**fake, "HARNESS_STOP_AT_STEP": str(2 * SPE)})
    check(case, rc == 0, "first part exits 0 (interrupted after step %d)" % (2 * SPE))
    if rc:
        return
    check(case, (d / ("checkpoint-%d" % (2 * SPE))).is_dir(), "checkpoint-%d written" % (2 * SPE))
    shutil.copy(d / "run_timing.json", ROOT / case / "timing_a.json")
    cfg["overwrite_output_dir"] = False  # as the orchestrator's final run: resume if a checkpoint exists
    rc = launch(case + "_b", cfg, fake)
    check(case, rc == 0, "resumed part exits 0")
    if rc:
        return
    st = state(d)
    steps_b = [s for s, _ in json.load(open(d / "run_timing.json"))["step_end"]]
    check(case, steps_b and steps_b[0] == 2 * SPE + 1 and steps_b[-1] == 4 * SPE,
          "the resumed process ran only steps %d..%d (%s..%s)" % (2 * SPE + 1, 4 * SPE, steps_b[:1], steps_b[-1:]))
    ev = evals(st)
    ev_steps = sorted({h["step"] for h in ev})
    check(case, ev_steps[:4] == [SPE, 2 * SPE, 3 * SPE, 4 * SPE], "eval history spans the interruption: %s" % ev_steps)
    check(case, st["best_model_checkpoint"].endswith("checkpoint-%d" % (2 * SPE)),
          "best = checkpoint-%d, written before the interruption (%s)" % (2 * SPE, st["best_model_checkpoint"]))
    ok, why = same_weights(d / "model.safetensors", d / ("checkpoint-%d" % (2 * SPE)) / "model.safetensors")
    check(case, ok, "final model == best checkpoint weights (%s)" % why)
    real2 = [h for h in ev if h["step"] == 2 * SPE][0]["eval_real_loss"]
    fe = final_eval(d)
    rel = abs(fe["eval_real_loss"] - real2) / real2
    check(case, rel < 1e-5, "post-training real eval loss == epoch-2 real loss (rel %.1e)" % rel)
    results[case].update({"steps_after_resume": steps_b, "eval_steps": ev_steps,
                          "best": st["best_model_checkpoint"]})


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    t0 = time.time()
    for f in (case_best_early_stop, case_trial_not_pruned, case_trial_pruned, case_resume, case_full_15):
        f()
    rep = {"test": "TE epoch-level control", "passed": not failures, "failures": failures,
           "minutes": round((time.time() - t0) / 60, 1), "cases": results,
           "n_train": N_TRAIN, "ga": GA, "n_val": N_VAL}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT / "TE_report.json", "w"), indent=1, default=str)
    print("\nTE %s (%d failed checks) in %.1f min -> %s" % ("PASSED" if not failures else "FAILED", len(failures),
                                                           rep["minutes"], OUT / "TE_report.json"))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
