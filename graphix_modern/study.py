"""t5-base study orchestrator on the modern stack (PROTOCOLO.md §2-6): port of
seq2seq/run_t5base_study.py + seq2seq/run_optuna_search.py onto graphix_modern/train.py.

For each (benchmark, arm) cell, arm in {rgat, plain}:
  1. search -- Optuna (TPE, MedianPruner) on the train-carved split: n_trials trials, each
               a fresh training process on the LR schedule of the final run (final_epochs)
               stopped after search_epochs (a trial is a prefix of the run it predicts),
               loss-only validation at every epoch end, early stopping (search_patience),
               pruning from epoch 1; objective = the trial's lowest validation eval_loss.
  2. final  -- the best trial's hyperparameters, from stock t5-base, seed 1, up to
               final_epochs with early stopping on validation eval_loss (final_patience);
               the lowest-eval_loss checkpoint is loaded at the end (load_best_model_at_end).
  3. dev    -- the official dev, once: greedy generation up to 512 tokens, exact match and
               execution, with the final weights loaded strictly (GRAPHIX_INIT_STATE_DICT).
Both arms of a benchmark share base config, data, filters, search space and budget, seeds,
stopping rule and decoding; only GRAPHIX_MODEL_VARIANT differs.

What is the same as the legacy orchestrator (deliberately, line by line): search space,
TPE sampler seed 1 + number of trials already in the study (1 on a fresh study; a restart
does not replay the sampler's random phase), MedianPruner(n_startup_trials=2,
n_warmup_steps=0), a trial killed with the process is marked FAIL at the next start and
counts towards the budget, trial outcome rules (pruned marker -> PRUNED; rc 0 with a best
metric -> COMPLETE; otherwise FAIL), the trial and final configs, the environment variables.

What differs (interface only; PORTABILIDADE.md "orquestrador"):
  - every run goes through scripts/runpod/entry.py -> graphix_modern/train.py: the fp32
    guard fails a run on autocast/TF32/non-"highest" matmul precision/non-fp32 or non-CUDA
    parameters; run_timing.json and run_env.json are written next to each run;
  - configs that set fp16/bf16/tf32/torch_compile are refused (strict FP32);
  - the model is t5-base as safetensors (data_all_in/t5-base-st, PORTABILIDADE.md) and is
    saved as model.safetensors; the dev evaluation loads that file;
  - outputs live under one root ($WS/study by default) instead of three mounts;
  - post-conditions are checked and recorded: the final model must equal the best
    checkpoint bit for bit, and a final run is complete only when that check passed
    (FINAL_DONE); an interrupted final run resumes from its last complete checkpoint; a
    checkpoint left incomplete by an interruption during its save, and a final run
    interrupted before its first complete checkpoint, are moved aside (never deleted);
  - every child gets an explicit environment without MKL_THREADING_LAYER (child_env()).

  python graphix_modern/study.py [--benches spider sciencebenchmark] [--arms rgat plain]
                                 [--root DIR] [--smoke]
Run from the project root inside the a5 image, with scripts/runpod/common.sh sourced
(scripts/runpod_study.sh does both). Re-running resumes: every finished stage is skipped.
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
import optuna  # noqa: E402

WS = Path(os.environ.get("WS", "/workspace"))
ENTRY = "scripts/runpod/entry.py"
MODEL = "data_all_in/t5-base-st"  # t5-base as safetensors (PORTABILIDADE.md)

SPIDER = "data_all_in/data/output"
SCIB = "data_all_in/data/sciencebenchmark/output"
SPLITS = "data_all_in/data/splits"
# identical to seq2seq/run_t5base_study.py BENCHES
BENCHES = {
    "spider": {
        "base_config": "configs/study_t5base_spider.json",
        "max_nodes": "512",
        "train": (f"{SPLITS}/spider_train.json", f"{SPIDER}/graph_pedia_total.bin"),
        "val": (f"{SPLITS}/spider_val.json", f"{SPIDER}/graph_pedia_total.bin"),
        "dev": (f"{SPIDER}/seq2seq_dev_dataset.json", f"{SPIDER}/graph_pedia_total.bin"),
    },
    "sciencebenchmark": {
        "base_config": "configs/study_t5base_sciencebenchmark.json",
        "max_nodes": "1400",
        "train": (f"{SPLITS}/sciencebenchmark_train.json", f"{SCIB}/graph_pedia_train.bin"),
        # validation examples come from train, so their graphs live in the train store
        "val": (f"{SPLITS}/sciencebenchmark_val.json", f"{SCIB}/graph_pedia_train.bin"),
        "dev": (f"{SCIB}/seq2seq_dev_dataset.json", f"{SCIB}/graph_pedia_dev.bin"),
    },
}
STRICT_FP32_KEYS = ("fp16", "bf16", "tf32", "torch_compile", "fp16_full_eval", "bf16_full_eval")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--benches", nargs="+", default=["spider", "sciencebenchmark"], choices=list(BENCHES))
    p.add_argument("--arms", nargs="+", default=["rgat", "plain"], choices=["rgat", "plain"])
    p.add_argument("--n-trials", type=int, default=6)
    p.add_argument("--search-epochs", type=int, default=3)
    p.add_argument("--search-patience", type=int, default=1)
    p.add_argument("--startup-trials", type=int, default=2)
    p.add_argument("--pruner-warmup-steps", type=int, default=0)
    p.add_argument("--warmup-max", type=float, default=0.1)
    p.add_argument("--final-epochs", type=int, default=15)
    p.add_argument("--final-patience", type=int, default=3)
    p.add_argument("--root", default=str(WS / "study"))
    p.add_argument("--smoke", action="store_true",
                   help="end-to-end check on tiny data, as the legacy --smoke: separate root "
                        "(<root>_smoke), 32 train examples, 2 trials of 1 epoch, final of 2 "
                        "epochs (patience 1), 20 dev examples")
    a = p.parse_args(argv)
    if a.smoke:
        a.root = a.root.rstrip("/") + "_smoke"
        a.n_trials, a.search_epochs, a.final_epochs, a.final_patience = 2, 1, 2, 1
    return a


# ---- bookkeeping ------------------------------------------------------------------------------
class Study:
    def __init__(self, a):
        self.a = a
        self.root = Path(a.root)
        for d in ("optuna", "trials", "final", "configs"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self._lock = open(self.root / ".lock", "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit("another orchestrator holds %s/.lock" % self.root)

    def log(self, event, **kw):
        line = json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kw}, default=str)
        print(line, flush=True)
        with open(self.root / "events.jsonl", "a") as f:
            f.write(line + "\n")


ENV_PROBE = r"""
import json, os, platform, subprocess, sys
import torch, transformers, dgl, optuna
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "runpod"))
import gitinfo
try:
    commit = gitinfo.commit(".")
except Exception as e:
    commit = "unknown (%s)" % e
env = {"code_commit": commit, "image_digest": os.environ.get("GRAPHIX_IMAGE_DIGEST"),
       "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
       "cudnn": torch.backends.cudnn.version(), "transformers": transformers.__version__,
       "dgl": dgl.__version__, "optuna": optuna.__version__,
       "precision": "fp32 strict (TF32/autocast/compile off; fp32 guard in scripts/runpod/entry.py)",
       "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    env.update(gpu=p.name, gpu_capability="%d.%d" % (p.major, p.minor), gpu_memory_gb=round(p.total_memory / 2 ** 30, 1))
try:
    env["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                            text=True).strip()
except Exception:
    env["driver"] = None
print(json.dumps(env))
"""


def child_env():
    """Environment of every child process, always passed explicitly (env=...).

    MKL_THREADING_LAYER is removed. mkl-service, when numpy is first imported, calls the C
    setenv() for it: INTEL if libgomp is not loaded yet, GNU otherwise. os.environ does not
    see that call, but a child started without env= inherits it. A parent that imported
    numpy before torch (optuna does) therefore hands MKL_THREADING_LAYER=INTEL to its
    children, and a training process, which imports torch (libgomp) first, then aborts:
    "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1" (TS, 2026-10-06). Every
    validated run (phase A, phase B, TE) started with the variable unset, so mkl chose GNU;
    unsetting it keeps the children identical to those runs whoever launched the study."""
    env = dict(os.environ)
    if env.pop("MKL_THREADING_LAYER", None) is not None:
        print("study: MKL_THREADING_LAYER inherited from the launcher, removed for the runs", flush=True)
    return env


def environment():
    """Versions, GPU and commit, from a child process (the orchestrator does not import torch)."""
    out = subprocess.run([sys.executable, "-c", ENV_PROBE], env=child_env(), capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("environment probe failed:\n" + out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def base_config(S, spec):
    cfg = json.load(open(spec["base_config"]))
    for k in STRICT_FP32_KEYS:
        if cfg.get(k):
            raise SystemExit("%s sets %s: the study runs in strict FP32" % (spec["base_config"], k))
    cfg.update({"model_name_or_path": MODEL, "cache_dir": str(WS / "cache" / "hf" / "transformers")})
    if S.a.smoke:
        cfg["max_train_samples"] = 32
    return cfg


def data_env(spec, eval_key, arm):
    (train_ds, train_gp), (eval_ds, eval_gp) = spec["train"], spec[eval_key]
    env = child_env()
    for k in ("GRAPHIX_ALLOW_TF32", "GRAPHIX_ALLOW_CPU_RGAT", "GRAPHIX_ALLOW_OOM_SKIP", "GRAPHIX_PROFILE"):
        env.pop(k, None)
    env = {k: v for k, v in env.items() if not k.startswith("HARNESS_")}
    env.update({
        "GRAPHIX_TRAIN_DATASET_PATH": train_ds, "GRAPHIX_TRAIN_GRAPH_PEDIA_PATH": train_gp,
        "GRAPHIX_EVAL_DATASET_PATH": eval_ds, "GRAPHIX_EVAL_GRAPH_PEDIA_PATH": eval_gp,
        "GRAPHIX_MAX_GRAPH_NODES": spec["max_nodes"],
        "GRAPHIX_MODEL_VARIANT": arm,
    })
    return env


def run(cmd, env, log_path):
    with open(log_path, "a") as f:
        return subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)


def epoch_settings(cfg, out_dir, run_name, epochs):
    for k in ("eval_steps", "save_steps"):
        cfg.pop(k, None)
    cfg.update({
        "run_name": run_name, "output_dir": str(out_dir), "num_train_epochs": epochs,
        "evaluation_strategy": "epoch", "save_strategy": "epoch", "save_total_limit": 2,
        "load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
        "greater_is_better": False, "predict_with_generate": False, "report_to": [],
    })
    return cfg


def complete_checkpoint(ck):
    return (ck / "trainer_state.json").is_file() and (ck / "model.safetensors").is_file()


def same_weights(a, b):
    from safetensors.torch import load_file
    ta, tb = load_file(str(a)), load_file(str(b))
    return set(ta) == set(tb) and all(ta[k].shape == tb[k].shape and bool((ta[k] == tb[k]).all()) for k in ta)


# ---- 1. search ----------------------------------------------------------------------------------
def sample(trial, warmup_max):
    # PROTOCOLO.md §4, identical to seq2seq/run_optuna_search.py
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 1e-3, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, warmup_max),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [8, 16, 32, 64]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.1]),
    }


def search(S, bench, arm, spec):
    a = S.a
    name = "%s_t5base_%s" % (bench, arm)
    storage = "sqlite:///%s" % (S.root / "optuna" / (name + ".db"))
    study = optuna.create_study(study_name=name, storage=storage, load_if_exists=True, direction="minimize")
    # a trial still RUNNING at startup was killed with the previous process: fail it
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
            S.log("stale_trial_failed", study=name, trial=t.number)
    finished = [t for t in study.trials if t.state.is_finished()]
    if len(finished) >= a.n_trials:
        return study
    seed = 1 + len(study.trials)
    study = optuna.create_study(
        study_name=name, storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=a.startup_trials, n_warmup_steps=a.pruner_warmup_steps))
    S.log("search_start", study=name, finished=len(finished), seed=seed)
    out_root = S.root / "trials" / name
    out_root.mkdir(parents=True, exist_ok=True)
    base = base_config(S, spec)
    done = len(finished)
    while done < a.n_trials:
        trial = study.ask()
        params = sample(trial, a.warmup_max)
        out_dir = out_root / ("trial_%d" % trial.number)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(base)
        cfg.update(params)
        cfg = epoch_settings(cfg, out_dir / "run", "optuna-trial_%d" % trial.number, a.final_epochs)
        cfg["overwrite_output_dir"] = True
        cfg_path = out_dir / "trial_config.json"
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        marker = out_dir / "PRUNED"
        if marker.exists():
            marker.unlink()
        env = data_env(spec, "val", arm)
        env.update({
            "GRAPHIX_EARLY_STOPPING_PATIENCE": str(a.search_patience), "GRAPHIX_LOSS_ONLY_EVAL": "1",
            "GRAPHIX_OPTUNA_STORAGE": storage, "GRAPHIX_OPTUNA_STUDY": name,
            "GRAPHIX_OPTUNA_TRIAL_ID": str(trial._trial_id), "GRAPHIX_OPTUNA_PRUNED_MARKER": str(marker),
            "GRAPHIX_OPTUNA_STARTUP_TRIALS": str(a.startup_trials),
            "GRAPHIX_OPTUNA_WARMUP_STEPS": str(a.pruner_warmup_steps),
        })
        if a.final_epochs > a.search_epochs:
            env["GRAPHIX_STOP_AFTER_EPOCHS"] = str(a.search_epochs)
        S.log("trial_start", study=name, trial=trial.number, params=params)
        t0 = time.time()
        rc = run([sys.executable, ENTRY, str(cfg_path)], env, out_dir / "train.log")
        state_file = out_dir / "run" / "trainer_state.json"
        best, epochs, steps = None, None, None
        if state_file.exists():
            st = json.load(open(state_file))
            best, epochs, steps = st.get("best_metric"), st.get("epoch"), st.get("global_step")
        if marker.exists():
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            outcome = "pruned"
        elif rc == 0 and best is not None:
            study.tell(trial, best)
            outcome = "complete"
        else:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            outcome = "fail(rc=%d)" % rc
        rec = {"trial": trial.number, "outcome": outcome, "best_eval_loss": best, "epochs_run": epochs,
               "global_step": steps, "minutes": round((time.time() - t0) / 60, 1), "params": params}
        with open(out_root / "trials.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        S.log("trial_end", study=name, **rec)
        # weights are the bulk of the disk cost and are not needed after the trial
        run_dir = out_dir / "run"
        if run_dir.exists():
            for p in list(run_dir.glob("checkpoint-*")) + list(run_dir.glob("*.safetensors")) + list(run_dir.glob("*.bin")):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        done += 1
    study = optuna.load_study(study_name=name, storage=storage)
    S.log("search_end", study=name, n_complete=len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        states=[t.state.name for t in study.trials])
    return study


# ---- 2. final -----------------------------------------------------------------------------------
def final(S, bench, arm, spec, study):
    a = S.a
    out = S.root / "final" / ("study-t5base-%s-%s" % (bench, arm))
    run_dir = out / "run"
    done = out / "FINAL_DONE"
    if done.exists():
        return out
    best = study.best_trial
    cfg = dict(base_config(S, spec))
    cfg.update(best.params)
    cfg = epoch_settings(cfg, run_dir, out.name, a.final_epochs)
    cfg["overwrite_output_dir"] = False  # resume from the last checkpoint after an interruption
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "final_config.json"
    if cfg_path.exists() and json.load(open(cfg_path)) != cfg:
        raise SystemExit("%s exists with a different config: refusing to mix runs" % cfg_path)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    if run_dir.exists():
        # A process killed while saving leaves a checkpoint directory without its last file
        # (transformers 4.57 writes trainer_state.json after the model, optimizer, scheduler
        # and RNG states); resuming from it fails ("Can't find a valid checkpoint", TS
        # 2026-10-06). Such directories are moved aside (never deleted) so the run resumes
        # from the last complete checkpoint.
        for ck in sorted(run_dir.glob("checkpoint-*")):
            if not complete_checkpoint(ck):
                aside = out / ("incomplete-%s-%s" % (ck.name, stamp))
                ck.rename(aside)
                S.log("final_incomplete_checkpoint_moved_aside", cell=out.name, checkpoint=ck.name, to=str(aside))
        if not [c for c in run_dir.glob("checkpoint-*")] and any(run_dir.iterdir()):
            aside = out / ("run.aborted-%s" % stamp)
            run_dir.rename(aside)  # interrupted before its first complete checkpoint: nothing to resume
            S.log("final_moved_aside", cell=out.name, to=str(aside))
    env = data_env(spec, "val", arm)
    env.update({"GRAPHIX_EARLY_STOPPING_PATIENCE": str(a.final_patience), "GRAPHIX_LOSS_ONLY_EVAL": "1"})
    resumed = run_dir.exists() and bool(list(run_dir.glob("checkpoint-*")))
    resume_from = max((int(c.name.split("-")[1]) for c in run_dir.glob("checkpoint-*")), default=None) if resumed else None
    S.log("final_start", cell=out.name, best_trial=best.number, val_loss=best.value, params=best.params,
          resume=resumed, resume_from_step=resume_from)
    rc = run([sys.executable, ENTRY, str(cfg_path)], env, out / "train.log")
    if rc != 0:
        S.log("final_end", cell=out.name, rc=rc)
        raise SystemExit("final training failed for %s (rc=%d); see %s/train.log" % (out.name, rc, out))
    st = json.load(open(run_dir / "trainer_state.json"))
    best_ckpt = st.get("best_model_checkpoint")
    post = {"global_step": st["global_step"], "max_steps": st["max_steps"], "epoch": st["epoch"],
            "best_metric": st.get("best_metric"), "best_model_checkpoint": best_ckpt,
            "eval_losses": [(h["epoch"], h["eval_loss"]) for h in st["log_history"] if "eval_loss" in h]}
    ok = best_ckpt is not None and same_weights(run_dir / "model.safetensors", Path(best_ckpt) / "model.safetensors")
    post["final_equals_best_checkpoint"] = ok
    S.log("final_end", cell=out.name, rc=rc, **post)
    if not ok:
        raise SystemExit("final model of %s is not the best checkpoint (%s)" % (out.name, best_ckpt))
    json.dump(post, open(done, "w"), indent=1)
    return out


# ---- 3. dev -------------------------------------------------------------------------------------
def dev_eval(S, arm, spec, final_dir):
    out = final_dir / "dev_eval"
    if (out / "run" / "eval_results.json").exists():
        return
    cfg = json.load(open(final_dir / "final_config.json"))
    cfg.update({"run_name": final_dir.name + "-dev", "output_dir": str(out / "run"), "model_name_or_path": MODEL,
                "do_train": False, "do_eval": True, "predict_with_generate": True, "overwrite_output_dir": True})
    if S.a.smoke:
        cfg["max_val_samples"] = 20
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "dev_config.json"
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    env = data_env(spec, "dev", arm)
    env["GRAPHIX_INIT_STATE_DICT"] = str(final_dir / "run" / "model.safetensors")
    S.log("dev_start", cell=final_dir.name)
    rc = run([sys.executable, ENTRY, str(cfg_path)], env, out / "eval.log")
    res_path = out / "run" / "eval_results.json"
    res = json.load(open(res_path)) if res_path.exists() else None
    S.log("dev_end", cell=final_dir.name, rc=rc, results=res)
    if rc != 0:
        raise SystemExit("dev evaluation failed for %s (rc=%d)" % (final_dir.name, rc))


def main(argv=None):
    a = parse_args(argv)
    S = Study(a)
    S.log("pipeline_start", args=vars(a), environment=environment())
    for bench in a.benches:
        spec = BENCHES[bench]
        for arm in a.arms:
            study = search(S, bench, arm, spec)
            if not [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]:
                S.log("no_complete_trials", study=study.study_name)
                continue
            final_dir = final(S, bench, arm, spec, study)
            dev_eval(S, arm, spec, final_dir)
    S.log("pipeline_end")


if __name__ == "__main__":
    main()
