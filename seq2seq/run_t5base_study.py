"""t5-base study driver (protocol: PROTOCOLO.md). For each (benchmark, arm) cell, with
arm in {rgat, plain}: a short Optuna search, then one final training from stock T5 with
the best hyperparameters, then scoring on the official dev. Both arms of a benchmark share
the base config, data, search space/budget, seeds and decoding; only
GRAPHIX_MODEL_VARIANT differs.

Stages (each skipped if its output already exists, so re-running the container resumes):
  1. search  -- seq2seq/run_optuna_search.py on the train-carved split (splits/*.json):
                few trials stopped after --search-epochs of the final --final-epochs
                LR schedule (a trial is a prefix of the run it predicts), loss-only eval
                on the validation slice, early stopping + MedianPruner.
  2. final   -- best params, up to --final-epochs with early stopping on validation
                eval_loss (patience --final-patience). Dev is never seen here.
  3. dev     -- generation + exact match / execution on the official dev, loading the
                final weights strictly via GRAPHIX_INIT_STATE_DICT.
Run inside the training container (scripts/run_t5base_study.sh).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
from seq2seq.utils.optuna_callback import _PKGS  # noqa: F401  (adds optuna to sys.path)

import optuna

# Defaults are the container mounts of scripts/run_t5base_study.sh; a pod without
# docker-in-docker (scripts/run_t5base_study_pod.sh) points these at its volume instead.
STUDY_DIR = os.environ.get("GRAPHIX_STUDY_DIR", "/optuna_studies")
RUNS_DIR = os.environ.get("GRAPHIX_RUNS_DIR", "/train_db_id")

SPIDER = "data_all_in/data/output"
SCIB = "data_all_in/data/sciencebenchmark/output"
SPLITS = "data_all_in/data/splits"
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benches", nargs="+", default=["spider", "sciencebenchmark"])
    p.add_argument("--arms", nargs="+", default=["rgat", "plain"])
    p.add_argument("--n-trials", type=int, default=6)
    p.add_argument("--search-epochs", type=int, default=3)
    p.add_argument("--search-patience", type=int, default=1)
    p.add_argument("--startup-trials", type=int, default=2)
    p.add_argument("--warmup-max", type=float, default=0.1)
    p.add_argument("--final-epochs", type=int, default=15)
    p.add_argument("--final-patience", type=int, default=3)
    p.add_argument("--storage-dir", default=STUDY_DIR)
    p.add_argument("--smoke", action="store_true",
                   help="end-to-end check on tiny data: separate *_smoke studies/dirs, 32 train "
                        "examples, 1 search epoch, 2 final epochs, 20 dev examples")
    return p.parse_args()


def base_config(spec, a):
    if not a.smoke:
        return spec["base_config"]
    cfg = json.load(open(spec["base_config"]))
    cfg["max_train_samples"] = 32
    path = f"{STUDY_DIR}/smoke_{Path(spec['base_config']).name}"
    json.dump(cfg, open(path, "w"), indent=2)
    return path


def log(event, **kw):
    line = json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kw})
    print(line, flush=True)
    with open(f"{STUDY_DIR}/t5base_study.jsonl", "a") as f:
        f.write(line + "\n")


def data_env(spec, eval_key, arm):
    (train_ds, train_gp), (eval_ds, eval_gp) = spec["train"], spec[eval_key]
    env = dict(os.environ)
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


def search(bench, arm, spec, a):
    study_name = f"{bench}_t5base_{arm}" + ("_smoke" if a.smoke else "")
    storage = f"sqlite:///{a.storage_dir}/{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True,
                                direction="minimize")
    # A trial still RUNNING at startup was killed with the previous container: fail it,
    # or it would count towards nothing and block nothing but confuse the status.
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
            log("stale_trial_failed", study=study_name, trial=t.number)
    finished = [t for t in study.trials if t.state.is_finished()]
    if len(finished) >= a.n_trials:
        return study
    # Seed from the trial count: a restart with a fixed seed replays the sampler's random
    # phase and duplicates earlier configurations.
    seed = 1 + len(study.trials)
    log("search_start", study=study_name, finished=len(finished), seed=seed)
    rc = run([sys.executable, "seq2seq/run_optuna_search.py",
              "--base-config", base_config(spec, a), "--study-name", study_name,
              "--storage-dir", a.storage_dir, "--out-root", f"{RUNS_DIR}/optuna",
              "--n-trials", str(a.n_trials),
              "--max-epochs", str(a.search_epochs), "--schedule-epochs", str(a.final_epochs),
              "--patience", str(a.search_patience), "--warmup-max", str(a.warmup_max),
              "--startup-trials", str(a.startup_trials), "--warmup-steps", "0",
              "--seed", str(seed),
              "--best-config-out", f"configs/{study_name}_optuna_best.json"],
             data_env(spec, "val", arm), f"{RUNS_DIR}/optuna/{study_name}.driver.log")
    study = optuna.load_study(study_name=study_name, storage=storage)
    log("search_end", study=study_name, rc=rc, n_complete=len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
    return study


def final(bench, arm, spec, study, a):
    out = Path(f"{RUNS_DIR}/study-t5base-{bench}-{arm}" + ("-smoke" if a.smoke else ""))
    # Trainer output goes to out/run: HF refuses a non-empty output_dir without a
    # checkpoint, and out/ already holds the config and log written below.
    run_dir = out / "run"
    if (run_dir / "train_results.json").exists() and (run_dir / "pytorch_model.bin").exists():
        return out
    best = study.best_trial
    cfg = json.load(open(base_config(spec, a)))
    cfg.update(best.params)
    for k in ("eval_steps", "save_steps"):
        cfg.pop(k, None)
    cfg.update({
        "run_name": out.name, "output_dir": str(run_dir), "num_train_epochs": a.final_epochs,
        "evaluation_strategy": "epoch", "save_strategy": "epoch", "save_total_limit": 2,
        "load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
        "greater_is_better": False, "predict_with_generate": False,
        # resume from the last checkpoint if the container was restarted mid-run
        "overwrite_output_dir": False, "report_to": [],
    })
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "final_config.json"
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    env = data_env(spec, "val", arm)
    env.update({"GRAPHIX_EARLY_STOPPING_PATIENCE": str(a.final_patience), "GRAPHIX_LOSS_ONLY_EVAL": "1"})
    log("final_start", cell=out.name, best_trial=best.number, val_loss=best.value, params=best.params)
    rc = run([sys.executable, "seq2seq/run_seq2seq_train.py", str(cfg_path)], env, out / "train.log")
    log("final_end", cell=out.name, rc=rc)
    if rc != 0:
        raise SystemExit(f"final training failed for {out.name} (rc={rc}); see {out}/train.log")
    return out


def dev_eval(arm, spec, final_dir, a):
    out = final_dir / "dev_eval"
    if (out / "run" / "eval_results.json").exists():
        return
    cfg = json.load(open(final_dir / "final_config.json"))
    cfg.update({"run_name": f"{final_dir.name}-dev", "output_dir": str(out / "run"),
                "model_name_or_path": json.load(open(spec["base_config"]))["model_name_or_path"],
                "do_train": False, "do_eval": True, "predict_with_generate": True,
                "overwrite_output_dir": True})
    if a.smoke:
        cfg["max_val_samples"] = 20
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "dev_config.json"
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    env = data_env(spec, "dev", arm)
    env["GRAPHIX_INIT_STATE_DICT"] = str(final_dir / "run" / "pytorch_model.bin")
    log("dev_start", cell=final_dir.name)
    rc = run([sys.executable, "seq2seq/run_seq2seq_train.py", str(cfg_path)], env, out / "eval.log")
    res_path = out / "run" / "eval_results.json"
    res = json.load(open(res_path)) if res_path.exists() else None
    log("dev_end", cell=final_dir.name, rc=rc, results=res)


def main():
    a = parse_args()
    if a.smoke:
        a.n_trials, a.search_epochs, a.final_epochs, a.final_patience = 2, 1, 2, 1
    log("pipeline_start", args=vars(a))
    for bench in a.benches:
        spec = BENCHES[bench]
        for arm in a.arms:
            study = search(bench, arm, spec, a)
            if not [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]:
                log("no_complete_trials", study=study.study_name)
                continue
            final_dir = final(bench, arm, spec, study, a)
            dev_eval(arm, spec, final_dir, a)
    log("pipeline_end")


if __name__ == "__main__":
    main()
