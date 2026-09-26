"""Optuna hyperparameter search over the existing training script.

Each trial is a fresh `run_seq2seq_train.py` subprocess (avoids GPU-memory/global-state
leaks between trials) driven with optuna's ask/tell API. Objective: best dev eval_loss,
with per-epoch early stopping and MedianPruner across trials. Run inside the training
container (see scripts/run_optuna.sh).

NOTE: the objective is dev eval_loss and dev is also the set the thesis reports on, so
the best trial's dev numbers are optimistically biased. The best hyperparameters are
written to a config for one clean final run; state the selection-on-dev caveat when
reporting.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
from seq2seq.utils.optuna_callback import _PKGS  # noqa: F401  (adds optuna to sys.path)

import optuna


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", default="configs/train_sciencebenchmark_t5small_rgat.json")
    p.add_argument("--study-name", default="scibench_t5small_rgat")
    p.add_argument("--storage-dir", default="/optuna_studies")
    p.add_argument("--out-root", default="/train_db_id/optuna")
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--timeout-hours", type=float, default=None)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3, help="early stopping patience, in epochs")
    p.add_argument("--startup-trials", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--best-config-out", default="configs/train_sciencebenchmark_t5small_rgat_optuna_best.json")
    return p.parse_args()


def sample(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 1e-3, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [8, 16, 32, 64]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.1]),
    }


def trial_config(base, params, out_dir, max_epochs):
    cfg = dict(base)
    cfg.update(params)
    for k in ("eval_steps", "save_steps"):
        cfg.pop(k, None)
    cfg.update({
        "run_name": f"optuna-{Path(out_dir).name}",
        "output_dir": str(out_dir),
        "num_train_epochs": max_epochs,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "predict_with_generate": False,
        "overwrite_output_dir": True,
        "report_to": [],
    })
    return cfg


def main():
    a = parse_args()
    base = json.load(open(a.base_config))
    storage = f"sqlite:///{a.storage_dir}/{a.study_name}.db"
    study = optuna.create_study(
        study_name=a.study_name, storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=a.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=a.startup_trials, n_warmup_steps=a.warmup_steps),
    )
    out_root = Path(a.out_root) / a.study_name
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "trials.jsonl"
    deadline = time.time() + a.timeout_hours * 3600 if a.timeout_hours else None

    done = len([t for t in study.trials if t.state.is_finished()])
    while done < a.n_trials and (deadline is None or time.time() < deadline):
        trial = study.ask()
        params = sample(trial)
        out_dir = out_root / f"trial_{trial.number}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / "trial_config.json"
        json.dump(trial_config(base, params, out_dir / "run", a.max_epochs), open(cfg_path, "w"), indent=2)
        marker = out_dir / "PRUNED"
        if marker.exists():
            marker.unlink()

        env = dict(os.environ)
        env.update({
            "GRAPHIX_EARLY_STOPPING_PATIENCE": str(a.patience),
            "GRAPHIX_LOSS_ONLY_EVAL": "1",
            "GRAPHIX_OPTUNA_STORAGE": storage,
            "GRAPHIX_OPTUNA_STUDY": a.study_name,
            "GRAPHIX_OPTUNA_TRIAL_ID": str(trial._trial_id),
            "GRAPHIX_OPTUNA_PRUNED_MARKER": str(marker),
            "GRAPHIX_OPTUNA_STARTUP_TRIALS": str(a.startup_trials),
            "GRAPHIX_OPTUNA_WARMUP_STEPS": str(a.warmup_steps),
        })
        t0 = time.time()
        with open(out_dir / "train.log", "w") as log:
            rc = subprocess.call([sys.executable, "seq2seq/run_seq2seq_train.py", str(cfg_path)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        state_file = out_dir / "run" / "trainer_state.json"
        best, epochs = None, None
        if state_file.exists():
            st = json.load(open(state_file))
            best, epochs = st.get("best_metric"), st.get("epoch")

        if marker.exists():
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            outcome = "pruned"
        elif rc == 0 and best is not None:
            study.tell(trial, best)
            outcome = "complete"
        else:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            outcome = f"fail(rc={rc})"

        with open(summary_path, "a") as f:
            f.write(json.dumps({"trial": trial.number, "outcome": outcome, "best_eval_loss": best,
                                "epochs_run": epochs, "minutes": round((time.time() - t0) / 60, 1),
                                "params": params}) + "\n")
        # Weights are the bulk of the disk cost and aren't needed after the trial.
        run_dir = out_dir / "run"
        if run_dir.exists():
            for p in list(run_dir.glob("checkpoint-*")) + list(run_dir.glob("*.bin")):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        done += 1

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials; not writing a best config.")
        return
    best_trial = study.best_trial
    final = dict(base)
    final.update(best_trial.params)
    for k in ("eval_steps", "save_steps"):
        final.pop(k, None)
    final.update({
        "run_name": f"{a.study_name}-best",
        "output_dir": f"/train_db_id/graphix-{a.study_name}-best",
        "num_train_epochs": a.max_epochs,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "predict_with_generate": True,
    })
    json.dump(final, open(a.best_config_out, "w"), indent=4)
    print(f"best trial #{best_trial.number}: eval_loss={best_trial.value:.4f} params={best_trial.params}")
    print(f"wrote {a.best_config_out}")


if __name__ == "__main__":
    main()
