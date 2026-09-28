import json
import os
import sys

from transformers import TrainerCallback

# Optuna isn't in the training image; the launcher installs it with
# `pip install --target` into a mounted dir. Append (never prepend) so the image's own
# packages keep precedence and only genuinely missing modules resolve from there.
_PKGS = os.environ.get("GRAPHIX_OPTUNA_PKGS", "/optuna_studies/.pkgs")
if os.path.isdir(_PKGS) and _PKGS not in sys.path:
    sys.path.append(_PKGS)


class OptunaPruningCallback(TrainerCallback):
    """Reports eval_loss to a shared Optuna study after each evaluation and stops the
    run when the study's pruner says this trial is unpromising. Runs in the training
    subprocess; the driver owns ask/tell and marks the trial PRUNED via `pruned_marker`."""

    def __init__(self, storage: str, study_name: str, trial_id: int, pruned_marker: str):
        import optuna

        # The pruner isn't persisted in storage, so it must match the driver's.
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(os.environ.get("GRAPHIX_OPTUNA_STARTUP_TRIALS", "4")),
            n_warmup_steps=int(os.environ.get("GRAPHIX_OPTUNA_WARMUP_STEPS", "2")),
        )
        self.study = optuna.load_study(study_name=study_name, storage=storage, pruner=pruner)
        self.trial = optuna.trial.Trial(self.study, trial_id)
        self.pruned_marker = pruned_marker
        self.step = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        # The final trainer.evaluate() after training (load_best_model_at_end) runs at the
        # same global_step as the last epoch's evaluation; reporting it again would add a
        # phantom step the pruner could use to prune an already finished trial.
        if state.global_step == getattr(self, "_last_step", None):
            return
        self._last_step = state.global_step
        self.trial.report(float(metrics["eval_loss"]), self.step)
        self.step += 1
        if self.trial.should_prune():
            with open(self.pruned_marker, "w") as f:
                json.dump({"step": self.step, "epoch": state.epoch}, f)
            control.should_training_stop = True
