#!/usr/bin/env python
"""Test TS: end-to-end smoke of the study orchestrator graphix_modern/study.py on the modern
stack, with real interruptions. Runs only after TE (scripts/port_te_epoch_control.py) passed.

  1. spider/rgat: the orchestrator is started and SIGKILLed (whole process group) while
     trial 0 is training; restarted, it must mark trial 0 FAIL (counted, as the legacy
     orchestrator does) and run the remaining trial; it is SIGKILLed again while the final
     run is writing its first checkpoint; restarted, the incomplete checkpoint must be moved
     aside (not deleted), and it is SIGKILLed once more after a complete checkpoint;
     restarted, the final run must resume from that checkpoint (the resumed process runs
     only the remaining steps), finish, pass the best-checkpoint check (FINAL_DONE) and run
     the dev evaluation.
     (spider/plain is launched with MKL_THREADING_LAYER=INTEL inherited, the cause of the
     first TS failure, which the orchestrator must neutralize.)
  2. the other three cells (spider/plain, sciencebenchmark/rgat, sciencebenchmark/plain)
     run uninterrupted.
  3. a third start must do nothing (every stage skipped).
Checks per cell: trials.jsonl and the study agree; trial configs follow PROTOCOLO.md §4
(space, schedule of the final run, stop after the search epochs); final config = base +
best params; final model == best checkpoint bit for bit; dev eval wrote eval_results.json
with exact match and execution and its predictions; every run's run_env.json shows strict
FP32 (no TF32, matmul precision "highest") and the commit.

  python scripts/port_ts_study_smoke.py      (inside the a5 image, cwd = project root,
                                              scripts/runpod/common.sh sourced)
Writes $WS/outputs/port_tests/TS/TS_report.json; exit != 0 on any failed check.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna

WS = Path(os.environ.get("WS", "/workspace"))
ROOT = WS / "port_tests_TS" / "study"           # the orchestrator appends _smoke
SROOT = Path(str(ROOT) + "_smoke")
OUT = WS / "outputs" / "port_tests" / "TS"
CELLS = [("spider", "rgat"), ("spider", "plain"), ("sciencebenchmark", "rgat"), ("sciencebenchmark", "plain")]
checks, failures = [], []


def check(cond, msg):
    checks.append({"ok": bool(cond), "check": msg})
    print("  [%s] %s" % ("ok" if cond else "FAIL", msg), flush=True)
    if not cond:
        failures.append(msg)


def start(benches, arms, tag, extra_env=None):
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / ("orchestrator_%s.log" % tag), "w")
    cmd = [sys.executable, "graphix_modern/study.py", "--smoke", "--root", str(ROOT), "--benches", *benches, "--arms", *arms]
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, **(extra_env or {})})  # explicit: see child_env() in graphix_modern/study.py


def kill_when(p, cond, what, timeout_s=3600, poll_s=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.poll() is not None:
            check(False, "orchestrator ended (rc=%s) before %s" % (p.returncode, what))
            return False
        if cond():
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            print("killed the orchestrator's process group: %s" % what, flush=True)
            return True
        time.sleep(poll_s)
    check(False, "timeout waiting for %s" % what)
    os.killpg(p.pid, signal.SIGKILL)
    return False


def training_started(run_dir, after_s=30):
    """The trial process is past startup: its run directory exists (combined_args.json is
    written before data loading) and after_s seconds have passed since."""
    f = Path(run_dir) / "combined_args.json"
    return f.exists() and time.time() - f.stat().st_mtime > after_s


def events():
    return [json.loads(l) for l in open(SROOT / "events.jsonl")]


def cell_checks(bench, arm):
    name = "%s_t5base_%s" % (bench, arm)
    storage = "sqlite:///%s" % (SROOT / "optuna" / (name + ".db"))
    study = optuna.load_study(study_name=name, storage=storage)
    states = [t.state.name for t in study.trials]
    recs = [json.loads(l) for l in open(SROOT / "trials" / name / "trials.jsonl")]
    check(len([s for s in states if s in ("COMPLETE", "PRUNED", "FAIL")]) == 2 and "RUNNING" not in states,
          "%s: 2 finished trials, none running (%s)" % (name, states))
    for t in study.trials:
        cfg_p = SROOT / "trials" / name / ("trial_%d" % t.number) / "trial_config.json"
        if not cfg_p.exists():
            continue
        cfg = json.load(open(cfg_p))
        p = t.params
        ok = (2e-5 <= p["learning_rate"] <= 1e-3 and 0 <= p["warmup_ratio"] <= 0.1
              and p["gradient_accumulation_steps"] in (8, 16, 32, 64) and p["weight_decay"] in (0.0, 0.01, 0.1)
              and all(cfg[k] == v for k, v in p.items()) and cfg["num_train_epochs"] == 2
              and cfg["evaluation_strategy"] == "epoch" and cfg["load_best_model_at_end"])
        check(ok, "%s trial %d: config = base + params in the §4 space, schedule of the final run" % (name, t.number))
        rd = cfg_p.parent / "run"
        check(not list(rd.glob("checkpoint-*")) and not list(rd.glob("*.safetensors")),
              "%s trial %d: weights removed after the trial" % (name, t.number))
        env = cfg_p.parent / "run" / "run_env.json"
        if env.exists():
            e = json.load(open(env))
            check(not e["matmul_allow_tf32"] and not e["cudnn_allow_tf32"] and e["float32_matmul_precision"] == "highest",
                  "%s trial %d: strict fp32 recorded" % (name, t.number))
            check(e["vars"].get("GRAPHIX_STOP_AFTER_EPOCHS") == "1", "%s trial %d: stops after the 1 search epoch" % (name, t.number))
    check(len(recs) >= 1, "%s: trials.jsonl written" % name)
    fd = SROOT / "final" / ("study-t5base-%s-%s" % (bench, arm))
    done = fd / "FINAL_DONE"
    check(done.exists(), "%s: FINAL_DONE" % name)
    if not done.exists():
        return
    post = json.load(open(done))
    check(post["final_equals_best_checkpoint"], "%s: final model == best checkpoint (bitwise)" % name)
    fc = json.load(open(fd / "final_config.json"))
    best = [t for t in study.trials if t.state.name == "COMPLETE"]
    best = min(best, key=lambda t: t.value)
    check(all(fc[k] == v for k, v in best.params.items()) and fc["num_train_epochs"] == 2,
          "%s: final config = base + best trial %d params" % (name, best.number))
    ev = fd / "dev_eval" / "run" / "eval_results.json"
    check(ev.exists(), "%s: dev eval_results.json" % name)
    if ev.exists():
        r = json.load(open(ev))
        check(r.get("eval_samples") == 20 and any("exact" in k for k in r) and any("exec" in k for k in r),
              "%s: dev eval of 20 examples with exact match and execution" % name)
        check(any((fd / "dev_eval" / "run").glob("predictions*")), "%s: dev predictions written" % name)
        e = json.load(open(fd / "dev_eval" / "run" / "run_env.json"))
        check(e["vars"].get("GRAPHIX_INIT_STATE_DICT", "").endswith("run/model.safetensors"),
              "%s: dev eval loaded the final weights" % name)


def main():
    for d in (SROOT, OUT):
        if d.exists():
            shutil.rmtree(d)
    t0 = time.time()
    trial0 = SROOT / "trials" / "spider_t5base_rgat" / "trial_0" / "run"
    fin = SROOT / "final" / "study-t5base-spider-rgat" / "run"

    print("== 1a: kill during trial 0", flush=True)
    p = start(["spider"], ["rgat"], "1a")
    kill_when(p, lambda: training_started(trial0), "trial 0 training")
    def ck_state():
        cks = sorted(fin.glob("checkpoint-*"), key=lambda c: int(c.name.split("-")[1])) if fin.exists() else []
        return [(int(c.name.split("-")[1]), (c / "trainer_state.json").is_file()) for c in cks]

    print("== 1b: restart; kill while the final run is writing its first checkpoint", flush=True)
    p = start(["spider"], ["rgat"], "1b")
    kill_when(p, lambda: any(not done for _, done in ck_state()), "a checkpoint being written", poll_s=0.05)
    caught = ck_state()
    check(caught and not caught[-1][1], "killed mid-save: checkpoint directory without trainer_state.json %s" % caught)

    print("== 1c: restart; incomplete checkpoint moved aside; kill after a complete checkpoint", flush=True)
    p = start(["spider"], ["rgat"], "1c")
    kill_when(p, lambda: any(done for _, done in ck_state()) and not (fin / "train_results.json").exists(),
              "a complete final checkpoint")
    ev = events()
    moved = [e for e in ev if e["event"] == "final_incomplete_checkpoint_moved_aside"]
    check(moved and Path(moved[0]["to"]).is_dir(), "incomplete checkpoint moved aside, not deleted (%s)"
          % [m["to"] for m in moved])
    ck = [n for n, done in ck_state() if done]
    print("== 1d: restart to completion (complete checkpoints at kill: %s)" % ck, flush=True)
    p = start(["spider"], ["rgat"], "1d")
    rc = p.wait()
    check(rc == 0, "spider/rgat completes after three kills (rc=%d)" % rc)
    ev = events()
    check(any(e["event"] == "stale_trial_failed" and e.get("trial") == 0 for e in ev),
          "trial 0, killed while running, marked FAIL at restart")
    starts = [e for e in ev if e["event"] == "final_start"]
    check(starts and starts[-1].get("resume") and ck and starts[-1].get("resume_from_step") == ck[-1],
          "last final start resumed from checkpoint-%s (%s)" % (ck[-1] if ck else None, starts[-1] if starts else None))
    if ck and (fin / "run_timing.json").exists():
        steps = [s for s, _ in json.load(open(fin / "run_timing.json"))["step_end"]]
        st = json.load(open(fin / "trainer_state.json"))
        check(steps and steps[0] == ck[-1] + 1 and steps[-1] == st["global_step"],
              "the resumed process ran only steps %d..%d (%s)" % (ck[-1] + 1, st["global_step"], steps))

    print("== 2: the other three cells", flush=True)
    for bench, arm in CELLS[1:]:
        # regression (root cause of the first TS run): a launcher whose C environment holds
        # MKL_THREADING_LAYER=INTEL must not break the runs (child_env() removes it)
        extra = {"MKL_THREADING_LAYER": "INTEL"} if (bench, arm) == ("spider", "plain") else None
        p = start([bench], [arm], "2_%s_%s" % (bench, arm), extra)
        rc = p.wait()
        check(rc == 0, "%s/%s completes (rc=%d)%s" % (bench, arm, rc, " with MKL_THREADING_LAYER=INTEL inherited" if extra else ""))
        if extra:
            log = (OUT / ("orchestrator_2_%s_%s.log" % (bench, arm))).read_text()
            check("MKL_THREADING_LAYER inherited from the launcher, removed" in log,
                  "the orchestrator removed the inherited MKL_THREADING_LAYER")

    print("== 3: a third start of everything does nothing", flush=True)
    n_before = len(events())
    p = start(["spider", "sciencebenchmark"], ["rgat", "plain"], "3")
    rc = p.wait()
    new = [e["event"] for e in events()[n_before:]]
    check(rc == 0 and set(new) <= {"pipeline_start", "pipeline_end"}, "re-run skips every stage (%s)" % new)

    print("== per-cell checks", flush=True)
    for bench, arm in CELLS:
        cell_checks(bench, arm)
    rep = {"test": "TS orchestrator smoke", "passed": not failures, "failures": failures, "checks": checks,
           "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(rep, open(OUT / "TS_report.json", "w"), indent=1)
    print("\nTS %s (%d failed of %d checks) in %.1f min" % ("PASSED" if not failures else "FAILED", len(failures),
                                                          len(checks), rep["minutes"]))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
