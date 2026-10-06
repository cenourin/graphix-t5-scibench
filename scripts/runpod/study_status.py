#!/usr/bin/env python
"""Read-only status of a study root (scripts/runpod_study.sh status). Reads only plain files
written by the launcher and graphix_modern/study.py (state.json, pid, events.jsonl,
trials.jsonl, FINAL_DONE, eval_results.json): no Optuna/SQLite access, no signals, no GPU.

  python scripts/runpod/study_status.py STUDY_ROOT
"""
import json
import os
import sys
import time
from pathlib import Path

CELLS = [("spider", "rgat"), ("spider", "plain"), ("sciencebenchmark", "rgat"), ("sciencebenchmark", "plain")]


def jl(path):
    out = []
    if path.exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # a line being written
    return out


def ago(ts):
    try:
        t = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
        m = (time.time() - t) / 60
        return "%.0f min ago" % m if m < 120 else "%.1f h ago" % (m / 60)
    except Exception:
        return ""


def main(root):
    root = Path(root)
    ld = root / "launcher"
    st = json.load(open(ld / "state.json")) if (ld / "state.json").exists() else {}
    pid = (ld / "pid").read_text().strip() if (ld / "pid").exists() else None
    alive = bool(pid) and os.path.isdir("/proc/%s" % pid)
    print("study root : %s" % root)
    print("launcher   : %s%s (rc=%s) started %s ended %s" % (
        st.get("status", "never started"), " [pid %s alive]" % pid if alive else "",
        st.get("rc"), st.get("started"), st.get("ended")))
    if st.get("status") == "running" and not alive:
        print("             WARNING: state says running but the runner is gone (pod stopped or killed);"
              " 'start' resumes")
    launch = json.load(open(ld / "launch.json")) if (ld / "launch.json").exists() else {}
    if launch:
        print("commit     : %s  args %s" % (launch.get("commit"), launch.get("study_args")))
    ev = jl(root / "events.jsonl")
    if ev:
        e = ev[-1]
        print("last event : %s %s (%s)" % (e["time"], e["event"], ago(e["time"])))
    print()
    print("%-26s %-24s %-14s %s" % ("cell", "trials (C/P/F)", "final", "dev"))
    for bench, arm in CELLS:
        name = "%s_t5base_%s" % (bench, arm)
        recs = jl(root / "trials" / name / "trials.jsonl")
        n = {k: sum(1 for r in recs if r["outcome"].startswith(k)) for k in ("complete", "pruned", "fail")}
        best = min((r["best_eval_loss"] for r in recs if r["outcome"] == "complete"), default=None)
        running = [e for e in ev if e.get("study") == name and e["event"] == "trial_start"]
        ended = [e for e in ev if e.get("study") == name and e["event"] == "trial_end"]
        trials = "%d/%d/%d" % (n["complete"], n["pruned"], n["fail"])
        if best is not None:
            trials += " best %.4f" % best
        if len(running) > len(ended):
            trials += " (trial %d running)" % running[-1]["trial"]
        fd = root / "final" / ("study-t5base-%s-%s" % (bench, arm))
        if (fd / "FINAL_DONE").exists():
            post = json.load(open(fd / "FINAL_DONE"))
            fin = "done ep %.0f" % post["epoch"]
        elif fd.exists():
            cks = sorted(fd.glob("run/checkpoint-*"), key=lambda c: int(c.name.split("-")[1]))
            fin = "in progress" + (" (ckpt %s)" % cks[-1].name.split("-")[1] if cks else "")
        else:
            fin = "-"
        res = fd / "dev_eval" / "run" / "eval_results.json"
        if res.exists():
            r = json.load(open(res))
            dev = "EM %.3f EX %.3f (n=%s)" % (r.get("eval_exact_match", float("nan")), r.get("eval_exec", float("nan")),
                                             r.get("eval_samples"))
        else:
            dev = "running" if fd.exists() and (fd / "dev_eval").exists() else "-"
        print("%-26s %-24s %-14s %s" % ("%s/%s" % (bench, arm), trials, fin, dev))
    fails = [e for e in ev if e["event"] in ("stale_trial_failed", "final_incomplete_checkpoint_moved_aside",
                                              "final_moved_aside", "no_complete_trials")]
    if fails:
        print()
        print("notable events:")
        for e in fails[-8:]:
            print("  %s %s %s" % (e["time"], e["event"], {k: v for k, v in e.items() if k not in ("time", "event")}))


if __name__ == "__main__":
    main(sys.argv[1])
