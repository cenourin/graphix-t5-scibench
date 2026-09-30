#!/usr/bin/env python
"""Session report: $WS/benchmarks/$REPORT_NAME.json and .md (docs/RUNPOD_PHASE_B.md §12-13).

  python scripts/runpod/report.py

Reads $WS/outputs/$SESSION (preflight.json, stages.jsonl, results/, monitor/cpu.csv) and
writes hardware, versions, throughput, VRAM, utilization per stage, profile shares, the
dominant bottleneck, the cost estimate (scripts/runpod/estimate_cost.py), the comparison
with the GTX 1070 baseline (docs/port/phase_b/reference_1070/1070_baseline_fp32.json, when
present and this is not that run), and a RECOMMENDATION for the next phase-B experiment.
Nothing is applied: the recommendation is text.
"""
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import estimate_cost  # noqa: E402

WS = Path(os.environ.get("WS", "/workspace"))
SESSION = os.environ.get("SESSION", "phase_b_s1")
REPORT_NAME = os.environ.get("REPORT_NAME", "4090_baseline_fp32")
SOUT = WS / "outputs" / SESSION
BENCH = WS / "benchmarks"
REF = Path("docs/port/phase_b/reference_1070/1070_baseline_fp32.json")
PRICE = float(os.environ.get("RUNPOD_GPU_USD_PER_HOUR", "0.74"))



def _commit():
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "runpod"))
    try:
        import gitinfo
        return gitinfo.commit(".")
    except Exception as e:
        return "unknown (%s)" % e

def jl(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def stage_windows():
    starts, ends = {}, {}
    p = SOUT / "stages.jsonl"
    if not p.exists():
        return {}
    for line in open(p):
        e = json.loads(line)
        (starts if e["event"] == "start" else ends)[e["stage"]] = e  # last of each
    out = {}
    for k, st in starts.items():
        en = ends.get(k)
        if en is not None and en["epoch_s"] < st["epoch_s"]:
            en = None  # an end from an earlier invocation (e.g. the report measuring itself)
        out[k] = {"start": st["epoch_s"], "end": en["epoch_s"] if en else None, "rc": en.get("rc") if en else None,
                  "seconds": (en["epoch_s"] - st["epoch_s"]) if en else None}
    return out


def train_windows(res):
    w = {}
    for k, v in res.items():
        t = v.get("timing") if isinstance(v, dict) else None
        if t and t.get("train_begin") and t.get("train_end"):
            w[k + " (training only)"] = {"start": t["train_begin"], "end": t["train_end"]}
    return w


def utilization(windows):
    p = SOUT / "monitor" / "cpu.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(open(p)))
    out = {}
    for name, w in windows.items():
        if w["end"] is None:
            continue
        r = [x for x in rows if w["start"] <= float(x["epoch_s"]) <= w["end"]]
        gu = [float(x["gpu_util"]) for x in r if x.get("gpu_util")]
        gm = [float(x["gpu_mem_used_mib"]) for x in r if x.get("gpu_mem_used_mib")]
        gp = [float(x["gpu_power_w"]) for x in r if x.get("gpu_power_w") not in (None, "", "[N/A]")]
        out[name] = {"samples": len(r),
                     "gpu_util_mean": sum(gu) / len(gu) if gu else None,
                     "gpu_mem_used_max_mib": max(gm) if gm else None,
                     "gpu_power_mean_w": sum(gp) / len(gp) if gp else None,
                     "cpu_percent_mean": sum(float(x["cpu_percent"]) for x in r) / len(r) if r else None,
                     "cpu_count_visible": int(r[0]["cpu_count"]) if r else None,
                     "ram_used_max_gb": max(float(x["ram_used_gb"]) for x in r) if r else None,
                     "python_rss_max_gb": max(float(x["top_python_rss_gb"]) for x in r) if r else None}
    return out


def profile_shares(res, bench):
    """Shares of the profiled training wall time (train_begin -> train_end, CUDA-synchronized).
    With gradient checkpointing (ScienceBenchmark), every encoder block -- RGAT included -- is
    run again inside backward, so the RGAT hook fires twice per step: half of its calls (and
    of graph_to's) are recomputation that is also part of the backward time. The forward
    shares below keep only the forward half; backward keeps the recompute."""
    if not res:
        return None
    p, t = res["profile"], res["timing"]
    cfg = jl(SOUT / "configs" / ("profile_%s_rgat.json" % bench)) or {}
    gc = bool(cfg.get("gradient_checkpointing"))
    f = 0.5 if gc else 1.0
    wall = t["train_end"] - t["train_begin"]
    sec = lambda k: p.get(k, {}).get("seconds", 0.0)
    rgat_fwd, gto_fwd = sec("rgat") * f, sec("graph_to") * f
    top = {"data_wait": sec("data_wait"), "prepare_inputs (CPU->GPU)": sec("prepare_inputs"),
           "graph_build (graph batch on CPU)": sec("graph_build"),
           "graph_to (graph.to(device), per RGAT layer, forward)": gto_fwd,
           "rgat forward (net of graph_to)": rgat_fwd - gto_fwd,
           "encoder T5 forward (net of rgat)": sec("encoder") - rgat_fwd,
           "decoder T5 forward": sec("decoder"), "lm_head": sec("lm_head"),
           "backward" + (" (incl. block recompute: RGAT %.1f s)" % (sec("rgat") - rgat_fwd) if gc else ""): sec("backward"),
           "optimizer (step+sched+zero_grad)": sec("optimizer")}
    top["other (loss, Python, logging, unaccounted)"] = wall - sum(top.values())
    return {"train_wall_s": wall, "gradient_checkpointing": gc, "rgat_calls": p.get("rgat", {}).get("calls"),
            "seconds": top, "share": {k: v / wall for k, v in top.items()}}


def recommend(shares, util, est, bench):
    """Rank the phase-B candidates by the share of step time each can act on. Text only.
    Backward cannot be split by component here, so candidates acting on compute get a range:
    at least their forward share, at most forward + all of backward."""
    if not shares:
        return {"text": "no profile available", "ranking": []}
    s = shares["share"]
    g = lambda pred: sum(v for k, v in s.items() if pred(k))
    bwd = g(lambda k: k.startswith("backward"))
    dense = g(lambda k: k.startswith(("encoder T5", "decoder T5", "lm_head")))
    rgat = g(lambda k: k.startswith("rgat forward"))
    host = g(lambda k: k.startswith(("data_wait", "prepare_inputs", "graph_build")))
    gto = g(lambda k: k.startswith("graph_to"))
    other = g(lambda k: k.startswith("other"))
    cand = [
        ("TF32", dense, dense + bwd, "matmul-heavy T5 parts (forward + their share of backward); smallest numeric change of the precision options"),
        ("BF16 autocast", dense, dense + bwd, "same parts as TF32, larger speedup and larger numeric change; after TF32"),
        ("RGAT in pure PyTorch (replace DGL kernels)", rgat, rgat + bwd, "large change; needs T3/T5/T6 re-validation"),
        ("DataLoader (workers/pin_memory/prefetch)", host, host, "CPU-side input path; order and content must stay identical"),
        ("remove repeated graph.to(device) (once per step)", gto, gto, "semantics-preserving; today the graph is copied once per RGAT layer"),
        ("torch.compile", other, other, "Python overhead; DGL ops force graph breaks; last"),
    ]
    cand.sort(key=lambda c: -c[2])
    top = cand[0]
    text = ("Next experiment: %s -- it acts on %.0f%%-%.0f%% of the profiled step time (%s). Input path and "
            "graph copies take %.1f%% together, so semantics-preserving changes there cannot pay off. Precision "
            "changes need a protocol deviation and their own numeric check (T5/T6/T10-style) before any study run."
            % (top[0], 100 * top[1], 100 * top[2], top[3], 100 * (host + gto)))
    overlap = [x[0] for x in cand[1:] if x[2] >= top[1] and x[1] <= top[2] and x[2] - x[1] > 0.05]
    if overlap:
        text += (" Caveat: the range of %s overlaps with %s's -- backward (%.0f%% of the step) is not split by "
                 "component, so this profile cannot rank them; a direct A/B probe (one change at a time, same "
                 "cells) decides." % (", ".join(overlap), top[0], 100 * bwd))
    c = (est.get("cells") or {}).get("%s_rgat" % bench) or {}
    if c.get("startup_minutes") is not None and c.get("epoch_minutes"):
        runs = 6 + 1 + 1
        text += (" Per-run startup (data, graph store, model; warm cache) is %.1f min, paid by each of the %d runs of a "
                 "cell (%.1f min in total) and not accelerated by the GPU." % (c["startup_minutes"], runs, runs * c["startup_minutes"]))
    return {"text": text, "ranking": [{"candidate": x[0], "share_min": x[1], "share_max": x[2], "why": x[3]} for x in cand]}


def main():
    pre = jl(SOUT / "preflight.json") or {}
    windows = stage_windows()
    R = SOUT / "results"
    res = {p.stem: json.load(open(p)) for p in R.glob("*.json")} if R.exists() else {}
    util = utilization({**windows, **train_windows(res)})
    est = estimate_cost.estimate(R, PRICE)
    shares = {b: profile_shares(res.get("profile_%s_rgat" % b), b) for b in ("spider", "sciencebenchmark")}
    rec = {b: recommend(shares[b], util, est, b) for b in shares if shares[b]}
    throughput = {}
    for k, v in res.items():
        if k.startswith("probe_"):
            tr, tim = v["train"], v["timing"]
            sps = estimate_cost.per_sample_seconds(tim)
            throughput[k] = {"seconds_per_sample": sps, "samples_per_s": 1 / sps if sps else None,
                             "steps_per_s_at_ga8": 1 / (8 * sps) if sps else None,
                             "train_runtime_s": tr.get("train_runtime"), "peak_vram_gb": tr.get("train_peak_vram_gb"),
                             "peak_vram_reserved_gb": tr.get("train_peak_vram_reserved_gb"),
                             "n_train": tim.get("n_train"), "val_eval_s": v["eval"].get("eval_runtime"),
                             "startup_s": tim["train_begin"] - tim["process_start"]}
    ref = jl(REF) if REPORT_NAME != "1070_baseline_fp32" else None
    comparison = None
    if ref:
        comparison = {}
        for k, t in throughput.items():
            r = ref.get("throughput", {}).get(k)
            if r and r.get("seconds_per_sample") and t.get("seconds_per_sample"):
                comparison[k] = {"s_per_sample_1070": r["seconds_per_sample"], "s_per_sample_here": t["seconds_per_sample"],
                                 "speedup_vs_1070": r["seconds_per_sample"] / t["seconds_per_sample"]}
    out = {"report": REPORT_NAME, "session": SESSION, "precision": {"mode": "fp32", "tf32": False, "autocast": False, "compile": False},
           "batch": {"per_device_train_batch_size": 1, "gradient_accumulation_steps_probe": 8},
           "commit": _commit(),
           "image_digest": os.environ.get("GRAPHIX_IMAGE_DIGEST"), "preflight": pre,
           "stages": windows, "utilization": util, "results": res, "throughput": throughput,
           "profile": shares, "recommendation": rec, "cost_estimate": est, "comparison_1070": comparison}
    BENCH.mkdir(parents=True, exist_ok=True)
    name = REPORT_NAME
    prev = jl(BENCH / (name + ".json"))
    if prev and prev.get("session") != SESSION:  # never overwrite another session's report
        name = "%s_%s" % (REPORT_NAME, SESSION)
    json.dump(out, open(BENCH / (name + ".json"), "w"), indent=1, default=str)
    (BENCH / (name + ".md")).write_text(markdown(out))
    print("wrote", BENCH / (name + ".json"), "and .md")


def markdown(o):
    pre = o["preflight"]
    v = lambda k: (pre.get(k) or {}).get("value")
    L = ["# %s" % o["report"], "",
         "Session `%s`, commit `%s`, image `%s`. Precision: **FP32 strict** (TF32 off, autocast off, compile off), "
         "batch 1, GA 8 in the probes. Estimates below come from a few hundred steps; they are not measurements of the study."
         % (o["session"], o["commit"], o["image_digest"]), "",
         "## Hardware and versions", "",
         "| item | value |", "|---|---|"]
    for k in ("gpu_model", "compute_capability", "vram_gb", "driver", "torch", "torch_cuda_runtime", "dgl",
              "transformers", "datasets", "numpy", "python", "env_fingerprint", "empirical_fp32_matmul_not_tf32"):
        L.append("| %s | %s |" % (k, v(k)))
    L += ["", "## Stages", "", "Stage rows include startup and evaluation; rows marked *(training only)* cover train_begin → train_end of that run.", "",
          "| stage | seconds | rc | GPU util mean | GPU mem max (MiB) | CPU % mean | RAM max (GB) |",
          "|---|---|---|---|---|---|---|"]
    rows = list(o["stages"].items()) + [(k, {"seconds": u.get("seconds"), "rc": ""}) for k, u in o["utilization"].items() if k.endswith("(training only)")]
    for k, w in rows:
        u = o["utilization"].get(k, {})
        f = lambda x, d=0: ("%.*f" % (d, x)) if isinstance(x, (int, float)) else "–"
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (k, f(w["seconds"], 1), w["rc"], f(u.get("gpu_util_mean")),
                                                        f(u.get("gpu_mem_used_max_mib")), f(u.get("cpu_percent_mean")),
                                                        f(u.get("ram_used_max_gb"), 1)))
    L += ["", "## Throughput (profiling off)", "", "| probe | s/sample | samples/s | steps/s (GA 8) | peak VRAM alloc/reserved (GB) | n_train | val eval (s) | startup (s) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, t in o["throughput"].items():
        L.append("| %s | %.4f | %.2f | %.3f | %s / %s | %s | %.0f | %.0f |" % (
            k, t["seconds_per_sample"], t["samples_per_s"], t["steps_per_s_at_ga8"], t["peak_vram_gb"],
            t["peak_vram_reserved_gb"], t["n_train"], t["val_eval_s"] or 0, t["startup_s"]))
    s50 = o["results"].get("steps50")
    if s50:
        L += ["", "## 50 steps (T10 replayed)", "",
              "Mechanics identical to the 1070 reference (steps, LR per step, order of 406 items): **%s**. Loss vs the six 1070 T10 runs, median: mean_rel %.2e, max_rel %.2e, RMSE %.2e; within the T10 same-stack envelope: **%s** (reported, not a gate). Train runtime %.1f s, peak VRAM %s GB."
              % (s50["mechanics_identical_to_1070"], s50["median_vs_1070"]["mean_rel"], s50["median_vs_1070"]["max_rel"],
                 s50["median_vs_1070"]["rmse"], s50["loss_within_t10_envelope"], s50["train_runtime_s"] or 0, s50["train_peak_vram_gb"])]
    for b, sh in o["profile"].items():
        if not sh:
            continue
        L += ["", "## Time per stage — %s/RGAT (profiling on, CUDA-synchronized)" % b, "",
              "Profiled training wall time %.1f s. Shares are what matters here; absolute times are inflated by the synchronization." % sh["train_wall_s"],
              "", "| section | seconds | share |", "|---|---|---|"]
        for k, s in sorted(sh["seconds"].items(), key=lambda x: -x[1]):
            L.append("| %s | %.2f | %.1f%% |" % (k, s, 100 * sh["share"][k]))
        dom = max(sh["share"].items(), key=lambda x: x[1])
        L.append("")
        L.append("**Dominant: %s (%.0f%%).**" % (dom[0], 100 * dom[1]))
    L += ["", "## Cost estimate", "", estimate_cost.fmt_table(o["cost_estimate"]),
          "", "Checkpoint save (t5-base + RGAT weights, Adafactor state, RNG): %s s per epoch. One-time HF datasets cache build on a fresh volume (not repeated per run): %s. Startup uses the warm-cache value. Assumptions: `cost_estimate.assumptions` in the JSON." % (
              "%.1f" % o["cost_estimate"]["checkpoint_save_s"] if o["cost_estimate"]["checkpoint_save_s"] else "n/a",
              ", ".join("%s %.0f s" % kv for kv in o["cost_estimate"].get("one_time_hf_cache_build_s", {}).items()) or "n/a")]
    if o["comparison_1070"]:
        L += ["", "## Comparison with the GTX 1070 (same probes, same code and image)", "", "| probe | 1070 s/sample | here s/sample | speedup |", "|---|---|---|---|"]
        for k, c in o["comparison_1070"].items():
            L.append("| %s | %.4f | %.4f | %.2f× |" % (k, c["s_per_sample_1070"], c["s_per_sample_here"], c["speedup_vs_1070"]))
    L += ["", "## Recommendation for the next phase-B experiment (not applied)", ""]
    for b, r in o["recommendation"].items():
        L.append("**%s:** %s" % (b, r["text"]))
        L.append("")
        L.append("| candidate | share of step it acts on (min–max) | note |")
        L.append("|---|---|---|")
        for c in r["ranking"]:
            L.append("| %s | %.1f%%–%.1f%% | %s |" % (c["candidate"], 100 * c["share_min"], 100 * c["share_max"], c["why"]))
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
