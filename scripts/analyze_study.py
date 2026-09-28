#!/usr/bin/env python3
"""Pre-registered analysis of the t5-base study (PROTOCOLO.md §7). Stdlib only.

Inputs: per-example scores of each cell's dev predictions, made by
scripts/score_predictions.py into train_db_id/rescore/study-t5base-<bench>-<arm>.jsonl.
Cells without scores are reported as missing, never skipped silently.

  1. EM and EX per cell, 95% percentile bootstrap CI (10 000 resamples, seed 0).
  2. Q1, rgat vs plain per benchmark and metric: exact McNemar (two-sided) on the
     examples both cells could score, paired bootstrap CI of the difference, Holm
     correction over the 4 tests.
  3. ScienceBenchmark, secondary: the same excluding dev examples whose gold SQL occurs
     verbatim in the train set (dataset leak).
  4. ScienceBenchmark per database, descriptive.

Usage: scripts/analyze_study.py [--scores-dir train_db_id/rescore] [--out FILE.md]
"""
import argparse
import json
import math
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCHES = ["spider", "sciencebenchmark"]
ARMS = ["rgat", "plain"]
METRICS = [("em", "EM"), ("ex", "EX")]
SCIB_TRAIN = REPO / "data_all_in/data/sciencebenchmark/output/seq2seq_train_dataset.json"
N_BOOT, SEED = 10_000, 0


def load(scores_dir, bench, arm):
    f = scores_dir / f"study-t5base-{bench}-{arm}.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else None


def acc(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(xs, rng):
    if not xs:
        return (float("nan"), float("nan"))
    n = len(xs)
    stats = sorted(acc([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(N_BOOT))
    return stats[int(0.025 * N_BOOT)], stats[int(0.975 * N_BOOT) - 1]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: binomial test of b vs c discordant pairs at p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [None] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired(rows_a, rows_b, key, keep=lambda r: True):
    """Rows scored by both cells, after checking both cells saw the same examples."""
    assert len(rows_a) == len(rows_b), "cells have different dev sets"
    pairs = []
    for ra, rb in zip(rows_a, rows_b):
        assert (ra["question"], ra["gold"]) == (rb["question"], rb["gold"]), f"row {ra['idx']} differs"
        if ra[key] is not None and rb[key] is not None and keep(ra):
            pairs.append((ra[key], rb[key]))
    return pairs


def paired_diff_ci(pairs, rng):
    n = len(pairs)
    if not n:
        return (float("nan"), float("nan"))
    stats = sorted(
        sum(a - b for a, b in (pairs[rng.randrange(n)] for _ in range(n))) / n for _ in range(N_BOOT))
    return stats[int(0.025 * N_BOOT)], stats[int(0.975 * N_BOOT) - 1]


def pct(x):
    return "–" if x != x else f"{100 * x:.1f}"


def leaked_golds():
    norm = lambda s: " ".join(s.lower().split())
    return {norm(ex["query"]) for ex in json.load(open(SCIB_TRAIN))}, norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores-dir", default=str(REPO / "train_db_id/rescore"))
    p.add_argument("--out")
    a = p.parse_args()
    sd = Path(a.scores_dir)
    cells = {(b, arm): load(sd, b, arm) for b in BENCHES for arm in ARMS}
    train_golds, norm = leaked_golds()
    not_leaked = lambda r: norm(r["gold"]) not in train_golds
    L = ["# Estudo t5-base: resultados (análise pré-registrada, PROTOCOLO.md §7)\n"]

    L.append("## 1. EM e EX por célula (dev oficial, IC 95% bootstrap)\n")
    L.append("| Célula | n | EM % [IC] | EX % [IC] |\n|---|---|---|---|")
    for (b, arm), rows in cells.items():
        if rows is None:
            L.append(f"| {b} / {arm} | – | *sem resultado* | *sem resultado* |")
            continue
        rng, parts = random.Random(SEED), []
        for key, _ in METRICS:
            xs = [r[key] for r in rows if r[key] is not None]
            lo, hi = boot_ci(xs, rng)
            parts.append(f"{pct(acc(xs))} [{pct(lo)}, {pct(hi)}] (n={len(xs)})")
        L.append(f"| {b} / {arm} | {len(rows)} | {parts[0]} | {parts[1]} |")

    L.append("\n## 2. Q1: rgat vs plain (McNemar exato pareado, Holm sobre 4 testes)\n")
    tests = []
    for b in BENCHES:
        ra, rp = cells[(b, "rgat")], cells[(b, "plain")]
        for key, name in METRICS:
            if ra is None or rp is None:
                tests.append((b, name, None))
                continue
            pairs = paired(ra, rp, key)
            bb = sum(1 for x, y in pairs if x == 1 and y == 0)
            cc = sum(1 for x, y in pairs if x == 0 and y == 1)
            lo, hi = paired_diff_ci(pairs, random.Random(SEED))
            diff = acc([x for x, _ in pairs]) - acc([y for _, y in pairs])
            tests.append((b, name, dict(n=len(pairs), b=bb, c=cc, diff=diff, lo=lo, hi=hi,
                                        p=mcnemar_exact(bb, cc))))
    done = [t for t in tests if t[2] is not None]
    for t, adj in zip(done, holm([t[2]["p"] for t in done])):
        t[2]["p_holm"] = adj
    L.append("| Benchmark | Métrica | n pareado | só rgat acerta | só plain acerta | Δ rgat−plain (p.p.) [IC] | p | p Holm |\n|---|---|---|---|---|---|---|---|")
    for b, name, r in tests:
        if r is None:
            L.append(f"| {b} | {name} | *sem resultado* | | | | | |")
        else:
            L.append(f"| {b} | {name} | {r['n']} | {r['b']} | {r['c']} | {100*r['diff']:+.1f} "
                     f"[{100*r['lo']:+.1f}, {100*r['hi']:+.1f}] | {r['p']:.3g} | {r.get('p_holm', float('nan')):.3g} |")
    L.append("\nH0 (diferença zero) é rejeitada só onde p Holm < 0,05. Holm é aplicado apenas "
             "sobre os testes com resultado; com células faltando, a correção fica incompleta.")

    L.append("\n## 3. ScienceBenchmark sem os exemplos cujo SQL aparece no train\n")
    rows = next((r for r in (cells[("sciencebenchmark", arm)] for arm in ARMS) if r), None)
    if rows is None:
        L.append("*sem resultado*")
    else:
        leaked = [r for r in rows if not not_leaked(r)]
        L.append(f"Exemplos excluídos: {len(leaked)} ({', '.join(sorted({r['db_id'] for r in leaked})) or '–'}).\n")
        L.append("| Célula | EM % | EX % |\n|---|---|---|")
        for arm in ARMS:
            rr = cells[("sciencebenchmark", arm)]
            if rr is None:
                L.append(f"| {arm} | – | – |")
                continue
            kept = [r for r in rr if not_leaked(r)]
            L.append(f"| {arm} | " + " | ".join(
                pct(acc([r[k] for r in kept if r[k] is not None])) for k, _ in METRICS) + " |")

    L.append("\n## 4. ScienceBenchmark por banco (descritivo)\n")
    L.append("| Célula | Banco | n | EM % | EX % |\n|---|---|---|---|---|")
    for arm in ARMS:
        rr = cells[("sciencebenchmark", arm)]
        if rr is None:
            continue
        for db in sorted({r["db_id"] for r in rr}):
            sub = [r for r in rr if r["db_id"] == db]
            L.append(f"| {arm} | {db} | {len(sub)} | " + " | ".join(
                pct(acc([r[k] for r in sub if r[k] is not None])) for k, _ in METRICS) + " |")

    txt = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
