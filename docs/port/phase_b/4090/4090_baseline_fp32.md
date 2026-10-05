# 4090_baseline_fp32

Session `phase_b_s2`, commit `cc5f69b352f4e092cb9d728b8d9200662113ffe8`, image `sha256:371f61af7521069e62bb8973c2c42584377659440b5902d025876fb94d859037`. Precision: **FP32 strict** (TF32 off, autocast off, compile off), batch 1, GA 8 in the probes. Estimates below come from a few hundred steps; they are not measurements of the study.

## Hardware and versions

| item | value |
|---|---|
| gpu_model | NVIDIA GeForce RTX 4090 |
| compute_capability | 8.9 |
| vram_gb | 23.53 |
| driver | 570.195.03 |
| torch | 2.4.0 |
| torch_cuda_runtime | 12.1 |
| dgl | 2.4.0+cu121 |
| transformers | 4.57.6 |
| datasets | 2.21.0 |
| numpy | 1.26.4 |
| python | 3.11.9 |
| env_fingerprint | {'got': 'd6d923a223971ddb93299963c68d31671260cce4e5906d7ce971ff0658d8b26d', 'expected': 'd6d923a223971ddb93299963c68d31671260cce4e5906d7ce971ff0658d8b26d', 'image': 'silveirabruno/graphix-modern@sha256:371f61af7521069e62bb8973c2c42584377659440b5902d025876fb94d859037'} |
| empirical_fp32_matmul_not_tf32 | 8.103687136086285e-07 |

## Stages

Stage rows include startup and evaluation; rows marked *(training only)* cover train_begin → train_end of that run.

| stage | seconds | rc | GPU util mean | GPU mem max (MiB) | CPU % mean | RAM max (GB) |
|---|---|---|---|---|---|---|
| preflight | 19.3 | 0 | 0 | 4 | 7 | 11.1 |
| smoke_train | 17.0 | 0 | 4 | 3890 | 8 | 14.2 |
| smoke_resume | 18.0 | 0 | 7 | 3906 | 7 | 13.9 |
| smoke_eval | 14.1 | 0 | 6 | 1960 | 7 | 13.2 |
| steps50 | 46.5 | 0 | 50 | 7288 | 7 | 14.0 |
| probe_spider_rgat | 142.7 | 0 | 55 | 7512 | 6 | 14.0 |
| profile_spider_rgat | 39.6 | 0 | 27 | 7284 | 6 | 13.9 |
| devgen_spider | 22.2 | 0 | 6 | 1960 | 7 | 13.3 |
| probe_spider_plain | 82.1 | 0 | 57 | 3584 | 7 | 13.2 |
| probe_sciencebenchmark_rgat | 538.9 | 0 | 44 | 4492 | 6 | 16.1 |
| profile_sciencebenchmark_rgat | 68.0 | 0 | 54 | 4486 | 7 | 15.4 |
| devgen_sciencebenchmark | 65.0 | 0 | 7 | 2058 | 6 | 14.9 |
| probe_sciencebenchmark_plain | 153.7 | 0 | 68 | 3050 | 7 | 14.8 |
| report | – | None | – | – | – | – |
| probe_sciencebenchmark_plain (training only) | – |  | 80 | 3050 | 7 | 14.5 |
| profile_sciencebenchmark_rgat (training only) | – |  | 80 | 4486 | 6 | 14.4 |
| probe_sciencebenchmark_rgat (training only) | – |  | 90 | 4492 | 6 | 14.6 |
| probe_spider_plain (training only) | – |  | 69 | 3584 | 7 | 12.7 |
| profile_spider_rgat (training only) | – |  | 57 | 7284 | 6 | 13.0 |
| probe_spider_rgat (training only) | – |  | 64 | 7512 | 6 | 12.9 |

## Throughput (profiling off)

| probe | s/sample | samples/s | steps/s (GA 8) | peak VRAM alloc/reserved (GB) | n_train | val eval (s) | startup (s) |
|---|---|---|---|---|---|---|---|
| probe_sciencebenchmark_plain | 0.0624 | 16.02 | 2.003 | 2.31 / 2.51 | 4233 | 6 | 19 |
| probe_sciencebenchmark_rgat | 0.1251 | 7.99 | 0.999 | 3.64 / 3.92 | 4233 | 14 | 270 |
| probe_spider_plain | 0.0328 | 30.49 | 3.811 | 2.69 / 3.04 | 7345 | 3 | 10 |
| probe_spider_rgat | 0.0559 | 17.89 | 2.237 | 6.11 / 6.87 | 7345 | 10 | 16 |

## 50 steps (T10 replayed)

Mechanics identical to the 1070 reference (steps, LR per step, order of 406 items): **True**. Loss vs the six 1070 T10 runs, median: mean_rel 1.96e-02, max_rel 9.28e-02, RMSE 2.20e-02; within the T10 same-stack envelope: **True** (reported, not a gate). Train runtime 29.0 s, peak VRAM 6.09 GB.

## Time per stage — spider/RGAT (profiling on, CUDA-synchronized)

Profiled training wall time 18.6 s. Shares are what matters here; absolute times are inflated by the synchronization.

| section | seconds | share |
|---|---|---|
| backward | 8.15 | 43.8% |
| rgat forward (net of graph_to) | 3.54 | 19.0% |
| decoder T5 forward | 2.21 | 11.9% |
| encoder T5 forward (net of rgat) | 1.74 | 9.4% |
| optimizer (step+sched+zero_grad) | 1.70 | 9.2% |
| other (loss, Python, logging, unaccounted) | 0.89 | 4.8% |
| data_wait | 0.16 | 0.9% |
| graph_to (graph.to(device), per RGAT layer, forward) | 0.13 | 0.7% |
| lm_head | 0.05 | 0.3% |
| prepare_inputs (CPU->GPU) | 0.02 | 0.1% |
| graph_build (graph batch on CPU) | 0.01 | 0.1% |

**Dominant: backward (44%).**

## Time per stage — sciencebenchmark/RGAT (profiling on, CUDA-synchronized)

Profiled training wall time 43.8 s. Shares are what matters here; absolute times are inflated by the synchronization.

| section | seconds | share |
|---|---|---|
| backward (incl. block recompute: RGAT 6.4 s) | 28.98 | 66.1% |
| rgat forward (net of graph_to) | 6.25 | 14.3% |
| encoder T5 forward (net of rgat) | 3.25 | 7.4% |
| decoder T5 forward | 2.36 | 5.4% |
| optimizer (step+sched+zero_grad) | 1.72 | 3.9% |
| other (loss, Python, logging, unaccounted) | 0.75 | 1.7% |
| data_wait | 0.25 | 0.6% |
| graph_to (graph.to(device), per RGAT layer, forward) | 0.15 | 0.4% |
| lm_head | 0.06 | 0.1% |
| prepare_inputs (CPU->GPU) | 0.02 | 0.0% |
| graph_build (graph batch on CPU) | 0.02 | 0.0% |

**Dominant: backward (incl. block recompute: RGAT 6.4 s) (66%).**

## Cost estimate

| cell | s/sample | steps/epoch (GA 8/16/32/64) | min/epoch | US$/epoch | trial (3 ep) | 6 trials | final | dev eval | cell total |
|---|---|---|---|---|---|---|---|---|---|
| spider_rgat | 0.056 | 918/459/229/114 | 7.0 | 0.087 | 0.36 h / $0.26 | 2.1 h / $1.58 | 0.5–1.8 h | 0.03 h | 1.2–3.9 h / $0.91–2.91 |
| spider_plain | 0.033 | 918/459/229/114 | 4.1 | 0.050 | 0.21 h / $0.15 | 1.2 h / $0.92 | 0.3–1.0 h | 0.03 h | 0.7–2.3 h / $0.54–1.70 |
| sciencebenchmark_rgat | 0.125 | 529/264/132/66 | 9.1 | 0.112 | 0.46 h / $0.34 | 2.8 h / $2.04 | 0.6–2.3 h | 0.09 h | 1.6–5.1 h / $1.22–3.80 |
| sciencebenchmark_plain | 0.062 | 529/264/132/66 | 4.5 | 0.056 | 0.23 h / $0.17 | 1.4 h / $1.03 | 0.3–1.1 h | 0.09 h | 0.9–2.6 h / $0.65–1.94 |

**4 cells: 4.5–14.0 GPU-hours, US$ 3.33–10.35 at US$ 0.74/h** (upper bound: no pruning, 15-epoch finals).

Checkpoint save (t5-base + RGAT weights, Adafactor state, RNG): 2.0 s per epoch. One-time HF datasets cache build on a fresh volume (not repeated per run): spider 0 s, sciencebenchmark 250 s. Startup uses the warm-cache value. Assumptions: `cost_estimate.assumptions` in the JSON.

## Comparison with the GTX 1070 (same probes, same code and image)

| probe | 1070 s/sample | here s/sample | speedup |
|---|---|---|---|
| probe_sciencebenchmark_plain | 0.3557 | 0.0624 | 5.70× |
| probe_sciencebenchmark_rgat | 0.7828 | 0.1251 | 6.26× |
| probe_spider_plain | 0.1582 | 0.0328 | 4.82× |
| probe_spider_rgat | 0.2645 | 0.0559 | 4.73× |

## Recommendation for the next phase-B experiment (not applied)

**spider:** Next experiment: TF32 -- it acts on 22%-65% of the profiled step time (matmul-heavy T5 parts (forward + their share of backward); smallest numeric change of the precision options). Input path and graph copies take 1.7% together, so semantics-preserving changes there cannot pay off. Precision changes need a protocol deviation and their own numeric check (T5/T6/T10-style) before any study run. Caveat: the range of BF16 autocast, RGAT in pure PyTorch (replace DGL kernels) overlaps with TF32's -- backward (44% of the step) is not split by component, so this profile cannot rank them; a direct A/B probe (one change at a time, same cells) decides. Per-run startup (data, graph store, model; warm cache) is 0.3 min, paid by each of the 8 runs of a cell (2.2 min in total) and not accelerated by the GPU.

| candidate | share of step it acts on (min–max) | note |
|---|---|---|
| TF32 | 21.5%–65.3% | matmul-heavy T5 parts (forward + their share of backward); smallest numeric change of the precision options |
| BF16 autocast | 21.5%–65.3% | same parts as TF32, larger speedup and larger numeric change; after TF32 |
| RGAT in pure PyTorch (replace DGL kernels) | 19.0%–62.8% | large change; needs T3/T5/T6 re-validation |
| torch.compile | 4.8%–4.8% | Python overhead; DGL ops force graph breaks; last |
| DataLoader (workers/pin_memory/prefetch) | 1.0%–1.0% | CPU-side input path; order and content must stay identical |
| remove repeated graph.to(device) (once per step) | 0.7%–0.7% | semantics-preserving; today the graph is copied once per RGAT layer |

**sciencebenchmark:** Next experiment: RGAT in pure PyTorch (replace DGL kernels) -- it acts on 14%-80% of the profiled step time (large change; needs T3/T5/T6 re-validation). Input path and graph copies take 1.0% together, so semantics-preserving changes there cannot pay off. Precision changes need a protocol deviation and their own numeric check (T5/T6/T10-style) before any study run. Caveat: the range of TF32, BF16 autocast overlaps with RGAT in pure PyTorch (replace DGL kernels)'s -- backward (66% of the step) is not split by component, so this profile cannot rank them; a direct A/B probe (one change at a time, same cells) decides. Per-run startup (data, graph store, model; warm cache) is 0.3 min, paid by each of the 8 runs of a cell (2.7 min in total) and not accelerated by the GPU.

| candidate | share of step it acts on (min–max) | note |
|---|---|---|
| RGAT in pure PyTorch (replace DGL kernels) | 14.3%–80.4% | large change; needs T3/T5/T6 re-validation |
| TF32 | 12.9%–79.1% | matmul-heavy T5 parts (forward + their share of backward); smallest numeric change of the precision options |
| BF16 autocast | 12.9%–79.1% | same parts as TF32, larger speedup and larger numeric change; after TF32 |
| torch.compile | 1.7%–1.7% | Python overhead; DGL ops force graph breaks; last |
| DataLoader (workers/pin_memory/prefetch) | 0.6%–0.6% | CPU-side input path; order and content must stay identical |
| remove repeated graph.to(device) (once per step) | 0.4%–0.4% | semantics-preserving; today the graph is copied once per RGAT layer |

