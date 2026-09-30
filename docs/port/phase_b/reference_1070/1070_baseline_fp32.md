# 1070_baseline_fp32

Session `local_1070_final`, commit `83516ddc8e2ca2fc2da5a662fd974dcc7277af4b`, image `None`. Precision: **FP32 strict** (TF32 off, autocast off, compile off), batch 1, GA 8 in the probes. Estimates below come from a few hundred steps; they are not measurements of the study.

## Hardware and versions

| item | value |
|---|---|
| gpu_model | NVIDIA GeForce GTX 1070 |
| compute_capability | 6.1 |
| vram_gb | 7.92 |
| driver | 535.288.01 |
| torch | 2.4.0 |
| torch_cuda_runtime | 12.1 |
| dgl | 2.4.0+cu121 |
| transformers | 4.57.6 |
| datasets | 2.21.0 |
| numpy | 1.26.4 |
| python | 3.11.9 |
| env_fingerprint | {'got': 'd6d923a223971ddb93299963c68d31671260cce4e5906d7ce971ff0658d8b26d', 'expected': 'd6d923a223971ddb93299963c68d31671260cce4e5906d7ce971ff0658d8b26d', 'image': 'silveirabruno/graphix-modern@sha256:371f61af7521069e62bb8973c2c42584377659440b5902d025876fb94d859037'} |
| empirical_fp32_matmul_not_tf32 | 8.105800148706163e-07 |

## Stages

Stage rows include startup and evaluation; rows marked *(training only)* cover train_begin → train_end of that run.

| stage | seconds | rc | GPU util mean | GPU mem max (MiB) | CPU % mean | RAM max (GB) |
|---|---|---|---|---|---|---|
| preflight | 63.1 | 0 | 0 | 294 | 22 | 2.4 |
| smoke_train | 34.4 | 0 | 16 | 3656 | 27 | 5.2 |
| smoke_resume | 48.0 | 0 | 11 | 3672 | 21 | 5.3 |
| smoke_eval | 26.7 | 0 | 14 | 1726 | 29 | 4.7 |
| steps50 | 175.3 | 0 | 71 | 7054 | 27 | 6.2 |
| probe_spider_rgat | 608.8 | 0 | 64 | 7278 | 27 | 5.2 |
| profile_spider_rgat | 119.1 | 0 | 47 | 7050 | 27 | 4.7 |
| devgen_spider | 25.5 | 0 | 13 | 1726 | 28 | 4.7 |
| probe_spider_plain | 361.0 | 0 | 58 | 3350 | 26 | 4.7 |
| probe_sciencebenchmark_rgat | 1693.9 | 0 | 91 | 4258 | 29 | 6.9 |
| profile_sciencebenchmark_rgat | 303.0 | 0 | 77 | 4252 | 28 | 6.9 |
| devgen_sciencebenchmark | 62.6 | 0 | 20 | 1800 | 26 | 6.3 |
| probe_sciencebenchmark_plain | 776.1 | 0 | 82 | 2814 | 29 | 6.3 |
| report | 0.2 | 0 | 45 | 130 | 25 | 1.8 |
| probe_spider_plain (training only) | – |  | 60 | 3348 | 26 | 3.9 |
| profile_spider_rgat (training only) | – |  | 59 | 7050 | 26 | 4.1 |
| probe_spider_rgat (training only) | – |  | 66 | 7278 | 26 | 4.1 |
| probe_sciencebenchmark_plain (training only) | – |  | 84 | 2814 | 29 | 5.5 |
| profile_sciencebenchmark_rgat (training only) | – |  | 87 | 4252 | 29 | 5.8 |
| probe_sciencebenchmark_rgat (training only) | – |  | 92 | 4258 | 29 | 5.7 |

## Throughput (profiling off)

| probe | s/sample | samples/s | steps/s (GA 8) | peak VRAM alloc/reserved (GB) | n_train | val eval (s) | startup (s) |
|---|---|---|---|---|---|---|---|
| probe_spider_plain | 0.1582 | 6.32 | 0.790 | 2.69 / 3.04 | 7345 | 18 | 23 |
| probe_spider_rgat | 0.2645 | 3.78 | 0.473 | 6.11 / 6.87 | 7345 | 50 | 24 |
| probe_sciencebenchmark_plain | 0.3557 | 2.81 | 0.351 | 2.31 / 2.51 | 4233 | 37 | 24 |
| probe_sciencebenchmark_rgat | 0.7828 | 1.28 | 0.160 | 3.64 / 3.92 | 4233 | 97 | 26 |

## 50 steps (T10 replayed)

Mechanics identical to the 1070 reference (steps, LR per step, order of 406 items): **True**. Loss vs the six 1070 T10 runs, median: mean_rel 1.76e-02, max_rel 7.37e-02, RMSE 1.73e-02; within the T10 same-stack envelope: **True** (reported, not a gate). Train runtime 150.6 s, peak VRAM 6.09 GB.

## Time per stage — spider/RGAT (profiling on, CUDA-synchronized)

Profiled training wall time 91.2 s. Shares are what matters here; absolute times are inflated by the synchronization.

| section | seconds | share |
|---|---|---|
| backward | 43.10 | 47.2% |
| rgat forward (net of graph_to) | 15.90 | 17.4% |
| decoder T5 forward | 9.86 | 10.8% |
| optimizer (step+sched+zero_grad) | 9.58 | 10.5% |
| encoder T5 forward (net of rgat) | 8.53 | 9.4% |
| other (loss, Python, logging, unaccounted) | 3.02 | 3.3% |
| data_wait | 0.44 | 0.5% |
| graph_to (graph.to(device), per RGAT layer, forward) | 0.42 | 0.5% |
| lm_head | 0.27 | 0.3% |
| prepare_inputs (CPU->GPU) | 0.07 | 0.1% |
| graph_build (graph batch on CPU) | 0.04 | 0.0% |

**Dominant: backward (47%).**

## Time per stage — sciencebenchmark/RGAT (profiling on, CUDA-synchronized)

Profiled training wall time 266.0 s. Shares are what matters here; absolute times are inflated by the synchronization.

| section | seconds | share |
|---|---|---|
| backward (incl. block recompute: RGAT 42.7 s) | 177.94 | 66.9% |
| rgat forward (net of graph_to) | 42.19 | 15.9% |
| encoder T5 forward (net of rgat) | 22.46 | 8.4% |
| decoder T5 forward | 9.71 | 3.7% |
| optimizer (step+sched+zero_grad) | 9.57 | 3.6% |
| other (loss, Python, logging, unaccounted) | 2.47 | 0.9% |
| data_wait | 0.67 | 0.3% |
| graph_to (graph.to(device), per RGAT layer, forward) | 0.52 | 0.2% |
| lm_head | 0.34 | 0.1% |
| prepare_inputs (CPU->GPU) | 0.07 | 0.0% |
| graph_build (graph batch on CPU) | 0.06 | 0.0% |

**Dominant: backward (incl. block recompute: RGAT 42.7 s) (67%).**

## Cost estimate

| cell | s/sample | steps/epoch (GA 8/16/32/64) | min/epoch | US$/epoch | trial (3 ep) | 6 trials | final | dev eval | cell total |
|---|---|---|---|---|---|---|---|---|---|
| spider_rgat | 0.264 | 918/459/229/114 | 33.2 | 0.410 | 1.67 h / $1.23 | 10.0 h / $7.41 | 2.2–8.3 h | 0.06 h | 5.6–18.4 h / $4.18–13.61 |
| spider_plain | 0.158 | 918/459/229/114 | 19.7 | 0.243 | 0.99 h / $0.73 | 5.9 h / $4.40 | 1.3–4.9 h | 0.06 h | 3.4–10.9 h / $2.50–8.09 |
| sciencebenchmark_rgat | 0.783 | 529/264/132/66 | 56.9 | 0.701 | 2.85 h / $2.11 | 17.1 h / $12.66 | 3.8–14.2 h | 0.11 h | 9.6–31.4 h / $7.13–23.26 |
| sciencebenchmark_plain | 0.356 | 529/264/132/66 | 25.7 | 0.317 | 1.29 h / $0.96 | 7.8 h / $5.74 | 1.7–6.4 h | 0.11 h | 4.4–14.3 h / $3.29–10.59 |

**4 cells: 23.1–75.1 GPU-hours, US$ 17.10–55.55 at US$ 0.74/h** (upper bound: no pruning, 15-epoch finals).

Checkpoint save (t5-base + RGAT weights, Adafactor state, RNG): 1.7 s per epoch. One-time HF datasets cache build on a fresh volume (not repeated per run): spider 1 s, sciencebenchmark 6 s. Startup uses the warm-cache value. Assumptions: `cost_estimate.assumptions` in the JSON.

## Recommendation for the next phase-B experiment (not applied)

**spider:** Next experiment: TF32 -- it acts on 20%-68% of the profiled step time (matmul-heavy T5 parts (forward + their share of backward); smallest numeric change of the precision options). Input path and graph copies take 1.1% together, so semantics-preserving changes there cannot pay off. Precision changes need a protocol deviation and their own numeric check (T5/T6/T10-style) before any study run. Caveat: the range of BF16 autocast, RGAT in pure PyTorch (replace DGL kernels) overlaps with TF32's -- backward (47% of the step) is not split by component, so this profile cannot rank them; a direct A/B probe (one change at a time, same cells) decides. Per-run startup (data, graph store, model; warm cache) is 0.4 min, paid by each of the 8 runs of a cell (3.1 min in total) and not accelerated by the GPU.

| candidate | share of step it acts on (min–max) | note |
|---|---|---|
| TF32 | 20.5%–67.7% | matmul-heavy T5 parts (forward + their share of backward); smallest numeric change of the precision options |
| BF16 autocast | 20.5%–67.7% | same parts as TF32, larger speedup and larger numeric change; after TF32 |
| RGAT in pure PyTorch (replace DGL kernels) | 17.4%–64.7% | large change; needs T3/T5/T6 re-validation |
| torch.compile | 3.3%–3.3% | Python overhead; DGL ops force graph breaks; last |
| DataLoader (workers/pin_memory/prefetch) | 0.6%–0.6% | CPU-side input path; order and content must stay identical |
| remove repeated graph.to(device) (once per step) | 0.5%–0.5% | semantics-preserving; today the graph is copied once per RGAT layer |

**sciencebenchmark:** Next experiment: RGAT in pure PyTorch (replace DGL kernels) -- it acts on 16%-83% of the profiled step time (large change; needs T3/T5/T6 re-validation). Input path and graph copies take 0.5% together, so semantics-preserving changes there cannot pay off. Precision changes need a protocol deviation and their own numeric check (T5/T6/T10-style) before any study run. Caveat: the range of TF32, BF16 autocast overlaps with RGAT in pure PyTorch (replace DGL kernels)'s -- backward (67% of the step) is not split by component, so this profile cannot rank them; a direct A/B probe (one change at a time, same cells) decides. Per-run startup (data, graph store, model; warm cache) is 0.4 min, paid by each of the 8 runs of a cell (3.5 min in total) and not accelerated by the GPU.

| candidate | share of step it acts on (min–max) | note |
|---|---|---|
| RGAT in pure PyTorch (replace DGL kernels) | 15.9%–82.8% | large change; needs T3/T5/T6 re-validation |
| TF32 | 12.2%–79.1% | matmul-heavy T5 parts (forward + their share of backward); smallest numeric change of the precision options |
| BF16 autocast | 12.2%–79.1% | same parts as TF32, larger speedup and larger numeric change; after TF32 |
| torch.compile | 0.9%–0.9% | Python overhead; DGL ops force graph breaks; last |
| DataLoader (workers/pin_memory/prefetch) | 0.3%–0.3% | CPU-side input path; order and content must stay identical |
| remove repeated graph.to(device) (once per step) | 0.2%–0.2% | semantics-preserving; today the graph is copied once per RGAT layer |

