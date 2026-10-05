# Phase B, first session on the RTX 4090 (RunPod): FP32 baseline

This session measures the ported stack (MODERN_A, validated in phase A; see `PORTABILIDADE.md`) on the RTX 4090 **without changing anything**:
- FP32 strict: TF32 off, autocast off, `torch.compile` off.
- Batch 1, the same code and image as phase A.

It does not start the study (no Optuna, no final training) and does not touch `PROTOCOLO.md`. Precision/speed changes come later, one at a time, after this baseline and a protocol deviation entry.

> **Never run `scripts/run_t5base_study_pod.sh` on this image.**
> - That launcher (and `seq2seq/run_t5base_study.py`, `seq2seq/run_optuna_search.py`) drives the **legacy** entrypoint `seq2seq/run_seq2seq_train.py`, which was never validated on the modern stack.
> - A missing `tenacity` import is the symptom of running it: only the legacy PICARD wrapper imports it.
> - The study orchestrator has not been ported to `graphix_modern/train.py` yet.

## 1. Pod

| Field | Value |
|---|---|
| Container image | `silveirabruno/graphix-modern@sha256:371f61af7521069e62bb8973c2c42584377659440b5902d025876fb94d859037` (a5 = `phase-a-final`; by digest, never by tag alone) |
| GPU | 1× RTX 4090 in the network volume's datacenter (EU-RO-1) |
| Volume | the 100 GB network volume, mounted at `/workspace` |
| Environment variables | none required (`GRAPHIX_IMAGE_DIGEST` is filled in by the scripts; the check is the in-container fingerprint of the image's Python environment, `docker/modern/a5_env_fingerprint.json`) |

Nothing is installed in the container:
- The image has no `git`; the scripts read the commit from `.git/` directly.
- Outbound SSH ports are blocked on the local network, so work through the pod's **web terminal**.

## 2. Data (already on the volume, 2026-09-29)

`scripts/prepare_runpod_data.sh` sent the unversioned data through the RunPod S3 API: 450 files, 23.02 GB, manifest sha256 `5f40d547…f149a`, including the T10 references.

Validation, once, on the pod:
```bash
cd /workspace/data
sha256sum manifests/DATA_MANIFEST.sha256        # 5f40d54754a93f6718568bcc24afd5249a6eb76ca2494cc3ad5af95cf24f149a
wc -l < manifests/DATA_MANIFEST.sha256          # 450
LOG="manifests/verify_$(date -u +%Y%m%dT%H%M%SZ).log"
sha256sum -c manifests/DATA_MANIFEST.sha256 > "$LOG" 2>&1; echo "rc=$?"   # rc=0
grep -vc ': OK$' "$LOG"                          # 0
grep -c ': OK$' "$LOG"                           # 450
date -u > manifests/DATA_VERIFIED
```

## 3. Code on the pod (no git in the image)

The image has no `git`, and outbound SSH is blocked here, so code goes up the same way as the data: through the S3 API.

Each commit gets its **own new folder**, so no existing file is overwritten or deleted. A sync with deletion onto the old clone could, through the S3 API, reach the data behind the `data_all_in/data` symlink. The scripts find their repository from their own location.

On this machine, with the commit pushed:
```bash
cd /tmp && rm -rf gx && git clone -q --branch trained-models /home/user/text2sql/graphix-t5-scibench gx
SHA=$(git -C gx rev-parse --short HEAD); git -C gx fetch -q --tags
AWS_PROFILE=runpod /home/user/text2sql/graphix-t5-scibench/.venv-tools/bin/aws s3 sync --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io --no-progress gx s3://jhvgb9yz8y/project/graphix-t5-scibench-$SHA/
echo $SHA
```

On the pod (web terminal), for that folder:
```bash
P=/workspace/project/graphix-t5-scibench-<sha>
ln -sfn /workspace/data            $P/data_all_in/data
ln -sfn /workspace/data/t5-base-st $P/data_all_in/t5-base-st
```
The first clone (`/workspace/project/graphix-t5-scibench`, at `3463bf0`) stays as it is.

## 4. The session (one command)

```bash
cd /workspace/project/graphix-t5-scibench-<sha>
EXPECT_COMMIT=<sha> bash scripts/runpod_phase_b_probe.sh 2>&1 | tee /workspace/outputs/phase_b_s1_console.log
```

Stages, each with a timeout, stopping at the first failure (exit ≠ 0):

1. **preflight** (`scripts/runpod_preflight.sh`)
   - Volume mounted and writable, free space.
   - Commit (from `.git`) and tag.
   - Data: manifest hash, `DATA_VERIFIED`, re-hash of every non-SQLite file.
   - GPU = RTX 4090, capability 8.9, ≥ 23 GB, driver, GPU idle.
   - Exact library versions, and the image environment fingerprint.
   - CUDA in torch and in DGL.
   - TF32/compile/autocast off, plus an **empirical** fp32 matmul test: relative error below 1e-4, which TF32 would exceed.
2. **smoke** (Spider/RGAT, `graphix_modern/train.py` through `scripts/runpod/entry.py`)
   - Train 3 steps and write a checkpoint.
   - Resume from checkpoint-3 to step 6; the resumed process must execute exactly steps 4–6.
   - Eval of 16 dev examples with generation, exact match and execution.
3. **steps50**: T10 replayed exactly (same config, initial weights, dropout neutralized).
   - Steps, LR per step and the order of the 406 items must equal the 1070 reference.
   - The loss is compared with the T10 same-stack envelope (reported, not a gate).
4. **probe**: 4 cells × 250 optimizer steps (GA 8), dropout on, profiling **off**, then the study's validation (loss only, full val split).
5. **profile**: RGAT cells, 40 steps with `GRAPHIX_PROFILE=1` (CUDA sync at each section): time shares.
6. **devgen**: 32 dev examples with generation, for the dev-eval term of the cost.
7. **report**: `/workspace/benchmarks/4090_baseline_fp32.{json,md}`.
   - Hardware and versions, throughput, VRAM, GPU/CPU utilization (training windows).
   - Time per section, and the dominant bottleneck.
   - Cost estimate (`RUNPOD_GPU_USD_PER_HOUR`, default 0.74): per epoch, trial, 6 trials, final, 4 cells.
   - Comparison with the GTX 1070 baseline (`docs/port/phase_b/reference_1070/1070_baseline_fp32.json`).
   - A **recommendation** for the next phase-B test, not applied.
8. Model/optimizer weights of the session are deleted (configs, logs, curves and results are kept), and it prints **`STOP THE POD NOW`**.

Safety:
- One session at a time (`flock` on `/workspace/benchmarks`).
- A new `SESSION` name is required for a second full session, so outputs are never mixed.
- Everything is written to `/workspace` (outputs, checkpoints, benchmarks, cache).
- `entry.py` fails a run if autocast, TF32, non-"highest" matmul precision, non-fp32 parameters or a non-CUDA model show up at any forward.
- The RGAT path refuses to run without CUDA.
- Ctrl-C stops cleanly.

Monitoring (the session also samples GPU/CPU every second into `/workspace/outputs/<session>/monitor/`):
```bash
tail -f /workspace/outputs/phase_b_s1_console.log
nvidia-smi dmon -s pucm          # or: watch -n1 nvidia-smi
```

## 5. Expected time and cost

The same session ran on the GTX 1070 (`docs/port/phase_b/reference_1070/1070_baseline_fp32.md`) in about 1 h 12 min.

On the 4090, the first session also builds the HF datasets cache once:
- measured locally: 123 s for Spider and 699 s for ScienceBenchmark, which reads DB content from the 15 GB skyserver file;
- CPU-bound, so the 4090 does not speed it up;
- the cache stays in `/workspace/cache` for later sessions.

## 5a. Result (2026-10-05)

Session `phase_b_s2` passed every stage in 20 min; its report is in `docs/port/phase_b/4090/`. The study keeps strict FP32 (decision recorded in `PORTABILIDADE.md`). The first attempt, `phase_b_s1`, was interrupted and discarded (`docs/incidentes.md`).

A second full session needs a new name, given on the command line (`SESSION=phase_b_s3 EXPECT_COMMIT=... bash scripts/runpod_phase_b_probe.sh 2>&1 | tee /workspace/outputs/phase_b_s3_console.log`). Always keep the `tee`, and keep the web terminal open until `STOP THE POD NOW`.

## 6. After the session

1. Stop the pod as soon as `STOP THE POD NOW` appears (stop, don't delete; never delete the volume).
2. Bring back `/workspace/benchmarks/` and `/workspace/outputs/<session>/` (S3 API, `aws s3 sync`).
3. Decide the next phase-B experiment from the report, one change at a time. Any precision change needs its own equivalence check and a protocol deviation before any study run.
