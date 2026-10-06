# The t5-base study on the RTX 4090 (RunPod)

The study of `PROTOCOLO.md` (4 cells: Spider and ScienceBenchmark, RGAT and plain), run by `graphix_modern/study.py` on the modern stack, in strict FP32, through `scripts/runpod_study.sh`.

- **Validation before the pod:** the gate TE (epoch-level control, 38/38) and the end-to-end test TS (4-cell smoke with real kills, resume and idempotence, 77/77) passed locally on the a5 image. See `PORTABILIDADE.md`, "Orquestrador do estudo na stack moderna", and `docs/port/orchestrator/`.
- **Cost:** the phase-B baseline estimates 4.5–14 GPU-hours (US$ 3.3–10.4) for the whole study (`docs/port/phase_b/4090/`).

> Never run `scripts/run_t5base_study_pod.sh` or `seq2seq/run_t5base_study.py` on this image (legacy entrypoint, see `docs/incidentes.md`).

## What the launcher guarantees

**Before starting**, `start` checks two things:
- The orchestrator and the training code are **byte-identical** to the files validated by TE/TS (`docs/port/orchestrator/tested_files.sha256`).
- The full **preflight** passes (`scripts/runpod_preflight.sh`):
  - volume and free space (at least 30 GB);
  - commit equal to `EXPECT_COMMIT`;
  - data hashes;
  - RTX 4090 with capability 8.9;
  - library versions and the image's environment fingerprint;
  - strict FP32, including the empirical matmul test.

If either check fails, nothing starts.

**While running:**
- **Detached:** the study runs in the background (`setsid` + `nohup`, stdin closed), so **closing the web terminal does not stop it**.
- **State on the volume:** the PID, logs and state live under the study root, `/workspace/study` (or `/workspace/study_smoke` for the smoke):
  - `launcher/pid`, `launcher/state.json`, `launcher/orchestrator.log`;
  - the monitors, `launcher/gpu.csv` and `launcher/cpu.csv`;
  - the orchestrator's own outputs: `events.jsonl`, `optuna/`, `trials/`, `final/`.
- **One study per root:** a second `start` while one is running is refused.
- **Strict FP32:** every training/evaluation run goes through `scripts/runpod/entry.py`, whose guard fails the run if TF32, autocast, non-"highest" matmul precision, or non-fp32/non-CUDA parameters ever appear.

**On restart** (after `stop`, a crash, or a stopped pod), running `start` again resumes, with the validated semantics:
- a trial interrupted while running is marked **FAIL** and counts towards the 6 trials (as in the legacy code);
- the final run resumes from its **last complete checkpoint**;
- an incomplete checkpoint (interrupted during its save) is **moved aside**, never deleted;
- the early-stopping counter restarts on resume (as in the legacy code); the best checkpoint and metric are kept;
- every finished stage is skipped.

## 1. Code on the volume (from this machine, pod stopped)

As in `docs/RUNPOD_PHASE_B.md` §3: one new folder per commit, via the S3 API, never `--delete`, then check its hash manifest on the pod.

## 2. Pod

- **Image:** `silveirabruno/graphix-modern@sha256:371f61af7521069e62bb8973c2c42584377659440b5902d025876fb94d859037`.
- **GPU:** 1× RTX 4090, EU-RO-1.
- **Volume:** the network volume at `/workspace`.
- **Access:** use the web terminal.

Then create the two symlinks in the new folder:
```bash
P=/workspace/project/graphix-t5-scibench-<sha>
cd $P && sha256sum -c --quiet ../graphix-t5-scibench-<sha>.sha256 && echo CODE OK
ln -sfn /workspace/data            $P/data_all_in/data
ln -sfn /workspace/data/t5-base-st $P/data_all_in/t5-base-st
```

## 3. Smoke on the 4090 (before the real study)

Same orchestrator on tiny data, in its own root (`/workspace/study_smoke`), all 4 cells: 2 trials of 1 epoch, a 2-epoch final, 20 dev examples.
```bash
cd /workspace/project/graphix-t5-scibench-<sha>
EXPECT_COMMIT=<sha> bash scripts/runpod_study.sh start --smoke
```
`start` prints the preflight, then `STUDY STARTED in the background`.

To prove the detachment, **close the web terminal**, open a new one, and run:
```bash
cd /workspace/project/graphix-t5-scibench-<sha>
bash scripts/runpod_study.sh status --smoke
```
The smoke is done when `status --smoke` shows `launcher   : finished (rc=0)`, and every cell shows `done ep 2` under `final` and EM/EX under `dev`. Send that output.

## 4. The real study

Only after the smoke has passed:
```bash
cd /workspace/project/graphix-t5-scibench-<sha>
EXPECT_COMMIT=<sha> bash scripts/runpod_study.sh start
```

## Following it (read-only, does not touch the process)

```bash
bash scripts/runpod_study.sh status          # state, PID, per-cell trials (complete/pruned/fail), final, dev
bash scripts/runpod_study.sh logs            # follows the orchestrator log; Ctrl-C stops only the tail
bash scripts/runpod_study.sh logs run        # follows the log of the training/eval run in progress
nvidia-smi                                   # or: watch -n 5 nvidia-smi
```
Add `--smoke` right after the subcommand for the smoke root.

`status` reads plain files only. It never opens the Optuna database, never sends signals and never uses the GPU.

## Stopping, and the end

- **To interrupt:** `bash scripts/runpod_study.sh stop` (SIGTERM to the study's process group). Resume later with `start`, which follows the semantics above.
- **At the end**, the log and `status` show `STUDY finished` and **`STOP THE POD NOW`**. Stop the pod (stop, never delete), then bring back `/workspace/study/` through the S3 API.
- **If the pod is stopped mid-study:** `status` says the runner is gone, and `start` resumes.
