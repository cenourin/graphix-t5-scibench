# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal fork of Graphix-T5 (vendored from `AlibabaResearch/DAMO-ConvAI`'s `graphix/` subdirectory — see the fork notice in `README.md`), used for an undergraduate thesis (TCC) that starts from the Spider-trained Graphix-T5 baseline and evaluates/adapts it toward [ScienceBenchmark](https://github.com/ckosten/sciencebenchmark_dataset) (3 real-world DBs: `cordis`, `oncomx`, `sdss`). The base architecture is a graph-aware T5 for text-to-SQL: relational-graph-attention (RGAT) layers injected into the T5 encoder so it can reason over question/schema graphs, not just token sequences.

**This is the canonical repo for the TCC** — `origin` points to `git@github.com-cenourin:cenourin/graphix-t5-scibench.git` (the author's own GitHub), not upstream Alibaba. If you ever find yourself working out of a sibling directory that's a full clone of `AlibabaResearch/DAMO-ConvAI`, that's leftover scratch space from before this fork existed — don't commit or push there.

Everything here is designed to run inside the `eyuansu62/graphix-text-to-sql:v2` Docker image (built from `Dockerfile`, which also carries a large Thrift/PICARD/C++ toolchain for the PICARD-constrained-decoding integration). There is no supported bare-metal Python setup for training/eval — always go through `make` or an equivalent `docker run`. (A `.venv-tools/` venv exists on the host for lightweight tooling only — `gdown` for dataset downloads, `matplotlib` for the plots under `docs/figures/` — not for running the model.)

## Commands

All entrypoints are `make` targets that wrap `docker run --gpus all` (see `Makefile`). Each mounts specific host directories into `/app`, `/train`, `/train_data`, `/transformers_cache`, `/stanza_resources`, `/eval`, `/train_db_id` so large assets (Spider + ScienceBenchmark data, T5 weights, Stanza models, checkpoints) stay on the host instead of being baked into the image.

```bash
make pull-train-image   # docker pull eyuansu62/graphix-text-to-sql:v2
make pre_process        # run the full preprocessing DAG (data_all_in/run/run_pre.sh) -- Spider
make train              # seq2seq/run_seq2seq_train.py configs/train.json -- Spider, t5-base+RGAT
make eval               # seq2seq/run_seq2seq_eval.py configs/eval.json (PICARD via use_picard/launch_picard)
```

ScienceBenchmark and ablation runs (t5-small/t5-base, with/without RGAT) don't have dedicated `make` targets yet — they're launched as direct `docker run -d --name <run>` commands using the same mounts as `make train`, pointed at one of the `configs/train_sciencebenchmark_*.json` / `configs/train_spider_*.json` files (see "Experiment configs" below), plus two env vars the Makefile's `train` target doesn't set: `DGLBACKEND=pytorch` and `HOME=/tmp` (without them DGL tries to write its default-backend config to a non-writable `/.dgl` and crashes before training starts).

There is no lint/build step; `tests/` (`test_dataset.py`, `test_picard_client.py`) are pytest-based and exercise `seq2seq/utils/dataset_loader.py` and the PICARD client — they assume the same dependency set as the training image (transformers, torch, PICARD), so run them inside the container, not on the host.

Docker containers run as `--user $(shell id -u):$(shell id -g)` (not the original hardcoded `13011:13011`) so output files land owned by the invoking host user instead of an unrelated UID.

## Preprocessing pipeline (Spider) — the part most likely to need debugging

`make pre_process` runs `data_all_in/run/run_pre.sh`, which chains four shell scripts as **plain sequential `python3` calls with no `set -e`** — a failure in one stage does not stop the next stage from starting, and does not stop later stages from silently consuming whatever partial/empty output the failed stage left behind. When a stage fails, check timestamps/sizes in `data_all_in/data/output/` against this DAG rather than trusting the shell's exit status:

1. **`run_syntax.sh`** — merges `train_spider.json` + `train_others.json` (`merge_train.py`), then for each of train/dev: `preprocess/process_dataset.py` (schema linking, DB introspection over the 166 Spider SQLite DBs → `tables.bin`, `train.bin`/`dev.bin`) → `preprocess/inject_syntax.py` (Stanza dependency parsing → `train_syntax.json`/`dev_syntax.json`, using `relation_prompt.py`'s relation vocabulary).
2. **`run_preprocessing_training.sh`** / **`run_preprocessing_dev.sh`** — each: `seq2seq/run_peteshaw_{train,dev}.py` (schema serialization in "peteshaw" format) → `map_subword_question.py` → `map_subword_schema.py` → `map_subword_schema_linking.py`, all three reading and rewriting the *same* `seq2seq_{train,dev}_dataset.bin` path in place. These three scripts (backed by shared logic in `map_subword_serialize.py`) convert word-level relation matrices into subword-level matrices, matching T5's subword tokenization to Stanza's word-level dependency parse. Each script loads the entire split into memory, mutates every example, and does one `pickle.dump()` at the end — **there is no incremental/streaming write**, so peak RAM scales with full split size (train's 8577 examples is ~8x dev's 1034).
3. **`Graph_Processing.py`** — converts subword relation matrices into actual `dgl.graph()` objects (`SubwordGraphProcessor`), drops the now-redundant matrix fields per example, and writes both `seq2seq_{train,dev}_dataset.json` (the slimmed final dataset) and `graph_pedia_{train,dev}.bin` (the graph objects, keyed by example index).
4. **`run_graph_all.sh`** — `graph_pedia_merge.py` merges `graph_pedia_train.bin` + `graph_pedia_dev.bin` into `graph_pedia_total.bin`. Spider's `run_seq2seq_train.py` defaults (`GRAPHIX_TRAIN_GRAPH_PEDIA_PATH`/`GRAPHIX_EVAL_GRAPH_PEDIA_PATH`) both point at this single merged file — train and dev share it, disambiguated by `graph_idx` offset (see `inject_syntax.py`), so unlike ScienceBenchmark there's no separate train/dev graph_pedia file to override for Spider runs.

**Known constraint**: step 2's train-side accumulate-then-pickle pattern can OOM on a host with limited RAM (confirmed via kernel OOM-killer on a ~27GB-RAM host) well before reaching step 3; dev's smaller split does not hit this. Any fix must not change the per-example transformation logic in `map_subword_*.py`/`Graph_Processing.py` — those are the parts that have to stay faithful to the published method — sharding/streaming the *orchestration* (how many examples are held in memory at once) is the safe lever.

`map_subword_serialize.py::question_subword_matrix` uses `tokenizer(question_words, is_split_into_words=True, ...)` rather than tokenizing a joined string — this is intentional and load-bearing: joining pre-tokenized words with spaces and re-tokenizing does not reproduce the same word boundaries the upstream word-level relation matrices were built against, which breaks the `assert len(processed_question_toks) + 1 == len(question_dict)` a few lines below. Don't "simplify" this back to string-joining.

Local, no-download model/resource paths that preprocessing configs (`configs/data_pre_train.json`, `configs/data_pre_dev.json`, and their `*_sciencebenchmark.json` counterparts) and shell scripts point at instead of downloading from the Hub: `data_all_in/t5-large/` (tokenizer/model used to pre-tokenize for graph construction), `data_all_in/t5-base/` and `data_all_in/t5-small/` (stock HF checkpoints used as fine-tuning starting points — see "Experiment configs"), `stanza_resources/` (mounted at `/stanza_resources` with `STANZA_RESOURCES_DIR` set). `data_all_in/data/spider/` already contains Spider (166 SQLite DBs under `database/`) and `data_all_in/data/sciencebenchmark/` already contains ScienceBenchmark (3 SQLite DBs under `database/`, converted from the original Postgres dumps) — do not re-download any of these.

## ScienceBenchmark — fully wired in, extensively evaluated

ScienceBenchmark preprocessing runs the same DAG shape as Spider (`data_all_in/run/run_syntax_sciencebenchmark.sh`, `run_preprocessing_{training,dev}_sciencebenchmark.sh`) against `data_all_in/data/sciencebenchmark/`, producing `data_all_in/data/sciencebenchmark/output/{seq2seq_train_dataset.json, seq2seq_dev_dataset.json, graph_pedia_train.bin, graph_pedia_dev.bin}` — 4732 train examples (seed+synth combined), 299 dev examples across `cordis`/`oncomx`/`sdss`. Unlike Spider, ScienceBenchmark's train and dev graph_pedia are **two separate files** (both 0-indexed by `graph_idx`, so a merged file would silently collide) — any run must set `GRAPHIX_TRAIN_GRAPH_PEDIA_PATH`/`GRAPHIX_EVAL_GRAPH_PEDIA_PATH` (and the matching `GRAPHIX_TRAIN_DATASET_PATH`/`GRAPHIX_EVAL_DATASET_PATH`) env vars pointing at these files explicitly; `run_seq2seq_train.py`'s defaults are Spider's paths.

The original Postgres→SQLite conversion blocker (dumps needed restoring into Postgres then converting to SQLite at `database/<db_id>/<db_id>.sqlite`, db_ids `cordis_temporary`/`oncomx_v1_0_25_small`/`skyserver_dr16_2020_11_30`) is resolved — the SQLite files already exist under `data_all_in/data/sciencebenchmark/database/` on the training host. They're gitignored (too large for git — `skyserver_dr16_2020_11_30.sqlite` alone is 15GB) and not reproducible by rerunning `download_sciencebenchmark.sh` alone (that script only fetches the Postgres dumps, not the converted SQLite files); a fresh checkout instead pulls a prebuilt data-only Docker image — see `docker/scibench-data/README.md` for the pull/rebuild commands and why that image's `skyserver` layer is split into ~500MB chunks.

`oncomx`'s schema graphs are uniformly large: all 1165 training examples produce RGAT graphs between 1022–1308 nodes (no small subset exists to sample from for a VRAM-constrained run). `GRAPHIX_MAX_GRAPH_NODES` (default 512, read once at module load in `run_seq2seq_train.py`) filters training examples by graph size *before* the model variant is even selected, so it applies to plain (non-RGAT) runs too even though they don't need it — set it high (1400 for ScienceBenchmark, 900 covers Spider's max of 809) to avoid dropping examples a plain run has no VRAM reason to exclude.

## Model architecture

`seq2seq/models/modeling_t5.py` is a fork of HF's `T5ForConditionalGeneration` with graph awareness spliced into the encoder stack:
- `T5LayerRGAT` wraps an `RGAT_Layer` (imported from `.graphix.rgat_tuning`) and is invoked from `T5Block.forward` via a `graph_batch` argument threaded all the way down from `T5ForConditionalGeneration.forward` → `T5Stack.forward` → each block. Both `T5LayerRGAT` and its `relation_emb` embedding derive their dimensions from `config.d_model` generically — the RGAT architecture isn't hardcoded to t5-base's 768; it works with t5-small's 512 too (untrained from scratch, since only a t5-base+RGAT checkpoint has ever been pretrained on Spider).
- `graph_batch` is a list of per-example dicts (the `graph_pedia` entries produced by `Graph_Processing.py`: `graph`, `edges`, `question_mask`/`schema_mask`, `question_subword_dict`, `schema_to_ids`, `node_idx`, ...), consumed in `T5LayerRGAT.graph_caption` to inject structural (RGAT) representations into the token hidden states before they continue through the normal FFN/attention path.
- When `gradient_checkpointing` is enabled, `T5Stack.forward`'s `torch.utils.checkpoint.checkpoint()` call wraps each block in a closure that must explicitly re-pass `graph_batch`/`relation_emb` (captured by Python closure, not through checkpoint's own tensor args) — omitting them silently drops RGAT during checkpointed training. `seq2seq/models/graphix/rgat.py::Model.gradient_checkpointing_enable()` must also delegate to `self.pretrain_model` explicitly, since `Model` is a plain `PreTrainedModel` wrapper (not a `T5PreTrainedModel`) and the HF default `gradient_checkpointing_enable()`'s `self.apply(...)` cascade never reaches the wrapped model's real `T5Stack` instances otherwise.
- `seq2seq/utils/dataset_graph.py::TokenizedDataset` is what pairs each tokenized example back up with its `graph_pedia` entry at `__getitem__` time — it needs `graph_pedia` (for `new_struct_in`, the serialized schema text) regardless of whether RGAT is actually used, so even the plain ablation below still requires it in the data pipeline.

**`GRAPHIX_MODEL_VARIANT` env var** (`run_seq2seq_train.py`) selects between `models.graphix.rgat.Model` (default, RGAT-aware) and `models.graphix.plain.Model` (`GRAPHIX_MODEL_VARIANT=plain` — a standard `T5ForConditionalGeneration`, no graph structure at all) for a no-RGAT ablation baseline, reusing the same `TokenizedDataset` pipeline.

`configs/train.json` and `configs/eval.json` are the reference Spider run configs (`data_training_args`/`training_args` in HF `Seq2SeqTrainingArguments` style): note `schema_serialization_type: "peteshaw"` must match what the preprocessing scripts produced, and `eval.json`'s `use_picard`/`launch_picard`/`picard_*` keys gate the PICARD constrained-decoding integration (`seq2seq/utils/picard_model_wrapper.py`, the `picard/` submodule, `picard.thrift`) — as of this writing PICARD doesn't actually intercept generation for the RGAT-aware model path (`seq2seq/models/graphix/rgat_picard.py` bypasses `model_cls_wrapper`), a known integration gap documented in `docs/sciencebenchmark-eval-results.tex`.

## Experiment configs (ScienceBenchmark + Spider ablations)

`configs/train_sciencebenchmark_*.json` and `configs/train_spider_*.json` are the ablation/fine-tuning configs used for the TCC, beyond the two reference configs above:
- `train_sciencebenchmark_t5small.json`, `train_sciencebenchmark_t5base_plain.json` — no-RGAT baselines (`GRAPHIX_MODEL_VARIANT=plain`) on ScienceBenchmark.
- `train_sciencebenchmark_t5small_rgat.json` — t5-small **with** RGAT (untrained from scratch) on ScienceBenchmark.
- `train_sciencebenchmark_full_gc.json` — t5-base+RGAT full fine-tune on ScienceBenchmark, all domains including `oncomx`, requires `gradient_checkpointing: true` (see above) to fit in 8GB VRAM.
- `train_sciencebenchmark_fewshot.json` — small balanced sample (140 cordis + 140 sdss + 20 smallest-graph oncomx), more epochs.
- `train_spider_t5small_plain.json`, `train_spider_t5base_plain.json` — no-RGAT baselines on Spider, for comparison against the RGAT-pretrained reference checkpoint.

All of these were launched as direct `docker run -d` commands (not `make` targets) — see git log / `docs/sciencebenchmark-eval-results.tex` for the exact invocations and results. Every ScienceBenchmark config needs the `GRAPHIX_*_PATH` env vars from the section above; every plain-variant config needs `GRAPHIX_MODEL_VARIANT=plain`; all need `DGLBACKEND=pytorch`/`HOME=/tmp`.

## Hardware-driven eval constraints

Graphix-3B (`train_db_id/Graphix-3B/pytorch_model.bin`, ~14GB, tracked via Git LFS) is too large to run at the original `eval.json` settings (`per_device_eval_batch_size: 4`, `num_beams: 4`) on an 8GB-VRAM GPU. When adapting eval for constrained VRAM, reduce batch size and beams explicitly in a copy of the config rather than assuming `fp16` alone closes the gap — validate with a single example / batch size 1 / beam 1 before scaling up. All ScienceBenchmark/ablation training in this repo runs on the same single 8GB-VRAM consumer GPU (no tensor cores), which is why `gradient_checkpointing`, `GRAPHIX_MAX_GRAPH_NODES`, and small effective batch sizes (`per_device_train_batch_size=1` + `gradient_accumulation_steps=32`) show up throughout the configs above.

## TCC results and write-up

`docs/sciencebenchmark-eval-results.tex` (+ compiled `.pdf`, + `docs/figures/*.png`/`.pdf`) is the canonical, actively-maintained results document — zero-shot/beam4/PICARD eval on ScienceBenchmark, all fine-tuning attempts (with their metric tables, training curves, per-config error matrices, and documented methodological limitations), and the RGAT-vs-plain-T5 ablation findings. Compile it standalone with a `\documentclass{article}` + `\usepackage{graphicx,booktabs}` wrapper via `docker run texlive/texlive:latest pdflatex` (it has no preamble of its own — designed for `\input{}` into a larger TCC document). Check this file for exact current numbers rather than citing figures from memory — it is the source of truth and gets updated as new runs complete.
