# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Graphix-T5: a graph-aware T5 model for text-to-SQL parsing (Spider benchmark), built by injecting relational-graph-attention (RGAT) layers into the T5 encoder so it can reason over question/schema graphs, not just token sequences. Reference checkpoint: Graphix-3B + PICARD (SOTA on Spider at publication time). See `README.md` for the paper citation and background images (`graphix.png`, `dp.png`, `graphix-3b-picard.png`).

Everything here is designed to run inside the `eyuansu62/graphix-text-to-sql:v2` Docker image (built from `Dockerfile`, which also carries a large Thrift/PICARD/C++ toolchain for the PICARD-constrained-decoding integration). There is no supported bare-metal Python setup — always go through `make`.

## Commands

All entrypoints are `make` targets that wrap `docker run --gpus all` (see `Makefile`). Each mounts specific host directories into `/app`, `/train`, `/train_data`, `/transformers_cache`, `/stanza_resources`, `/eval`, `/train_db_id` so large assets (Spider data, T5 weights, Stanza models, checkpoints) stay on the host instead of being baked into the image.

```bash
make pull-train-image   # docker pull eyuansu62/graphix-text-to-sql:v2
make pre_process        # run the full preprocessing DAG (data_all_in/run/run_pre.sh)
make train              # seq2seq/run_seq2seq_train.py configs/train.json
make eval               # seq2seq/run_seq2seq_eval.py configs/eval.json (PICARD via use_picard/launch_picard)
```

There is no lint/build step; `tests/` (`test_dataset.py`, `test_picard_client.py`) are pytest-based and exercise `seq2seq/utils/dataset_loader.py` and the PICARD client — they assume the same dependency set as the training image (transformers, torch, PICARD), so run them inside the container, not on the host.

Docker containers run as `--user $(shell id -u):$(shell id -g)` (not the original hardcoded `13011:13011`) so output files land owned by the invoking host user instead of an unrelated UID.

## Preprocessing pipeline (the part most likely to need debugging)

`make pre_process` runs `data_all_in/run/run_pre.sh`, which chains four shell scripts as **plain sequential `python3` calls with no `set -e`** — a failure in one stage does not stop the next stage from starting, and does not stop later stages from silently consuming whatever partial/empty output the failed stage left behind. When a stage fails, check timestamps/sizes in `data_all_in/data/output/` against this DAG rather than trusting the shell's exit status:

1. **`run_syntax.sh`** — merges `train_spider.json` + `train_others.json` (`merge_train.py`), then for each of train/dev: `preprocess/process_dataset.py` (schema linking, DB introspection over the 166 Spider SQLite DBs → `tables.bin`, `train.bin`/`dev.bin`) → `preprocess/inject_syntax.py` (Stanza dependency parsing → `train_syntax.json`/`dev_syntax.json`, using `relation_prompt.py`'s relation vocabulary).
2. **`run_preprocessing_training.sh`** / **`run_preprocessing_dev.sh`** — each: `seq2seq/run_peteshaw_{train,dev}.py` (schema serialization in "peteshaw" format) → `map_subword_question.py` → `map_subword_schema.py` → `map_subword_schema_linking.py`, all three reading and rewriting the *same* `seq2seq_{train,dev}_dataset.bin` path in place. These three scripts (backed by shared logic in `map_subword_serialize.py`) convert word-level relation matrices into subword-level matrices, matching T5's subword tokenization to Stanza's word-level dependency parse. Each script loads the entire split into memory, mutates every example, and does one `pickle.dump()` at the end — **there is no incremental/streaming write**, so peak RAM scales with full split size (train's 8577 examples is ~8x dev's 1034).
3. **`Graph_Processing.py`** — converts subword relation matrices into actual `dgl.graph()` objects (`SubwordGraphProcessor`), drops the now-redundant matrix fields per example, and writes both `seq2seq_{train,dev}_dataset.json` (the slimmed final dataset) and `graph_pedia_{train,dev}.bin` (the graph objects, keyed by example index).
4. **`run_graph_all.sh`** — `graph_pedia_merge.py` merges `graph_pedia_train.bin` + `graph_pedia_dev.bin` into `graph_pedia_total.bin`.

**Known constraint**: on a host with ~27GB RAM, step 2's train-side accumulate-then-pickle pattern OOMs (confirmed via `/var/log/kern.log` — kernel OOM-killer, not an assertion or code bug) well before reaching step 3; dev's smaller split does not hit this. Any fix must not change the per-example transformation logic in `map_subword_*.py`/`Graph_Processing.py` — those are the parts that have to stay faithful to the published method — sharding/streaming the *orchestration* (how many examples are held in memory at once) is the safe lever.

`map_subword_serialize.py::question_subword_matrix` uses `tokenizer(question_words, is_split_into_words=True, ...)` rather than tokenizing a joined string — this is intentional and load-bearing: joining pre-tokenized words with spaces and re-tokenizing does not reproduce the same word boundaries the upstream word-level relation matrices were built against, which breaks the `assert len(processed_question_toks) + 1 == len(question_dict)` a few lines below. Don't "simplify" this back to string-joining.

Local, no-download model/resource paths that preprocessing configs (`configs/data_pre_train.json`, `configs/data_pre_dev.json`) and shell scripts point at instead of downloading from the Hub: `data_all_in/t5-large/` (tokenizer/model), `stanza_resources/` (mounted at `/stanza_resources` with `STANZA_RESOURCES_DIR` set, and `DGLBACKEND=pytorch`/`HOME=/tmp` set to stop DGL from trying to write to a non-writable home). `data_all_in/data/spider/` already contains Spider (166 SQLite DBs under `database/`) — do not re-download any of these.

## ScienceBenchmark (new dataset, download prep done, not yet wired into the pipeline)

`data_all_in/download_sciencebenchmark.sh` fetches [ScienceBenchmark](https://github.com/ckosten/sciencebenchmark_dataset) — 3 real-world DBs (`cordis`/EU research projects, `oncomx`/biomedical, `sdss`/astronomy) — into `data_all_in/data/sciencebenchmark/` (gitignored, same untracked-data convention as `data_all_in/data/spider/`). It pulls the JSON splits (`dev.json`/`seed.json`/`synth.json`/`tables.json` per DB) straight from GitHub, and the 3 PostgreSQL dumps from Google Drive via `gdown`, run out of a local `.venv-tools/` venv (the host Python is externally-managed / PEP 668, so `pip install --user` fails — `gdown` has to live in that venv). The data is already downloaded (~7.8GB in `postgres_dumps/`); re-running the script is a no-op (it skips files that already exist).

Compared to Spider, ScienceBenchmark is depth-over-breadth: only 3 DBs (vs Spider's 166) but each is much larger (6–25 tables vs Spider's average 5.3), and total volume is smaller (300 seed + 4,432 synth + 299 dev = 5,031 examples vs Spider's 11,840). The example JSON already carries a `sql` field in the same parse-tree shape `preprocess/process_sql.py`/Spider produce, and `tables.json` uses Spider's exact field names (`table_names_original`, `column_names_original`, `column_types`, `foreign_keys`, `primary_keys`), so format-wise it should slot into `process_dataset.py` with little friction.

**The one real blocker**: `data_all_in/preprocess/common_utils.py` and `common_utils_proton.py` hardcode `sqlite3.connect(db_dir/db_id/db_id.sqlite)` for DB introspection — they cannot open the Postgres dumps ScienceBenchmark ships. Before any `run_syntax.sh`-style preprocessing can run on this data, the dumps need to be (1) restored into a real Postgres server (`CREATE EXTENSION pg_trgm;` first, then `pg_restore`) and (2) converted to SQLite files at `data_all_in/data/sciencebenchmark/database/<db_id>/<db_id>.sqlite` (db_ids: `cordis_temporary`, `oncomx_v1_0_25_small`, `skyserver_dr16_2020_11_30`). Neither step is automated yet.

## Model architecture

`seq2seq/models/modeling_t5.py` is a fork of HF's `T5ForConditionalGeneration` with graph awareness spliced into the encoder stack:
- `T5LayerRGAT` wraps an `RGAT_Layer` (imported from `.graphix.rgat_tuning`) and is invoked from `T5Block.forward` via a `graph_batch` argument threaded all the way down from `T5ForConditionalGeneration.forward` → `T5Stack.forward` → each block.
- `graph_batch` is a list of per-example dicts (the `graph_pedia` entries produced by `Graph_Processing.py`: `graph`, `edges`, `question_mask`/`schema_mask`, `question_subword_dict`, `schema_to_ids`, `node_idx`, ...), consumed in `T5LayerRGAT.graph_caption` to inject structural (RGAT) representations into the token hidden states before they continue through the normal FFN/attention path.
- `seq2seq/utils/dataset_graph.py::TokenizedDataset` is what pairs each tokenized example back up with its `graph_pedia` entry at `__getitem__` time.

`configs/train.json` and `configs/eval.json` are the two run configs (`data_training_args`/`training_args` in HF `Seq2SeqTrainingArguments` style): note `schema_serialization_type: "peteshaw"` must match what the preprocessing scripts produced, and `eval.json`'s `use_picard`/`launch_picard`/`picard_*` keys gate the PICARD constrained-decoding integration (`seq2seq/utils/picard_model_wrapper.py`, the `picard/` submodule, `picard.thrift`).

## Hardware-driven eval constraints

Graphix-3B (`train_db_id/Graphix-3B/pytorch_model.bin`, ~14GB, tracked via Git LFS) is too large to run at the original `eval.json` settings (`per_device_eval_batch_size: 4`, `num_beams: 4`) on an 8GB-VRAM GPU. When adapting eval for constrained VRAM, reduce batch size and beams explicitly in a copy of the config rather than assuming `fp16` alone closes the gap — validate with a single example / batch size 1 / beam 1 before scaling up.
