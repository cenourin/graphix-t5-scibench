# Repository Guidelines

## Project Structure & Module Organization

The workspace combines standalone SchemaGraphSQL scripts with the nested `graphix-t5-scibench/` Git checkout. `schemagraphsql_sqlite.py` implements SQLite introspection, graph selection, SQL generation, and validation. `run_schemagraphsql_benchmark.py` runs inference; `evaluate_schemagraphsql_metrics.py` evaluates saved predictions offline. Root-level `test_*.py` files cover these components.

Prompts live in `SCHEMAGRAPHSQL_prompts.md` and `schemagraphsql_single_call_prompt.txt`. Evaluation artifacts belong under `results/`. Within Graphix, `seq2seq/` contains model code, `configs/` experiment settings, `data_all_in/` preprocessing and datasets, and `docs/` and `estudo/` reports and figures.

## Build, Test, and Development Commands

Run standalone commands from the workspace root using Python 3.10+ with `requests` installed; no compilation is required.

- `python3 -m unittest -v test_schemagraphsql_sqlite.py test_schemagraphsql_metrics.py`: run pipeline and metric tests.
- `python3 schemagraphsql_sqlite.py --sciencebenchmark cordis --inspect-schema`: inspect local tables without calling the API.
- `bash configurar_openwebui.sh`: configure credentials interactively in `.env`.
- `bash executar_schemagraphsql.sh --sciencebenchmark cordis --question "How many projects are there?" --mode paper --execute`: run one question.
- `python3 -u run_schemagraphsql_benchmark.py --output results/new_run`: run the benchmark using `.openwebui-key` by default.
- `python3 -u evaluate_schemagraphsql_metrics.py`: evaluate saved predictions without model calls.

Graphix has separate legacy dependencies. Follow its README and Docker setup before running `make pre_process`, `make train`, or `make eval` inside that directory; these workflows require datasets and GPU resources.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and variables, and `PascalCase` classes. Preserve existing type annotations and explicit error handling. Root scripts have no configured formatter; Graphix configures Black and isort with 120-character lines in `pyproject.toml`. Avoid unrelated formatting changes.

## Testing Guidelines

Root tests use `unittest`; Graphix tests use pytest. Name tests `test_*.py` and test methods `test_*`. Cover graph paths, read-only SQL enforcement, nested reference queries, and metric edge cases. The SDSS smoke test requires local data. No coverage threshold is configured. Keep live API checks separate from automated tests.

## Commit & Pull Request Guidelines

Nested Git history uses scoped conventional messages, such as `fix(docker): ...` and `docs(sciencebenchmark): ...`. Keep commits focused. PRs should describe behavior changes, commands tested, data prerequisites, and affected metrics. Link relevant issues and include figures when changing visual reports.

## Security & Evaluation Integrity

Never commit `.env`, `.openwebui-key`, database files, or credentials in logs. Keep benchmark databases read-only. Preserve previous result directories and report denominators, timeouts, missing predictions, and comparison rules explicitly.
