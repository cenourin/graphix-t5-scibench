# Manifesto dos datasets e bancos

`DATASETS.sha256` traz o sha256 de cada arquivo de dados e de cada banco SQLite usados no estudo t5-base: Spider e ScienceBenchmark, com 187 arquivos.
- **Caminhos:** relativos à raiz dos datasets, que contém `spider/` e `sciencebenchmark/`. Nesta máquina a raiz é `data_all_in/data/`; no outro computador é `schemagraphsql_experiment/datasets/`.
- **Ordenação:** `LC_ALL=C`.

Os arquivos em si não ficam no git por causa do tamanho (o `skyserver` sozinho tem 15 GB); fica só a identidade deles.

## Como conferir

Dentro da raiz dos datasets:
```bash
LC_ALL=C sha256sum -c /caminho/do/repo/docs/datasets/DATASETS.sha256
```
Cada linha deve terminar em `OK`. Uma linha `FAILED` aponta exatamente o arquivo que diverge.

O hash conjunto dos 166 bancos do Spider, rodado dentro de `spider/database/`, é `c0bf09e77ad05e72efac5701e6f97e3e6334da3f081f47e9d8d8aaf436060f16`:
```bash
find . -name '*.sqlite' | LC_ALL=C sort | xargs sha256sum | sha256sum
```
Sem `LC_ALL=C`, o `sort` ordena de outro jeito em locales como pt_BR, e o hash conjunto muda mesmo com os arquivos idênticos.

## Registro
- **2026-10-09:** gerado na máquina principal, a partir de `data_all_in/data/`. Na mesma data, os 166 bancos do Spider e os 12 arquivos-chave do outro computador (`schemagraphsql_experiment/datasets`) foram conferidos como idênticos.
- **Origem dos dados:**
  - Spider: distribuição oficial da Yale.
  - ScienceBenchmark: SQLite convertidos dos dumps Postgres oficiais (veja `CLAUDE.md`). Não são reproduzíveis só baixando; guarde as cópias.
