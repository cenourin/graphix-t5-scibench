# Procedência dos avaliadores em `third_party/`

Copiados **sem modificação** de `/app/third_party/` da imagem LEGACY_REFERENCE `eyuansu62/graphix-text-to-sql:v2` em 2026-09-29 (passo A6.1, `PORTABILIDADE.md`). São os avaliadores que produziram todos os EM/EX já medidos neste repositório (`seq2seq/metrics/spider/*.py` os importa). Antes deste passo eles existiam só dentro da imagem legada.

## Origem

Os arquivos são **idênticos, byte a byte,** às cabeças dos forks que o PICARD usa como submódulos (`ServiceNow/picard` → `.gitmodules`):

| Diretório | Fork | Commit | Arquivos |
|---|---|---|---|
| `spider/` | `elementai/spider` | `2965d67a2d7b0dc5dcb0ade84349b444e5cebd98` | `evaluation.py`, `process_sql.py`, `README.md` |
| `test_suite/` | `elementai/test-suite-sql-eval` | `61d71b8e902c0a5cac9ef81987fdcaebc27bfbc6` | `evaluation.py`, `exec_eval.py`, `parse.py`, `process_sql.py`, `README.md` |

| Arquivo | SHA-256 |
|---|---|
| `spider/evaluation.py` | `223ed0961d750f379694b9d05331310c1c93be0602c11fd081a62a2cdac42bfb` |
| `spider/process_sql.py` | `2ef748af026ae42552361094db08145fd02eb0d7489aae6dc9a124c8662aabcb` |
| `test_suite/evaluation.py` | `b8c2a6e1c02312f454582435bf4a84764cb1125bf30508787dc90403b71636bb` |
| `test_suite/exec_eval.py` | `ab33ee23b22f859cf6c83e09240ef5b494667f675b888539a51ad967824cd31d` |
| `test_suite/parse.py` | `0e364a7465d3603934a4a23ba01543824317febed100c1aaf221e1b8156d26a5` |
| `test_suite/process_sql.py` | `54fe3816de82cf240fa06b62529925a80ef596b18d7aae8f9dff33e0d5ee9e83` |

## Não são os avaliadores oficiais

Os 6 arquivos **diferem** dos oficiais da Yale (`taoyds/spider@b7b5b8c` e `taoyds/test-suite-sql-eval@e97acc5`, os commits fixados na outra máquina; ver a memória `reference_second_machine`). Algumas das diferenças:
- imports relativos (pacote Python);
- reformatação;
- `evaluate_one` devolvendo dicionários e scores por turno;
- `exec_eval.py` normalizando `'> ='` → `'>='` e usando um diretório temporário próprio.

**Implicação para o relatório.** Todos os EM/EX medidos até hoje vêm dos forks ElementAI. Isso é adequado para comparações internas, já que todos os modelos usam o mesmo avaliador. Para comparar com a literatura, o relatório final deve pontuar as predições também com os avaliadores oficiais fixados, e declarar qual avaliador produziu cada número.

## Dependências (fixadas em `docker/modern/requirements-a2.in`)

- **nltk 3.7**, igual ao legado. O EM passa pelo `word_tokenize`.
- **Dados `punkt`:** `docker/modern/nltk_data/tokenizers/punkt.zip` (SHA-256 `9a74e3cc0057021b12984c07cc5e46cb746385cf90f49b7d6fe806fb71610144`), copiado da imagem legada. Depois da extração, o `PY3/english.pickle` é idêntico ao do legado.
- **sqlparse 0.4.2**, igual ao legado.
