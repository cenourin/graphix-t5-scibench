# Estudo t5-base na RTX 4090: execução, scoring oficial e análise pré-registrada

Execução do estudo de `PROTOCOLO.md` (sem alteração) na stack moderna, em FP32 estrito, e o scoring e a análise dos §6 e §7. Os resultados estão em **`RESULTADOS.md`**, gerado por `scripts/analyze_study.py` sem alteração.

## 1. Execução do estudo

| Item | Valor |
|---|---|
| Orquestrador | `graphix_modern/study.py`, validado pelo TE (38/38) e pelo TS (77/77); veja `PORTABILIDADE.md` e `docs/port/orchestrator/` |
| Commit no pod | `01e006e`. O orquestrador e o treino são idênticos byte a byte ao `8554e77`, e o `start` conferiu isso por hash |
| Imagem | `silveirabruno/graphix-modern@sha256:371f61af7521069e62bb8973c2c42584377659440b5902d025876fb94d859037` (a5) |
| Hardware | 1× RTX 4090 (capability 8.9, driver 570.195.03), RunPod EU-RO-1 |
| Precisão | FP32 estrito: TF32, autocast e compile desligados, com guarda em todo `forward` (`scripts/runpod/entry.py`). Os 16 runs registram isso em `run_env.json` |
| Lançamento | `EXPECT_COMMIT=01e006e bash scripts/runpod_study.sh start`, depois de um smoke aprovado na mesma 4090 (`/workspace/study_smoke`) |
| Início → fim | 2026-10-06 03:58:43 → 12:51:50 UTC (8 h 53 min), `finished`, rc=0 |
| Incidentes | Nenhum: nenhum trial FAIL, nenhum resume, nenhum checkpoint incompleto |

Os metadados de cada célula estão em `run/<célula>/`:
- `trials.jsonl`;
- `final_config.json`;
- `FINAL_DONE`, que contém a checagem de que o modelo final é idêntico bit a bit ao melhor checkpoint;
- `final_trainer_state.json`;
- `dev_eval_results.json`, com os números calculados durante o treino;
- `final_run_env.json`.

O log de eventos e o estado do launcher estão em `run/`.

| Célula | Trials C/P/F | Hiperparâmetros escolhidos (LR, warmup, GA, WD) | Treino final |
|---|---|---|---|
| spider/rgat | 3/3/0 | 2,93e-4; 0,083; 32; 0,01 (trial 4) | early stop na época 6, melhor na 3 |
| spider/plain | 3/3/0 | 2,93e-4; 0,083; 32; 0,01 (trial 4) | early stop na época 6, melhor na 3 |
| sciencebenchmark/rgat | 3/3/0 | 1,65e-4; 0,042; 32; 0,0 (trial 1) | early stop na época 10, melhor na 7 |
| sciencebenchmark/plain | 3/3/0 | 2,93e-4; 0,083; 32; 0,01 (trial 4) | early stop na época 9, melhor na 6 |

## 2. Cópia dos resultados

- **O que foi copiado:** `/workspace/study` foi baixado inteiro pela API S3 para `train_db_id/study_4090_2026-10-06/`, que fica fora do git (13,58 GB, 588 arquivos).
- **Conferência contra o volume:** os 588 arquivos batem em contagem e em tamanho, um a um, com o inventário do volume (`volume_listing.txt`).
  - Um arquivo, `events.jsonl`, falhou no primeiro download com um erro 522 passageiro da API S3. Foi baixado de novo e conferido.
- **Manifesto sha256 da cópia:** `study_copy_manifest.sha256`, cujo próprio sha256 é `624e7677bcc36dfa6daa137d7579cc4fb9d20b49c7d152e040e8b80927c6816f`.
- **Integridade dos arquivos grandes:** em cada célula, o `model.safetensors` final e o do melhor checkpoint, baixados separadamente, são idênticos byte a byte.
- **O que fica no volume:** o volume e os pesos foram preservados, sem nenhuma limpeza.

## 3. Scoring oficial (PROTOCOLO.md §6 e desvios de 2026-09-28)

**Comando:** `bash docs/study_4090/score.sh`, a partir da raiz do repositório.
- **Pontuador:** `scripts/score_predictions.py`, com os avaliadores ElementAI vendorizados em `third_party/` (procedência em `third_party/PROVENANCE.md`).
- **Ambiente:** dentro da imagem que o script documenta, `eyuansu62/graphix-text-to-sql@sha256:1fb86bb618456dbc0dba73d13fd7af7219eda4bad22ad9d3a476254dd66d6614`, com `--cpus 1`.
- **Condições:** uma célula por vez, depois do fim do treino, com a máquina ociosa (load ≈ 0,5, GPU parada).
- **Análise:** em seguida, `scripts/analyze_study.py` sem alteração.

`scoring_environment.txt` registra a data, o commit, a imagem, Python 3.7.10, sqlite 3.33.0 e os sha256 do pontuador, da análise, dos avaliadores e das 4 predições. `outputs.sha256` registra os sha256 das saídas.

**Execução.** A primeira rodada (2026-10-06) foi interrompida pelo fim da sessão do agente, no meio do ScienceBenchmark. Para que as 4 células fossem pontuadas na mesma rodada, as saídas parciais foram movidas de lado (`train_db_id/rescore_study_4090.interrupted-20261006/`) e o scoring rodou **do zero**, desacoplado, em 2026-10-07 (13:34–14:08 UTC). As saídas parciais da primeira rodada são idênticas byte a byte às da segunda, nas três células que ela chegou a gravar.

**Exemplos pontuáveis.**
- **Spider:** 1034 de 1034 em EM e em EX.
- **ScienceBenchmark:** dos 299 exemplos do dev, 293 chegam ao modelo (o filtro remove 6). Desses, 290 são pontuáveis em EM (índices 216, 255 e 256 falham no parser) e 288 em EX (o SQL de referência falha nos índices 193, 198, 232, 233 e 254). Os conjuntos são os mesmos nos dois braços, e os testes pareados usam a interseção.
- **Diferença entre convenções:** a avaliação durante o treino pontuou o EX em 289 exemplos, e o pontuador offline em 288. É a dependência de carga prevista no desvio de 2026-09-28.
  - O `RESULTADOS.md` reporta o EX sobre os pontuáveis, com o n.
  - Os `*.summary.json` guardam também o EX na convenção do avaliador (falha conta como erro): rgat 0,1468 e plain 0,1706.

## 4. Resultados

Os números estão em `RESULTADOS.md`:
- tabela principal com ICs;
- 4 testes McNemar com Holm;
- ScienceBenchmark sem os 5 exemplos vazados do `oncomx`;
- ScienceBenchmark por banco.

Pelo critério pré-registrado, **a H0 não é rejeitada em nenhum dos 4 testes**: o menor p de Holm é 0,285, e todos os ICs da diferença rgat − plain incluem o zero. As limitações do §8 do protocolo valem integralmente:
- uma seed por célula;
- busca curta;
- dev pequeno no ScienceBenchmark.

## 5. O que ainda não foi feito

- **Avaliadores oficiais da Yale:** as predições ainda não foram pontuadas com eles. `third_party/PROVENANCE.md` exige isso para comparar com a literatura; é uma etapa separada, na segunda máquina, com os commits fixados.
- **Q3:** a comparação descritiva com o Graphix-3B ainda não foi feita.
- **BIRD:** continua ausente (§2).
