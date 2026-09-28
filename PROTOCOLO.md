# Protocolo do estudo t5-base (pré-registro)

Escrito e commitado em 2026-09-26, **antes** de qualquer resultado deste estudo. Uma mudança feita depois de ver resultados entra na seção "Desvios", com data e motivo, e o texto original fica como está.

Código: `seq2seq/run_t5base_study.py` (orquestrador), `scripts/run_t5base_study.sh` (lançador), `configs/study_t5base_{spider,sciencebenchmark}.json`. Riscos gerais do repositório: `RISCOS.md`.

## 1. Perguntas e hipóteses

- **Q1 (principal):** injetar RGAT no encoder do T5 melhora o text-to-SQL em relação ao mesmo T5 sem RGAT, com treino e seleção idênticos?
  - H0: a diferença de exact match (EM) e de execução (EX) no dev entre `rgat` e `plain` é zero.
  - Teste: McNemar pareado por exemplo, bicaudal, α = 0,05, feito separadamente por benchmark e por métrica.
- **Q2:** o efeito do RGAT, se houver, se mantém fora do Spider? Isso se responde comparando o efeito no Spider com o efeito no ScienceBenchmark.
- **Q3 (descritiva, sem teste):** como o t5-base treinado aqui se posiciona em relação ao Graphix-3B original, avaliado na outra máquina? Essa comparação **não é controlada**, porque tamanho e pré-treino diferem, e é reportada só como referência.

## 2. Desenho

Fatorial 2 × 2:

- braço: `rgat` ou `plain` (`GRAPHIX_MODEL_VARIANT`);
- benchmark: Spider ou ScienceBenchmark.

O BIRD entra como terceiro benchmark quando os bancos do train existirem (`docs/bird-data.md`). Até lá ele fica fora, e isso é declarado.

Dentro de um benchmark, os dois braços compartilham tudo, exceto a variante do modelo:
- config base;
- dados e filtros (`GRAPHIX_MAX_GRAPH_NODES`, que também se aplica ao plain, de modo que os exemplos são os mesmos);
- espaço e orçamento de busca;
- seeds;
- critério de parada;
- decodificação.

**Inicialização:** todos os modelos partem dos pesos originais do HF, `data_all_in/t5-base`. Não há fine-tuning a partir de checkpoints Graphix (`RISCOS.md` R0.1).

### 2.1 Configuração por célula

Comum a todas as células:
- t5-base (~223M parâmetros), fp32, Adafactor;
- batch 1 por passo, com acumulação de gradiente escolhida pela busca (§4);
- seed de treino 1;
- decodificação greedy até 512 tokens (§6).

| Célula | `GRAPHIX_MODEL_VARIANT` | Entrada máx. | Grafo máx. (`GRAPHIX_MAX_GRAPH_NODES`) | Gradient checkpointing | Batch de avaliação | Config base |
|---|---|---|---|---|---|---|
| Spider / rgat | `rgat` | 512 tokens | 512 nós | não | 8 | `configs/study_t5base_spider.json` |
| Spider / plain | `plain` | 512 | 512 | não | 8 | idem |
| ScienceBenchmark / rgat | `rgat` | 1536 | 1400 | sim | 1 | `configs/study_t5base_sciencebenchmark.json` |
| ScienceBenchmark / plain | `plain` | 1536 | 1400 | sim | 1 | idem |

O limite de nós também vale para o plain, para que as duas células de um benchmark vejam exatamente os mesmos exemplos. O gradient checkpointing não muda a conta, só troca memória por tempo; ele fica ligado nos dois braços do ScienceBenchmark para que o custo seja o mesmo.

### 2.2 Ambiente

| Item | Valor |
|---|---|
| Imagem Docker | `eyuansu62/graphix-text-to-sql:v2` (congela o ambiente) |
| Python | 3.7.10 |
| torch | 1.9.0, CUDA 11.1 |
| DGL | 0.8.2 |
| transformers | 4.17.0 |
| Hardware | 1× GTX 1070 8 GB, 27 GB RAM, 4 núcleos |

As 4 células rodam todas nesta máquina. O pico de VRAM e o nome da GPU ficam em `train_results.json` / `eval_results.json` de cada run.

## 3. Dados

| Benchmark | Treino | Validação (seleção) | Relatório |
|---|---|---|---|
| Spider | 7594 (`splits/spider.json`) | 983, bancos disjuntos do treino | dev oficial, 1034 |
| ScienceBenchmark | 4259 (`splits/sciencebenchmark.json`) | 473, esqueletos de SQL disjuntos dentro de cada banco | dev oficial, 299 (293 pontuáveis, ver §7) |

As divisões vêm de `scripts/make_splits.py`, com seed 42. **O dev oficial nunca é usado para escolher nada:** nem hiperparâmetros, nem época, nem checkpoint. Ele só é usado uma vez por célula, no fim.

## 4. Busca de hiperparâmetros (Optuna)

- **Espaço, igual em todas as células:**
  - `learning_rate`: log-uniforme em [2e-5, 1e-3];
  - `warmup_ratio`: uniforme em [0, 0,1];
  - `gradient_accumulation_steps`: {8, 16, 32, 64};
  - `weight_decay`: {0, 0,01, 0,1}.

  Otimizador Adafactor, batch 1 por dispositivo.
- **Orçamento:** 6 trials por célula, com TPE. A seed do sampler é 1 no início de cada estudo, então os dois braços recebem as mesmas configurações iniciais.
- **Fidelidade:** cada trial usa o agendamento de learning rate do treino final (15 épocas) e para depois da época 3. Assim, um trial é um prefixo exato do run que ele prevê.
  - O warmup vai só até 0,1 (1,5 época) para que a busca veja o LR depois do warmup.
- **Objetivo:** a menor `eval_loss` na validação, calculada só com teacher forcing, sem geração.
  - Early stopping com paciência de 1 época.
  - MedianPruner com n_startup_trials=2 e poda a partir da época 1.

## 5. Treino final

- **Hiperparâmetros:** os do melhor trial, com treino do zero e seed 1.
- **Duração:** até 15 épocas, com early stopping na `eval_loss` de validação (paciência de 3 épocas).
- **Checkpoint usado:** o de menor `eval_loss` de validação (`load_best_model_at_end`).

## 6. Avaliação

- **Onde:** no dev oficial, uma única vez por célula.
- **Métricas:**
  - EM: avaliador do Spider (`seq2seq/metrics/spider`);
  - EX: test-suite / execução nos bancos SQLite.
- **Decodificação, idêntica em todas as células:**
  - greedy (`num_beams` 1);
  - sem `no_repeat_ngram_size`;
  - até 512 tokens.

  Não usamos PICARD, que não funciona com o modelo RGAT neste repositório. Para a Q3, o Graphix-3B tem que ser reportado com essa mesma decodificação. Números com beam 4 podem aparecer, mas separados.
- **Carregamento dos pesos:** estrito, via `GRAPHIX_INIT_STATE_DICT`. Se alguma chave não bater, o run falha em vez de avaliar pesos não treinados.
- **Alinhamento:** predição e gabarito ficam alinhados mesmo quando o filtro descarta exemplos do dev. Isso foi corrigido em 2026-09-26; antes, os runs do ScienceBenchmark pontuavam 76 de 293 predições contra o gabarito errado.

## 7. Análise (definida antes dos resultados)

1. **Tabela principal:** EM e EX por célula, com IC de 95% por bootstrap (10 000 reamostragens de exemplos, seed 0).
2. **Q1:** McNemar exato pareado entre `rgat` e `plain`, por benchmark e métrica. Também reportamos a diferença com IC bootstrap pareado.
   - São 4 testes (2 benchmarks × 2 métricas), e reportamos o p bruto e o p com correção de Holm.
3. **ScienceBenchmark:** o número principal é o do dev de 293 exemplos. Como número secundário, reportamos também o resultado sem os 3 exemplos do `oncomx` cujo SQL aparece idêntico no train.
4. **Resultados por banco no ScienceBenchmark** (`cordis`, `oncomx`, `sdss`): só descritivos.
5. **O que é reportado:** tudo, inclusive resultados nulos, negativos, trials podados e falhas.

## 8. Limitações conhecidas

- **Uma seed por célula:** a variância entre seeds não é estimada, e o McNemar mede incerteza de amostragem dos exemplos, não de treino.
- **Busca curta e de baixa fidelidade:** 6 trials de 3 épocas. O ótimo encontrado é local, e isso vale igualmente para os dois braços.
- **Loss de validação como critério:** ela é um substituto imperfeito de EM/EX.
- **Dev pequeno no ScienceBenchmark:** 293 exemplos; diferenças de poucos pontos percentuais têm pouco poder estatístico.
- **Hardware:** uma GTX 1070 (8 GB); ver §2.1. Isso explica o filtro de grafo com mais de 512 nós no Spider (o dev não perde exemplos) e o gradient checkpointing no ScienceBenchmark (nos dois braços).
- **BIRD:** fica ausente até existirem os bancos do train.

## 9. Execução

```
scripts/run_t5base_study.sh     # ordem: spider/rgat, spider/plain, sciencebenchmark/rgat, sciencebenchmark/plain
```

Saídas:
- `optuna_studies/<bench>_t5base_<arm>.db`;
- `train_db_id/optuna/<bench>_t5base_<arm>/trial_N/`;
- `train_db_id/study-t5base-<bench>-<arm>/` (treino final e `dev_eval/`);
- log de eventos em `optuna_studies/t5base_study.jsonl`.

O código roda a partir de um snapshot (`train_db_id/t5base_study_code/`, com `COMMIT`).

## 10. Desvios

(nenhum até agora)
