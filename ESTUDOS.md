# ESTUDOS: guia de todos os experimentos, relatórios e gráficos do TCC

Este arquivo é só um índice: não move nada de lugar. Cada número vem de `train_db_id/<run>/all_results.json`/`eval_results.json`, `eval/<pasta>/…json` ou dos `.tex`, sempre citados. Cada gráfico foi aberto e conferido a olho, não só pelo nome.

---

## Resumo: a história do TCC em 10 linhas

> ⛔ **Errata (26/09/2026).** Os runs **#4, #5, #6 e #13** (fine-tunings a partir do checkpoint Graphix do Spider) **não carregaram os pesos do checkpoint**. O prefixo `pretrain_model.` não bate com os nomes que o `from_pretrained` espera, e o modelo inteiro (T5 + RGAT) começou aleatório: perda no passo 1 de 14,5 a 15,5, contra 2,9 a 4,3 nos runs a partir do T5 original. Esses runs **não são fine-tuning** e não entram em comparações. Os itens 3, 5 e 6 abaixo foram corrigidos. Zero-shot (#1 a #3) e runs a partir do T5 original (#0, #7 a #12) **não são afetados**. Detalhes em `RISCOS.md`, R0.1.

1. O ponto de partida é o **Graphix-T5**, um T5 com camadas **RGAT** (atenção sobre o grafo pergunta↔esquema). Treinado no **Spider**, chega a 41,9% EM / 44,8% exec.
2. **Zero-shot no ScienceBenchmark** (cordis/oncomx/sdss): cai para **9,0% / 8,87%**. O `sdss` fica em 0% até nas perguntas fáceis. Beam search (n=4) melhora um pouco (9,34% / 9,56%). O PICARD não teve efeito porque não estava de fato ligado (falha de integração).
3. ~~**Fine-tuning do RGAT no ScienceBenchmark**: todas as tentativas ficaram **piores que o zero-shot**.~~ ⛔ **Inválido (errata):** as tentativas partiram de pesos aleatórios, não do modelo do Spider. Se o fine-tuning ajuda **não foi respondido**, e por decisão de escopo não será refeito (os modelos novos partem todos do T5 original). Texto original, para registro: Sem oncomx, few-shot e completo com gradient checkpointing deram entre 0% e 0,34% de exec. No caminho apareceram bugs (sem warmup, empate no "best checkpoint", gradient checkpointing quebrado), todos corrigidos.
4. **Ablação RGAT vs. T5 puro**: o **T5 sem RGAT ganha ou empata em todas as 4 comparações** (2 tamanhos × 2 benchmarks). No Spider, t5-base puro fica em 41,1% contra 41,9% do RGAT. No ScienceBenchmark, **t5-base puro = 8,19% exec**, o melhor fine-tuning do trabalho. ⚠️ **Errata:** a célula t5-base × ScienceBenchmark comparava o T5 puro (pesos do T5 original) com o RGAT (#6, pesos aleatórios), então essa comparação tem um fator de confusão e fica em aberto até o estudo novo. As outras 3 comparações continuam válidas.
5. **"Será que o RGAT só precisava de mais épocas?"** Treinando por 10 épocas, o RGAT melhora de 6 a 8× (t5-small 4,10% exec, t5-base 2,73%). Mesmo assim continua atrás do T5 puro e custa mais que o dobro do tempo. ⚠️ **Errata:** só o t5-small vale. O t5-base (2,73%, #13) partiu de pesos aleatórios.
6. **Conclusão atual**: com 1 GPU de 8 GB e poucas horas por run, aumentar o modelo rende mais do que adicionar estrutura de grafo. ⚠️ **Errata:** conclusão **suspensa**. Ela se apoiava em parte na comparação t5-base do ScienceBenchmark, que tem fator de confusão (item 4). Será reavaliada no estudo novo (t5-base + RGAT a partir do T5 original, Optuna, validação separada; ver `RISCOS.md`).

---

## Tabela comparativa de todos os runs

⛔ = run inválido (pesos do checkpoint não carregados, ver errata no topo). Dev do ScienceBenchmark: 289 exemplos pontuados em EM e 287 em exec. Dev do Spider: 1034 exemplos. O tempo é o `train_runtime`.

| # | Run (`train_db_id/` ou `eval/`) | Dataset | Modelo | RGAT? | Épocas | EM | exec | eval_loss | Tempo de treino |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `graphix-base-scibench-tcc` (**checkpoint de referência**, apesar do nome) | Spider | t5-base | ✅ | 5 | 41,88% | 44,78% | 0,2666 | 3h37 |
| 1 | `eval/sciencebenchmark` (zero-shot, greedy) | SciBench | t5-base (#0) | ✅ | – | 9,00% | 8,87% | não encontrado | – |
| 2 | `eval/sciencebenchmark_beam4` | SciBench | t5-base (#0) | ✅ | – | 9,34% | 9,56% | 0,6885 | – |
| 3 | `eval/sciencebenchmark_picard` | SciBench | t5-base (#0) | ✅ | – | 9,00% | 8,87% | não encontrado | – |
| 4 ⛔ | `graphix-scibench-finetune` (sem oncomx) → reavaliado em `eval/sciencebenchmark_finetuned` (ckpt-300) | SciBench | t5-base | ✅ | 3 | 0,00% | 0,34% | 4,2321 ¹ | 44min |
| 5 ⛔ | `graphix-scibench-fewshot` (300 ex.) | SciBench | t5-base | ✅ | 10 | 0,00% | 0,00% | 4,8754 | 1h08 |
| 6 ⛔ | `graphix-scibench-finetune-gc` (+oncomx, gc) | SciBench | t5-base | ✅ | 3 | 0,00% | 0,34% | 1,6765 | 9h42 |
| 7 | `graphix-scibench-t5small-plain` | SciBench | t5-small | ❌ | 3 | 1,38% | 1,37% | 1,1825 | 2h53 |
| 8 | `graphix-spider-t5small-plain` | Spider | t5-small | ❌ | 5 | 13,93% | 16,25% | 0,5965 | 1h43 |
| 9 | `graphix-spider-t5base-plain` | Spider | t5-base | ❌ | 5 | 41,10% | 43,81% | 0,2526 | 3h12 |
| 10 | `graphix-scibench-t5base-plain` | SciBench | t5-base | ❌ | 3 | **7,27%** | **8,19%** | 0,8010 | 4h13 |
| 11 | `graphix-scibench-t5small-rgat` (RGAT do zero) | SciBench | t5-small | ✅ | 3 | 1,04% | 0,68% ² | 1,1517 | 3h46 |
| 12 | `graphix-scibench-t5small-rgat-10ep` | SciBench | t5-small | ✅ | 10 | 5,19% | 4,10% | 0,9243 | 6h59 |
| 13 ⛔ | `graphix-scibench-finetune-gc-10ep` | SciBench | t5-base | ✅ | 10 | 0,00% | 2,73% | 1,4567 | 16h13 |

¹ O run #4 tem dois `eval_loss`. O `train_db_id/graphix-scibench-finetune/all_results.json` diz 2,7353, mas esse valor é do **checkpoint-50**, salvo por engano pelo bug de empate, e foi pontuado em apenas 194 exemplos. O 4,2321 vem de `eval/sciencebenchmark_finetuned/eval_results.json` (checkpoint-300, o que vale). A tabela principal do `.tex` mostra 2,7353 com nota de rodapé explicando.
² ⚠️ `train_db_id/graphix-scibench-t5small-rgat/all_results.json` registra `eval_exec_scored_examples = 286`, não 287 como nos outros runs. O `.tex` diz que todas as configurações usam 289/287. A diferença é de 1 exemplo e não muda a conclusão, mas vale conferir.

**Runs que não entram em nenhuma comparação** (smoke/piloto): `graphix-gc-smoke`, `graphix-scibench-smoke`, `graphix-base-scibench-tcc-pilot` (3 épocas, 1,26% EM no Spider) e `Graphix-3B` (checkpoint original de 3B do upstream). `eval/eval_results.json` (EM=1,0) é um eval smoke do Graphix-3B (`configs/eval_smoke.json`) e não é resultado real.

---

## Mapa dos relatórios

| Relatório | O que cobre | Papel |
|---|---|---|
| **`docs/sciencebenchmark-eval-results.tex` / `.pdf`** | Tudo: zero-shot, beam4, PICARD, os 4 fine-tunings com RGAT, t5-small puro, limitações, tabela consolidada, justificativa de hiperparâmetros e resumo das ablações e das 10 épocas | **PRINCIPAL**, é a fonte da verdade. Não tem preâmbulo, foi feito para `\input{}` no TCC |
| `estudo/spider_ablation.tex` / `.pdf` | Runs #8 e #9: T5 puro (small/base) no Spider contra a referência RGAT | Anexo detalhado da §"Ablação completa" do principal |
| `estudo/scibench_ablation.tex` / `.pdf` | Runs #10 e #11, fechando a matriz 2×2 no ScienceBenchmark (#7, #10, #11, #6) | Anexo detalhado da §"Ablação completa" |
| `estudo/more_epochs_ablation.tex` / `.pdf` | Runs #12 e #13 (10 épocas), under/overfitting e logs de validação por checkpoint | Anexo detalhado da §"Revisão: o RGAT precisava de mais treino?" |

Ordem de leitura sugerida: principal §1–§4, depois `spider_ablation`, `scibench_ablation` e `more_epochs_ablation`.

---

## Experimentos, na ordem da linha do tempo

### E0. Checkpoint de referência: Graphix-T5 (t5-base + RGAT) no Spider
- **Pergunta**: reproduzir o baseline do artigo no nosso hardware.
- **Config**: `configs/train.json` → `train_db_id/graphix-base-scibench-tcc`. Spider, t5-base, RGAT, 5 épocas, sem warmup.
- **Resultado**: 41,88% EM / 44,78% exec / eval_loss 0,2666 (`train_db_id/graphix-base-scibench-tcc/all_results.json`).
- **Gráficos**: nenhum. Não existe curva de treino deste run.
- **Relatório**: principal, §1 (texto) e tabela `tab:full-ablation`.

### E1. Zero-shot no ScienceBenchmark
- **Pergunta**: o modelo treinado no Spider funciona em bancos científicos sem re-treino?
- **Config**: `configs/eval_sciencebenchmark.json` → `eval/sciencebenchmark/`.
- **Resultado**: 9,00% EM / 8,87% exec (`eval/sciencebenchmark/final_metrics_v2.json`). Por banco: cordis 12/15%, oncomx 14,1/11,1%, sdss 0/0% (`final_metrics_per_db.json` e tabela 1 do `.tex`). Das previsões erradas, 68% são SQL inválido, e nenhuma dessas 179 roda no SQLite (`sql_invalido_validation.json`).
  - ⚠️ `eval/sciencebenchmark/final_metrics.json` (mais antigo, 0% e só 197 exemplos em exec, com o sdss pulado) está **desatualizado**. O correto é o `_v2`.
- **Gráficos**: aparece como as 3 primeiras barras de `docs/figures/overview_comparison.png` e `prf1_comparison.png` (ver E7).
- **Relatório**: principal, §1 "Avaliação no ScienceBenchmark (zero-shot)" (resultado agregado, dificuldade, análise de erros).

### E2. Beam search (n=4)
- **Pergunta**: decodificar melhor ajuda?
- **Config**: `configs/eval_sciencebenchmark_beam4.json` → `eval/sciencebenchmark_beam4/`.
- **Resultado**: 9,34% EM / 9,56% exec / 0,6885 (`eval_results.json`). É o melhor número zero-shot.
- **Gráficos**: barra "Zero-shot +beam4" em `overview_comparison.png`.
- **Relatório**: principal, §1.5 "Busca em feixe".

### E3. PICARD
- **Pergunta**: restringir a geração por gramática reduz o SQL inválido?
- **Config**: `configs/eval_sciencebenchmark_picard.json` → `eval/sciencebenchmark_picard/`.
- **Resultado**: idêntico ao zero-shot, 9,00% / 8,87% (`eval/sciencebenchmark/final_metrics_picard.json`). O PICARD nunca foi chamado (bug de integração em `rgat_picard.py`).
- **Gráficos**: barra "Zero-shot +PICARD" em `overview_comparison.png`, idêntica à primeira.
- **Relatório**: principal, §1.4 "Decodificação restrita por gramática (PICARD)".

### E4. Fine-tuning completo sem oncomx
- **Pergunta**: re-treinar no ScienceBenchmark melhora? (Por falta de VRAM, o oncomx ficou de fora.)
- **Config**: `configs/train_sciencebenchmark.json` → `train_db_id/graphix-scibench-finetune`. Reavaliação em `configs/eval_sciencebenchmark_finetuned.json` → `eval/sciencebenchmark_finetuned/`. t5-base+RGAT, 3 épocas, **warmup 0**, `metric_for_best_model=exact_match` (bug).
- **Resultado**: 0,00% EM / 0,34% exec / 4,2321 (ckpt-300). Pior que o zero-shot.
- **Gráficos**:
  - `docs/figures/full_no_oncomx_training.png`: à esquerda, train loss (azul, cai de ~15,5 para ~1), eval loss (vermelho, 2,7 → 1,77) e LR (tracejado, começa já no pico 5e-5, **sem warmup**). À direita, EM e exec ao longo das épocas. **Conclusão**: a loss cai normalmente, mas o EM fica em 0 o treino todo e o exec só sai de 0 na época ~2,3. Esse é o retrato do colapso sem warmup e do empate que quebrou a escolha do "melhor checkpoint".
  - `docs/figures/full_no_oncomx_error_matrix.png`: heatmap de categoria de erro × banco (checkpoint-300). **Conclusão**: 97 dos 99 exemplos de oncomx viram SQL inválido (o modelo nunca viu esse banco). No cordis o erro dominante é agregação errada (72) e no sdss é "outro erro semântico" (62).
  - `estudo/full_logs/scibench_t5base_rgat_no_oncomx_full.png`: mesmas curvas mais a tabela de 134 hiperparâmetros (ver seção final).
- **Relatório**: principal, §2.1 "Fine-tuning completo" e §2.5 limitações (iv) e (v).

### E5. Fine-tuning few-shot
- **Pergunta**: com uma amostra pequena e balanceada que inclui oncomx, o resultado melhora?
- **Config**: `configs/train_sciencebenchmark_fewshot.json` → `train_db_id/graphix-scibench-fewshot`. 300 ex. (140 cordis + 140 sdss + 20 oncomx), 10 épocas, warmup 0,1.
- **Resultado**: 0,00% / 0,00% / 4,8754 / 1h08 (`all_results.json`). O checkpoint salvo é o do passo 47 de 90.
- **Gráficos**:
  - `docs/figures/fewshot_training.png`: a train loss cai de ~14,5 para ~2,4. Só existem **2 pontos de eval** (épocas ~1,6 e ~5,2), e EM/exec ficam em 0 nos dois. **Conclusão**: dados de menos e checkpoints de menos. O modelo nunca gerou SQL válido.
  - `docs/figures/fewshot_error_matrix.png`: 281 de 293 exemplos são SQL inválido, em todos os bancos. **Conclusão**: o modelo não aprendeu a gerar SQL.
  - `estudo/full_logs/scibench_t5base_rgat_fewshot_full.png`: versão com a tabela de hiperparâmetros.
- **Relatório**: principal, §2.2 "Fine-tuning few-shot".

### E6. Correção do gradient checkpointing e fine-tuning completo com oncomx
- **Pergunta**: incluir todos os domínios (depois de corrigir o bug do gc) resolve?
- **Config**: `configs/train_sciencebenchmark_full_gc.json` → `train_db_id/graphix-scibench-finetune-gc`. 3 épocas, warmup 0,1, gc ligado, `metric_for_best_model=eval_loss`.
- **Resultado**: 0,00% / 0,34% / 1,6765 / 9h42 (`all_results.json`). A eval_loss caiu 2,5× em relação ao E4, mas **as métricas discretas não mudaram**.
- **Gráficos**:
  - `docs/figures/full_gc_oncomx_training.png`: a LR agora tem rampa de warmup, e a eval loss cai de forma suave (3,1 → 1,68). À direita, EM fica em 0 o tempo todo e o exec só sobe para 0,34% no último ponto. **Conclusão**: eval_loss melhor não significa acertar mais SQL.
  - `docs/figures/full_gc_oncomx_error_matrix.png`: o oncomx deixou de ser só SQL inválido (45 inválidos, 47 com coluna errada), mas o cordis continua dominado por agregação errada (70). **Conclusão**: ver o oncomx no treino mudou o tipo de erro, não a taxa de acerto.
  - `estudo/full_logs/scibench_t5base_rgat_oncomx_full.png`: versão com a tabela de hiperparâmetros (conferida).
- **Relatório**: principal, §2.3 "Correção do gradient checkpointing…".

### E7. Baseline t5-small sem RGAT (ScienceBenchmark) e visão consolidada
- **Pergunta**: o RGAT está ajudando? Um T5 menor e sem grafo faz melhor?
- **Config**: `configs/train_sciencebenchmark_t5small.json` → `train_db_id/graphix-scibench-t5small-plain` (`GRAPHIX_MODEL_VARIANT=plain`). 3 épocas.
- **Resultado**: 1,38% / 1,37% / 1,1825 / 2h53 (`all_results.json`). Supera todos os fine-tunings com RGAT.
- **Gráficos**:
  - `docs/figures/t5small_plain_training.png`: a eval loss cai de 2,1 para 1,18. O exec sai de 0 já na época 1 (pico de 1,7% na época 1,36) e o EM sobe até 1,38%. **Conclusão**: com warmup, o modelo produz SQL válido cedo.
  - `docs/figures/t5small_plain_error_matrix.png`: o erro dominante passa a ser **coluna errada** (106), e o SQL inválido cai para 81, concentrado no sdss (45). 3 corretos no cordis. **Conclusão**: o modelo já gera SQL estruturalmente plausível e erra nomes de colunas.
  - `docs/figures/overview_comparison.png`: barras de EM (verde) e exec (laranja) das **7 configurações** E1–E7, divididas em "zero-shot" e "com fine-tuning". **Conclusão**: o zero-shot (~9%) ganha de qualquer fine-tuning da primeira rodada, e entre os fine-tunings só o t5-small puro sai do zero.
  - `docs/figures/prf1_comparison.png`: precision, recall e F1 macro por componente SQL para as mesmas 7. **Conclusão**: P e R seguem o ranking do EM, mas o **F1 engana** (few-shot = 98,6% por "acordo vazio"). Não use o F1 como medida de qualidade.
  - `docs/figures/warmup_comparison.png`: à esquerda, a LR com e sem warmup. À direita, o exec de validação de `full_no_oncomx` (sem warmup) contra `t5small_plain` (com warmup). **Conclusão**: com warmup, o exec aparece mais de 1 época antes. É a justificativa empírica do `warmup_ratio=0,1`. ⚠️ A comparação não é controlada: os dois runs também diferem em modelo e em RGAT.
  - `estudo/full_logs/scibench_t5small_plain_full.png`: versão com a tabela de hiperparâmetros.
- **Relatório**: principal, §2.4 "Baseline t5-small sem RGAT", §3 "Comparação final", §4 "Tabela consolidada" e §4.3 "Justificativa dos hiperparâmetros" (warmup).

### E8. Ablação no Spider: T5 puro contra a referência RGAT
- **Pergunta**: o RGAT ajuda pelo menos no Spider, o benchmark para o qual foi desenhado?
- **Config**: `configs/train_spider_t5small_plain.json` → `graphix-spider-t5small-plain` e `configs/train_spider_t5base_plain.json` → `graphix-spider-t5base-plain`. 5 épocas, warmup 0,1.
- **Resultado**: t5-small 13,93% / 16,25% / 0,5965 / 1h43. t5-base **41,10% / 43,81%** / 0,2526 / 3h12. A referência RGAT tem 41,88% / 44,78%, uma diferença menor que 1 ponto.
- **Gráficos**:
  - `estudo/spider_ablation_comparison.png`: exec por época dos dois modelos puros, com uma linha pontilhada na referência RGAT (44,8%). **Conclusão**: o t5-base puro quase encosta na referência. O t5-small fica em ~16%, e isso se deve ao tamanho do modelo, não ao RGAT.
  - `estudo/spider_t5small_plain_training.png` e `estudo/spider_t5base_plain_training.png`: loss (train/eval/LR) à esquerda, EM e exec à direita. **Conclusão**: nos dois, as métricas sobem até o fim sem platô, ou seja, mais épocas ainda ajudariam um pouco.
  - `estudo/full_logs/spider_t5small_plain_full.png` e `spider_t5base_plain_full.png`: versões com a tabela de hiperparâmetros.
- **Relatório**: `estudo/spider_ablation.tex` (inteiro) e principal, §5 "Ablação completa" (tabela `tab:full-ablation` e a última figura).

### E9. Ablação 2×2 no ScienceBenchmark (tamanho × RGAT)
- **Pergunta**: com o tamanho do modelo fixo, o RGAT ajuda?
- **Config**: `configs/train_sciencebenchmark_t5base_plain.json` → `graphix-scibench-t5base-plain` e `configs/train_sciencebenchmark_t5small_rgat.json` → `graphix-scibench-t5small-rgat`, completando os runs #7 e #6.
- **Resultado**: t5-base puro **7,27% / 8,19%** / 0,8010 / 4h13, o melhor fine-tuning do trabalho. t5-small+RGAT 1,04% / 0,68% / 1,1517 / 3h46. **A comparação mais controlada do TCC** é t5-small com RGAT contra sem RGAT, e o sem RGAT vence.
- **Gráficos**:
  - `estudo/scibench_ablation_comparison.png`: exec por época das 4 configurações. **Conclusão**: o t5-base sem RGAT (laranja) sobe até ~8% e as outras 3 ficam presas abaixo de 2%.
  - `estudo/scibench_t5base_plain_training.png`: a eval loss vai de 1,3 a 0,80 e o EM/exec sobe quase sempre (com uma queda na época 1,7). **Conclusão**: é a configuração que melhor aprende.
  - `estudo/scibench_t5small_rgat_training.png`: a eval loss cai direitinho (2,05 → 1,15), mas o exec tem pico de 2,05% na época 2,04 e **cai** para 0,68% no final. **Conclusão**: o checkpoint escolhido por eval_loss não é o de melhor exec.
  - `estudo/full_logs/scibench_t5base_plain_full.png` e `scibench_t5small_rgat_full.png`: versões com a tabela de hiperparâmetros.
- **Relatório**: `estudo/scibench_ablation.tex` (inteiro) e principal, §5 "Ablação completa" (penúltima figura).

### E10. Mais épocas para o RGAT (3 → 10)
- **Pergunta**: o RGAT perdeu só por falta de treino?
- **Config**: `configs/train_sciencebenchmark_t5small_rgat_10ep.json` → `graphix-scibench-t5small-rgat-10ep` e `configs/train_sciencebenchmark_full_gc_10ep.json` → `graphix-scibench-finetune-gc-10ep`.
- **Resultado**: t5-small+RGAT 5,19% / 4,10% / 0,9243 / 6h59. t5-base+RGAT 0,00% / 2,73% / 1,4567 / 16h13. O sub-treino era **parte** da explicação, mas o RGAT continua atrás do t5-base puro (8,19% em 3 épocas e 4h13).
- **Gráficos**:
  - `estudo/more_epochs_ablation_curves.png`: um painel por modelo, com eval_loss (linha cheia) e exec (tracejado) para 3 épocas (cinza) e 10 épocas (vermelho). **Conclusão**: nas duas durações a eval_loss continua caindo e o exec sobe bem com 10 épocas. ⚠️ No t5-base, o exec chega a **~4,4% na época ~7,4** e cai para 2,73% no checkpoint escolhido por eval_loss. O relatório não comenta esse pico.
  - `estudo/more_epochs_ablation_summary.png`: barras de eval_loss final (esquerda) e exec/EM final (direita) das 4 rodadas. **Conclusão**: é o resumo visual da tabela. (Cosmético: os rótulos do eixo X de "t5-base + RGAT completo" se sobrepõem.)
  - `estudo/more_epochs_train_vs_eval_loss.png`: train loss por passo contra eval loss por checkpoint, 10 épocas, eixo Y cortado em 4. **Conclusão**: nenhum dos dois faz overfitting (a eval nunca sobe). O t5-small está em underfitting e o t5-base está chegando ao limite (a eval fica plana enquanto a train ainda cai). (Cosmético: o título tem um `\` sobrando em "vs.\ eval".)
- **Relatório**: `estudo/more_epochs_ablation.tex` (inteiro) e principal, §5.1 "Revisão: o RGAT precisava de mais treino?". Os runs de 10 épocas **não** têm figura em `full_logs/`.

---

## Arquivos confusos ou redundantes

### Imagens que nenhum `.tex` usa (órfãs)
- **As 8 imagens de `estudo/full_logs/*_full.png`**. Foram geradas por `make_full_config_plots.py` e cada uma tem a tabela de 134 hiperparâmetros do `combined_args.json` mais as mesmas curvas de loss e métricas. Servem como registro de reprodutibilidade, mas não aparecem em nenhum relatório.
- **Imagens da raiz** (`graphix.png`, `dp.png`, `graphix-3b-picard.png`): nenhum `.tex` usa, mas o **`README.md` usa** (linhas 19, 44 e 74). Vieram do repositório original da Alibaba:
  - `graphix.png`: diagrama comparando RATSQL, T5, GNN-T5 e **Graphix-T5**. Útil para o capítulo de fundamentação do TCC.
  - `dp.png`: exemplo do grafo pergunta↔esquema (dependências sintáticas + ligação com as colunas).
  - `graphix-3b-picard.png`: print de terminal com as métricas do **Graphix-3B+PICARD no Spider** (77,08% EM / 80,95% exec). ⚠️ É do artigo/upstream, **não é resultado deste TCC**.
- `render.html` (raiz): gráfico pyecharts ("Awesome-pyecharts") do upstream. Não tem relação com o TCC.

### Imagens duplicadas ou que mostram a mesma coisa
Cada run tem até **3 versões** da curva de treino:

| Run | `docs/figures/` | `estudo/` | `estudo/full_logs/` |
|---|---|---|---|
| E4 sem oncomx | `full_no_oncomx_training.png` | – | `scibench_t5base_rgat_no_oncomx_full.png` |
| E5 few-shot | `fewshot_training.png` | – | `scibench_t5base_rgat_fewshot_full.png` |
| E6 +oncomx gc | `full_gc_oncomx_training.png` | – | `scibench_t5base_rgat_oncomx_full.png` |
| E7 t5-small puro | `t5small_plain_training.png` | – | `scibench_t5small_plain_full.png` |
| E8 Spider small | – | `spider_t5small_plain_training.png` | `spider_t5small_plain_full.png` |
| E8 Spider base | – | `spider_t5base_plain_training.png` | `spider_t5base_plain_full.png` |
| E9 t5-base puro | – | `scibench_t5base_plain_training.png` | `scibench_t5base_plain_full.png` |
| E9 t5-small RGAT | – | `scibench_t5small_rgat_training.png` | `scibench_t5small_rgat_full.png` |

- A versão `_full` = a versão `_training` + a tabela de hiperparâmetros.
- O **padrão de nomes é inconsistente**: `docs/figures/t5small_plain_*` não diz o dataset, enquanto `estudo/` usa `scibench_`/`spider_`. `full_gc_oncomx` e `scibench_t5base_rgat_oncomx` são o mesmo run com nomes diferentes.
- O nome do run `graphix-base-scibench-tcc` confunde: é o **checkpoint treinado no Spider** (E0), não no ScienceBenchmark.

### Outros problemas encontrados
- **Os scripts que geram os gráficos não estão no repo.** Ficam em `~/.claude/jobs/42c42bb9/tmp/` (`make_plots.py`, `make_hparam_plot.py`, `make_spider_plots.py`, `make_scibench_ablation_plots.py`, `make_more_epochs_plots.py`, `make_train_vs_eval_loss_plot.py`, `make_full_config_plots.py`, `categorize_errors.py`, `compute_prf1.py`), que é uma pasta temporária. Se ela sumir, não dá mais para regenerar as figuras. **Recomendação forte**: copiar para `scripts/plots/` no repo.

  | Script | Gera |
  |---|---|
  | `make_plots.py` | `docs/figures/*_training.png`, `*_error_matrix.png`, `overview_comparison.png`, `prf1_comparison.png` |
  | `make_hparam_plot.py` | `docs/figures/warmup_comparison.png` |
  | `make_spider_plots.py` | `estudo/spider_*.png` |
  | `make_scibench_ablation_plots.py` | `estudo/scibench_*.png` |
  | `make_more_epochs_plots.py` | `estudo/more_epochs_ablation_{curves,summary}.png` |
  | `make_train_vs_eval_loss_plot.py` | `estudo/more_epochs_train_vs_eval_loss.png` |
  | `make_full_config_plots.py` | `estudo/full_logs/*_full.png` |

- `eval/sciencebenchmark/final_metrics.json` está desatualizado (use o `_v2`). `eval/eval_results.json` é um smoke test.
- O `MEMORY.md`, item 9, ainda descreve a ablação como "queued"/"ongoing", mas ela já terminou (E8–E10).

### Proposta de estrutura (só proposta, nada foi movido)

```
figuras/
  00_contexto_graphix/          graphix.png, dp.png, graphix-3b-picard.png   (manter cópia na raiz p/ README)
  01_zero_shot/                 (hoje sem figura própria; aparece nos comparativos)
  02_ft_sem_oncomx/             ft_t5base_rgat_sem_oncomx__curvas.png, __erros.png
  03_ft_fewshot/                ft_t5base_rgat_fewshot__curvas.png, __erros.png
  04_ft_gc_com_oncomx/          ft_t5base_rgat_oncomx_gc__curvas.png, __erros.png
  05_t5small_puro_scibench/     scibench_t5small_puro__curvas.png, __erros.png
  06_comparativo_geral/         comparativo_7configs_em_exec.png, comparativo_7configs_prf1.png, efeito_warmup.png
  07_ablacao_spider/            spider_comparativo_exec.png, spider_t5small_puro__curvas.png, spider_t5base_puro__curvas.png
  08_ablacao_scibench_2x2/      scibench_2x2_exec.png, scibench_t5base_puro__curvas.png, scibench_t5small_rgat__curvas.png
  09_rgat_10_epocas/            rgat_3vs10ep__curvas.png, rgat_3vs10ep__resumo.png, rgat_10ep__train_vs_eval.png
  anexo_hiperparametros/        <run>__hiperparametros.png   (os 8 de full_logs/)
relatorios/
  principal_sciencebenchmark.tex/.pdf
  anexo_A_ablacao_spider.tex/.pdf
  anexo_B_ablacao_scibench.tex/.pdf
  anexo_C_rgat_10_epocas.tex/.pdf
scripts/plots/                  (copiar os make_*.py de ~/.claude/jobs/…)
```

Convenção sugerida: `<dataset>_<modelo>_<rgat|puro>[_variante]__<tipo>.png`. Se aplicar essa estrutura, será preciso atualizar os `\includegraphics` dos 4 `.tex` e o `README.md`, e recompilar os PDFs.
