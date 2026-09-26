# RISCOS: análise de riscos e prevenção de erros do estudo comparativo

Escopo: o estudo ampliado decidido em 2026-09-26.
- **t5-base + RGAT** treinado no **Spider**, no **BIRD** e no **ScienceBenchmark**, com early stopping e busca Optuna reduzida.
- **Inferência do Graphix-3B** (checkpoint original) nos mesmos 3 benchmarks.
- A busca Optuna do **t5-small + RGAT** que já está rodando.

Hardware: 1 GTX 1070 (8 GB, Pascal), 27 GB de RAM, 4 núcleos.

Cada risco traz: **gravidade** (A = invalida uma conclusão, M = atrasa ou enfraquece, B = incômodo), **probabilidade**, **prevenção**, **detecção** e **plano B**. Quando um risco já aconteceu, isso está marcado como **CONFIRMADO**, com a evidência.

---

## 0. Incidentes já confirmados (resolver antes de qualquer run novo)

### R0.1 Fine-tunings a partir do checkpoint Graphix começaram com pesos aleatórios. CONFIRMADO, gravidade A
- **O que acontece.** O `run_seq2seq_train.py` usa o `models/graphix/rgat.Model`, que chama `from_pretrained()` do T5 interno. Os pesos salvos por esse wrapper têm o prefixo `pretrain_model.`, então o HF não reconhece nenhum parâmetro: avisa "not used" e "newly initialized" e deixa **todos** os pesos aleatórios, inclusive o T5 base. Só o caminho de avaliação (`rgat_picard.Model`) remove o prefixo e carrega corretamente.
- **Evidência.**
  - Loss de treino no passo 1 (`trainer_state.json`):
    - Runs afetados: `graphix-scibench-finetune`, `-fewshot`, `-finetune-gc` e `-finetune-gc-10ep` começam em **14,5 a 15,5**.
    - Runs que partem do t5 original ficam entre **2,9 e 4,3**.
  - Teste de carga isolado no `graphix-base-scibench-tcc`: 0 parâmetros carregados.
- **Impacto.**
  - Os runs **#4, #5, #6 e #13** do `ESTUDOS.md` **não são fine-tuning** do modelo do Spider. São treinos do zero, com o T5 também aleatório.
  - Caem estas conclusões:
    - "fine-tuning é pior que o zero-shot" (item 3 do resumo);
    - o número "t5-base RGAT 10 épocas = 2,73% exec" (item 5).
  - Não afetados: os runs que partem de `data_all_in/t5-*` (#0, #7 a #12) e a busca Optuna atual (t5-small original).
- **Decisão (2026-09-26).** Todos os modelos treinados partem do T5 original (`data_all_in/t5-*`). O fine-tuning a partir do checkpoint Graphix saiu do escopo, e o risco deixa de afetar os runs novos.
- **Prevenção (o que ainda vale).**
  1. Corrigir o carregamento no `rgat.Model` para remover o prefixo, como o `rgat_picard` já faz.
  2. Adicionar uma checagem que **aborta** o treino se algum parâmetro vier "newly initialized" quando o `model_name_or_path` for um checkpoint Graphix.
  3. ~~Refazer os fine-tunings afetados.~~ Fora do escopo por decisão. No TCC, a transferência Spider → ScienceBenchmark é avaliada só por zero-shot vs. treino direto.
  4. Marcar #4, #5, #6 e #13 como inválidos no `ESTUDOS.md` e no `.tex`, sem apagar.
- **Detecção permanente.** Registrar a loss do passo 1 em todo run. Uma loss muito maior que a de um run a partir do T5 original indica pesos não carregados.

### R0.2 O código montado ao vivo afeta a busca em andamento. CONFIRMADO, gravidade A
- **O que acontece.** O container `optuna-search` monta `seq2seq/` como leitura e escrita, e cada trial roda um novo subprocesso `run_seq2seq_train.py`. **Qualquer edição em `seq2seq/` entra nos próximos trials.** Corrigir o R0.1 durante a busca mudaria o código no meio do estudo.
- **Prevenção.**
  - Cada job longo roda a partir de um **snapshot congelado** do código: uma cópia ou `git worktree` num commit fixo, montada somente leitura.
  - O hash do commit é registrado no diretório do run.
  - Para a busca atual, qualquer correção deve ser um no-op no caminho do t5-small original. Mais seguro ainda é editar num worktree separado até a busca terminar.
- **Detecção.** Salvar o `git rev-parse HEAD` e o `git diff` no início de cada trial.

---

## 1. Validade científica (desenho do estudo)

| # | Risco | Grav. | Prob. | Prevenção | Detecção / plano B |
|---|---|---|---|---|---|
| 1.1 | **Seleção e relato no mesmo dev** deixam os números otimistas | A | Certa na busca atual | Estudo novo: validação tirada do train, dev só para o relato final e test oficial do Spider como teste extra | Busca do t5-small: declarar como limitação. Opcional: repetir no protocolo novo |
| 1.2 | **Vazamento na validação do ScienceBenchmark**: os exemplos `synth` são gerados a partir dos `seed`, e uma divisão aleatória põe "parentes" dos dois lados | A | Alta | Dividir por **grupo** (template ou seed de origem). Se não houver ligação explícita, usar só `seed` como validação e tirar do treino os `synth` derivados deles | Medir o overlap de SQL normalizado entre treino e validação. Tem de ser 0 |
| 1.3 | **Validação do Spider e do BIRD sem separar bancos**: o teste real é entre domínios, e validar em bancos já vistos no treino superestima o resultado | M | Alta | Validação **por banco** (bancos inteiros fora do treino), com semente fixa | Conferir que a interseção de `db_id` entre treino e validação é vazia |
| 1.4 | **Early stopping por `eval_loss`** escolhe uma época que não é a melhor em exec | M | Média | Declarar que a loss é um proxy. No treino final, avaliar exec em todas as épocas e reportar os dois critérios | Correlação entre loss e exec por época, que já existe nos runs de 10 épocas |
| 1.5 | **Uma semente só**: diferenças de 1 a 2 pp são ruído (ScienceBenchmark tem 289/287 exemplos pontuáveis) | A | Certa | Treino final com **3 sementes** nas configurações principais, ou pelo menos no t5-base × 3 benchmarks | IC por bootstrap e **teste de McNemar pareado** entre modelos. Sem significância, a conclusão é "empate" |
| 1.6 | **Busca desigual**: um benchmark recebe mais trials ou mais épocas e a comparação fica viciada | M | Média | Mesmo espaço, mesmo número de trials, mesmo pruner e mesma paciência nos 3 benchmarks. Registrar o orçamento gasto em horas de GPU | Tabela de orçamento por benchmark no relatório |
| 1.7 | **Graphix-3B vs t5-base não é comparação controlada**: tamanho, receita de treino e versão do pré-processamento são diferentes | M | Certa | Tratar o 3B como **referência externa**, não como ablação. O controle é t5-base com RGAT vs sem RGAT | Texto do TCC separa "ablação controlada" de "referência" |
| 1.8 | **Tentação de ajustar ao resultado**: mexer em filtros, métrica ou checkpoint depois de ver o dev | A | Média | Congelar o protocolo (splits, espaço de busca, métricas e critério de checkpoint) **antes** dos runs, com commit | Todo desvio vira uma entrada de "desvio de protocolo" com a justificativa |
| 1.9 | **Reprodução do 3B "não bate" com o artigo** e é tratada como erro nosso ou mascarada | M | Alta (sem PICARD) | Alvo = número do artigo **sem PICARD**. Reportar a diferença como está | Se a diferença passar de ~2 pp no Spider, investigar pré-processamento e decodificação antes de seguir para os outros benchmarks |
| 1.10 | **Filtro `GRAPHIX_MAX_GRAPH_NODES` descarta exemplos** e os conjuntos deixam de ser comparáveis | M | Alta no BIRD | Valor alto, igual em todas as variantes. Contar e reportar os descartados | Log "n exemplos removidos por split". O dev **nunca** é filtrado |

## 2. Graphix-3B (inferência na CPU em fp32)

| # | Risco | Grav. | Prob. | Prevenção | Detecção / plano B |
|---|---|---|---|---|---|
| 2.1 | **OOM na RAM**: 14,2 GB de pesos. O carregador atual em fp32 teria pico de cerca de 28 GB (dicionário + modelo), e o OOM-killer pode matar **a busca na GPU** | A | Alta | Carregar em shards (pesos → modelo, um bloco por vez; pico de ~15 GB). Container com `--memory` limitado (cgroup), para matar só ele | `docker stats` e ausência de OOM no `dmesg`. Plano B: pausar a busca durante a carga |
| 2.2 | **Pesos não carregados em silêncio** (como no R0.1) | A | Média | `load_state_dict(strict=True)` e comparar alguns tensores com o arquivo depois de carregar | Piloto no dev do Spider: EM próximo do artigo. Se der próximo de 0%, os pesos estão errados |
| 2.3 | **Precisão diferente da original**: o carregador atual converte para **bf16** e a 1070 não tem bf16 nativo | M | Certa se não for mudado | Tornar a precisão configurável (padrão bf16, para não mudar runs antigos) e usar **fp32** neste estudo | Comparar 20 saídas fp32 vs bf16 e reportar a diferença |
| 2.4 | **Tempo na CPU**: 1034 (Spider) + 299 (SciBench) + 1534 (BIRD) exemplos, com 4 núcleos divididos com o treino | M | Alta | Piloto de 20 exemplos para medir s/exemplo antes de rodar tudo. Checkpoint das previsões a cada N exemplos, para poder retomar | Se passar de ~3 dias, rodar quando a GPU estiver livre ou usar o offload na GPU (declarado) |
| 2.5 | **Disputa de CPU atrasa a busca na GPU** (dataloader) | B | Alta | `--cpus 2` no container do 3B | Comparar s/it da busca antes e depois |
| 2.6 | **O smoke test antigo (EM=1,0, loss=0,0) não prova nada** | B | Certa | Não citar como resultado (o `ESTUDOS.md` já não cita) | — |
| 2.7 | **Entrada do 3B diferente da do treino original** (serialização do esquema, conteúdo do banco, `normalize_query`, `target_with_db_id`) | A | Média | Usar exatamente os flags do `configs/eval.json` e do artigo e documentar cada um | Se o Spider ficar muito abaixo do artigo, conferir estes flags primeiro |

## 3. BIRD (benchmark novo)

| # | Risco | Grav. | Prob. | Prevenção | Detecção / plano B |
|---|---|---|---|---|---|
| 3.1 | **Download grande** (~35 GB) e corrompido ou incompleto | M | Média | Conferir checksums e tamanhos. Baixar em background com retomada | Contagem de bancos: 69 (train) + 11 (dev) |
| 3.2 | **OOM no pré-processamento**: o passo `map_subword_*` já causa OOM no train do Spider com 27 GB (documentado). O BIRD tem mais exemplos e esquemas maiores | A | Alta | Processar em shards, sem mudar a lógica por exemplo (regra do `CLAUDE.md`). `set -e` no orquestrador | Conferir tamanhos e timestamps de cada saída intermediária. O DAG não falha sozinho |
| 3.3 | **Pipeline do Spider não serve para o BIRD**: campo `evidence`, nomes com espaços, bancos enormes na ligação por conteúdo | A | Alta | Adaptador próprio. Decidir e documentar **com ou sem evidence** (o padrão do BIRD é com) e aplicar igual a todos os modelos | Smoke em 50 exemplos antes do conjunto inteiro |
| 3.4 | **Truncamento em 512 tokens**: esquemas do BIRD passam do limite e o modelo não vê a coluna certa | A | Alta | Medir a % de entradas truncadas **antes** de treinar e reportar. Considerar poda do esquema, igual para todos | Relatório de comprimento por exemplo |
| 3.5 | **Grafos grandes estouram a VRAM** no RGAT | M | Alta | Medir a distribuição de nós. Gradient checkpointing (já corrigido) | Plano B: batch 1 com acumulação maior (mesmo batch efetivo) |
| 3.6 | **Métrica errada**: o BIRD reporta EX (e VES), não EM | M | Média | Usar o script oficial de EX do BIRD e reportar o EM do Spider só como complemento | Validar o script com as SQLs gold (EX gold = 100%) |
| 3.7 | **Consultas lentas travam a avaliação por execução** (também no `sdss`, de 15 GB) | M | Alta | Timeout por consulta (o oficial do BIRD) e contar os timeouts à parte | Log de timeouts por modelo |
| 3.8 | **Test do BIRD é oculto** | B | Certa | Protocolo: validação tirada do train e dev como teste. Declarar | — |

## 4. Optuna e treino

| # | Risco | Grav. | Prob. | Prevenção | Detecção / plano B |
|---|---|---|---|---|---|
| 4.1 | **Poucos trials** (~8): o TPE mal sai da fase aleatória | M | Certa | Declarar como busca reduzida e fixar `n_startup_trials`. Mesma semente de sampler nos 3 benchmarks | Mostrar a curva "melhor valor × trial". Se ainda estiver caindo, dizer que a busca não convergiu |
| 4.2 | **MedianPruner com poucos trials** poda uma configuração boa que começa devagar (warmup alto) | M | Média | `n_warmup_steps` ≥ 2 épocas e 4 trials iniciais sem poda (já é assim) | Listar os trials podados e em que época |
| 4.3 | **Espaço de busca não cobre o ótimo** (lr no limite) | M | Média | Checar se o melhor lr está numa borda | Se estiver, ampliar e declarar |
| 4.4 | **Trial falha** (OOM ou NaN) e é contado como ruim ou some | M | Média | Estado FAIL separado de PRUNED (já existe). Guardar o log de cada trial | O loop de monitoramento marca FAIL e NaN no `STATUS.md` |
| 4.5 | **Checkpoints enchendo o disco**: 451 GB livres, t5-base tem ~1 GB por checkpoint | B | Baixa | `save_total_limit: 2` nos trials (já é assim). Apagar trials que não venceram só depois de fechar o estudo | Alerta abaixo de 50 GB livres |
| 4.6 | **fp16 na 1070**: T5 dá NaN e Pascal é lento em fp16 | M | Alta se ligado | fp32 no treino (como hoje) | Detectar NaN na loss e abortar |
| 4.7 | **Retomada do study com código diferente** mistura trials de versões diferentes | A | Média | O snapshot do R0.2 e o hash do commit gravado como `user_attr` em cada trial | Recusar a retomada se o hash mudou |

## 5. Operação e infraestrutura

| # | Risco | Grav. | Prob. | Prevenção | Detecção / plano B |
|---|---|---|---|---|---|
| 5.1 | **Fila de GPU de ~10 a 12 dias**: queda de energia, reboot ou atualização de driver | M | Média | Tudo retomável (Optuna em SQLite, checkpoint por época). Ordem de prioridade na fila | Loop de monitoramento. Após reboot, conferir `docker ps` e retomar |
| 5.2 | **Dois jobs de GPU ao mesmo tempo** (OOM de VRAM, tempos sem sentido) | M | Média | Fila única: um container de GPU por vez | `nvidia-smi` antes de cada lançamento |
| 5.3 | **Container sem `DGLBACKEND=pytorch` e `HOME=/tmp`** cai antes de treinar (documentado) | B | Média | Script de lançamento único para todos os runs | Falha em menos de 1 min |
| 5.4 | **Imagem Docker mudou** e a reprodução muda | M | Baixa | Registrar o digest da imagem em cada run | — |
| 5.5 | **Artefatos grandes no git** (study DB, checkpoints, BIRD) | B | Média | `.gitignore` (o `optuna_studies/` já está). Revisar o `git status` antes de cada commit | — |

## 6. Relato e documentação

| # | Risco | Grav. | Prob. | Prevenção |
|---|---|---|---|---|
| 6.1 | **Números antigos inválidos** (R0.1) continuam no texto do TCC | A | Certa até ser corrigido | Errata no `ESTUDOS.md` e no `.tex` principal. Runs marcados como inválidos, não apagados |
| 6.2 | **Número digitado de memória** diverge do arquivo | M | Média | Toda tabela é gerada por script a partir de `all_results.json` e `eval_results.json` |
| 6.3 | **Mistura de runs com protocolos diferentes** na mesma tabela | M | Média | Coluna "protocolo" (antigo com seleção no dev vs novo com validação do train) em toda tabela |
| 6.4 | **Significância exagerada** | A | Média | Todo "A > B" acompanhado de IC ou McNemar. Sem isso, escrever "sem diferença detectável" |

---

## Checklist antes de cada run

1. O código está congelado em commit e o hash foi gravado no diretório do run?
2. A GPU está livre (`nvidia-smi`)? O container tem `--memory`?
3. O config é novo? Nunca editar um config que já produziu um resultado.
4. Os splits vêm dos arquivos congelados e a checagem de sobreposição dá 0?
5. A loss do passo 1 está na faixa esperada (checagem do R0.1)?
6. Quantos exemplos foram filtrados, por tamanho de grafo ou truncamento, e isso foi registrado?

## Ordem proposta (menor risco primeiro)

1. **Sem GPU, agora:** piloto de 20 exemplos do 3B na CPU em fp32 (limite de RAM), download do BIRD, divisão dos splits e checagem de sobreposição, errata do R0.1 no `ESTUDOS.md` e no `.tex`.
2. **Depois da busca do t5-small:** trava contra pesos não carregados (R0.1, num worktree) e busca t5-base × Spider → ScienceBenchmark → BIRD.
3. **Por último:** treinos finais com 3 sementes, testes estatísticos e o relatório.
