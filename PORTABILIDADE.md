# Portabilidade para uma stack moderna (RTX 4090)

Análise de 2026-09-28. **Decisão 51 revista:** o ambiente final é uma stack moderna, otimizada para a RTX 4090. A imagem legada passa a servir só como **LEGACY_REFERENCE**, para testes de equivalência.

**Fora do escopo deste documento:** o protocolo experimental (`PROTOCOLO.md`) não é alterado aqui.

Nada foi portado ainda. O documento propõe versões, lista as incompatibilidades e define o plano incremental.

**Regra das fases:**
- **Fase A, equivalência funcional:** fp32 estrito (`allow_tf32=False` para matmul e cuDNN), sem autocast e sem `torch.compile`.
- **Fase B, otimização:** TF32, BF16, `torch.compile` e DataLoader. Só começa depois que a Fase A passar nos testes.

## 1. Fontes consultadas

| Fonte | O que confirma |
|---|---|
| `pytorch/pytorch` → `RELEASE.md`, "Release Compatibility Matrix" e "CUDA Support Matrix" (branch main, lida em 2026-09-28) | Python e CUDA suportados por cada versão do torch; arquiteturas de GPU por build |
| `dmlc/dgl` → releases do GitHub (a última é a **v2.4.0**, de 2024-09-03) | "torch 2.4 & CUDA 12.4 are now supported"; "numpy 2.x is now supported" |
| dgl.ai → "Get Started" (página oficial de instalação) | PyTorch **2.1.x a 2.4.x**; CUDA 11.8/12.1/12.4; Python 3.8 a 3.12; Linux Ubuntu ≥ 20.04 |
| `data.dgl.ai/wheels` (índice de wheels) | há wheels `dgl-2.5.0` para torch 2.5/2.6, **sem release nem notas oficiais**, e por isso **não usadas** como base |
| Commits do `dmlc/dgl` | o último é de 2025-07-31: o projeto está praticamente parado |
| PyPI (metadados) e código de `transformers 4.57.6` e `datasets 2.21.0` | requisitos de Python e torch; APIs presentes, renomeadas ou removidas (§4) |

Da matriz oficial do PyTorch, os pontos que decidem:
- **Pascal (a GTX 1070 local) só aparece nos builds CUDA 12.6 do torch 2.7 em diante.** Até o torch 2.6, os builds CUDA 12.x ainda incluem sm_50/60.
- **A Ada (8.9) não é listada explicitamente em nenhum build.** Ela roda os kernels sm_86 por compatibilidade binária dentro da família 8.x, o que vale para torch e DGL. Isso é confirmado no smoke do pod.
- **Driver do pod:** reporta CUDA 13.2 e é retrocompatível com runtimes 12.x. A versão do driver não é a do runtime do container.

## 2. LEGACY_REFERENCE

A imagem `eyuansu62/graphix-text-to-sql:v2` (30,9 GB, Ubuntu 18.04):

| Pacote | Versão |
|---|---|
| Python | 3.7.10 |
| torch | 1.9.0, CUDA 11.1 (kernels sm_37…sm_86); TF32 **ligado** por padrão |
| transformers | 4.17.0 |
| tokenizers | 0.11.6 |
| datasets | 1.18.4.dev0 |
| DGL | 0.8.2 |
| numpy | 1.21.4 |
| scipy | 1.6.2 |
| networkx | 2.6.3 |
| sentencepiece | 0.1.96 |
| stanza | 1.1.1 |

O pré-processamento (Stanza, construção dos grafos) **continua no legado**. A stack moderna consome os artefatos já gerados (§5, passo A0).

## 3. Matriz de versões modernas

### 3.1 MODERN_A, a primeira versão moderna (Fase A)

Mantém o DGL, dentro da faixa **oficialmente suportada** por ele.

| Componente | Versão | Justificativa |
|---|---|---|
| Python | **3.11** | Suportado pelo torch 2.4 (3.8–3.12) e pelo DGL 2.4 (3.8–3.12). Mais maduro que o 3.12 para wheels científicos da época. |
| torch | **2.4.0** | O DGL suporta oficialmente a série 2.4, mas o wheel `dgl-2.4.0+cu121` declara **`Requires-Dist: torch<=2.4.0`** (metadado conferido em 2026-09-29). Por isso foi usado o 2.4.0, e não o 2.4.1 aprovado inicialmente: com o 2.4.1, a instalação do DGL rebaixava o torch por conta própria. Suporta Ada, BF16, TF32 e `torch.compile`. |
| CUDA runtime | **12.1** (build `cu121`) | O CUDA **estável** do torch 2.4 (o 12.4 é "experimental" na matriz oficial). O DGL 2.4 publica `cu121`. **Inclui Pascal**, então o mesmo build roda na 1070 local e na 4090. |
| DGL | **2.4.0 + cu121** | Última versão oficial. Suporta torch 2.4. |
| transformers | **4.57.6** | Último patch da série 4; exige torch ≥ 2.2 e Python ≥ 3.9. A série 5.x ficou de fora: a 5.17 exige torch ≥ 2.5, e a série 5 removeu APIs de que o projeto ainda depende (§4). |
| tokenizers | a que a transformers 4.57.6 fixar | O tokenizador é conferido token a token (teste T1). |
| datasets | **2.21.0** | Última série 2.x. **Ainda tem `load_metric` e scripts de carregamento** (com `trust_remote_code`), dos quais o projeto depende. A 3.0 removeu `load_metric`, e a 4.0 removeu os scripts. |
| numpy | **1.26.4** | O DGL 2.4 declara suporte a numpy 2, mas o numpy 2 não traz ganho aqui e aumentaria o número de variáveis no teste de equivalência. Pode subir depois da Fase A. |
| sentencepiece, scipy, networkx | as que os pacotes acima exigirem | Não afetam o modelo, e o código só usa APIs estáveis delas. |

**Implementação (A1, 2026-09-29): imagem `graphix-modern:a1`**
- `docker/modern/Dockerfile`: base `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` fixada pelo digest `sha256:68c022c2…`, com Python 3.11.9, torch 2.4.0, CUDA 12.1, cuDNN 9.1.0, numpy 1.26.4 e Ubuntu 22.04.
- `docker/modern/requirements-a1.in` → `requirements-a1.lock`: tudo o que é acrescentado à base, instalado com `--no-deps`.
- **Dependências indiretas fixadas na mesma época** do datasets 2.21 e do DGL 2.4: pandas 2.2.3, pyarrow 17.0.0, scipy 1.14.1, fsspec 2024.6.1 e protobuf 5.28.2. Sem isso, o pip escolheria pandas 3, pyarrow 25, scipy 1.17 e protobuf 7.
- **Resultado:** tokenizers 0.22.2 (exigido pela transformers 4.57.6) e sentencepiece 0.2.0. O `pip check` passou limpo.
- **Verificado na GTX 1070:** CUDA disponível, arquiteturas sm_50…sm_90 e troca de mensagens do DGL na GPU.

### 3.2 MODERN_B, o alvo possível da Fase B

É o MODERN_A com o RGAT reescrito em PyTorch puro, sem DGL em tempo de execução.

| Componente | Versão | Justificativa |
|---|---|---|
| Python | 3.12 | Suportada pelo torch 2.8 (3.9–3.13) |
| torch | **2.8.x**, build **cu126** | CUDA 12.6 é estável no 2.8, e é o **único build do 2.8 que ainda inclui Pascal**, então o mesmo build roda na 1070 e na 4090. Não uso o mais recente (2.14): o 2.8 tem mais de um ano de correções e é a versão do template do RunPod. |
| DGL | **só no LEGACY_REFERENCE**, para exportar os grafos | Sem wheels oficiais depois do torch 2.4. |
| transformers, datasets, numpy | como no MODERN_A | A única mudança é tirar o DGL. |

**Quando passar do A para o B:** só se a Fase B mostrar que o DGL impede um ganho real, e a justificativa entra aqui. Os limites conhecidos do DGL são três:
- os kernels SpMM do DGL **não aceitam 16 bits** (erro já reproduzido neste repositório), então o RGAT continua em fp32 dentro do BF16;
- o `torch.compile` quebra o grafo em toda chamada ao DGL;
- o `graph.to(device)` roda por camada e por exemplo.

A troca exige demonstrar equivalência **contra o MODERN_A** (teste T3 com os mesmos pesos).

## 4. Incompatibilidades de API do projeto

Cada linha foi conferida no código da versão-alvo, não inferida.

| # | Onde no projeto | API legada | Situação na versão-alvo | Ação no port |
|---|---|---|---|---|
| 1 | `models/modeling_t5.py` (fork do T5 4.17, Graphix) | T5 com cache em tupla, `past_key_value`, `checkpoint()` direto, `parallelize()` | O T5 da transformers 4.57.6 usa `Cache`/`EncoderDecoderCache`, `cache_position`, `past_key_values` (o nome antigo está obsoleto, com remoção prevista para a 4.58) e `_gradient_checkpointing_func` | **Reaplicar as alterações Graphix (§6) sobre o `modeling_t5.py` da 4.57.6**, num arquivo vendorizado novo. Os remendos de precisão (§6.2) ficam de fora, e `parallelize` é removido. |
| 2 | `models/graphix/rgat_tuning.py` | `fn.copy_edge` | **Removida** no DGL 2.x (virou `fn.copy_e`) | trocar |
| 3 | `rgat_tuning.py`, `modeling_t5.py` | `number_of_nodes()` / `number_of_edges()` | obsoletos (`num_nodes()` / `num_edges()`) | trocar |
| 4 | `graph_pedia_*.bin` (dados) | pickle/shelve de `DGLGraph` do DGL 0.8.2 + Python 3.7 | desserialização no DGL 2.4 **sem garantia** | **exportação única** no legado para um formato neutro (§5, A0) |
| 5 | `rgat_tuning.py` | `except DGLError` → pula o RGAT daquele exemplo | continua possível | no port **vira erro**, porque a equivalência não pode mascarar falha. `empty_cache` + retry é removido (contorno de 8 GB). |
| 6 | `utils/picard_model_wrapper.py` | `transformers.generation_utils` e `transformers.generation_logits_process` | **Módulos removidos** (hoje em `transformers.generation`) | o PICARD não é usado. O import passa a acontecer só com `use_picard=True`. |
| 7 | Vários (`file_utils.copy_func`, docstrings) | `transformers.file_utils` | existe como compatibilidade na 4.57, e some na série 5 | migrar para `transformers.utils` |
| 8 | `models/modeling_auto.py` | internos de `auto_factory` (`_get_model_class`, `CONFIG_MAPPING_NAMES`, `model_type_to_module_name`, `replace_list_option_in_docstrings`) | API interna; mudou entre versões | usar a classe Graphix-T5 diretamente, sem a fábrica automática |
| 9 | `dynamic_module_utils.get_class_from_dynamic_module` | assinatura antiga | assinatura mudou | não é usado no caminho de treino; remover o import |
| 10 | Configs e scripts (`study_t5base_*.json`, `run_optuna_search.py`, `run_t5base_study.py`) | `evaluation_strategy` | **Renomeado** para `eval_strategy`; o nome antigo não é mais aceito | renomear as chaves (novos configs; os configs legados ficam como estão) |
| 11 | `utils/trainer.py`, `run_seq2seq_train.py` | `Trainer(tokenizer=...)` | obsoleto (`processing_class`), removido na 5.0 | trocar |
| 12 | `utils/trainer.py` (`evaluate`/`predict`/`prediction_step` sobrescritos) | contrato interno do `Seq2SeqTrainer` 4.17 (`evaluation_loop`, `_memory_tracker`, `gen_kwargs`) | `evaluation_loop` com a mesma assinatura. `prediction_step` e o tratamento de `gen_kwargs` mudaram. | reescrever as sobrescritas contra a 4.57.6. Teste T8 (geração). |
| 13 | `utils/dataset_loader.py`, `metrics/*` | `datasets.load_metric`, `datasets.Metric`, scripts de dataset | existem na 2.21 (obsoletos); scripts exigem `trust_remote_code=True` | Fase A: `datasets==2.21.0` + `trust_remote_code=True`. Depois: carregadores diretos e chamada direta das métricas (sai o `datasets`). |
| 14 | `run_seq2seq_train.py` (tokens extras `" <="`, `" <"`) | `AddedToken` do tokenizers 0.11 | os padrões de `AddedToken` (lstrip/rstrip/normalized) mudaram entre versões | teste T1: `input_ids` idênticos. Se divergirem, fixar os parâmetros do `AddedToken` explicitamente. |
| 15 | `modeling_t5.py` (gradient checkpointing) | `torch.utils.checkpoint.checkpoint` sem `use_reentrant` | o torch 2.4 avisa, e o padrão vai mudar | fixar `use_reentrant=True` (comportamento legado) na Fase A |
| 16 | `run_seq2seq_train.py` (`GRAPHIX_INIT_STATE_DICT`), retomada do HF | `torch.load` | no torch 2.4 o `weights_only` padrão é False; no 2.6+ é True | passar `weights_only=True` explícito (os `state_dict` são só tensores) |
| 17 | Treino (`adafactor: true`) | `TrainingArguments.adafactor` → `Adafactor(scale_parameter=False, relative_step=False)` | ainda existe na 4.57.6 | conferir que os kwargs do otimizador são os mesmos (teste T9) |
| 18 | `run_seq2seq_train.py` | `graph_pedia` via `pickle` + `shelve` (`dbm.dumb`) | o formato `dbm.dumb` é legível no 3.11, mas o conteúdo é `DGLGraph` (ver #4) | carregar do formato neutro |
| 19 | `modeling_t5.py` (remendos fp16/bf16) | upcasts manuais para contornar a Pascal | desnecessários na Ada e com autocast | **removidos** do port (§6.2), com registro |

## 5. Plano incremental

Cada passo termina num teste verde, e nenhum passo da Fase B começa com a Fase A pendente.

**Fase A: equivalência funcional**, sempre em `modern_fp32_reference` (fp32, TF32 off, sem autocast, sem compile).

| Passo | O que é feito | Teste que fecha o passo |
|---|---|---|
| A0 | **No legado:** exportar cada entrada do `graph_pedia` para um formato neutro (`src`/`dst` em ordem de ID, `num_nodes`, relações e demais campos) + manifesto sha256 | T2: reconstruir cada grafo no legado a partir do export e comparar com o `DGLGraph` original, com arestas e ordem idênticas |
| A1 | Ambiente MODERN_A reproduzível (imagem ou venv com versões fixas) | imports; `torch.cuda` na 1070; versões registradas |
| A2 | Tokenizador | **T1**: `input_ids` idênticos em todos os exemplos de train, val e dev dos dois benchmarks |
| A3 | Port do RGAT para o DGL 2.4 (#2, #3, #5) | **T3**: com os mesmos pesos e entrada, a saída do `RGAT_Layer` legado × moderno tem max \|Δ\| ≤ 1e-5 |
| A4 | Port do T5 Graphix sobre o T5 4.57.6 (#1, #15, #19) | **T4**: um checkpoint legado carrega com `strict=True`, mesma lista de parâmetros e shapes. **T5**: logits max \|Δ\| relativo ≤ 1e-4 e loss \|Δ\| ≤ 1e-5, com dropout off |
| A5 | Gradientes | **T6**: cosseno ≥ 0,9999 por tensor e o mesmo conjunto de parâmetros com gradiente não nulo, incluindo `relation_emb` e as 12 camadas RGAT |
| A6 | Infra de treino e avaliação (#6–#14, #16–#18) | **T7**: checkpoint salvo e recarregado sem perdas. **T8**: geração greedy com SQL idêntico. **T9**: mesmo otimizador e hiperparâmetros efetivos |
| A7 | Treino curto de 50 passos, mesma seed e mesma ordem de dados, legado × moderno | **T10**: curvas de loss com divergência ≤ ~1e-3 nos primeiros passos. Qualquer divergência maior precisa ser explicada antes de seguir. *(Critério substituído: ver a linha T10 em §5.1.)* |

**Profiling desde a Fase A (decidido em 2026-09-28).** Medir o tempo de cada etapa, sem mudar nenhum comportamento: sem sincronizar a GPU fora do modo de profiling e sem mudar ordem, dtype ou dispositivo. Serve para, na Fase B, atacar o gargalo real em vez de supor que ele está no T5. Nada é otimizado antes de T1 a T10 ficarem verdes.

| Etapa medida | Onde |
|---|---|
| DataLoader (espera por batch) | loop de treino |
| Montagem e reconstrução do grafo (`graph_factory`) | `rgat.Model` |
| Cópia CPU → GPU do batch | `Trainer._prepare_inputs` |
| `graph.to(device)` | `RGAT_Layer.forward` |
| RGAT (`propagate_attention` + FFN) | `RGAT_Layer.forward` |
| Encoder T5 (sem o RGAT) | `T5Stack` do encoder |
| Decoder T5 + `lm_head` | `T5Stack` do decoder |
| Backward | `Trainer.training_step` |
| Passo do otimizador | loop de treino |

O profiling é ativado por variável de ambiente e fica desligado nas runs normais. Com ele ligado, cada medida usa `torch.cuda.synchronize()` nos limites da etapa, o que atrasa a execução; por isso os números de throughput do benchmark (B6) vêm de runs **sem** o profiling.

**O que não muda nas Fases A e B**, até uma discussão separada depois do B7, porque mudaria a dinâmica numérica do treino e não é portabilidade:
- batch físico 1;
- acumulação de gradiente em {8, 16, 32, 64};
- filtros de exemplos;
- seeds;
- splits;
- Adafactor;
- schedule de learning rate.

- **Onde rodam os testes** *(corrigido em 2026-09-29, depois do A3)*:
  - **T3 a T10 não têm referência na CPU.** O DGL 0.8.2 do legado não roda o RGAT na CPU (16 de 16 grafos falham no SpMM via libxsmm).
  - **O LEGACY_REFERENCE válido para o caminho RGAT é a GTX 1070**, com os dois ambientes na mesma GPU. Como a Pascal não tem TF32, o lado legado fica em fp32 estrito.
  - **O T1 (tokenizador) e o T2 (grafos) não passam pelo RGAT** e continuam como estavam.
  - **Na 4090 do pod** não há legado a comparar. Ali roda só o moderno, e ele é comparado com as saídas de referência gravadas na 1070.
  - A premissa original ("testes primeiro na CPU, para comparação determinística") fica registrada aqui como superada.
- **Pipeline:** `scripts/port_pipeline.sh` roda em ordem todos os testes já implementados, cada um no seu ambiente, para no primeiro que falhar e grava `data_all_in/data/port_tests/pipeline_summary.json` com o commit e as imagens. Cada passo novo acrescenta o seu teste ao pipeline, e só é dado como concluído quando o pipeline **inteiro** passa.
- **Fallback para CPU no caminho RGAT agora é erro fatal** (`docs/incidentes.md`): o `run_seq2seq_train.py` recusa RGAT sem CUDA, e o `RGAT_Layer` legado não pula mais o RGAT em silêncio.
- **Conjunto fixo:** 8 exemplos do Spider e 8 do ScienceBenchmark, incluindo grafos grandes do `oncomx`.
- **Diferenças numéricas esperadas:**
  - a ordem das somas (atomics do DGL, kernels de redução) e as versões de cuBLAS e cuDNN dão erro relativo de ~1e-6 a 1e-5 por operação, acumulado em 12 camadas;
  - divergência na geração só é aceitável se vier de um empate de logits, e cada caso é investigado.

**Fase B: otimização para a Ada.** Cada passo é medido contra o `modern_fp32_reference`.

| Passo | O que é feito |
|---|---|
| B1 | `--precision tf32` (`allow_tf32=True` para matmul e cuDNN), registrado nos metadados |
| B2 | `--precision bf16`: `torch.autocast(bfloat16)` com pesos e estado do Adafactor em fp32. Mede VRAM, throughput e estabilidade da loss. Com DGL, o RGAT fica em fp32. |
| B3 | *(condicional)* MODERN_B: RGAT em PyTorch puro, validado contra o MODERN_A com os testes T3 a T6 |
| B4 | `--compile` opcional, com os graph breaks documentados |
| B5 | DataLoader: levar a montagem do grafo do `forward` para o collate; depois `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor` e `non_blocking` |
| B6 | Instrumentação por run (GPU, capability, driver, versões, precisão, TF32, compile, batch, acumulação, tokens/s, passos/s, s/época, pico de VRAM) e benchmark na 4090: A fp32, B fp32+TF32, C bf16, D bf16+compile, no mesmo conjunto |
| B7 | Recomendação da configuração para a 4090. **A mudança no protocolo vem só depois disso, como proposta separada.** |

### 5.1 Andamento

| Passo | Status | Evidência |
|---|---|---|
| A0 | ✅ 2026-09-29 | **T2 passou em 100%**: 14 642 grafos (9611 do Spider, 4732 do train e 299 do dev do ScienceBenchmark), 125,2 milhões de arestas e 14 642 associações exemplo ↔ grafo. Export rodado a partir do commit `dbf5832`; uma segunda execução gerou artefatos idênticos byte a byte. Evidências em `data_all_in/data/graph_export/manifest.json` e `T2_report.json`. |
| A1 | ✅ 2026-09-29 | Imagem `graphix-modern:a1` (commit `7c46c63`), publicada como `silveirabruno/graphix-modern:a1`, digest `sha256:7f9513344460f65cda11f29236ea724025ae0786c86bf441efc5d6c14d276a75` |
| A2 | ✅ 2026-09-29 | **T1 passou**: `input_ids`, `labels` e `attention_mask` idênticos nos 14 642 exemplos. O tokenizador tem os mesmos 32 102 tokens, e os tokens extras `" <="`/`" <"` ficam em 32100/32101. Relatório em `data_all_in/data/port_tests/T1/T1_report.json` (`scripts/port_t1_tokenizer.py`). |
| A3 | ✅ 2026-09-29 | RGAT portado para `graphix_modern/rgat_tuning.py` com escopo mínimo (`copy_e`; `DGLError` não é mais capturado; sai o `empty_cache`+retry). **T3 na GPU passou**: saída da camada com max\|Δ\| entre 9,5e-7 e 2,3e-6 nos 16 exemplos do conjunto fixo (`tests/port/fixture.json`). Relatório estágio a estágio em `data_all_in/data/port_tests/T3/T3_report_cuda.json`. |
| A4 | ✅ 2026-09-29 | T5 da transformers 4.57.6 com as alterações Graphix, gerado por `scripts/port_build_modeling_t5.py` (diff para o upstream em `docs/port/modeling_t5_graphix.diff`). **T4 passou nos 7 gates**: 451 `named_parameters` idênticos; o checkpoint legado carrega com `strict=True` (0 ausentes, 0 inesperadas, tudo bit a bit). **T5 passou**: nos 14 exemplos usáveis, \|Δloss\| ≤ 9,5e-7 e logits com erro relativo ≤ 9,0e-7, sem nenhum ponto de controle interno acima de 1e-4. Os 2 grafos do Spider maiores que 512 tokens, que o treino descarta, ficaram de fora. |
| A5 | ✅ 2026-09-29 | **T6 passou.** Os gradientes foram acumulados sobre os exemplos em três cenários: R1, Spider sem checkpointing; R2, Spider com checkpointing; R3, ScienceBenchmark com checkpointing. O cosseno mínimo por tensor foi 1,00000000 / 1,00000000 / 0,99999916 (gate: 0,9999). O conjunto de parâmetros com gradiente é o mesmo (438, sendo 181 Graphix; 13 sem gradiente, incluindo o `filter` e o `relation_emb` do decoder). Dentro de cada ambiente, o checkpointing não altera os gradientes (diferença relativa ~1e-6). |
| A6.1 | ✅ 2026-09-29 | Avaliadores em `third_party/`: são os forks ElementAI do PICARD, **não os oficiais**; procedência e implicações em `third_party/PROVENANCE.md`. nltk 3.7 e o mesmo `punkt` do legado, com `word_tokenize` idêntico. |
| A6.2a | ✅ 2026-09-29 | **TD passou.** Filtros (treino, validação e dev), exemplos do dev (gabaritos), alinhamento, esquemas e tamanho do treino são idênticos nas duas bases. O campo P fica fora do gate, como "diferença conhecida e não funcional" (ver achados). |
| A6.2b | ✅ 2026-09-29 | **TW passou.** `graph_batch` idêntico nos 14 exemplos, e \|Δloss\| ≤ 9,5e-7 no braço RGAT e ≤ 1,4e-6 no plain. |
| A6.2c | ✅ 2026-09-29 | **Gate TRACE passou**, idêntico ao legado: N=10, GA=4, 3 épocas; 6 atualizações; sobras atravessando a fronteira (`w.grad` 1,5); LR por passo; ordem nas 3 épocas. Trainer em `graphix_modern/trainer.py`: laço da 4.17, sampler do torch 1.9, loss ÷ GA, OOM fatal. |
| A6.2d | ✅ código 2026-09-29 | `graphix_modern/train.py`: port do `run_seq2seq_train.py` com a mesma interface (config JSON e variáveis `GRAPHIX_*`). Mudanças de interface: export do A0, `evaluation_strategy` → `eval_strategy`, PICARD recusado, pesos iniciais em safetensors ou com `weights_only`, modelo salvo em safetensors. Smoke de ponta a ponta pendente. |
| A6.2e | ✅ código 2026-09-29 | `graphix_modern/profiling.py`, desligado por padrão (`GRAPHIX_PROFILE=1` liga). Seções: `data_wait`, `prepare_inputs`, `graph_build`, `graph_to`, `rgat`, `encoder`, `decoder`, `lm_head`, `backward`, `optimizer` e `evaluate`. Grava `profile.json` na pasta do run. Com gradient checkpointing, `rgat` e `encoder` incluem a recomputação feita no backward. |
| T9 | ✅ 2026-09-29 | Adafactor com os mesmos hiperparâmetros e agrupamento de parâmetros, mesmo scheduler e **LR idêntico em cada passo** (diferença 0,0) em três configurações do estudo. `max_steps` 6885 no Spider, igual ao do trial legado de 2026-09-28. |
| Pipeline na `a5` | ✅ 2026-09-29 | T2, T1, T3, A4_build, T4, T5, T6, TD, TW e TRACE verdes na imagem `silveirabruno/graphix-modern:a5` (`sha256:371f61af…`), commit `33d939f`. |
| Smoke do ponto de entrada | ✅ 2026-09-29 | Spider, 64 exemplos, 2 épocas e GA 16 na GTX 1070: treino com avaliação só da loss (2,912 → 2,680), melhor checkpoint recarregado, `model.safetensors`, `profile.json`. Depois, avaliação no dev (20 exemplos) com `GRAPHIX_INIT_STATE_DICT` estrito: geração, EM/EX pelos avaliadores e decodificação com a limpeza de espaços do legado. Arquivos em `data_all_in/data/port_tests/TE/`. |
| T8 | ✅ 2026-09-29 | Geração greedy pelo mesmo caminho do `prediction_step`: `max_length` 512, cache (`Cache` da 4.57), `graph_idx`, decode com a limpeza de espaços do legado. **Tokens gerados idênticos em 14/14 exemplos** nos três cenários: RGAT com pesos do T4, RGAT treinado no smoke e plain com o T5 padrão. |
| T10 (A7) | 🔁 critério redefinido em 2026-09-29, antes da nova rodada | Os dois pontos de entrada reais, com dropout neutralizado e os mesmos pesos iniciais; Spider, 206 micro-batches, GA 8, 2 épocas (25 atualizações/época, 6 sobras). **Mecânica idêntica em todas as rodadas até aqui:** `global_step` 50, `max_steps` 50, LR em cada passo, ordem dos 406 itens, sobras entrando na 1ª atualização da época 2. **Histórico do critério de loss:** (1) o limite original de max \|Δloss\| ≤ 1e-3 foi invalidado pela constatação de que o próprio legado varia mais do que isso entre execuções nominalmente idênticas (GPU não determinística: somas atômicas no DGL e no cuBLAS; 50 passos amplificam 1e-6 até 10⁻²). (2) O critério seguinte, de máximo contra envelope com **um único par por stack**, passou em 28/09 (cross 6,8e-2 contra 7,7e-2 e 9,0e-2), mas **reprovou** na rodada do pipeline de 29/09 (commit fea4973): cross 1,16e-1 contra 6,0e-2 e 7,6e-2. A análise dos 5 runs dessa rodada mostrou que o máximo de um único par é ruidoso demais: os 4 pares cross-stack ficam entre 5,4e-2 e 1,16e-1 no max e entre 1,69e-2 e 2,08e-2 na média, contra 6,0e-2 a 7,6e-2 e 1,74e-2 a 2,27e-2 nos pares da mesma stack, e sem viés de sinal (médias das curvas entre 1,2945 e 1,2998). **Portanto, o "passou" de 28/09 foi uma única amostra favorável, não evidência suficiente.** (3) **Critério atual, fixado antes da nova rodada**, com o run como unidade estatística (os passos de uma curva são autocorrelacionados). São 3 runs por stack, o que dá 6 pares da mesma stack (o envelope) e 9 pares cross-stack; o run perturbado em 1e-6 serve só de diagnóstico. **Passa se:** a mecânica for exatamente idêntica nos 6 runs; a **mediana** dos 9 pares cross-stack em média relativa, máximo relativo e RMSE da curva não exceder o maior valor da mesma stack; e não houver viés consistente de sinal (reprova se todas as médias de loss dos runs legados ficarem do mesmo lado de todas as modernas **e** a distância entre as stacks superar a maior distância dentro de uma stack). Usa-se a mediana, e não "todos os 9", porque com 15 sorteios intercambiáveis o maior cai entre os 9 cross em 60% das vezes mesmo sem efeito de stack. O `first_divergence_step` é **descritivo**: mede o instante da primeira diferença de arredondamento, não sua magnitude, e os kernels das duas stacks já diferem em ~1e-6 (T5/T6), então é esperado que o cross-stack saia da 4ª casa decimal 1 ou 2 passos antes. É uma equivalência empírica dentro da variabilidade intrínseca da GPU, não um teste formal de significância, porque 3 runs por stack não sustentam inferência forte (`scripts/port_t10_training.py`, `scripts/port_t10_noise.sh`). |
| T7 | ✅ 2026-09-29 (a reconfirmar no pipeline final, que parou no T10) | Retomada isolada com todas as execuções partindo do mesmo `checkpoint-30`. **G1:** `global_step` 50 e LR igual ao do run ininterrupto nos passos 31 a 50. **G2:** os 160 itens treinados nos passos 31 a 50 são exatamente os do run ininterrupto. **G3:** loss do passo 31 idêntica entre duas retomadas (0,5441). **G4:** duas retomadas idênticas nos passos 32 a 35 (0,0), enquanto o controle com o estado do Adafactor esvaziado se afasta até 6,8e-3, o que mostra que o estado do otimizador é restaurado e importa. Primeiro desenho (comparar com um run ininterrupto) descartado: os dois já divergem por ruído de GPU antes do checkpoint (`scripts/port_t7_resume.py`, `scripts/port_t7_resume2.sh`). |

**Achado do T10: fragmentação de memória na GTX 1070.** O T10 moderno parou por OOM no passo 11 de 50, fatalmente, sem pular micro-batch. A memória alocada era a mesma do legado (6,08 GB contra o pico de 6,26 GB no legado), mas o alocador do torch 2.4 tinha 1,49 GB reservados e sem uso. Com `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, que é uma configuração só do alocador e não muda nenhuma conta, o moderno completou os 50 passos com pico de 6,09 GB alocados e 6,65 GB reservados, **abaixo do legado**. Nos 24 GB da 4090 isso é irrelevante, mas a configuração fica valendo nos runs locais.

**Achados do smoke do ponto de entrada:**
1. **`adam_eps` é uma chave morta nas configs.** O `TrainingArguments` chama o campo de `adam_epsilon`, e a 4.17 ignorava chaves desconhecidas em silêncio, então o valor nunca foi usado. Isso não afeta o estudo, porque o Adafactor não usa o epsilon do Adam. O port aceita chaves extras, como o legado, mas lista quais ignora.
2. **A 4.57 remove colunas também de `torch.utils.data.Dataset`.** Ela embrulha o collator num `RemoveColumnsCollator`, que descartava o `graph_idx` na avaliação. A 4.17 só removia colunas de `datasets.Dataset`. O port desliga essa remoção.
3. **O safetensors grava os pesos amarrados uma vez só** (`shared.weight`). A carga estrita restaura as três chaves amarradas (embeddings do encoder e do decoder e `lm_head`) a partir dele, e só elas.
4. **Um no-op acidental do legado.** No `_post_process_function`, o `np.where(label_ids != -100, ...)` é aplicado a uma lista Python irregular; a comparação dá um único `True` e os labels voltam inalterados. O numpy recente recusa o array irregular. O port reproduz o efeito, sem alterar os labels, que aliás não contêm `-100`.

**Achados do A6:**
1. **Imagem `a5` consolidada.** O accelerate 1.3.0 é o mínimo que funciona com a transformers 4.57.6: o metadado declara ≥ 0.26, mas a 1.1.1 e a 1.2.1 falham em `unwrap_model(keep_torch_compile=...)`. O rapidfuzz 2.0.5 do legado não tem wheel para Python 3.11 e o sdist traz Cython incompatível com 3.11. Um estágio de build regenera o Cython (3.0.0b1, porque a 3.0.0a10 pedida saiu do PyPI) a partir do `.pyx` original, com o SHA-256 do sdist conferido. O algoritmo, em C++, não muda. O `fuzz.ratio` dá o mesmo valor do legado.
2. **O campo P (dev pré-processado pelo `datasets`, com valores do banco) é uma diferença conhecida e não funcional.** Três evidências:
   - **nenhum consumidor funcional:** o modelo lê o `TokenizedDataset` (T1), a métrica lê os exemplos do dev (E), e o P só aparece num log de duplicatas;
   - **sem referência estável:** o `bridge_content_encoder` junta valores num `set()` e usa `SELECT DISTINCT` sem `ORDER BY`, e **duas execuções do mesmo legado já divergem em 9 exemplos do Spider**;
   - **não é problema de versão:** a diferença persiste entre legado e moderno com o mesmo rapidfuzz 2.0.5 e `PYTHONHASHSEED=0`.

   O rapidfuzz 2.0.5 foi fixado mesmo assim.
3. **`decode` do tokenizador.** O padrão de `clean_up_tokenization_spaces` mudou (4.17 `True`, 4.57 `False`). O trainer moderno fixa `True` explicitamente nas três decodificações (entradas, labels e SQL predito).
4. **Semântica de treino da 4.17, confirmada empiricamente** (`scripts/port_trace_epochs.py`):
   - ⌊N/GA⌋ atualizações por época;
   - as sobras entram na 1ª atualização da época seguinte, cada micro-batch ÷GA;
   - na última época, o treino para em `max_steps` e **as sobras nem são processadas**.

   O Trainer padrão da 4.57 faz 9 atualizações em vez de 6 e, com um `forward(**kwargs)`, **não divide a loss por GA**.
5. **`RandomSampler` do torch 2.x.** Ele consome uma permutação extra por época (`randperm(n)[:num_samples % n]`, avaliado mesmo com resto 0), o que muda a ordem dos exemplos a partir da época 2. O port usa o `__iter__` do torch 1.9.
6. **Retomada com torch < 2.6.** O Trainer 4.57 bloqueia o `torch.load` de `optimizer.pt` e `scheduler.pt`. O port carrega esses arquivos com `weights_only=True` direto, **só** para checkpoints dos nossos próprios runs; a CVE-2025-32434 torna perigoso carregar arquivos de terceiros.

**Achados do A5:**
1. **A maior diferença relativa por elemento (1,4e-2, no R3) fica no `ffn.feedforward.0` do RGAT**, o `Linear` que vem **antes** de uma ReLU. O `feedforward.2`, depois da ReLU, quase não diverge. É o padrão de ReLU no limiar: a diferença de ~1e-6 do forward vira o sinal de algumas pré-ativações muito próximas de zero, e a derivada da ReLU é descontínua. O L2 relativo do tensor fica em 1,3e-3, e o cosseno em 0,9999992.
2. **O Graphix não admite backward com dropout desligado.** O `graph_caption` escreve in-place na fatia que o RGAT acabou de consumir. Com dropout ativo, o autograd guardou a saída nova do dropout, e tudo funciona. Com dropout como identidade (`p=0` ou `eval()`), o autograd guarda a própria fatia, e o backward falha, **no legado e no moderno igualmente**. Não afeta o treino, que sempre tem dropout. O teste usa um dropout que devolve `x * 1.0`. O comportamento in-place foi mantido no port, porque a Fase A exige equivalência, não limpeza.
3. **A atenção do T5 faz dropout pela função** (`nn.functional.dropout`), não por um módulo `nn.Dropout`. Na primeira tentativa, o harness não zerou esse dropout, e os gradientes divergiam até dentro do mesmo ambiente. Foi um erro do teste, corrigido zerando também `T5Attention.dropout`.

**Achados do A4:**
1. **Checkpoints `.bin` bloqueados.** A transformers 4.57 se recusa a ler `.bin` (pickle) com torch < 2.6, por causa da CVE-2025-32434, e o MODERN_A precisa do torch 2.4 por causa do DGL. O t5-base original foi convertido uma vez para `data_all_in/t5-base-st/model.safetensors` (`scripts/port_convert_t5_safetensors.py`), com os 258 tensores conferidos bit a bit e SHA-256 no `manifest.json`. O `data_all_in/t5-base/` do legado ficou intocado. **Consequência para o A6:** checkpoints do treino moderno devem ser salvos em safetensors; carregar `.bin` antigos exige `torch.load(..., weights_only=True)` direto, fora da transformers.
2. **Inicialização das camadas Graphix.** No 4.57, o `from_pretrained` inicializa as chaves ausentes só via `_init_weights`, que não conhece os `nn.Linear`, `nn.Embedding` e `nn.LayerNorm` genéricos do RGAT e do `relation_emb`. Um port ingênuo deixaria essas camadas com memória não inicializada. O port marca os módulos acrescentados pelo Graphix e aplica a inicialização padrão do torch, que era a que eles recebiam no legado. O T4-G7 confere isso contra a distribuição teórica.
3. **Critérios corrigidos no próprio T4, antes de passar** (erros do teste, não do port):
   - **G4:** o `T5Config` da 4.17 não tem `relative_attention_max_distance`, e o código legado usa o 128 fixo. O gate passou a comparar o valor **efetivo** de cada implementação.
   - **G7:** comparava duas amostras aleatórias com tolerância de ~2σ. Passou a conferir cada ambiente contra a distribuição teórica, com limite de 5σ.
4. **Checagem de reprodutibilidade no pipeline:** o `modeling_t5.py` commitado precisa ser idêntico ao que o gerador produz.

**Diagnóstico do T3, estágio a estágio (GPU):**
- **Onde nasce a diferença:** `edge_feats`, `k` e `v` são idênticos bit a bit. A primeira diferença aparece em `q`, a única projeção com bias: ~2e-6 absoluto, relativo ≤ 1e-6. Vem do kernel `addmm` do cuBLAS no torch 1.9 × 2.4, não do DGL.
- **Estágios que passam de 1e-5 em valor absoluto:** `score` (até ~69 em módulo), `wv` e `z`, que são somas com valores grandes. O erro relativo deles fica em 3 a 6e-7, o que equivale a poucos ULPs de float32 e é a diferença do `q` propagada. A normalização `o = wv/z` volta à escala 1.

**Achado: o RGAT legado não roda na CPU.** O DGL 0.8.2 falha em **16 de 16** grafos na CPU ("Failed to generate libxsmm kernel for the SpMM operation"). Essa versão não tem o `dgl.use_libxsmm`, que permitiria desligar o libxsmm. Consequências:
1. **O plano de testes muda.** O dispositivo de referência de todos os testes que passam pelo RGAT (T3 a T10) passa a ser a **GPU** (GTX 1070). Como a Pascal não tem TF32, o lado legado fica em fp32 estrito por construção.
2. **Confirmação do incidente de 2026-09-26.** Qualquer run legado que caiu para a CPU treinou **sem RGAT**, porque o `except DGLError` do legado pulava a camada em silêncio. Isso confirma o diagnóstico do incidente de perda de GPU.
3. **O moderno funciona na CPU.** O DGL 2.4 roda o RGAT na CPU (16 de 16, autoconsistente), então testes só do moderno podem usar a CPU.

O comparador do T3 também passou a recusar comparação com zero exemplos: antes, uma comparação vazia contava como aprovação.

## 6. Alterações Graphix no T5 (o que o port preserva)

O `seq2seq/models/modeling_t5.py` é o `modeling_t5.py` da transformers 4.17 (1845 linhas) com 199 linhas alteradas.

### 6.1 Algoritmo, que é portado exatamente

| Classe | Função | Alteração | Motivo | Equivalente moderno | Dificuldade |
|---|---|---|---|---|---|
| `T5LayerRGAT` (nova) | `__init__` | `RGAT_Layer(d_model, d_model, heads=1, feat_drop=0.2)`, `T5LayerNorm`, 2 dropouts e um `filter = Linear(d_ff, d_model)` **nunca usado** | camada de grafo | copiar. O `filter` morto fica, para os checkpoints baterem. | baixa |
| `T5LayerRGAT` | `forward` | `x + dropout(dropout_gnn(elu(graph_caption(layer_norm(x)))))` | injeção do grafo | copiar | baixa |
| `T5LayerRGAT` | `graph_caption(_one)` | por exemplo: primeiros `num_nodes` estados; `relation_emb(rel_ids)`; RGAT; **escrita in-place** no tensor normalizado | nós = subwords da pergunta e do esquema | copiar e validar os gradientes (T6) | média |
| `T5Block` | `__init__` | encoder: `self.rgat_layer = T5LayerRGAT(config)` | uma camada RGAT por bloco do encoder | acrescentar | baixa |
| `T5Block` | `forward` | kwargs `graph_batch` e `relation_emb`. No encoder, o RGAT roda depois do FF e antes do clamp de fp16. | a ordem define o modelo | acrescentar ao forward com cache moderno | média |
| `T5Stack` | `__init__` | `relation_emb = nn.Embedding(25, d_model)`, no encoder **e no decoder** (o do decoder não é usado) | 25 relações de `GRAPHIX_RELATIONS` | acrescentar os dois | baixa |
| `T5Stack` | `forward` | repassa `graph_batch` e `relation_emb` aos blocos, inclusive no gradient checkpointing | sem isso, o RGAT some com checkpointing | via `_gradient_checkpointing_func` | média |
| `T5ForConditionalGeneration` | `forward` | aceita `graph_batch` e o repassa ao encoder | ponto de entrada | acrescentar | baixa |
| `graphix/rgat.py` → `Model` | `graph_factory`, `graph_postprocess`, `generate` | monta o `graph_batch` pelo `graph_idx`; relação → ID; passa o grafo ao `generate` | pareamento exemplo-grafo | sem mudança de lógica. O `generate` moderno filtra os kwargs do encoder pela assinatura, então o `graph_batch` precisa estar nela. | média |

### 6.2 Remendos de precisão, que não são algoritmo e não são portados

Foram adicionados por sessões anteriores para avaliar o Graphix-3B em 16 bits numa GPU Pascal. **Em fp32 não fazem nada**, então não afetam a equivalência da Fase A.

| Local | O que faz | Por que sai |
|---|---|---|
| `T5DenseReluDense.forward`, `T5DenseGatedGeluDense.forward` | FFN em fp32 quando a entrada é 16 bits | no BF16 a faixa de expoente é a do fp32, sem o estouro do fp16 |
| `T5Attention.forward` | QKᵀ e AV em fp32 quando a entrada é 16 bits | o cuBLAS da Pascal não tem GEMM batched em bf16; na Ada não há essa limitação |
| `T5ForConditionalGeneration.forward` | `lm_head` em fp32 quando a entrada é 16 bits | com autocast, a cross-entropy já roda em fp32. Volta como opção documentada se a Fase B mostrar instabilidade. |

## 7. O que depende de você

1. **Aprovar a matriz MODERN_A** para a Fase A: Python 3.11, torch 2.4.1+cu121, DGL 2.4.0+cu121, transformers 4.57.6, datasets 2.21.0, numpy 1.26.4.
2. **Onde fica o ambiente moderno.** Recomendo uma **imagem Docker própria**, `FROM` uma imagem oficial `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` com as versões fixadas. No RunPod, a mesma imagem vira o pod, e aqui ela roda com `docker run`. Um venv no template do RunPod dependeria do Python do template. A imagem precisa ser publicada numa conta sua do Docker Hub.
3. **Começar pelo A0** (exportação dos grafos no legado), que não muda nada no código de treino.
