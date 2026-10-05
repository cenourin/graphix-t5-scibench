# Histórico técnico de incidentes

Incidentes que afetam a validade de runs. Nada é apagado: os runs afetados ficam marcados, com a evidência (`PROTOCOLO.md` §7.5: falhas e trials podados são reportados).

## 2026-09-26: perda de GPU; trials RGAT treinaram na CPU **sem RGAT**

**O que aconteceu.** Às 02:44 (horário local), um `systemctl daemon-reload` disparado pelo snapd fez o container `optuna-search` perder o acesso à GPU ("Failed to initialize NVML"). Processos novos passaram a ver `torch.cuda.is_available() == False`, e o HF Trainer **caiu para a CPU sem erro**. Na CPU, o DGL 0.8.2 falha no SpMM ("Failed to generate libxsmm kernel for the SpMM operation"), e o `RGAT_Layer` legado capturava esse `DGLError` e **pulava o RGAT** daquele exemplo, também sem erro.

**Evidência direta** (reproduzida no teste T3, `PORTABILIDADE.md` §5.1): o DGL 0.8.2 falha na CPU em **16 de 16** grafos do conjunto fixo. Na CPU, o RGAT legado **nunca** roda.

**Runs afetados.** Estudo `scibench_t5small_rgat` (t5-small + RGAT, ScienceBenchmark), contando as linhas "RGAT_Layer.forward: DGLError … skipping graph augmentation" em `train_db_id/optuna/scibench_t5small_rgat/trial_N/train.log`:

| Trial | Início → fim (UTC) | RGAT pulado | Estado no estudo | Validade |
|---|---|---|---|---|
| 0 | 2026-09-26 01:23 → 05:59 | 0 | COMPLETE (eval_loss 0,8193) | válido. Terminou 15 min depois da perda de GPU, mas o processo já tinha o contexto CUDA aberto e continuou na GPU. |
| 1 | 05:59 → 12:45 | **81 917** | FAIL (rc=-9, morto por falta de memória) | **inválido: treinou na CPU, sem RGAT** |
| 2 | 12:45 → 14:35 | **21 102** | FAIL (marcado à mão) | **inválido: treinou na CPU, sem RGAT** |
| 3 | 14:35 → 14:39 | 0 | FAIL (duplicata do trial 0 por repetição do sampler) | não é resultado (2 min) |
| 4 | 14:40 → (parado) | 0 | RUNNING (busca interrompida) | incompleto |

**Nenhum dos trials inválidos entrou em resultado.** Os dois já estavam como FAIL, então não foram considerados pelo TPE nem pela escolha do melhor trial.

**Limitação que não dá para verificar.** Os 14 runs mais antigos em `train_db_id/graphix-*` não guardaram arquivo de log: a saída ia para o `docker logs` e os containers foram removidos. Para eles, não há como provar que **nenhum** exemplo teve o RGAT pulado. Rodaram na GPU, onde o DGL funciona, mas o comentário no próprio `rgat_tuning.py` legado registra uma falha de libxsmm num fine-tuning do ScienceBenchmark mesmo com o grafo na GPU. Portanto, **resultados RGAT antigos podem conter exemplos isolados sem RGAT**, e isso fica declarado como limitação ao citá-los. Runs novos sempre gravam log em arquivo.

**Correções:**
1. O snap foi travado (`sudo snap refresh --hold`) para não repetir o `daemon-reload`. Lembrar de `sudo snap refresh --unhold` ao fim dos treinos.
2. **O fallback silencioso virou erro fatal** (2026-09-29):
   - `seq2seq/run_seq2seq_train.py` se recusa a iniciar a variante RGAT sem CUDA. A exceção explícita, para testes, é `GRAPHIX_ALLOW_CPU_RGAT=1`.
   - O `RGAT_Layer` legado não captura mais o `DGLError`: a exceção sobe e o run falha.
   - O port moderno (`graphix_modern/rgat_tuning.py`) nasceu sem essa captura.

   Isso só altera o comportamento em caso de erro: a conta dos runs corretos é a mesma (o T3 continua passando).
3. O status monitorado no loop passou a conferir CUDA dentro do container.

## 2026-09-29: launcher legado executado no pod da imagem a5 (tentativa inválida)

**O que aconteceu.** No primeiro pod com a imagem a5 (`silveirabruno/graphix-modern@sha256:371f61af…`), antes de existir a sessão da Fase B, foi rodado direto o orquestrador **legado**, a partir do clone em `3463bf0`:

```
PYTHONPATH=/workspace/project/graphix-t5-scibench GRAPHIX_ALLOW_TF32=0 GRAPHIX_CODE_COMMIT=3463bf0 \
GRAPHIX_STUDY_DIR=/workspace/outputs/smoke/studies GRAPHIX_RUNS_DIR=/workspace/outputs/smoke/runs \
DGLBACKEND=pytorch python seq2seq/run_t5base_study.py --benches spider --arms rgat --smoke
```

Foi só `--smoke`: nunca `--probe` e nunca o estudo. Os dois trials falharam na importação, antes de carregar dados ou modelo: `ModuleNotFoundError: No module named 'tenacity'`, vindo de `seq2seq/utils/picard_model_wrapper.py`, que só o entrypoint legado `seq2seq/run_seq2seq_train.py` importa. O driver terminou com "No completed trials; not writing a best config".

**Por que não foi corrigido instalando o `tenacity`.** O entrypoint legado nunca foi validado na stack moderna (a Fase A validou `graphix_modern/train.py`). Fazê-lo rodar na a5 seria rodar código sem validação. A imagem a5 continua intacta, e o orquestrador do estudo será portado para `graphix_modern/train.py`.

**Artefatos e limpeza.**
- Em `/workspace/outputs/smoke/` ficaram o banco Optuna `spider_t5base_rgat_smoke.db`, os logs e os configs dos 2 trials (9 arquivos, 123 KB). Uma cópia está arquivada fora do repositório, em `pod_archive/outputs_smoke/` na máquina local. Os arquivos foram apagados do volume em 2026-10-05.
- Ficou também o symlink `/workspace/project/graphix-t5-scibench/data_all_in/t5-base -> /workspace/data/t5-base-st` no clone antigo. É inofensivo e foi mantido.
- `/workspace/optuna_studies`, `/workspace/runs`, `/workspace/study_run` e `/workspace/study_code` foram conferidos pela API S3 em 2026-10-05: não existem.

**Validade.** Nenhum resultado. Nenhum passo de treino foi executado.

**Prevenção.** `docs/RUNPOD_PHASE_B.md` e `docs/RUNPOD.md` avisam para não rodar o launcher legado na imagem moderna. A sessão da Fase B usa só `scripts/runpod_phase_b_probe.sh`.

## 2026-09-30: sessão `phase_b_s1` interrompida (incompleta, descartada)

**O que aconteceu.** A primeira sessão da Fase B na 4090 (`SESSION=phase_b_s1`, commit `cc5f69b`) passou no preflight, nas três etapas de smoke e no steps50. O processo do `probe_spider_rgat` também terminou: os resultados foram gravados às 11:39:15 UTC. Logo depois, o launcher parou sem gravar o evento de fim da etapa, sem mensagem de erro e sem chegar às etapas seguintes. O monitor de CPU parou no mesmo segundo. Isso é compatível com o pod ter sido parado, ou com o terminal web ter sido fechado e encerrado o processo; não há erro nos logs.

**Validade.** A sessão está incompleta e **nenhum número dela é usado**. A baseline da 4090 é a sessão `phase_b_s2` (2026-10-05), que rodou do início ao fim com o mesmo commit e a mesma imagem (`docs/port/phase_b/4090/`). Em s1, a inicialização do steps50 e do probe levou 321 s e 348 s, contra 10–16 s em s2, e o smoke_train levou 389 s no total, contra 17 s. Isso é compatível com a primeira montagem do cache de datasets do Spider no volume; em s2 o cache já existia.

**Artefatos.**
- `/workspace/outputs/phase_b_s1/` (logs, resultados e monitor) fica no volume, como evidência.
- Os pesos em `/workspace/checkpoints/phase_b_s1/` (74 arquivos, 6,75 GB) não foram limpos porque a sessão parou antes da etapa de limpeza. Os arquivos sem pesos foram arquivados em `pod_archive/checkpoints_phase_b_s1_noweights/` na máquina local, e a pasta foi apagada do volume em 2026-10-05.

**Prevenção.** Rodar a sessão sempre com `| tee` para um arquivo no volume e não fechar o terminal web antes de "STOP THE POD NOW". Uma sessão interrompida nunca é retomada: roda-se outra, com um `SESSION` novo, e o launcher se recusa a reutilizar um nome de sessão.
