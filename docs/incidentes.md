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
