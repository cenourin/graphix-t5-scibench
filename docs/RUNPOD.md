# Rodar o estudo t5-base num pod (RunPod ou similar)

O estudo roda nesta máquina (GTX 1070), por decisão de 2026-09-26. Este guia deixa pronto o caminho para um pod, caso isso mude.

**Regra do protocolo:** as 4 células rodam num só lugar (`PROTOCOLO.md` §2.2). Não se divide o estudo entre a 1070 e um pod. Se ele mudar de máquina, isso entra em "Desvios" no `PROTOCOLO.md`, e as células já treinadas são refeitas.

## 1. Por que o lançador é outro

Aqui, `scripts/run_t5base_study.sh` roda `docker run` na imagem `eyuansu62/graphix-text-to-sql:v2`. Um pod **já é** um container: a imagem é escolhida na criação do pod, e dentro dele não há Docker. Por isso:

- o pod é criado **com a própria imagem** `eyuansu62/graphix-text-to-sql:v2`, e o ambiente fica idêntico: Python 3.7.10, torch 1.9.0 + CUDA 11.1, DGL 0.8.2, transformers 4.17.0;
- `scripts/run_t5base_study_pod.sh` roda o orquestrador direto com `python`. Ele aponta `GRAPHIX_STUDY_DIR`/`GRAPHIX_RUNS_DIR` para o volume e mantém a regra do snapshot do código (`RISCOS.md` R0.2).

## 2. Qual GPU escolher

O torch 1.9 com CUDA 11.1 traz kernels até a arquitetura **Ampere** (sm_86).

| GPU | Situação |
|---|---|
| RTX 3090 / A5000 / A6000 / A40 (24–48 GB, Ampere) | **recomendadas** |
| A100 (Ampere) | funciona, mas é cara para este tamanho de modelo |
| RTX 4090 / L4 / L40S (Ada), H100 (Hopper) | **não testadas**; podem não rodar. Se usar, rode o smoke antes. O lançador avisa. |

O protocolo cabe em 8 GB, então qualquer uma das GPUs acima sobra. Uma GPU maior só acelera o treino **se** o protocolo mudar (batch maior, sem checkpointing, bf16), e isso seria um desvio registrado, igual para as 4 células.

## 3. Volume persistente

O `/workspace` do RunPod persiste entre reinícios do pod. Tudo o que importa fica nele:

```
/workspace/
├── graphix-t5-scibench/      repositório (git clone) + dados extraídos em data_all_in/
├── optuna_studies/           estudos Optuna (*.db), t5base_study.jsonl, pipeline.out
├── runs/                     trials (optuna/) e runs finais (study-t5base-<bench>-<arm>/)
├── study_code/               snapshot congelado do código (COMMIT + uncommitted.diff)
└── study_run/                diretório de execução (links para o snapshot e os dados)
```

O que se pode descartar: o próprio container, porque o lançador reinstala o optuna no volume se faltar. O que não se pode perder: `optuna_studies/` e `runs/`.

## 4. Dados (~31 GB)

Nesta máquina:

```
scripts/pack_for_pod.sh /caminho/com/espaco     # gera graphix_study_data.tar + manifesto sha256
```

O tar leva só o que o estudo lê, em caminhos relativos à raiz do repositório:

| Conteúdo | Tamanho |
|---|---|
| Bancos SQLite do ScienceBenchmark | 16 GB |
| Saída pré-processada do ScienceBenchmark | 9,8 GB |
| Saída pré-processada do Spider (`graph_pedia_total.bin` etc.) | 3,8 GB |
| `t5-base` original | 0,85 GB |
| Bancos do Spider | 0,84 GB |
| Splits (`data_all_in/data/splits/`) | 79 MB |

O manifesto `pod_data.sha256` vai dentro do tar e é conferido no pod. Os splits também podem ser regerados com `scripts/make_splits.py` (seed 42), e o resultado tem que bater com `splits/*.json`, que estão versionados.

## 5. Passo a passo no pod

```bash
cd /workspace
git clone git@github.com:cenourin/graphix-t5-scibench.git   # branch trained-models
cd graphix-t5-scibench && git checkout trained-models
# copie graphix_study_data.tar para cá (scp/rsync/runpodctl send), depois:
tar -xf graphix_study_data.tar && sha256sum --quiet -c pod_data.sha256 && echo dados OK

scripts/run_t5base_study_pod.sh --smoke     # ~30-40 min, 4 células com dados mínimos
# se o smoke passar: confira o pico de VRAM em /workspace/runs/study-t5base-*-smoke/*/…results.json
rm -rf /workspace/study_code /workspace/study_run /workspace/optuna_studies/*_smoke.db
scripts/run_t5base_study_pod.sh             # estudo completo
tail -f /workspace/optuna_studies/t5base_study.jsonl
```

Se o pod reiniciar, rode `scripts/run_t5base_study_pod.sh` de novo. Etapas terminadas são puladas, trials interrompidos viram FAIL e a busca continua com seed nova.

## 6. Ao terminar

Traga de volta `optuna_studies/` e `runs/`. Os pesos (`pytorch_model.bin`) dos runs finais são artefatos do estudo e precisam ser guardados. Os dos trials já são apagados automaticamente.
