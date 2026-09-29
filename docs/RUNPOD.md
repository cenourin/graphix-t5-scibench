# Rodar o estudo t5-base num pod do RunPod

Alvo: **1× RTX 4090 24 GB**, RAM de 31 a 46 GB, *network volume* de 100 GB em `/workspace` e disco raiz efêmero.

**Regra do protocolo:** as 4 células rodam num só lugar (`PROTOCOLO.md` §2.2). Se o estudo for para o pod, isso entra em "Desvios" **antes** do lançamento: hardware GTX 1070 → RTX 4090, TF32 desligado. Nenhuma célula fica pela metade em cada máquina.

## 1. Imagem: use a do projeto, não o template PyTorch 2.8

O código é Graphix-T5 com ambiente antigo e fixo. Tudo vem da imagem `eyuansu62/graphix-text-to-sql:v2`, que tem 30,9 GB:

| Item | Versão |
|---|---|
| Ubuntu | 18.04 |
| Python | 3.7.10 |
| torch | 1.9.0 + CUDA 11.1 |
| DGL | 0.8.2 |
| transformers | 4.17.0 |
| tokenizers | 0.11.6 |
| datasets | **1.18.4.dev0** |
| numpy | 1.21.4 |
| scipy | 1.6.2 |
| networkx | 2.6.3 |
| sentencepiece | 0.1.96 |
| stanza | 1.1.1 |

**Não migre para o template RunPod PyTorch 2.8.**
- O modelo é um fork do `modeling_t5` do transformers 4.17.
- O DGL não tem wheels para torch 2.8.
- O `datasets` da imagem é uma versão de desenvolvimento, que não se instala por pip.
- Portar mudaria a numérica e invalidaria a comparação com tudo o que já foi medido.

A imagem **é** o arquivo de dependências travado deste projeto. Um `requirements.txt` feito à mão seria menos fiel.

**Compatibilidade com a 4090 (compute capability 8.9).**
- O torch da imagem traz kernels até sm_86, e o DGL até sm_80.
- Pela regra de compatibilidade binária da CUDA, um kernel compilado para 8.x roda em qualquer 8.y com y ≥ x. **Deve funcionar, e o smoke confirma.**
- GPUs 9.x (H100) ou mais novas não rodam, e o lançador recusa.
- O driver do host (CUDA 13.2) é retrocompatível com o runtime 11.1 da imagem. A versão que o `nvidia-smi` mostra é a do driver, não a da imagem.

**TF32.** O torch 1.9 liga TF32 por padrão em GPUs Ampere e mais novas. O `run_seq2seq_train.py` desliga, para manter o fp32 do protocolo. Isso não muda nada na 1070. Religar exige `GRAPHIX_ALLOW_TF32=1` e um registro de desvio.

## 2. Criar o pod

| Campo | Valor |
|---|---|
| Container image | `eyuansu62/graphix-text-to-sql:v2` |
| Container disk | ≥ 50 GB. A imagem tem 31 GB, e o valor exato que o RunPod exige para imagens grandes não foi verificado; na dúvida, sobre. |
| Volume | o network volume de 100 GB, montado em `/workspace` |
| Expose TCP ports | `22` |
| Docker command | `bash -c 'mkdir -p ~/.ssh && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && service ssh start && sleep infinity'` |

A imagem já tem `sshd`. O RunPod injeta a sua chave pública em `$PUBLIC_KEY`. A imagem não tem Jupyter nem terminal web; o acesso é por SSH.

## 3. Espaço no volume de 100 GB

| Conteúdo | Tamanho |
|---|---|
| Dados (`scripts/pack_for_pod.sh`) | 31 GB |
| Repositório | < 0,2 GB |
| Estudos Optuna + logs | < 0,1 GB |
| Checkpoints dos trials (2 por vez; os pesos são apagados ao fim de cada trial) | ~2 GB, transitório |
| 4 runs finais (modelo + 2 checkpoints cada) | ~12 GB |
| **Total** | **~46 GB** |

Os dados detalhados:

| Dado | Tamanho |
|---|---|
| Bancos SQLite do ScienceBenchmark (o `skyserver` sozinho tem 15 GB) | 16 GB |
| Saída pré-processada do ScienceBenchmark | 9,8 GB |
| Saída pré-processada do Spider | 3,8 GB |
| `t5-base` original | 0,85 GB |
| Bancos do Spider | 0,84 GB |
| Splits | 79 MB |

**Transfira em streaming.** Um tar de 31 GB mais os dados extraídos ocupariam 62 GB. Enviar 31 GB leva ~1,4 h a 50 Mbit/s de upload, e ~7 h a 10 Mbit/s.

Não há cache do Hugging Face a configurar: o modelo e os dados vêm de caminhos locais, e nada é baixado da Hub.

## 4. Passo a passo

Nesta máquina:

```bash
ssh -p <PORTA> root@<IP_DO_POD> 'cd /workspace && git clone https://github.com/cenourin/graphix-t5-scibench.git && cd graphix-t5-scibench && git checkout trained-models'
# repositório privado: use um token de acesso só de leitura na URL, ou uma deploy key; não deixe credencial gravada no volume
scripts/pack_for_pod.sh - | ssh -p <PORTA> root@<IP_DO_POD> 'cd /workspace/graphix-t5-scibench && tar -xf -'
```

No pod:

```bash
cd /workspace/graphix-t5-scibench
sha256sum --quiet -c pod_data.sha256 && echo dados OK
nvidia-smi && python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

scripts/run_t5base_study_pod.sh --smoke          # ~30-60 min: as 4 células com dados mínimos
tail -f /workspace/optuna_studies/t5base_study.jsonl
# conferir: 4x dev_end rc=0; pico de VRAM em /workspace/runs/study-t5base-*-smoke/run/train_results.json
# a velocidade real por época NÃO sai do smoke (32 exemplos); medir no 1º trial de verdade

rm -rf /workspace/study_code /workspace/study_run /workspace/optuna_studies/*_smoke.db \
       /workspace/optuna_studies/smoke_*.json /workspace/optuna_studies/t5base_study.jsonl \
       /workspace/runs/optuna/*_smoke* /workspace/runs/study-t5base-*-smoke
# registrar o desvio de hardware no PROTOCOLO.md, commitar, e só então:
scripts/run_t5base_study_pod.sh
```

Monitoramento no pod: `tail -f /workspace/optuna_studies/t5base_study.jsonl`, `nvidia-smi dmon -s mu` (memória e uso por segundo) e `tail -f /workspace/runs/optuna/*/trial_*/train.log`.

## 5. Retomada: o que sobrevive a parar o pod

Tudo o que importa fica em `/workspace`, e o lançador é idempotente. Depois de reiniciar, basta rodar `scripts/run_t5base_study_pod.sh` de novo.

| Etapa | O que acontece ao retomar |
|---|---|
| Trial Optuna interrompido | Vira FAIL, e a busca continua com seed nova (1 + nº de trials), para não repetir configurações. O trial perdido custa no máximo ~3 épocas. |
| Treino final interrompido | **Retomada completa** a partir do último checkpoint por época (`overwrite_output_dir: false`). O checkpoint do HF guarda pesos, estado do Adafactor, scheduler, `trainer_state` (época e passo) e o estado dos RNGs. Não usa AMP, então não há scaler. |
| Avaliação no dev interrompida | Refeita do zero (~minutos). |
| Célula terminada | Pulada. |

A retomada no meio de uma época repete a ordem dos dados via RNG salvo, mas não é garantida como idêntica bit a bit. Se acontecer, isso vai para "Desvios".

## 6. Rastreabilidade de cada execução

- **`pipeline_start` em `t5base_study.jsonl`:** commit do código (com a marca `+uncommitted` se houver diff), versões de Python, torch, CUDA, cuDNN, transformers e DGL, GPU e capability, VRAM, driver e o estado do TF32.
- **Cada run:** `combined_args.json` (todos os hiperparâmetros), `trainer_state.json` (curva de loss), `train_results.json` e `eval_results.json` (tempo, pico de VRAM alocada e reservada, GPU).
- **Código:** `study_code/COMMIT` + `uncommitted.diff` (o snapshot congelado).

## 7. Ao terminar

1. Trazer de volta `/workspace/optuna_studies/` e `/workspace/runs/`, por exemplo com `rsync -a` via SSH. Os pesos dos 4 runs finais são artefatos do estudo.
2. A análise (`scripts/score_predictions.py` + `scripts/analyze_study.py`) pode rodar aqui ou no pod, desde que as 4 células sejam pontuadas nas mesmas condições e com a máquina ociosa (`PROTOCOLO.md`, "Desvios").
3. Encerrar o pod. O volume continua sendo cobrado até ser apagado.

## 8. Checklist

- [ ] Pod criado com a imagem do projeto (não o template PyTorch 2.8), com container disk ≥ 50 GB, volume em `/workspace` e porta 22 exposta.
- [ ] `nvidia-smi` mostra a 4090, e `torch.cuda.is_available()` dá `True` dentro do pod.
- [ ] Repositório em `/workspace/graphix-t5-scibench`, branch `trained-models`, commit anotado.
- [ ] Dados transferidos e `sha256sum -c pod_data.sha256` passou.
- [ ] Smoke passou nas 4 células, com o pico de VRAM anotado.
- [ ] Artefatos do smoke apagados.
- [ ] Desvio de hardware registrado e commitado no `PROTOCOLO.md`.
- [ ] Estudo lançado. O `pipeline_start` mostra a GPU certa e `allow_tf32: false`.
- [ ] Ao final: resultados copiados e pod encerrado.
