# Graphix-3B — setup completo

## Backbone (T5-3B vanilla)

Config real do checkpoint, em `train_db_id/Graphix-3B/config.json`:

| Parâmetro | Valor |
|---|---|
| `d_model` (hidden size) | 1024 |
| `d_ff` (FFN interno) | 16384 |
| `d_kv` (dim por head) | 128 |
| `num_heads` | 32 |
| `num_layers` (encoder) | 24 |
| `num_decoder_layers` | 24 |
| `feed_forward_proj` | relu |
| `vocab_size` | 32128 |
| `dropout_rate` | 0.1 |
| `layer_norm_epsilon` | 1e-6 |
| Checkpoint em disco | 14.23 GB (`pytorch_model.bin`, salvo em fp32 originalmente) |
| Parâmetros totais | ~2.85B |

## Camada RGAT injetada

`T5LayerRGAT` (`modeling_t5.py:369`), uma por bloco do **encoder**, todos os 24:

```python
RGAT_Layer(config.d_model, config.d_model, num_heads=1, feat_drop=0.2)
```

`ndim=edim=1024` (mesmo `d_model` do T5), **1 head só** (não usa os 32 heads do
T5), dropout de 0.2 nas features. Só entra no encoder — o decoder do T5
permanece intocado, sem consciência do grafo.

## Batch / beam / geração

Dois perfis: o que rodamos (smoke test, validado) vs. o alvo original do paper.

| | `eval_smoke.json` (rodado, validado) | `eval.json` (original/alvo) |
|---|---|---|
| `per_device_eval_batch_size` | 1 | 4 |
| `num_beams` | 1 | 4 |
| `max_val_samples` | 1 | 1034 (dev inteiro) |
| `val_max_target_length` | 128 | 512 |
| `val_max_time` | 180s | 2400s |
| `use_picard`/`launch_picard` | false | true |
| `max_source_length` (preprocessing) | 1024 | 1024 |

## Precisão e memória — adaptações pra caber em 8GB de VRAM

Nada disso é o setup "padrão" do paper — foi construído nesta sessão pra rodar
o Graphix-3B numa GTX 1070 de 8GB:

- Checkpoint carregado em **bf16** (não fp16, não fp32) — downcast por-tensor
  durante `torch.load()`, streaming, pra não duplicar o pico de memória.
- **Offloading CPU↔GPU camada-a-camada**: cada `T5Block` do encoder/decoder
  migra pra GPU só durante seu próprio forward, com prefetch assíncrono
  (`torch.cuda.Stream`) do próximo bloco enquanto o atual computa. Só ~2
  blocos residem na GPU por vez, dos 48 total (24 encoder + 24 decoder).
- Upcast local pra fp32 em 3 pontos específicos (FFN, os dois matmuls batched
  da atenção, `lm_head`) — não é o comportamento padrão do HF T5, foi
  adicionado por causa do overflow documentado em
  `fp16-to-bf16-overflow.md`.
- Ops do DGL (RGAT) sempre em fp32, independente do resto — DGL não suporta
  16-bit nenhum.

## Treino vs. avaliação

Tudo acima é especificamente pro **eval/inferência** — não há treino do
Graphix-3B configurado neste repo (o checkpoint já vem pronto, baixado do
Hugging Face). `configs/train.json` é o perfil de treino deste repo, e esse
aponta pro **T5-large** (`t5-large`), não pro 3B.
