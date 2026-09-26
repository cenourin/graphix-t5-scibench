# BIRD: procedência dos dados

Os dados ficam em `data_all_in/data/bird/`, que o git ignora. Baixados em 2026-09-26.

| Arquivo local | Origem | Conteúdo | sha256 |
|---|---|---|---|
| `bird_dev_20251106_package.zip` | Google Drive `13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG` (link "BIRD Dev Complete Package" do README de `birdsql/bird_sql_dev_20251106`) | Pacote do **mini-dev**. Dele só usamos os 11 bancos SQLite do dev (`dev/dev_databases/`, 1,4 GB) e o `dev_tables.json` | `aeb211c0e39010bbdae3838bb5e8bd27dc446ed77495b1709f85ccc9bf67f2be` |
| `dev/dev_20251106.json` | HF `birdsql/bird_sql_dev_20251106`, `data/dev_20251106-00000-of-00001.json` | 1534 perguntas do dev, **revisão oficial de 2025-11-06** (perguntas, `evidence` e SQL corrigidos) | `ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb` |
| `train/train_filtered.jsonl` | HF `birdsql/bird23-train-filtered`, `data/train-00000-of-00001.jsonl` | 6601 perguntas de treino (**versão filtrada oficial**, das 9428 originais), 69 bancos | `2ee64aa593e1adc59be7d5b9ce2395372a8ee27fdd94a3fb4d2fe09f3cd34bc8` |

**Pendente:** os bancos SQLite do train (~33 GB) só estão em `https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip`, que esta máquina não alcança (conexão TCP recusada na porta 443; o DNS resolve). O mesmo vale para o `dev.zip` original, que traz o dev da versão de 2023.

**Ressalva de comparabilidade:** números no dev de 2025-11-06 não são diretamente comparáveis aos da literatura, que usa o dev original de 2023. Dentro deste estudo todos os modelos usam a mesma versão, então a comparação interna é válida.
