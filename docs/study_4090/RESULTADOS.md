# Estudo t5-base: resultados (análise pré-registrada, PROTOCOLO.md §7)

## 1. EM e EX por célula (dev oficial, IC 95% bootstrap)

| Célula | n | EM % [IC] | EX % [IC] |
|---|---|---|---|
| spider / rgat | 1034 | 51.0 [47.9, 54.1] (n=1034) | 52.1 [49.2, 55.2] (n=1034) |
| spider / plain | 1034 | 53.1 [50.0, 56.1] (n=1034) | 54.0 [51.0, 57.0] (n=1034) |
| sciencebenchmark / rgat | 293 | 12.8 [9.0, 16.6] (n=290) | 14.9 [10.8, 19.1] (n=288) |
| sciencebenchmark / plain | 293 | 14.8 [10.7, 19.0] (n=290) | 17.4 [13.2, 21.9] (n=288) |

## 2. Q1: rgat vs plain (McNemar exato pareado, Holm sobre 4 testes)

| Benchmark | Métrica | n pareado | só rgat acerta | só plain acerta | Δ rgat−plain (p.p.) [IC] | p | p Holm |
|---|---|---|---|---|---|---|---|
| spider | EM | 1034 | 57 | 79 | -2.1 [-4.3, +0.1] | 0.0714 | 0.285 |
| spider | EX | 1034 | 68 | 87 | -1.8 [-4.3, +0.5] | 0.148 | 0.444 |
| sciencebenchmark | EM | 290 | 7 | 13 | -2.1 [-5.2, +1.0] | 0.263 | 0.444 |
| sciencebenchmark | EX | 288 | 6 | 13 | -2.4 [-5.6, +0.3] | 0.167 | 0.444 |

H0 (diferença zero) é rejeitada só onde p Holm < 0,05. Holm é aplicado apenas sobre os testes com resultado; com células faltando, a correção fica incompleta.

## 3. ScienceBenchmark sem os exemplos cujo SQL aparece no train

Exemplos excluídos: 5 (oncomx_v1_0_25_small).

| Célula | EM % | EX % |
|---|---|---|
| rgat | 11.9 | 14.1 |
| plain | 13.7 | 16.3 |

## 4. ScienceBenchmark por banco (descritivo)

| Célula | Banco | n | EM % | EX % |
|---|---|---|---|---|
| rgat | cordis_temporary | 100 | 14.0 | 14.0 |
| rgat | oncomx_v1_0_25_small | 99 | 18.2 | 24.7 |
| rgat | skyserver_dr16_2020_11_30 | 94 | 5.5 | 5.5 |
| plain | cordis_temporary | 100 | 12.0 | 13.0 |
| plain | oncomx_v1_0_25_small | 99 | 26.3 | 33.0 |
| plain | skyserver_dr16_2020_11_30 | 94 | 5.5 | 5.5 |
