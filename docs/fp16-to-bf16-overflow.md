# De overflow em fp16 a bf16: a cadeia causal completa

Contexto: durante o smoke test do Graphix-3B (1 exemplo do dev set do
Spider, batch=1, beam=1, em GPU de 8GB), a avaliação retornava predição
vazia e `eval_loss=NaN`. O diagnóstico (hooks por bloco/sublayer no
encoder) localizou a origem em `T5LayerFF` do bloco 19: sua saída virava
`inf`, e a camada RGAT seguinte convertia isso em `nan`, contaminando os
blocos 20–23 e tudo a jusante. Este documento descreve a cadeia causal
completa até a correção final.

## (1) Por que a magnitude das ativações cresce monotonicamente entre blocos do encoder

T5 usa **Pre-LN** (pré-normalização): cada sublayer (self-attention, FFN)
aplica LayerNorm na *entrada* antes da transformação, mas soma a saída
*não normalizada* de volta no residual stream original —
`x = x + f(LN(x))`. Isso é estruturalmente diferente do Post-LN clássico
(Transformer original), onde a normalização acontece *depois* da soma
residual e reseta a escala a cada bloco.

Em Pre-LN, nada re-normaliza o residual stream em si — só a cópia que
entra em cada sublayer é normalizada. O resultado é que o residual
acumula, de forma aditiva e sem teto, a saída bruta de toda sublayer de
todo bloco anterior. Não existe mecanismo de clipping ou rescaling entre
blocos. Esse crescimento (empiricamente próximo de `~sqrt(profundidade)`
ou pior, dependendo da inicialização) é uma característica **conhecida e
documentada** de transformers Pre-LN profundos — é inclusive parte do
motivo pelo qual o T5 original do Google foi treinado em bfloat16 em TPU,
e por que a comunidade HuggingFace já alertava sobre instabilidade de T5
em fp16 puro.

`T5LayerNorm` é RMSNorm (sem subtração de média, sem bias) — normaliza a
variância da entrada da sublayer, mas isso não impede o residual em si de
crescer. Com 24 blocos no encoder do T5-3B, medimos magnitude crescendo
até **~272384** no bloco 23 (o último) — ou seja, o problema é cumulativo,
não localizado num bloco "quebrado".

## (2) Por que o upcast local para fp32 dentro do FFN não bastou sozinho

O padrão tentado primeiro foi: pegar a entrada fp16 → cast para fp32 →
computar `wi` / ativação / `wo` em fp32 (precisão total, sem erro de
arredondamento) → cast do resultado de volta para fp16 antes de devolver
ao resto do pipeline.

O problema é que isso resolve um problema de **precisão** (erro de
arredondamento durante o cálculo), não de **faixa dinâmica** (o valor
final não cabe no formato de destino). O valor verdadeiro calculado em
fp32 no bloco 19 já excede **65504** — o maior valor finito representável
em fp16 (`(2 − 2⁻¹⁰) × 2¹⁵`). Não importa quão exato foi o cálculo
intermediário: no downcast final para fp16, o IEEE 754 não tem como
representar um número maior que o teto do formato — ele vira `inf`
(overflow), por definição de arredondamento de overflow, independentemente
da precisão usada para chegar lá. Por isso "computar em fp32 e guardar em
fp16" é uma correção de *rounding error*, inútil contra *magnitude
overflow*.

## (3) Por que bf16 resolve a faixa mas não a precisão

fp16 e bf16 ocupam os mesmos 16 bits, mas dividem esse orçamento de forma
diferente:

| Formato | Sinal | Expoente | Mantissa | Faixa (~max) | Precisão relativa |
|---|---|---|---|---|---|
| fp16 | 1 | 5 bits | 10 bits | 65504 | ~2⁻¹⁰ (~3–4 dígitos) |
| bf16 | 1 | 8 bits | 7 bits | ~3.4×10³⁸ | ~2⁻⁷ (~2–3 dígitos) |
| fp32 | 1 | 8 bits | 23 bits | ~3.4×10³⁸ | ~2⁻²³ |

bf16 tem o **mesmo campo de expoente do fp32** — mesma faixa dinâmica,
então 272384 cabe nele sem problema algum (nem chega perto do teto). O
preço é que ele sacrifica 3 bits de mantissa a mais que o fp16, então cada
valor armazenado tem mais ruído de quantização relativo. Ou seja: bf16
não é "melhor" que fp16 de forma absoluta — ele **troca precisão por
faixa**. Isso transforma o modo de falha de "overflow catastrófico → inf
→ nan → contamina tudo a jusante, irrecuperável" para "ruído de
arredondamento comum, do mesmo tipo que qualquer treino/inferência em
precisão mista já tolera", que normalizações e residuais absorvem
normalmente.

## (4) Pontos que precisaram de upcast manual vs. os que já vinham protegidos

**Já protegidos nativamente no T5 do HuggingFace** (confirmado durante o
diagnóstico, não modificados):

- **`T5LayerNorm.forward`** — RMSNorm já faz upcast para fp32 internamente
  antes de calcular a variância (`hidden_states.to(torch.float32)` →
  `variance = hidden_states.pow(2).mean(...)` → `rsqrt`), porque **elevar
  ao quadrado** um valor fp16 já moderadamente grande estoura quase
  instantaneamente.
- **Softmax dentro de `T5Attention`** — HF já calcula softmax em fp32
  (`dtype=torch.float32` seguido de `.type_as(...)`), porque
  **exponenciação** (`exp()`) de logits mesmo moderados também estoura
  fácil em fp16/bf16.

Ambos os casos envolvem operações não-lineares (quadrado, exponencial)
que amplificam a faixa dinâmica muito mais rápido que uma multiplicação
de matriz simples — por isso os autores da biblioteca defensivamente já
protegeram exatamente esses dois pontos, e só esses, como prática padrão.

**Precisaram de upcast manual** (adicionado no fix) — porque uma
`nn.Linear`/matmul comum não eleva ao quadrado nem exponencia, então não
recebe essa proteção defensiva por padrão, mesmo que o **residual stream
acumulado** que passa por ela já esteja fora de faixa:

- **`T5DenseReluDense`/`T5DenseGatedGeluDense.forward`** (o FFN
  `wi` → ativação → `wo`) — ponto de origem do bug, bloco 19.
- **`T5Attention.forward`**, especificamente os dois matmuls **batched**
  (4D): `QK^T` e `attn_weights @ V`. Esse upcast teve uma segunda
  motivação, independente da faixa: nesta GPU (Pascal, compute capability
  6.1, sem tensor cores), o cuBLAS não suporta GEMM batched em bf16
  (`CUBLAS_STATUS_NOT_SUPPORTED`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP` exige
  tensor cores) — só matmul 2D simples usa um caminho de cuBLAS que não
  depende de tensor cores. Por isso foram usadas variáveis com sufixo
  `_mm` separadas, para o KV-cache (`present_key_value_state`) continuar
  em bf16 e não vazar para fp32 silenciosamente entre passos de geração.
- **Projeção final `lm_head`** — mesmo padrão de upcast-local-e-downcast
  do FFN, protegendo a projeção para o vocabulário.

## Por que isso não é um bug de implementação

O overflow não veio de nenhum erro de código — vem de uma propriedade
estrutural conhecida de transformers Pre-LN sem renormalização do
residual entre blocos, combinada com a decisão (do Google, no T5
original) de treinar em bfloat16, faixa suficiente para esse crescimento,
nunca fp16. Rodar esse checkpoint em fp16 é operar fora do regime de
precisão em que ele foi treinado e validado — o comportamento correto e
esperado é exatamente esse: a magnitude do residual eventualmente excede
o teto do fp16, em algum bloco previsível pela profundidade do modelo, e
a correção certa é usar o formato de faixa compatível (bf16), não
"consertar" a matemática do FFN.
