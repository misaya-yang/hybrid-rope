# OLMo-2 1.485B Fresh General-Data Q/K Adaptation

## Status

`RUNNING / NOT YET REVIEWER-USABLE`

The matched training arms and contamination gate are complete. The registered
autoregressive 2WikiMultiHopQA, complete 13-family RULER, official YaRN, and
repository fixed-ramp matrix is still running. This owner must not be promoted
until every registered cell, raw-generation hash, and final summary is present.

Relevant retained concerns: `RDz6s.1`, `RDz6s.2`, `RzWsa.3`, `RzWsa.4`,
`R27bE.2`, `R27bE.5`, `AC.2`, and `AC.4`.

## Question and direct answer

This experiment asks whether EVQ can be adapted from the untouched mature
OLMo-2 Instruct checkpoint using only fresh Q/K LoRA and generic data, without
inheriting a Q/K/V/O adapter or training on 2Wiki/RULER task-family rows. It
also asks whether official YaRN or the repository fixed-index smooth-ramp
transform changes the Native-versus-EVQ result.

The downstream answer remains pending. The completed teacher-forced endpoint
already shows a clear tradeoff: Native has lower 4K natural-text NLL, while EVQ
has lower 8K and 16K NLL. This is probability-modeling evidence only and is not
treated as QA, retrieval, or RULER capability.

## Registered matched protocol

| Field | Native arm | EVQ arm |
| --- | --- | --- |
| Base checkpoint | untouched OLMo-2-0425-1B-Instruct | same |
| Actual model size | 1.485B parameters | same |
| Adaptation | fresh Q/K-only LoRA | same |
| LoRA rank / alpha | 64 / 128 | same |
| Trainable parameters | 8,388,608 in 64 tensors | same |
| Trainable projections | all-layer `q_proj` and `k_proj`; no V/O/readout | same |
| Data schedule | LongAlign full-token, LongAlign full-token, Tulu assistant-only | same |
| Steps / global batch | 600 / 8 | same |
| LR / warmup | `1e-4` / 30 steps | same |
| Optimizer | fused AdamW, betas 0.9/0.95, no weight decay | same |
| Physical train length | at most 4,096 tokens | same |
| Position IDs | ordinary contiguous positions only | same |
| Seed | 20260727 | same |
| Active training frequency | Native RoPE | EVQ-Cosh |

The only registered method variable is the frequency table active during
otherwise matched fresh Q/K-only adaptation. The learned LoRA tensors
subsequently diverge as a consequence of that intervention, not as an
additional experimental variable.

Matched receipts:

- initial adapter tensor SHA-256:
  `b1280c2001a3bd1bf588f10838969a5d2225a8d042d466176d865369740280c0`;
- row-selection SHA-256:
  `23e41570f599055fe8d678473a4a3f7da32e038f793d3af2d6e4e1f4a5c2f2cd`;
- Native adapter SHA-256:
  `3b8735c9b0cd6b65238f256ce0d9395c6e835be9ba7583fca485bc1e675c4740`;
- EVQ adapter SHA-256:
  `26157b7b586449bcc1a3ad5ff574ae4428640a047bc3fdad90a40e71d2653c79`.

The schedule executes 400 LongAlign full-token steps and 200 Tulu
assistant-only steps. Both arms consume 19,656,000 dense tokens and
13,729,836 supervised tokens under the same selected rows and order.

## Data and overlap boundary

Training uses prepared views of:

- `zai-org/LongAlign-10k`, revision
  `12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc`;
- `allenai/tulu-3-sft-olmo-2-mixture-0225`, revision
  `d91a0785ade02942520280fb484866fce41e448f`.

The registered exact-token audit found:

- `0/200` exact 2Wiki test-question occurrences in the prepared training
  views;
- `0/780` exact raw or chat-rendered RULER test-prompt occurrences in the
  prepared training views.

This supports “no direct benchmark-family supervision” and
“benchmark-unsupervised evaluation after generic adaptation.” It does not
prove document-, topic-, or upstream-corpus-level non-contamination and
therefore is not labeled clean unseen-task transfer.

## Natural-text NLL/PPL

Single-seed evaluation on 16 held-out natural-text rows per length:

| Length | Native NLL | Native PPL | EVQ NLL | EVQ PPL |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 2.2889 | 9.8642 | 2.8803 | 17.8190 |
| 8K | 3.9317 | 50.9912 | 3.0876 | 21.9245 |
| 16K | 5.1079 | 165.3291 | 3.3819 | 29.4257 |

Interpretation: under the matched generic-data Q/K-only protocol, EVQ pays a
material in-window modeling cost but degrades much less at 2× and 4× context.
These numbers do not establish autoregressive task capability.

## Registered capability matrix

The following cells are registered and must all be resolved before promotion:

1. untouched Native base at 4K;
2. Native fresh-QK and EVQ fresh-QK at 4K/8K/16K;
3. Native/EVQ substrates with official Transformers YaRN at factor 2 for
   4K/8K and factor 4 for 4K/16K;
4. Native/EVQ substrates with the repository fixed-index smooth-ramp
   transform under the same factors and lengths.

### 2WikiMultiHopQA

`PENDING_MATRIX`

The endpoint uses 200 examples per cell, greedy autoregressive generation,
normalized token F1, normalized exact match, terminal EOS, and full generated
token IDs. Deterministic answer-filtered distractor filling reaches the
registered physical budgets; this is a task-family long-QA protocol, not an
unaltered LongBench leaderboard run.

### Complete 13-family RULER

`PENDING_MATRIX`

The endpoint uses all 13 families, 20 examples per family-length cell, greedy
autoregressive generation, and the official family-specific string-match
score. RULER is reported separately from NLL/PPL.

### Official YaRN versus repository fixed ramp

`PENDING_MATRIX`

Official YaRN is the Transformers operator including its attention scaling.
The repository control is a fixed-index 20%–90% smoothstep ramp without that
attention scaling. It is never described as official YaRN.

## Claim boundary and send gate

- Evidence tier remains `DESIGN_ONLY_OR_PENDING` until the capability matrix
  and raw hashes close; it can then become `POST_SUB_RAW_HASH_BACKED`.
- The experiment is single-seed.
- Any 4K deficit must sit next to an 8K positive.
- Any weak or negative 16K result must sit next to the 8K result it limits.
- NLL/PPL cannot substitute for QA or RULER.
- “No direct benchmark-family supervision” is not clean unseen-task transfer.
- Fixed-factor official YaRN is not a fully tuned YaRN sweep.
- The comparison isolates the Native-versus-EVQ training frequency table at
  the method level; it does not decompose which internal Cosh coordinates
  cause the outcome.
- This is not a direct FMRoPE comparison and does not support universal SOTA,
  universal no-harm, or replacement of range-scaling methods.

Do not cite this owner while any `PENDING_MATRIX` marker remains.
