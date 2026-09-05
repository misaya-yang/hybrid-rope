# Log-p2 unit-gain Q/K-LoRA canonical replay preflight

- **Date:** 2026-09-03 project time (2026-09-04 on the work machine).
- **Status:** `EXECUTED / RESULT OWNED ELSEWHERE`. See
  [`LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md`](../../results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md).
- **Question:** Does the existing 96-step rank-8 Q/K adapter trained with the
  exact log-p2 table at unit gain improve unseen natural-text likelihood or
  generated-task behaviour over the same frozen table and unit gain?
- **Primary estimand:** adapter minus frozen-weight outcome on identical rows,
  with the sign reported separately at 1x and 4x.
- **Training action:** none; this is a provenance repair and evaluation of an
  already completed private adapter.

## 1. Why this replay exists

The current single-static owner specifies a future log-p2 plus `c=.074`
same-substrate LoRA. A private 2026-09-02 adapter initially appeared to satisfy
that description, but source recovery shows that it used the same table with
`attention_scaling=1.0`. It therefore cannot be promoted as execution of the
current specification.

It does supply a narrower matched experiment: table and unit gain are identical
between frozen and adapted systems, and only all-layer Q/K rank-8 residuals
change. Canonical natural replay can determine whether the specialized
positional training signal transfers beyond its training view.

## 2. Recovered provenance

- Training script SHA-256:
  `1e5409c3a54ecfe6c5fb645c4ad73de1a5c85ad369cc451b129aca54500c93f9`.
- Training receipt SHA-256:
  `2ced79686df85cb5fdea2cf949239b1fea90f9fa2ca15898a218dfcd6791c166`.
- Adapter safetensors SHA-256:
  `48f54cf5b6e761dd81e03e8f2516da69cc9ee95ea9252bf529dc1044911cacd9`.
- Adapter config SHA-256:
  `261f152fcfa788381fc8bda91af08ab90b61f6d2f18eefb5a5d024c95aab8436`.
- Base checkpoint weight/config SHA-256:
  `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f` /
  `0d15ebb6cb8d998513b46ef337214176a6fd59fe5f16b30387c70d5f87795a9c`.
- Installed table float32/file SHA-256:
  `56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b` /
  `ee968fcefe9f91a6bec7bff6eae45585d2876f03625378a739031a3155b59316`.
- Trainable scope: all-layer `q_proj` and `k_proj`, rank `8`, alpha `16`, no
  dropout or bias, `1,048,576` trainable parameters.
- Training: 96 steps, seed `20260902`, learning rate `5e-5`, `LLLSS` schedule,
  correct plus answer-span-deranged variants, table fixed, unit gain fixed.
- Short/long training manifest SHA-256:
  `11433edaa494966ec1c08fc19f29ac48ccfe07a461928137d133c685d0f77849` /
  `375f4066058ef5ceddefeacf0eeddf236306d51939d0d95fea5b50be5943dec8`.

The recovered script was private and is not a tracked repository artifact.
Its hash match plus receipt/raw-log/adapter readback establishes this executed
artifact's identity; it does not make the training protocol a manuscript owner.

## 3. Frozen replay protocol

Both arms use `legacy_u_p2_log_s4`, `attention_scaling=1.0`, one static table at
every length, and the same current evaluator:

1. frozen base weights;
2. the recovered rank-8 Q/K adapter, merged for inference.

Stage A evaluates the 20 fixed PG-19 documents at 1x and 4x. Stage B opens only
after finite matched Stage-A results and evaluates Qasper, MultiFieldQA-en,
HotpotQA, 2WikiMQA, and GovReport on their identical fixed 1x/4x rows. Report
task cells and paired rows; do not pool likelihood and generation.

The natural token manifest SHA-256 is
`74022bf36d444a1735baab72bda0312b9867dd38c9f85ece376049b5f35f66f3`.
The canonical evaluator SHA-256 is
`cfc3d7c2893e95cd88a160e728fae76d17e7e1f8be9ae2c9edcca6b28f9d2871`.

## 4. Outcome interpretation

| Outcome | Interpretation |
| --- | --- |
| 4x improves without material 1x loss | supports transfer of this unit-gain Q/K adaptation recipe; does not validate `c=.074` |
| positional training endpoints improve but natural replay does not | specialized objective mismatch; do not continue the recipe |
| 1x improves but 4x degrades | ordinary local adaptation, not long-context repair |
| both natural endpoints degrade or remain unresolved | closes this exact adapter/protocol as a useful continuation |
| identity, merge, row, or control failure | `UNRESOLVED`; no parameter rescue |

No outcome tests the scale-orbit theorem, proves Q/K sufficiency, selects rank
or budget, or licenses transplanting the adapter to another table/gain.
