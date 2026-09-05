# Log-p2 plus c=.074 Q/K-LoRA preflight

- **Date:** 2026-09-04.
- **Status:** `EXECUTED / RESULT OWNED ELSEWHERE`. See
  [`LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md`](../../results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md).
- **Question:** On the retained one-table log-p2 substrate, does 96-step
  all-layer Q/K rank-8 adaptation improve natural 4x behaviour over frozen
  weights without a material 1x cost?
- **Primary estimand:** adapted minus frozen outcome with the exact table and
  fixed `attention_scaling=1+0.074 log(4)` in both arms.
- **Evidence boundary:** prospective experiment. It cannot identify p2 or
  `.074` as theory-derived, prove Q/K sufficiency, or validate scale-orbit
  quantities.

## Why this run is live

The recovered unit-gain Q/K adapter supplies a positive transfer screen on 20
paired PG-19 documents: adapter-favouring NLL deltas are `0.01734` at 1x and
`0.10622` at 4x, with paired bootstrap intervals excluding zero. Its five-task
generation macro is heterogeneous (`-0.01723` at 1x and `+0.01023` at 4x; both
intervals cross zero). This justifies testing the exact retained gain, but does
not permit gain transplantation to count as same-substrate adaptation.

## Frozen training contract

- released OLMo-2-0425-1B-Instruct weight/config SHA-256:
  `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f` /
  `0d15ebb6cb8d998513b46ef337214176a6fd59fe5f16b30387c70d5f87795a9c`;
- legacy-u p2 log-s4 float32/file SHA-256:
  `56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b` /
  `ee968fcefe9f91a6bec7bff6eae45585d2876f03625378a739031a3155b59316`;
- fixed gain coefficient/scalar: `.074` / `1.102585782722872`;
- all-layer `q_proj,k_proj` only; rank `8`, alpha `16`, dropout/bias `0`;
- 96 optimizer steps, seed `20260902`, AdamW, learning rate `5e-5`, no weight
  decay, ten-step warmup then cosine decay;
- repeating `LLLSS` length schedule; each step accumulates the paired correct
  and answer-span-deranged views before one optimizer update;
- short/long data manifest SHA-256:
  `11433edaa494966ec1c08fc19f29ac48ccfe07a461928137d133c685d0f77849` /
  `375f4066058ef5ceddefeacf0eeddf236306d51939d0d95fea5b50be5943dec8`;
- training script: `scripts/train/train_log_p2_c074_qk_lora.py`, SHA-256
  `2a2ee346c465ad1f451be0baee751e507134919f1edcf570af3d17c9e8553e2a`.

Stop training at 96 steps or immediately on identity drift, non-finite loss or
gradient, OOM, escaped trainable scope, missing BF16/Flash execution, or author
stop. There is one seed because this is a candidate screen, not a variance
claim. No rank, alpha, learning-rate, gain, or budget sweep is authorized.

## Frozen evaluation and decision

Stage A uses the current formal evaluator and the same 20 PG-19 rows at 1x/4x.
The frozen c=.074 arm may be reused only after exact row, table, gain,
checkpoint, manifest, evaluator, and decoder identities match. Report paired
NLL deltas and intervals.

If 4x PG-19 improves without a material 1x loss, run the same five natural
generation tasks as the unit-gain replay at 1x/4x. Otherwise stop this exact
candidate. Preserve raw rows, receipts, hashes, failures, and partial outputs
outside the repository. A positive screen promotes only this fixed
table/gain/adapter/protocol; a negative screen closes only this candidate.

Stage C opens only if Stage A improves but Stage B is mixed or negative. It
evaluates the fresh core-4 RULER panel at 4K/8K/16K, 20 rows per task, to
distinguish likelihood-only adaptation from improvement on the structured
positional capability that motivated the training views. The data manifest
SHA-256 is
`0e184255f006c212c9fc1db08860800cc3018c9de434e69a0d1d6207665fdb31`.
Both frozen and adapter arms use the adapter-capable evaluator
`scripts/eval/target_free_ruler_smoke.py`, SHA-256
`06811fcac3e5be7967c228ab24f3738eb32c42e0bf174d63e6c946f2b0959c4f`;
the baseline is rerun because earlier receipts predate this exact evaluator.
No full-13 expansion opens from Stage C.
