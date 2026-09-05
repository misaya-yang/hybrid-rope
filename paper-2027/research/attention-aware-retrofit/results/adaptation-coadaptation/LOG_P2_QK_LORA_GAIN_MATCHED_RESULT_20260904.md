# Log-p2 Q/K LoRA improves likelihood but not measured capability

- **Date:** 2026-09-04.
- **Status:** `COMPLETE / VALID MATCHED REPLAYS / PG-19 POSITIVE / NATURAL
  GENERATION AND RULER NOT PROMOTED`.
- **Evidence labels:** completed training and endpoint values are
  **Observations**; the exact 96-step `c=.074` recipe is a **Negative result**
  for broad capability improvement; mechanistic explanations remain
  **Interpretations**.
- **Question:** Does low-rank Q/K adaptation on the same static log-p2 substrate
  repair the learned weight--table mismatch, first for a recovered unit-gain
  adapter and then for the retained `c=.074` table/gain?
- **Preflights:**
  [`unit-gain replay`](../../preflights/adaptation-coadaptation/LOG_P2_UNIT_GAIN_QK_LORA_REPLAY_PREFLIGHT_20260903.md)
  and
  [`c=.074 training/evaluation`](../../preflights/adaptation-coadaptation/LOG_P2_C074_QK_LORA_PREFLIGHT_20260904.md).

## 1. Decision

Q/K-only adaptation is sufficient to lower natural PG-19 NLL on both tested
gain substrates, but it is not sufficient to improve the measured generated
capability. The exact current-method `c=.074` adapter improves paired PG-19 at
1x and 4x, has unresolved/slightly negative five-task macros, and lowers fresh
core-4 at 4K and 16K. It is therefore not promoted as a solution to RoPE
extension or as an improvement to the retained static method.

This result identifies a useful boundary: likelihood repair and positional
task repair are different estimands. A table can become easier for the adapted
model to predict under teacher forcing without becoming more reliable at
retrieval, tracking, or answer generation.

## 2. Frozen adaptations

Both adapters use the exact float32 table
`56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b`,
all-layer `q_proj,k_proj` rank `8`, alpha `16`, no dropout/bias, seed
`20260902`, 96 optimizer steps, and the same repeating `LLLSS` correct-plus-
answer-span-deranged training views. Each has `1,048,576` trainable parameters.

The recovered 2026-09-02 artifact used `attention_scaling=1.0`; it was not an
execution of the current method. The new matched run fixes scaling to
`1+0.074 log(4)=1.102585782722872` throughout training and evaluation. It
completed in `402.56 s`, with peak reserved memory `28,219,277,312` bytes and
finite losses/gradients at all 96 steps. These are training diagnostics, not
performance evidence.

## 3. PG-19 paired replay

Positive deltas below favour the adapter by reporting frozen minus adapted NLL.
Each cell has 20 identical documents and 10,240 target tokens.

| Gain | Length | Frozen NLL | Adapted NLL | Delta | Paired 95% interval |
| --- | --- | ---: | ---: | ---: | ---: |
| unit | 1x | `3.303623` | `3.286281` | `+0.017342` | `[+0.011467,+0.024484]` |
| unit | 4x | `3.581615` | `3.475397` | `+0.106218` | `[+0.078272,+0.143641]` |
| `.074` | 1x | `3.104234` | `3.090126` | `+0.014107` | `[+0.009616,+0.019405]` |
| `.074` | 4x | `3.081946` | `3.063204` | `+0.018742` | `[+0.013260,+0.025634]` |

The unit-gain baseline was rerun cleanly after an initial logging-directory
error allowed two evaluator processes to overlap and produced 44 rather than
40 rows. The contaminated output is preserved and marked **Invalid**; only the
isolated 40-row rerun enters this report.

## 4. Natural generated tasks

Qasper, MultiFieldQA-en, HotpotQA, 2WikiMQA, and GovReport use identical fixed
rows at 1x/4x. Intervals resample paired rows within task and then equal-weight
the five task means.

| Gain | Length | Adapter minus frozen macro | Paired 95% interval |
| --- | --- | ---: | ---: |
| unit | 1x | `-0.01723` | `[-0.04915,+0.00561]` |
| unit | 4x | `+0.01023` | `[-0.00883,+0.03525]` |
| `.074` | 1x | `-0.00582` | `[-0.02054,+0.00352]` |
| `.074` | 4x | `-0.00233` | `[-0.02420,+0.02074]` |

The task splits are heterogeneous. Unit gain improves GovReport 1x by
`+0.01039` with interval `[+0.00521,+0.01601]`, but decreases HotpotQA and
Qasper 1x by `-0.05` each. Under `.074`, no task-length cell has an interval
strictly above zero; Qasper 4x is `-0.02648` and 2WikiMQA 1x is `-0.03125`.
PG-19 improvement therefore cannot be relabeled as natural generation repair.

## 5. Fresh core-4 RULER at c=.074

Both arms use the adapter-capable evaluator with the same script hash, table,
gain, data manifest, decoder, scorer, and 20 rows per task/length. CPU-only
preflight verified all 240 rows, and the 4K one-row-per-task smoke scored `4/4`
for both arms.

| Length | Frozen | Adapted | Delta | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| 4K | `.8375` | `.8275` | `-.0100` | `[-.0400,+.0100]` |
| 8K | `.7225` | `.7250` | `+.0025` | `[0,+.0075]` |
| 16K | `.4025` | `.3800` | `-.0225` | `[-.0600,+.0050]` |

The 16K decrease is concentrated in numeric multikey (`-.10`), while VT gains
`.01`; single-key and UUID multikey are unchanged. The small 8K increase is a
VT-only `.01` change after task-equal averaging. This is redistribution, not a
robust capability gain.

## 6. Artifact identity

- Recovered unit-gain training receipt / adapter tensor:
  `2ced79686df85cb5fdea2cf949239b1fea90f9fa2ca15898a218dfcd6791c166` /
  `48f54cf5b6e761dd81e03e8f2516da69cc9ee95ea9252bf529dc1044911cacd9`.
- Unit-gain PG-19 clean frozen/adapted result JSON:
  `34ab78224fa68344e1bb9f0458f6a1ed23ef4fa8949f103afc37a548b3ed391d` /
  `4bb5e6c01acb8ef87a12e6a3f26e62d6c1826aa7112930b14deefa37ae3b579a`.
- Unit-gain five-task frozen/adapted result JSON:
  `81cb66167f30cdeb2b3bf5fc6ff6455bd240bd238be57b77c017cdb5a4411d7b` /
  `1419dd69968aa0df4dbfdb97b8b85050c2ec6c2be4fda7f127b7a900dbb96af8`.
- `.074` training script / receipt / adapter tensor:
  `2a2ee346c465ad1f451be0baee751e507134919f1edcf570af3d17c9e8553e2a` /
  `50a6a901bf3aad16629c2c297d653472dec424377016ed60f4405fea72b96726` /
  `a9f4483adbf6e1d056f8bfbaff274571b975fb288c44f0a207f78214889eaca0`.
- `.074` PG-19 frozen/adapted result JSON:
  `dc673cbd0225ee7f78b927f4973658bfefd705da5af10987cf4e2a4dd9c69503` /
  `e7f66b9b90a09cc727e40e503c13630bd67f3eef260526cb3939f7e78603b07a`.
- `.074` five-task adapted result JSON:
  `b86c1e4bd4d843f3cf64fbca19a6e2e1ae3ac2f3690e72922d5582f9fb5a7580`.
- `.074` RULER frozen/adapted result JSON:
  `78e72e7b67767c391d599324c753b3aaec75f3616fa7f8d6d1d7b5fd0281f9b5` /
  `3e4d372312d5e60d20c40fb115a195b2a1018147d4503062dd57ecb29b9bf950`.

Raw predictions, rows, adapter weights, training logs, manifests, and console
output remain private on the work machine.

## 7. Supported and unsupported claims

Supported: Q/K-only rank-8 adaptation can improve paired PG-19 NLL on the exact
static table at both unit gain and `.074`; the gain changes the size and
downstream expression of that improvement; and the exact `.074` recipe does
not improve the measured five-task or core-4 macros.

Unsupported: that LoRA solves RoPE position encoding, that PG-19 predicts
retrieval/generation, that Q/K is universally sufficient, that rank 8 or 96
steps is optimal, or that another gain/budget would rescue capability. The
single seed supports a candidate decision, not training-variance claims.

## 8. Stop decision

Stop this exact adapter recipe. Do not sweep rank, alpha, gain, schedule, or
steps to rescue it. The useful scientific result is the estimand split:
same-substrate Q/K adaptation repairs likelihood while leaving structured and
natural generated capability unresolved or slightly worse. Further work must
change the learning signal or derive a behavioural prediction before consuming
more GPU; repeating this proxy objective is not progress.
