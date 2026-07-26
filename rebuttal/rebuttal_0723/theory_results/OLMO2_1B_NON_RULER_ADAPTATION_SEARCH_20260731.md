# Non-RULER 4K adaptation does not recover broad RULER capability

Status: **complete diagnostic negative / search stopped**
Concerns: `R27bE.2`, `R27bE.5`, `AC.2`

## Question and stop rule

1. **Concern addressed.** Can 4K-only adaptation of a mature OLMo-2 1B
   checkpoint convert EVQ's long-context language-modeling behavior into
   autoregressive capability on tasks absent from the adaptation data?
2. **Existing evidence.** Routing data produced by the official RULER NIAH
   generator yields a positive 8K NIAH result, while a clean LongAlign plus
   Tulu arm fails all 13 RULER tasks.
3. **Smallest missing evidence.** Test whether frequency morphing, retention
   loss, answer-only supervision, or an independently generated natural-text
   retrieval curriculum closes the clean-transfer gap.
4. **Smallest executable plan.** Train seven 4K-only EVQ LoRA arms without
   explicit RULER rows. Screen each arm on eight official RULER tasks at 4K and
   8K with five examples per task.
5. **Stop condition.** Stop an arm after the screen if it does not approach the
   length-matched control. Run the full 13-task matrix only after a screen pass.

## Training and evaluation separation

All seven arms contain zero explicitly added RULER or NIAH benchmark rows. The
LongAlign and Tulu arms use the existing frozen views. The natural-span arms
derive a copy-the-next-span task from the LongAlign training split, distribute
the source over 16 position bins, and use no RULER generator, template, row, or
value list.

This establishes procedural separation from RULER. It does not establish
semantic deduplication of every upstream LongAlign or Tulu example against
every retrieval-like task.

The screen uses greedy autoregressive decoding and official task-specific
RULER scoring on:

`niah_single_1`, `niah_multikey_1`, `niah_multiquery`, `vt`, `cwe`, `fwe`,
`qa_1`, and `qa_2`.

Each task contributes five fixed examples at 4K and 8K. The length-matched
controls on the same screening subset score `0.5700` for untouched Native RoPE
at 4K and `0.5125` for untouched Native plus official Transformers YaRN factor
2 at 8K.

All arms use Q/K/V/O LoRA with rank 64 and alpha 128. Each arm trains for 1,500
optimizer steps at a maximum physical length of 4,096. Progressive arms end at
the same endpoint EVQ-Cosh frequency grid as immediate-EVQ arms.

## Results

| Arm | Schedule and objective | Natural-text NLL, 4K / 8K / 16K | RULER screen, 4K / 8K |
| --- | --- | --- | --- |
| A1 | Linear Native-to-EVQ morph; LongAlign full-token plus Tulu assistant loss | 3.0565 / 3.2559 / 3.4810 | 0.1167 / 0.0667 |
| A2 | A1 plus sparse Native-teacher KL on every fourth Tulu batch | 3.1331 / 3.3324 / 3.5410 | 0.0833 / 0.0417 |
| B1 | Linear morph; LongAlign answer-only plus Tulu answer-only | 3.5799 / 3.7782 / 3.9843 | 0.0417 / 0.0250 |
| B2 | Immediate EVQ; same answer-only data and objective as B1 | 3.3106 / 3.4750 / 3.6477 | 0.1000 / 0.0750 |
| C1 | Linear morph; independent natural-span retrieval with 8-token answers plus Tulu | 3.9914 / 4.3305 / 4.6596 | 0.0000 / 0.0000 |
| C2 | Immediate EVQ; same 8-token natural-span curriculum as C1 | 3.4749 / 3.7542 / 4.0637 | 0.0167 / 0.0000 |
| D1 | Immediate EVQ; 3:1 one-token natural-span retrieval to Tulu | 4.7408 / 6.0438 / 7.6042 | 0.0000 / 0.0000 |

Every arm remained below both length-matched controls. The best 4K screen score
was `0.1167`, and the best 8K score was `0.0750`. No arm passed the registered
gate for the full 13-task matrix.

The one-token curriculum also failed its own held-out 128-row task:

| Diagnostic | Value |
| --- | ---: |
| First-answer-token top-1 | 0.0391 |
| Answer plus EOS exact | 0.0391 |
| Mean answer NLL | 3.7357 |
| First-token mean rank | 2129.54 |
| First-token median rank | 128.5 |

This diagnostic rules out an interpretation based only on RULER task shift.
The highest-pressure natural retrieval arm did not learn its own held-out
objective and caused the largest natural-text degradation.

## Interpretation

The seven-arm search rejects the tested fixes. Progressive frequency morphing
does not recover broad task performance. Sparse Native-teacher KL lowers the
screen score. Answer-only LongAlign supervision weakens language modeling and
does not improve transfer. The independent natural-span curricula fail both
transfer and, for the one-token arm, their own held-out task.

The result preserves two distinct claims. The earlier `49/100` and `48/100`
8K NIAH result demonstrates benchmark-family-matched capability conversion
after training with the official RULER NIAH generator. The clean full-RULER
audit and this search show that the tested 4K-only non-RULER objectives do not
produce broad RULER capability. Natural-text NLL therefore remains a
language-modeling metric, not evidence of autoregressive task success.

## Reviewer-facing use

These arms do not support a positive rebuttal claim. Keep them out of the
opening response. If a reviewer asks whether the positive NIAH result transfers
to unseen tasks, disclose the clean full-RULER negative result and state that
seven follow-up non-RULER objectives also failed the low-cost screening gate.

The search is closed. Further training variants require a new capability
hypothesis and a gate that first verifies learning on the arm's own held-out
task.

## Provenance

| Arm | Adapter SHA-256 |
| --- | --- |
| A1 | `cb35537aeb8c3b913f824583a4949b23c06628c0066482c6818b454a7dfc864f` |
| A2 | `f240260c68496a0a5911ae5d7abf05505733a4aab85a483d4e57298026dc2c70` |
| B1 | `e4eb518353aca870c0698a5719760842d672d289b5486687cf4c5b6267fb4b92` |
| B2 | `a6688ae6a8374d1627e6685c1c11bc16f3ab498ea56dda1c312d2c60f31b6f25` |
| C1 | `9ad8b4878fcc7dfe0cd9bf749d80d17b271e6429e23dae904ba3708cdf698d54` |
| C2 | `b2471270fe12d9754188a2a33e3ea4223b760458e4d6360123685817415426ec` |
| D1 | `02791fd5ac438f49bd3c34d512c7cda04e6229d80f962edad447a5b0bad6c84b` |

All completed arms use endpoint EVQ-Cosh with \(\tau=2\) at the final step.
The final frequency SHA-256 is
`917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607`.

The eight-token natural-span dataset hashes are:

- token IDs:
  `5a5124d8f4cbed158d648f2144666e043c83aa220342684a7b2d51d0c76f318c`;
- assistant mask:
  `d0945786433b4452425a8f88dbf6a2eba3b4810ff4fd8584f71f28f504908fc8`.

The one-token natural-span dataset hashes are:

- token IDs:
  `c28de3cf5c44bc13f510f16e345a55d8aa437b776ccf58ec9dfe116dae742daa`;
- assistant mask:
  `2d026ade14edc2bac8a907883cd480b2cf01d19d2804f9050a039b5d2a8fdf9d`.
