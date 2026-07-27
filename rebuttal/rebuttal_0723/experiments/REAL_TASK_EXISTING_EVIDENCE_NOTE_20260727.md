# Existing Real-Document QA Evidence

Status: **completed supporting evidence — no new experiment required**

Concerns: `RDz6s.1`, `AC.2`

## Result

The 1.485B OLMo-2 EVQ continuation kept every backward pass at physical 4K and was evaluated autoregressively on two real-document QA families:

| Task | 4K | 8K | 16K |
| --- | ---: | ---: | ---: |
| HotpotQA | 25 | 25 | 10 |
| SQuAD QA | 15 | 10 | 10 |

These are measurable absolute capability results on real documents and human-written questions after 4K-only training.

## Claim boundary

No matched Native arm received the same continuation mixture. Therefore:

- use these scores only as absolute, task-adapted capability evidence;
- do not describe them as an EVQ-versus-Native win or causal allocation evidence;
- do not claim unseen-task transfer or broad real-world superiority.

The matched 1.485B NIAH and matched 8B RULER results remain the comparison evidence; this 1.485B HotpotQA/SQuAD result supplies only the bounded real-document endpoint.

---

## Reading trap — record this before anyone re-reads the owner

The per-task table in `theory_results/OLMO2_1B_4K_RULER_FAMILY_ADAPTATION_20260726.md`
has two numeric columns, "Legacy two-seed mean" and "New 4K task-family
adaptation". **Both arms are EVQ.** The comparison is between two adaptation
recipes, not against Native. The owner lists "an EVQ-versus-Native causal
advantage" among the claims it does *not* support, "because no matched Native
arm received this same continuation mixture".

On 2026-07-27 a draft misread that table as EVQ-versus-Native, concluded that
QA had not improved, and nearly wrote a false concession into the `Dz6s` reply
stating that the real-task part of the concern remained open. It does not: the
matched **LLaMA-3-8B** comparison shows Native-LoRA at zero normalized-exact on
all 13 tasks at 16K while EVQ-LoRA has normalized-exact successes including
QA2 (`ruler_qa_hotpot`).

**Rule:** before writing any "X did not improve" or "X is worse" sentence,
confirm the identity of both arms in the table being read.

## What the two QA families actually are

Both are pinned by SHA-256 in `experiments/olmo2_1b_evq/prepare_ruler_data.py`:

| Task | Source |
| --- | --- |
| `ruler_qa_squad` | SQuAD v2 dev (`squad_dev_v2.json`) |
| `ruler_qa_hotpot` | HotpotQA distractor dev (`hotpot_dev_distractor_v1.json`) |

Real passages and human-written questions; the long-context instance is
constructed by padding with distractor documents to the target length. Correct
external phrasing is **"built from real-document QA datasets"**, not
"non-synthetic".

## Missing control and how to close it

**A matched Native-LoRA arm receiving the identical 13-family continuation
mixture.** Everything else exists: the continuation data (480 official-generator
rows — 96 each for VT, CWE, FWE, SQuAD QA, HotpotQA — plus 128 paired NIAH
replay and 128 LongAlign replay rows), the training script, the evaluation
harness, and the frozen EVQ parent adapter.

Run the same recipe on the Native substrate:

- `OLMo-2-0425-1B-Instruct`, native RoPE;
- same 480 + 128 + 128 rows, same order;
- rank-64 / alpha-128 Q/K/V/O LoRA, physical 4K backward passes only;
- three deterministic passes, 276 optimizer steps, global batch 8, BF16,
  fused AdamW, lr 2e-5, 14 warmup steps;
- evaluate the same 13 tasks at 4K/8K/16K with the same harness and seeds.

Reference cost: the EVQ continuation took **431.6 s** wall clock including
first compile, at approximately 30.3K physical token/s on an RTX 5090.
Evaluation over the 39 cells dominates the total.

## What it buys

1. Converts this note's result from an absolute capability statement into a
   second **matched** comparison, at a second model scale, on the same 13
   families including both real-document QA tasks.
2. Removes the "compared to what?" handle from `paste/REVIEWER_zWsa.md` §3,
   currently the weakest sentence in that reply.
3. Gives `RzWsa.3` ("my score would increase if the method improves RULER") a
   two-model answer instead of a one-model answer.
4. Lets `AC.2` be answered with matched real-task evidence at both 1.485B and 8B.

## Boundaries that will still apply after the run

- One continuation seed per arm unless more are run.
- Generator families are shared between continuation training and evaluation,
  so this stays task-family-adapted length transfer, not unseen-task transfer.

## Status

**Not blocking.** The current `paste/` texts are consistent with the evidence as
it stands: the LLaMA matched result carries the real-task claim, and this OLMo
result is reported without a comparison. If the Native arm completes before
submission, update `paste/REVIEWER_zWsa.md` §3 and `paste/AC_PUBLIC.md` to
report it as a matched comparison and delete the "no Native counterpart" clause.
