# Existing Real-Document QA Evidence

Status: **completed matched evidence — no new experiment required**

Concerns: `RDz6s.1`, `AC.2`

## Result

The 1.485B OLMo-2 EVQ continuation kept every backward pass at physical 4K and was evaluated autoregressively on two real-document QA families:

| Task | 4K | 8K | 16K |
| --- | ---: | ---: | ---: |
| HotpotQA | 25 | 25 | 10 |
| SQuAD QA | 15 | 10 | 10 |

These are measurable absolute capability results on real documents and human-written questions after 4K-only training.

## Matched result and claim boundary

The matched Native arm has now completed the same continuation and evaluation.
Native/EVQ official macro is `82.16%/37.51%` at 4K,
`0.08%/21.29%` at 8K, and `0%/6.13%` at 16K. At 16K, Native is zero
on every task while EVQ retains non-zero SQuAD and HotpotQA scores of `10%`
each.

This supports matched, task-adapted length transfer, including real-document
QA endpoints. It does not establish unseen-task transfer, broad real-world
superiority, or pure interior-shape attribution. Native's large 4K advantage
must remain adjacent to the 8K/16K result.

---

## Reading trap — record this before anyone re-reads the owner

The per-task table in `theory_results/OLMO2_1B_4K_RULER_FAMILY_ADAPTATION_20260726.md`
has two numeric columns, "Legacy two-seed mean" and "New 4K task-family
adaptation". **Both arms are EVQ.** The comparison is between two adaptation
recipes, not against Native. The owner lists "an EVQ-versus-Native causal
advantage" among the claims it does *not* support, "because no matched Native
arm received this same continuation mixture".

On 2026-07-27 a draft misread that table as EVQ-versus-Native. The table
remains EVQ-only and must not be used as the matched comparison. Its original
boundary predates the completed control. The matched result is instead owned by
`theory_results/OLMO2_1B_MATCHED_RULER_CONTINUATION_20260727.md`.

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

## Completed matched control

The Native-LoRA arm received the identical 13-family continuation:

- `OLMo-2-0425-1B-Instruct`, native RoPE;
- same 480 + 128 + 128 rows, same order;
- rank-64 / alpha-128 Q/K/V/O LoRA, physical 4K backward passes only;
- three deterministic passes, 276 optimizer steps, global batch 8, BF16,
  fused AdamW, lr 2e-5, 14 warmup steps;
- the same 13 tasks were evaluated at 4K/8K/16K with the same harness and
  seeds.

The Native continuation completed in `308.8 s`, produced adapter SHA-256
`da8111055a23ea9f56ff675399eebc2c47c5afbedb6b7079dc12598889280d80`,
and ended at validation NLL `0.5020`. Its evaluation contains all 39 cells and
780 generations.

## What it establishes

1. A second matched 13-family comparison at 1.485B scale, including both
   real-document QA tasks.
2. A direct answer to `RzWsa.3`: RULER improves at 2× under matched
   task-family supervision on both OLMo-2 and LLaMA-3-8B.
3. Matched real-task evidence for `AC.2` at both 1.485B and 8B.

## Boundaries

- One continuation seed per arm unless more are run.
- Generator families are shared between continuation training and evaluation,
  so this stays task-family-adapted length transfer, not unseen-task transfer.

## Status

**Complete and promoted.** The reviewer-facing wording is synchronized in
`paste/REVIEWER_zWsa.md`, `paste/REVIEWER_Dz6s.md`, `paste/AC_PUBLIC.md`, and
`paste/AC_CONFIDENTIAL.md`; the standalone owner is
`theory_results/OLMO2_1B_MATCHED_RULER_CONTINUATION_20260727.md`.

## Claude Opus 5 handoff — 2026-07-28

This experiment is complete. **Do not rerun it.** Use the following read order:

1. `theory_results/OLMO2_1B_MATCHED_RULER_CONTINUATION_20260727.md`;
2. the `E-OLMO-RULER-FAMILY` row in
   `theory_results/REVIEWER_USABLE_EVIDENCE_LEDGER_20260726.md`;
3. `01_REBUTTAL_PLAYBOOK.md`;
4. the four reviewer-facing files listed in **Status** above.

The reviewer-facing matched Native/EVQ official RULER macro is:

| Length | Native | EVQ |
| --- | ---: | ---: |
| 4K | 82.16% | 37.51% |
| 8K | 0.08% | 21.29% |
| 16K | 0.00% | 6.13% |

After the matched continuation, validation NLL is `0.5020` for Native and
`0.9881` for EVQ. The defensible interpretation is that Native remains stronger
in-window, while EVQ retains substantially more task-family-adapted capability
at 2× and 4× length. At 8K, Native is nonzero only on CWE (`1%`); EVQ is
nonzero on 10 of 13 tasks, including VT (`31%`), CWE (`22%`), FWE (`60%`),
SQuAD (`10%`), and HotpotQA (`25%`). At 16K, Native is zero on all 13 tasks;
EVQ remains nonzero on seven task cells, including FWE (`50%`), SQuAD (`10%`),
and HotpotQA (`10%`).

Mandatory claim boundaries:

- one continuation seed per arm;
- task-family supervision, not clean unseen-task transfer;
- combined EVQ frequency substrate plus LoRA adaptation, not pure Cosh
  attribution;
- no 32K claim;
- keep Native's 4K advantage adjacent to the 8K/16K EVQ advantage;
- do not reuse the earlier EVQ-only table as a Native comparison.

The frozen 12-file evidence bundle checksum-manifest SHA-256 is
`b8b668ac1fa4fbf626d4805888d15fed5902d937fb91829753ebe9413a172830`;
the standalone owner records the individual artifact hashes. No additional GPU
work is needed for this comparison. Claude should work only on concern
alignment, wording, and character-budget compression unless the author
explicitly opens a new experiment question.
