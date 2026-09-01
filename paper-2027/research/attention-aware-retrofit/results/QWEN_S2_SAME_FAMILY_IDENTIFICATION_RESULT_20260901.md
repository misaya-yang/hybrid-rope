# Same-generation Qwen s2 identification (2026-09-01)

## Decision

**Status: `FIXED_BASELINE_PANEL_COMPLETE / INDEPENDENT_CONFIRMATION_UNRESOLVED`.**

Both checkpoint resolvers pass: Native at 32K and deterministic YaRN2 at
64K are nonzero. The unchanged K64 C2-s2 profile passes the Native RULER
point gate and remains useful at 64K, but is not shown superior to YaRN2.
K32 retained a historical sample-level Native/long crossing, but its paired
intervals did not establish physical-versus-index ordering. The later
independent N80 confirmation also fails to identify that ordering and does not
reproduce the old physical long advantage.

Thus this stage supports useful fixed-s2 behavior on the two Qwen checkpoints,
not a causal K effect, a universal physical coordinate, or SOTA. Its registered
sample increase is now complete; the superseding interpretation is owned by
[`K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901`](K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md).

Owners:

- [Preregistration](../preflights/QWEN_S2_SAME_FAMILY_IDENTIFICATION_PREFLIGHT_20260901.md).
- [K32 complete panel and raw identities](../evidence/QWEN_K32_MATCHED_S2_BASELINE_RECEIPT_20260901.json).
- [K64 complete panel and raw identities](../evidence/QWEN_K64_MATCHED_S2_BASELINE_RECEIPT_20260901.json).
- [K32 historical-output replay](../evidence/K32_HISTORICAL_RULER_REPLAY_RECEIPT_20260901.json).

## 1. Matched scientific contract

The checkpoints are the existing Qwen2.5-0.5B-Instruct (K32) and
Qwen2.5-1.5B-Instruct (K64) artifacts, with b=1e6 and configured reference
32768. Target 65536 fixes s=2. Gemma's calibrated 4096 reference is not
transferred to these checkpoints. Their Native RULER resolvers are not a new
two-family natural/capability reference-frontier calibration.

The boundaries remain `.7382780681078285/.366403835112904`; C2 and index use
c=.074 and amplitude `1.0512928913614359`. YaRN retains the published fixed
beta bounds and amplitude `1.0693147180559945`, verified against the installed
HF CPU initializer. There is no boundary, scale, gain or profile search.
Each long profile is fixed before prefill and used unchanged at both lengths.

Use the four official tasks, 20 rows/task/length, seed 20260822. All compared
arms at a checkpoint have matching tokenizer and per-cell input hashes.
The K64 data live in two historical manifests, but their 32K/64K cell hashes
also match the corresponding K32 cells. A different manifest wrapper is not
treated as a different input, and inputs alone do not make distinct models
a causal K comparison.

This completion adds 480 generations: K32 YaRN2 and K64 YaRN2/C2-s2, each
at both lengths. Another 24 fixed K32 canaries verify historical decoded
outputs. Historical rows are not relabeled independent new samples.

## 2. Complete length curves

Task-vector order is single / mk2 / mk3 / VT.

| Checkpoint | Fixed profile | 32K task vector | 32K macro | 64K task vector | 64K macro | Native retention |
| --- | --- | --- | ---: | --- | ---: | ---: |
| K32 | Native | `1/.80/.25/.54` | .6475 | `.75/.20/0/.16` | .2775 | 1.0000 |
| K32 | physical-x | `1/.50/.15/.44` | .5225 | `1/.50/.05/.47` | .5050 | .806950 |
| K32 | normalized-index | `1/.60/.25/.46` | .5775 | `1/.40/0/.35` | .4375 | .891892 |
| K32 | YaRN2 | `1/.40/.35/.50` | .5625 | `1/.25/.05/.46` | .4400 | .868726 |
| K64 | Native | `1/.85/.50/.93` | .8200 | `1/.30/0/.88` | .5450 | 1.0000 |
| K64 | C2-s2 | `1/.85/.30/.92` | .7675 | `1/.65/.15/.73` | .6325 | .935976 |
| K64 | YaRN2 | `1/.65/.50/.92` | .7675 | `1/.55/.30/.93` | .6950 | .935976 |

The `.875` Native **point** gate passes K32 index and both K64 long profiles.
K32 physical and YaRN2 fail that point gate. In particular, Native cost is
not exclusive to the frozen coupling profile. These are RULER capability
ratios; no Qwen natural-NLL/PPL double-gate claim is made.

## 3. Paired uncertainty changes the interpretation

The reproducer is
[`summarize_qwen_s2_baseline_completion.py`](../../../../scripts/analysis/summarize_qwen_s2_baseline_completion.py).
It verifies terminal rows, raw hashes, paired inputs/references/scorer,
checkpoint/Native tensors and static s2/gain identity. It uses 10,000 paired
row-bootstrap replicates, seed 202609025, within the four fixed task strata.
No observations are removed; these intervals condition on one checkpoint,
fixed tasks and the recorded evaluation contract.

| Contrast | 32K delta [pointwise 95% CI] | 64K delta [pointwise 95% CI] |
| --- | --- | --- |
| K32 physical − Native | -.1250 `[-.2200,-.0300]` | +.2275 `[.1275,.3300]` |
| K32 physical − index | -.0550 `[-.1450,.0325]` | +.0675 `[-.0050,.1425]` |
| K32 physical − YaRN2 | -.0400 `[-.1275,.0450]` | +.0650 `[-.0175,.1475]` |
| K64 C2 − Native | -.0525 `[-.1300,.0225]` | +.0875 `[-.0075,.1800]` |
| K64 C2 − YaRN2 | .0000 `[-.0700,.0700]` | -.0625 `[-.1625,.0350]` |

K32 physical-versus-Native has opposite, resolved sample effects at the two
lengths. The stronger **physical-versus-index** claim does not follow: both
intervals span zero, also under the reported two-length Bonferroni sensitivity.
Likewise the K64 C2/YaRN comparison is unresolved, not an equivalence result.
Do not use `.01--.03` differences or a single point-gate crossing as a method
ranking. All complete per-task results and sensitivity intervals remain in
the compact receipts, including unfavorable outcomes.

## 4. Execution provenance and limitations

The exact K64 Native-32K historical runner (`bbc8a41f…`) was recovered from
Git; its greedy/prompt/scorer and imported helper behavior is unchanged by the
new profile-target and post-score token metadata additions. The latter
changes do not themselves alter the default Native frequency tensor.

K32 historical runner source was not fully recovered. Its 480 raw rows and
official scores were verified; the registered first-row replay across three
profiles, two lengths and four tasks gives **24/24 identical full decoded
predictions and official scores**, including old failures. This is bounded
decoded-output parity, not token/EOS parity or full-matrix execution identity.

K64 Native64K `.5450` remains a raw-hash-verified historical same-data anchor.
Its older runner SHA `d2e0a518c83af2a34eeaf25615f80818f9dff271d5ab59f508542d6bddd535d2`
was not recovered from the bounded Git history search, so complete execution
equality with current P2 is not asserted. C2 and YaRN2 in this stage are both
current-run comparisons. No missing historical source is silently replaced
by a neighboring version.

The first K32 launch stopped at a CPU receipt-schema mismatch; the first K64
attempt stopped at the checkpoint-bound data-manifest check. Both were fixed
before any affected evaluation rows were produced. The original manifests
were not retargeted or overwritten: K64 uses its own verified input owners.
These are implementation/preflight events, not failed method arms.

## 5. What this says about K, scale and the next experiment

K64 physical/index are the same counterfactual-K64 construction in exact
arithmetic; formula/runtime float rounding can yield distinct tensor hashes
but no independent coordinate intervention. Hence only one K64 C2-s2 arm is
run. The K64 zero contrast cannot be used as a measured K trend.

The results support a bounded statement: the frozen two-parameter geometry
can be reused at s2 with useful capability on these checkpoints, and K64
Native retention improves relative to its previously failed s4 point gate.
They do not establish arbitrary-scale generalization or isolate model size,
training, attention-head structure and rotary budget from one another.

The registered
[`K32 independent crossing confirmation`](../preflights/K32_PAIRED_CROSSING_CONFIRMATION_PREFLIGHT_20260901.md)
has completed on seed 202609026 with 80 rows/task. Physical/index 64K macros
are `.46625/.46125`; physical-minus-index is `+.0050` with corrected interval
`[-.038125,.048750]`. Normalized-index passes the 32K Native point gate while
physical fails. The conditional Native-Q/K entrance therefore failed; the CPU
KL module remains unused infrastructure, not an executed mechanism result or
validated predictor.
