# Frozen coupling: P0 Native-only reference-length identification

## Decision and fixed scope (user priority correction)

The next round keeps the existing frozen two-parameter law and uses the
already available checkpoints. It does not fit a new curve, boundary, gain,
or effective context length. The purpose is to identify why K32 has a
Native/long Pareto crossing and why the K128 screen collapses at 16K before
constructing a general method. An offline selector is not substituted for
identification. The initially proposed YaRN/NTK GPU panel is deferred; no
baseline arm was launched. CPU baseline-export code is an unrun future asset.

Execution order is P0 reference length, P1 reference-correct K128, P2
same-family K triangulation, P3 Native compatibility mechanism, then P4
selector and P5 broad matched baselines. Later stages require evidence from
the preceding stage. Extra compute does not authorize extra free parameters.

Existing evidence belongs to
[`FROZEN_2D_COUPLING_TRANSPORT_RESULT_20260901`](../../results/coupling-transfer/FROZEN_2D_COUPLING_TRANSPORT_RESULT_20260901.md).
Its Native-versus-external canaries close only the loader-branch difference:
both branches share the custom attention backend and handwritten generation.
They do not establish stock-HF or phase-arithmetic correctness.

## A. Completed implementation controls before P0

Code: [`audit_rope_runtime_parity.py`](../../../../../scripts/eval/audit_rope_runtime_parity.py).

- Exact existing Gemma-1.1 instruction artifact and existing paired data.
- Native table, original checkpoint config, no model or frequency update.
- First row of `niah_single_1` and `niah_multikey_2` at 4096/8192/16384:
  six diagnostic rows, not a capability confirmation.
- Compare stock-HF SDPA plus `generate` with the existing custom Flash
  backend plus handwritten greedy decoding on identical input IDs and
  explicitly matched EOS and generation budgets.
- Record prefill and one teacher-forced cached-step logits, stock cached
  versus uncached continuation, raw generated IDs, scores, runtime source
  hashes, Native tensor identity, timing and peak memory.
- Flash-only execution, no quadratic full-length eager fallback.
- Freeze the script/data/weight/config identities in the raw run manifest.

Any discrepancy must be diagnosed before a new capability matrix. Agreement
excludes the tested implementation differences only; both paths still share
the installed Gemma implementation and PyTorch Flash kernel. If phase
arithmetic is suspect, compare realized phases with integer-position/FP64
reference, and test IEEE-FP32 arithmetic as an implementation control, not
as a new RoPE candidate. A small Q/K slice can supply an eager reference
without materializing a full long-context attention matrix. The six original
canaries are never counted as fresh P0 calibration examples.

After the first fresh Native NLL result, one bounded shared-kernel check is
registered: the first two natural calibration documents at 4096 and 8192,
same weights/inputs/Native table, Flash versus explicit FP32 attention with
128-query chunks. It never constructs a full quadratic attention matrix.
This is an explicit implementation reference, not a silent fallback or a
candidate. If either document has mean suffix-NLL disagreement above `.05`,
pause interpretation and diagnose the implementation; otherwise retain the
Native endpoint diagnosis. No length is selected by this check.

## B. Hypothesis, confound and fixed construction

**H0:** a mismatched behavioral reference length or request scale may explain
the existing K128 negative before the frozen law itself is rejected.

**Confounds:** config capacity, documented training length, task-specific
behavioral range, changing target difficulty, increasing distractor count,
format failures and shared implementation errors are different quantities.

The [official Gemma technical report](https://arxiv.org/html/2403.08295v4)
explicitly states training context 8192 in its architecture section. This
supersedes speculation that the family was simply pretrained at 4K. It does
not establish the task-level range of this exact instruction redistribution,
nor every instruction/RLHF sequence length. `L_config=8192`, documented
family training context, and `L_ref(protocol)` remain separate fields.

Use exact existing Gemma-1.1 weights, original config, original Native
frequency tensor, gain one, and grid `{1024,2048,4096,8192}`. There are no
candidate tables in P0. No final long holdout score is an input to the data
builder, selector, thresholds or confirmation.

### B.1 Fresh paired inputs

Freeze both calibration and confirmation manifests before inference:

| Family | Calibration | Independent confirmation | Paired target |
| --- | ---: | ---: | --- |
| FineWeb-Edu natural continuation | 32 documents | 64 new documents | identical final 256 target tokens across nested prefixes |
| Controlled two-code retrieval | 64 blueprints | 128 new blueprints | same two requested codes and six distractor records across lengths |

Natural documents are selected by source order, minimum token length and
text identity only, from source row 20000 onward, excluding supplied prior
source-text hashes. No loss-based selection; no appended EOS for an arbitrary
document cut. Each request includes one BOS and its scored continuation within
the stated total length.

Capability is a fresh deterministic generator, not reused RULER rows. Seeds
are `202609011/202609012`; four evidence-depth strata and query order are
balanced in advance. The number of facts never increases with length; only
neutral filler changes. A compact version removes filler but preserves all
eight records and the query. The complete requested continuation is
`code1, code2.` after a fixed assistant prefix, followed by tokenizer EOS,
within 48 generated tokens. Only outer whitespace is normalized; no substring,
first-number extraction or partial-answer success. Raw token IDs, terminal
token, exact text and format failures are retained.

The grid length includes the generation budget. Natural and capability are
different families but do not establish task-independent validity. Increased
historical content/filler remains part of the operating-range estimand;
this is not a pure causal RoPE-distance intervention.

### B.2 Preregistered decision rule

The reference anchor is 1024. Reuse the existing practical retention margin
`.875`, now with explicitly paired estimands:

- natural: `exp[-mean_doc(NLL_L-NLL_1024)] >= .875`, equivalently
  paired degradation at most `-log(.875)` nats/token;
- capability: `exact_L / exact_1024 >= .875`;
- instrument competence: 1024 and compact exact success each at least `.75`.

Calibration proposes the largest contiguous passing grid point **separately
for each family**. Require identical frontiers, adequate instrument competence
and no fail-then-pass reentry. Otherwise abstain; do not take the minimum,
average or best-looking family. The provisional point is frozen before
confirmation inference.

Confirmation evaluates only 1024, the selected point and its immediate next
grid point if one exists, plus compact controls. Accept the selected operating
point only if all four requirements pass one-sided 98.75% bounds (Bonferroni
familywise .05). Natural differences use 10,000 paired document-bootstrap
replicates, seed `202609013`. Capability retention uses conservative
Clopper-Pearson marginal bounds, each tail .00625, rather than degenerate
all-success bootstrap certainty; also report paired blueprint-bootstrap
intervals, stratified by depth. Competence uses one-sided Clopper-Pearson
bounds at .0125. Exact algebraic anchor identity is not a population estimate.

If confirmation fails or is unresolved, abstain without descending the grid.
The next point only determines whether a frontier was located; it may not
promote the selected point. An accepted 8192 is right-censored at config
capacity. Distinguish a confirmed operating point from a confirmed upper
boundary and from historical training exposure.

## C. Conditional later stages, not an immediate queue

Only after P0 freezes `L_ref` may P1 rebuild coordinates with unchanged
`x_H,x_L,c,G` and use the unique ratio `s=target/L_ref`, never a scale sweep.
P1 compares Native, physical, index and deterministic YaRN; an old config-based
table may be a labeled mis-reference control. P2 then uses the closest
same-generation Qwen pair, each with a Native resolver and nonzero YaRN2.
P3 may diagnose signed finite Q/K distortion, but layer/head/distance are
diagnostic strata, not parameters. Selector and expanded SOTA baselines wait.

Cell-average/P3 residual remains closed. No new `G(x;K)`, per-head/per-layer
curve or benchmark-derived residual is authorized by these controls.

## D. One measurement repair after the first P0 abstention

The first calibration completed and retained its `ABSTAIN` verdict. The
capability instrument failed even in compact contexts. Raw-output inspection
identified a specific measurement confound: most short-context failures contain
both requested codes in order but violate the two-answer separator/period
format. Those are still failures under the original exact contract; the old
scores are never changed or used to force a 4K reference.

Exactly **one** new instrument is preregistered before further GPU work:

- eight records, one queried code and seven distractors;
- fresh seeds `202609021/202609022`, 64/128 independent blueprints;
- query positions cycle uniformly over all eight record positions; depth is
  stratified into four fixed bins;
- complete six-digit code plus period and terminal EOS after `The code is `,
  with only outer whitespace normalization and no substring scoring;
- same length grid, Native model, practical margins, bootstrap/CP rules and
  independent-confirmation decision; no frequency or gain intervention;
- reuse the already frozen natural **input** documents, keeping confirmation
  model outcomes unopened. Re-evaluated calibration natural rows are repeats,
  not additional independent documents;
- first run compact controls, then 1024. If either point estimate is below
  `.75`, stop this instrument immediately, abstain, and do not change another
  prompt or score rule;
- only if those controls pass, finish the same frozen grid and apply the
  original common-frontier rule. Disagreement still means abstention.

This is not a claim that the original two-code task was repaired: query count
and multi-target interference also decrease. An accepted reference would be
explicitly specific to natural continuation plus **single-code retrieval**.
No original result is upgraded, no long holdout chooses the repair, and no
additional probe search is registered.

K, model family, head dimension, learned coefficients and training history
are confounded in the current checkpoints. Same-family Qwen K32/K64 reduces
some confounds but cannot identify the causal effect of K or training
exposure. No new model download is currently necessary, and the protected
1.485B checkpoint must remain intact. No automatic shutdown is scheduled
for this newly authorized session.

## Evidence ceiling

This is a preregistration, not a completed P0 GPU result. Frozen `G` and a
scale-consistent formula do not by themselves prove
behavioral universality, Native compatibility or SOTA. No third rescue
candidate, parameter sweep, long-label fit, or hierarchical table is opened.
