# ICLR 2027 narrative and revision guide

> **September 2026 status.** This is the active, durable manuscript narrative
> guardrail for the 9/17 internal abstract freeze and 9/25 full-paper submission.
> It is not a facts authority, numerical owner, research agenda, receipt store,
> or live action queue. Current state lives only in [`HANDOFF.md`](HANDOFF.md).

## Authority and use

Before editing the manuscript:

1. follow [`../AGENTS.md`](../AGENTS.md) for rules and claim ceilings;
2. use [`../INDEX.md`](../INDEX.md) to locate the canonical owner;
3. read the current source/PDF and [`HANDOFF.md`](HANDOFF.md), rather than an old
   review locator or plan;
4. apply this guide to presentation only after the fact, protocol, theorem, and
   evidence role are verified.

External-model reviews, historical verdicts, and old revision plans are
adversarial inputs. They may identify a current defect, but they do not authorize
an edit or upgrade evidence.

## Objective and reviewer memory

The objective is to maximise ICLR 2027 acceptance probability by making the
scientific object, exact theory, and evidence strength understandable quickly.
Do not turn the paper into a sentence-by-sentence defence of a historical review.

> **Reviewer memory:** A finite RoPE table is not exhausted by base or range. At
> fixed sampled support, its interior allocation is an identifiable,
> controllable design coordinate for the effective positional basis and for
> model behaviour in the tested protocols.

Use this chain before adding or moving prose:

1. **Claim:** write \(x_k=-\log\omega_k=a+Rz_k\). Support \((a,R)\) and
   interior allocation \(z\) are distinct coordinates of a finite table.
2. **Evidence:** full sin/cos geometry gives the exact basis account;
   fixed-support interventions identify the behavioural effect; three
   protocol-separated stages test its consequences through the model lifecycle.
3. **Implication:** with the same \(K\), support, and RoPE operator, tables can
   form different effective positional bases. EVQ-Cosh is a closed-form,
   zero-learned-parameter construction on that coordinate.

State the conclusion first, then attach the nearest material scope clause. Do
not make an unproved limitation the grammatical subject of a paragraph.

## Scientific object and novelty boundary

`z` changes interior frequencies. Never call the method “not changing
frequencies.” The question is how a finite set of frequency samples allocates
phase resolution **within fixed support**, and hence what effective positional
basis a nominal rotary budget provides.

State the field-level contrast precisely without turning Related Work into a
taxonomy. Widely used RoPE extensions act through a scalar base, range, phase
transport, frequency-dependent transforms, learned tables, or combinations of
these. The contribution is the explicit separation of support and interior
allocation, fixed-support causal identification of allocation, exact geometry
of the resulting basis, and a closed-form construction on that coordinate.

- A scalar-base change moves the geometric support path while leaving its
  normalised allocation uniform. Frequency-dependent transport can induce a
  non-uniform realised `z`; that does not erase the fixed-support distinction.
- Whole-vector or broader positional methods change more than a single fixed
  RoPE table. Preserve those operator differences.
- Learned-table work such as LeRoPE or AdaRoPE is relevant allocation work, not
  mechanism validation or a matched comparator for this paper.
- `z` is not defined as an extrapolation-only knob. The paper may state the
  broader coordinate implication only at the strength supported by completed
  evidence.

## What the paper may say

Strong, accurate formulations include:

- `z` is a third finite-table design coordinate beyond support location and
  span;
- at fixed support, changing `z` changes the positional basis and trained-model
  behaviour in the controlled protocols;
- nominal rotary dimensions can collapse to far fewer effective positional
  dimensions under the stated measure;
- the same \(K\), sampled support, and operator can yield different effective
  positional bases.

Keep the epistemic layers explicit: geometry diagnoses the basis; controlled
trained-model protocols establish behavioural consequences; EVQ-Cosh supplies
the analytic construction. The construction is unique **only** for its stated
convex surrogate.

The complete forbidden-claim ceilings live in `AGENTS.md`. In particular, do
not imply that `z` is necessary, sufficient, the only extrapolation mechanism,
universally better, SOTA, globally optimal, or near-optimal. Do not present
static effective rank as a monotone LM-quality predictor; slow bands as dead or
reclaimable; LeRoPE as validation of EVQ; the 1.485B result as converged or
multi-seed; or the collection of model sizes as a scaling law.

## Causal attribution and synergy contract

The central empirical attribution comes from a **pure-`z` contrast**. Between
the compared arms, hold sampled support/base, the RoPE operator, checkpoint or
training contract, gain, routing, data, and evaluation fixed; change only
interior allocation `z`. The frequency vector changes because `z` changes. That
is the intended intervention, not a confound.

Once the non-marginal pure-`z` effect is established, experiments may combine
`z` with adaptation, range transport, gain, routing, or another component and
report the combined system. They may not assign a particular share of the gain
to one component without a matched marginal or factorial interaction.

Preserve the three evidence stages without pooling metrics, uncertainty, or
causal readings:

1. **Fully frozen intervention:** allocation remains consequential after
   pretraining under a deterministic per-checkpoint intervention.
2. **Matched adaptation:** pretrained representations can exploit a changed
   basis under a separately scoped adaptation protocol.
3. **From-training / co-adapted:** fixed-support training identifies `z`, while
   architecture, scale, and modality studies retain their own protocol roles.

Current owners, results, and maximum claims for these stages live in
`INDEX.md` §3 and the routed claim-level owners. Do not copy their changing
numbers or status into this guide.

The weights-by-table crossing explains why the stages cannot be spliced: weights
co-adapt with the installed table, so static geometry alone does not order
frozen-model LM quality. The frozen-transplant theorem has its own exact scope
and does not block approximate adaptation or new operators.

## FMRoPE and neighbouring methods

FMRoPE is related work and the paper-faithful fixed-support protocol control. It
is not the paper's opponent or a separate narrative axis. The body may state the
support-retargeting result once to establish that support and allocation are
distinct but interacting; the appendix and canonical owner retain the protocol,
numbers, and per-seed detail. Do not expand it into a winner/loser loop. Never
write “we beat FMRoPE” or “we beat YaRN.”

Related Work is a compact attribution section. Its job is to locate the
fixed-support `z` estimand and the construction boundary, not to teach every
range method or reproduce an external review's comparison matrix.

## Body allocation and edit workflow

Preserve this reading order:

1. finite-table coordinate and counterintuitive spectral-budget observation;
2. fixed-support identification of `z`;
3. exact full-pair theory and EVQ-Cosh as a bounded analytic construction;
4. fully frozen, matched-adaptation, and from-training consequences;
5. compact related-work positioning and the field implication.

Foreground the theorem–identification–lifecycle loop. Compress repeated history,
reviewer-rebuttal framing, metric ledgers, and protocol repetition. Route long
implementation provenance, extra figures, and proofs to the appendix without
removing sound theory merely to create apparent body space.

### Pre-edit questions

- What exact current passage is defective or low leverage?
- Which canonical owner governs the proposed change?
- Which evidence stage does it belong to?
- Does the edit strengthen the main reader path, or merely litigate an old review?
- Can the same outcome be achieved by replacing or deleting less useful prose?

### Post-edit questions

- Does the 30-second reading recover finite table → `z` → exact geometry →
  controlled identification → lifecycle consequences?
- Are conclusion and necessary scope adjacent, with the result leading?
- Are `Geo`, `Native`, `FMRoPE`, anchored EVQ-Cosh, EVQ-Cosh, YaRN-style, and the
  MLA wavelength-blend operator kept distinct?
- Are likelihood, capability, seed, row, and trajectory units still separate?
- If the edit changes science rather than presentation, was the canonical owner
  updated first?

## September boundary

The September submission uses completed evidence. Do not admit an old panel
item, theory exploration, or experiment design merely because it appears in an
archived document. New submission compute is outside the active revision scope;
post-submission research remains routed by `INDEX.md` §6 and requires separate
authorization.

Current tasks, freeze progress, validation receipts, and author decisions belong
only in `HANDOFF.md`. This guide changes only when the owner-supported narrative
contract itself changes.
