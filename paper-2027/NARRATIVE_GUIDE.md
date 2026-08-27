# ICLR 2027 narrative and revision guide

> **Purpose.** Keep every local manuscript edit serving one reviewer judgment:
> finite RoPE tables have an identifiable interior-allocation coordinate that
> changes the effective positional basis and matters in the tested protocols.
> This guide governs narrative and revision discipline only. It is **not** a
> fourth facts, numbers, or evidence authority. For those, follow
> [`../AGENTS.md`](../AGENTS.md) -> [`../INDEX.md`](../INDEX.md) ->
> [`HANDOFF.md`](HANDOFF.md) -> the routed canonical owner.

## Objective and reviewer memory

The objective is to maximise ICLR 2027 acceptance probability: make the
scientific object, exact theory, and evidence strength understandable quickly.
Do not turn the paper into a sentence-by-sentence defence of a historical
review.

> **Reviewer memory:** A finite RoPE table is not exhausted by base or range:
> at fixed sampled support, its interior allocation is an identifiable,
> controllable design coordinate for the effective positional basis and for
> long-context behaviour in the tested protocols.

Use this chain before adding or moving prose:

1. **Claim:** write \(x_k=-\log\omega_k=a+Rz_k\). Support \((a,R)\) and
   interior allocation \(z\) are independent coordinates of a finite table.
2. **Evidence:** full sin/cos subspace geometry gives the exact
   effective-dimension account; a fixed-support three-seed intervention
   identifies the behavioural effect; three protocol-separated routes test
   persistence through the model lifecycle.
3. **Implication:** with the same \(K\), support, and RoPE operator, tables can
   form different effective positional bases. EVQ-Cosh is one closed-form,
   zero-learned-parameter witness on that axis.

State the conclusion first, then attach the nearest material scope clause. Do
not make an unproved limitation the grammatical subject of a paragraph.

## The scientific object and novelty boundary

`z` numerically changes interior frequencies. Never call the method “not
changing frequencies.” The question is how a finite set of frequency samples
allocates phase resolution **within fixed support**, and hence what effective
positional basis a nominal rotary budget actually provides.

- A scalar-base/range method changes support \((a,R)\), realised phases, or
  range transport while retaining a geometric path. A pinned-support change in
  `z` cannot be recreated by another scalar base.
- Whole-vector or broader positional methods change more than a single fixed
  RoPE table. Do not flatten those operator differences away.
- Learned-table work such as LeRoPE or AdaRoPE is relevant allocation work, not
  a validation of this paper's mechanism or a matched comparator.
- `z` is not an extrapolation-only knob. The strongest completed controls are
  long-context controls, but the design coordinate reallocates in-window and
  out-of-window representation alike. A closed-form zero-parameter
  window-optimal allocation remains future work, not a completed result.

## What the paper may say

Strong, accurate formulations include:

- `z` is a third independent design coordinate beyond support location and
  span in a finite RoPE table.
- It is an identifiable and controllable determinant of positional geometry
  and of long-context behaviour **in the tested protocols**.
- Nominal rotary dimensions can collapse to far fewer effective positional
  dimensions under the stated measure.
- The same \(K\), sampled support, and operator can yield different effective
  positional bases.

Keep the epistemic layers explicit: geometry diagnoses the basis; trained and
controlled protocols establish behavioural consequences; EVQ-Cosh supplies a
minimal analytic construction. The construction is unique **only** for its
stated convex surrogate.

Never claim or imply that `z` is necessary, sufficient, the only mechanism of
extrapolation, universally better, SOTA, globally optimal, or near-optimal.
Never present static effective rank as a monotone predictor of LM quality;
slow bands as dead, unused, or reclaimable; or LeRoPE as validation of EVQ.
Do not call the 1.485B row converged or multi-seed, or present the 50.9M to
1.485B collection as a scaling law.

## Causal attribution and synergy contract

The paper's central empirical attribution must come from a **pure-`z`
contrast**.  Between the compared arms, hold sampled support/base, the RoPE
operator, checkpoint or training contract, attention gain, routing, data, and
evaluation fixed; change only interior allocation `z`.  The resulting frequency
vector changes because `z` changes.  That is the intended intervention, not a
confound.

Two completed anchors own this causal claim:

- the 151.9M fixed-support three-training-seed comparison identifies `z`
  during co-adapted training;
- the mature OLMo/Qwen same-support frozen controls hold checkpoint, support,
  gain, routing, rows, and decoder fixed while changing `z`, showing that the
  coordinate remains consequential after pretraining.

Once the non-marginal pure-`z` main effect is established, experiments may
combine `z` with adaptation, range transport, gain, routing, or other non-base
components and report the resulting **combined or synergistic method
contribution**.  They do not need to decompose every added component merely to
show that the combined system improves.  A matched marginal or factorial
interaction is required only when the prose assigns a particular share of the
gain to one added component or claims a component-specific causal interaction.
Keep base/support fixed for the paper's `z`-centred combined-method claim.  If
base/support also changes, do not describe the total improvement as a pure
allocation effect; route the pure effect to the anchors above and state the
combined intervention separately.

This is the disentanglement contract requested by the historical NeurIPS
panel: isolate allocation from base/range, parameterisation, tuning effort,
and deployment machinery before assigning causality.

## Keep FMRoPE in its correct place

FMRoPE is historical reviewer context, related work, and the paper-faithful
fixed-support protocol control. It is not this paper's main question, an
opponent, or a separate narrative axis. Do not restart a body-prose loop about
“fixed-support win versus target-matched loss.” The fixed-support control
establishes the identified coordinate; the target-matched result is a
deployment boundary and belongs in the required appendix/protocol context,
not in the paper's remembered main story. Never write “we beat FMRoPE” or
“we beat YaRN.”

## Evidence routes: preserve their identities

Do not pool metrics, uncertainty, or causal readings across these routes.

| Route | Canonical route(s) | Strongest reviewer-facing claim | Boundary that must travel with it |
| --- | --- | --- | --- |
| Zero-training frozen intervention | [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823`](research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md); session/fresh-natural owners routed by [`research/attention-aware-retrofit/README.md`](research/attention-aware-retrofit/README.md) | Holding the tested frozen checkpoint, support, amplitude, rows, and decoder fixed while changing `z` shows that allocation remains consequential after pretraining. | It is a per-checkpoint deterministic intervention with evaluation-row uncertainty, not training-seed or population uncertainty. Frozen derived and movement-profile-ramp profiles are **not** EVQ-Cosh; Native and official YaRN are reference rows, not substitutes for the pure-`z` control. |
| Matched adaptation | [`OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729`](../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md); [`EVQ_8B_ADAPTATION_EVIDENCE_20260724`](../rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md) | Matched adaptation supports task-family length transfer at 1.485B and remote-source causal use in the tested 8B adapted model. | Adaptation-only evidence; retain its single-trajectory and endpoint scope. RULER/2Wiki are task-family adaptation, not unseen-task transfer; 8B is not pretraining-scale evidence. |
| From-training / co-adapted | [`EXACT_RANGE_151M_3SEED_RESULT_20260820`](research/EXACT_RANGE_151M_3SEED_RESULT_20260820.md); [`table18_mla_3seed_aggregate.json`](../data/curated/table18_mla_3seed_aggregate.json); routed 750M/1.485B owners in [`research/README.md`](research/README.md) | Fixed-support three-seed training identifies `z`; architecture and full-parameter studies repeat the qualitative long-context crossover across their own protocols. | 50.9M is configuration/shape breadth, not the first point of a scale series. The 1.485B row is a pre-specified early-training, same-initialisation/same-scientific-recipe trend, not converged, bitwise paired, or multi-seed scale evidence. |

The 50M weights-by-table crossing explains why these routes cannot be spliced:
weights co-adapt with the installed table, so static geometry alone does not
order frozen-model LM quality. The exact frozen-transplant theorem has its own
scope: it blocks exact position-independent invertible Q/K compensation for
unequal frequency multisets; it does not block approximate adaptation or new
operators.

## Body allocation and edit workflow

The nine-page body should preserve this reading order:

1. finite-table coordinate and counterintuitive spectral-budget observation;
2. fixed-support identification of `z`;
3. related-work classifier and exact full-pair theory;
4. EVQ-Cosh as a bounded analytic witness;
5. three labelled evidence routes, then practical implications.

Foreground the theorem--identification--lifecycle loop. Compress repeated
history, reviewer rebuttal framing, metric ledgers, and protocol detail.
Route FMRoPE target-matching detail, long implementation provenance, extra
figures, and proofs to the appendix where they retain their evidential role.
Do not remove sound appendix theory merely to create apparent body space.

### Pre-edit checklist

- Read the reviewer-memory sentence and the claim -> evidence -> implication
  chain above.
- Read the canonical owner before touching a number, metric, protocol, or
  interpretation.
- Identify which one of the three evidence routes the edit belongs to.
- Ask whether the edit strengthens the main reader path, or merely litigates a
  historical local concern. Route the latter to related work or appendix.

### Post-edit checklist

- Does the 30-second reading still recover finite table -> `z` -> exact
  geometry -> controlled identification -> three lifecycle routes?
- Are the conclusion and scope clause adjacent, with contribution leading?
- Did the edit preserve `Geo`/`Native`/`FMRoPE`/anchored EVQ-Cosh/EVQ-Cosh
  identities and avoid mixing the three evidence routes?
- Did it avoid every forbidden inference above and keep Cosh, static geometry,
  1.485B, 8B, and frozen profiles within their owner-defined scope?
- If the edit changes a fact rather than presentation, stop and return to the
  canonical owner; do not silently update the science to fit the prose.

## Authority boundary

This guide is a revision guardrail, not a replacement for repository authority.
`AGENTS.md` supplies rules and claim ceilings; `INDEX.md` supplies durable
routing; `HANDOFF.md` supplies live state; the routed raw/research owner
supplies numbers and protocol identity. On conflict, do not revise science from
this guide—verify the canonical owner and update this guide later only when
the owner-supported narrative has changed.
