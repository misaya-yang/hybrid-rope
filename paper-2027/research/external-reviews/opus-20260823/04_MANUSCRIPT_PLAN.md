# Manuscript plan

- **Date:** 2026-08-23
- **Status:** PLAN, conditional on experiments that have not run. Nothing here
  authorises a manuscript edit.
- Governing constraints: `AGENTS.md` §2 (claim ceilings, nine-page budget,
  locked nomenclature) and `HANDOFF.md` §3 (implemented state).

## 1. The gate

**Do not touch the manuscript until E1 (`03`) has run.** The current PDF is
built, validated, and internally consistent (9 body pages, 30 total, 0 undefined
refs, 223 focused tests). Inserting the retrofit today would add a second
analysis unit (inference-time operator on a frozen checkpoint) with a weaker
claim ceiling and a large attack surface, in exchange for evidence whose
strongest numbers sit on a synthetic retrieval assay.

E1 costs ~10 GPU-minutes and determines which of three plans applies.

## 2. Plan A — E1 shows geometric ≪ budgeted at fixed support

This is the outcome worth writing.

**What it buys.** The paper's central claim currently rests on training-time
identification (151.9M exact-range, three seeds). Plan A adds a second,
independent instance of the *same* identification on a **released 1.485B
checkpoint with zero training and zero learned parameters**. Same coordinate,
same $(a,R)$ held fixed, same $z$ varied, different regime. That is exactly the
kind of second pillar that moves a borderline score, and it costs no pages if it
displaces weaker material.

**Where it goes.** One subsection at the end of the experiments, ≤ 3/4 page,
displacing lower-leverage material — not stacked on top (`AGENTS.md` §2).
Candidate for displacement: part of the mature-studies discussion, whose
protocol-specific hedging currently consumes space without carrying a claim.

**One figure panel, not a table.** Rotation count on the $x$ axis, per-channel
frequency ratio $\omega^{\text{native}}_k/\omega'_k$ on the $y$ axis, three
curves (geometric flat at the stretched endpoint, YaRN ramping 32→1, budgeted
ramping ≈16→≈7), with the resulting RULER-13 macro annotated on each. That single
panel carries the whole argument: same endpoints, different interior, order-of-
magnitude different outcome.

**Draft claim ceiling** (do not exceed):

> At a fixed spectral range and a fixed attention temperature, the interior
> allocation of a released checkpoint's RoPE table decides whether that
> checkpoint retains long-range retrieval beyond its native window, with no
> training and no learned parameters. On OLMo-2-1B-Instruct, three tables sharing
> the same two frequency endpoints span RULER-13 macro from `X` (geometric) and
> `0.2382` (the published NTK-by-parts split) to `0.6772` (a split derived from
> the redundancy accounting of §T) at $2\times$ the native window. The same
> derived construction, frozen and transferred without re-tuning, improves
> core-4 RULER on Qwen2.5-1.5B-Instruct at $2\times$ its own 32K window
> (`0.6700` vs `0.6025` for the same one-deployment YaRN control).

**Mandatory accompanying sentences.** These are not optional hedges; omitting any
of them makes the passage indefensible:

1. The band split is the operative quantity, and a plain linear ramp placed at
   the derived split reproduces the table to within `0.009` mean $|\log_2|$
   (OLMo). State it; do not let a reviewer discover it.
2. RULER is a retrieval assay, not unseen-task transfer (existing ceiling).
3. On real documents the picture is heterogeneous: Qasper `+0.065` (CI excludes
   zero), 2WikiMQA `+0.010` (CI includes zero), six-task $4\times$ macro
   `−0.006` (CI includes zero).
4. Native-path preservation is a construction property, and on 2WikiMQA it
   *costs* `0.0108` F1 relative to applying the long table everywhere.
5. Single checkpoint per model; deterministic operators, so the reported
   intervals are evaluation-sampling only.

## 3. Plan B — E1 shows `nearest_yarn_ramp_s4` ≈ budgeted, geometric ≪ both

Still publishable, smaller, and it must be written honestly.

**Reframe to:** the spectral-budget accounting *predicts where the band split
belongs*; the published constants place it about seven times too low; the
consequence on a mature checkpoint is `0.2382 → 0.6772`. The operator family is
YaRN's. Say so in the first sentence.

**Placement.** Not a method subsection. A short empirical corollary inside the
theory section — the redundancy measure makes a quantitative, transferable
prediction and the prediction is checkable on released checkpoints. Half a page.

**Forbidden under Plan B:** any language implying a new operator, a new method, a
new family, or a deployment recommendation. Nomenclature must not invent a name
for the table.

## 4. Plan C — E1 shows geometric ≈ budgeted

Nothing enters the manuscript. Record the negative in `research/` alongside the
LeRoPE-oracle and $\kappa_{\text{att}}$ falsifications, and stop the line. The
zero-training work remains a legitimate internal engineering result with no
paper-facing claim.

## 5. Changes required regardless of outcome (owner hygiene, zero GPU)

These are defects in existing internal material, independent of any experiment.

1. **Re-lead the owner.** `SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md` §1
   and `HANDOFF.md` §6 both open with "versus official Transformers YaRN
   configured once at factor four". Replace with the fixed-support,
   fixed-amplitude framing. The science is unchanged; the current wording is what
   invites the "you just beat a badly tuned baseline" reading.
2. **Give full-200 Qasper an owner.** It is complete, matched, and the strongest
   natural-document evidence in the line, and it appears nowhere (`01` §3).
3. **Correct the Qasper counterexample sentence.** The owner currently says
   "Qasper is the clear counterexample" on the basis of a 20-row $2\times$ bucket
   (`0.1056` vs `0.1485`). At full 200 rows and a 16K window the direction
   reverses (`0.2457` vs `0.1803`). Different estimands — say both, claim
   neither as *the* Qasper result.
4. **Stop presenting the order crossings as a property.** Replace
   `order_crossings_retained: true` framing with the artefact finding (`02` §4),
   and record that `max_points` is not scale-free.
5. **Downgrade the two null results in prose.** `+0.0097` on 2Wiki and `−0.0063`
   on the $4\times$ macro are indistinguishable from zero; both currently read as
   outcomes.
6. **Label core-4 as the selection set** wherever `0.7175 / 0.4075` appears. The
   OLMo confirmation numbers are the unseen-9 row.
7. **Disclose the YaRN factor-2 control.** `0.5375` at 8K on identical data and
   checkpoint. Matched-operator comparison at fixed $s$ is the right design, but
   suppressing the length-matched baseline is what gets a paper accused of
   cherry-picking.

## 6. Nomenclature

If anything is promoted, extend the locked table in `AGENTS.md`:

| Term | Use |
| --- | --- |
| `same-support geometric` | the $\theta'=\theta\,s^{d/(d-2)}$ arm; note its identity with published NTK-aware base scaling on first use |
| `derived band split` | the $\approx16\to\approx7$ rotation thresholds obtained from the redundancy measure |
| `published band split` | YaRN's $\beta_{\text{fast}}=32,\ \beta_{\text{slow}}=1$ |

`Native` keeps its existing meaning (unmodified pretrained checkpoint) and is
never interchangeable with `Geo`. Do not name the budgeted table as a method
under Plan B.

## 7. Explicitly out of scope

- Any claim of cross-model universality from two checkpoints.
- Any 128K claim until E2 lands.
- Any statement about Qwen in-window behaviour until E4 lands.
- Any deployment or systems framing. This is identification evidence; the
  serving policy is the vehicle, not the contribution.
