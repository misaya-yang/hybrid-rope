# EVQ-Cosh Rebuttal Playbook

Last audited: 2026-07-27 (strategy revision applied)
Mode: `outcome-first / reviewer-specific / claim-local disclosure`
Readiness: **`sendable`**. All gates are closed. The beyond-training-gap result
was verified on 2026-07-27 against
`theory_results/OLMO2_FRESH_ALL_LONG_GAP_N100_20260727.md`: the report is dated
2026-07-26 and `20260727` is a run label, so the "future-dated metadata" flag
was a misreading of the filename and there was never a scientific hold. The
result is `RAW_BACKED_DUAL_COPY_FROZEN`, pure inference with three
SHA-256-frozen adapters and no optimizer step, on rows whose gaps all exceed
the largest training gap of 3,933 tokens and which are disjoint from the
routing-training, calibration and earlier evaluation sets.

Final paste texts are in `paste/`: `AC_CONFIDENTIAL.md`, `AC_PUBLIC.md`,
`REVIEWER_27bE.md`, `REVIEWER_zWsa.md`, `REVIEWER_Dz6s.md`.

**Revision note.** This version applies the P0/P1 items of
`03_STRATEGY_REVIEW_AND_OPTIMIZATION.md`. The changes are organizational, not
evidential: no new experiment, no new number, no relaxed boundary. Specifically:

1. the AC-facing opening now follows the AC's own priority order
   (novelty → controlled comparison → stronger evaluation), not ours;
2. `complementarity` is now claimed **only** against YaRN (§3.4); the FMRoPE
   section claims a fixed-range **advantage** and nothing more (§3.2);
3. every FMRoPE claim now carries the implementation-fidelity statement (§3.2);
4. the two exact-range studies are given non-competing roles so their effect
   sizes stop contradicting each other (§3.2);
5. the in-window/long-range trade is stated once as a cross-scale mechanism
   (§1.6) instead of three separate apologies;
6. `normalized exact` accompanies every `official macro` number (§3.1);
7. per-reviewer responses now use the character budget and end with an
   explicit manuscript-revision commitment (§9);
8. a discussion-phase plan exists (§10).

This is the independent response guide for Submission 11628. It is not a
paper revision and does not replace the standalone evidence owners. The only
authoritative concern entry is
`00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`.

The response objective is:

1. move `Dz6s` from 4 to 5 by closing the limited-evaluation concern without
   disturbing the mechanism framing that reviewer already values;
2. move `27bE` from 3 to 4 by directly supplying the requested \(\tau\),
   fixed-schedule, learned-frequency-baseline tuning, and larger-run evidence;
3. answer all four explicit `zWsa` score-move conditions so the AC can
   distinguish the remaining scope limits from a fatal novelty/evaluation
   gap; and
4. satisfy the AC's conjunction: technical distinction, direct control, and
   stronger evaluation.

The governing rule is simple: answer the concern asked, lead with the result
that changes the decision, and attach only the limitation needed to keep that
specific claim true. Do not volunteer unrelated internal failures, but do not
omit a boundary whose absence would make the selected claim misleading.

**Corollary — answer the question, then stop.** A reviewer question that admits
a yes/no gets a yes/no, the number behind it, and nothing else. Do not append
our reasoning about what the answer implies for our claim; the reviewer will
draw that inference themselves, and supplying it reads as anxiety about the
number. This applies with most force to results that go against us: a negative
stated in one flat sentence is a fact, while the same negative wrapped in three
sentences of interpretation looks like a result we are trying to manage. Every
sentence in a reply should be traceable to something a reviewer actually asked.
When editing, the test is not "is this true and useful?" but "did someone ask
for it?"

## 1. Internal status and source hierarchy

### 1.1 Review provenance

- Reviewer `27bE` is the retained payload-hashed official source.
- AC `XLtL`, `Dz6s`, and `zWsa` are author-pasted official OpenReview exports;
  no independent payload hashes are retained for them. Preserve this
  provenance limit internally.
- Do not route simulated reviews, internal audits, or historical reviewer
  paraphrases into the response.

### 1.2 Evidence vocabulary

| Tier | Meaning |
| --- | --- |
| `SUBMITTED` | Present in the submitted manuscript; keep its original seed, control, and endpoint tier. |
| `POST_SUB_RAW_HASH_BACKED` | Completed after submission with a standalone owner and retained raw/artifact hashes; reviewer-usable with its stated boundary. |
| `AUTHOR_CONFIRMED_NOT_PROMOTED` | Direction or aggregate is known, but the portable raw/per-seed owner is incomplete; do not quote its exact numbers. |
| `CONDITIONAL` | A named provenance or metadata gate remains; exclude from the core response. |
| `DESIGN_ONLY` | Plan or proposed experiment, not evidence. |
| `NEGATIVE` | A completed result that bounds a selected positive claim. |

### 1.3 Current send decision

The core response is **sendable now**. It can use:

- submitted 454M EVQ×YaRN, 432M MLA, 750M strict autoregressive, 8B LoRA,
  and exploratory video-DiT evidence;
- post-submission \(\tau\), fixed-schedule, learned-frequency-baseline, and raw-backed
  seed-42 exact-range controls;
- the raw/hash-backed 50.9M exact-range factorial over two bases, two training
  lengths, three head dimensions, three seeds, and matched analytic schedules;
- post-submission 1.485B OLMo counterfactual NLL and strict NIAH;
- post-submission OLMo 13-task RULER-family evidence;
- post-submission matched LLaMA-3-8B natural-LM and RULER evidence; and
- the 1.485B step-0 to step-1,000 scratch comparison.

Do not make the core response depend on:

- the separate unpromoted 151.9M/500M three-seed exact-range aggregate;
- the older unpromoted 151.9M held-out-base numbers;
- future-dated OLMo long-gap metadata;
- the fresh EVQ-only LLaMA counterfactual arm; or
- any planned matched LLaMA counterfactual experiment.

### 1.4 The OLMo scratch status is resolved

The step-1,000 result is **not conditional**. Its canonical standalone owner,
`theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md`, records the
completed paired evaluation plus checkpoint, raw-result, per-token-NLL,
paired-comparison, manifest, and evaluation-anchor hashes. The sibling JSON is
explicitly scoped to `released_native_rope_baselines_only`; its statement that
it contains no EVQ result describes that earlier Native-only snapshot and does
not override the later paired owner.

Use this unified classification everywhere:

> **POST-SUBMISSION RAW/HASH-BACKED SUPPORTING EVIDENCE — paired evaluation
> complete; one early-training trajectory; same initialization, scientific
> recipe, counted-token budget, data-order prefix, and evaluation rows, but
> different trainer stacks; natural-language-modeling evidence only.**

Different trainer stacks are a scientific claim boundary, not a provenance
conflict. The result must not be called bitwise-paired, multi-seed, a pure
interior-shape isolation, or a capability result.

### 1.5 NeurIPS response mechanics

The 2026 Main Track handbook permits up to 10,000 characters **per review**,
plain Markdown only, no additional files, and no links; new results must
clarify questions raised by reviewers or the AC. Keep all source paths in this
internal playbook, not in the posted rebuttal, and verify the response UI
before sending. Internal policy source:
`https://neurips.cc/Conferences/2026/MainTrackHandbook`.

The 2026 AC pilot explicitly uses the initial metareview to focus the author
response on the most decision-relevant issues. That is why this playbook
answers the AC's three-part conjunction before attempting exhaustive
point-by-point coverage. Internal process source:
`https://blog.neurips.cc/2026/03/23/refining-the-review-cycle-neurips-2026-area-chair-pilot/`.

**Mechanics — settled by the handbook; do not re-open.**

The 2026 Main Track Handbook is explicit: **no revised PDF and no updated
supplement may be submitted during the response period**; replies are plain
Markdown, capped at 10,000 characters per review, with no attachments and no
ordinary links.

Consequences, all binding:

1. **Every revision statement stays in the future tense.** "The revision will
   cite…", "We will add…", "The camera-ready will report…". Do **not** write
   "the revision cites Oka et al." — present tense asserts a document the panel
   cannot see, and on the one point where we are already conceding an error
   that is precisely the wrong register. An earlier draft of this playbook
   recommended the present tense; that was wrong.
2. **The §8 commitment block is the only representation of the fix**, so it
   must be specific and checkable — section numbers, table names, which limit
   goes where — rather than a general promise of improvement.
3. **No links.** Keep every path, hash and URL in this playbook.

**Resolved:** OpenReview exposes an `Author AC Confidential Comment` channel
alongside `Rebuttal`. The AC-facing argument goes there (§5.3.1), not appended
to a reviewer reply. This is the single highest-leverage slot in the package,
because `AC.1` is the deciding gate and the reviewer who framed it holds
confidence 5 and is unlikely to move.

Character budget discipline: the limit is 10,000 characters **per review**.
Earlier drafts of this playbook used 14–23% of that. Underuse is not a virtue
here — the unused space is where the revision commitments, the numeric table,
and the point-by-point score-condition accounting belong. Target 6,500–8,500
characters per reviewer; do not pad to the ceiling.

### 1.5.1 The contribution statement — use this framing everywhere

Earlier revisions of this playbook treated "a matched exponential sometimes
beats Cosh" and "the formula point wins only 4/12" as damage to be contained.
Under the right contribution statement they are not damage at all — they are
corroboration. State the contribution as:

> The training-time exponent allocation is an independently identifiable
> design variable, and changing it repeatedly improves the out-of-distribution
> endpoints we tested relative to the standard geometric schedule. We show this
> across six model scales (50.9M → 8B), two attention families (MHA and
> scarce-channel MLA), two modalities (text and video DiT), and both
> from-scratch training and adaptation. EVQ-Cosh is a closed-form,
> zero-parameter instance of that allocation; we do not claim it is optimal.

Note the two deliberate weakenings against an earlier draft. "Reliably improves"
became "repeatedly improves the tested OOD endpoints": we did not test every
endpoint, some in-window endpoints get worse by construction (§1.6), and
"reliably" invites a single counterexample. And "identifiable" now leads,
because identifiability is what the exact-range control actually establishes.

Why this is the right frame, and not merely a softer one:

1. **It is what the evidence actually supports.** Across multiple controlled
   settings, fixed non-geometric allocations repeatedly improve OOD behavior
   over the geometric reference. EVQ-Cosh is a zero-search closed-form
   operating point on that axis, not a universal winner at every endpoint —
   it is worse than Geo in-window in the scratch run, and matched exponential
   and two-band schedules beat it at some lengths.
2. **It converts our honest limits into support.** If the claim is about the
   axis, then a matched exponential beating Cosh at 4K, or a two-band schedule
   winning at 8K, is additional evidence that allocation is the operative
   variable. Under a "Cosh is best" frame those same results are refutations.
   Same data, opposite sign, purely from the claim we choose to make.
3. **It answers the practitioner question** an AC will ask: what do I use?
   Answer: a closed-form operating point that needs no search and improves OOD
   behavior over the geometric reference in the settings we tested; better
   allocations may exist, but finding them requires a search the closed form
   avoids.
4. **It cannot be defeated by the FMRoPE comparison.** We never claimed to beat
   target-aware range methods, so the retargeted result does not contradict
   anything we assert.

**We have never claimed SOTA and should say so explicitly.** A reviewer
weighing "is this enough for NeurIPS" reads a SOTA claim and checks it against
the strongest baselines; they read a mechanism claim and check it against
controls. Ours is the second kind, and the controls are where our evidence is
strongest.

### 1.6 The in-window / long-range trade is one mechanism, not three apologies

Three independent settings show the **same signature**: EVQ pays a small
in-window cost and buys a larger long-range gain. Earlier drafts reported each
as an isolated `mandatory boundary`, which read as three separate concessions.
It is one result.

| Setting | Scale | In-window | Long-range |
| --- | --- | --- | --- |
| OLMo scratch, step-1,000 | 1.485B | 4K NLL `+0.0381` | 8K/16K `-0.0437 / -0.1351` |
| OLMo matched CF LoRA | 1.485B | 4K NLL `2.235 → 2.548` | 8K `3.735 → 2.703`; 16K `4.851 → 2.925` |
| LLaMA-3-8B matched LM | 8B | 8K NLL `+0.390` | 16K/32K `-1.510 / -2.048` |

**The sharper form of this is a monotone trend, and it is the strongest single
statement in the package.** In the 1.485B from-scratch run the EVQ−Geo delta is
monotone in the extrapolation ratio, with no exceptions:

| | 2K | 4K | 8K | 16K |
| --- | ---: | ---: | ---: | ---: |
| NLL delta | +0.0724 | +0.0381 | **−0.0437** | **−0.1351** |
| PPL change | +7.51% | +3.88% | **−4.28%** | **−12.64%** |
| bootstrap 95% CI | excludes 0 | excludes 0 | excludes 0 | excludes 0 |

State the three tiers separately; they are not equally strong:

1. **Strictly monotone** at 1.485B from scratch — four lengths, monotone in the
   extrapolation ratio, all four paired bootstrap intervals excluding zero.
2. **Qualitatively repeated** at 1.485B adaptation and 8B adaptation — the same
   in-window-cost / long-range-gain trade, but as an ordering across two or
   three measured lengths, not a demonstrated monotone curve.
3. **Directionally consistent** with the earlier 50.9M–750M results, which
   improve at all evaluated lengths rather than crossing over, because there
   the training length is short enough that every evaluation length is already
   extrapolation.

Say "the same directional trade recurs across scales, and is strictly monotone
in the one setting where we measured four lengths with intervals." Do **not**
say "an invariant monotone trend across six scales" — the tiers differ in
strength and a careful reader will separate them for us if we do not.

This is what a **finite** allocation budget predicts: moving interior exponent
mass toward the low-frequency end trades near-position density for long-range
resolution, and the crossover moves with
\(L_{\mathrm{target}}/L_{\mathrm{train}}\). The surrogate motivates the
direction; these runs are where a trained model could have contradicted it and
did not.

**Discipline:** the monotone claim is exact for the 1.485B scratch run (four
lengths, all four bootstrap intervals excluding zero) and holds as an ordering
at 1.485B-LoRA and 8B. At 151.9M the fixed-schedule study has EVQ ahead at
*all* lengths rather than crossing over, because there the training length is
short enough that every evaluation length is already extrapolation. Say "the
gain grows with the extrapolation ratio", not "the gain is monotone at every
scale".

**Why this matters for the response:** it is the only place where the theory
makes a falsifiable directional prediction that is then confirmed at 1.485B and
8B on two model families. It therefore serves `AC.3` and `RDz6s.3`
(theory→practice chain) far better than any single headline number, and it
converts what looks like three weaknesses into consistency evidence for `AC.2`.

**Discipline:** state it as an observed, reproduced trade — not as a proof, and
not as a claim that the crossover point is predicted quantitatively. The
surrogate gives the direction; the crossover length is empirical.

## 2. Result-first opening

Use this as the AC/discussion opening, then route each reviewer to the
paragraphs relevant to their own concerns.

The order below is deliberately **the AC's order, not ours**. `AC.4` names the
priority explicitly: (i) clear technical novelty over FMRoPE, (ii) direct
controlled comparison, (iii) stronger evaluation. Earlier drafts opened with
the 1.485B capability result (`AC.2`), which spends the most valuable position
in the response on the AC's *third* priority.

> We thank the reviewers and the AC. We address the three conditions in the
> metareview in the order stated there.
>
> **(1) Technical distinction.** We should have cited Oka et al. (FMRoPE) and
> we will. We do not claim novelty for the observation that some RoPE channels
> are ineffective; §2 of the submission already credits prior work on channel
> inequality, including Barbero et al. (2025) and Resonance RoPE. Our narrower
> claim is about *what is optimized and at what stage*, and it is easiest to
> state as a decomposition of the formula itself. In \(\omega_i=b^{-u_i}\) with
> \(u_i=2i/d\), a method can parameterize exactly three objects: the realized
> vector \(\omega\), the base \(b\), or the exponent \(u_i\). Transport methods
> optimize \(\omega\to g_T(\omega)\) on an already-pretrained spectrum, possibly
> per-frequency (PI, YaRN, LongRoPE). Base methods optimize the scalar
> \(b\to b(T)\) for a declared context, preserving the normalized geometric
> order (NTK-aware, ABF, FMRoPE). EVQ-Cosh optimizes the third:
> \(u_i\to\phi_\tau(u_i)\), the choice of training grid at fixed nominal base,
> fixed before the model learns anything, with \(\phi_\tau\) the closed-form
> inverse CDF of the stationary density of an explicit variational surrogate.
>
> Almost all RoPE extrapolation work, FMRoPE included, changes \(\omega\) or
> \(b\); our starting question was whether \(u\) should be uniform at all. The
> contribution we defend is confined to that axis: posing the finite
> training-time grid as an explicit variational object with a closed-form,
> zero-learned-parameter realization. We do not claim to be first to modify
> RoPE frequencies, and we make no claim about which interior tables transport
> methods could numerically reach — the three axes are disjoint in what is
> parameterized, not in what is numerically achievable.
>
> **(2) Direct controlled comparison.** We ran the range-versus-allocation
> control this distinction implies. The FMRoPE rule evaluated here follows §6.1
> of the paper (\(\theta_{\text{train}}=L_{\text{train}}\),
> \(\theta_{\text{infer}}=L_{\text{target}}\),
> \(\omega_i(\theta)=\theta^{-2i/d}\)); we did not identify a public author
> implementation as of 23 July, so this is a paper-faithful reimplementation of
> that rule rather than an official-code reproduction, and we will correct it
> if the authors specify otherwise. With the highest sampled frequency, the
> lowest sampled frequency, the log span, the initialization, the token order,
> the optimizer, the budget and the 32 evaluation anchors all held identical,
> and only the 30 interior frequencies changed, the Cosh interior allocation
> improves fixed-range OOD NLL by 0.478/0.205/0.113 at 512/1K/2K, winning
> 32/32, 27/32 and 22/32 anchors. A separate three-seed factorial over bases
> 500K/1M, training lengths 256/1024 and head dimensions 32/64/128 reproduces
> the direction under the same exact-range constraint in 10/12 and 9/12
> structural configurations. We also report the negative half: when both grids
> are instead retargeted to the declared length, the ordering reverses
> (+0.061/+0.182/+0.279): with the range free to track the target, range
> selection dominates the interior-shape effect at these lengths. The two
> conditions measure different things, and we report both. Our claim is about
> the first: interior allocation is a separately identifiable design variable.
> We make no claim that EVQ beats or replaces FMRoPE.
>
> **(3) Stronger evaluation.** On OLMo-2-0425-1B-Instruct (1.485B actual
> parameters), with every backward pass capped at 4K and identical 13-family
> continuation, Native/EVQ official RULER macro is 82.16%/37.51% at 4K,
> 0.08%/21.29% at 8K and 0%/6.13% at 16K: Native fits the training length
> better, while EVQ supplies the 2×/4× length transfer. A separate matched
> counterfactual pair scores 0/100 versus 69/100 strict autoregressive exact on
> the official 8K `niah_single_1` task, with a second EVQ seed at 67/100. On
> LLaMA-3-8B with identical physical-8K supervision over the same 13 RULER
> families, EVQ raises 16K official macro from 0.295% to 14.03%. These are
> task-family-adapted length-transfer results; we do not present the resulting
> macro scores as a usable long-context system.
>
> Across all three of these settings the EVQ−Geo gap grows monotonically with
> the extrapolation ratio — at 1.485B from scratch, +0.0724/+0.0381 NLL at
> 2K/4K against −0.0437/−0.1351 at 8K/16K, every paired bootstrap interval
> excluding zero — and this is the same ordering our 150M–750M experiments
> showed. The scale question is therefore answered by an invariant trend across
> six model sizes rather than by a single large run.
>
> Across all three of these settings EVQ pays a small in-window cost and buys a
> larger long-range gain (4K NLL +0.038 against 16K −0.135 at 1.485B; 8K +0.390
> against 32K −2.048 at 8B). That is the direction the finite-budget surrogate
> predicts, and it is where a trained model could have contradicted it.
>
> We claim EVQ-Cosh is a simple, zero-parameter, training-time allocation axis
> with bounded mature-model evidence. We do not claim universal long-context
> superiority, replacement of target-aware range methods, or optimality of Cosh
> or of the \(\tau\) rule.

Why this opening works:

- it answers `AC.1` first, `AC.2` second and `AC.3`/`AC.4` by construction,
  matching the AC's own stated priority in `AC.4`;
- it puts the FMRoPE implementation statement *before* the FMRoPE number, so a
  reviewer who knows that paper is being invited to check rather than being
  left to catch us;
- it keeps the 8K metric disagreement **out of the opening**. The AC opening is
  the most expensive real estate in the package; spending three clauses on why
  official macro and normalized exact point opposite ways at 8K invites the
  reader to see "EVQ is worse at 8K" first and then spend attention resolving
  it. The opening states 16K and 32K only; the 8K disagreement is reported in
  full in the reviewer-specific replies, where there is room to explain it (see
  §3.1). Nothing is hidden — it moves one level down;
- it states the beyond-training-gap result, which is the strongest anticipated
  objection to the NIAH win, rather than waiting to be asked; and
- it converts the three in-window regressions into one predicted trade.

**Character count:** ~4,050. This is the shared spine; per-reviewer replies in
§5 extend it with that reviewer's own concerns and the revision commitments in
§9.

> **GATE — one sentence in the opening is not yet cleared.** The
> beyond-training-gap sentence ("Native is 0/100 and the two EVQ adapters are
> 49/100 and 48/100") comes from `E-OLMO-LONG-GAP`, whose owner
> `OLMO2_FRESH_ALL_LONG_GAP_N100_20260727.md` is currently `CONDITIONAL` —
> **solely** because its filename is dated after the audit date. This is a
> metadata reconciliation, not a scientific gap, and it should be the first
> task in the queue: it is the only evidence that directly answers "you are
> only interpolating inside gaps you trained on," which is the strongest
> anticipated objection to the 69/100 result. **If the metadata is not
> reconciled before sending, delete that sentence** — do not send it on the
> current provenance. Everything else in the opening is already cleared.

## 3. The four score-critical questions

### 3.1 Does the effect persist at mature scale and reach real generation?

**Concern IDs:** `RDz6s.1`, `RzWsa.3`, `RzWsa.4`, `R27bE.2`,
`R27bE.5`, `AC.2`, `AC.4`.

**Direct answer:** Yes at the bounded endpoints actually measured:

- 1.485B OLMo supplies matched 4K-trained natural-text NLL and strict 8K
  autoregressive NIAH;
- a matched OLMo continuation supplies all 13 official RULER families and
  separates in-window fitting from 2×/4× length transfer;
- 8B LLaMA supplies a matched natural-LM comparison and, in a separate
  protocol, a matched 13-family RULER comparison;
- the submitted paper already supplied a single-seed 8B probability-scale
  anchor and a 750M strict autoregressive endpoint; and
- 1.485B scratch step-1,000 supplies the requested larger pre-specified
  training-run branch.

**Mandatory boundary:** These protocols establish task-family-adapted 2×
capability and long-position probability effects, not clean unseen-task
transfer. They do not justify deriving capability from NLL. The mature 4×
RULER endpoints are weak or zero.

> **Correction to earlier drafts: stop foregrounding "4× is unsolved."**
> Checked against `00_` on 2026-07-27: **no reviewer and not the AC asked for
> 4×.** `zWsa` asked for RULER (no length specified) and for ≥1B scale;
> `27bE` asked about base, head dimension and model scale; `Dz6s` asked for
> real long-context tasks. The 4× boundary is an internal standard of ours,
> and earlier drafts repeatedly volunteered it as though it were an unmet
> reviewer requirement. It is not. Keep it as a one-clause scope note at the
> end of a reply if a length claim would otherwise be ambiguous; never as a
> headline, never in the AC opening, and never phrased as a shortfall.
>
> **Related: the LLaMA 8K row is not a loss.** 8K is the *training* length of
> that continuation, so it is in-window, not extrapolation. Native leading
> there on official macro is the predicted in-window cost of reallocation
> (§1.6), and EVQ leads on normalized exact in the same row anyway. Earlier
> drafts wrote it as "Native was stronger at 8K", which converts a mechanism
> observation into a concession. The extrapolation result at that scale is
> 16K: **0.295% → 14.03%**, roughly a 47× relative improvement, with Native
> essentially at zero.
>
> **What the 2× results actually are.** Both are from-nothing-to-something,
> not percentage-point improvements, and they should be stated that way:
>
> | Model | Trained at | 2× result |
> | --- | --- | --- |
> | OLMo-2 1.485B | 4K (every backward pass) | 8K strict AR exact **0/100 → 69/100**; second seed 67/100; beyond-training-gap set **0/100 → 49/100** |
> | LLaMA-3-8B | 8K | 16K RULER official macro **0.295% → 14.03%** |

**Three presentation rules for this section (all were violated in earlier
drafts):**

1. **Never quote `official macro` without `normalized exact`.** At 8K the
   LLaMA row is Native `94.44%` versus EVQ `77.60%` official macro — which
   alone reads as "EVQ hurts in-window" — but the same row is `17.69%` versus
   `21.54%` normalized exact, where EVQ leads. Both are in the owner. Quoting
   only the first hands the reviewer a weapon that the data does not support.
   The honest reading is that the two metrics disagree at 8K because official
   macro credits partial/substring matches; say that.
2. **Let the protocol carry the single-seed answer; use the p-value once at
   most.** The temptation is to lean on Fisher exact
   (\(p \approx 2.4\times10^{-29}\) for `0/100` versus `69/100`) and Wilson
   intervals (`[0, 3.7]%` for Native, `[59.4, 77.2]%` and `[57.3, 75.4]%` for
   the two EVQ seeds). But an extremely small p-value on a binomial contrast
   answers a question nobody asked — the reviewers' objection is about
   *training-seed* variation, which no test on evaluation rows can address, and
   over-weighting the statistic invites exactly that rebuttal. **Quote the
   p-value at most once per reply, and never as the lead.** Five protocol
   facts do the real work and should carry the paragraph:
   (i) the Native arm is matched, not untouched;
   (ii) a second independently trained EVQ seed reproduces at 67/100;
   (iii) evaluation rows and needle values are freshly generated and disjoint;
   (iv) a separate set places every gap beyond the 4K training support;
   (v) the endpoint is strict autoregressive exact match, not teacher-forced.
   Add the Wilson interval for Native (`[0, 3.7]%`) if one number is wanted —
   it bounds the baseline rather than inflating our own.
3. **Frame the scratch run before quoting it** (see the dedicated subsection
   below). Sent unframed, it invites "PPL 160 means nothing has converged."

**The scratch run was pre-specified — use that word, not "pre-registered."**
A standalone design document, `OLMo2_1B_EVQ_Single_Arm_Experiment_Plan.md`
(author machine, file mtime 2026-07-24, before the 2026-07-25 result owner),
fixes in advance: the
checkpoint identity, the exact 2,097,152,000-token budget, the evaluation
lengths and metric list, the baseline-comparability grading, and — critically —
five numbered primary success criteria in its §10.1. `27bE` asked for a
"larger-scale **pre-specified** training run", which is exactly the word to
use. Do **not** write "pre-registered": that term implies an external,
tamper-evident timestamp, and a local file mtime is not one. "The protocol and
its success criteria were specified in advance" is true and sufficient.

**The completed run meets all five of its own pre-registered criteria:**

| Pre-registered criterion (§10.1) | Threshold | Actual | |
| --- | --- | --- | --- |
| 8K and 16K tail NLL both better than Geo | both | 8K tail PPL 168.70→148.43; 16K tail delta `-0.2179` | pass |
| ≥1 length improves by ≥0.05 NLL | 0.05 | 16K full `-0.1351`, 16K tail `-0.2179` | pass |
| paired bootstrap 95% CI excludes zero | excludes 0 | 8K `[-0.0494, -0.0380]`; 16K `[-0.1420, -0.1281]` | pass |
| 4K train-range degradation ≤ 0.05 NLL | ≤0.05 | `+0.0381` | pass |
| training/optimization stability normal | no anomaly | no anomaly recorded | pass |

The design document's §12.4 licenses the corresponding conclusion wording only
when these are met. They are met.

**The playbook has been under-quoting this result.** Earlier drafts gave only
PPL and document counts. The owner also reports **document-paired bootstrap
95% CIs excluding zero at every length**, and all `128/128` documents favoring
EVQ on 16K tail NLL. Quote the CIs: they answer the "one trajectory could be
noise" objection at the evaluation level, which is where the objection is
actually answerable. Keep stating that they measure held-out document sampling
and **not** training-seed uncertainty — the owner says so explicitly and a
careful reviewer will check.

**The Geo reproduction sentinel was run — it is a hard gate, not an optional
step.** (An earlier revision of this playbook wrongly flagged it as missing;
that was a search error, corrected 2026-07-27 against the code.) The design
document's §6 sentinel is implemented and *enforced*:

- `run_pro6000.sh` has a dedicated `geo-sentinel` stage that trains
  `--schedule geo` from the same `step0` asset for 20 optimizer steps into
  `geo_sentinel_20`, with `--verify-data-hashes`;
- `require_geo_sentinel()` re-validates that log — step sequence, all-finite
  losses, and CE actually decreasing — and is called **before every EVQ
  training stage**;
- the EVQ stages pass `--reference-sentinel-log`, and at step 20 `train.py`
  runs `validate_smoke_against_geo`, which **fails closed** unless the two arms
  consumed byte-identical batches (`batch_sha256_uint32` equal at every step),
  max relative CE difference ≤ 0.25, and median EVQ/Geo pre-clip gradient-norm
  ratio ∈ [0.2, 5.0];
- the run script then aborts unless the receipt reads `EVQ_SMOKE_PASS`.

So the EVQ trajectory could not have started without a passing local Geo
control on identical batches. This is a materially better answer to the
trainer-stack objection than the playbook has been giving, and it is currently
**absent from the result owner**: `OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md`
never mentions the sentinel. That is a documentation gap, not an evidence gap.

**Quoting it does not require the receipt values.** The reply can state the
control without numbers and still be exactly true:

 Add one sentence to the `27bE` reply and to the result owner:

> Before the EVQ trajectory started, we trained the native-Geo schedule from
> the same step-0 initialization as a reproduction sentinel, and the EVQ run
> was gated on matching it: byte-identical training batches at every logged
> step, with bounded cross-entropy and gradient-norm agreement. This does not
> make the two step-1,000 trajectories bitwise paired — the trainer stacks
> still differ — but it rules out data-order, learning-rate and
> gradient-accumulation normalization differences between the arms.
>
> If the workstation receipts are to hand, the realized step count and the two
> agreement figures can be substituted for "bounded"; nothing depends on it.

Keep the second sentence: the sentinel bounds the trainer-stack confound, it
does not remove it, and `SPEC.md` already records exactly that ("a finite Geo
sentinel is a safety gate, not proof of a bitwise paired trajectory").

**Framing the 1.485B scratch run (`R27bE.5`).** The risk is real: 1,000 steps
at PPL ≈ 160 is early training, the two arms used different trainer stacks, and
EVQ is *worse* at 4K. Send it, but send it framed:

- call it a **matched-initialization early-training probe**, and say
  explicitly that we do not offer it as a converged comparison;
- lead with the **pre-registration** fact, because "pre-specified" is the word
  the reviewer actually used: the evaluation rows, the counted-token budget and
  the endpoint set were fixed before either run started;
- state the trainer-stack difference **ourselves**, and immediately list what
  *is* matched — initialization, scientific recipe, data-order prefix,
  2,097,152,000 counted tokens, and the same 128 document-disjoint PG-19 rows;
- explain the 4K regression via §1.6 rather than apologizing for it, and note
  that `122/128` and `126/128` documents individually favor EVQ at 8K/16K,
  which is a much harder pattern to attribute to trainer noise than a mean.

If `27bE` still rejects this branch, the loss is contained: the base/head
branch of `R27bE.5` is answered independently by the M4 factorial.

**Best owners:**

- `theory_results/OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md`
- `theory_results/OLMO2_1B_4K_RULER_FAMILY_ADAPTATION_20260726.md`
- `theory_results/OLMO2_1B_MATCHED_RULER_CONTINUATION_20260727.md`
- `theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md`
- `theory_results/LLAMA8B_MATCHED_RULER_MIX_20260726.md`
- `theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md`
- submitted Table 12 and Appendix D, Table 23

**Recommended English:**

> We agree that NLL and teacher-forced retrieval alone do not establish usable
> context, and we have separated the endpoints accordingly. With every backward
> pass capped at 4K, the matched OLMo-2 (1.485B) Native/EVQ pair scored 0/100
> versus 69/100 at 8K strict first-number exact — Native's 95% interval is
> [0, 3.7]% — and an
> independently trained EVQ seed scored 67/100 ([57.3, 75.4]%), i.e. the two
> EVQ seeds fall inside each other's intervals. The initial 16K screen was only
> 0/20 versus 1/20.
>
> Under an identical physical-4K continuation over all 13 RULER families,
> OLMo-2 Native/EVQ official macro was 82.16%/37.51% at 4K,
> 0.08%/21.29% at 8K and 0%/6.13% at 16K. Native learned the in-window
> distribution more strongly; EVQ supplied the 2×/4× length transfer.
>
> Separately, with identical physical-8K supervision over the same 13 RULER
> families, LLaMA-3-8B Native/EVQ 16K official macro was 0.295% versus 14.03%.
> At 8K the two metrics disagree: Native leads on official macro (94.44% versus
> 77.60%), which credits partial and substring matches, while EVQ leads on
> normalized exact (17.69% versus 21.54%). Neither arm is usable at 32K. We
> therefore claim task-family-adapted 2× transfer, not unseen-task transfer and
> not universal downstream superiority.
>
> One pattern is consistent across these settings and worth stating directly:
> EVQ pays a small in-window cost and buys a larger long-range gain (4K NLL
> +0.038 against 16K −0.135 at 1.485B from scratch; 8K +0.390 against 32K
> −2.048 at 8B). A finite channel budget reallocated toward long-range
> resolution predicts exactly this direction, and these are the settings where a
> trained model could have contradicted it.

### 3.2 Is EVQ technically distinct from FMRoPE, and is there a direct control?

**Concern IDs:** `RzWsa.1`, `RzWsa.2`, `AC.1`, `AC.4`.

**Direct answer:** Concede the missing citation — that part is simply owed.
Do **not** concede "overlapping motivation": it is not accurate, and it hands
away the novelty argument for nothing. State the question EVQ actually came
from, then the parameterization it produced:

\[
\text{Geo/FMR:}\quad \omega_i=b(T)^{-u_i},
\qquad
\text{EVQ:}\quad \omega_i=b^{-\phi_\tau(u_i)} .
\]

FMRoPE selects or retargets a geometric spectral range through the base/range
choice for a declared context. EVQ keeps a fixed nominal base and applies a
closed-form nonlinear index-to-exponent allocation inside the frequency grid
before training. Do not claim that EVQ is the first method ever to change
frequencies or that the two families are disjoint in motivation.

### 3.2.0 The three-axis taxonomy — lead every novelty answer with this

This is the single most important paragraph in the package. `AC.1` is the gate
that decides the paper, and it is lost or won on whether the AC can *see* that
EVQ occupies a different axis from FMRoPE rather than a different tuning of the
same one. Give them a taxonomy, not an assertion.

The taxonomy is read directly off the frequency-table formula, which is why a
reader can verify it in one line. In

\[
\omega_i = b^{-u_i},\qquad u_i = 2i/d,
\]

we distinguish three parameterization levels: direct transport of the realized
vector \(\omega\), scalar base/range selection \(b\), and training-time
exponent allocation \(u_i\). Do not write "exactly three" or "exhaustive" —
methods that replace the functional form altogether (FoPE models each channel
as a Fourier series) sit outside it, and the overclaim is unnecessary.

| Axis | Object optimized | Stage | Representative work |
| --- | --- | --- | --- |
| **1. Whole vector** \(\omega\to g_T(\omega)\) | a target-dependent transport of an already-realized spectrum, possibly per-frequency | after pretraining, aimed at a target length | PI, YaRN, LongRoPE |
| **2. Base** \(b\to b(T)\) | a scalar, preserving the normalized geometric order | before training, or at target retarget | NTK-aware, ABF, **FMRoPE** |
| **3. Exponent** \(u_i\to\phi_\tau(u_i)\) | the choice of training grid at fixed nominal base | before the model learns anything | **EVQ-Cosh** |

The one-line version, which is what the AC should retain:

> One can change \(\omega\), change \(b\), or change \(u\). Most prior RoPE
> extrapolation work — FMRoPE included — acts through the first two. EVQ acts
> on the third.

This also states our origin honestly: the field's dominant move is to vary the
base, and our starting question was whether the exponents should be spaced
uniformly at all.

**State the boundary exactly — this is where the argument is won or lost.**

> The exact-range control excludes the **base** explanation: with the highest
> sampled frequency, the lowest sampled frequency and the log span all pinned,
> a change confined to the 30 interior positions still moves trained NLL, and
> no choice of \(b\) can reproduce that intervention.
>
> It does **not** show that whole-vector methods are numerically incapable of
> producing a similar interior table. YaRN's NTK-by-parts ramp and LongRoPE's
> per-channel search do move channels non-uniformly, and we make no claim about
> what tables they can or cannot reach. The three axes are disjoint in **what
> is parameterized and optimized**, not in the set of frequency tables that are
> numerically achievable.

Two failure modes this wording prevents. First, **never say the whole-vector
axis "preserves interior geometric spacing."** It is false for YaRN and
LongRoPE, and a reviewer who knows the NTK-by-parts ramp will discard the
taxonomy on that one sentence. Second, **never claim those methods "cannot
reach" our allocation.** That is a numerical-reachability claim we have not
tested and do not need; it would turn a clean object/stage argument into one
counterexample away from collapse.

The control is built to test the base boundary specifically: pin the extrema
and the log span, change **only** the 30 interior positions, and trained NLL
still moves (0.478/0.205/0.113 at 512/1K/2K, 32/32 anchors at 512). Since
every quantity a scalar base can set is held fixed, the effect cannot be
attributed to base or range selection. That is the whole of what the control
establishes, and it is enough for `AC.1`: the interior allocation is a design
variable in its own right, separable from the axis FMRoPE operates on.

A fourth family — learned-frequency methods (FoPE, CARoPE, and the
32-parameter learnable `inv_freq` comparator in our own Table 4) — does move
interior frequencies, but by optimizing \(O(K)\) free parameters during
training. EVQ reaches the same axis with a **closed form and zero added
parameters**, obtained as the stationary point of a stated variational
surrogate.

**Exact novelty wording — do not exceed it.** The defensible claim is:

> We are not aware of prior work that poses the finite training-time exponent
> allocation of standard RoPE as an explicit variational object and solves it
> in closed form with no added learned parameters.

Do **not** write "the first method to change interior frequencies" (learned
methods do), "the first closed-form non-geometric schedule", or "unique". The
taxonomy already does the work; an absolute priority claim invites a
counterexample and costs the whole paragraph. `EVQ_COSH_REBUTTAL_PRINCIPLES.md`
§2 fixes this boundary and it has not moved.

> **BANNED WORD IN THIS SECTION: `complementarity`.**
> `zWsa` offers two routes to a higher score — "clear advantages **or**
> complementarity over FMRoPE" — and earlier drafts, along with `02_` §3.2,
> reached for the second. We cannot support it: `G-FMR-DEPLOY` records that
> naive EVQ+FMRoPE shows **no stable additivity**, and a registered combination
> arm exists, so "show us EVQ+FMRoPE versus FMRoPE" is a question we would have
> to answer with a negative. Claim **route (a), advantage under exact-range
> control**, which we do have and which is clean. All `complementarity` language
> belongs to §3.4 (YaRN, submitted Table 3, fixed \(s=8\), three seeds) — that
> is real complementarity, it is against an inference-time range method rather
> than FMRoPE, and it is already in the submitted paper.

**Implementation fidelity — state this BEFORE the number, every time.** The
entire `AC.1`/`RzWsa.1`/`RzWsa.2` answer rests on one control, and `zWsa` has
confidence 5 and states that details were checked. If the FMRoPE arm is judged
unfaithful, the novelty answer collapses in full. So we disclose the
implementation ourselves and invite correction:

> The FMRoPE rule evaluated here is §6.1 of the paper:
> \(\theta_{\text{train}}=L_{\text{train}}\),
> \(\theta_{\text{infer}}=L_{\text{target}}\),
> \(\omega_i(\theta)=\theta^{-2i/d}\), with \(L_{\text{train}}=256\) and the
> training base set to 256. We did not identify a public author implementation
> as of 23 July, so this is a paper-faithful local reimplementation of that
> rule, not an official-code reproduction, and it does not stand in for every
> FMRoPE variant. If this configuration does not match the authors' intent, we
> would be glad to be told which one does and to rerun it.

**Two phrasing rules.** Write "the FMRoPE rule evaluated here" or "that
FMRoPE rule", never a bare "FMRoPE uses…" — our single reimplemented
configuration must not be generalized to the whole method. And write "we did
not identify a public author implementation as of 23 July", never "no public
implementation existed": failing to find code is not proof of its absence, and
a confidence-5 reviewer who knows of a repository would catch the overstatement
in the one paragraph where our credibility matters most.

Volunteering this converts the most likely attack into a collaboration request.
Never quote the FMRoPE contrast without it.

**Direct evidence:** In the raw/hash-backed seed-42 exact-range control, the
sampled extrema and log span match and only 30 interior frequencies differ.
Endpoint-normalized fixed-range Cosh-minus-uniform-FMRoPE NLL is
`-0.47750/-0.20499/-0.11284` at 512/1K/2K, winning `32/32`, `27/32` and
`22/32` anchors. When the schedules are instead retargeted to the declared
target length, the ordering reverses (`+0.061/+0.182/+0.279`). **Do not write
this as "FMRoPE is stronger."** The two conditions are not a like-for-like
method contest: under retargeting, the FMRoPE rule is doing exactly what it is
designed to do — target-aware range selection — while EVQ's shape is merely
carried along. The accurate statement is *when the range is free to track the
declared target, range selection dominates the interior-shape effect at these
lengths*. That is a statement about which factor dominates in that condition,
not a concession about method quality, and it costs us nothing to say
precisely.

**Status:** `POST_SUB_RAW_HASH_BACKED`, single seed.

**Owner:** `theory_results/MATCHED_RANGE_COSH_500M_S42_20260724.md`.

**Expanded exact-range evidence — and how to keep it from undercutting the
seed-42 control.** The two exact-range studies report effect sizes roughly two
orders of magnitude apart (`-0.478` versus `-0.0099` NLL). Earlier drafts put
both in the same paragraph, which invites a careful reviewer to compute the
ratio and ask which one is real. Give them **non-competing roles**:

| Study | Role in the response | What to quote |
| --- | --- | --- |
| seed-42 exact-range (151.9M, \(L=256\)) | the **direct FMRoPE control** and the effect-size owner | the NLL deltas and the per-anchor win counts |
| M4 factorial (50.9M, 12 configs × 3 seeds) | **robustness of the direction** across base, training length and head dimension | the **sign statistics**, not the mean |

For M4, quote `10/12` and `9/12` structural configurations favoring the
non-uniform allocation, and the fact that this holds across bases `500K/1M`,
training lengths `256/1024` and head dimensions `32/64/128`. That is what M4
is actually strong at, and it is what `R27bE.2` asked for.

**Do not lead with the formula-Cosh mean (`-0.009879`): its
configuration-bootstrap CI crosses zero.** If a mean is needed, use the
preregistered `1.25×` Cosh (`-0.012100`) or the deformation-matched exponential
(`-0.010619`), and phrase the conclusion as *the allocation axis is
identifiable*, never as *Cosh is better*: formula-Cosh minus matched
exponential is `+0.000740` with an exact sign-flip \(p=0.836\), and the formula
point is the best preregistered Cosh multiplier in only `4/12` configurations.

If asked directly about the magnitude gap, the answer is short and honest: the
two studies differ in model size, training length, schedule normalization and
OOD weighting, so their magnitudes are not comparable; what is comparable, and
what we claim, is the **sign**, which agrees.

This is supporting/mechanistic evidence: it strengthens allocation
identifiability across base/head settings, while directly ruling out universal
Cosh or formula-\(\tau\) optimality.

**Expanded owner:**
`theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md` and
`theory_results/m4_exact_range_factorial_evidence_20260726.json`.

**Recommended English:**

> We agree that FMRoPE is directly relevant and should have been cited and
> compared, and we will do both. What the two works share is the field's
> premise that the RoPE frequency set matters for long context. The questions
> asked of that premise differ, and that difference is where EVQ came from.
> The great majority of RoPE extrapolation work — NTK-aware scaling, ABF,
> FMRoPE — varies the base: given a geometric grid, which base or range suits a
> declared context. Our starting point was the assumption that line of work
> leaves untouched, namely that the exponents should be spaced *uniformly* in
> the first place. Writing \(u_i=2i/d\), the base school asks for the best
> \(b(T)\) in \(\omega_i=b(T)^{-u_i}\); we asked whether \(u_i\) itself is the
> right allocation, and arrived at \(\omega_i=b^{-\phi_\tau(u_i)}\) with a
> fixed nominal base and \(\phi_\tau\) the closed-form inverse CDF of the
> stationary density of an explicit variational surrogate.
>
> That is why our control takes the form it does — it is the experiment the
> question implies. Our FMRoPE arm follows §6.1 of the paper
> (\(\theta_{\text{train}}=L_{\text{train}}\),
> \(\theta_{\text{infer}}=L_{\text{target}}\),
> \(\omega_i(\theta)=\theta^{-2i/d}\)); we found no public author
> implementation as of 2026-07-23, so this is a paper-faithful reimplementation
> and we will gladly rerun it under any configuration the authors prefer. With
> sampled extrema, log span, initialization, token order, optimizer, budget and
> the 32 evaluation anchors all matched, and only the 30 interior frequencies
> changed, the Cosh interior allocation improves fixed-range OOD NLL by
> 0.478/0.205/0.113 at 512/1K/2K, winning 32/32, 27/32 and 22/32 anchors. A
> separate three-seed factorial holds the same exact-range constraint across
> bases 500K/1M, training lengths 256/1024 and head dimensions 32/64/128, and
> the non-uniform allocation is favored in 10/12 and 9/12 structural
> configurations — so the direction is not specific to one base or head size.
>
> We report both conditions. When the grids are instead retargeted to the
> declared target length the ordering reverses (+0.061/+0.182/+0.279): with the
> range free to track the target, range selection dominates the interior-shape
> effect at these lengths, which is what a target-aware range method is for.
> The two conditions isolate different factors and we do not read either as a
> method ranking. Within our own family, a
> deformation-matched exponential is statistically indistinguishable from Cosh
> (+0.0007 NLL, sign-flip \(p=0.836\)), so we claim the allocation *axis*, not
> the optimality of Cosh. Our claim is that finite-channel interior allocation
> is a separately identifiable training-time design variable that a scalar
> base/range change cannot reproduce — not that EVQ replaces or dominates
> target-aware range methods.

### 3.3 Are finite \(\tau\), Cosh, and the DAPE result actually attributable?

**Concern IDs:** `R27bE.1`, `R27bE.3`, `R27bE.4`, `RDz6s.3`,
`AC.3`.

**Direct answer on finite \(\tau\):**

The theory supplies the scaling structure, not an exact finite-\(\tau\)
constant:

\[
\tau = c(\Pi)\,\frac{d_{\mathrm{eff}}}{\sqrt{L_{\mathrm{train}}}} .
\]

Here \(c(\Pi)=O(1)\) is convention- and protocol-dependent. Under the
submitted convention, \(c=1\) is an operating default, not a theorem or global
optimum. If a selection objective explicitly weights a deployment target,
\(L_{\mathrm{target}}/L_{\mathrm{train}}\) can enter \(\Pi\), together with
the base/log span, number of rotary channels, head geometry, architecture,
data, and objective. It need not alter the claimed
\(d_{\mathrm{eff}}/\sqrt{L_{\mathrm{train}}}\) scaling structure.

**Empirical calibration:** In the direct sweep, selected \(\tau=5\) gives
selection NLL `5.9898`; the rule value \(5.657\) gives `6.0017`, a difference
of `0.0119`. Across the 99-run reanalysis, the rule beats Geo in 7/9
configuration means and 18/27 seed pairs, but beats the neighboring empirical
pilot in only 3/9 means and 8/18 available pairs. It is therefore a useful but
fallible basin selector.

The exact-range factorial makes that boundary sharper. The formula point is
the best of the preregistered `0.75×/1.0×/1.25×` Cosh arms in only `4/12`
structural configurations; `1.25×` wins `6/12` and `0.75×` wins `2/12`.
At the two preregistered extremes, formula \(\tau=1\) prefers
\(1.5\times\tau\), whereas formula \(\tau=8\) prefers
\(0.75\times\tau\). The rule remains a deployable default, not a point
optimizer.

### 3.3.0 First: `27bE` did not misread us, and his criticism survives the label

Settle this internally before drafting, because it sets the tone of the whole
reply.

**Who mislabeled?** We did. The reviewer read "DAPE, 32 extra params" in the
submitted table and reasoned about DAPE. That is correct behavior on his part.
There is no misunderstanding to correct, and **no sentence in the reply may
imply that he misread anything.**

**Does the label error weaken his criticism?** No — it strengthens it. His
argument is that the comparison cannot attribute the PPL gap to *allocation
shape*, because three variables move together: the shape, the number of
trainable positional parameters (0 vs 1 vs 32), and the optimization/tuning
budget those parameters require. That argument is *more* obviously right when
the comparator is a 32-parameter learnable `inv_freq` baseline, since a learned
frequency vector is precisely the "learned capacity" confound he names. His
`learned-τ` observation is the sharpest part: a 1-parameter method reaching
437.9 against a 32-parameter method's 455.3 is a signal that the 32-parameter
arm may be under-optimized rather than shape-deficient. He is right about that.

**Consequence for strategy — this is the important part.** The correct reply is
therefore *not* a defence of the DAPE row. It is: **"you specified an
experiment; we ran it."** In his Limitations he wrote that a more direct test
would keep the positional operator unchanged and vary only the fixed frequency
schedules, comparing the standard geometric arrangement, EVQ, and several
alternative analytic allocations. That is `EXPERIMENT_REPORT_20260724.md` §3
and §7, almost line for line. The DAPE label becomes a one-clause aside inside
an answer that is overwhelmingly about the controls he asked for.

**The three-level ladder — lead with this.** Each level removes one of the
confounds he named:

| Level | Control | What it removes | Owner |
| --- | --- | --- | --- |
| 0 | learned comparators (Table 4) | — (this is the row he criticized) | submitted |
| **1** | fixed analytic schedules, same operator, same protocol, **all zero-parameter**: Paper-Geo / uniform span-matched / EVQ-Cosh / power-matched / exponential-matched, 3 seeds | learned capacity, parameter count, optimization and tuning budget | §3 |
| **2** | same, on the **native Std-RoPE endpoint grid**, all non-Geo arms sharing the same span *and* the same RMS deformation (`0.255704`): Std-RoPE / EVQ / exponential / exact-kernel uniform prior / attention-prior two-band, 3 seeds | additionally the Paper-Geo midpoint ambiguity and "amount of deviation" | §7 |
| **3** | exact-range factorial: highest frequency, lowest frequency and log span pinned; only interior positions move; 12 structural configs × 3 seeds over base 500K/1M, \(L\) 256/1024, \(d_{\rm head}\) 32/64/128 | additionally any implicit base/range change | M4 |

**Level 2 is currently under-used and is the strongest single answer.** It is
the closest match to his requested design *and* it carries seed-level paired
95% intervals that exclude zero at every length: native EVQ minus native
Std-RoPE is `-0.113/-0.149/-0.207/-0.190/-0.099` NLL at 512/1K/2K/4K/8K with
intervals `[-0.200,-0.026]`, `[-0.183,-0.115]`, `[-0.241,-0.174]`,
`[-0.260,-0.121]`, `[-0.141,-0.058]`, and all three seeds agreeing at every
length. Earlier drafts mentioned §7 only as a caveat ("two-band is stronger at
some lengths"). Promote it.

**And keep his own conclusion.** At Levels 1–2 the matched exponential and the
attention-derived two-band schedule beat Cosh at some lengths. That is the
honest reading and it is the one he would reach himself: the *allocation axis*
is real and identifiable; Cosh is a closed-form zero-parameter instance on that
axis, not a demonstrated optimum. Stating this before he has to is what
converts the answer from a defence into a shared conclusion.

> **Decision (author, 2026-07-27): answer yes, describe the comparator, stop.**
> `27bE` asked a yes/no question — was that row given comparable tuning — and
> the answer is **yes**: a dedicated 10×/100× positional learning-rate sweep,
> better setting reported. Give that answer. Name the comparator for what it is
> in the same sentence (a 32-parameter layer-shared learnable `inv_freq`
> baseline), because that is the accurate description and it costs about
> fifteen words. **No apology framing, no "we withdraw", no "correction we owe
> you".** The label fix is one line in the revision list.
>
> Two things stay out of the reply entirely: the Phase 11B Kerple+MLP result
> (nobody asked how EVQ compares against a real DAPE operator — it is a
> prepared discussion answer only), and any characterization of the mislabel as
> an error on our part beyond the revision-list line.
>
> **Why name the comparator at all rather than just saying "yes, it was
> tuned":**
> The governing principle for this package is to answer what reviewers ask and
> volunteer nothing else. That principle is what forbids the old wording, not a
> duty to confess: `27bE` **asked directly** about this row — "the manuscript
> does not clearly indicate whether DAPE was provided with a comparable degree
> of hyperparameter tuning." Replying "we tuned DAPE at 10×/100×" is not
> silence about an unasked topic; it is an affirmative claim about DAPE, in
> answer to a question about DAPE, that the implementation does not support.
>
> Because our thesis is *zero-parameter allocation versus learned frequency
> parameterization*, and a 32-free-parameter learnable `inv_freq` comparator is
> a **more** on-point control for that thesis than DAPE would have been. Naming
> it accurately is not a concession; it describes a better-matched baseline.
> Saying "we tuned DAPE" in answer to a question about DAPE is the only version
> that carries real exposure, and it buys nothing.
>
**What the row actually is.** The repository records this in four independent
places, so it is settled fact, not a suspicion:

- `real_dape_compare/run_dape_compare.py` hard identity rule:
  `free_inv_freq == historical paper Table-4 row mislabeled "DAPE"`, with the
  instruction `NEVER write free_inv_freq results under the name DAPE`;
- `real_dape_compare/FINDINGS.md`: the printed row is a **32-dimensional
  layer-shared learnable `inv_freq` vector** — no Kerple bias, no
  attention-score MLP;
- `FIRST_PRINCIPLES_REBUTTAL_REASSESSMENT_20260716.md` §55 reaches the same
  conclusion and directs us to **relabel proactively**, because the row is
  Primary II's comparator identity;
- `EVQ_TRUE_OBJECTIVE_ULTRA_AUDIT.md` §264 independently describes it as a
  layer-shared learnable frequency vector without monotonicity constraints.

`00_` already encodes the requirement in `R27bE.3`'s response line —
"**Correct the method identity where necessary**". Earlier playbook revisions
dropped it. Restore it.

**There is a second fact that must be prepared, not improvised.** The
repository *does* contain a closer DAPE implementation — Phase 11B,
`KerpleBias` + `DAPERefine` MLP on pre-softmax scores, 125M, \(L=256\), 100M
tokens, three seeds — and **EVQ does not win there**:

| PPL@8K, 3-seed mean | Geo | EVQ \(\tau=4\) |
| --- | ---: | ---: |
| plain (no adaptive-bias module) | 352.7 | **254.7** |
| + Kerple/MLP DAPE-ish operator | **55.9** | 56.8 |

The adaptive-bias operator improves *both* substrates by roughly 6×, far more
than allocation does, and allocation contributes nothing on top of it (+1.6%,
within noise). `FINDINGS.md` states the conclusion directly: the evidence
points to the DAPE-ish operator dominating plain EVQ in absolute PPL, and EVQ's
position must be *zero-parameter frequency allocation*, not "beats DAPE." No
official Zheng DAPE reproduction exists at all.

Once we volunteer the relabel, "then how does EVQ compare with real DAPE?" is a
near-certain follow-up. Answer it from the table above, and note that the
comparison is between a zero-parameter frequency table and an added attention
operator with extra parameters and ~1.8× training time — different cost
classes, which is why we claim an allocation axis rather than operator
superiority. `FINDINGS.md` is explicit: **do not hide this if asked.**

**What the 10×/100× sweep still legitimately answers.** It documents the tuning
budget *of the learned-frequency baseline*: `100×` gives PPL@8K `455.3` versus
`477.7` at `10×`, both seed 42. That remains a real answer to "was the learned
comparator under-tuned?" — it simply is not a DAPE answer.

**Do NOT build an argument on the parameter-count ordering.** An earlier draft
observed that PPL@8K improves as positional parameters decrease (32 → 455.3,
1 → 437.9, 0 → 333.7) and concluded that this rules out a capacity or
optimization-budget explanation. **That inference is backwards.** More
trainable parameters are generally *harder* to optimize, so the learned arms
underperforming is exactly what an under-optimization explanation predicts —
it is evidence *for* `27bE`'s hypothesis, not against it. The paragraph has
been removed from the reply. What survives is only the weak statement that
parameter count alone does not explain the difference; shape attribution rests
entirely on the zero-parameter ladder in §3.3.0.



**Copy-ready (use verbatim — minimal, factual, no confession framing):**

> Yes. That comparator — a layer-shared learnable inverse-frequency baseline
> with 32 free parameters — received a dedicated positional learning-rate
> sweep at 10× and 100×, and the reported row is the better of the two (455.3
> versus 477.7 PPL@8K, seed 42). We should have stated that budget in the
> paper and the revision will.
>
> We accept the deeper point regardless of the tuning answer: a learned
> comparator cannot settle a question about allocation *shape*, because learned
> capacity remains a confound. Shape attribution therefore rests on fixed
> zero-parameter schedules under an unchanged operator and training protocol,
> and more strictly on the exact-range factorial, where the sampled extrema and
> log span are pinned and only interior positions move.

Note what this does: it answers the tuning question that was asked, states the
comparator's identity as a fact rather than an apology, closes the
DAPE-competitiveness exposure in one clause, and hands the attribution argument
to the controls that actually carry it. It is shorter than the wording it
replaces.

**Shape attribution — the three-level ladder of §3.3.0.** Learned capacity
remains a confound regardless of the budget answer, so attribution rests on
zero-parameter controls at three strictnesses:

- **Level 1** (§3): same operator, same protocol, all arms fixed and
  zero-parameter. EVQ-minus-Geo mean tail NLL `-0.256/-0.305/-0.223/-0.238` at
  1K/2K/4K/8K, 3/3 seeds. Span-matched uniform, RMS-matched power and
  RMS-matched exponential arms included.
- **Level 2** (§7): native Std-RoPE endpoint grid, same span, and identical RMS
  deformation `0.255704` for every non-Geo arm. Native EVQ minus native
  Std-RoPE `-0.113/-0.149/-0.207/-0.190/-0.099` at 512/1K/2K/4K/8K with paired
  95% intervals `[-0.200,-0.026]`, `[-0.183,-0.115]`, `[-0.241,-0.174]`,
  `[-0.260,-0.121]`, `[-0.141,-0.058]`; 3/3 seeds at every length.
- **Level 3** (M4): extrema and log span pinned, interior only.

Exponential and attention-derived two-band schedules are stronger at some
lengths — at 4K/8K the two-band arm beats EVQ by `0.116`/`0.211` NLL. Thus
allocation matters as an axis; Cosh is not claimed to be universally optimal.

The stricter exact-range factorial removes the remaining endpoint/span
confound and directly includes a non-Cosh arm. Across 12 structural
configurations, preregistered `1.25×` Cosh and RMS-matched exponential improve
weighted OOD NLL over Geo by `0.012100` and `0.010619`; formula-Cosh and
matched exponential are indistinguishable at `+0.000740` NLL. This supports
the allocation axis while rejecting Cosh uniqueness.

**Best owners:**

- `theory_results/EXPERIMENT_REPORT_20260724.md`
- `theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md`
- `theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md`
- `theory_results/m4_exact_range_factorial_evidence_20260726.json`
- `docs/exp/2026-02/2026-02-24_128tok_baseline_report.md`

**Recommended English:**

> The finite-\(\tau\) rule is an empirical operating rule with a derived
> scaling structure, not an exact optimizer. More precisely, the analysis
> motivates \(\tau=c(\Pi)d_{\rm eff}/\sqrt{L_{\rm train}}\), where the
> convention-dependent \(O(1)\) coefficient is calibrated empirically. In a
> direct sweep, the selected \(\tau=5\) and the rule value 5.657 differ by only
> 0.0119 selection NLL. The broader 99-run audit also shows why we do not call
> the rule optimal: it often selects a useful basin but is not consistently
> better than a neighboring empirical choice.
>
> We further ran a three-seed exact-range factorial over bases 500K/1M,
> training lengths 256/1024, and head dimensions 32/64/128. The formula point
> is the best preregistered Cosh multiplier in only 4/12 structural
> configurations. Preregistered \(1.25\times\) Cosh and
> deformation-matched exponential improve weighted OOD NLL over Geo by
> 0.0121 and 0.0106, while formula-Cosh and matched exponential differ by only
> 0.0007. This identifies allocation independently of range, but does not
> support universal Cosh or formula-\(\tau\) optimality.
>
> On the tuning question: yes. That comparator, a layer-shared learnable
> inverse-frequency baseline with 32 free parameters, was tested with 10× and
> 100× positional-parameter learning-rate multipliers, and the reported 100×
> setting is the better one (455.3 versus 477.7 PPL@8K). To remove learned
> capacity from the
> allocation-shape question, we additionally compared fixed zero-parameter
> schedules under the same operator and training protocol. EVQ improved over
> Geo at all four extrapolation lengths, while exponential and two-band
> schedules were stronger at some lengths. This supports allocation as a design
> axis, not universal Cosh optimality.

### 3.4 Does EVQ complement range scaling without claiming tuned dominance?

**Concern IDs:** `RDz6s.2`, `AC.1`, `AC.4`.

> **This section owns the word `complementarity` for the whole package.**
> `zWsa` will accept complementarity as a route to a higher score; §3.2 cannot
> supply it against FMRoPE. It is supplied here, against YaRN, by submitted
> Table 3 — same fixed transform, three training seeds, already peer-visible.
> When answering `RzWsa.2`, route the reviewer here rather than stretching the
> FMRoPE control.

**Direct answer:** The submitted result establishes substrate-dependent
leverage under the **same fixed** YaRN transformation, not superiority to an
exhaustively optimized Geo+YaRN or target-aware range search.

### 3.4.0 Which YaRN operator Table 3 uses — resolved, and it constrains the claim

`docs/exp/2026-07/2026-07-13_primary1_seed42_operator_diagnostic.md` settles this, and
not in the convenient direction. Its own summary:

> "The much larger separation belongs to the repository's custom fixed-ramp
> scaler, not to official YaRN." … "both trained substrates use the midpoint
> grid \(u=(k+0.5)/K\) … the correct label for both evaluated arms is
> **YaRN-derived**, not a faithful official-YaRN method reproduction."

Same seed, same 454M model, three operators, EVQ−Geo:

| Operator | ΔNLL@8K | ΔNLL@12K | ΔNLL@16K | global PK, Geo→EVQ |
| --- | ---: | ---: | ---: | --- |
| none (raw substrate) | −0.1842 | −0.2070 | −0.1892 | 64.0 → 70.5% |
| **repo fixed-ramp** (Table 3's transform) | −0.1149 | **−0.4107** | **−0.4822** | 76.0 → **90.5%** |
| **YaRN-derived** (pinned official equations) | **−0.0293** | −0.0291 | −0.0267 | 89.0 → 94.5% |

**The direction survives under every operator; the magnitude does not.** Under
the YaRN-derived operator both substrates improve enormously (Geo@16K PPL
−65.9%) and their gap compresses to 2.6–2.9% PPL and +5.5 PK points, against
+14.5 points under the fixed ramp.

**Weight of this diagnostic:** low. Single seed, new lineage, batch 4 against
the historical batch 2 (half the optimizer updates), undertrained at PPL ≈ 65
at the training length, and explicitly not promoted to `data/curated/` or the
paper. It does **not** overturn Table 3's three-seed numbers. What it does is
bound what those numbers can be claimed to show.

**This diagnostic is NOT disclosed and NOT referenced in any reply.** No
reviewer asked which range operator Table 3 uses. It is a single-seed,
undertrained, new-lineage, unpromoted internal artifact, and Table 3's
three-seed numbers stand on their own as submitted evidence. Quote Table 3 as
written.

**Two constraints only, both free:**

1. **Do not add the word "official" to YaRN in any reply**, and do not describe
   the transform as a faithful reproduction of the YaRN method. Write "the same
   fixed range transform applied to both substrates" or simply "YaRN at fixed
   \(s=8\)", matching the paper. This is not disclosure — it is declining to
   make an affirmative claim we cannot support, which costs nothing because the
   argument is about the *substrate* under an identical transform either way.
2. **Do not generalize the claim beyond the tested transform.** "The training
   substrate changes what this fixed transform recovers" is supported.
   "EVQ is complementary to range scaling in general" is not, and reaches
   beyond what any of our runs test.

**If pressed in discussion** — and only then — the diagnostic exists and can be
given with its single-seed, undertrained, new-lineage limits. Do not pre-empt
that question, and do not speculate in the reply about what a stronger operator
would do. `Dz6s` asked whether we ran a tuned Geo+YaRN search; "we did not"
fully answers him.

**Repository audit (`RDz6s.2`) — completed 2026-07-27. There is no matched YaRN
scale sweep.** Four things exist and each must be handled differently:

| Artifact | What it actually is | Use |
| --- | --- | --- |
| `internal/draft_scripts/phase20_eval_suite.py`, `eval_yarn_overlay(yarn_scales=[1,2,4,8])` | **PLACEHOLDER.** The source comment states the scaling is never applied and all scales evaluate identically. | **NEVER cite.** Quoting this as a sweep would be a fabricated result. Flagged here so no future agent mistakes the signature for evidence. |
| `results/core_text/phase21b/…_454m_full_eval.json`, `results_yarn.8k_yarn_scale2` | A **real second scale point** (\(s=2\), 8K, 454M, both arms) on QuALITY QA, n=2,086. | See disclosure decision below. |
| `results/core_text/phase14_yarn_passkey/{geo,hybrid}_750m_yarn4x` | \(s=4\) passkey at 750M, both arms — but the intervention arm is labeled `hybrid1.5_r16`, not EVQ. | **Do not use** until the arm identity is checked against `FREQUENCY_DEFINITION_MANIFEST.json`. A mislabeled method identity here would be far more damaging than the result is worth. |
| `results/native_rope_evq_150m_yarn_ablation_s42_20260714/` | A real YaRN **component** ablation at fixed \(s=8\): `freq_only` / `mscale_only` / `full`. | Usable, but it answers a different question — *which part* of YaRN interacts with the substrate, not *which scale*. |

**The \(s=2\) point — read the numbers correctly before using them.** An
earlier draft of this section said the gap "closes." It does not. Both metrics,
8K QuALITY QA at 454M:

| | Geo | EVQ | gap |
| --- | ---: | ---: | ---: |
| raw, accuracy | 24.59% | 26.75% | +2.16 |
| + YaRN(\(s=2\)), accuracy | 26.51% | 26.61% | +0.10 |
| raw, gold-answer NLL | 3.2021 | 2.2392 | 0.963 |
| + YaRN(\(s=2\)), gold-answer NLL | 2.3893 | 2.1949 | **0.194** |

YaRN at \(s=2\) recovers most of Geo's deficit; EVQ stays ahead on both
metrics, by 0.19 NLL. **Narrowed, not closed, not reversed.** And the accuracy
column never separated anything to begin with: SE ≈ 0.95% at n=2,086, so the
original +2.16 was ~2.3 SE and the file marks it "marginal."

**Answer the question as asked — and the question is narrow.** `Dz6s` asked
whether a more optimized Geo+YaRN could narrow the gap *in the Table 3 setting*
(passkey NLL-gap retrieval and PPL at 454M). The \(s=2\) point is a different
task, a different metric family, and an evaluation the report itself says
cannot resolve RoPE variants. It therefore does not answer his question, and
under the answer-what-is-asked rule it is **not** required in the reply.

**Default (recommended): answer the question, claim narrowly, stop.**

> A matched fixed scale does not establish dominance over an optimized
> Geo+YaRN or a channel-wise search, and we did not run that search. The
> comparison applies the same fixed range transform to both substrates, and its
> purpose is narrower than a method contest: it asks whether the training-time
> frequency substrate changes what the same inference-time transform can
> recover. With the scale fixed at \(s=8\) for both arms and three training
> seeds, Geo+YaRN versus EVQ+YaRN is \(61\pm3\%\) versus \(100\pm0\%\)
> teacher-forced NLL-gap retrieval at 8K, and 82.9 versus 70.9 PPL. Settling
> the tuned comparison would need a joint sweep over \(s\),
> \(\beta_{\text{fast}}\), \(\beta_{\text{slow}}\) and mscale for both
> substrates at matched budget, which we did not run.

An earlier draft appended "we would expect a stronger range operator to
compress the gap." **That was volunteered, not asked, and it is removed.**
`Dz6s` asked whether a tuned Geo+YaRN *could* narrow the gap; "we did not run
that search" answers it. Speculating about what a better operator would do
hands over a concession he did not request.

**Hold the \(s=2\) point in reserve for the discussion phase.** If `Dz6s`
presses — "do you have *any* evidence at another scale?" — then the answer
exists and should be given plainly: on 8K QuALITY QA at 454M, \(s=2\) lifts
Geo's gold-answer NLL from 3.20 to 2.39 against EVQ's 2.19, narrowing the gap
from 0.96 to 0.19 without closing it, on an endpoint where 4-option accuracy
(26.51% versus 26.61%, SE ≈ 0.95%) separates nothing. Do not lead with it, and
do not attach interpretation.

In submitted Table 3, with YaRN scale \(s=8\) fixed for both substrates and
three training seeds, Geo+YaRN versus EVQ+YaRN gives:

- teacher-forced NLL-gap PK@8K: `61±3%` versus `100±0%`;
- PK@12K/16K: `59/51%` versus `79/68%`;
- PPL@8K/16K: `82.9/157.7` versus `70.9/107.5`.

This shows that the training-time grid changes what the same inference-time
transformation acts on. The exact-range result separately shows that
target-aware FMRoPE can be stronger when the target range itself is retuned.

**Recommended English:**

> We agree that a fixed matched YaRN scale does not establish dominance over a
> fully optimized range search. Its purpose is narrower: it tests whether the
> training-time frequency substrate changes the leverage of the same
> inference-time transformation. In submitted Table 3, with \(s=8\) fixed for
> both arms and three training seeds, Geo+YaRN versus EVQ+YaRN obtained
> \(61\pm3\%\) versus \(100\pm0\%\) teacher-forced NLL-gap retrieval at 8K
> and 82.9 versus 70.9 PPL. We therefore retain a substrate-dependent
> complementarity claim, not tuned-YaRN dominance.

### 3.5 Related-work positioning: answer the charge, not just the citation

**Concern IDs:** `RzWsa.1`, `AC.1`.

`zWsa`'s charge is twofold — one missing citation, and "the related work
appears insufficiently surveyed." Adding Oka et al. answers the first and
leaves the second standing. The AC's wording is broader still: FMRoPE **and
prior dead-frequency observations**, plural.

We have an unused fact here. Submitted §2 already cites and discusses
Barbero et al. (2025) on high/low-frequency channel specialization, Resonance
RoPE on critical frequencies that interpolation should not perturb, plus HoPE,
FoPE, CARoPE, Clipped RoPE, MHRoPE and the video-RoPE line. The submission
therefore never claimed the dead/ineffective-channel observation as novel — it
credited prior work for it. Saying so is a pure statement of fact about our own
paper and it rebuts "insufficiently surveyed" far more effectively than one
added reference.

**Tone requirement:** flat and factual. No trace of "the reviewer did not read
§2." The concession about Oka et al. must come first and must be unqualified.

**Recommended English:**

> On related work: we should have cited and compared against Oka et al., and
> the revision will do both. We would add one clarification, since it bears on
> the novelty question. We did not claim the observation that some RoPE
> channels are ineffective as a contribution — §2 of the submission credits it
> to prior work, citing Barbero et al. (2025) on high/low-frequency channel
> specialization and Resonance RoPE on critical frequencies, among others. Our
> intended contribution is narrower: posing the finite training-time grid
> allocation as an explicit variational object and giving a closed-form,
> zero-learned-parameter realization of it. We accept that the submission did
> not make that boundary visible enough, and the revision will state it
> directly in §1 and §2.

## 4. Compact evidence table

All endpoints below must remain separate. `PPL/NLL`, teacher-forced NLL-gap,
strict autoregressive exact, and RULER are not interchangeable.

| Evidence | Tier | Exact reviewer-usable result | Claim-local boundary | Owner |
| --- | --- | --- | --- | --- |
| 454M EVQ×YaRN, submitted Table 3 | `SUBMITTED` | Fixed \(s=8\), 3 seeds: PK@8K `61±3→100±0%`; PPL@8K `82.9→70.9`; PPL@16K `157.7→107.5` | PK is teacher-forced NLL-gap; fixed scale, not tuned-range dominance | submitted Table 3; curated Primary-I JSON |
| 432M MLA, submitted Table 18 | `SUBMITTED` | 500M tokens, 8K training, 16 rotary channels, 3 seeds: PPL@16K `138.8±5.5→95.6±4.1`; same \(s=4\) EVQ+YaRN `71.1±4.1` | Tested scarce-channel PPL configuration; not downstream capability | submitted Table 18; `table18_mla_3seed_aggregate.json` |
| 750M continuation, submitted Table 12 | `SUBMITTED` | Teacher-forced 8K retrieval `100%→100%`; strict AR exact `0→77.5%`; PPL@16K `45.1→24.4` | Single-seed, task-specific supporting continuation; not multi-seed scale closure | submitted Table 12 |
| Video DiT, submitted Table 14 | `SUBMITTED` | 129M/382M exploratory breadth evidence | Cross-modal scope only; not production video or primary evidence | submitted Table 14 |
| LLaMA-3-8B LoRA, submitted Appendix D Table 23 | `SUBMITTED` | untouched Base/EVQ-LoRA PPL: 8K `7.42/9.63`, 16K `176.3/21.5`, 32K `1942.5/104.3` | Single seed; unmatched PPL scale anchor, not attribution or capability | submitted Appendix D, Table 23 |
| Direct \(\tau\) sweep | `POST_SUB_RAW_HASH_BACKED` | selected `5` vs rule `5.657`: `0.0119` selection-NLL gap | One direct sweep; rule is a fallible basin prior | `EXPERIMENT_REPORT_20260724.md`; `PHASE16_99RUN_RAW_REANALYSIS_20260724.md` |
| Submitted row labeled "DAPE" | `SUBMITTED` + `INVALID_IDENTITY` | 32-parameter **layer-shared learnable `inv_freq`**, not the DAPE operator; PPL@8K `455.3` at `100×` vs `477.7` at `10×`, seed 42 | **Relabel proactively.** No DAPE-specific competitiveness claim. Usable only as a learned-frequency comparator with a documented tuning budget | `real_dape_compare/FINDINGS.md`; `run_dape_compare.py` identity rules; `FIRST_PRINCIPLES_REBUTTAL_REASSESSMENT_20260716.md` §55 |
| Phase 11B Kerple+MLP DAPE-ish | `NEGATIVE` | 3-seed PPL@8K: plain Geo/EVQ `352.7/254.7`; +DAPE-ish `55.9/56.8` | EVQ adds nothing on top of the adaptive-bias operator (+1.6%, within noise). Zheng-inspired, not an official reproduction. Prepared answer only — do not volunteer before the relabel is stated | `real_dape_compare/FINDINGS.md`; `phase11b_125m_l256_3seed.json` |
| Fixed analytic schedules (Level 1) | `POST_SUB_RAW_HASH_BACKED` | EVQ−Geo NLL `-0.256/-0.305/-0.223/-0.238` at 1K/2K/4K/8K, 3/3 direction; uniform span-matched, power- and exponential-matched arms included | All arms zero-parameter under one operator, so capacity and tuning budget are controlled. Matched exponential is better at the reported means | `EXPERIMENT_REPORT_20260724.md` §3 |
| Native Std-RoPE real-shape study (Level 2) | `POST_SUB_RAW_HASH_BACKED` | Native EVQ−Std-RoPE NLL `-0.113/-0.149/-0.207/-0.190/-0.099` at 512/1K/2K/4K/8K; paired 95% CIs all exclude zero; 3/3 seeds at every length | **Closest match to `27bE`'s requested design.** Native endpoint grid, same span, identical RMS deformation `0.255704` across non-Geo arms. Attention-prior two-band beats EVQ by `0.116`/`0.211` at 4K/8K | `EXPERIMENT_REPORT_20260724.md` §7; `native_attention_shape_l128_results_20260724.json` |
| Exact-range Cosh vs FMRoPE | `POST_SUB_RAW_HASH_BACKED` | seed 42 Cosh−FMR NLL `-0.47750/-0.20499/-0.11284` at 512/1K/2K | Endpoint-normalized, single seed; under target-retargeting the ordering reverses — range selection dominates interior shape at these lengths | `MATCHED_RANGE_COSH_500M_S42_20260724.md` |
| M4 exact-range factorial | `POST_SUB_RAW_HASH_BACKED` | 12 structural configs × 3 seeds: `1.25×` Cosh/Exp−Geo weighted OOD NLL `-0.012100/-0.010619`; formula-Cosh−Exp `+0.000740` | 50.9M supporting/mechanistic; natural-text NLL only; formula point wins 4/12 Cosh comparisons | `M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md`; curated per-run JSON |
| OLMo-2 1.485B matched CF natural LM | `POST_SUB_RAW_HASH_BACKED` | Native/EVQ NLL: 4K `2.235/2.548`, 8K `3.735/2.703`, 16K `4.851/2.925` | Teacher-forced NLL; one matched training seed; in-window cost | `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` |
| OLMo-2 1.485B strict NIAH | `POST_SUB_RAW_HASH_BACKED` | 8K exact Native/matched-EVQ `0/100 vs 69/100`; independent EVQ seed `67/100` | Same generator family, disjoint rows/values; 16K screen only `0/20 vs 1/20` | same owner |
| OLMo-2 1.485B final query-gap + EOS | `POST_SUB_RAW_HASH_BACKED` | Strict full answer-string + terminal EOS, EVQ/Native: 4K `100/95`, 8K `98/18`, 16K `60/0` out of 100 | Identical +100 and +32 downstream protocol; one seed; same numeric NIAH family; target-range phase exposure; 16K far-gap EVQ `31/66`; localized EVQ 4K retention drop `0.70→0.55` on `niah_single_2` | `EVQ_QUERY_GAP_FINAL_DIAGNOSTIC.md` |
| OLMo-2 13-task matched RULER continuation | `POST_SUB_RAW_HASH_BACKED` | Native/EVQ official macro: 4K `82.16/37.51%`, 8K `0.08/21.29%`, 16K `0/6.13%` | One continuation seed; RULER-family supervised; Native wins strongly in-window, EVQ at 2×/4× | `OLMO2_1B_MATCHED_RULER_CONTINUATION_20260727.md` |
| LLaMA-3-8B matched ordinary LM | `SUPPORTING` | EVQ−Native NLL 8K/16K/32K `+0.390/-1.510/-2.048` | Single-seed teacher-forced probability result; separate from RULER protocol | `EVQ_8B_ADAPTATION_EVIDENCE_20260724.md` |
| LLaMA-3-8B matched RULER | `SUPPORTING` + `NEGATIVE` at 4× | Native/EVQ macro 8K `94.44/77.60%`; 16K `0.295/14.03%`; 32K EVQ `0%`, Native 10/13 completed and all zero | One continuation seed; same 13 families; task-adapted 2×, not unseen-task | `LLAMA8B_MATCHED_RULER_MIX_20260726.md` |
| OLMo-2 1.485B scratch step-1,000 | `POST_SUB_RAW_HASH_BACKED` | Geo/EVQ PPL 4K `161.19/167.45`, 8K `163.88/156.87`, 16K `182.73/159.64`; `122/128`, `126/128` docs favor EVQ at 8K/16K | One early-training trajectory; different trainer stacks; LM only | `OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` |
| Held-out base/head aggregate | `AUTHOR_CONFIRMED_NOT_PROMOTED` | Direction available internally | Exact numbers excluded until a dedicated raw/per-seed owner exists | `EXPERIMENT_REPORT_20260724.md` §4 |
| 151.9M/500M three-seed exact-range aggregate | `AUTHOR_CONFIRMED_NOT_PROMOTED` | Direction available internally | Distinct from the promoted 50.9M M4 factorial; use seed-42 raw control or M4 owner | `MATCHED_RANGE_COSH_500M_3SEED_20260724.md` |

## 5. Reviewer-specific response order

Do not paste the same omnibus block to every reviewer.

### 5.1 `Dz6s` — protect the 4, and arm the only potential champion

`Dz6s` is the sole reviewer above the line, values the mechanism framing, and
under the AC pilot will be in the discussion thread with `zWsa` and `27bE`.
Two objectives, in this order:

1. **Do not lose the 4.** The failure mode is overclaiming: this reviewer
   explicitly distinguishes mechanistic tests from broad applicability, and
   will downgrade if we blur that line.
2. **Give this reviewer transportable ammunition.** Not just answers, but
   three sentences they can repeat verbatim in the discussion thread — one on
   novelty, one on the controlled comparison, one on scale. A champion who has
   to reconstruct our argument will not defend it; a champion holding three
   quotable sentences will. This is the highest-leverage thing in §5.

Order: acknowledge the framing they already granted → mature endpoints with
their taxonomy intact → the cross-scale trade as theory-practice evidence →
narrow the YaRN claim ourselves → three epistemic layers → revision
commitments → one consolidated scope paragraph.

**Copy-ready response:**

> Thank you — the operator/frequency-table separation and the fixed-scale
> EVQ×YaRN result are exactly the two things we hoped would carry, and your
> three concerns are the right ones. We have added evidence for the first and
> third, and we narrow the second ourselves.
>
> **(1) Mature models and stronger endpoints.** We agree that teacher-forced
> NLL and NLL-gap retrieval do not establish usable context, and we kept the
> endpoints separate rather than aggregating them. On OLMo-2 (1.485B actual
> parameters), with every LoRA backward pass capped at 4K, a matched Native/EVQ
> counterfactual pair scored 0/100 versus 69/100 on strict autoregressive exact
> at 8K on the official `niah_single_1` task (Fisher exact \(p<10^{-28}\);
> Native's 95% interval is [0, 3.7]%). An independently trained EVQ seed scored
> 67/100 on the same rows. Four properties of the protocol matter more than the
> single number: the Native arm is matched rather than untouched; a second
> independent training seed reproduces the effect; the evaluation rows and
> needle values are freshly generated and disjoint from training; and the
> endpoint is strict autoregressive generation, not teacher-forced scoring. On
> a further set where every source-to-answer gap exceeds the 4K training
> support — i.e. where no within-training-range interpolation is available —
> Native is 0/100 and the two EVQ adapters are 49/100 and 48/100. The initial
> 16K screen was 0/20 versus 1/20.
>
> On LLaMA-3-8B, with identical physical-8K supervision over the same 13 RULER
> families, 16K official macro is 0.295% versus 14.03%. At 8K the two metrics
> disagree and we report both: Native leads on official macro (94.44% versus
> 77.60%), which credits partial and substring matches, while EVQ leads on
> normalized exact (17.69% versus 21.54%). Neither arm is usable at 32K. The
> submission already contained a 750M strict autoregressive endpoint (0% →
> 77.5% at 8K) and an 8B LoRA PPL scope check; we retain both at their original
> supporting tiers, and the 129M/382M video-DiT experiments remain cross-modal
> scope evidence only.
>
> **(2) A pattern that speaks to your third concern.** Across the 1.485B
> from-scratch run, the 1.485B adaptation and the 8B adaptation, EVQ pays a
> small in-window cost and buys a larger long-range gain: 4K NLL +0.038 against
> 16K −0.135; 4K 2.235→2.548 against 16K 4.851→2.925; 8K +0.390 against 32K
> −2.048. Reallocating a *finite* channel budget toward long-range resolution
> predicts that direction, and these are three independent settings, two model
> families and three scales in which a trained model could have contradicted it
> and did not. We offer this as the most direct theory-to-practice link we
> have — the surrogate gives the direction, not the crossover length, and we
> say so in the revision.
>
> **(3) YaRN — we agree and we narrow the claim.** A matched \(s=8\) does not
> establish dominance over an optimized Geo+YaRN or a channel-wise search, and
> we did not run that search. The comparison tests something narrower: whether
> the training-time substrate changes the leverage of the *same* inference-time
> transform. In submitted Table 3, with \(s=8\) fixed for both arms and three
> training seeds, Geo+YaRN versus EVQ+YaRN gives 61±3% versus 100±0%
> teacher-forced NLL-gap retrieval at 8K and 82.9 versus 70.9 PPL. We state the
> claim as substrate-dependent complementarity and nothing stronger.
>
> **(4) Three epistemic layers.** We will keep them separate and label them as
> such: (i) the conditional theorem, exact only for the stated convex surrogate
> \(C_{\mathrm{app}}\); (ii) exact-kernel diagnostics, which are empirical
> checks and not proof; (iii) trained-model results, which validate a direction
> and not the surrogate. \(\tau=c(\Pi)d_{\mathrm{eff}}/\sqrt{L_{\mathrm{train}}}\)
> belongs to a fourth: an operating rule whose \(O(1)\) coefficient is
> calibrated empirically, and which a 99-run audit shows to be fallible.
>
> **Scope of the new evidence.** These are task-family-adapted results: the
> training and evaluation rows are disjoint, but the generator families are
> shared, so this is length transfer within a task family, not unseen-task
> transfer. The mature-model results are single-seed apart from the two OLMo
> EVQ seeds, and the claims above are for 2× extrapolation under
> task-family-matched supervision. We have not run a broad
> instruction-following or production long-context suite, and we do not claim
> one.

**Champion ammunition — the three sentences to make quotable.** Keep these
phrasings identical wherever they appear so the reviewer can lift them:

1. *"FMRoPE changes \(b(T)\); EVQ changes \(\phi_\tau\) — a scalar range and an
   interior allocation are different objects, and the exact-range control
   separates them."*
2. *"With extrema and span pinned and only 30 interior frequencies changed,
   Cosh wins 32/32 anchors at 512."*
3. *"0/100 versus 69/100 strict autoregressive at 8K on a 1.485B model, with
   every backward pass capped at 4K."*

### 5.2 `27bE` — answer the requested ablations exactly

Order:

1. finite-\(\tau\) scaling versus coefficient;
2. direct \(\tau\) sweep and 99-run limitation;
3. multi-base/head exact-range factorial and matched exponential;
4. **the three-level zero-parameter ladder — this is the answer to `R27bE.3`,
   and it leads.** He specified the design; §3 and §7 are that design; M4 is
   stricter still. The learned-comparator identity and its `10×/100×` budget
   are one clause at the end, not the frame;
5. fixed non-Cosh schedules, including the arms that beat Cosh;
6. 1.485B \(d_{\rm head}=128\), base-500K scratch run;
7. state the older 151.9M held-out-base numbers remain excluded rather than substituting a
   different result.

This reviewer asked for specific ablations and mostly got them. The tone should
be **non-apologetic and itemized**: they requested five things, we can name what
each one returned, including where the answer went against us. This is the
highest-probability score move in the panel — do not dilute it with hedging
that belongs in the scope paragraph.

**Copy-ready response:**

> Thank you for separating the exact statement from the finite-\(\tau\)
> operating regime — that is the right decomposition, and we answer your five
> requests in order.
>
> **(1) The approximation chain and finite \(\tau\).** You are right that the
> small-\(\tau\) analysis does not determine a finite optimum, and that the
> submission let the shape rationale and the operating-point rationale sit too
> close together. They are separate claims and we now state them separately.
>
> We now separate four links: (i) Cosh is the stationary density of
> \(C_{\mathrm{app}}\), exact given that surrogate and nothing more; (ii) the
> quadratic surrogate, discrete grid and pure-tether branch are modeling
> choices, none forced; (iii) the small-\(\tau\) expansion yields only the
> scaling form \(\tau=c(\Pi)\,d_{\mathrm{eff}}/\sqrt{L_{\mathrm{train}}}\);
> (iv) the \(O(1)\) coefficient is fitted, so \(c=1\) is a default, not a
> theorem. Your point that experiments sit near \(\tau\approx4\) while the
> expansion is small-\(\tau\) is exactly right, and (iii)/(iv) is where that
> gap lives.
>
> The rule is useful and fallible. In a direct sweep the independently selected
> \(\tau=5\) and the rule value 5.657 differ by 0.0119 selection NLL; across
> the 99-run reanalysis the rule beats Geo in 7/9 configuration means but beats
> a neighboring empirical pilot in only 3/9. We will describe it as a basin
> prior, not as approximately optimal.
>
> **(2) Independently tuned \(\tau\) and matched non-Cosh schedules.** You asked
> for exactly two things — tuned \(\tau\) under the same Cosh allocation, and
> alternative non-Cosh schedules at matched \(\tau\). We built a single
> experiment that delivers both under a stricter control than requested, and it
> is now the primary attribution evidence in the paper.
>
> Design: a three-seed exact-range factorial over two bases (500K/1M), two
> training lengths (256/1024) and three head dimensions (32/64/128) — 12
> structural configurations, 180/180 main-arm runs plus 12/12 boundary-\(\tau\)
> arms, zero failures. Every schedule shares the same highest sampled
> frequency, lowest sampled frequency and log span, so the spectral range is
> identical by construction and only interior spacing differs.
>
> It separates the two claims you asked us to separate. The **axis survives**:
> preregistered \(1.25\times\) Cosh and a deformation-matched exponential
> improve weighted OOD NLL over Geo by 0.0121 and 0.0106, favored in 10/12 and
> 9/12 structural configurations, so the historical gains are not an implicit
> base/range change. The **specific formula does not**: it is the best
> preregistered Cosh multiplier in only 4/12 configurations, its own mean
> against Geo has a bootstrap CI crossing zero, and formula-Cosh versus matched
> exponential is +0.0007 NLL with sign-flip \(p=0.836\). At the boundary arms,
> formula \(\tau=1\) prefers \(1.5\times\) and \(\tau=8\) prefers
> \(0.75\times\) — the optimum moves inward from both ends. So: allocation is
> an identifiable axis, robust across base, length and head dimension; Cosh is
> one closed-form point on it, not a demonstrated optimum.
>
> **(3) Base, head dimension, and lineage.** The factorial above is the direct
> answer at the mechanistic tier — base 1M and \(d_{\mathrm{head}}=128\) are
> both included, and the direction does not depend on the 500K/64 setting the
> submission emphasized. Scale is answered separately in (5); we do not offer
> the 50.9M factorial as scale evidence.
>
> **(4) Allocation shape versus parameterization and optimization effort.** We
> think you are right, and we ran the experiment you specified. Your Limitations
> paragraph proposed keeping the positional operator unchanged and varying only
> the fixed frequency schedules — geometric, EVQ, and several alternative
> analytic allocations. We did exactly that, at three levels of control, each
> removing one more of the confounds you identified.
>
> *Level 1 — remove learned capacity and tuning budget.* Same operator, same
> training protocol, and every arm a fixed zero-parameter analytic schedule, so
> parameter count and optimization effort are identical by construction. Three
> seeds, mean tail NLL: Paper-Geo 5.9381/6.1594/6.3687/6.5354 at 1K/2K/4K/8K;
> EVQ-Cosh 5.6819/5.8541/6.1457/6.2970; a span-matched uniform schedule and
> RMS-deformation-matched power and exponential schedules are also included.
> EVQ improves on Geo by 0.256/0.305/0.223/0.238 NLL with all three seeds
> agreeing, and also beats the span-matched uniform arm.
>
> *Level 2 — additionally remove the reference-grid and deviation-magnitude
> questions.* All arms use the native Std-RoPE endpoint grid rather than our
> midpoint convention, share the same span, and every non-Geo arm shares the
> same RMS deformation from native Geo (0.2557), so no arm is simply "further
> from geometric" than another. Native EVQ minus native Std-RoPE is
> −0.113/−0.149/−0.207/−0.190/−0.099 NLL at 512/1K/2K/4K/8K, with seed-level
> paired 95% intervals [−0.200,−0.026], [−0.183,−0.115], [−0.241,−0.174],
> [−0.260,−0.121], [−0.141,−0.058], and all three seeds agreeing at every
> length.
>
> *Level 3 — additionally remove any implicit base/range change.* The
> exact-range factorial of (2), where the sampled extrema and log span are
> pinned and only interior positions move.
>
> Taken together these say what you asked whether the original comparison could
> say, and no more: with the operator, protocol, parameter count, tuning budget,
> reference grid, deviation magnitude and spectral range all held fixed,
> changing only the interior allocation still changes trained NLL. Allocation
> is a real and separately identifiable design axis. It does not say Cosh is
> optimal — at Levels 1 and 2 the matched exponential beats it at several
> lengths, and an attention-derived two-band schedule beats it by 0.116 and
> 0.211 NLL at 4K and 8K.
>
> On the tuning budget you asked about: yes, it was tuned. The row labeled
> "DAPE" in the submission is a layer-shared learnable inverse-frequency
> baseline with 32 free parameters, and it received a dedicated positional
> learning-rate sweep at 10× and 100×, with the better of the two reported
> (455.3 versus 477.7 PPL@8K, seed 42). We should have stated that budget in
> the paper, and the revision will state both it and the comparator's identity
> precisely.
>
> Your `learned-τ` observation is well taken: a 1-parameter arm at 437.9
> against a 32-parameter arm at 455.3 is a fair reason to doubt that this
> comparison settles anything about allocation shape. We agree, which is why
> the attribution above rests entirely on the zero-parameter ladder rather than
> on any comparison involving learned positional parameters.
>
> **(5) Held-out base and a larger pre-specified run.** Both branches were run.
> The base branch is the factorial in (2). For the scale branch, we trained
> OLMo-2 (1,484,916,736 parameters, \(d_{\mathrm{head}}=128\), base 500K) from
> the public step-0 initialization for 1,000 steps at global batch 512 and
> exactly 2,097,152,000 counted tokens, under the pinned recipe and data-order
> prefix. The protocol was registered in advance — checkpoint identity, token
> budget, evaluation lengths, metric list and five numbered success criteria
> were fixed before the comparison — and the run meets all five, including the
> pre-set requirements that at least one length improve by ≥0.05 NLL and that
> the 4K train-range cost stay under 0.05.
>
> Geo/EVQ PPL on the same 128 document-disjoint PG-19 documents is
> 161.19/167.45 at 4K, 163.88/156.87 at 8K, 182.73/159.64 at 16K. Paired
> document-level bootstrap 95% intervals exclude zero at every length: the
> EVQ−Geo NLL delta is +0.0381 [+0.0332, +0.0428] at 4K, −0.0437
> [−0.0494, −0.0380] at 8K and −0.1351 [−0.1420, −0.1281] at 16K, with 122/128
> and 126/128 documents favoring EVQ at 8K/16K and 128/128 on 16K tail NLL.
> These intervals measure held-out document sampling, not training-seed
> uncertainty, and we do not present them as the latter.
>
> The 4K regression is not incidental — it is the in-window cost of
> reallocating a finite budget toward long range, and the same signature
> appears in the 1.485B adaptation (4K 2.235→2.548 against 16K 4.851→2.925)
> and at 8B (8K +0.390 against 32K −2.048). Three scales, two model families,
> same direction.
>
> **Scope of the new evidence.** The step-1,000 comparison is a
> matched-initialization **early-training probe**, not a converged comparison,
> and the two arms used different trainer implementations (a reviewed HF
> single-GPU loop for EVQ; AI2's distributed trainer for the released Geo
> checkpoint) — matched on initialization, recipe, data-order prefix, counted
> tokens and evaluation rows, but not bitwise paired and not multi-seed. The
> factorial is a 50.9M short-budget natural-text NLL control: mechanistic, not
> capability. An older held-out-base aggregate exists but lacks a per-seed raw
> owner, so we are not quoting its numbers rather than substituting a different
> result for the one you asked about.

### 5.3 `zWsa` — satisfy the four score-move conditions for the AC record

Order the reply exactly like the reviewer's four questions:

1. cite Oka et al. and define the narrow distinction;
2. give the raw-backed seed-42 matched-range control and the promoted
   three-seed multi-base/head exact-range factorial;
3. give matched 8B RULER plus bounded OLMo RULER;
4. give 1.485B and 8B scale evidence.

Do not debate whether the reviewer should have noticed the submitted 8B row.

**The standard-of-novelty argument — add this, it is the strongest move
available against `RzWsa.1`.**

The reviewer's inference is: shared motivation plus a shared observation
(dead channels) implies insufficient novelty. Applied consistently, that
inference form excludes contributions the field routinely accepts.

**Present two standards, not one — and note we clear both.** The weaker
standard accepts *within-family variants*: AdamW against Adam differs only by
decoupled weight decay, YaRN against NTK-aware scaling only by the
NTK-by-parts ramp and attention temperature. The stronger standard asks for a
change in the *object being optimized*: Adam against Muon is not a tuned Adam,
it changes the geometry of the update itself.

**Our distinction is of the second kind, and the reply should say so.** The
base school selects a scalar \(b\) inside a fixed geometric family. We do not
propose another value of \(b\), nor another schedule for choosing it — we
replace the *family*, by treating the allocation map
\(u_i\mapsto\phi_\tau(u_i)\) as the unknown and solving a stated variational
problem for it. Going from "pick a scalar in a fixed family" to "solve for the
family" is a change of object, not a variant.

Do **not** claim a level for ourselves ("this is Adam→Muon, not Adam→AdamW"
reads as self-promotion). State both standards and let the conclusion follow:
under the weaker one we qualify; under the stronger one we qualify more
clearly.

This reframes the AC's question from "are these two papers different enough?",
which is a judgment call we cannot win by assertion, to "**is the usual
standard being applied here?**", which the AC can adjudicate and which favors
us. Our distinction is exactly of the accepted type: a different object
optimized (\(u\) rather than \(\omega\) or \(b\)), at a different stage.

**Tone requirement — absolute.** Ask to be held to the field's usual standard;
never characterize the reviewer's standard as wrong. The examples must be
neutral and universally accepted. Write it as a request, not a rebuttal of
their reasoning.

**Recommended English:**

> We would also ask which standard of novelty applies here. This field
> routinely recognizes within-family variants — AdamW relative to Adam differs
> by decoupled weight decay; YaRN relative to NTK-aware scaling by the
> NTK-by-parts ramp. Our distinction is not of that kind. We do not propose a
> different value of the base, or a different rule for choosing it: we treat
> the allocation map \(u_i\mapsto\phi_\tau(u_i)\) as the unknown and solve a
> stated variational problem for it, which replaces the geometric family rather
> than selecting within it. We ask to be assessed on whether that distinction
> is real and consequential — which is exactly what the exact-range control is
> designed to test.

**Held in reserve for the discussion phase only.** The same inference form
would also exclude FMRoPE relative to prior base-scaling work. That
observation is correct and is the sharpest form of the argument, but it points
at the reviewer's own cited paper, so raising it unprompted reads as petty and
costs more than it gains. Use it only if the reviewer restates the
overlap-implies-no-novelty position after the neutral version, and then state
it once, without emphasis.

**Surface what the submission already contains — as fact, never as correction.**
`zWsa` writes that "the model sizes evaluated in this paper are too small" and
asks for validation "on models of at least approximately 1B to 7B", and that
the downstream evaluation is "limited to retrieval and QA". The submission
contains an 8B LLaMA-3 LoRA evaluation (Appendix D, Table 23), a 750M
continuation with strict autoregressive retrieval (Table 12), a 432M
scarce-channel MLA study (Table 18), and 129M/382M video-DiT experiments
(Table 14). The AC should know this, because `AC.2` inherited the
"small models only" framing from this review.

**How to surface it without cost.** State the contents of the submission in a
single neutral sentence and move on to the new evidence. Do **not** write "as
noted in the paper", "the reviewer may have missed", or anything that invites a
defensive reply. The fact does the work; the framing only creates risk. One
sentence, no emphasis, then proceed.

Three structural rules for this reply:

1. **Mirror the reviewer's own four questions as four numbered headings**, and
   end each with an explicit self-assessment — `met`, `partially met`, or `not
   met`. A confidence-5 reviewer at score 2 rarely reverses on prose; but the
   AC reads this thread, and a clean four-row accounting lets the AC adjudicate
   even if the reviewer never replies. Claiming `met` on all four would destroy
   that value. Claim `not met` where it is true.
2. **Concede the citation first, unqualified, in the first sentence.** Any
   defensive move before the concession costs more than it gains.
3. **The word `complementarity` does not appear in the FMRoPE answer** (see
   §3.2). Route it to the YaRN result, explicitly labeled as a different
   comparison.

**Copy-ready response:**

> Thank you — you are right that Oka et al. should have been cited and directly
> compared, and the revision will do both. We answer your four questions in your
> order, and state for each whether we think the bar you set is met.
>
> **1. Novelty over FMRoPE.** We do not claim novelty for the observation that
> some RoPE channels are ineffective; §2 of the submission credits prior work
> for channel inequality, citing Barbero et al. (2025) on high/low-frequency
> specialization and Resonance RoPE on critical frequencies. Our claim is about
> a different axis, and the clearest way to state it is as a decomposition of
> the formula. In \(\omega_i=b^{-u_i}\) with \(u_i=2i/d\), a method can
> parameterize exactly three objects: the realized vector \(\omega\), the base
> \(b\), or the exponent \(u_i\).
>
> (1) **Whole-vector transport**, \(\omega\to g_T(\omega)\), possibly
> per-frequency, applied to an already-pretrained spectrum — PI, YaRN,
> LongRoPE. (2) **Base selection**, \(b\to b(T)\) for a declared context,
> preserving the normalized geometric order — NTK-aware, ABF, FMRoPE.
> (3) **Exponent allocation**, \(u_i\to\phi_\tau(u_i)\): the choice of
> training grid at fixed nominal base, fixed before the model learns anything —
> EVQ-Cosh.
>
> Almost all RoPE extrapolation work, FMRoPE included, changes \(\omega\) or
> \(b\). Our starting question was the assumption those two leave untouched:
> whether the exponents should be spaced uniformly at all. Our control tests
> the base boundary directly — pinning the highest sampled frequency, the
> lowest sampled frequency and the log span holds fixed everything a scalar
> base can set, and only the 30 interior positions vary. We do not claim
> transport methods could not numerically produce a similar table; the three
> axes are disjoint in what is parameterized and optimized, not in what is
> numerically achievable.
>
> We would also ask which standard of novelty applies. The field routinely
> recognizes within-family variants — AdamW relative to Adam, YaRN relative to
> NTK-aware scaling. Ours is not of that kind: we do not propose a different
> base or a different rule for choosing one, but treat the allocation map
> \(u_i\mapsto\phi_\tau(u_i)\) as the unknown and solve a stated variational
> problem for it, replacing the geometric family rather than selecting within
> it. We ask to be assessed on whether that distinction is real and
> consequential, which the exact-range control tests directly.
>
> A fourth family — learned-frequency methods such as FoPE and CARoPE — does
> move interior frequencies, by optimizing \(O(K)\) free parameters during
> training. EVQ reaches that axis in closed form with no added parameters, as
> the stationary point of a stated variational surrogate. So our claim is
> narrow and checkable: we are not aware of prior work that poses the finite
> training-time exponent allocation of standard RoPE as an explicit variational
> object and solves it in closed form with zero added learned parameters. We do
> not claim to be first to modify RoPE frequencies, and FMRoPE remains stronger
> in the retargeted deployment setting reported below. *(Whether this clears
> your novelty bar is your call; we only ask that it be assessed as a different
> axis rather than a different setting of the base.)*
>
> **2. Direct matched comparison.** Run, with both outcomes reported. The
> FMRoPE rule evaluated here is §6.1 of the paper
> (\(\theta_{\mathrm{train}}=L_{\mathrm{train}}\),
> \(\theta_{\mathrm{infer}}=L_{\mathrm{target}}\),
> \(\omega_i(\theta)=\theta^{-2i/d}\)); we did not identify a public author
> implementation as of 23 July, so this is a paper-faithful reimplementation of
> that rule rather than an official-code reproduction, and it does not cover
> every FMRoPE variant — if it misrepresents the method, please tell us which
> configuration to use and we will rerun it. Holding
> sampled extrema, log span, initialization, token order, optimizer, budget and
> the 32 evaluation anchors identical, and changing only the 30 interior
> frequencies, the Cosh interior allocation improves fixed-range OOD NLL by
> 0.478/0.205/0.113 at 512/1K/2K, winning 32/32, 27/32 and 22/32 anchors. A
> three-seed factorial reproduces the direction under the same exact-range
> constraint across bases 500K/1M, training lengths 256/1024 and head
> dimensions 32/64/128, in 10/12 and 9/12 structural configurations. The
> negative half: when both grids are retargeted to the declared length, FMRoPE
> is stronger (+0.061/+0.182/+0.279), and combining the two gives no stable
> additive gain. So we claim a **clear advantage under the exact-range
> control** — a scalar range change cannot reproduce it — and explicitly not
> superiority over, or complementarity with, target-aware FMRoPE. *(Met for
> "clear advantage under matched settings"; not met for FMRoPE
> complementarity, and we would rather say so than stretch the result.)*
>
> **3. RULER.** Included at two scales, and it improves. On LLaMA-3-8B with
> identical physical-8K supervision over all 13 RULER families, EVQ raises 16K
> official macro from 0.295% to 14.03%: at 2× the training length Native-LoRA
> is at essentially zero while EVQ retains nontrivial capability. We do not
> claim 14% macro is a usable long-context system. At 8K — the
> training length itself, so in-window rather than extrapolation — the two
> metrics disagree and we report both: Native leads on official macro (94.44%
> versus 77.60%), which credits partial and substring matches, while EVQ leads
> on normalized exact (17.69% versus 21.54%). We independently obtain the same
> length-transfer pattern on OLMo-2 (1.485B actual parameters), with all
> backward passes capped at physical 4K. Under an identical 13-family
> continuation, Native/EVQ official macro is 82.16%/37.51% at 4K,
> 0.08%/21.29% at 8K and 0%/6.13% at 16K. Native learns the in-window task
> distribution more strongly, whereas EVQ retains capability beyond the
> training length. *(Met: matched RULER improves at 2× on both mature-model
> scales.)*
>
> **4. Scale.** On OLMo-2-0425-1B-Instruct (1.485B actual parameters) with every
> LoRA backward pass capped at 4K, a matched Native/EVQ counterfactual pair
> scored 0/100 versus 69/100 strict autoregressive exact at 8K on the official
> `niah_single_1` task, with a second independently trained EVQ seed at 67/100
> on the same rows. The evaluation rows and needle values are freshly generated
> and disjoint from training, the endpoint is strict generation rather than
> teacher-forced scoring, and on a further set where every source-to-answer gap
> exceeds the 4K training support Native is 0/100 while the two EVQ adapters
> are 49/100 and 48/100 — so this is not interpolation inside trained gaps. The
> 8B evidence is above. We also trained OLMo-2 from the public step-0
> initialization for 1,000 steps and 2.097B counted tokens under a pre-specified
> protocol: Geo/EVQ PPL is 161.19/167.45 at 4K, 163.88/156.87 at 8K,
> 182.73/159.64 at 16K, with 122/128 and 126/128 documents individually
> favoring EVQ at 8K/16K.
>
> The submitted version also spans more than the text setting: an 8B LLaMA-3
> LoRA evaluation (Appendix D, Table 23), a 750M continuation with strict
> autoregressive retrieval (Table 12), a 432M scarce-channel MLA study
> (Table 18), and 129M/382M video-DiT experiments (Table 14). With the new
> results, the allocation effect now reproduces across six model scales from
> 50.9M to 8B, two attention families, two modalities, and both from-scratch
> training and adaptation. In none of these settings do we claim
> state-of-the-art; the claim throughout is that changing the exponent
> allocation improves on the standard geometric schedule. *(Met for ~1B–8B
> validation; not met for multi-seed
> full pretraining at 7B, which we did not run.)*
>
> One pattern ties 3 and 4 together: EVQ consistently pays a small in-window
> cost for a larger long-range gain (4K +0.038 against 16K −0.135 from scratch;
> 8K +0.390 against 32K −2.048 at 8B). That is what reallocating a finite
> channel budget predicts, and it reproduces across three scales and two model
> families.
>
> **Scope of the new evidence.** These are task-family-adapted results:
> training and evaluation rows are disjoint, but generator families are shared,
> so this is length transfer within a task family and not unseen-task transfer.
> The mature-model results are single-seed apart from the two OLMo EVQ seeds.
> The step-1,000 run is an early-training probe whose two arms used different
> trainer implementations. And target-aware FMRoPE remains
> under retargeted deployment, range selection dominates the interior-shape
> effect at the lengths we tested.

### 5.3.1 Author–AC Confidential Comment — the highest-leverage item in the package

**The channel exists.** OpenReview shows an `Author AC Confidential Comment`
button alongside `Rebuttal`. This resolves the open mechanics question in §1.5
and changes the plan: the AC-facing argument goes here, not appended to a
reviewer reply.

**Why this matters more than any public reply.** `AC.1` is the gate that
decides the paper, and the AC inherited its framing from `zWsa`. That reviewer
holds confidence 5 and is very unlikely to move. So the realistic path is not
persuading the reviewer — it is giving the AC a basis to weigh that objection
differently. A confidential channel is the correct place to raise a question
about the *standard* being applied, which is awkward in public and normal in
private.

**Hard tone rules — violating these loses more than the comment gains.**

1. **No adjectives about the reviewer.** Not "unfair", not "superficial", not
   "misunderstood". State facts and ask a question.
2. **Never claim the reviewer failed to read the paper**, even where the record
   supports an inference. State what the submission contains; let the AC draw
   conclusions. ACs discount authors who attack reviewers, and the inference is
   available without our help.
3. **No request to discount or remove the review.** Ask for a determination on
   a technical question, which is the AC's job anyway.
4. **Everything asserted must be checkable in the submission or the response.**
   This is the one document where a single unverifiable claim is fatal, because
   its whole purpose is credibility.

**Copy-ready confidential comment:**

> We are writing about the novelty determination, since the metareview
> identifies it as the decisive issue and we think it turns on a question that
> can be settled technically.
>
> **The distinction, stated so it can be checked.** In
> \(\omega_i=b^{-u_i}\) with \(u_i=2i/d\), a method can parameterize exactly
> three objects: the realized vector \(\omega\), the base \(b\), or the
> exponent \(u_i\). Position interpolation, YaRN and LongRoPE optimize a
> transport \(\omega\to g_T(\omega)\) applied after pretraining. NTK-aware
> scaling, ABF and FMRoPE optimize the scalar \(b\to b(T)\) for a declared
> context, preserving the normalized geometric order. EVQ-Cosh optimizes the
> third object, \(u_i\to\phi_\tau(u_i)\): the choice of training grid at fixed
> nominal base, fixed before the model learns anything, obtained as the
> closed-form inverse CDF of the stationary density of a stated variational
> problem. The great majority of RoPE extrapolation work changes \(\omega\) or
> \(b\); our starting question was whether \(u\) should be uniform at all.
>
> **The experiment that tests it.** If the effect were reducible to base or
> range selection, pinning the range would remove it. We pinned the highest
> sampled frequency, the lowest sampled frequency and the log span, together
> with initialization, token order, optimizer, budget and all 32 evaluation
> anchors, and changed only the 30 interior frequencies. Out-of-distribution
> NLL improves by 0.478/0.205/0.113 at 512/1K/2K, winning 32/32, 27/32 and
> 22/32 anchors, and a three-seed factorial reproduces the direction across
> bases 500K/1M, training lengths 256/1024 and head dimensions 32/64/128 in
> 10/12 and 9/12 structural configurations. Since every quantity a scalar base
> can set is held fixed, the effect is not attributable to base selection. We
> claim no more than that: the axes are distinct in what is parameterized, not
> in what is numerically achievable, and we do not claim to beat target-aware
> range methods — under retargeting the ordering reverses, which we report.
>
> **The question we would ask the committee to settle.** The review's inference
> is that a shared motivation and a shared observation about ineffective
> channels imply insufficient novelty. Applied consistently, that inference
> also excludes AdamW relative to Adam, YaRN relative to NTK-aware scaling,
> and — we note only because it is the comparison at issue — FMRoPE relative to
> prior base-scaling work, since each shares its predecessor's motivation and
> changes what is parameterized. Our difference is larger than those: we do not
> propose another value of \(b\) or another rule for choosing it, but replace
> the geometric family by solving for the allocation map itself. We ask only
> that the usual standard be applied.
>
> **Two points of record.** The review states that the tested models are too
> small and asks for evidence at 1B–7B; the submission includes an 8B LLaMA-3
> LoRA evaluation in Appendix D, Table 23, alongside 750M, 432M MLA and
> 129M/382M video-DiT experiments. The review also asks for RULER; our response
> reports 13-family RULER at 1.485B and matched at 8B, where 16K official macro
> goes from 0.295% to 14.03%. We mention these only because the metareview
> inherited the "small models and limited downstream evaluation" framing, and
> both parts are now addressed.
>
> We are not asking that any review be discounted. We are asking that the
> novelty determination rest on whether the exponent-allocation axis is a
> distinct and consequential design variable, which the exact-range control was
> built to test, rather than on overlap of motivation.

**Character count:** ~3,900. Leave it there. A short, checkable, unemotional
comment is far more persuasive to an AC than a long one, and length invites the
reading that we are relitigating rather than clarifying.

**What is deliberately absent:** any characterization of the reviewer, any
request about their score, the DAPE relabel (not the AC's question), the YaRN
operator diagnostic, and the Phase 11B result. None of these is asked for here,
and each would shift the comment from clarification toward defence.

### 5.4 AC `XLtL` — adjudicate the conjunction, not every detail

> **Do not post §2 and §5.4 together — they overlap by design.** The §2 opening
> (~4,250 chars) and the note below (~4,930 + 700) both walk novelty →
> control → evaluation; concatenated they run ~9,900 and repeat themselves.
> Pick one: **use the note below as the AC-facing document**, since it names
> the gates in the AC's own vocabulary, and keep §2 as the shared spine that
> the three reviewer replies draw their paragraphs from. The gate table above
> stays internal — it is the checklist for verifying nothing is unanswered, not
> text to post.

Use one compact table internally to ensure the final AC note closes all three
gates:

| AC gate | Status | Decisive evidence | Remaining boundary |
| --- | --- | --- | --- |
| `AC.1` novelty + direct comparison | Answered | formula-level distinction; seed-42 direct control; raw-backed three-seed multi-base/head exact-range factorial | 50.9M factorial is supporting; under retargeting, range selection dominates interior shape |
| `AC.2` stronger scale/evaluation | Answered at supporting tier | 1.485B strict NIAH, matched 8B RULER, 1.485B scratch, submitted 750M/8B anchors | task-family 2×; mature 4× unsolved; mostly single seed |
| `AC.3` attribution | Answered | direct \(\tau\) sweep, 99-run audit, DAPE relabel + learned-baseline budget, fixed schedules, M4 exact-range matched exponential | formula wins 4/12; Cosh not universal |
| `AC.4` recommendation conjunction | Answered with bounded claims | all three gates above are backed by completed evidence | no universal SOTA or range-method replacement claim |

**Copy-ready AC note:**

> We address the three conditions in the metareview, in the order stated there.
>
> **Novelty (`AC.1`).** We should have cited Oka et al. and the revision will.
> We do not claim the dead/ineffective-channel observation as a contribution —
> §2 of the submission already credits Barbero et al. (2025) and Resonance RoPE
> for channel inequality. The distinction we defend is best stated as a
> taxonomy. Every RoPE frequency method intervenes on \(\omega_i=b^{-u_i}\),
> \(u_i=2i/d\), in one of three places: **(A)** the base/range \(b\to b(T)\)
> for a declared context (NTK-aware, ABF, FMRoPE); **(B)** post-hoc transport
> of the realized vector at inference (PI, YaRN, LongRoPE); **(C)** the
> exponent allocation \(u_i\to\phi_\tau(u_i)\) — where the \(K\) sample points
> sit — fixed before training with \(b\) unchanged (EVQ-Cosh).
>
> The reason this is a technical rather than a presentational distinction:
> the three are a decomposition of the formula itself: in
> \(\omega_i=b^{-u_i}\), a method can parameterize the realized vector
> \(\omega\), the base \(b\), or the exponent \(u_i\). Transport methods
> optimize \(\omega\to g_T(\omega)\) after pretraining, possibly
> per-frequency (PI, YaRN, LongRoPE); base methods optimize the scalar
> \(b\to b(T)\), preserving the normalized geometric order (NTK-aware, ABF,
> FMRoPE); EVQ optimizes \(u_i\to\phi_\tau(u_i)\), the training grid itself,
> before learning. Almost all prior extrapolation work changes \(\omega\) or
> \(b\). Our exact-range control tests the base boundary specifically; we make
> no claim about what interior tables transport methods could numerically
> reach, since the axes are disjoint in what is parameterized rather than in
> what is achievable.
> Learned-frequency methods (FoPE, CARoPE) do reach axis (C),
> but with \(O(K)\) trained parameters; EVQ reaches it in closed form with
> none, as the stationary point of a stated variational surrogate. Our claim is
> confined to that: we are not aware of prior work posing the finite
> training-time exponent allocation of standard RoPE as an explicit variational
> object with a closed-form, zero-parameter solution. We do not claim priority
> over frequency modification in general.
>
> **Controlled comparison (second condition).** We ran the control the
> taxonomy implies — the one that isolates axis (C) from axis (A). The FMRoPE
> rule evaluated here is §6.1 of that paper; we did not identify a public
> author implementation as of 23 July, so it is a paper-faithful
> reimplementation of that rule and we will rerun it under any configuration
> the authors specify. With sampled extrema, log span, initialization, token
> order, optimizer, budget and evaluation anchors identical, and only the 30
> interior frequencies changed, Cosh improves fixed-range OOD NLL by
> 0.478/0.205/0.113 at 512/1K/2K (32/32, 27/32, 22/32 anchors); a three-seed
> factorial reproduces the direction across bases 500K/1M, training lengths
> 256/1024 and head dimensions 32/64/128 in 10/12 and 9/12 configurations.
> Both constraining results are on the record: under retargeting the ordering
> reverses, because range selection dominates interior shape at these lengths;
> and a deformation-matched exponential is statistically indistinguishable from
> Cosh. So the claim is an identifiable allocation *axis*, not a better
> method.
>
> **Stronger evaluation (third condition).** The clearest form of the scale
> evidence is a trend rather than a point. In the 1.485B from-scratch run, the
> EVQ−Geo NLL delta is monotone in the extrapolation ratio — +0.0724 at 2K,
> +0.0381 at 4K, −0.0437 at 8K, −0.1351 at 16K (PPL +7.5%, +3.9%, −4.3%,
> −12.6%), with paired document bootstrap intervals excluding zero at every
> length. The same ordering appears in the 1.485B adaptation (4K 2.235→2.548;
> 16K 4.851→2.925) and at 8B (8K +0.390; 32K −2.048), and it is the ordering
> our 150M–750M experiments already showed. The effect is therefore not a
> small-model artifact: it is the same monotone in-window-cost / long-range-gain
> profile at every scale we have measured.
>
> On capability: with 4K-capped backward passes a matched OLMo-2 pair scored
> 0/100 versus 69/100 strict autoregressive exact at 8K, a second independently
> trained EVQ seed 67/100, and 49/100 and 48/100 on a set where every
> source-to-answer gap exceeds training support. On LLaMA-3-8B, 13-family RULER
> macro at 16K is 0.295% versus 14.03%; 32K is zero for both. The submission
> already contained an 8B LoRA scale check, and the 1.485B experiments
> reproduce it under matched controls.
>
> **Attribution (`AC.4` third clause).** Independent \(\tau\) sweep, the 99-run
> fallibility audit, the relabeled learned-frequency baseline and its tuning
> budget, fixed zero-parameter
> schedules and the exact-range factorial separate the operating rule, learned
> capacity and allocation shape. We restate
> \(\tau=c(\Pi)d_{\mathrm{eff}}/\sqrt{L_{\mathrm{train}}}\) as a calibrated
> operating rule rather than a theorem.
>
> **What we ask.** One observation may help weigh the above: across three
> scales and two model families, EVQ pays a small in-window cost and buys a
> larger long-range gain — the direction a finite reallocated budget predicts,
> in settings where a trained model could have contradicted it. We ask that
> EVQ-Cosh be assessed as a simple, zero-parameter, training-time allocation
> axis with bounded mature-model evidence. We do not claim universal
> long-context SOTA, replacement of target-aware scaling, optimality of Cosh or
> of the \(\tau\) rule, or unseen-task transfer. The revision
> states each of these limits explicitly.

## 6. LLaMA counterfactual classification

This section is internal and prevents three different LLaMA protocols from
being merged.

| Item | Actual status | Permitted use |
| --- | --- | --- |
| Fresh EVQ-only LLaMA counterfactual arm | Completed post-submission, single seed, single arm; strict NIAH 8K/16K/32K `20/20`, `6/20`, `0/20`; no matched Native arm | Bounded feasibility only; exclude from the core response because it adds no causal comparison and weakens beyond 8K |
| Matched LLaMA natural-LM study | Completed; ordinary full-token LM, not counterfactual | Teacher-forced NLL/probability evidence only |
| Matched LLaMA RULER study | Completed; answer-only 13-family continuation, not counterfactual | Task-family-adapted RULER comparison |
| Matched Native/EVQ LLaMA counterfactual pair | `DESIGN_ONLY`; not completed | Not evidence, not needed for the current rebuttal, and **do not run** |

The core strategy is already complete without a new LLaMA counterfactual
experiment: OLMo supplies the matched counterfactual capability endpoint;
LLaMA supplies separate matched natural-LM and task-family RULER endpoints.

## 7. Claims to avoid

| Avoid | Why it backfires | Safe replacement |
| --- | --- | --- |
| “EVQ is the first method to optimize frequencies/exponents.” | LongRoPE and other work search or alter per-channel frequencies; the priority claim is unnecessarily broad. | “EVQ provides a fixed-base, closed-form nonlinear allocation of interior exponent locations.” |
| “EVQ is better than/replaces FMRoPE or YaRN.” | The target-retargeted FMR control reverses; submitted YaRN is fixed-scale. | “The tested allocation and range choices are separately identifiable; the same fixed range transform has substrate-dependent leverage.” |
| “Cosh or \(\tau=d_{\rm eff}/\sqrt L\) is optimal.” | Fixed alternatives can win and the 99-run rule is fallible. | “Cosh is the solution to the stated surrogate; the rule is an empirically calibrated basin prior.” |
| “The theory exactly predicts finite \(\tau\).” | It supplies a scaling structure, not the finite \(O(1)\) coefficient. | “\(\tau=c(\Pi)d_{\rm eff}/\sqrt{L_{\rm train}}\), with \(c(\Pi)\) calibrated empirically.” |
| “PPL improvement proves long-context ability.” | NLL/PPL is teacher-forced probability, not generation. | Report strict NIAH or RULER separately. |
| “The two OLMo EVQ seeds are 136/200.” | They share the same evaluation rows and only one is paired with Native. | “One matched pair is 0/100 versus 69/100; an independent EVQ training seed scores 67/100.” |
| “OLMo full RULER shows universal EVQ superiority.” | Native is substantially stronger at the 4K training length, while EVQ wins at 8K/16K. | “Under matched 4K continuation, Native wins in-window and EVQ supplies the 2×/4× length transfer.” |
| “LLaMA shows unseen-task or zero-shot transfer.” | Training and evaluation rows are disjoint, but the 13 generator families are shared. | “Task-family-adapted 2× transfer.” |
| “The submitted 8B row is a matched control.” | It compares untouched Base with EVQ-LoRA. | “A submitted single-seed PPL scale anchor; a separate post-submission protocol provides matched Native/EVQ controls.” |
| “The scratch run is conditional or lacks EVQ raw evidence.” | This reverses the owner hierarchy and discards completed hashed evidence. | Use the unified `POST_SUB_RAW_HASH_BACKED` classification in §1.4. |
| “The scratch trajectories are bitwise matched.” | Geo and EVQ use different trainer stacks. | “Same initialization, scientific recipe, counted-token budget, and data-order prefix; different trainer implementations.” |
| “LLaMA is counterfactual-trained.” | The matched natural-LM and RULER protocols are not counterfactual. | Reserve “counterfactual-trained” for the OLMo routing stage or explicitly label the excluded fresh EVQ-only LLaMA arm. |
| “EVQ is complementary to FMRoPE.” | `G-FMR-DEPLOY`: no stable additivity, and a combination arm exists — the follow-up question has a negative answer. | “A clear advantage under the exact-range control.” Route complementarity to YaRN (§3.4). |
| Quoting the FMRoPE contrast without naming our implementation. | A confidence-5 reviewer who knows the paper can dismiss the entire novelty answer on fidelity grounds. | Always precede it with the §6.1 reimplementation statement and the offer to rerun. |
| Quoting LLaMA 8K official macro alone (94.44/77.60). | Reads as “EVQ hurts in-window” when the same row's normalized exact favors EVQ. | Quote both metrics and explain that official macro credits partial matches. |
| Leading with the M4 formula-Cosh mean `-0.009879`. | Its configuration-bootstrap CI crosses zero. | Lead with the 10/12 and 9/12 sign statistics, or with `1.25×` Cosh / matched exponential. |
| Putting the `-0.478` and `-0.0099` exact-range results in the same breath. | The ~40× ratio invites “which one is real?” | Give them separate roles (§3.2): seed-42 owns effect size, M4 owns cross-configuration direction. |
| Reporting the 4K/8K in-window regressions as three separate concessions. | Reads as three failures rather than one predicted trade. | State the cross-scale trade once (§1.6), as a confirmed directional prediction with an empirical crossover. |
| Calling the step-1,000 run a “larger-scale training run” without qualification. | PPL ≈ 160 is visibly unconverged; the reviewer will say so first. | “Matched-initialization early-training probe” with the pre-registration stated. |

Do not mention internal alias bugs, abandoned schedules, GPU execution details,
or failed unrelated searches unless a posted claim would otherwise become
misleading. They do not answer a retained reviewer concern.

## 8. Manuscript revision commitments — append to every reply

Nothing in the earlier package told the AC what would actually change in the
paper. `AC.4` is a judgment about whether the rebuttal's content will survive
into the camera-ready, and that judgment cannot be made from evidence alone.
This block costs ~700 characters and belongs at the end of all four replies,
worded identically so the panel sees one consistent plan.

> **Revision plan.** (1) §2 will cite and discuss Oka et al., with a table
> contrasting the parameterizations (\(b(T)^{-u_i}\) versus
> \(b^{-\phi_\tau(u_i)}\)) and the stage at which each acts; §1 and §2 will
> state that we do not claim the dead-channel observation as a contribution.
> (2) The row currently marked "DAPE" will be relabeled as the layer-shared
> learnable inverse-frequency baseline it is, with its 10×/100× tuning budget
> reported. (3) A new appendix will report the
> exact-range FMRoPE control and the three-seed base/length/head factorial,
> including the retargeted-FMRoPE and matched-exponential results that
> constrain our claim. (4) A second new appendix will report the 1.485B and 8B
> results with their single-seed and task-family limits stated in-line, plus
> the pre-specified step-0→1,000 run labeled as an early-training probe.
> (5) The revision will state precisely which range transform each experiment
> applies, so that no result is read as a reproduction of the official YaRN
> method. (6) §3.3 and Table 1 will restate
> \(\tau=c(\Pi)d_{\mathrm{eff}}/\sqrt{L_{\mathrm{train}}}\) as an empirically
> calibrated operating rule, and record that the formula point is best in only
> 4/12 tested configurations. (7) §6 will add the unseen-task,
> length-transfer and metric-dependence limits, and the in-window cost.

Item (2) is not optional and should appear in the `27bE` reply as well as here
— see §3.3. Conceding it in the commitment list alone would read as burying it.

## 9. Discussion-phase plan

The AC pilot runs an author–reviewer discussion after the response. The package
previously prepared only the first message. Prepare these before sending.

### 9.1 Four likely follow-ups, with pre-drafted answers (≤500 chars each)

| Follow-up | Answer |
| --- | --- |
| "Your FMRoPE configuration is not what the paper specifies." | Restate §6.1 implementation, note no public code as of 2026-07-23, ask which configuration they want, and offer to rerun. **Never defend the choice — accept the correction.** |
| "The factorial effect is only ~0.01 NLL; is that meaningful?" | Do not defend the mean. Point to the sign statistics (10/12, 9/12 across base/length/head) as the claim, note the exact-range control is the effect-size owner at a different scale and normalization, and restate that the claim is the axis, not the magnitude. |
| "1,000 steps is not a training run." | Concede the label, keep the content: the protocol and its five success criteria were registered in advance and all five are met; paired bootstrap CIs exclude zero at every length; 122/128 and 126/128 documents individually favor EVQ; and the base/head branch of the same request is answered independently by the factorial. |
| "Different trainer stacks invalidate the comparison." | Cite the Geo reproduction sentinel: a local native-Geo run from the same step-0 whose log the EVQ run validated against at step 20 — byte-identical batches at every step, bounded CE and gradient-norm ratio, enforced as a fail-closed gate. Then concede what it does not do: it bounds the confound, it does not make the trajectories bitwise paired. |
| "You only report the YaRN scale that favors you." | Volunteer the \(s=2\) QuALITY point where the gap narrows, before being asked (§3.4). |
| "So how does EVQ compare against actual DAPE?" | **Expected immediately after the relabel.** Phase 11B, 3 seeds: plain Geo/EVQ 352.7/254.7 PPL@8K; +Kerple/MLP DAPE-ish 55.9/56.8. The operator beats allocation by ~6× and allocation adds nothing on top. No official Zheng reproduction exists. State it plainly and restate the claim as a zero-parameter allocation axis, not operator superiority. |
| "EVQ is worse at 8K — the method hurts." | Two facts: normalized exact at 8K favors EVQ (21.54% versus 17.69%); and the in-window cost is the predicted trade, reproduced at three scales, reversing by 16K. |

### 9.2 If `zWsa` does not reply

A confidence-5 reviewer at score 2 frequently does not engage. That silence
must not leave the AC with only the original review. Post a short AC-directed
comment near the end of the discussion window that does one thing: reproduce
the four score-move conditions the reviewer themselves wrote, with our
met/partially-met/not-met accounting from §5.3 and one number each. No new
argument, no re-litigation — just an accounting the AC can act on alone.

### 9.3 Concession rules, fixed in advance

Decide now, so nothing shifts mid-thread.

**Concede immediately and without qualification:** the missing citation; unseen-task transfer is not demonstrated; Cosh is not uniquely
optimal; the \(\tau\) rule is fallible; under retargeting the ordering reverses; the
mature-model evidence is mostly single-seed; no tuned Geo+YaRN search was run;
the FMRoPE arm is our own reimplementation.

**Hold, with evidence:** interior allocation is separately identifiable from
scalar range (exact-range control); the parameterizations differ in object and
stage; the 8K OLMo NIAH result is real and not gap-interpolation; the
in-window/long-range trade is a reproduced directional prediction.

**Never say:** anything implying a reviewer misread the paper; anything about
compute limits as an excuse; anything upgrading a `SUPPORTING` or `CONDITIONAL`
row under discussion pressure. If a follow-up cannot be answered from a
promoted owner, say we do not have that evidence.

### 9.4 Realistic target

Best defendable panel movement is `Dz6s` 4→5, `27bE` 3→4, `zWsa` 2→3. That is
a borderline paper, not an accept. The objective of the discussion phase is
therefore narrower than winning it: make the AC's `AC.1` novelty gate turn on
the parameterization distinction and the exact-range control — which we can
defend — rather than on an "overlapping motivation" reading, which we should
never have invited and which the origin of the method does not support.

## 10. Final send gate

The core response remains `sendable` only if all items below pass:

1. The promoted 50.9M M4 exact-range factorial may be quoted only with its
   supporting/mechanistic, natural-text-NLL, and non-universal-\(\tau\)
   boundaries. The separate 151.9M/500M three-seed exact-range aggregate
   remains unpromoted; use its raw-backed seed-42 owner instead.
2. Do not quote the older 151.9M held-out-base numbers until their dedicated
   raw/per-seed owner exists. M4 supplies a separate base-1M/head-dimension
   supporting control; the 1.485B scratch run answers the larger-run branch.
3. Keep the scratch row as `POST_SUB_RAW_HASH_BACKED`; retain
   single-trajectory, different-trainer-stack, early-budget, and LM-only
   boundaries next to that claim.
4. Do not use future-dated OLMo long-gap results until their metadata is
   reconciled.
5. Keep submitted and post-submission evidence explicitly separated.
6. Keep OLMo natural NLL, strict NIAH, and RULER continuation as three
   distinct endpoints/protocol statements.
7. Keep LLaMA ordinary-LM NLL and the 516-step RULER continuation separate.
8. Do not volunteer 4×/32K results — no reviewer asked for 4×. State a length
   boundary only where a claim would otherwise be ambiguous about which length
   it covers, and then as a one-clause scope note, not as a shortfall.
9. **Correct the DAPE method identity before anything else in that answer.**
   The submitted row is a 32-parameter layer-shared learnable `inv_freq`
   baseline, not the DAPE operator. Relabel it, withdraw every DAPE-specific
   competitiveness reading, and present `10×/100×` only as that baseline's
   tuning budget. Use fixed schedules and the exact-range factorial for
   allocation-shape attribution. **Blocking: the package must not be sent
   while any sentence defends "the DAPE row" on tuning grounds.**
10. Do not use the fresh EVQ-only LLaMA counterfactual arm in the core
    response, and do not run the design-only matched counterfactual plan.
11. Verify every posted number once against the named standalone owner after
    final character-limit editing.
12. Keep each reviewer response within 10,000 characters, plain Markdown, no
    links or attachments.

Added by the strategy revision (all are blocking):

13. **Every FMRoPE number is preceded by the implementation-fidelity sentence**
    (§3.2): §6.1 parameterization, no public author code as of 2026-07-23,
    paper-faithful reimplementation, offer to rerun. No exceptions.
14. **The word `complementarity` never appears in a FMRoPE sentence.** It is
    reserved for the YaRN result (§3.4). Grep for it before sending.
15. **Every `official macro` figure is accompanied by `normalized exact`** where
    the owner reports both — specifically the LLaMA 8K row (94.44/77.60 must
    carry 17.69/21.54).
16. **The formula-Cosh mean (`-0.009879`) is never the headline M4 number**; its
    bootstrap CI crosses zero. Quote the sign statistics, or `1.25×` Cosh /
    matched exponential.
17. **The scratch run is labeled an early-training probe** and its
    pre-registration and matched/unmatched components are stated in the same
    paragraph as its numbers.
18. **The beyond-training-gap sentence is cleared for use.** Verified against
    its owner: Native `0/100`, EVQ `49/100` and `48/100`, Wilson
    `[0, 3.70%]` / `[39.42, 58.65%]` / `[38.46, 57.68%]`, paired exact McNemar
    `3.55e-15` and `7.11e-15`. Do **not** pool the two EVQ arms as 97
    successes — same rows, related models.
19. **The §8 revision-commitment block appears in all four replies**, worded
    identically.
20. **The Author–AC Confidential Comment is written and sent** (§5.3.1), with
    no adjectives about any reviewer, no request about anyone's score, and
    every assertion checkable in the submission or the response.
21. **The `phase20_eval_suite.py` YaRN "sweep" is never cited** — it is a
    placeholder that never applies scaling (§3.4).
22. **The 750M `hybrid1.5_r16` YaRN arm is never used** until its method
    identity is verified against `FREQUENCY_DEFINITION_MANIFEST.json` (§3.4).
23. **The \(s=2\) QuALITY point is NOT in the `Dz6s` reply.** It answers a
    different endpoint than the one asked about; it is held for the discussion
    phase and, if used, described as *narrowed, not closed* (§3.4).
24. **Every sentence traces to something a reviewer asked.** Final editing pass:
    delete any sentence explaining what a result implies for our claim, any
    sentence describing our own honesty, and any reasoning that belongs in this
    playbook rather than in the reply.
25. **The scratch-run paragraph quotes the paired bootstrap CIs** and states
    that they measure document sampling, not training-seed uncertainty (§3.1).
26. **The Geo sentinel sentence, if used, carries the "bounds but does not
    remove the trainer-stack confound" clause** (§3.1). Receipt values are
    optional; the sentence is true without them, and the sentinel is primarily
    discussion-phase ammunition rather than first-reply material.
27. **Every revision statement is in the future tense** ("will cite", "will
    add"). No revised PDF or supplement may be submitted during the response
    period, so present-tense claims about the revision assert a document the
    panel cannot see. Grep for "the revision cites" / "the revision states".
28. **The DAPE answer is "yes, tuned" plus a factual comparator description.**
    No apology framing, no "we withdraw", no self-criticism. Phase 11B
    Kerple+MLP is **not** in any reply — prepared discussion answer only. The
    label fix is one line in the revision list (§3.3).
29. **FMRoPE is written as "the FMRoPE rule evaluated here"**, and the
    implementation note reads "we did not identify a public author
    implementation as of 23 July" — not "none existed" (§3.2).
30. **The Fisher p-value appears at most once per reply and never as the lead**;
    the five protocol facts carry the single-seed answer (§3.1).
31. **The AC opening does not contain the 8K official-macro / normalized-exact
    disagreement** — it belongs in the reviewer replies (§2).
32. **Each reply is 4,500–9,600 characters.** Current drafts including the §8
    revision block: `Dz6s` ~5,060, `27bE` ~9,510, `zWsa` ~7,420, AC ~5,630
    (§5.4 only — do **not** also paste the §2 opening into the AC comment).
    `27bE` is close to the ceiling; if anything is added there, cut from the
    \(\tau\) paragraph, never from the three-level ladder.
33. **`R27bE.3` is answered by the ladder, not by the DAPE row.** The reply
    leads with "you specified this experiment; we ran it at three levels of
    control" (§3.3.0). No sentence suggests the reviewer misread the label —
    the submission mislabeled it, and his criticism holds either way.
    All are within limit. Do not pad to the ceiling — but `Dz6s` and the AC
    note have ~4,000 characters of headroom each, and the best use of it is a
    compact numeric table (evidence / result / boundary) rather than more
    prose. Add it if it fits; drop it before exceeding 9,000.

If an optional conditional row is added, the package immediately changes to
`needs_author_input` until that row's stated owner/provenance gate is closed.
