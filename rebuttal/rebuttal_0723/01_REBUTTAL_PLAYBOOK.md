# EVQ-Cosh Rebuttal Playbook

Last updated: 2026-07-24. Status: **internal authoring guide; not a response to paste verbatim**.

This is the single reviewer-facing decision ledger for Submission 11628; the
official concerns live in `00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`. Standalone
reports own evidence details; this file owns the hook, ordering, routes, and gates.

## 1. Decision summary

> RoPE supplies only \(K\) frequency pairs. Our question is how those finite
> channels should be allocated on the training-time log-frequency grid.

EVQ-Cosh is not base selection. Range-scaling and range-transport methods act
primarily on the support used by an existing geometric exponent order,
\(\omega_k=B(T)^{-u_k}\). EVQ instead optimizes the non-geometric interior
allocation, \(\omega_k=B^{-\Phi_\tau(u_k)}\), before training. Its novelty is
the optimization object, the stated variational construction, and a closed-form
inverse-CDF schedule with zero learned parameters.

The exact-range control is the experimental identity test: sampled highest and
lowest frequencies, log-frequency span, model, initialization, token order,
optimizer, training budget, and evaluation anchors are identical; only the
\(K-2\) interior frequencies change. At seed 42, fixed-range paired OOD NLL
improves by \(0.478/0.205/0.113\) at 512/1K/2K. Thus finite-channel allocation
cannot be reduced to scalar range selection. Target-aware FMRoPE remains
stronger after retargeting; this limits deployment advantage, not method identity.
Cosh empirical all-schedule optimality was never claimed; \(\tau=d_{\mathrm{eff}}/\sqrt{L_{\mathrm{train}}}\) is only an operating prior.
OLMo-2 1B is the most important pending evidence; the scale gap is not closed.

The score-changing evidence order is:

1. exact-range identification, with seed 42 complete and seeds 137/256
   registered as confirmation;
2. fixed zero-parameter shape controls and native-span controls;
3. three-seed jointly held-out base/head robustness;
4. the \(\tau\) selection diagnostic and broader held-out audit;
5. target-aware FMRoPE as the required deployment boundary;
6. historical scarce-channel MLA evidence;
7. pending OLMo-2 1B scale transfer.

## 2. Paper hook and method identity

The same hook must lead the summary, AC.1 response, evidence ledger, and send
gate:

> EVQ-Cosh treats the finite training-time RoPE frequency grid as an allocation
> problem. It changes where a fixed number of channels are placed inside
> log-frequency support and derives one allocation family from a variational
> surrogate. It does not redefine the RoPE operator or select an inference-time
> target range.

### 2.1 Two design axes

The following formulas are schematic method-family descriptions:

\[
\text{NTK-aware / YaRN / FMRoPE:}\qquad
\omega_k=B(T)^{-u_k},
\]

\[
\text{EVQ-Cosh:}\qquad
\omega_k=B^{-\Phi_\tau(u_k)}.
\]

The first family primarily changes scale, range, or train-to-target transport
for an inherited geometric exponent order. EVQ changes the finite set of
interior exponent locations used during training. These are the
**range-scaling / range-transport axis** and the **finite-grid allocation
axis**. Calling them separate axes does not predict additive gains.

### 2.2 Prior-art boundary

| Prior-art family | Main object changed | Stage and search cost | Boundary relative to EVQ-Cosh |
| --- | --- | --- | --- |
| NTK-aware scaling, YaRN, FMRoPE | Scale, range, or target transport of the RoPE spectrum | Usually specified for context extension or target-aware deployment | Does not construct a variational training-time interior allocation |
| LongRoPE / LongRoPE2 | Target-specific per-dimension rescaling, search, and adaptation; LongRoPE2 uses PPL-guided evolutionary search | Applied to a pretrained checkpoint and target context, with search/adaptation | Can produce nonuniform per-dimension scales, but answers a post-hoc target-extension problem rather than a pretraining allocation problem |
| Learned-frequency methods | \(O(K)\) frequency or frequency-scaling parameters | Learned during optimization | Tests learnable frequency freedom; EVQ fixes a one-parameter closed-form grid before training and adds no learned parameters |
| EVQ-Cosh | Non-geometric placement of the \(K-2\) interior channels inside log-frequency support | Closed-form inverse-CDF construction before training | New optimization object plus a variational, zero-learned-parameter realization |

Use one novelty standard consistently. Earlier base modification does not erase
the contribution of a new FMRoPE analysis or selection rule; likewise, earlier
observations that frequency usage matters do not cover EVQ's optimization
object or variational construction. This is a scope distinction, not a
criticism of NTK-aware scaling, YaRN, FMRoPE, LongRoPE, or learned frequencies.

Concurrent post-submission LeRoPE (arXiv:2607.10134) independently reports
language-modeling value from learning one scalar per frequency up to 2.5B
parameters. It supports the importance of frequency-grid optimization, but it
appeared after submission, learns \(O(K)\) frequency parameters, and does not
validate EVQ-Cosh or raw extrapolation. It cannot replace our pending 1B
evidence.

Repository experiment names must remain protocol-accurate. The submitted
“YaRN” row is the repository-defined fixed-index smooth-ramp scaler, not an
official YaRN implementation. The local FMRoPE control follows the published
base/range rule, but no response should imply that its authors' code was used.

## 3. Pre-submission research continuity

The project did not begin with EVQ-Cosh or with the FMRoPE comparison. Before
submission it explored **anchored-sigmoid frequency warping**: retain a
high-frequency anchor and smoothly deform the low-frequency grid. The schedule
was an empirical frequency-grid intervention, not EVQ-Cosh and not evidence for
the Cosh surrogate.

The same line was adapted to Qwen2.5-7B-Instruct with Q/K/V/O LoRA and evaluated
under a fixed full LongBench-21 protocol. Its result is a preserved boundary,
not a positive benchmark: seed 42 changed the average from 44.44 to 44.08
(\(-0.35\)); seed 1337 changed 44.47 to 44.05 (\(-0.42\)). Retrieval-oriented
tasks improved in places, while multi-hop tasks such as 2WikiMQA, Musique, and
HotpotQA declined; the FDR result is `no_improvement`.

This trajectory has three legitimate uses:

1. it records a pre-submission program on frequency-grid optimization rather
   than a distinction invented in response to FMRoPE;
2. it shows the progression from empirical anchored warping to a principled
   variational allocation;
3. it shows that mature-model adaptation and downstream transfer were explored
   and that their negative boundary was retained.

It is not scientific proof of novelty, not positive 7B downstream evidence,
and not an EVQ-Cosh result. Git history may support internal chronology, but it
must not replace the technical novelty argument or be linked proactively in a
reviewer response.

## 4. Linchpin: exact-range identification

The exact-range control is not one more ablation. It is the controlled
identification experiment for AC.1:

- sampled highest frequency: exactly identical;
- sampled lowest frequency: exactly identical;
- log-frequency span: exactly identical;
- model, initialization, token order, optimizer, training budget, and frozen
  evaluation anchors: exactly identical;
- only the \(K-2\) interior frequencies differ.

\[
\boxed{\text{same sampled support}+\text{different interior allocation}
\Longrightarrow\text{different trained OOD behavior}}
\]

At seed 42 and fixed range, Cosh minus uniform FMRoPE is
\(-0.478/-0.205/-0.113\) paired NLL at 512/1K/2K. Cosh wins 32/32, 27/32, and
22/32 frozen anchors. The supported identification claim is:

> Finite-channel allocation cannot be reduced to scalar base/range selection.

After both grids are target-retargeted, the differences reverse to
\(+0.061/+0.182/+0.279\). Uniform FMRoPE is then stronger. Preserve this
boundary because it separates method identity from deployment ranking:

- the fixed-range result identifies an interior-allocation effect;
- the retargeted result shows that this Cosh grid does not improve the tested
  target-aware FMRoPE deployment;
- neither result establishes additive gains or empirical independence;
- neither result establishes universal empirical optimality of Cosh.

Seed 42 is complete. Seeds 137 and 256 remain registered confirmation runs.
Until both finish, all exact-range wording must say **single-seed diagnostic**,
not three-seed result.

## 5. Score-changing evidence ledger

Paper tier and rebuttal role are separate fields. The ledger is ordered by its
ability to answer the AC, not by experiment date.

| Rank / ID | Concern and response role | Artifact and status | Protocol and result | Safe reviewer takeaway | Mandatory boundary |
| --- | --- | --- | --- | --- | --- |
| 1. `E-MATCHED-RANGE` | `AC.1`, `AC.3`; identifies method object | `theory_results/MATCHED_RANGE_COSH_500M_S42_20260724.md`; `RAW_BACKED`; seed 42 complete, seeds 137/256 registered | Identical sampled extrema/span and matched training protocol; only \(K-2\) interior frequencies differ. Cosh minus uniform FMR is \(-0.478/-0.205/-0.113\) NLL at 512/1K/2K; anchor wins 32/32, 27/32, 22/32. After target retargeting, differences are \(+0.061/+0.182/+0.279\) | Finite-channel allocation is not reducible to scalar base/range selection | Single seed until registered runs finish; target-aware FMR is stronger; no additive, large-model, or all-schedule claim |
| 2. `E-SHAPE` | `R27bE.3`, `R27bE.4`, `AC.3`; isolates fixed zero-parameter shape | `theory_results/EXPERIMENT_REPORT_20260724.md` §3; `REPORT_BACKED`; post-submission three-seed ablation | EVQ minus Paper-Geo at 1K/2K/4K/8K is \(-0.256/-0.305/-0.223/-0.238\) NLL; all seeds agree. Matched exponential is also competitive | Allocation shape is a distinct tested variable; Cosh is one analytic instance | Competitive non-Cosh schedules do not contradict the stated surrogate theorem |
| 3. `E-NATIVE-SPAN` | `R27bE.3`, `AC.1`, `AC.3`; removes midpoint/span confounds | Report §7 and native-shape JSON; `AGGREGATE_BACKED`; post-submission three-seed mechanism check | Native endpoint grid with matched span/deformation: EVQ minus Std-Geo is \(-0.113/-0.149/-0.207/-0.190/-0.099\) NLL at 512/1K/2K/4K/8K; all paired CIs exclude zero | The effect is not a Paper-Geo midpoint artifact; nonlinear interior allocation remains consequential after matching span | Alternative shapes can be stronger at some lengths |
| 4. `E-HELDOUT` | `R27bE.2`, `R27bE.5`, `AC.2`; tests base/head transfer | Report §4; `REPORT_BACKED`; post-submission three-seed robustness | Jointly held-out base \(1\)M and \(d_{\mathrm{head}}=128\): EVQ minus Paper-Geo is \(+0.069\) at 512 and \(-0.802/-0.664/-0.433/-0.287/-0.212\) NLL at 1K/2K/4K/8K/16K; seed-level CIs exclude zero | Gain is not confined to base 500K or \(d_{\mathrm{head}}=64\) | One jointly held-out 151.9M configuration; small in-domain cost; scale gap remains open |
| 5. `E-TAU` | `R27bE.1`, `R27bE.4`, `AC.3`; bounds the operating rule | Report §2 and `theory_results/PHASE16_99RUN_RAW_REANALYSIS_20260724.md`; `REPORT_BACKED` | On selection anchors the predefined formula point is \(0.0119\) NLL from selected \(\tau=5\). Across the staged nine-configuration study it beats Geo in 7/9 means but the selected neighbor in only 3/9 held-out comparisons | The rule is a useful, fallible basin prior | \(0.0119\) is not held-out; the rule is not a point predictor or trained-model optimum |
| 6. `E-FMR-500M` | `AC.1`; records deployment boundary | Report §6; `REPORT_BACKED`; post-submission single-seed diagnostic | 151.9M, seed 42, 500M tokens: raw EVQ minus Paper-Geo is \(-0.360/-0.221/-0.130\) NLL at 512/1K/2K; target-base FMR beats raw EVQ by \(1.271/1.946/1.570\) | Training-time allocation and target-aware range transport answer different decisions | Different deployment contracts; no ranking claim across contracts |
| 7. `E-PRIMARY-III` | `R27bE.2`, `AC.2`; historical scarce-channel context | Provenance M3, `data/curated/eval_3seeds_full_results.json`, and `data/curated/table18_mla_3seed_aggregate.json`; `CURATED_RAW_BACKED_WITH_SCOPE` | 432M MLA, 8K/500M, 16 rotary frequency pairs, three seeds | Existing evidence that allocation sensitivity can increase under a scarce rotary budget | Architecture-specific \(d_{\mathrm{eff}}\); not production-identical MLA |
| 8. `E-OLMO-1B-PENDING` | `AC.2`; tests scale transfer | Pending; no result may be quoted | OLMo-2 1B from public step 0; train only the EVQ intervention branch; compare with the released matched Geo milestone; first gate at 2.1B tokens | If positive, directly tests full-RoPE, large-base, approximately 1B transfer | Pending evidence only; do not say the scale gap is closed |

## 6. Concern-by-concern response route

### AC.1: novelty relative to FMRoPE and dead-frequency observations

Use this order without detours.

**First, define the object.**

> FMRoPE changes the base/range used by a geometric exponent order,
> \(\omega_k=B(T)^{-u_k}\). EVQ-Cosh instead changes the finite training-time
> interior allocation, \(\omega_k=B^{-\Phi_\tau(u_k)}\), through a closed-form
> inverse CDF with zero learned parameters. The new object is the placement of
> a fixed number of channels inside log-frequency support.

**Second, identify it experimentally.**

> We matched the sampled highest and lowest frequencies, log-span, model,
> initialization, token order, optimizer, training budget, and evaluation
> anchors. Only the \(K-2\) interior frequencies differ. At seed 42 and fixed
> range, the Cosh allocation improves paired OOD NLL by 0.478/0.205/0.113 at
> 512/1K/2K. This rules out scalar range selection as an equivalent
> description of the intervention.

Until seeds 137/256 finish, introduce that paragraph as a seed-42 diagnostic.
After they finish, replace it only with the registered aggregate.

**Third, position prior work fairly.**

> Prior work established the importance of frequency range, under-used bands,
> target-specific rescaling, and learned frequencies. Our contribution is the
> constructive training-time problem of allocating a finite channel budget,
> together with a variational, closed-form, zero-learned-parameter instance.

**Fourth, state the deployment boundary.**

> When both grids are target-retargeted, uniform FMRoPE is stronger. We
> therefore use the exact-range control to establish method identity, not
> superiority over target-aware range transport.

The anchored-sigmoid/Qwen trajectory may be mentioned only if a reviewer
explicitly suggests that the allocation framing was invented after submission.
Even then, use it as chronology and preserve its negative downstream boundary;
the novelty argument must remain the optimization object, construction, and
exact-range evidence.

### R27bE.1 / R27bE.4 / AC.3: surrogate, Cosh, and \(\tau\)

Treat these as epistemic boundaries, not withdrawn claims:

1. **Surrogate theorem.** Under the stated positivity, normalization, and
   function-space conditions, Cosh is the unique minimizer of the convex
   \(\mathcal C_{\mathrm{app}}\). The theorem is not about the oscillatory
   kernel, attention, LM loss, or every empirical schedule.
2. **Closed-form branch.** The deployed grid is the homogeneous pure-tether
   solution of the stated variational surrogate. Fisher-forced extensions are
   separate and are not claimed to vanish.
3. **Empirical schedule family.** Competitive exponential or other shapes are
   compatible with the paper's claim. No empirical all-schedule optimum was
   proposed.
4. **Operating prior.** \(\tau=d_{\mathrm{eff}}/\sqrt{L_{\mathrm{train}}}\) is
   a practical basin default motivated by the small-\(\tau\) scaling argument.
   It is not a global optimum or an exact point prediction for trained models.

Report both sides of `E-TAU`: \(0.0119\) on selection anchors from selected
\(\tau=5\), and only 3/9 held-out wins over the selected neighbor in the broader
study. Pair it with `E-SHAPE`, where Cosh improves Geo under the tested protocol
while a matched exponential remains competitive.

### R27bE.3: learned-frequency/DAPE comparison

Lead with fixed schedules, not capacity:

> Learned-frequency comparisons mix allocation freedom with learned capacity
> and tuning. We therefore compare fixed zero-parameter schedules under the
> same RoPE operator, initialization, token order, optimizer, and evaluation
> protocol. Cosh and matched non-Cosh schedules improve the geometric control,
> isolating finite-grid shape without adding trainable frequency parameters.

Use `E-SHAPE` and `E-NATIVE-SPAN`. The historical learned-frequency row is not
the primary attribution evidence.

### R27bE.2 / R27bE.5 / AC.2: base, head dimension, scale, and evaluation

Lead with `E-HELDOUT`, then state immediately that it is one 151.9M
configuration. Use historical Primary III only within its registered
scarce-channel scope.

The remaining scale answer is `E-OLMO-1B-PENDING`:

- start from the public OLMo-2 1B step-0 state;
- train only the EVQ intervention branch;
- compare against the released matched Geo milestone;
- test transfer to full RoPE, a large base, and an approximately 1B
  architecture;
- treat 2.1B tokens as the first gate;
- continue to later public Geo milestones only after a positive first gate.

Until a report and provenance entry exist, this is a plan, not evidence.

### AC.4: what can change the recommendation

The response package is: method identity, exact-range identification, fixed
shape attribution, held-out base/head transfer, bounded \(\tau\) evidence, and
an explicit scale-status statement. Do not expand the response into a new
method program.

## 7. Response exclusions

Omit unless a reviewer directly triggers them:

- detailed anchored-sigmoid implementation, Qwen task-by-task values, or Git
  chronology; their only rebuttal role is bounded research continuity;
- internal adversarial or simulated reviews;
- 8B readout/sparse-conversion diagnostics;
- fresh scarcity, band-pruning, residual, T-PIG, or other exploratory tracks;
- GPU runtime, cleanup, and implementation-plan details;
- broad pre-rebuttal audit history.

Do not claim:

- that separate design axes must combine additively;
- that EVQ improves target-aware FMRoPE deployment;
- that Cosh is the empirical optimum across real schedules or objectives;
- that the \(\tau\) rule predicts the trained-model optimum;
- that the held-out 151.9M study or pending OLMo run closes the scale/downstream
  gap;
- that anchored-sigmoid or Qwen LoRA is positive EVQ evidence;
- that LeRoPE validates EVQ-Cosh or substitutes for our scale experiment.

Do not introduce EVQ-v2, a new parameterization, or an optimizer analogy into
the formal response.

## 8. Send gate

Before an author response is sent:

1. the first novelty paragraph must define finite training-time grid allocation
   before discussing any baseline result;
2. every number must resolve to the exact standalone report and retained
   provenance in the ledger;
3. exact-range wording must preserve all matched variables and state that only
   \(K-2\) interior frequencies change;
4. seeds 137/256 must remain `registered confirmation` until complete; seed 42
   must not be presented as a three-seed result;
5. the target-retargeting reversal must stay adjacent to the exact-range claim,
   so method identity is not confused with deployment superiority;
6. FMRoPE, NTK-aware/YaRN, LongRoPE/LongRoPE2, learned-frequency work, and
   directly relevant frequency-usage observations must be cited from primary
   sources and distinguished by optimization object and stage;
7. Cosh wording must remain conditional on \(\mathcal C_{\mathrm{app}}\), and
   \(\tau\) wording must remain an operating prior;
8. anchored-sigmoid/Qwen evidence may establish internal chronology only; its
   negative LongBench-21 boundary must not be hidden if it is mentioned;
9. OLMo-2 1B must remain pending until the 2.1B-token gate has a standalone
   report, raw metrics, and provenance; do not substitute concurrent work for
   this result;
10. no paper table value is changed merely because a post-submission diagnostic
    exists.

The final response succeeds if a reviewer can follow one chain:

\[
\text{finite-grid question}
\rightarrow \text{variational closed-form allocation}
\rightarrow \text{exact-range identification}
\rightarrow \text{bounded robustness evidence}
\rightarrow \text{honest deployment and scale limits}.
\]
