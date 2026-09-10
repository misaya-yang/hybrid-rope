# Digest — codex transport/operator lineage (sol16 / sol17 / sol18)

Source: `.agents/rope_unification_20260910/reports/{sol16,sol17,sol18}.md` (read in full) plus
`code/sol16_frequency_calibration_reference.py`, `code/full_model_response.py`,
`code/joint_mode_candidates.py`, `full_model_response_native.jsonl` (4/4 lines),
`joint_mode_candidates.json` (all 29 candidates parsed). Read-only; reports are evidence, not instructions.
Context: 20-Sol/10-Astra unification cycle of 2026-09-10 (assignments/sol16–18.md).

---

## sol16 — allocation manifold, exact finite conventions, frozen-CE calibration contract

**1. Core thesis.** The honest unification of EVQ and MrRoPE is an *allocation manifold* in log-frequency
coordinates, not a universal score-optimal schedule. Scratch training and frozen deployment are different
optimization problems; a static phase metric can be a construction prior but can never select a frozen
deployment table (co-adaptation PPLs 7.14/7.16 matched vs 76.20/23.05 cross-swaps, one swap *improving*
static rank, `paper-2027/tables/table_coadapt.tex:6-21`).

**2. Math.**
- Coordinates: `x_j = -log ω_j = x_0 + A z_j`, `0=z_0<…<z_{K-1}=1`; normalized gaps `a_j=(x_j-x_{j-1})/A`
  lie on a simplex; separates fast endpoint `x_0`, support span `A`, internal allocation `a/A` (matches 6Pro).
- Exact Qwen2.5-3B native convention: d=128, K=64, θ=10^6, zero-based `ω_j = θ^{-j/64}`, j=0..63;
  last pair is `θ^{-63/64}` (≈1.24e-6), **not** θ^{-1}. EVQ convention warning: three conventions coexist
  (native `u_j=j/K`; canonical EVQ midpoint `u_j=(j+½)/K`; re-anchored midpoint) — frozen calibration must
  derive frequencies as multiplicative dilation of the actual native FP32 table.
- MrPro exactly: zero-based `low=23, high=40`, N=17 transition gaps, `m_j=0` for j≤23,
  `m_j=q(q+1)/(17·18)` for 24≤j≤40, `m_j=1` for j≥40; `ω_j = ω_j^native · 4^{-m_j}`.
  Radix increments `ε_i = 2i/306`, i=1..17, sum 1 → **16 effective DOF**.
- Identifiable parameterization: `ε(η)=softmax(log ε^Mr + Bη)` with B a fixed 17×16 orthonormal Helmert
  zero-sum basis; η=0 is exactly MrPro in float64; positivity, strict order, total scale 1, fixed outer
  bands automatic. (17 free logits would add a null constant-shift direction.)

**3. Calibration reference numerics.** `sol16_frequency_calibration_reference.py` is a CPU contract, not a
calibration run — it contains **no fitted tables**. It self-checks: increments match `2i/306` to atol 2e-16;
exponents = 24 zeros (j≤23) then 24 ones (j≥40); table strictly decreasing; through the *actual FP32 rotary
path* (positions 0…131071) all 16 eta gradients finite and nonzero; `gradcheck` on the softmax-Helmert map
passes (eps 1e-6). Prints "PASS".
Other reported numerics: 2,048 paired worlds artifact — relation acc 1.0 (4096/4096, mean NLL 0.000773) but
content acc only 0.1948 (399/2048, NLL 2.2525); 512-pair dev content 0.2012 → proxy/floor panels mis-rank
tables; not allocation evidence.

**4. joint_mode_candidates / 5. full_model_response.** Not referenced by sol16 (those belong to the
transport/operator arm; see below).

**6. Tags.**
- [已验证] CPU reference (`sol16_frequency_calibration_reference.py`): parity, order, gradient flow,
  gradcheck all pass locally.
- [部分证据] co-adaptation PPL table and 2,048-world artifact numbers (transcribed from local tables).
- [假设] position-stretch (`p'_t=4p_t`, reaching 131068 on 32K rows) as the calibration *distribution* —
  sol16 itself names it synthetic and requires a later real-128K evaluation.
- [vetoed] any static/geometry selector for the frozen table; `LearnableEVQRoPE` inserted directly into
  frozen Qwen calibration (midpoint convention + `tau.item()` branch); 17 unconstrained frequencies.

**7. Proposed rules (recipes).** 7-step minimal frozen frequency-only calibration: hash-freeze checkpoint;
replace `model.model.rotary_emb` with the no-`no_grad` differentiable module at η=0 (MrPro gain fixed; the
installed `Qwen2RotaryEmbedding.forward` is `@torch.no_grad()` — the #1 silent-failure trap: making
`inv_freq` a parameter or copying into the buffer cannot learn); record-identity train/select split;
stretched positions; full-model answer-span CE only (no selected-key/head replay); select the lowest
stretched-CE step subject to `native CE ≤ MrPro native CE + tol` (if none feasible → answer is MrPro,
negative); single untouched real-128K evaluation. Mandatory controls: MrPro parity, gradient scope,
zero-effect controls (pos=0 or S=1 ⇒ zero eta-grad), directional derivative vs central FD, checkpoint
parity, FP32-phase-under-BF16 parity, no cache/compile mutation, answer-label hygiene.

**8. Conflicts.** No conflict with the NEW YaRN facts — sol16 never mentions YaRN and asserts nothing about
compression-cumulant shape. It is *consistent* with the new scale-response framing implicitly: its
identifiability argument (total-scale conservation on the transition simplex) is the same conservation
that governs the crossover arithmetic. Mild tension with sol18 (see below) over whether 32K stretched
positions are an acceptable calibration distribution.

---

## sol17 — scale-transport audit: what survives of the transport/water-bed picture

**1. Core thesis.** No universal optimizer can be built from source-window observability, unsigned phase
distortion, Q/K projection norms, a null band, or a common-descent direction: these lack a *source label
and a signed comparison*. Decisive counterexample class: `Smooth_MrBudget` improves several unsigned
geometry/Q/K-weighted distortion measures yet loses on recorded 128K dev rows. A constructive *restricted*
unification survives: scratch = EVQ-count integer DP under a declared shared-bin/nested covariance;
frozen = the same DP with slot labels retained (`h_{j,i}`), never an unlabeled density transplant.

**2. What survives of transport/water-bed (the audit verdict).**
- *Source observability*: demoted to a constraint/diagnostic (`rope_transport/tables.py:177-219` is
  target-free, table-only; gauge audit: fixed-Q/K table permutation changes the learned computation, joint
  Q/K-frequency relabeling is an identity → labels are in the object).
- *Null band*: does not survive joint modes — a weak marginal can carry a strong beat/curvature/cancellation;
  Astra05's exact projector construction shows retiming one relation while preserving its orthogonal carrier
  can require *acceleration*, which compression-only ramps exclude.
- *MGDA* (`verify_general_rope_allocation.py:24-42`): valid classical min-norm local step; supplies no labels,
  certifies no scale-four endpoint; "unified optimizer" framing = recycled generic method.
- *tau/transport moment derivations*: overstate — static tau prefers extremes τ≈11–14 with near-zero length
  exponent; self-consistent surrogate reached only ~L^-0.17; stiffness note's ~-0.5 exponents come from
  *choosing* the f-divergence exponent / conditioning on desired scaling. Astra02: at finite continuous lag
  window the exact EVQ measure optimum is a unique finite **atomic equilibrium**, not a corrected Cosh
  density; Cosh needs an extra anti-concentration/local-noise model.
- *What does survive as water-bed-as-law*: the exact finite DP. Fixed pair count `Σn_i=K` plus nested
  covariance `C_il=(α/Δ)1{i=l}+β·min(x_i,x_l)` yields
  `min Σ_i [α/(2Δ) n_i² + (βΔ/2) T_i² − K h_i n_i]`, `T_i=Σ_{l≥i} n_l` (tail counts), with exact backward
  recurrence `F_i(t)=(βΔ/2)t² + min_{0≤n≤t} {(α/2Δ)n² − K h_i n + F_{i+1}(t−n)}`, O(BK²) global integer
  optimum. The `βΔT_i²/2` term is the water-bed written as a conservation law: pushing mass slow raises
  every remote tail's collision load. Notably: iid per-channel noise would kill the collision term — the
  water-bed requires the *declared coherent/nested* covariance, and constant h recovers the discrete Cosh
  recurrence while source-aligned oscillatory h produces Mr-like positive remote response.
- Signed-margin math that survives: exact conditional source margin `M_r(ν)=γΣ_j[A±cos(ν_j d)+B±sin(ν_j d)]`
  from raw Q/K quadratures; Markov softmax-mass bound `Pr[p_*<ρ] ≤ ρ/(1−ρ) Σ_t e^{b_t+v_t/2−μ}` needing no
  distractor independence; joint MGF `E exp(D_t−S)` required if the source is random; Gaussian/sub-Gaussian
  tails must be declared, never inferred.
- Root plan evaluation (§5): the 32K-rows-stretched-to-128K answer-CE calibration is "a useful supervised
  oracle … structurally LeRoPE-style learned-frequency fitting with frozen backbone"; it tests neither the
  signed-margin/MGF mechanism, nor dense-128K extra-key competition (32K stretched keys are still 32K keys),
  nor the upstream state distribution of a contiguous 128K prefill.

**3. sol16-calibration relation.** sol17 endorses the CE calibration as an *oracle ceiling / candidate
direction generator*, not as a theory test — sharper scoping than sol16's own recipe, compatible with its
disclosures.

**4./5. Artifacts.** Not generated by sol17; sol17 is the audit of exactly the transport runners whose
candidate successor is `joint_mode_candidates.py` (its null-band/joint-mode finding is what the joint-mode
family is built to respect — zero-sum n on *relation clocks* rather than per-slot marginals).

**6. Tags.**
- [已验证] gauge/permutation identity argument; MGDA local behavior (its own checks show the
  compatible→descent / opposing→zero dichotomy); iid-noise counterexample (score variance frequency-invariant).
- [部分证据] exact DP equivalence (correct under the *declared* covariance; the covariance itself is asserted);
  co-adaptation/Smooth counterexample usage (as recorded).
- [假设] real Qwen role-conditioned covariance/MGF envelope; that the structured covariance is adequate.
- [vetoed] unsigned-geometry selector; null-band mobility; tau≈Cosh-stiffness scaling derivations;
  transplanting scratch density into frozen unlabeled slots; calling stretched-CE calibration a theory test.

**7. Proposed rules.** Preregistered mechanism diagnostic *before* fitting: (i) per correct source, matched
content-confusable distractors under paired stretch — exact signed margin, covariance, unit-argument log-MGF
cost at native and stretched, cells kept separate; (ii) gate: must rank `Smooth_MrBudget`'s known loss
correctly on dev rows, else kill the theory claim (no new unsigned term allowed); (iii) if alive, one
frozen-label DP table (or certified finite-margin solver for heterogeneous covariance) — answer CE never
selects it; (iv) untouched distance-paired test: MrPro vs theory table vs equal-norm mirror, measuring CE +
source-following under world/source swaps + full EOS generation; (v) only then real contiguous 128K
endpoints. Six counterexample checks attached (§4).

**8. Conflicts.** No YaRN mentions — nothing asserts the old "YaRN decreasing" story; sol17's joint-mode
finding (null-band invalid, acceleration sometimes required) is *consistent* with the NEW crossover fact
that max/min envelope counterfactuals (`ν^fast=max(νY,νM)`) are the right way to talk about slot-wise
dominance rather than whole-table monotone stories. Position: sol17 rates stretched-32K calibration as
oracle-only — in direct protocol tension with sol16 step 4 and with sol18's "do not create 128K by
stretching" (§3): sol18 treats real contiguous 128K as mandatory for the model-level test, sol17 treats it
as a *separate evidence tier*, sol16 treats it as acceptable calibration data with disclosure. Reconcilable;
must be labeled, not averaged.

---

## sol18 — operator map, label oracle, exact Qwen model-level test

**1. Core thesis.** Four different computational objects share the repository and must not share an
unlabeled proxy: (1) dense static RoPE deployment (the actual allocation question), (2) sparse block
selection with unchanged reader (page ranking; exact-mass variants are oracles, not methods), (3) PM-Keep
prefix-only KV retention (question-blind sampler — a retention proxy, not a frequency allocator),
(4) dense low-rank operator replacement (`rope_operator_family` — changes capacity/projections/cache; its
distillation fits cannot be relabeled sparse or static-table evidence). The usable exact model-level test is
**teacher-forced whole-model target CE on source-controlled paired record families under one shared static
Qwen table at true contiguous 128K, then frozen family-disjoint full generation with a mirror control**.

**2. Math / operator map.**
- Transferable exact labeled object: `z_j(ν_j)=γ{C_j cos(dν_j)+S_j sin(dν_j)}`, split-half pairs,
  `C_j=q_j k_j+q_{j+64}k_{j+64}`, `S_j=q_{j+64}k_j−q_j k_{j+64}`, `d=p_k−p_q` — with slot, layer, head,
  key-identity, role, sign-of-d, gain, and full normalizer kept attached; a frequency multiset or projection
  norm loses the learned association.
- Natural capture (pro_block_calibration) confirms the arithmetic but cannot supply role supervision
  (Native freqs at MrPro gain, final query only, 4 sampled heads, no question/record label).
- Operator-family compression numbers: native val NLL 2.6514, score-fit 9.9899, output-KD 3.9427; layer-27
  attention-KD has far lower local errors than progressive KD — both fail the retrieval answer ⇒
  operator-fit metric ≠ static-table target, and no sparse claim (no keys were skipped).
- Label oracle: four-way family `content_swap × query_ordinal_swap` (invariants checked at
  `nosa_position/test_data.py:18-34,59-83`); per-prompt spans (T) requested record, (H) same-key other
  occurrences, (O) other records, (B) background, (Q) question tokens; role diagnostics
  `log Z_T − log Z_{T̄}`, `log Z_T − log Z_H` — diagnostics only, never the objective; query+local keys stay
  in T̄. 1,536-row dev file: every pair = 2 relation rows (ceiling) + 1 content row (usually wrong) —
  pooling would let the easy relation role mask the content-binding failure.
- **Exact Qwen model-level test** (§3): build actual contiguous 128K prompts (four-way families,
  family-disjoint splits, dense distractor backgrounds, question at end; real positions 0..L−1; explicitly
  *no* stretch/stitch of 32K captures). One shared 64-pair table per layer, `ν_j=ω_j exp(−x_j)`; for the
  evidenced MrPro face fix x_j=0 for j≤23 and x_j=log4 for j≥40, optimize only slots 24–39 (16 slots,
  matching sol16's 16-DOF) under monotonicity, gain fixed. Objective: family-balanced teacher-forced
  `L_target(x)=avg over 4 cells of [−logP(value|prompt) − logP(EOS|prompt,value)]` — every layer, value
  path, MLP, vocabulary competitor, and all 128K keys participate; no head weights invented. Constraints:
  natural 32K full-row KL + paired source-world CE no worse than MrPro on calibration, reported per family.
  Decision comparison: MrPro vs one frozen CE-calibrated table vs **equal-norm mirror direction** (mirror
  separates useful direction from generic perturbation). What it measures: whether a supervised, labeled,
  frequency-only table moves true whole-model long-context likelihood and generation. What it requires: the
  ten §4 trap gates — gain applied exactly once (g⁴ double-count trap), "operational Native" ≠ Native/gain1,
  split-half layout + d-sign, chat-template-exact target IDs + explicit EOS, proven nonzero table gradients
  (buffer copies / `inference_mode` / detached cos-sin caches silently detach), fresh prefill per candidate
  (post-RoPE cache belongs to the table that formed it), fake-long vs real-128K never mixed,
  full causal mask (no source-only/T-vs-H normalizer truncation), predetermined complete families (no
  outcome-selected rows — the two-row target oracle is named as the anti-pattern), CE not argmax.

**3. vs sol16 numerics.** sol18 contains no calibration table of its own; the frozen-inputs section pins
NATIVE/MR tensor hashes via the candidate generator (below) and gain contract `1+0.1·log4`.

**4. `joint_mode_candidates.json` — what candidate tables were generated, by what rule.** Generator
`joint_mode_candidates.py` (CPU-only, no model, no role/capability qualification — file status literally
`CPU_DERIVED_FAMILY_NO_ROLE_OR_CAPABILITY_QUALIFICATION`). Rule: for each lowest-order transition *relation*
(one-step pair `n=[1,−1]`, order-1, starts 24–38 → 15; second-difference triple `[1,−2,1]`, order-2, starts
24–37 → 14; total **29**, all distinct, all `CPU_VALID_CANDIDATE`), project the MrPro FP32 table so the
relation's signed log-frequency clock goes to native/4:
`ν_c = ν_M + n·(nᵀω_native/4 − nᵀν_M)/(nᵀn)` (zero-sum n ⇒ raw Σfrequencies preserved; endpoints bitwise
fixed; strictly decreasing; FP64 projection identity + FP32 rounding residuals reported per candidate).
Key reproduced numbers: MrPro source `Σm = 29.3333`; **0/29** relations are already retimed ×4 by MrPro
(`mr_already_retimes_native_by4` false everywhere; relative error of the Mr clock to the ×4 target spans
1.10–3.76 — i.e., on these adjacent-slot beat modes MrPro's effective period extension is only
mr_clock/native-clock ≈ 0.85–1.00×, the exact-kernel fact that the ramp retimes slots, not low-order
relation clocks); candidate `ΔΣm` ranges −0.01826…+0.00185 (zero-sum in *frequency* does NOT conserve
Σ of log-compression exponents — the water-bed is coordinate-dependent, echoing sol17);
`max_abs_delta_phase_at_128k` 0.34–58.31 rad (no linear-extrapolation guarantee); 26/29 candidates stay in
the [0,1] compression box, 3 pairs (s24_25, s25_26, s26_27) put one slot faster than MrPro (allowed by
design; `slots_faster_than_mr` reported). Qualification line: clocks/amplitudes/geometry may only *filter
directions*; promotion requires whole-model losses + task endpoints.

**5. `full_model_response_native.jsonl` — what model-level measurements exist.** Runner
`full_model_response.py`: Qwen2.5-3B, weights frozen, MrPro table + gain installed, 64-slot FP32
log-period parameter `δ` (`freq=base·exp(−δ)`) patched into `rotary_emb.forward` without `no_grad`;
rows = historical dev `niah_multikey_2` / `niah_multiquery` at **true native 32768 positions** (hence
"_native"; not stretched), splits `_0`/`_1`; per row records the whole-prefix teacher-forced correct-answer
CE gradient `∂loss/∂δ` (64 floats), per-token losses, loss, sha, seconds, peak memory. 4/4 rows present:

| row | loss | ‖grad‖ | top-|g| slots | peak 16.5 GB, ~13–14 s |
|---|---|---|---|---|
| multikey_2_0 | 0.0132 | 50.9 | slot0 −40.2, slot2 +26.6 | fast-band dominated |
| multikey_2_1 | 0.1106 | 688.7 | slot0 −640.9, slot3 +162.4 | ditto |
| multiquery_0 | 0.3087 | 70.7 | slot6 +47.4, slot4 −24.9 | first-token loss 4.26 |
| multiquery_1 | 0.4746 | 460.5 | slot1 −382.7, slot0 +158.4 | first-token 7.36, tok27 6.75 |

Substantive readings: (a) the gradient plumbing of sol16/sol18's trap list works end-to-end on GPU
(nonzero finite whole-model frequency gradients through checkpointing); (b) at native 32K positions the
answer-CE response is concentrated in the **fast slots 0–6, which the MrPro face freezes at m=0** —
transition slots 24–39 barely respond, empirically confirming sol17's warning that 32K rows (stretched or
not) exercise mostly the fast band and that the 128K transition-band question needs true long phases;
(c) multikey dev rows sit near CE ceiling (0.013/0.11), matching sol18's label warning (relation/short
answers dominate signal; content binding is the failure mode), and these are *fitted historical dev rows*,
not a holdout (manifest scope line says exactly that).

**6. Tags.**
- [已验证] four-operator taxonomy against the actual sources; PM-Keep offset-mapping losslessness; NOSA/Qwen
  sparse reader semantics; operator-family NLL triple (2.6514/9.9899/3.9427) as recorded; joint-mode FP64/FP32
  projection identities (assertion-gated generator, 29/29 pass); 4 jsonl gradient records exist and are finite.
- [部分证据] the model-level test exists only as *design + plumbing pilot* (jsonl is direction measurement at
  32K, explicitly "not a claim of successful extension"); Astra01/03/06 conditional laws are compatible
  mechanisms, not Qwen measurements; E1/P2 baseline contracts must match or be excluded.
- [假设] that contiguous-128K four-way CE calibration lands inside a transferable region (mirror-vs-table
  outcome unknown); Astra09's frozen-cotangent direction agreeing with the CE gradient.
- [vetoed] relabeling natural captures / sparse-attention / PM-Keep scores / operator-family fits as
  static-table evidence; stretched-position or B64-all-scan oracles as dense-table tests; outcome-selected
  rows; rescuing a failed transfer with head mass, covariance fit, geometry distortion, or operator-family
  results; backprop through argmax.

**7. Proposed rules.** Execute the §3 test exactly: qualification gates (target/gain/position/gradient) →
projected constrained optimization *from* MrPro over slots 24–39 with exact forward/backward CE +
backtracking → freeze one table + equal-norm mirror → family-disjoint held-out decision (target CE,
answer+EOS CE, greedy value+EOS, per-family reporting, all regressions) → real 128K PPL/passkey/generation.
Finite differences or checkpointed recomputation are allowed; detached Q/K replay is a *proposal filter only*.
Bridge to theory arm: test Astra09's predicted direction against the exact CE gradient on the same
families; disagreement ⇒ model-level target wins and a frozen-state assumption gets diagnosed.

**8. Conflicts.** No YaRN mentions in any of the three reports — the old "YaRN decreasing vs MrPro
increasing" story is asserted nowhere here, so nothing to retract, but nothing incorporates the NEW facts
either (neither the η-scale-response crossover, the Σ(ν−ω)² = 0.4841 comparison, nor the max/min
counterfactuals appear in sol16–18; the unification reports predate or ignore the YaRN-vs-MrPro analysis).
Direct protocol conflict: sol16 §step-4 stretches 32K→128K positions and calls it the calibration
distribution; sol18 §3 forbids stretching for the model-level test and sol17 §5 rates stretched rows as
oracle-only with three named blind spots. jsonl fact (b) above sides with sol18/sol17: at 32K the
transition band response is negligible. Second conflict: sol16 says "16 DOF via ε(η)" while sol18
optimizes x_j on slots 24–39 (16 slots) — same simplex, different coordinates; sol16's Helmert form
additionally guarantees strict monotonicity by construction, sol18 imposes it as a constraint; pick one or
prove parity. Third: joint_mode `ΔΣm ≠ 0` shows "total scale conservation" is coordinate-relative —
qualifies (does not refute) both sol16's "exact total scale" claim (true in ε-space, not in m-space) and
any water-bed intuition read off raw frequencies.

---

## Lineage verdict (5 lines)

1. The transport/water-bed picture is **demoted as an optimizer** — sol17's audit (unsigned observability,
   null-band vs joint modes, MGDA locality, tau-moment overreach, Astra02's atomic optimum) and the
   Smooth_MrBudget counterexample class leave it no derivation power; none of sol16–18 revives it.
2. It **survives as a KKT shadow**: the water-bed term `βΔT_i²/2` is exactly the `Σn_i=K` conservation of
   the declared-nested-covariance bridge DP (constant-h relaxation ⇒ discrete Cosh recurrence), and sol16's
   softmax-Helmert simplex is the same conservation law in continuous coordinates — bookkeeping, not mechanism.
3. The shadow is coordinate-dependent: joint_mode candidates conserve Σfrequency (zero-sum n) while shifting
   Σm by −0.018…+0.002 and pushing 3/29 candidates out of the compression box — any "conservation law" must
   name its coordinate before carrying argument weight.
4. **sol18's model-level test survives as the validation protocol** (preregistered four-way source control,
   one table + equal-norm mirror, fresh-prefill/gain/layout/gradient gates) conditional on sol16's
   no-`no_grad` implementation fixes and sol17's evidence-tiering of the 4 pilot jsonl rows (32K CE response
   lives in frozen fast slots 0–6; transition band needs true 128K phases).
5. Executed so far: CPU reference PASS, 29 CPU candidates (unqualified), 4 gradient-measurement rows —
   **zero model-level calibration results exist**; the next gate is running sol18's test once, with the
   new YaRN-vs-MrPro η-facts and max/min counterfactuals still untested by this lineage.
