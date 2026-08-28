# Protected-ramp theory analysis (2026-08-28)

- **Status:** `THEORY_ANALYSIS_NOT_EXECUTED` — analysis only. Contains no new
  experiment, no compute authorization, and no reviewer-facing claim.
- **Superseding check:** independently recomputed and refined (with five
  corrections C1–C5) in
  [`PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md`](PROTECTED_RAMP_RIGOROUS_COMPOSITE_20260828.md);
  where the two disagree, the composite's recomputed numbers govern.
- **Direction update (2026-08-28):** the protected-ramp scan direction below
  is superseded by author directive — the composite's §9 now carries the
  leave-one-band-out frequency-band attribution design (which band's movement
  owns the 4K cost / the 16K benefit). The recommendations in this document
  about scanning new protection formulas are retired; its boundary scales and
  mass arithmetic remain valid inputs to that design.
- **Role:** internal theory study of the "protect fast bands, move slow bands"
  direction as a fixed-support interior-$z$ problem; companion analysis to
  [`../preflights/PROTECTED_PROGRESSIVE_SHIFT_PREFLIGHT_20260827.md`](../preflights/PROTECTED_PROGRESSIVE_SHIFT_PREFLIGHT_20260827.md)
  (frozen, not executed).
- **Method note:** five independent parallel analyses (boundary theory;
  evidence archaeology; YaRN↔$z$ correspondence; exact-geometry quantification
  with CPU-only closed-form computation; adversarial critique). All CPU
  reconstructions are shape-faithful recomputations from owner-code frozen
  constants, not new evidence; every cited number carries its owner.
- **Governing agenda:** `INDEX.md` §6 (decision order row A: submit as-is;
  this line is post-submission method development) and §6.2 (the
  matched-content 2×2 bridge is the next decision-valuable protocol).
- **Date:** 2026-08-28. NeurIPS 2026 decision is pending (notifies 2026-09-24);
  nothing here depends on it.

## 0. The question under analysis

Given a known extension factor $s$, can pure $z$-reallocation at fixed
support and gain achieve the YaRN-like pattern — protect the frequencies that
should stay, move only those that should move, smooth transition — and can the
protection boundary be derived from $L_{native}$, $s$, and each frequency's
training-time phase coverage? Target object: the frozen protected-progressive
preflight (rotation cutoffs $r=1$ and $r=32$, smoothstep attenuation
$h_k = \mathrm{smoothstep}(\mathrm{clip}((32-r_k)/31,0,1))$ applied to the
existing simple progressive curve $m_{old}$), goal of pushing the static-s4
4K in-window cost toward 0 while retaining long-range gain.

**Headline verdicts (developed below):**

1. **Two protection landmarks are exactly derivable**: the lower edge
   $r_L=1$ (window fundamental; exact) and the upper edge $r_U=s$ (demand
   extinction; exact under the coverage premise). A third, model-relative
   scale $r_{orth}=K/\ln b + \tfrac12$ marks the onset of per-pair in-window
   uniqueness. **$r=32$ is not derived** — it is YaRN's inherited
   $\beta_{fast}$ default and sits an order of magnitude above every derived
   scale (§1).
2. **The frozen preflight, applied to the actual verified curve, is nearly the
   identity map.** Three independent analyses converge: the simple progressive
   curve already embodies fast-band protection ($m_{old}\approx 0$ at
   $r\ge 32$); its movement lives at $r\lesssim 7$–$11$, where the smoothstep
   attenuates at most 16%; protection removes $\approx 2\%$ of movement mass.
   Prediction: $E\approx D$, short-window gate fails, retention gate passes
   trivially. The protocol is falsifiable by arithmetic at zero compute (§2).
3. **The real open question is redirected**: which *mid-band* pairs own the
   +0.1236 4K cost, and can it be removed without surrendering the
   extrapolation gain? No existing owner attributes that cost by frequency —
   the gap every protection theory must fill (§3).
4. **"Band protection = fixed-support $z$ allocation" is approximately true,
   and the payoff is decided by boundary placement, a pure-$z$ fact** — but
   only after adopting YaRN's shifted support and gain as the reference frame;
   YaRN itself is support+allocation+gain (§4).
5. **Static geometry underdetermines the trade-off** (in-window basis quality
   saturates fast; extrapolation-prior geometry even prefers Native); the
   binding constraint is learned-coordinate compatibility, so no closed-form
   retention$(j)$ exists (§5). One genuine theoretical contradiction surfaced
   (§6).

## 1. Derivable protection scales

Setup: pair $k$ has $x_k=-\ln\omega_k = a + R z_k$
(`../../sections/03_theory.tex` §1); shift rule
$\hat\omega_k=\omega_k[(1-m_k)+m_k/s]$ (eq. `movement-allocation`); native
rotations $r_k = \omega_k L_{native}/2\pi$. OLMo-2-0425-1B-Instruct
identity from
`../results/JOINT_MECHANISM_REPORT_20260822.md` line 4 and
`../analysis/TRANSPORT_RESIDUAL_ANALYSIS_20260822.md:52`: **$K=64$ pairs,
base $5\times10^5$, $L_{native}=4096$, $s=4$, amplitude 1.13863**. (The
$K=32$ appearing in `03_theory.tex`/`a1_proofs.tex` is the paper's generic
standard-MHA example, $d_{head}=d_{rope}=64$, not OLMo.)

**(a) Lower edge $r_L=1$ — DERIVED, exact.** At $r_k=1$ the pair is exactly
the window fundamental ($\omega L = 2\pi$); below it the cos/sin traces are
monotone arcs and the subspace collapses toward $\mathrm{span}\{1,\Delta\}$
(Prop 2, `03_theory.tex` lines 114–125; the 23 deepest OLMo pairs sit at
$r_2=2.00$, `FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`
§3.2). Movement of that band is information-cheap in-window. Independent
corroboration: native phase-safe fraction at 8K/16K is exactly 0.500
(`TRANSPORT_RESIDUAL_ANALYSIS_20260822.md` table), i.e. the 32/64 pairs with
$r\le1$ (COMPUTED). Drift argument: full shift of pair $k$ accumulates
$2\pi\, m\, r_k(1-1/s)$ cycles of phase drift across the window; demanding
sub-cycle drift gives $r \le s/(s-1) = 1.33$ at $s=4$ — the phase-coverage
edge and the free-movement edge coincide at $r\approx1$.

**(b) Uniqueness onset $r_{orth}=K/\ln b + \tfrac12$ — DERIVED, closed
form.** Adjacent pairs are $\approx 1 - b^{-1/K}$ rotations apart; neighbor
near-orthogonality (Fourier-type, Lemma `fourier-alias`, `a1_proofs.tex`
lines 111–127) requires spacing $\ge 1$ cycle. COMPUTED: $r_{orth}=5.39$
(OLMo), $5.15$ (Qwen). Below $\sim5$ rotations pairs collide with their grid
neighbors; above it each pair supplies a distinct in-window direction. This
is the theoretical home of the derived profile's knee ($\widetilde U_k$ has no
closed form; only its onset does).

**(c) Upper edge $r_U = s$ — DERIVED under coverage premise (A2).** A pair
shifted by $m$ has post-shift wavelength
$\hat\lambda = L/[r_k(1-m(s-1)/s)]$. It supplies *new non-aliasing coverage*
over $[L, sL]$ iff $\hat\lambda\in(L, sL]$, i.e.
$m > (1-1/r_k)\cdot s/(s-1) =: m_{min}(r_k)$; $m_{min}\le 1$ requires
$r_k\le s$. **Pairs with $r>s$ cannot reach any extended-window scale at any
allowed shift: moving them is in-window cost with zero extension gain under
(A2).** Pairs with $1<r\le s$ are the contested demand band; under the frozen
smoothstep their attenuation is negligible ($h(2)=0.997$, $h(4)=0.974$;
COMPUTED).

**(d) Prior dependence.** Under the attention-weighted prior $\mu_\alpha$,
$\alpha\ge1.5$, the resolution threshold $\phi^\*\to0$ with
$\omega_0 L = 2.0772 \Rightarrow r_{\phi^\*}=0.331$, and the O1 analysis
(`../../three_completions/optimization_notes.md` lines 22–35) reads "no
redistribution should happen." Every derived scale shifts with the prior; the
uniform-prior results above are the registered default.

**(e) Ordering (COMPUTED):** $r_{\phi^\*}=0.33 < r_L = 1 < s=4 \approx
r_{orth}=5.2\text{–}5.4 \ll 32$. **The entire derived structure lives in
$r\in[0.33,\sim5.4]$; $32$ is an order of magnitude above the last derived
scale.** $r=1$ reproduces a derived boundary; $r=32$ equals YaRN/LLaMA-3
$\beta_{fast}$, not $K$ for OLMo, and nothing in the theory produces it. The
transferable law is $r_U=s$ (scale-free in native-window coordinates), not
the partition it induces: at fixed $(1,32)$, OLMo protects 15 pairs
($\lambda<128$ tokens) while Qwen-32K protects 24 ($\lambda<1024$ tokens),
so the in-window/long-range trade-off is **not** transfer-invariant even
though the cutoffs are.

**Assumption ledger.** (A1) readout is relative phase $\omega\Delta$ under
the uniform window prior (registered default; O1 shows sensitivity). (A2)
extension value = non-aliasing scale coverage — CONJECTURE-grade: consistent
with the slope-only phase-preservation failure
(`../results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md` lines 65–69)
but contradicted in part by §6 below. (A3) frozen weights make basis-
substitution cost monotone in $\widetilde U_k\cdot$drift (supported by the
interaction contrast $-3.537$, `FULL_ROPE…REPORT` §5.3).

**Falsifiable predictions from the boundary theory** (CPU-checkable pre-GPU
where marked):
- **P1 (drift attribution):** E's in-window gain over D equals the drift
  energy $\sum_k m_{old,k} r_k$ removed by $(1-h_k)m_{old}$; pre-register
  that $\ge60\%$ of drift-weighted movement is removed while $\le5\%$ of the
  $r\in(1,4)$ band's is.
- **P2 (band-localized retention):** $\mathrm{Retention}(16K)\in[0.9,1.0]$
  because coverage is supplied by the $r\le4$ band with $h\approx1$; if
  $<0.8$, the demand-band model is falsified (attenuation there is only
  $\sim3\%$).
- **P3 (position-bin sign crossover):** $N_E-N_D\le0$ in 0–4K, $\ge0$ in
  8–16K, with the 0–4K improvement larger in magnitude.
- **P4 (cutoff invariance above $s$):** any upper cutoff $c\in[8,32]$ changes
  neither gate; $c\approx6\approx r_{orth}$ strictly lowers 4K cost at
  retention $\ge0.8$; $c\approx2<s$ collapses long-range retention. Sharpest
  isolation of derived $r_U=s$ vs inherited 32.
- **P5 ($s=2$ asymmetry):** the useless-movement band $(s,32)$ widens as $s$
  shrinks, so E-vs-D in-window repair is relatively larger and both retention
  gates easier at $s=2$ — with *unchanged* cutoffs.

## 2. The identity-map finding (three-way independent convergence)

Three analyses independently reconstructed the realized construction and
converge on the same arithmetic:

**Census (COMPUTED, validated against owner landmarks to 3 decimals —
reconstruction of $\omega_k=\theta^{-2k/128}$ reproduces the published
ramp_onset/full_interpolation values of
`../../external-reviews/opus-20260823/evidence/OPUS_RECOMPUTATION_20260823.json`):**

| Band | OLMo ($K=64$) | Qwen2.5-1.5B |
|---|---|---|
| $r\ge32$ (preflight fully protected) | 15, $k\le14$ ($r_{14}=36.9$, $r_{15}=30.1$) | 24, $k\le23$ |
| $1\le r<32$ (contested) | 17, $k=15..31$ | 16, $k=24..39$ |
| $r<1$ (sub-cycle) | 32, $k\ge32$ ($r_{32}=0.922$) | 24, $k\ge40$ |

**Displacement anatomy.** The frozen s4 table (SHA `a435d7…`, derived
allocation of `../results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`
§3.2) is built with $m_k=(1-\widetilde U_k)^2$; CPU reconstruction
(shape-faithful; hash path owned by the export pipeline) gives
$m_k\approx0$ for $k\le13$, rising $0.001\to0.671$ over $k=14..21$,
$\approx1$ from $k=22$ on. The successful curve is effectively a near-step
band-split with ramp split rule $k=20\to22$ (OLMo; ramp MSE $0.00072$ to the
derived profile) — i.e. **displacement is already concentrated on
$\lambda\gtrsim500$ tokens, $r\lesssim7$–$11$.** The effective protection
boundary of the successful curve is $r\approx16$, not 32.

**Consequence.** The protected set $r\ge32$ (15 pairs) is exactly where
$m_{old}\approx0$ already, and the smoothstep reaches only $h\ge0.84$ down to
$k=22$ — in the band that actually moved. Protection removes $\approx2\%$ of
total movement L1 mass (max single-pair attenuation $1-h = 0.16$ at OLMo
$k=21$, $0.20$ at Qwen $k=29$; total movement retained $99.2\%$ OLMo,
$99.0\%$ Qwen). Hence:

- **Prediction: $E\approx D$ nearly everywhere**; $|N_E-N_D|$ at 8K/16K
  within row noise; $\mathrm{Retention}(16K)\approx1.0$ (trivial pass);
  **short-window gate $N_E(4K)-N_A(4K)\le+0.01$ fails**, because the 4K cost
  lives in mid-band pairs ($k\approx18..22$) where $h$ only reached
  $0.84$–$0.97$.
- Adversarial estimate: **$P(\text{both gates})\approx5\text{–}10\%$**,
  with the short gate the binding one, contradicted by the curve's own
  construction.
- Fast-band protection of this curve is a weaker restatement of a design
  prior the curve already implements (design prior B,
  `../analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md` §3).

The structural contrast with the 2026-08-24 protected-band Cosh remains real
(that scheme protected the 7 mid-band pairs $k=32..38$,
$\lambda\in[L_{native},4L_{native}]$, and gave all 32 fast pairs to Cosh;
the new protection set is its complement) — but the new construction's lever
is almost entirely the transition band, and the question it can answer is
narrower than its framing.

## 3. Where the 4K cost lives: the measurement that does not exist yet

Existing evidence localizes cost by **position, not frequency**: co-adaptive
oracle (+0.001 on 0–4K, **+0.092 mid-band at 8K**, $-0.039$ tail, at tiny
displacement $\max|dz|=0.0013$, via `ZERO_TRAINING_MECHANISM_AND_CEILING`
§5 and `../results/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md`); mature
weights absorb the 4K hard-swap at full displacement ($+0.00098$). **No owner
attributes the static-s4 $+0.1236$ 4K cost
(`../results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md`) to
frequency bands.** The preflight's position-stratified E-vs-D bins would be
the first measurement — but per §2 they would measure a near-no-op.

The theory's prediction if a redirected attribution experiment were run
(prediction, not evidence): short-window recovery only for protection edges
pulled to $r\lesssim8$ (attenuating the $m=1$ plateau), with long-range
retention then decaying roughly linearly in removed movement mass. The
band that owns the cost is the $r\in[1,\sim16]$ transition band plus
amplitude — which is exactly the question actually worth answering.

## 4. YaRN ↔ $z$ correspondence

**Verified from the pinned reference implementation**
(`scripts/lib/rope/official_yarn.py`, `jquesnelle/yarn@995db5b`): thresholds
are rotation counts — `low = floor(find_correction_dim(32)) = 14`,
`high = ceil(find_correction_dim(1)) = 32` for OLMo — ramp linear in pair
index; amplitude `mscale = 1 + 0.1 ln s` ($=1.1386$ at $s=4$, the same value
in `JOINT_MECHANISM_REPORT` §4). (Paper-YaRN's ramp is linear in wavelength
rather than index — more conservative in the mid-band; flagged from
knowledge, unverified in detail.)

**Decomposition in $x_k=a+Rz_k$:**
- PI = pure support translation ($a\to a+\ln s$, $z$ geometric);
- NTK-aware = pure support stretch ($R\to R+\ln s$, $z$ geometric);
- **YaRN = NTK support stretch + non-geometric interior $z$-ramp + gain
  mscale. The fast endpoint is preserved; the slow endpoint is NOT**
  ($\gamma_{K-1}=1$ exactly since $\lambda_{63}\approx2.3\times10^6 \gg
  32L_{native}$; $\hat\omega_{K-1}=\omega_{K-1}/s$). YaRN is inherently
  support+allocation+gain; "band protection = fixed-support $z$" holds only
  after adopting YaRN's post-shift support and mscale as the frame — which is
  precisely the frame of every arm in `tab:frozen-fixed-support`
  (eq. `movement-allocation`, endpoints pinned, matched long gain).

**The payoff is boundary placement, a pure-$z$ fact.** At matched support +
matched gain (`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md` §3.2):
geometric 0.56 → official-YaRN ramp class 7.94 → derived 60.47 → ramp with
re-derived label-free boundary 61.04 (OLMo 16K RULER, %). The ramp shape
contributes $\approx+7.4$ over geometric; **relocating the boundary
contributes $\approx+60$**. Gain is not an uncontrolled confound: the basis×
mscale interaction is strongly positive at 4× ($+0.285$, ratio 2.48,
`JOINT_MECHANISM_REPORT` §3), which is why the protocol pins matched long
gain rather than dropping it. Qwen nuance: derived − official-YaRN
$=+0.0625$ with interval $[-0.0050,+0.1325]$ — OLMo decisive, Qwen
directional.

**Exact novelty boundary (for any future-cycle text).** May say: YaRN's
band-protection principle is a rule for interior exponent allocation on the
method's own extended support; isolating $z$ at fixed endpoints, fixed
log-span, fixed amplitude — a separation no published method makes and no
scalar-base or linear Q/K reparameterization can imitate (Thm obstruction) —
shows the payoff is decided by where the transition band sits: YaRN's fixed
rotation thresholds $[1,32]$ recover 7.9/100 on OLMo-16K while the same
linear-ramp family with one model-relative label-free boundary recovers 61.0.
May **not** say: that YaRN is fixed-support; that band protection is new;
that YaRN's thresholds are wrong in general (one checkpoint pair; and the
repo's own preflight adopts them); that mscale is inessential; that the
official-YaRN row is a pure-$z$ control.

## 5. Static geometry: the trade-off is flat where protection acts

Implemented the cross-gram of `../../appendix/a1_proofs.tex` lines 9–18 and
budget identity (Thm 1, lines 38–47). Sanity: the 23 OLMo slow pairs give
$r_2=2.000$, $\bar c=0.99993$ — exactly the frozen claim
(`03_theory.tex` lines 127–129). Surrogate family
$m_k=0\ (k<j)$, linear-to-1 thereafter, endpoints pinned:

| $j$ protected pairs | $\sum m$ | $r_2$, U$[0,4096]$ | $r_2$, U$[0,16384]$ |
|---|---|---|---|
| 0 (full progressive) | 32.0 | 7.13 | 10.46 |
| 15 (preflight $r\ge32$) | 24.5 | 7.45 | 10.98 |
| 25 | 19.5 | 7.77 | 11.48 |
| 32 ($r\ge1$ crossing) | 16.0 | 8.02 | 12.00 |
| 35 | 14.5 | **8.07 = native** | 12.24 |
| native | 0 | 8.07 | 12.64 |

Shape: concave, saturating exactly when the protected set absorbs the
sub-cycle band; beyond $j\approx32$ additional protection buys literally
nothing in-window. The whole in-window basis price of the unprotected
progressive curve is $0.94$ effective dims ($-12\%$); protecting the fastest
15 pairs recovers only $0.32$ of it. Mechanism: **99.16% of the native
table's in-window collision mass sits in ordered pairs touching the $r<1$
band; pairs with both $r\ge1$ hold 0.84%** — the fast band is already
near-orthogonal, so protecting it defends nothing movement could destroy.
Qwen replicates (native $r_2=13.93$ in-window; plateau by its own $r=1$
crossing at $j=40$).

Under the **extension-window** prior, static $r_2$ is *maximal at Native*
($12.64 > $ every moved table; Qwen $24.03 > 17.91$): geometry alone prefers
no movement — flatly contradicted behaviorally (Native PG-19 4× NLL 7.205 vs
binary-s4 3.098; RULER-16K 0.0000 vs 0.6047,
`SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md` §5.1/§5.4). The uniform-
prior Gram cannot see the binding constraint: a frozen model's learned
coefficients are written in the native phase coordinates (50M crossing:
$r_2$ *rises* $4.57\to12.54$ while perplexity collapses $7.14\to76.20$,
`03_theory.tex` lines 148–162). **Consequence: no closed-form
retention$(j)$ exists from geometry; protection-boundary theory must be
stated as movement-mass vs learned-coordinate compatibility, with (A3)
carrying the load.** This is the same moral as the standing audit rule:
static collision $\ne$ extrapolation mechanism.

## 6. The open theoretical contradiction (most valuable output)

The coverage-extinction result (§1c) says pairs with $r>s=4$ cannot buy
extension under any allowed shift. Yet the **empirically successful derived
curve moves pairs up to $r\approx16$** ($k=18$, onset landmark 16.268 in the
opus recomputation; $m$ rising from $k=14$, $r\approx30$ at negligible
amplitude through $k=22$, $r\approx7.2$ at full). If (A2) were the whole
story, that movement would be pure in-window cost — but it is part of the
table that produced 60.47 at 16K.

Candidate resolutions, all untested:
1. **(A2) is incomplete**: extrapolation value comes from pairwise
   collision/phase-separation reduction at extended positions, not per-pair
   scale coverage — consistent with length-dependent $z$ rankings (geometric
   wins at 8K then loses 512/512 at 16K,
   `FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md`).
2. **Mid-distance demand**: the 4–8K/8–12K region may genuinely require the
   $(4,32)$ band (P2's failure mode); the oracle's $+0.092$ mid-band penalty
   at 8K points the same way.
3. **Prior dependence**: under the attention prior $\mu_\alpha$ the demand
   scale shifts ($r_{\phi^\*}=0.33$), possibly legitimating mid-band
   movement that uniform-prior coverage forbids.

This is the real theory problem the direction exposes: **derive the correct
demand functional $G(r;s)$ whose optimizer matches the observed $r\approx16$
onset.** Until then, §1c stands as exact-but-insufficient, and every
protection-boundary claim above inherits the (A2) caveat.

## 7. Protocol audit of the frozen preflight (design defects)

1. **Arm B reachability (highest severity).** The gate keys on $N_E-N_A$ at
   0–4K under matched long gain, but the gain-only in-window cost (B−A at
   0–4K) has **never been measured** (session policy never applies mscale
   in-window; `JOINT_MECHANISM_REPORT` §3 has gain-only at RULER, not NLL).
   If $B-A>+0.01$, the short gate is unreachable for *any* $z$, and a
   "protection failure" verdict would actually be an amplitude fact. **B−A
   must be measured before any single-table verdict is read.**
2. **Endpoint trap.** §2 gives $m_{new}=m_{old}\cdot h$ with no endpoint-
   repinning rule. If the realized $m_{old}$ is nonzero at the endpoint pair,
   E's endpoint ≠ D's, the "shared exact endpoints" premise (§3) and the
   pure-$z$ label silently fail, and the §8 stop rule kills the run with no
   valid instance of the formula.
3. **Retention bin ambiguity.** $\mathrm{Retention}(8K)$ is undefined between
   three readings (4–8K bin / 0–8K cumulative / standalone 8K documents); the
   owners' 8K numbers (`2.75 vs 7.00` class) come from 8K-token documents,
   not bins of the 16K rows, and the denominator changes materially across
   readings. Must be pinned in the manifest before execution.
4. **Gate width vs noise.** $\pm0.01$ NLL point-estimate gate on one row set
   where paired per-document 4K-bin SE is plausibly $0.01$–$0.03$: marginal
   false passes/fails expected; decision does not condition on the A–F
   decomposition even though the interpretation table admits gain/support
   attribution can void it.
5. **Stage 2 circularity.** Stage 2 licenses a **re-embedded** table (Native
   support, Native amplitude, no gain) that is neither D nor E; conditioning
   the bridge on Stage 1 destroys the bridge's registered independence.
6. **Stage 3 is not the transfer it claims.** Fixed rotation cutoffs on
   Qwen-32K protect 24 pairs and span protection across nearly the whole
   table; the Qwen base curve at $s=2/4$ has no cost/gain owner, and the
   corrected Qwen 128K numbers are explicitly forbidden as results for this
   method. Unregistered as stated.
7. **Identity fork.** "Simple progressive curve" is not a name either owner
   uses; derived (hash `a435d7…`) and ramp are distinct tensors $\approx0.001$
   NLL apart, and the $+0.1236$ history attaches to the derived table.
   "Resolve at execution time" hides a genuine fork.

## 8. Strategic placement

- **This cycle:** manuscript frozen at `001a900`; `AUTHOR_VERDICTS_20260828.md`
  §7 bans new submission experiments; the preflight self-labels
  post-submission. A pass becomes a next-cycle method owner at best; nothing
  here enters ICLR 2027.
- **A failure is internally safe** (candidate-specific; AGENTS.md keeps
  internal negatives surfaced but out of reviewer-facing text; the paper never
  claims routing is necessary), but hardens `INDEX.md` §3.4 #11 against the
  single-table class.
- **Deployment need is already answered** by the bitwise-exact Native routing
  option (120/120); a $\le+0.01$ single table buys serving simplicity, not
  in-window performance. The open scientific content is narrow and
  identification-shaped: does the joint Pareto exist at zero training, and
  which band owns the cost.
- **Drift risk:** "push the cost to 0 while keeping the gain" is
  competition framing, which the author's 2026-08-28 ruling rejects for this
  paper; and `INDEX.md` §6.2 designates the matched-content 2×2 bridge — not
  this companion — as the next decision-valuable protocol. Priority must not
  invert.

## 9. Recommendations

0. **Zero-compute first.** Run the protection-coverage audit before any GPU
   authorization: realized $m_{old,k}$, $r_k$, $h_k$, per-band $|\Delta x_k|$
   and drift energy from the hashed frozen tensors, plus the P1 bookkeeping.
   All five analyses predict $E\approx D$; the audit turns that into a
   certificate at zero cost and pre-registers the expected null.
1. **Do not execute the frozen preflight as written.** Independent of
   compute authorization (which does not exist), §2 predicts its primary
   contrast is a near-no-op and §7 lists design defects that would muddy any
   verdict it did produce.
2. **Redirect to the real question** — frequency attribution of the 4K cost
   and the protection-vs-retention frontier at edges $r\lesssim8$ (P4/P5 give
   the pre-registerable shape). This requires a **new preflight** (with the
   §7 fixes: B−A first, endpoint repinning rule, retention-bin definition,
   decomposition-conditioned decision rule) and separate explicit compute
   authorization.
3. **Keep the matched-content 2×2 bridge first** in the post-submission
   queue per `INDEX.md` §6.2.
4. **Theory work worth keeping regardless of any experiment:** the exact
   boundaries $r_L=1$, $r_U=s$ and $r_{orth}=K/\ln b+\tfrac12$; the
   protection-budget decoupling (99.16% collision-mass concentration; flat
   $r_2(j)$); and the §6 contradiction (derive $G(r;s)$ whose optimizer
   reproduces the observed $r\approx16$ onset). These are next-cycle theory
   assets and are consistent with the paper's identification identity.

## 10. Do-not-reuse reminders (from evidence archaeology)

Old aliased Qwen tables (0.6700 @64K; 0.6175 @128K — artefact-inflated; only
0.6650/0.5400 valid; the 128K $n=5$ probe must not be cited); the $+0.1236$
and $+0.6922$ deltas as *baseline estimates* (valid only as owner-held
candidate outcomes); the 32-row vs 128-row co-adaptive sign flip; the 20-row
2Wiki learned-table screen; the direct-z pilot's four-row deltas;
$c=0.12$ gain sensitivity; ledger prohibitions
(`../evidence/METHOD_SELECTION_LEDGER_20260823.json`); Qwen rows in the
20260827 preflight Stage 3 are unverified futures, not results.
