# Protected-ramp rigorous composite (2026-08-28)

> **ARCHIVED LIFECYCLE NOTE (updated 2026-08-30):** superseded as an active
> research route. This document remains arithmetic and derivation provenance.
> Its §9 leave-one-band-out design is **invalid as written**, not merely
> conditional: abrupt restoration of B0/B0°/B1/B2 breaks strict frequency
> ordering, B0 changes sampled support, and B4 is a small nonzero intervention
> rather than an exact sham. No execution or new preflight may inherit those
> arms. Future attribution needs a newly derived monotonicity-checked cumulative
> or smoothly projected intervention.

- **Status:** `THEORY_ANALYSIS_NOT_EXECUTED`. Independent author-side
  re-derivation and CPU-only recomputation of the five-agent synthesis in
  [`PROTECTED_RAMP_THEORY_ANALYSIS_20260828.md`](PROTECTED_RAMP_THEORY_ANALYSIS_20260828.md).
  No model evaluation, no new evidence, no compute authorization. Where this
  document and the five-agent synthesis disagree, this document's recomputed
  numbers govern.
- **Purpose:** self-contained rigorous composite for external cross-analysis.
  Every closed form used is stated in Appendix A; every number is either
  DERIVED (algebra), COMPUTED (independent CPU implementation of the stated
  closed forms), or INHERITED (from an owner file, with path).
- **Addendum (2026-08-28, after external cross-analysis):** §9 adds the
  leave-one-band-out frequency-band attribution design — author-directed
  pivot from scanning new protected ramps to causal attribution of the
  existing successful curve ("which band's movement carries the short-window
  cost, which band's movement carries the long-range benefit"). Status
  unchanged: `DESIGN_NOT_EXECUTED`; no arm constructed, no evaluation, no
  compute authorization.
- **Repo context:** fixed-support RoPE identification paper. Table coordinate
  $x_k=-\ln\omega_k = a + R z_k$; shift rule
  $\hat\omega_k=\omega_k[(1-m_k)+m_k/s]$ with endpoints pinned; frozen
  protected-progressive protocol protects fast bands of the verified s=4 curve
  via rotation-count cutoffs $(1,32)$.

## 1. Setup

OLMo-2-0425-1B-Instruct: $K=64$ pairs, base $b=5\times10^5$,
$L_{native}=4096$, extension factor $s=4$, matched long gain $1.13863$
(INHERITED: `../results/JOINT_MECHANISM_REPORT_20260822.md` line 4,
`../analysis/TRANSPORT_RESIDUAL_ANALYSIS_20260822.md:52`). Native
$\omega_k = b^{-k/64}$, $k=0..63$. Rotations in the native window
$r_k=\omega_k L_{native}/2\pi$. Qwen2.5-1.5B: $K=64$, $b=10^6$,
$L_{native}=32768$.

Bands (COMPUTED, exact census):

| Band | OLMo pairs | Qwen pairs |
|---|---|---|
| $r\ge32$ | 15, $k\le14$ ($r_{14}=36.94$, $r_{15}=30.09$) | 24, $k\le23$ ($36.39/29.33$) |
| $1\le r<32$ | 17, $k=15..31$ | 16, $k=24..39$ |
| $r<1$ | 32, $k\ge32$ ($r_{31}=1.132$, $r_{32}=0.922$) | 24, $k\ge40$ ($1.151/0.927$) |
| collapse core $\omega L\le1$ | 23, $k\ge41$ | — |

Rotation-count thresholds of the frozen protocol
(`../preflights/PROTECTED_PROGRESSIVE_SHIFT_PREFLIGHT_20260827.md` §2):
$h_k=\mathrm{smoothstep}(\mathrm{clip}((32-r_k)/31,0,1))$,
$m^{new}_k=m^{old}_k\,h_k$; $h=0$ for $r\ge32$ (no shift), $h=1$ for $r\le1$
(full shift).

## 2. Verified exact results

**V1 — window-fundamental edge $r=1$.** DERIVED. $r_k=1 \iff \omega L=2\pi$:
below it the pair completes no cycle in-window and its subspace collapses
toward $\mathrm{span}\{1,\Delta\}$ (Prop. 2 of `../../../sections/03_theory.tex`).
Independent drift check: full shift accumulates
$\Delta\phi = 2\pi\, m\, r_k(s-1)/s$ cycles of phase drift across the window;
sub-cycle drift requires $r \le s/(s-1) = 1.333$ at $s=4$. The coverage edge
and the free-movement edge coincide at $r\approx1$.

**V2 — demand extinction at $r=s$.** DERIVED. Post-shift wavelength
$\hat\lambda = \lambda/(1-m(s-1)/s)$. The pair reaches beyond the native
window ($\hat\lambda > L$) iff
$m > m_{min}(r) := \dfrac{s(r-1)}{r(s-1)}$,
and $m_{min}\le1 \iff r\le s$ (algebra: $s(r-1)\le r(s-1) \iff r\le s$).
COMPUTED at $s=4$: $m_{min}(1.5)=0.444$, $m_{min}(2)=0.667$, $m_{min}(3)=0.889$,
$m_{min}(4)=1.000$, $m_{min}(4.0001)>1$, $m_{min}(8)=1.167$, $m_{min}(16)=1.25$.
**Pairs with $r>s$ cannot stretch their wavelength past $L$ at any allowed
shift**; moving them changes no coverage scale (interpretation "therefore zero
extension gain" requires premise A2 of §7 — see open problem O1).

**V3 — uniqueness-onset scale.** DERIVED closed form.
$r_{orth} := \dfrac{1}{1-b^{-1/K}}$, with asymptotic expansion
$r_{orth}=\dfrac{K}{\ln b}+\dfrac12+\dfrac{\ln b}{12K}+O(K^{-2})$.
COMPUTED: OLMo exact $5.3942$ vs expansion $5.3943$; Qwen $5.1504$ vs
$5.1505$. Origin: adjacent-pair rotation spacing is $r_k(1-b^{-1/K})$;
neighboring subspaces become Fourier-distinct in-window when that spacing
reaches one cycle (Lemma `fourier-alias` regime,
`../../../appendix/a1_proofs.tex`).

**V4 — spectral budget identity and collapse.** Independent reimplementation
of the closed-form cross-Gram (Appendix A.1) reproduces:
$r_2(\text{native}, U[0,4096]) = 8.067$; $r_2(\text{native}, U[0,16384]) =
12.641$; the 23 collapse-core pairs ($k\ge41$) alone: $r_2=2.0001$ (paper
claim: $2.00$, `03_theory.tex` line 127). COMPUTED.

**V5 — collision-mass concentration.** Of the native table's total pairwise
collision mass $\sum_{i\neq j}c_{ij}$ under $U[0,4096]$, **99.16%** sits in
ordered pairs touching the $r<1$ band; pairs with both $r\ge1$ carry
**0.84%**. COMPUTED. The fast band is near-orthogonal; protecting it defends
nothing that movement destroys.

**V6 — YaRN thresholds are rotation-count cutoffs.** COMPUTED from
$\mathrm{idx}(r)=128\ln(L/(2\pi r))/(2\ln b)$: OLMo $(\lfloor14.7\rfloor,
\lceil31.6\rceil)=(14,32)$; Qwen $(23,40)$. The frozen protocol's $(1,32)$
is exactly YaRN's $(\beta_{slow}=1,\beta_{fast}=32)$ in rotation coordinates.
INHERITED implementation identity: `scripts/lib/rope/official_yarn.py`
(pinned `jquesnelle/yarn@995db5b`).

**V7 — $32$ is not a derived scale.** The derived ordering (COMPUTED/DERIVED):
$r_{\phi^\*}=0.33$ (attention-prior resolution threshold; INHERITED from
`../../three_completions/optimization_notes.md` O1, not independently
re-derived here) $<\ r_L=1 <\ s=4 \approx r_{orth}=5.39 \ll 32$. No formula
in V1–V3 produces 32.

## 3. The protection transform on the realized curve (the identity-map result)

The verified s=4 curve ("derived allocation", table SHA
`a435d754…`, owner `../results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`
§3.2; construction owner `scripts/analysis/export_uniqueness_budgeted_tables.py`)
is $m^{old}_k=(1-\widetilde U_k)^2$ with $\widetilde U_k$ the min-max
normalised conditional residual energy of pair $k$ against all others under
the causal separation measure $p(\Delta)\propto L-\Delta$.

**Independent reconstruction** (COMPUTED, frozen definition re-implemented;
shape-faithful — my float32 hash drifts from `a435d7…`, candidate cause the
numpy-vs-torch float32 `pow`; the owner pipeline asserts the canonical hash):

| $k$ | $r_k$ | $\widetilde U^{raw}_k$ | $m^{old}_k$ | $h_k$ | $m^{old}(1-h_k)$ |
|---|---|---|---|---|---|
| 14 | 36.94 | 0.9762 | 0.0006 | 0.000 | 0.0006 |
| 17 | 19.97 | 0.9179 | 0.0067 | 0.335 | 0.0045 |
| 18 | 16.27 | 0.8596 | 0.0197 | 0.511 | 0.0096 |
| 19 | 13.25 | 0.7900 | 0.0441 | 0.655 | 0.0152 |
| 20 | 10.80 | 0.6542 | 0.1196 | 0.764 | 0.0283 |
| 21 | 8.79 | 0.1809 | 0.6709 | 0.842 | 0.1059 |
| 22 | 7.16 | 0.0012 | 0.9975 | 0.897 | 0.1026 |
| 23 | 5.84 | ~0 | 1.0000 | 0.935 | 0.0654 |
| 25 | 3.87 | 0 | 1.0000 | 0.976 | 0.0242 |
| 28 | 2.09 | 0 | 1.0000 | 0.996 | 0.0036 |
| 31 | 1.13 | 0 | 1.0000 | 1.000 | 0.0001 |
| 32+ | <0.92 | 0 | 1.0000 | 1.000 | 0 |

Mass accounting (COMPUTED, exact for the reconstructed profile):
$\sum_k m^{old}_k = 42.86$; protection removes $\sum m^{old}(1-h) = 0.428$,
i.e. **0.998% of movement mass; 99.002% retained**. Drift-weighted:
$\sum m r = 47.0$, removed $3.3$ (**7.03%**); of the removed drift, **94.94%
comes from the $r>4$ band** (provably coverage-useless by V2) and only
**1.05% from the demand band** $1<r\le4$. Ramp variant (split $20\to22$):
$\sum m = 42.50$, removed $0.339$ (0.797%).

**Conclusion (arithmetic):** the frozen protection is within one percent of
the identity on the verified curve. The curve already embodies fast-band
protection — $m^{old}_k\le0.007$ for all $k\le17$ (the entire $r\ge32$ band
and most of the contested band). The only substantive lever is the taper
$k=15..23$, and the demand band the theory says must move ($1<r\le4$,
$k=25..31$) is attenuated by ≤2.5%.

**New observation N1 — the uniqueness cliff.** The conditional residual
collapses from $0.181$ ($k=21$, $r=8.79$) to $0.0012$ ($k=22$, $r=7.16$):
from $k=22$ on, pairs are almost completely predictable from the rest under
the causal measure. The successful curve's step location is set by this
redundancy cliff at $r\approx7$–$9 \approx (1.3\text{–}1.6)\,r_{orth}$. This
is the first-principles anchor of the empirically successful boundary — same
decade as V3's $r_{orth}=5.39$, prior-dependent, and numerically computable
but without closed form (open problem O3).

## 4. Basis accounting of D vs E (exact)

COMPUTED with the independent closed-form implementation:

| Table | $r_2$, $U[0,4096]$ | $r_2$, $U[0,16384]$ |
|---|---|---|
| Native | **8.067** | **12.641** |
| D = derived s4 | 5.608 | 8.072 |
| E = protected | 5.690 | 8.090 |

D sacrifices $2.46$ effective dimensions in-window ($-30.5\%$) for its
measured long-range gain (owner-reported: 16K RULER $0.0056\to0.6047$, PG-19
$4\times$ NLL $7.205\to3.098$); protection E recovers $0.082$ of the $2.46$
(**3.3%**). Surrogate family $m_k=0\ (k<j)$, linear-to-1 thereafter
(COMPUTED, matches the five-agent table): $r_2(j)$ in-window $=7.13, 7.45,
7.60, 7.77, 8.03, 8.06, 8.07$ for $j=0,15,20,25,32,35,\ge40$ — concave,
saturating exactly when the protected set absorbs the $r<1$ band.

**Two structural facts follow.** (i) Even under the extension-window prior
$U[0,16384]$, static $r_2$ is maximal at Native ($12.641 > 8.09$): pure
geometry prefers no movement at every length, while behavior prefers movement
decisively — static geometry cannot see the binding constraint (learned
coefficients written in native phase coordinates; the 50M crossing raises
$r_2$ $4.57\to12.54$ while perplexity collapses $7.14\to76.20$,
`03_theory.tex` lines 148–162). No closed-form retention$(j)$ exists from
geometry alone. (ii) Under the most favorable geometric hypothesis (in-window
cost $\propto$ in-window basis loss), the maximum repair available to E is
$\approx 3.3\%$ of D's cost, i.e. $+0.1236\to\approx+0.1195$ versus the gate
$\le+0.01$. The co-adaptation channel (weights bound to exact phases) is not
repaired by removing one percent of movement. **The short-window gate fails
under both channels.**

## 5. Precedent synthesis: two independent supports for gate failure

**Support 1 (mass).** With 99.0% of movement retained, any damage model
approximately monotone in displacement predicts
$N_E(4K)-N_A(4K)\approx N_D(4K)-N_A(4K)\approx+0.12 \gg +0.01$.

**Support 2 (the 2026-08-24 protected-band Cosh precedent, sharpened here).**
Owner: `../results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md`. That
construction protected exactly the band $\lambda\in[L_{native},4L_{native}]$
$= r\in[0.25,1]$ = pairs $k=32..38$ (COMPUTED: 7 pairs, verified from the
census) and cut in-window damage from $+3.978$ (anchored full Cosh) to
$+0.692$ — protecting those 7 pairs removed ~83% of the cost. **This is
direct executed evidence that in-window damage concentrates in the
displacement of the $r\in[0.25,1]$ band.** In the protected-progressive
construction, exactly that band has $m^{old}=1$ and $h=1$ (COMPUTED: all 7
pairs fully shifted, zero protection). Even a strongly nonlinear damage
model calibrated to the only available protection datapoint therefore
predicts gate failure for E.

Caveats, stated: the Cosh construction differs in curve shape and in what the
complement did (global Cosh warp, including all fast pairs), so Support 2 is
directional evidence about damage concentration, not an additive
decomposition. The two supports are mechanistically independent (one
arithmetic, one precedent-based) and agree.

## 6. Corrections and refinements to the five-agent synthesis

- **C1 (capacity criterion — corrected).** "Wrap-free extension capacity
  requires $r\le1/s$ (25 OLMo pairs)" conflates two statements. COMPUTED:
  a **fully moved** pair ($m=1$) at position $sL$ has phase $2\pi r$ — exactly
  as native at $L$ — so wrap-free capacity under movement is $r\le1$: **32
  OLMo / 24 Qwen pairs**. A **protected** pair ($m=0$) at $sL$ has phase
  $2\pi s r$, wrap-free iff $r\le1/s$: **25 OLMo / 17 Qwen pairs**. Either
  way demand is comfortably met; the synthesis's conclusion is unchanged, the
  attribution is corrected.
- **C2 (mass figure — refined).** "~2% of movement removed" → exact
  $0.998\%$ unweighted, $7.03\%$ drift-weighted ($\sum mr$), with $94.94\%$
  of removed drift in the $r>4$ band.
- **C3 (attenuation statement — refined).** "max single-pair attenuation
  $1-h=0.16$ at $k=21$" is correct among pairs with substantial movement;
  raw $1-h$ reaches $1.0$ at $k=14,15$, where $m^{old}\approx0$. Mass
  removal peaks at $k=21$ ($0.106$) and $k=22$ ($0.103$).
- **C4 (basis loss of D — new exact value).** $r_2(D,4K)=5.608$
  ($-30.5\%$ vs native); the surrogate linear ramp's $-12\%$ understates the
  real curve's in-window basis price because the real curve is step-like.
- **C5 (demand band count — added).** The demand band $1<r\le4$ contains 7
  OLMo pairs ($k=25..31$); attenuation there is ≤2.5% (COMPUTED), confirming
  prediction P1's pattern quantitatively: $\ge94\%$ of removed drift is
  coverage-useless, ≤1.1% touches the demand band.

## 7. Open problems and conjecture ledger

- **O1 (the central contradiction, sharpened).** V2 proves movement of
  $r>4$ pairs buys no scale coverage at $s=4$. Yet the successful derived
  curve moves pairs with $m\ge0.1$ up to $r=10.8$ ($k=20$) and $m\ge0.5$ up
  to $r=8.8$ ($k=21$), and the resulting table is what produced the 16K gain.
  Therefore premise A2 ("extension value = non-aliasing scale coverage") is
  incomplete. Candidate mechanisms: (i) extension value includes pairwise
  collision/phase-separation reduction at extended positions, not per-pair
  coverage; (ii) mid-distance (4–8K) discrimination genuinely uses the
  $(4,32)$ band (the position-stratified oracle shows the mid-band penalty
  $+0.092$ at 8K, via `../analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md`
  §5); (iii) prior dependence (attention prior moves the resolution scale to
  $r_{\phi^\*}=0.33$). **The open theory problem: derive the demand
  functional $G(r;s)$ whose optimizer reproduces the observed cliff at
  $r\approx7$–$9$ and onset $r\approx11$.** Status: conjecture-grade.
- **O2 (prior dependence).** All derived scales move with the separation
  prior; under the attention-weighted prior the construction reads "no
  redistribution" (INHERITED, not re-derived here). Any protection-boundary
  claim is prior-relative.
- **O3 (no closed form for $\widetilde U(r)$).** The uniqueness cliff (N1) is
  numerically computable; no analytic form is known.
- **O4 (gain/amplitude outside this analysis).** Everything here is
  $z$-domain. The matched-gain arm B's in-window cost has never been measured;
  if $N_B(4K)-N_A(4K)>+0.01$, the short-window gate is unreachable for any
  $z$ (protocol defect, not settled by theory).
- **O5 (endpoint pinning).** The frozen formula $m^{new}=m^{old}h$ has no
  endpoint-repinning rule. In the realized profile $m^{old}_0=0$ and
  $m^{old}_{63}=1$ hold by min-max construction (COMPUTED), so E and D share
  endpoints here; the defect is latent, not active, for this curve.

## 8. Explicit questions for external cross-analysis

1. Is the V2 extinction argument correct as stated (wavelength-stretch
   readout of the shift rule), and does its contrapositive legitimately
   support "movement above $r=s$ is in-window cost with no coverage gain"
   only under A2?
2. Can the V1/V3 coincidence ($r_L=1$, drift edge $s/(s-1)$, $r_{orth}$) be
   unified into a single boundary functional, or are they genuinely distinct
   landmarks?
3. Propose a demand functional $G(r;s)$ resolving O1: it must make moving
   $r\in(4,\sim11)$ pairs valuable while making $r>32$ movement worthless,
   and it must predict the observed redundancy cliff near $r\approx7$–$9$.
4. Given §3–§5, is there any damage model consistent with both the 8/24
   precedent (protecting $r\in[0.25,1]$ removed ~83% of Cosh damage) and the
   identity-map arithmetic under which E passes the $+0.01$ short-window
   gate? (We believe no; a counterexample would be valuable.)
5. Is the C1-corrected capacity statement (moved: $r\le1$; protected:
   $r\le1/s$) complete, or does partial movement $m\in(0,1)$ create a third
   regime worth naming?

## 9. Invalidated addendum — leave-one-band-out restoration design

> **Do not execute.** The historical construction below is retained so the
> failure is auditable. For a restored block beginning at pair `j`, the pair
> immediately before the block remains scaled by
> `[(1-m_{j-1})+m_{j-1}/s]`, while pair `j` jumps to its Native frequency.
> Under the canonical OLMo `s=4` movement profile this violates the strict
> decreasing-order assertion used by
> `scripts/analysis/rope_transport/same_support_controls.py` for B0/B0°/B1/B2.
> Thus “set one band's movement to zero” is not a legal RoPE allocation for
> those arms. The endpoint and sham defects are separate and remain as recorded
> below.

**Directive.** External cross-analysis (GPT, 2026-08-28) endorsed §1–§8 and
sharpened the next question; the author then redirected the line: since the
identity-map result (§3) already proves any new protection formula is within
~1% of the verified curve, the missing layer is not "a better ramp" but
**causal attribution of which displacements produce which effects**. Author's
formulation: split the successful long table's frequencies into bands, restore
each band to Native one at a time, and observe (i) which restoration recovers
the 4K loss, (ii) whose restoration destroys 16K extrapolation — producing the
map `frequency band → (native cost, long-range benefit)`. YaRN's
"protect fast bands" is at best a surface correlate; the missing object is the
causal attribution of the transition band. This section records the historical
attempt and its predictions from V1–V7 and N1. The arm construction was later
invalidated before execution; status: `INVALID_DESIGN_NOT_EXECUTED`.

### 9.1 Arms and exact reachability

Reference table: the canonical derived s=4 table (SHA `a435d754…`, owners in
§3). The shift rule is pair-wise affine in $m_k$, so the historical construction
can algebraically install
$m'_k = m^{old}_k\cdot\mathbb 1[k\notin b]$ without a refit. The former inference
that this makes every arm a legal fixed-support allocation is false: algebraic
element-wise reachability does not preserve global frequency order, and B0 also
moves an endpoint. The invalidation note above governs.
Controls: $C^+ = D$ (restore nothing), $C^- =$ Native (restore everything —
exactly the Native checkpoint).

### 9.2 Band partition (OLMo, primary)

The author's split, with exact index boundaries DERIVED from
$r_k=(L/2\pi)\,b^{-k/64}$, i.e. $r_k=r_\star \iff k=64\ln(L/(2\pi r_\star))/\ln b$:

| Band | $r$ range | pairs $k$ | $n$ | movement in $D$ | note |
|---|---|---|---|---|---|
| B0 | $r<1$ | 32..63 | 32 | $m=1$ | collapse core $k\ge41$ inside; contains slow endpoint $k=63$ |
| B1 | $1\le r<4$ | 25..31 | 7 | $m=1$ | V2 demand band ($m_{min}\le1 \iff r\le s=4$) |
| B2 | $4\le r<8$ | 22..24 | 3 | $\approx1$ ($k{=}22$: $0.9975$) | N1 cliff foot at $k=22$ |
| B3 | $8\le r<16$ | 19..21 | 3 | $0.044/0.120/0.671$ | N1 cliff head at $k=21$ |
| B4 | $r\ge16$ | 0..18 | 19 | $\le0.02$ per pair (§3 table) | incl. 15 pairs $r\ge32$ with $m\le0.007$ |

Boundary values (COMPUTED from the closed form; $L=4096$, $b=5\times10^5$):
$r=1\iff k=31.61$; $r=4\iff k=24.83$; $r=8\iff k=21.46$; $r=16\iff k=18.08$.

**Endpoint note.** B0 contains the slow endpoint $k=63$ ($m^{old}=1$), so
restoring B0 also returns the slow endpoint to Native, i.e. **restores the
support** ($\widehat R\to R$). Variant $B0^\circ$: restore only $k=32..62$,
keep $k=63$ moved — separates the support-radius channel from the
interior slow-band channel. Both are predicted destructive at 16K; the
$B0$-vs-$B0^\circ$ contrast measures how much of the destruction runs through
support itself.

### 9.3 Static side, exactly pre-computable (no model)

Per-arm $r_2$ under $U[0,4096]$ and $U[0,16384]$ from Appendix A.1–A.2
(closed-form implementation exists from this session's audit,
`/tmp/protected_ramp_audit.py`):

- Predicted in-window ordering (to be confirmed by the pre-screen, not assumed
  as a theorem): $r_2(4K)$ rises along $C^+ \lesssim B3 \lesssim B2 \lesssim
  B1 \lesssim B0 \lesssim C^-$, with marginal repairs dominated by B0 (V5:
  99.16% of in-window collision mass touches the $r<1$ band), then B1, then
  B2; B3 small, B4 null.
- $r_2(16K)$ of every arm stays below Native ($12.641$, §4(i)): **static
  geometry is structurally blind to the benefit side** (the binding constraint
  is co-adaptation, not basis size). The pre-screen can rank the 4K cost
  channel; it can never rank the 16K benefit channel. Behavioral readout is
  mandatory.
- Historical B4 control proposal: its table differs from $C^+$ by at most the
  §3-listed displacements ($m\le0.02$ per pair). This is a real nonzero
  intervention, not a sham; a nonzero behavioural effect would therefore not
  invalidate the harness. A future design needs an exact duplicate determinism
  control.

### 9.4 Behavioral predictions — the 4K cost channel

Two hypotheses on the source of the $+0.1236$ NLL (4K FineWeb-Edu, owner
`../../../appendix/a6_mature_scale.tex` deployment paragraph):

- **H-cost-1 (slow-band displacement).** The 8/24 protected-band Cosh
  precedent (§5, Support 2) is executed evidence: protecting exactly
  $r\in[0.25,1]$ (7 pairs inside B0) removed ~83% of in-window Cosh damage.
  Extrapolating: B0 restoration recovers most of $+0.1236$.
- **H-cost-2 (transition-band phase distortion).** The cost sits in pairs that
  complete cycles in-window and whose phases the Native weights resolve
  ($r\in[1,8)$, B1∪B2): B1/B2 restoration recovers the cost.

The design separates them, with an asymmetry: under H-cost-1 cost and benefit
are entangled in B0 (the same band carries the 4K damage and, by C1, the
wrap-free capacity — a fully moved pair at $sL$ has phase $2\pi r$, exact iff
$r\le1$, which is precisely B0); under H-cost-2 the cost sits partly in B1,
whose movement is simultaneously the only legal coverage stretch (V2). **Under
either hypothesis plus A2, B2 is the only band whose restoration is a priori
Pareto-improving** — V2 proves $r>4$ movement buys no coverage scale, and both
cost hypotheses assign B2 a non-negative share of the in-window cost. If B2
restoration neither recovers 4K nor damages 16K, the cost is not additive in
per-band displacement at all (§9.6).

### 9.5 Behavioral predictions — the 16K benefit channel; three worlds

| Arm | 4K: recovery of $+0.1236$ | 16K: loss from $60.47$ |
|---|---|---|
| B0 ($r<1$) | H-cost-1: most; H-cost-2: ≈0 | large (wrap-free capacity, C1) |
| B1 ($1$–$4$) | H-cost-2: part; H-cost-1: ≈0 | large (V2: only band that stretches $\hat\lambda$ past $L$) |
| B2 ($4$–$8$) | partial under either hypothesis | ≈0 under A2; measurable ⇒ A2 falsified |
| B3 ($8$–$16$) | small | ≈0 under A2; measurable ⇒ A2 falsified |
| B4 ($\ge16$) | ≈0 | ≈0 (negative control) |

- **W1 (A2 holds).** 16K damage concentrates in B0+B1; B2/B3/B4 null.
  Attribution map: phase-exact coverage at $sL$ ← B0; wavelength stretch ← B1;
  cost ← B0 or B1∪B2 (whichever H-cost survives); free lever ← B2.
- **W2 (A2 incomplete — O1 resolved empirically).** B2 and/or B3 restoration
  measurably damages 16K. Then O1's demand functional $G(r;s)$ has support
  beyond $r=s=4$, and the N1 cliff at $r\approx7$–$9$ is the candidate zero
  crossing of $G$ — matching O1's conjecture that the optimizer of $G$
  reproduces the successful curve's onset $\approx11$ / cliff $\approx7$–$9$.
  Practical consequence: the protection boundary is the cliff, not $r=s$, and
  not $r=32$ (= YaRN $\beta_{fast}$, V7 — heuristic, not derived).
- **W3 (historical control reading, invalid).** The former rule treated any B4
  effect as protocol noise. Because B4 is not an exact duplicate, that inference
  is unsupported and cannot govern a future experiment.

The N1 cliff straddles the B2/B3 boundary (foot $k=22$, $r=7.16$; head
$k=21$, $r=8.79$), so any non-null B2 or B3 effect triggers the pre-specified
follow-up: single-pair dissection of $k=19..24$.

### 9.6 Non-additivity warning (design constraint)

Leave-one-band-out gives marginals, not a partition: NLL and RULER macro are
nonlinear in the table, so generally $\sum_b\mathrm{effect}(B_b)\neq
\mathrm{effect}(C^-)$. Report the interaction gap
$\mathrm{effect}(C^-)-\sum_b\mathrm{effect}(B_b)$ as a diagnostic; if it is
large, add the two-band restoration B0+B1 before any attribution conclusion.

### 9.7 Qwen variant (secondary, flagged)

Same partition by rotation census (COMPUTED, $b=10^6$, $L=32768$): $r<1$:
$k\ge40$ (24); $1\le r<4$: $k=34..39$ (6); $4\le r<8$: $k=31..33$ (3);
$8\le r<16$: $k=27..30$ (4); $r\ge16$: $k\le26$ (27). **Caveat:** the frozen
Qwen protocol evaluates a factor-4 table at $2\times$ Native ($64$K), so the
effective extension factor at the evaluation horizon is $2$; V2's demand
boundary shifts to $r\le2$. The Qwen demand-band algebra must be re-derived at
evaluation factor 2 in the preflight, not copied from OLMo.

### 9.8 Relation to §7–§8; governance

- Empirical fallback for §8 Q3 and the resolution path for O1: if $G(r;s)$
  resists closed form, W1/W2 decides between O1's candidate mechanisms
  (i)–(iii) directly.
- Supersedes the protected-ramp scan direction: §3's identity map already
  proved protection-formula scans near the verified curve are ≈identity;
  attribution is the missing causal layer.
- Governance: this archived arm list has no execution path. Any future band
  study must start from a new monotonicity-checked construction and preflight,
  obtain explicit compute authorization, and route completed numbers to a new
  canonical owner. No reviewer-facing claim follows from this invalid design.

## Appendix A — closed forms used (for independent checking)

**A.1 Cross-Gram.** $\Delta\sim\mathrm{Unif}[0,L']$,
$x_\omega=[\cos\omega\Delta,\sin\omega\Delta]$, $d=(\omega-\nu)L'$,
$q=(\omega+\nu)L'$, $a(t)=\sin t/t$, $b_\star(t)=(1-\cos t)/t$:
$H_{\omega\nu}=\frac12\begin{bmatrix}a(d)+a(q)&b_\star(q)-b_\star(d)\\ b_\star(q)+b_\star(d)&a(d)-a(q)\end{bmatrix}$;
$S_\omega=H_{\omega\omega}$; $Q_{\omega\nu}=S_\omega^{-1/2}H_{\omega\nu}S_\nu^{-1/2}$;
$c_{\omega\nu}=\tfrac12\|Q_{\omega\nu}\|_F^2\in[0,1]$. (Verified termwise
against $\mathbb E[\cos\omega\Delta\cos\nu\Delta]$ etc.)

**A.2 Budget identity.** $\Gamma$ block-assembled from $Q_{ij}$ with
$I_2$ diagonals: $\operatorname{tr}\Gamma=2K$,
$\operatorname{tr}\Gamma^2=2K+2K(K-1)\bar c$, hence
$r_2=(\operatorname{tr}\Gamma)^2/\operatorname{tr}\Gamma^2
=2K/(1+(K-1)\bar c)$. (Proof re-checked line by line against
`../../../appendix/a1_proofs.tex` lines 38–47.)

**A.3 Demand.** $\hat\lambda=\lambda/(1-m(s-1)/s)$;
$\hat\lambda>L \iff m>s(r-1)/(r(s-1))=m_{min}(r)$; $m_{min}\le1\iff r\le s$.
Full-movement phase at $sL$: $\hat\omega\cdot sL = \omega L$ (exactly);
protected phase at $sL$: $s\,\omega L$.

**A.4 Uniqueness onset.** Neighbor spacing in rotations
$\Delta r = r_k(1-b^{-1/K})$; distinctness threshold $\Delta r\ge1$ gives
$r_{orth}=1/(1-b^{-1/K})=K/\ln b+1/2+\ln b/(12K)+O(K^{-2})$ (Euler
expansion of $1/(1-e^{-x})$, $x=\ln b/K$).

**A.5 YaRN threshold map.** $\mathrm{idx}(r)=d_{head}\ln(L/(2\pi r))/(2\ln b)$
(rotation count → channel index), matching the pinned reference
implementation; OLMo $(14,32)$, Qwen $(23,40)$.

**Computation record.** All CPU arithmetic above was recomputed on 2026-08-28
from the stated closed forms and the frozen construction definition in
`scripts/analysis/export_uniqueness_budgeted_tables.py` (read-only; the
profile reconstruction is shape-faithful, hash drift documented in §3).
