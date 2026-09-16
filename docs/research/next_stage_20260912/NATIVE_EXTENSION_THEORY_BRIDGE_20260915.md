# Native enhancement and extension: a testable theory bridge

Date: 2026-09-15. Status: mathematical derivation and local CPU verification;
no model forward pass, GPU execution, or new model-performance result.

**Subsequent evidence review:** the [method-lineage and native-benefit audit](../reviews/NATIVE_BENEFIT_AND_METHOD_LINEAGE_REVIEW_20260915.md)
supersedes this note's proposed NCP-first research priority. Its mathematics remains valid for the stated reference objects;
the (+,+,-) reference-risk pattern is not adopted as a default model-accuracy prediction. Existing gain, formation/readout,
KV-factorial and positive native results must constrain the next design. This note does not establish a universal native method.

## 1. First principles: the contribution to pursue

The paper can make a stronger statement than “a better frequency curve improves
long context”: **internal allocation controls the distance dependence of learned
content comparisons. Learning, native deployment, and extension place different
requirements on that same controllable object.**

The forward argument should be **operator → content response → structural prior
→ explicit constructor → predicted response → task evidence**. Existing
experiments establish useful interventions and counterexamples; the new theory
should state assumptions that generate predictions before another model is run.
Below, an impossibility example identifies the needed assumption, NCP derives a
model-independent phase rule, and a finite content-margin bound makes its link to
real models measurable. Task performance remains the final empirical endpoint.

## 2. What the manuscript already establishes

Relevant sources are
[main theory](../../../paper-2027/sections/03_theory.tex),
[TailSpline construction](../../../paper-2027/sections/04_mature.tex),
[TailSpline proofs and controls](../../../paper-2027/appendix/a10_tailspline.tex),
[allocation responses](../../../paper-2027/appendix/a11_allocation_response.tex),
and [learned compatibility](../../../paper-2027/sections/03_compatibility.tex).

The existing foundation includes complete-pair rank, retained content coordinates,
the Q/K compensation obstruction, exact TailSpline minimization and T/C exchange,
two distance references, and exact softmax/value reweighting. Gains with lower
positional rank already rule out a simple “more rank means better tasks” account.
The new work below addresses the missing **signed content response**.

## 3. One coordinate system; two different frozen deployment priors

Write the original table as

\[
x_k=-\log\omega_k=a+Rz_k,\qquad \nu_k=\omega_ke^{-u_k}.
\]

For **native NCP**, \(u_0=u_{K-1}=0\), gain is one, and the original support
and window remain fixed. Hence \(z'_k=z_k+u_k/R\) is a direct interior
allocation intervention.

For **TailSpline extension**, \(u_k=m_k\log s\). Its high-frequency band
retains \(u=0\), and its fully interpolated tail has \(u=\log s\).
Changing from Native to this table also changes the support and the specified
gain. The pure interior comparison is **T versus P or C at the same extended
support and gain**, not the entire Native-to-extension change.

For fixed content, a pair contributes

\[
f_k(d;\nu)=C_k\cos(d\nu_k)+D_k\sin(d\nu_k)
=A_k\cos(d\nu_k-\alpha_k),
\quad A_k=\sqrt{C_k^2+D_k^2}.
\]

This is the common response object:

- NCP's reference task sets the correct pair's preferred phase to
  \(\alpha=0\), takes an isotropic unrelated key, and integrates over a
  declared triangular distance prior. Its proximal penalty limits departure
  from Native in log-frequency coordinates.
- TailSpline's tail preserves a content response under distance dilation:
  if the relevant reference phase is \(\alpha=\omega d_0\), evaluating at
  \(d=sd_0\) with \(\nu=\omega/s\) reproduces that phase exactly.
  Its transition joins the native and dilation references with a specified
  smooth extra-gap boundary prior.

**These are two priors on the same response, not two proven minimizers of one
actual-model loss.** In a real model, \(C,D,\alpha\) depend on the input,
layer, head, and preceding computation. Establishing how their signed responses
change is the empirical bridge. A cosine is legitimate after specifying content
phase; using cosine-only overlap as phase-independent positional geometry is a
different, invalid substitution already addressed by the manuscript.

### Why an assumption about content is necessary

For any nonzero rotation change \(A=R(d\nu)-R(d\omega)\), there are unit
vectors \(q,k\) with \(q^TAk>0\); replacing \(k\) by \(-k\) reverses the
sign. Thus no content-blind nontrivial frequency intervention improves every
possible content score. This statement applies whenever the changed kernels
differ at the distance being considered; exact phase aliases are the no-change
case.

A stronger explicit example uses the same fixed-support intervention
\(\omega=(1,1/2,1/10)\to\nu=(1,1/4,1/10)\) and distance \(d=2\).
Only the middle pair carries content; \(q=(1,0)\), and the distractor is the
negative of the unit correct key. Both Native margins are positive:

| Correct key in the active pair | Native margin | Changed margin | Difference |
|---|---:|---:|---:|
| \((1,0)\): aligned before rotation | \(2\cos1=1.080605\) | \(2\cos(1/2)=1.755165\) | +0.674561 |
| \((\cos1,-\sin1)\): matched to Native phase | 2 | \(2\cos(1/2)=1.755165\) | −0.244835 |

This does not refute a general native-enhancement method. It identifies what
“general” must mean: a **shared structural prior whose average gain transfers
across relevant model/content distributions**, not pointwise gain for arbitrary
Q/K. NCP assumes a useful population of pre-rotation aligned comparisons;
TailSpline assumes useful content responses that should transfer under distance
dilation. The remaining empirical question is whether those populations have
enough weight in actual model decisions.

## 4. New proposition A: NCP selects a phase scale, with a precise limitation

Use the proposal's definitions

\[
r(\theta)=\mathbb E_X\operatorname{softplus}(X-\cos\theta),
\quad X=\cos\psi,\quad\psi\sim U[0,2\pi),
\]
\[
\mathcal R(\phi)=2\int_0^1(1-t)r(\phi t)dt,
\quad q(\phi)=\frac{d\mathcal R}{d\log\phi},
\quad\lambda=9/4.
\]

### A1. Exact symmetry identifies the leading harmonic

The symmetry of \(X\), and
\(\operatorname{softplus}(v)-\operatorname{softplus}(-v)=v\), imply

\[
r(\theta)-r(\pi-\theta)=-\cos\theta.
\]

Indeed, letting \(H(c)=\mathbb E\operatorname{softplus}(X-c)\), replacing
\(X\) by \(-X\) in \(H(-c)\) gives \(H(c)-H(-c)=-c\).
Consequently its cosine series has the exact form

\[
r(\theta)=a_0-\tfrac12\cos\theta+
\sum_{j\ge1}a_{2j}\cos(2j\theta).
\]

All odd harmonics above one vanish. Thus the nonlinear softmax risk has a precise
relationship to a simple content-alignment response. Its even harmonics encode
the nonlinear correction; neither their presence nor the risk's name establishes
additional model benefit by itself.

### A2. Both ends of the relative intervention vanish

Define \(C(v)=\operatorname{sinc}^2(v/2)\), with mathematical
\(\operatorname{sinc}(v)=\sin(v)/v\). Termwise integration gives

\[
\mathcal R(\phi)=a_0+\sum_{n\ge1}a_n C(n\phi),
\]
\[
q(\phi)=\frac2\phi\sum_{n\ge1}\frac{a_n\sin(n\phi)}n
-\frac4{\phi^2}\sum_{n\ge1}\frac{a_n[1-\cos(n\phi)]}{n^2}.
\]

The smooth periodic risk makes these weighted coefficient sums absolutely
convergent. For \(A_1=\sum|a_n|/n\) and \(A_2=\sum|a_n|/n^2\),

\[
|q(\phi)|\le\frac{2A_1}{\phi}+\frac{8A_2}{\phi^2}.
\]

For a coordinate with inactive gap constraints, NCP either leaves \(u=0\)
or solves \(\lambda u=q(\phi e^{-u})\), with \(u\le2/9\). Therefore

\[
u^*(\phi)\le\frac1\lambda
\left(\frac{2e^{2/9}A_1}{\phi}+
\frac{8e^{4/9}A_2}{\phi^2}\right)=O(\phi^{-1}).
\]

At the other end, put
\(m=\mathbb E\sigma(X-1)>0\). The proposal's slow-phase expansion gives
\(q(\phi)=m\phi^2/6+O(\phi^4)\). Substitution into the root equation yields
the sharper coefficient

\[
\boxed{u^*(\phi)=\frac{2m}{27}\phi^2+O(\phi^4),\qquad\phi\to0.}
\]

Thus both very slow and very fast pairs have vanishing relative movement, while
an intermediate phase scale can receive appreciable movement. This conclusion
comes from the reference task and proximal rule; no manually selected peak or
half-turn eligibility boundary is required. The scalar conclusion is conditional
on inactive gap constraints; it should not be applied coordinate by coordinate
to a coupled active-constraint solution without another argument.

### A3. Small relative movement is not small whole-window phase movement

For the full window, the actual change is

\[
h(\phi)=\phi(1-e^{-u^*(\phi)}),
\qquad \|R(\phi e^{-u})-R(\phi)\|_2=2|\sin(h/2)|.
\]

The high-phase result gives \(h=O(1)\), **not** \(h\to0\). Numerical
evaluation along \(\phi=3\pi/2+2\pi j\) demonstrates this distinction for
the actual NCP reference risk:

| \(j\) | \(\phi\) | \(u^*\) | \(h\), radians | Rotation operator difference |
|---:|---:|---:|---:|---:|
| 100 | 633.030920 | 0.0006717246 | 0.425080 | 0.421887 |
| 1000 | 6287.897696 | 0.0000673486 | 0.423467 | 0.420310 |
| 10000 | 62836.565461 | 0.0000067366 | 0.423304 | 0.420151 |

This numerical sequence is a diagnostic illustration, not a proof of a global
nonzero limit. The proved point is that the available asymptotic bound does not
force phase compatibility. It motivates measuring \(d\Delta\omega\) and
signed content response at actual evidence distances. It does not justify
changing NCP after observing task scores.

## 5. New proposition B: a finite signed content-margin certificate

For one fixed query and one fixed key, set

\[
\theta_k=d\omega_k,\qquad \delta_k=d(\nu_k-\omega_k),
\quad J_k=-C_k\sin\theta_k+D_k\cos\theta_k.
\]

Taylor's theorem in **phase**, not in log-frequency displacement, gives

\[
f_k(\theta_k+\delta_k)-f_k(\theta_k)=J_k\delta_k+e_k,
\qquad |e_k|\le\tfrac12 A_k\delta_k^2.
\]

Proof: \(f_k''(\theta)=-C_k\cos\theta-D_k\sin\theta\) has absolute
value at most \(A_k\). The bound holds for every finite \(\delta\), although
it may be loose outside the small-phase regime. Both sine and cosine terms are
retained.

For a designated correct key \(+\) and a designated distractor \(-\), with
possibly different distances, define

\[
M=\sum_k(f_k^+-f_k^-),
\quad B=\sum_k(J_k^+\delta_k^+-J_k^-\delta_k^-),
\]
\[
E=\tfrac12\sum_k\left[A_k^+(\delta_k^+)^2+
A_k^-(\delta_k^-)^2\right].
\]

Then

\[
\boxed{B-E\le M(\nu)-M(\omega)\le B+E.}
\]

Hence \(B>E\) certifies a positive **fixed-content attention margin change**.
An existing margin \(M(\omega)>0\) stays positive if
\(M(\omega)+B-E>0\). A shared positive attention scale multiplies all three
quantities; a shared cosine/sine gain also factors through consistently.
Comparisons with different gains require explicitly including those gains.

For the phase reflection control \(\nu^R=2\omega-\nu\), each phase step
changes sign: \(B_R=-B\), \(E_R=E\), and the exact per-slot operator norm
\(2|\sin(d(\nu-\omega)/2)|\) is identical. Thus \(B>E\) certifies opposite
margin effects for the two directions despite equal phase perturbation norm.
Reflection is an attribution control, not an NCP optimizer output; positivity
and ordering must be checked without clipping, and installed FP32 mismatch must
be recorded. Equal **total log displacement** alone does not give this equality:
it can allocate much larger phase movement to high-frequency slots.

This is a more informative bridge than unweighted displacement:

- The same phase movement can improve or damage the margin depending on
  the measured \(C,D\) coefficients.
- NCP supplies \(\nu\) and a reference prediction; actual coefficients test
  whether its alignment prior is relevant to the model.
- T/C supplies a fixed zero-total exchange; actual coefficients identify which
  channels and evidence distances account for its signed attention response.
- If the bound is loose, the exact trigonometric difference remains cheap to
  compute. A loose bound is not evidence against the method.

The certificate fixes Q/K at the measured layer. It does not hold later hidden
states fixed in a real table-swapped forward pass, certify answer generation, or
identify a causal mediator by correlation alone. The manuscript's exact value
readout identity explains the next link: useful key reweighting must change the
useful value mixture before a task benefit can follow.

## 6. The new generalization hypothesis and its experiment

**Hypothesis:** a useful subset of content comparisons across model families
shares a window-normalized phase regime in which moderate slowdown improves
correct-relative-to-distractor evidence. A single public-config construction can
exploit that regime without learning a target table. The gain should follow the
measured signed content response, not just frequency index, total movement, or
positional rank.

NCP gives a precise transferable rule in \(\phi=(L-1)\omega\), with gap
constraints specified in native log-frequency coordinates. Rescaling
\(D\to cD\), \(\omega\to\omega/c\) leaves both \(\phi\) and native log
gaps unchanged, so the mathematical optimizer's \(u\) is unchanged, including
its coupled constraints. This exact scale equivariance supports a common
construction rule; it does not imply actual checkpoints have identical content
phases, head uses, or gains. Different native tables must be recomputed by the
same rule, with actual endpoints/order verified; the OLMo vector is not copied.

The strongest economical test uses fixed NCP versus Native on multiple model
families and the same native task contract. Keep all inputs in each model's
native window, use gain one, preserve endpoints/slots, and stratify by actual
normalized evidence distance. A complete first task panel tests the method;
independent confirmation and natural-text/QA evaluation test scope and cost.
A positive aggregate dominated by one tracking task is not broad native gain.

For mechanism, reuse those answer outputs and T/C/P extension outputs. On a
fixed source-order slice with known correct/distractor source spans, obtain a
reference Q/K trace only at predeclared query/span positions. Do not filter on
success, select favorable heads, or use the trace to tune the table. CPU replay
then gives exact and certified signed margins for **all** selected cases. Verify
the runtime's rotary sign/layout with an exact same-table no-op; retain a fixed
label-shuffled analysis null. The directional prediction must use actual
query/evidence distances: Section 9 now predicts a **reversal** at the farthest
4K binding positions. Test that signed prediction, including its negative case,
against correct/distractor responses and answers; do not assume longer distance
always means larger benefit.

A signed-margin/answer association supports contact with the claimed mechanism.
For causal mediation, one subsequent output experiment must apply the change to
a predefined layer group, with runtime-matched no-op and shuffled-group controls,
on a separate fixed slice. Current-layer replay alone is not that experiment.
A failure of the signed prediction weakens this explanation; no automatic curve
rescaling or parameter sweep follows.

## 7. How this strengthens the paper's argument

The theoretical result is a **conditional and falsifiable deployment principle**:
frequency allocation changes complete content-dependent distance responses;
structural priors determine useful directions; proximity and phase-aware controls
separate compatibility cost from directional utility. Native and extension use
this principle with different reference phases and boundary conditions.

The most valuable missing evidence is therefore a fixed-rule native result
that transfers across checkpoints, with unchanged-constructor natural-text/QA
costs and one discriminating mechanism comparison. Complete the existing T/C/P
attribution and Native deployment curves alongside that result. Together these
connect allocation learned with weights, allocation enhanced within native
support, and allocation transported for extension. They do not require claiming
that the two frozen constructors optimize one universal Transformer objective.

If NCP fails, retain its reference-risk theorem and the calibrated V1 positive
evidence at their proper scopes. Failure would distinguish the proposed prior
from the broader possibility of native enhancement, which the arbitrary-Q/K
counterexample neither proves nor rules out.

## 8. CPU verification performed for this note

Reproduce the receipt and tests from the repository root:

```bash
python3 -m experiments.native_enhancement_oral_20260915.theory
python3 -m pytest -q tests/test_native_enhancement_theory.py
```

The CLI optionally accepts `--out <new-json-path>` and refuses to overwrite an
existing receipt. The implementation is
[theory.py](../../../experiments/native_enhancement_oral_20260915/theory.py), with
[independent matrix tests](../../../tests/test_native_enhancement_theory.py).
No checkpoint, activation, network request, SSH, or GPU was used. Import tests
verify that torch and GPU runtimes are not loaded.

- NCP risk integrated on a deterministic 2048-by-2048 angular grid; first
  32 Fourier modes retained; scalar roots solved with Brent's method.
- Reflection identity maximum absolute error: \(5.56\times10^{-16}\).
- \(a_1=-0.5\); largest odd coefficient above one through mode 31:
  \(1.42\times10^{-17}\).
- \(m=0.2898446201581536\), hence \(2m/27=0.021469971863566934\).
  At \(\phi=0.1,0.01,0.001\), measured \(u/\phi^2\) was
  0.02147414044, 0.02147001386, 0.02146997242.
- Mathematical FP64 OLMo grid \(500000^{-k/64}\), \(D=4095\): largest
  relative slowdown 0.1671638685 at pair 33. This check does **not** certify
  bit identity with the installed Torch FP32 native table; that belongs to the
  constructor implementation audit.
- Seed 20260915, 10,000 random fixed-content comparisons, eight full rotary pairs
  each: zero margin-bound violations; maximum remainder/bound ratio 0.965943;
  3,490 positive certificates, none false. These use a shared fixed table,
  different correct/distractor distances, and random content; they are not model
  results. The earlier inline calculation used independently randomized phases
  and shifts and is superseded by this reproducible shared-table fixture.
- Explicit 2-by-2 rotation SVD versus \(2|\sin(\delta/2)|\), 5,000 draws:
  maximum absolute difference \(6.67\times10^{-16}\).

The original NCP proposal is a conversation attachment, not a Git-distributed
source: `a438001a-db48-48fd-8ddd-5b604bc0cb9e/pasted-text.txt`.
Its definitions needed for these new derivations are restated above. External
download links in the proposal were not used.

## 9. Frozen prediction: whole-window average gain can reverse at a final query

The execution owner has frozen a CPU-only receipt, `reports/reference_predictions.json`,
using [predictions.py](../../../experiments/native_enhancement_oral_20260915/predictions.py)
on the actual binding panel and installed Native/NCP tables. For each row it
averages **every token of the query-selected correct evidence record and every
frequency pair equally**, using the final prompt token as the query position.
It evaluates the single-distance risk \(r\), not the integrated risk
\(\mathcal R\). The reported means below are rounded reference-loss units,
not accuracy points or model NLL:

| Input cap | Near: Native minus NCP risk | Far: Native minus NCP risk | Far minus near |
|---|---:|---:|---:|
| 1K | +0.00003917 | +0.00337280 | +0.00333363 |
| 2K | +0.00004082 | +0.00914011 | +0.00909929 |
| 4K | +0.00003921 | −0.02122600 | −0.02126521 |

**Why this does not contradict NCP's theorem.** Let
\(g(d)=K^{-1}\sum_k[r(d\omega_k)-r(d\nu_k)]\). The optimizer controls
\(2\int_0^1(1-t)g(Dt)dt\): the distance distribution from two uniformly
sampled positions subsequently ordered in a window. A fixed final query with a
uniformly located key instead gives a *uniform* distance distribution; the binding
panel deliberately places its selected record near the end or beginning and
therefore gives two concentrated empirical distributions. Long distances have
vanishing density under the triangular prior but receive the far panel's entire
weight. The same 4K NCP table is retained at all three input lengths.

There is an exact reason a slowdown can harm the pointwise reference risk:
\(r'(\theta)=\sin\theta\,\mathbb E\sigma(X-\cos\theta)\). When both
old and new phases lie within \((\pi,2\pi)\), slowing toward \(\pi\)
**increases** risk; within \((0,\pi)\), slowing decreases it. Periodic
responses therefore permit positive average gain and negative far-distance gain
simultaneously. This interval argument explains the possibility of reversal;
the reported full-pair calculation establishes its sign for the actual panel.

**Prospective model hypothesis:** if this aligned-content reference transfers,
the NCP-minus-Native binding effect should have a positive far-minus-near
interaction at 1K and 2K and a negative interaction at 4K. Near effects are tiny
in the reference and do not imply detectable near accuracy changes. The sign
pattern is a falsifiable *lift from reference loss to model behavior*, not a
theorem about accuracy. A positive 4K far task effect would weaken this specific
prediction even if NCP is useful overall; a matching reversal would expose a
distance-distribution limitation of the frozen rule.

Retain all lengths, layouts, examples, and the original table. Do not remove 4K
far rows, retune NCP, or replace the triangular prior after seeing this result.
The binding prediction does not reduce three-hop chain reasoning to a one-pair
loss. The new knowledge being tested is **whether reference-distribution average
benefit transfers to the actual query-distance distribution, including a
precomputed adverse regime**.
