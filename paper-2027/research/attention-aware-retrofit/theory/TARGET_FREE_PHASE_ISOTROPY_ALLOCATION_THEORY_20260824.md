# Target-free phase-isotropy allocation

- **Date:** 2026-08-24
- **Status:** `THEORY_AND_CPU_SCREEN_INTERNAL_NOT_TRAINING_VALIDATED`
- **Role:** corrected theory and experiment contract for the next fixed-support
  allocation; not current manuscript evidence
- **Compute boundary:** no GPU run or training is authorized by this document

## Decision

The next method must be fixed before training and must not receive a deployment
length.  A table selected after seeing $L_{\rm target}$ solves the FMRoPE/PI
problem, not the target-free allocation problem studied here.

The desired object is one table that

1. preserves language modelling inside the training window;
2. improves the extrapolation curve at several unseen lengths without being
   constructed for any of them; and
3. changes only normalized interior allocation $z$ when used in the
   fixed-support identification protocol.

No completed static metric guarantees the first property.  The current Cosh
table is the aggressive reference: it strongly removes slow-band redundancy
and repeatedly improves extrapolation, but pays a measured in-window cost.
Pair-volume companding, developed during the present audit, is the opposite
reference: it protects the training geometry so strongly that its fixed-range
movement is probably too small to retain the desired extrapolation gain.

The primary new candidate is therefore **phase-isotropy companding**.  It is
more selective than pair volume about partially observed rotary pairs, but its
density saturates once a pair is well exposed.  This gives it a stronger
slow-tail intervention without Cosh's continued concentration at the fastest
endpoint.

This is a proposed construction, not a theorem of language-model optimality.

## 1. Problem contract

Fix the number of rotary pairs $K$, training length $L$, and sampled
frequency endpoints.  Write

\[
\omega(\phi)=\omega_{\max}e^{-R\phi},
\qquad
0=\phi_0<\phi_1<\cdots<\phi_{K-1}=1.
\]

Only the interior coordinates move.  The construction may use

\[
(K,L,\omega_{\max},\omega_{\min})
\]

and the causal geometry of a length-$L$ sequence.  It may not use a target
length, task labels, checkpoint activations, learned attention probabilities,
or held-out model loss.

Evaluation lengths remain necessary for falsification.  They are frozen only
after the table is built and never feed back into its definition.

## 2. The exact training object

For one rotary pair,

\[
x_\omega(d)=
\begin{bmatrix}
\cos(\omega d)\\
\sin(\omega d)
\end{bmatrix}.
\]

The structural causal separation distribution counts the number of query--key
pairs at each offset:

\[
p_L(d)=\frac{L-d}{\sum_{r=0}^{L-1}(L-r)},
\qquad d=0,\ldots,L-1.
\]

This is an attention-operator prior, not a content or learned-attention prior.
Its characteristic function is

\[
\chi_L(t)=\mathbb E_{d\sim p_L}e^{itd}.
\]

Define the physical self-Gram

\[
S_\omega=\mathbb E_{d\sim p_L}
[x_\omega(d)x_\omega(d)^\top].
\]

Its eigenvalues are exactly

\[
\lambda_\pm(S_\omega)
=\frac{1\pm|\chi_L(2\omega)|}{2}.
\]

Two consequences must remain distinct.

### Pair volume

\[
v_L(\omega)
=4\det S_\omega
=1-|\chi_L(2\omega)|^2.
\]

This measures the two-dimensional area exposed during training.  It is the
natural conservative score: a pair receives credit when its two quadratures
span nonzero volume, even if one is still much weaker than the other.

### Pair isotropy

\[
i_L(\omega)
=\frac{\lambda_-(S_\omega)}{\lambda_+(S_\omega)}
=\frac{1-|\chi_L(2\omega)|}{1+|\chi_L(2\omega)|}.
\]

This is the condition ratio of the complete pair.  It withholds credit until
both quadratures are observable.  That sharper transition is the reason it is
the main candidate rather than another in-window-preserving diagnostic.

For small $\omega$,

\[
|\chi_L(2\omega)|
=1-2\operatorname{Var}_{p_L}(d)\omega^2+O(\omega^4),
\]

so

\[
i_L(\omega)
=\operatorname{Var}_{p_L}(d)\omega^2+O(\omega^4),
\qquad
\operatorname{Var}_{p_L}(d)=\frac{(L-1)(L+2)}{18}.
\]

Once the training phases are well spread,
$|\chi_L(2\omega)|\to0$ and $i_L(\omega)\to1$.  The score therefore has
the shape needed by the problem: it suppresses poorly exposed slow pairs but
stops increasing inside the well-exposed band.

## 3. The proposed table

We deliberately choose a nearest-cell spectral coverage objective,

\[
\mathcal D_K[\rho]
=\frac{1}{12K^2}
\int_0^1\frac{i_L(\omega(\phi))}{\rho(\phi)^2}\,d\phi,
\qquad
\int_0^1\rho(\phi)\,d\phi=1.
\]

Its unique positive minimizer is

\[
\boxed{
\rho_{\rm iso}(\phi)
=\frac{i_L(\omega(\phi))^{1/3}}
{\int_0^1i_L(\omega(u))^{1/3}\,du}.}
\]

The finite table uses endpoint-inclusive inverse-CDF quantiles,

\[
\phi_k=F_{\rm iso}^{-1}\!\left(\frac{k}{K-1}\right),
\qquad k=0,\ldots,K-1.
\]

This construction has no learned positional parameter, stiffness multiplier,
uniform-mixture coefficient, or target horizon.  It preserves the sampled
support exactly.

The cube-root statement belongs only to the declared nearest-cell objective.
It is not imported into projection on the full selected span.  Neighboring
frequencies can cooperate in full-span projection, where the local squared
error can begin at fourth order and no universal density exponent has been
proved.

## 4. Why this is not another conservative solution

Three target-free schedules now bracket the design space.

| Construction | Training score | Slow-band behavior | Expected failure |
| --- | --- | --- | --- |
| pair volume | $v_L^{1/3}$ | mild compression | protects 1x but may leave too much extrapolation gain unrealized |
| **phase isotropy** | $i_L^{1/3}$ | sharper transition, then saturation | primary candidate; trained outcome unknown |
| physical-metric coverage | $(v_L\omega^2)^{1/3}$ | continued fast-end concentration | high rank/extrapolation pressure, Cosh-like in-window risk |

At the exact-range configuration $K=32,L=256,b=256$, normalized median
coordinates from the current CPU screen are approximately

\[
0.500\quad\text{(FMRoPE)},\qquad
0.483\quad\text{(pair volume)},\qquad
0.451\quad\text{(phase isotropy)},\qquad
0.190\quad\text{(anchored Cosh)}.
\]

Phase isotropy is not a small perturbation disguised as a new method, nor does
it push half the table into the fastest fifth of the log span.  It occupies the
interior of the existing empirical bracket.  The completed phase-chord table
also lies in this broad region, but it was defined from measured attention
mass; phase isotropy reaches a similar intervention scale using only the exact
training pair Gram.

## 5. Target-free recurrence is an audit, not a selector

For a fixed table, the normalized rotary-code distance at integer lag $n$ is

\[
d_\Omega^2(n,0)
=1-\frac1K\sum_{k=1}^K\cos(\omega_kn).
\]

Every finite table is recurrent: arbitrarily distant lags can return
arbitrarily close to the origin.  Consequently, the worst collision over all
future lengths is degenerate and cannot select a table without introducing a
hidden horizon.

The honest target-free diagnostic is therefore a recurrence curve, reported
after construction in geometric octaves:

\[
A_m(\Omega)
=\max_{2^mL\le n<2^{m+1}L}
\frac1K\sum_k\cos(\omega_kn),
\qquad m=0,1,2,\ldots.
\]

No particular octave chooses the method.  A candidate that improves early
octaves but develops a later peak has exposed its recurrence profile; it has
not been retroactively optimized for the peak.  Target-aware support transport
such as FMRoPE remains a separate oracle that is allowed to move the table
after receiving a requested horizon.

## 6. CPU screen

The following calculations are exploratory internal diagnostics.  They were
performed with the repository's canonical full-RoPE Gram routines and direct
integer-lag evaluation; they require a tracked script and receipt before any
outward use.

The identity

\[
4\det S_\omega=1-|\chi_L(2\omega)|^2
\]

matched direct weighted matrix determinants to below $2\times10^{-12}$.  The
small-$\omega$ expansion matched its leading term to roughly seven decimal
places in the tested lengths.

Across 21 unique configurations spanning $K\in\{16,32,64\}$, training
lengths 128 through 4096, bases 10K through 500K, and the exact-range
$b=256$ setting:

- pair volume improved the training full-subspace $r_2$, marginal phase
  coverage, and post-freeze 2x/4x/8x recurrence audit against Geo in every
  configuration;
- phase isotropy improved $r_2$ and marginal phase coverage in every
  configuration, and improved the recurrence audit in all configurations at
  2x and in 20/21 at 4x and 8x.

For the exact-range configuration:

| Static quantity | FMRoPE | pair volume | phase isotropy | anchored Cosh |
| --- | ---: | ---: | ---: | ---: |
| normalized median coordinate | 0.500 | 0.483 | **0.451** | 0.190 |
| full-subspace $r_2$ | 15.06 | 16.38 | **18.97** | 53.70 |
| mean marginal phase-cover radius | 0.540 | 0.489 | **0.421** | not used |
| max code correlation through 2x | 0.152 | 0.141 | **0.151** | 0.303 |
| max code correlation through 4x | 0.285 | 0.237 | **0.238** | 0.303 |
| max code correlation through 8x | 0.305 | 0.274 | **0.270** | 0.437 |

These numbers do not predict NLL.  They establish only that phase isotropy is
neither the conservative endpoint nor a Cosh replica, and that it passes the
static go/no-go checks without using the audited lengths to construct itself.

## 7. What the external trajectory analysis contributes

An independent external-model analysis correctly identified four exact
objects: training-frame conditioning, pair phase exposure, extrapolation
leverage, and code separation.  Its training Gram and phase-exposure
derivations are useful and lead directly to $S_\omega$, $v_L$, and
$i_L$ above.

Its final optimizer is not adopted.  It fixes $L_{\rm target}$, and its
minimum-separation definition includes adjacent lags, which can collapse local
resolution and far recurrence into the same number.  An epsilon-constraint
formulation also replaces scalar weights with uncalibrated thresholds; it does
not remove the selection problem.

The useful correction to $\mathcal C_{\rm app}$ is retained:

- a near-diagonal ridge in the exact frequency kernel can yield a local
  $\rho^2$ term after continuum scaling, with a coefficient that generally
  varies with $\phi$;
- $\min(\phi,\psi)$ is not the Taylor expansion of the full RoPE kernel.  It
  remains a nested slow-band Green-kernel model, with an additional restricted
  broadband interpretation under the historical cosine/log-uniform slice;
- Cosh is the constant-coefficient solution of that stated model, not the
  exact finite-$K$ rotary optimum.

These points clarify the present construction but do not make its trained
performance automatic.

## 8. Experiment contract

No new suite is required.  Use the existing 151.9M exact-range contract and
train one additional table.

### Gate A: durable CPU construction

Before GPU use:

1. implement the discrete causal $\chi_L$, phase-isotropy density, and
   endpoint quantiles in one analysis module;
2. assert endpoint identity, strict ordering, float32 table hash, and agreement
   between the determinant and characteristic-function formulas;
3. reproduce the 21-configuration screen and emit a machine-path-free receipt;
4. freeze the table before reading any new LM result.

### Gate B: one method-selection seed

Use seed 137 with the canonical FMRoPE and anchored-Cosh owners.  Within the
new arm, preserve initialization, stored-token prefix, row order, optimizer,
token budget, and evaluation anchors.  The table advances only if

- 256-token tail NLL is at most $+0.01$ above FMRoPE;
- all 512/1024/2048 contrasts favor phase isotropy over FMRoPE; and
- its weighted OOD gain is not merely the conservative pair-volume endpoint.

The last condition is evaluated by including pair volume only as a CPU
reference unless phase isotropy fails the first two gates; do not spend a
second training arm merely to confirm that the conservative table is
conservative.

### Gate C: independent seeds

If Gate B passes, freeze every construction and training choice and run the
remaining two canonical seeds.  Promotion requires all three seeds to preserve
the OOD direction and the mean 1x cost to remain at or below $+0.01$ NLL.

Mature-checkpoint retrofit remains a later, separate estimand.  A table that
works from initialization must still be transported into co-adapted weights;
the transplant obstruction and existing frozen controls continue to govern
that stage.

## 9. Paper routing

Nothing in this document changes the current ICLR manuscript.  The paper
should retain the full-subspace spectral-budget identity, fixed-support
identification, co-adaptation evidence, and Cosh as a minimal closed-form
construction under its declared surrogate.

If phase isotropy passes three training seeds, it can replace the construction
layer without changing the paper's identified variable (z).  If it fails,
the result rejects this target-free operating rule; it does not weaken the
existing evidence that allocation is a causal training variable.

## Canonical routes

- Full-RoPE geometry and counterexamples:
  `../../foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`
- Three-seed fixed-support owner:
  `../../EXACT_RANGE_151M_3SEED_RESULT_20260820.md`
- Completed phase-chord result:
  `../results/EXPERIMENT_REPORT_20260821.md`
- Failed attention-measure selector:
  `../../audits/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md`
- Failed LeRoPE structural-curvature oracle:
  `../../audits/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md`
- Current manuscript construction:
  `../../../sections/03_theory.tex`
