# What EVQ's local congestion approximation discards

Mathematical note, CPU only, 2026-09-10. This examines the original cosine
collision model in `docs/theory/EVQ_COSH_THEORY.tex`. It does not replace the
full rotary sin/cos calculation by a cosine-only performance claim, and it is
not a new frequency table submitted to a model.

## Exact interior decomposition

Put c = log b, omega = exp(-c phi), nu = exp(-c psi), and use the original
distance prior dt/(t log L) on [1,L]. For distinct frequencies the exact kernel is

\[
K_L(\phi,\psi)=\frac{\operatorname{Ci}(L\delta)-\operatorname{Ci}(\delta)
+\operatorname{Ci}(L\sigma)-\operatorname{Ci}(\sigma)}{2\log L},
\quad \delta=|\omega-\nu|,\quad\sigma=\omega+\nu.
\]

Write Ci(z) = gamma + log z + R(z), where
R(z) = integral from 0 to z of (cos t - 1)/t dt and |R(z)| <= z²/4.
Integration by parts at infinity also gives |Ci(z)| <= 2/z. Consequently,

\[
K_L(\phi,\psi)
=\frac{c\min(\phi,\psi)-\gamma+h_c(\phi-\psi)}{\log L}
+E_L(\phi,\psi),
\]

\[
h_c(x)=-\tfrac12\log(1-e^{-2c|x|}),\qquad
|E_L|\le\frac{1}{\log L}\left[
\frac1{L\delta}+\frac1{L\sigma}+\frac{\delta^2+\sigma^2}{8}\right].
\]

This follows from delta sigma = exp(-2c min(phi,psi))
times (1-exp(-2c|phi-psi|)). The displayed bound holds for positive delta;
it is useful in the interior regime delta,sigma << 1 and L delta >> 1.
It is deliberately not a uniform approximation at the diagonal, the fastest
endpoint, or the unresolved slow endpoint.

The omitted interaction is therefore a nonlocal, integrable logarithmic ridge.
It is not simply independent per-frequency noise. Replacing it by a delta
kernel is an additional locality approximation.

## The exact ridge multiplier and the sign of its first correction

On the real line,

\[
h_c(x)=\frac12\sum_{n=1}^{\infty}\frac{e^{-2cn|x|}}n,
\qquad A_0:=\int_{\mathbb R}h_c(x)dx=\frac{\pi^2}{12c}.
\]

For the Fourier transform convention h-hat(k) = integral h(x)exp(-ikx)dx,

\[
\widehat h_c(k)=\sum_{n=1}^{\infty}\frac{2c}{4c^2n^2+k^2}
=\frac{\pi}{2|k|}\coth\frac{\pi|k|}{2c}-\frac{c}{k^2},
\quad \widehat h_c(0)=A_0.
\]

The multiplier is positive, decreases with |k|, and is at most A0. Hence
the delta approximation A0 times the identity overpenalizes variation at
every nonzero density wavenumber. For a sufficiently smooth, decaying density,

\[
\langle\rho,h_c*\rho\rangle
=A_0\|\rho\|_2^2-\frac{\pi^4}{720c^3}\|\rho'\|_2^2
+\text{higher-order terms}.
\]

The negative sign is fixed by the kernel; adding a positive adjacent-gradient
penalty would move in the opposite direction. This expansion is valid only
for density variation scales with |k|/c small. The truncated functional is
not bounded below at large k and must never be optimized without its regime
restriction. The full nonlocal positive kernel remains well behaved.

For a density supported on [0,1], zero extension preserves the exact quadratic
Fourier identity. The derivative expansion additionally requires suitable
endpoint regularity; a positive endpoint density has a jump under zero
extension and cannot be inserted into the derivative formula without boundary
terms. Thus this expression is an interior diagnostic, not a ready-made
replacement functional for the finite-interval EVQ problem.

## Consequence for the allocation question

The EVQ locality approximation is most plausible for broad density variation.
It becomes less accurate when the finite table redistributes mass over one or
two frequency slots. In particular, the original variational argument does
not justify penalizing every sharp middle transition as intrinsically bad:
the actual ridge charges narrow variations less than its delta replacement.

This is a specific correction to a modeling step, not evidence that a sharp
transition improves a frozen model. The exact cosine collision objective also
omits signed content coefficients, sin/cos cross-relations, and learned use of
the channels. The existing source-subspace calculation supplies one of these
missing pieces, but even its Q/K-weighted version does not rank all observed
tables correctly.

Two additional limits matter before claiming an improved rule:

1. The diagonal layer is finite in the exact kernel: K(phi,phi) is bounded,
   whereas h diverges logarithmically. Its log-frequency width is of order
   1/(L c omega), so it varies strongly across the middle and slow bands.
2. A global ridge-only optimization would use this approximation precisely
   where it is least justified. The exact finite-window kernel, source slot
   association, and the required long-distance computation must be retained
   before an optimizer can be promoted to a deployment rule.

The useful prediction is about the mathematical approximation itself:
holding its other terms fixed, the local EVQ congestion term increasingly
overestimates the ridge cost as an allocation change becomes narrower in
log frequency. This can be checked analytically and with CPU arithmetic;
it is not a prediction of downstream accuracy.

## CPU checks

With b = 10^6, the exact ridge multiplier divided by its local-delta value is
0.991596 at k = pi, 0.967550 at 2 pi, 0.885888 at 4 pi, 0.599065 at k = 32,
and 0.355619 at k = 64. Direct adaptive quadrature of the defining Fourier
integral agrees with the closed form within 3.8e-15 on these five points.
Here k is an angular wavenumber of the density variation, not a rotary slot
index or an input length.

For L = 131072, direct Ci evaluation gives:

| phi, psi | Exact kernel | Interior approximation | Absolute error | Derived error bound |
|---|---:|---:|---:|---:|
| .20, .24 | .202667740 | .202568816 | .000098924 | .000143115 |
| .30, .34 | .319846598 | .319813337 | .000033261 | .000129247 |
| .40, .44 | .437205520 | .437057858 | .000147662 | .000486748 |
| .50, .54 | .553541756 | .554302379 | .000760623 | .001936025 |
| .65, .69 | .734646029 | .730169161 | .004476868 | .015378172 |

All five error bounds hold. The increasing slow-end error illustrates the
boundary limitation rather than licensing an extrapolation of the interior
formula into the unresolved band. Computed with SciPy 1.18.1 on the already
available CPU server; no model was loaded or executed.
