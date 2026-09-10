# Sol 02 — EVQ derivation, exact quantiles, and finite-window correction

## Decisive finding

EVQ-Cosh has a clean exact theorem, but the theorem is narrower than the phrase “exact EVQ” suggests.  The cosh density is the unique minimizer of the **projected pure-tether functional**

\[
\mathcal C_{\rm app}[\rho]
=\frac\alpha2\int_0^1\rho^2
+\frac\beta2\iint_{[0,1]^2}\rho(\phi)\rho(\psi)\min(\phi,\psi)\,d\phi d\psi,
\quad \rho\ge0,\quad\int\rho=1,
\]

for \(\alpha>0,\beta\ge0\).  It is not the minimizer of the original finite-window collision functional unless the kernel projection is exact.  At finite \(L\), the correct generalization retains the exact \(K_L\) and is naturally an integral problem in the continuum or a constrained gap problem for the actual finite table.  This distinction is material for unification with MrRoPE: a sharp slot transition is over-penalized by EVQ's local \(\delta\)-ridge replacement, so smoothness of a density is not a theorem-backed deployment criterion.

## 1. Reconstructed functional and proof

Let \(c=\log b\), \(\omega(\phi)=e^{-c\phi}\), and use the normalized log-uniform separation prior \(D_L(\Delta)=1/(\Delta\log L)\) on \([1,L]\).  The exact cosine collision kernel is

\[
K_L(\phi,\psi)=\int_1^L D_L(\Delta)
\cos(\omega(\phi)\Delta)\cos(\omega(\psi)\Delta)\,d\Delta.
\]

For \(\phi\ne\psi\), putting \(\delta=|\omega-\nu|\), \(\sigma=\omega+\nu\),

\[
K_L=\frac{\operatorname{Ci}(L\delta)-\operatorname{Ci}(\delta)
+\operatorname{Ci}(L\sigma)-\operatorname{Ci}(\sigma)}{2\log L}.
\]

On the diagonal the first Ci difference must be interpreted by its limit, \(\lim_{a\downarrow0}[\operatorname{Ci}(La)-\operatorname{Ci}(a)]=\log L\); writing the displayed Ci expression literally at \(\delta=0\) produces an undefined \(\infty-\infty\).  Equivalently,

\[
K_L(\phi,\phi)=\frac12+\frac{\operatorname{Ci}(2L\omega)-\operatorname{Ci}(2\omega)}{2\log L}.
\]

EVQ replaces this kernel by \(K_{\rm app}=\alpha\delta(\phi-\psi)+\beta\min(\phi,\psi)\).  The min kernel is the Green kernel of \(-d^2/d\phi^2\) with \(g(0)=0,g'(1)=0\), and

\[
\iint f(\phi)f(\psi)\min(\phi,\psi)d\phi d\psi
=\int_0^1\left(\int_s^1 f(u)du\right)^2ds\ge0.
\]

Thus \(\mathcal C_{\rm app}\) is coercive and strictly convex for \(\alpha>0\), and has a unique constrained minimizer.  With
\(g(\phi)=\int\rho(\psi)\min(\phi,\psi)d\psi\), stationarity is
\(\alpha\rho+\beta g+\nu=0\).  Since \(g''=-\rho\),

\[
\rho''-\tau^2\rho=0,\qquad \tau=\sqrt{\beta/\alpha}.
\]

The differentiated stationarity equation and mass constraint give
\(\rho'(1)=0\) and \(\rho'(0)=-\tau^2\).  Hence

\[
\boxed{\rho_\tau(\phi)=\frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau}}.
\]

It is positive everywhere, so the nonnegativity KKT constraint is inactive.  For \(\beta=0\), the unique solution is \(\rho\equiv1\), also the \(\tau\to0\) limit.

If the Fisher-resolution term \(-\mu_F\int\rho b^{-2\phi}\) is retained, the exact statement for the **surrogate** is instead

\[
\rho''-\tau^2\rho=\frac{4\mu_Fc^2}{\alpha}e^{-2c\phi},
\]

with a particular solution \(P e^{-2c\phi}\), \(P=(4\mu_Fc^2/\alpha)/(4c^2-\tau^2)\), away from resonance.  The pure cosh law is therefore exact only after removing this term, not merely after projecting the kernel.  Positivity of the Fisher-augmented stationary solution is not automatic and would require a KKT check.

## 2. Exact quantile and gap reformulation

The CDF and quantile of the pure-tether minimizer are

\[
F_\tau(\phi)=1-\frac{\sinh(\tau(1-\phi))}{\sinh\tau},\qquad
Q_\tau(u)=1-\frac1\tau\operatorname{asinh}((1-u)\sinh\tau).
\]

The derivative

\[
Q_\tau'(u)=\frac{\sinh\tau}{\tau\sqrt{1+(1-u)^2\sinh^2\tau}}
=\frac1{\rho_\tau(Q_\tau(u))}
\]

is strictly increasing in \(u\).  Therefore every equal-mass quantile grid has increasing exponent gaps toward the low-frequency end.  For any ordered grid \(0\le u_0<\cdots<u_{K-1}\le1\), the exact adjacent gaps are

\[
\boxed{g_k=Q_\tau(u_{k+1})-Q_\tau(u_k)
=\frac1\tau\left[
\operatorname{asinh}((1-u_k)\sinh\tau)
-\operatorname{asinh}((1-u_{k+1})\sinh\tau)
\right].}
\]

They satisfy \(g_{k+1}>g_k\) for an equally spaced \(u\)-grid.  For small \(\tau\),

\[
Q_\tau(u)=u-\frac{u(1-u)(2-u)}6\tau^2+O(\tau^4),
\]

so the frequency exponents move toward the fast end while the gaps tilt upward with \(k\).  This separates two often-confused claims: point locations shift to smaller \(\phi\); the low-frequency-end spacings become larger.

There are two different finite-table conventions in the sources:

* \(u_k=k/K\) anchors the fastest endpoint only and excludes \(u=1\).
* \(u_k=(k+1/2)/K\), used by the practical experiments and finite-transport proof, anchors neither endpoint.

Thus the endpoint-anchoring proposition does not describe the deployed midpoint EVQ table.  Any comparison to MrRoPE must freeze one convention; changing it changes every slot and can be as large as half a quantile cell.

For a general positive density, set \(Q=F^{-1}\).  Change of variables gives the exact continuum quantile form

\[
\mathcal E_L[Q]=\frac12\int_0^1\!\int_0^1K_L(Q(u),Q(v))\,du\,dv
-\mu_F\int_0^1 e^{-2cQ(u)}du,
\]

over nondecreasing \(Q:[0,1]\to[0,1]\).  This removes density/Jacobian bookkeeping because \(\rho(\phi)d\phi=du\).  Its interior first variation is

\[
\frac{\delta\mathcal E_L}{\delta Q(u)}
=\int_0^1\partial_1K_L(Q(u),Q(v))dv
+2c\mu_F e^{-2cQ(u)}.
\]

There is no cosh ODE for this exact finite-window functional.

For the actual \(K\)-pair table with equal channel weights, the exact finite objective is

\[
E_{L,K}(\boldsymbol\phi)=\frac1{2K^2}\sum_{i,j=0}^{K-1}K_L(\phi_i,\phi_j)
-\frac{\mu_F}{K}\sum_i e^{-2c\phi_i},
\quad 0\le\phi_0\le\cdots\le\phi_{K-1}\le1.
\]

Writing \(a=\phi_0\), \(g_r=\phi_r-\phi_{r-1}\ge0\), so \(\phi_m=a+\sum_{r\le m}g_r\), gives the exact gap gradient

\[
\boxed{\frac{\partial E_{L,K}}{\partial g_r}
=\sum_{m=r}^{K-1}\left[
\frac1{K^2}\sum_j\partial_1K_L(\phi_m,\phi_j)
+\frac{2c\mu_F}{K}e^{-2c\phi_m}
\right].}
\]

Interior optimal gaps make this cumulative force zero; nonnegative-gap and endpoint constraints add the usual complementary-slackness multipliers.  This is the concrete finite-window generalization to optimize or compare against a proposed EVQ/MrRoPE hybrid.  It is decision-sufficient and has no arbitrary candidate grid.

The kernel derivative can be evaluated without differentiating Ci:

\[
\partial_1K_L(\phi,\psi)=\frac{c\omega(\phi)}{\log L}
\int_1^L\sin(\omega(\phi)t)\cos(\omega(\psi)t)dt,
\]

with the continuous limiting interpretation when sum/difference frequencies coincide.

## 3. Why finite-window EVQ is not local Cosh

For distinct interior frequencies the exact decomposition in the correction note is

\[
K_L(\phi,\psi)=\frac{c\min(\phi,\psi)-\gamma+h_c(\phi-\psi)}{\log L}+E_L,
\qquad h_c(x)=-\tfrac12\log(1-e^{-2c|x|}).
\]

The ridge \(h_c\) is nonlocal and integrable.  Its Fourier multiplier is positive and decreases with wavenumber:

\[
\widehat h_c(k)=\frac{\pi}{2|k|}\coth\frac{\pi|k|}{2c}-\frac c{k^2},
\quad \widehat h_c(0)=\frac{\pi^2}{12c}.
\]

Replacing it by \(\widehat h_c(0)I\) overcharges every nonconstant density mode.  The first long-wave correction is negative,

\[
\langle\rho,h_c*\rho\rangle
=A_0\|\rho\|_2^2-\frac{\pi^4}{720c^3}\|\rho'\|_2^2+\cdots,
\]

but this truncated derivative functional is unbounded below at high wavenumber and has endpoint terms for positive finite-interval densities.  It cannot safely be optimized.  Retaining the full \(K_L\), as in the quantile/gap objective above, is the constructive fix.

## 4. Proved, modeled, and empirical claims

**Mathematically proved under stated assumptions**

1. PSD of the min kernel, strict convexity for \(\alpha>0\), existence/uniqueness, the cosh minimizer, its CDF/inverse, increasing gaps, geometric \(\tau\to0\) limit, single crossing, self-consistency identity, and finite quantile transport bounds.
2. Exact Ci representation (with a diagonal limit), exact nonlocal-ridge Fourier multiplier, and the fact that the delta replacement over-penalizes all nonzero density wavenumbers.
3. The post-hoc transplant obstruction in the appendix: position-independent invertible Q/K maps cannot exactly change a frozen model's rotary frequency multiset on an open interval.

**Modeling assumptions / approximations**

1. The cosine-only collision kernel itself omits the full sin/cos positional subspace, signed content coefficients, learned Q/K use, and nonlinear attention behavior.
2. Projecting \(K_L\) to \(\alpha I+\beta A^{-1}\) is the decisive approximation behind Cosh.  Its accuracy is nonuniform at the diagonal and endpoints.
3. The \(\tau\propto d_{\rm head}/\sqrt L\) calculation additionally assumes small \(\tau\), diffuse reference attention, full-RoPE MHA, positive \(Q_1\), and a fixed trade-off \(\lambda\).  It derives a scaling form inside that model, not a universal optimum.

**Empirical only**

1. Reported kernel-fit \(R^2\), the 99-run win counts, and the chosen coefficient \(c=1\).
2. Any downstream claim that Cosh or a sharper table improves language-model long-context behavior.

## 5. Training-from-scratch versus frozen deployment

For training from scratch, \(E_{L,K}\) can serve as a geometry regularizer or initialization prior while Q/K weights co-adapt.  A controlled proposal is to minimize the exact finite-window gap objective subject to a stated support/grid convention, then train it as a candidate.  Even here its collision term is only a prior; downstream behavior must decide.

For frozen deployment, table replacement changes the phase generators and cannot generally be absorbed by fixed position-independent maps.  The learned association between slot \(i\), frequency \(\omega_i\), and Q/K content must be preserved.  Therefore the constructive rule is a **displacement-budgeted finite-window step**, not a fresh density optimum:

\[
\min_{\boldsymbol\phi}\ E_{L_{\rm target},K}(\boldsymbol\phi)
+\frac12\sum_i w_i(\phi_i-\phi_i^{(0)})^2,
\]

with monotone gaps and hard-fixed slots where evidence says association is critical.  The weights \(w_i\) are deployment sensitivity terms to be measured from the frozen model (e.g. held-out loss curvature or controlled slot perturbations), not inferred from the cosine geometry.  EVQ supplies the broadband prior; MrRoPE-like selective preservation enters through the displacement weights or hard constraints.  This is a mathematically correct unification framework, but it does not prove downstream improvement.

## 6. Consequence for the next decision

Do not choose between EVQ and MrRoPE by density smoothness or by the Cosh functional alone.  Evaluate any proposed allocation on the exact \(E_{L,K}\) at both training and target windows, record its per-gap cumulative forces, and separately measure frozen-model slot sensitivity.  A candidate is justified for frozen testing only when it improves the exact finite-window objective without spending displacement on empirically sensitive slots.  The downstream long-context task result remains the acceptance criterion.

## Source anchors

* `docs/theory/EVQ_COSH_THEORY.tex:59-125, 130-214, 225-298`
* `paper-2027/appendix/a1_proofs.tex:315-367, 373-442, 451-561`
* `paper-2027/appendix/a1_proofs.tex:288-312` (frozen transplant obstruction)
* `paper-2027/appendix/budget_proofs.tex:1-55` (scope of finite geometry bounds)
* `docs/research/EVQ_NONLOCAL_KERNEL_CORRECTION_20260910.md:8-114, 116-138`

