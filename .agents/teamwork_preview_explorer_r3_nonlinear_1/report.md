# Research Report: Mathematical & Physical Impact of Non-Linear Warping $z \to f(z)$ vs. Linear Scaling $cz$ in RoPE Spectral Allocation

- **Agent:** `explorer_r3_nonlinear_1` (Focus R3)
- **Working Directory:** `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_explorer_r3_nonlinear_1`
- **Date:** 2026-09-01 (UTC)
- **Status:** Complete Research Synthesis & Theoretical Derivation
- **Target Manuscript:** ICLR 2027 (*RoPE Has a Spectral Budget*)

---

## Executive Summary

This report delivers a rigorous mathematical and physical investigation of non-linear spectral warping $z \mapsto f(z)$ compared to linear scaling $f(z) = cz$ within the Rotary Position Embedding (RoPE) framework.

Key findings established in this report:
1. **Linear Scaling Isomorphism:** Linear scaling $f(z) = cz$ is mathematically isomorphic to a uniform base dilation $b \mapsto b^c$. It modifies only the sampled log-frequency support span $R$, leaving the relative logarithmic channel density and normalized interior allocation $z_k = k/(K-1)$ strictly invariant.
2. **Non-Linear Spectral Warping:** Non-linear warping $f(z) \neq cz$ under fixed boundary support ($f(0)=0, f(1)=1$) constitutes a distinct, orthogonal design axis that reallocates channel capacity across frequency bands without altering the total spectral range.
3. **Channel Density Transformation:** The realized channel density in continuous frequency space is given by $\rho_f(\omega) = \frac{1}{\omega \ln b \cdot |f'(f^{-1}(-\log_b \omega))|}$. Warping curves with negative second derivative in log-frequency (such as Cosh) systematically transfer capacity from the redundant low-frequency collapse regime ($\omega L \ll 1$) into high- and mid-frequency bands.
4. **Physical Wave-Packet Dynamics:** RoPE attention query-key logits act as wave-packet superpositions $\Psi(\Delta) = \int A(k) e^{i(\omega(k)\Delta + \psi(k))} dk$. Geometric RoPE suffers from extreme exponential group velocity dispersion (GVD), causing phase stagnation at low frequencies and rapid dephasing at high frequencies. Non-linear warping redistributes group velocities $v_g(k) = d\omega/dk$, shaping the coherence length $L_{\text{coh}}(\Delta)$ across the context window.
5. **Variational Uniqueness of EVQ-Cosh:** EVQ-Cosh is the unique, closed-form minimizer of the convex quadratic surrogate functional $\mathcal{C}_{\text{app}}[\rho] = \frac{\alpha}{2}\int \rho^2 + \frac{\beta}{2}\iint \rho(\phi)\rho(\psi)\min(\phi,\psi)$. This uniqueness is strictly conditional on the surrogate and does not imply universal optimality for full-RoPE Grassmannian geometry or downstream language model cross-entropy loss.
6. **Post-Mortem on Falsified Routes:**
   - The *Arcsine Conjecture (O5)* fails because the full-RoPE kernel leading term $\min(\phi, \psi)^2$ is not a Green's function for any local differential operator, and free numerical optimization yields a monotone decreasing profile rather than a U-shape.
   - *Naive Collision/Logdet Minimization* fails because unconstrained minimization over a window $[0, L]$ drives frequencies into a harmonic Fourier comb ($\omega_k = 2\pi k / L$), causing catastrophic exact aliasing $\mathcal{R}(\Delta + L) = \mathcal{R}(\Delta)$ that destroys out-of-distribution extrapolation.

---

## 1. Isomorphism of Linear Scaling $f(z) = cz$ and Base Change $b \to b^c$

### 1.1 Parameterization and Coordinate Conventions

Let $d$ be the head dimension, and let $K = d/2$ denote the number of orthogonal 2D rotary pairs. In standard RoPE (Su et al., 2024), the discrete frequencies are defined as:
$$\omega_i = b^{-2i/d} = b^{-i/K}, \qquad i \in \{0, 1, \dots, K-1\},$$
where $b > 1$ is the base wavelength parameter (e.g., $b = 10{,}000$ or $500{,}000$).

We define two standard continuous coordinate parameterizations:
1. **Normalized Log-Frequency Coordinate $\phi \in [0, 1]$ (or $z \in [0, 1]$):**
   $$\omega(\phi) = b^{-\phi} = \exp(-\phi \ln b), \qquad \phi \in [0, 1].$$
   Here $\phi = 0$ corresponds to the fastest frequency ($\omega_{\max} = 1$), and $\phi = 1$ corresponds to the slowest frequency ($\omega_{\min} = b^{-1}$).
2. **Negative Exponent Coordinate $z \in [-1, 0]$:**
   $$\omega(z) = b^z = \exp(z \ln b), \qquad z \in [-1, 0],$$
   where $z = -2i/d$. Here $z = 0$ is the fast boundary and $z = -1$ is the slow boundary.

### 1.2 Mathematical Proof of Isomorphism

Let $\mathcal{F}_b = \{ \omega(z) = b^z \mid z \in [-1, 0] \}$ denote the continuous frequency spectrum parameterized by base $b$.

**Theorem 1 (Linear Scaling / Base Change Isomorphism).**
*Let $c > 0$ be a scalar dilation factor. The linear scaling transformation of the coordinate $T_c: z \mapsto cz$ on the frequency map $\omega_b(z) = b^z$ is algebraically isomorphic to a scalar base transformation $b \mapsto b' = b^c$.*

*Proof.*
Consider the transformed frequency function under coordinate scaling $T_c$:
$$\omega_{\text{scaled}}(z) := \omega_b(T_c(z)) = \omega_b(cz) = b^{cz}.$$
Using the algebraic identity for real exponentials $(x^y)^z = x^{yz}$:
$$b^{cz} = (b^c)^z.$$
Define the new base $b' := b^c$. Then:
$$\omega_{\text{scaled}}(z) = (b')^z = \omega_{b'}(z).$$
Conversely, let $b \mapsto b'$ be any base change. Since $b, b' > 1$, there exists a unique positive scalar $c = \frac{\ln b'}{\ln b} > 0$ such that $b' = b^c$. Then:
$$\omega_{b'}(z) = (b^c)^z = b^{cz} = \omega_b(cz) = \omega_b(T_c(z)).$$
Thus, the action of the multiplicative group $(\mathbb{R}_{>0}, \cdot)$ on the coordinate space via $z \mapsto cz$ is isomorphic to its action on the base parameter space via $b \mapsto b^c$. $\blacksquare$

### 1.3 Invariance of Relative Logarithmic Channel Density

In log-frequency space, define $x = -\ln \omega = -z \ln b$.
The differential spacing between two channels $z_1, z_2$ is:
$$\Delta x = x_2 - x_1 = -(z_2 - z_1) \ln b = -\Delta z \ln b.$$
Under linear scaling $z \mapsto cz$:
$$\Delta x' = -(c z_2 - c z_1) \ln b = -c \Delta z \ln b = -\Delta z \ln(b^c).$$
Consider the relative allocation ratio between any two intervals $[z_1, z_2]$ and $[z_3, z_4]$:
$$\frac{\Delta x'_{12}}{\Delta x'_{34}} = \frac{-c(z_2 - z_1)\ln b}{-c(z_4 - z_3)\ln b} = \frac{z_2 - z_1}{z_4 - z_3} = \frac{\Delta x_{12}}{\Delta x_{34}}.$$
Therefore, linear scaling $cz$ preserves all relative logarithmic interval ratios. It is a pure affine dilation (stretching) of the log-frequency axis.

### 1.4 Orthogonality in the Causal Coordinate System

Following the causal decomposition established in repository owner `ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`:
$$x_k = -\ln \omega_k = a + R z_k, \qquad z_0 = 0, \quad z_{K-1} = 1,$$
where:
- $a = x_0 = -\ln \omega_0$ is the fast boundary anchor;
- $R = x_{K-1} - x_0 = \ln(\omega_0 / \omega_{K-1})$ is the total log-frequency support span;
- $z_k \in [0, 1]$ is the normalized interior allocation vector.

Under linear scaling / base change $b \to b^c$:
$$a' = c a, \qquad R' = c R, \qquad z_k' = \frac{x_k' - a'}{R'} = \frac{c x_k - c a}{c R} = \frac{x_k - a}{R} = z_k.$$
**Corollary 1.1.** *Linear scaling $f(z) = cz$ operates exclusively on the support parameters $(a, R)$ and has zero projection onto the interior allocation coordinate $z$. It cannot reallocate capacity between high and low frequencies within a fixed support.*

---

## 2. Continuous Formulation of Non-Linear Warping $z \mapsto f(z)$

### 2.1 Boundary Conditions and Fixed Support

A non-linear warping is a differentiable, strictly monotonic mapping $f: [0, 1] \to [0, 1]$ satisfying the fixed support boundary constraints:
$$f(0) = 0, \qquad f(1) = 1, \qquad f'(\phi) > 0 \quad \forall \phi \in [0, 1].$$
When expressed in negative exponent coordinates $z \in [-1, 0]$:
$$f(-1) = -1, \qquad f(0) = 0, \qquad f'(z) > 0 \quad \forall z \in [-1, 0].$$

Because $f(0) = 0$ and $f(1) = 1$, the sampled support endpoints $\omega_0 = 1$ and $\omega_{K-1} = b^{-1}$ are strictly pinned:
$$a' = a, \qquad R' = R.$$
All modifications introduced by $f$ represent pure interior reallocation $z \to f(z)$.

### 2.2 Local Curvature and Non-Linear Distortion

The deviation of $f$ from linearity is governed by its local curvature $f''(\phi)$:
- **Linear baseline ($f''(\phi) = 0$):** $f(\phi) = \phi \implies z_k = \frac{k}{K-1}$ (Uniform log-spacing).
- **Strictly Concave Warping ($f''(\phi) < 0$):** $f'(\phi)$ is strictly decreasing. The mapping stretches intervals near $\phi = 0$ (high frequencies) and compresses intervals near $\phi = 1$ (low frequencies).
- **Strictly Convex Warping ($f''(\phi) > 0$):** $f'(\phi)$ is strictly increasing, compressing high frequencies and expanding low frequencies.

```
Log-Frequency Coordinate Warp:
  Uniform Grid phi_k:   0 ---- 1/4 ---- 2/4 ---- 3/4 ---- 1
  Concave Warp f(phi):  0 -------- 0.45 ---- 0.75 -- 0.90 - 1
                        [  Dense High Freq  ] [ Sparse Low ]
```

### 2.3 Structural Comparison: Linear Scaling vs. Non-Linear Warping

| Property | Linear Scaling $f(z) = cz$ | Non-Linear Warping $f(z) \neq cz$ |
| :--- | :--- | :--- |
| **Algebraic Nature** | Isomorphic to base change $b \to b^c$ | Functional deformation of spectrum |
| **Support Span $R = \ln(\omega_{\max}/\omega_{\min})$** | Modifies $R \mapsto cR$ | Strictly invariant ($R' = R$) |
| **Interior Allocation $z_k$** | Invariant ($z_k' = z_k$) | Warped ($z_k' = f(z_k) \neq z_k$) |
| **Relative Log-Bandwidth** | Constant across all octaves | Non-uniform across octaves |
| **Degrees of Freedom** | 1 scalar parameter ($c$) | Infinite-dimensional functional $\rho(\phi) \in L^2$ |
| **Causal Identification** | Support extension | Pure interior allocation (Exact-range) |

---

## 3. Spectral Channel Density Modification $\rho_f(\omega)$

### 3.1 Exact Derivation of Continuous Channel Density

Let $u \in [0, 1]$ represent the normalized channel rank (cumulative channel fraction from fast to slow).
Let $\phi(u) = f(u)$ map channel fraction $u$ to the log-frequency coordinate $\phi \in [0, 1]$, where $\omega(\phi) = \omega_0 b^{-\phi}$.
Without loss of generality, set $\omega_0 = 1$, so $\omega = b^{-\phi} = b^{-f(u)}$.

To find the channel density per unit frequency $\rho_f(\omega) = \left| \frac{du}{d\omega} \right|$:
1. Invert the frequency relation:
   $$\ln \omega = -f(u) \ln b \implies f(u) = -\frac{\ln \omega}{\ln b} = -\log_b \omega.$$
2. Since $f$ is strictly increasing, $f^{-1}$ exists:
   $$u(\omega) = f^{-1}(-\log_b \omega).$$
3. Differentiate $u$ with respect to $\omega$:
   $$\frac{du}{d\omega} = \frac{d}{d\omega} \left[ f^{-1}\left( -\frac{\ln \omega}{\ln b} \right) \right] = (f^{-1})'(-\log_b \omega) \cdot \left( -\frac{1}{\omega \ln b} \right).$$
4. By the Inverse Function Theorem, $(f^{-1})'(y) = \frac{1}{f'(f^{-1}(y))}$:
   $$\rho_f(\omega) = \left| \frac{du}{d\omega} \right| = \frac{1}{\omega \ln b \cdot \left| f'\left( f^{-1}\left( -\log_b \omega \right) \right) \right|}.$$

### 3.2 Analysis of Frequency Band Reallocations

#### Case 1: Standard Geometric RoPE ($f(u) = u \implies f'(u) = 1$)
$$\rho_{\text{geo}}(\omega) = \frac{1}{\omega \ln b}.$$
In linear frequency space, $\rho_{\text{geo}}(\omega) \propto 1/\omega$.
- As $\omega \to 0$, $\rho_{\text{geo}}(\omega) \to \infty$. A disproportionate physical channel density is concentrated in infinitesimal frequency intervals near zero.
- In log-frequency space, however, the density is uniform:
  $$\rho_{\text{geo}}^{\text{log}}(\phi) = \rho_{\text{geo}}(\omega) \left| \frac{d\omega}{d\phi} \right| = \frac{1}{\omega \ln b} \cdot (\omega \ln b) = 1.$$

#### Case 2: EVQ-Cosh Warping
In EVQ-Cosh, the continuous log-density is given by:
$$\rho_\tau(\phi) = \frac{\tau \cosh(\tau(1 - \phi))}{\sinh \tau}, \qquad \tau = \sqrt{\beta/\alpha} > 0.$$
The corresponding warping quantile function is $\phi(u) = 1 - \frac{1}{\tau}\operatorname{arcsinh}((1 - u)\sinh \tau)$, with derivative:
$$f'(u) = \frac{d\phi}{du} = \frac{\sinh \tau}{\tau \sqrt{1 + (1 - u)^2 \sinh^2 \tau}} = \frac{1}{\rho_\tau(\phi(u))}.$$
Substituting into the general density formula:
$$\rho_{\text{EVQ}}(\omega) = \frac{\rho_\tau(-\log_b \omega)}{\omega \ln b} = \frac{\tau \cosh\left(\tau(1 + \log_b \omega)\right)}{\omega \ln b \sinh \tau}.$$

### 3.3 Mitigation of Low-Frequency Spectral Collapse

**Proposition 1 (Low-Frequency Collapse; Full RoPE Report §3).**
*Over context window $[0, L]$, as $\omega L \to 0$, the rotary subspace $V_\omega = \operatorname{span}\{\cos(\omega\Delta), \sin(\omega\Delta)\}$ converges to $\operatorname{span}\{1, \Delta\}$. Under the attention softmax metric (annihilating the constant mode), the subspace converges to the centered 2D limit $\operatorname{span}\{\Delta - \mathbb{E}_p\Delta, \Delta^2 - \mathbb{E}_p\Delta^2\}$ at order $O((\omega L)^4)$.*

In geometric RoPE, all slow channels with $\omega L \ll 1$ occupy separate pairs $k$ but span virtually identical positional subspaces.
- In EVQ-Cosh, at the slow boundary $\phi = 1$ ($\omega = b^{-1}$):
  $$\rho_\tau(1) = \frac{\tau}{\sinh \tau} < 1 \quad (\text{for } \tau > 0).$$
  For $\tau = 2.0$, $\rho_2(1) = \frac{2}{\sinh 2} \approx 0.5516$. Low-frequency channel allocation is reduced by $\approx 45\%$.
- At the fast boundary $\phi = 0$ ($\omega = 1$):
  $$\rho_\tau(0) = \tau \coth \tau > 1.$$
  For $\tau = 2.0$, $\rho_2(0) = 2 \coth 2 \approx 2.0746$. High-frequency channel allocation is increased by $+107\%$.

**Theorem 2 (Single-Crossing Budget Shift; Lemma \ref{lem:budget-crossing}).**
*For every $\tau > 0$, $\rho_\tau(\phi)$ crosses the uniform geometric density $\rho \equiv 1$ exactly once, at:*
$$\phi_c(\tau) = 1 - \frac{1}{\tau}\operatorname{arcosh}\left(\frac{\sinh \tau}{\tau}\right) \le 1 - \frac{1}{\sqrt{3}} \approx 0.4226.$$
*Proof.* $\rho_\tau(\phi)$ is strictly decreasing in $\phi \in [0, 1]$. Since $\rho_\tau(0) = \tau\coth\tau > 1$ and $\rho_\tau(1) = \tau/\sinh\tau < 1$, the Intermediate Value Theorem guarantees a unique root $\phi_c$ to $\rho_\tau(\phi) = 1$. Solving $\cosh(\tau(1-\phi_c)) = \frac{\sinh\tau}{\tau}$ gives $\phi_c = 1 - \tau^{-1}\operatorname{arcosh}(\sinh\tau/\tau)$.
Using the Taylor expansion $\frac{\sinh\tau}{\tau} = \sum_{n=0}^\infty \frac{\tau^{2n}}{(2n+1)!} \ge \sum_{n=0}^\infty \frac{\tau^{2n}}{3^n (2n)!} = \cosh(\tau/\sqrt{3})$, applying $\operatorname{arcosh}$ yields $\operatorname{arcosh}(\sinh\tau/\tau) \ge \tau/\sqrt{3}$, hence $\phi_c \le 1 - 1/\sqrt{3}$. $\blacksquare$

---

## 4. Physical Wave-Packet Perspective: Dispersion, Velocities, and Coherence

### 4.1 RoPE Attention Logits as Wave-Packet Superpositions

For a single attention head with query $\mathbf{q}$ and key $\mathbf{k}$ separated by positional lag $\Delta$, the pre-softmax logit is:
$$\ell(\Delta) = \mathbf{q}^\top \mathcal{R}_\Omega(\Delta) \mathbf{k} = \sum_{j=0}^{K-1} A_j \cos(\omega_j \Delta + \psi_j) = \operatorname{Re}\left[ \sum_{j=0}^{K-1} A_j e^{i \psi_j} e^{i \omega_j \Delta} \right],$$
where $A_j, \psi_j$ are content-dependent amplitudes and phases.
In the continuum limit ($K \gg 1$), let $k \in [0, K]$ be the continuous mode index. The attention logit is a wave-packet:
$$\Psi(\Delta) = \int_0^K A(k) e^{i [\omega(k) \Delta + \psi(k)]} dk.$$

### 4.2 Phase Velocity vs. Group Velocity

In physical wave mechanics, where $\Delta$ represents spatial displacement and $k$ represents mode/channel index:
1. **Phase Velocity ($v_p$):** The rate at which the phase of an individual channel advances per unit mode index:
   $$v_p(k) = \frac{\omega(k)}{k}.$$
2. **Group Velocity ($v_g$):** The rate of envelope drift and interference beat frequency across adjacent channels:
   $$v_g(k) = \frac{d\omega(k)}{dk}.$$

#### Dispersion in Geometric RoPE
With $\omega(k) = b^{-k/K} = \exp\left(-\frac{k}{K} \ln b\right)$:
- Group velocity:
  $$v_g^{\text{geo}}(k) = \frac{d\omega}{dk} = -\frac{\ln b}{K} \exp\left(-\frac{k}{K}\ln b\right) = -\frac{\ln b}{K} \omega(k).$$
- Group Velocity Dispersion (GVD / second-order dispersion):
  $$\text{GVD}^{\text{geo}}(k) = \frac{d^2\omega}{dk^2} = \left(\frac{\ln b}{K}\right)^2 \omega(k) > 0.$$

Notice the severe exponential disparity in group velocity across the channel spectrum:
$$\frac{|v_g(0)|}{|v_g(K-1)|} = \frac{\omega(0)}{\omega(K-1)} = b.$$
For $b = 500{,}000$, the fastest channels travel $500{,}000\times$ faster in phase space than the slowest channels.

#### Dispersion in Non-Linear Warped RoPE
With $\omega(k) = b^{-f(k/K)}$:
$$v_g^{\text{warp}}(k) = -\frac{\ln b}{K} f'\left(\frac{k}{K}\right) \omega(k),$$
$$\text{GVD}^{\text{warp}}(k) = \left(\frac{\ln b}{K}\right)^2 \left[ (f'(k/K))^2 - \frac{K}{\ln b} f''(k/K) \right] \omega(k).$$
By engineering $f'(u)$, non-linear warping directly shapes the group velocity profile across mode index $k$.

```
Group Velocity Profile |v_g(k)|:
  Geometric:   High Fast ========\________ Very Low Slow (Stagnation)
  EVQ-Cosh:    High Fast ===========\_____ Balanced Intermediate Band
```

### 4.3 Wave-Packet Broadening, Dephasing, and Coherence Length

Consider the interference of a narrow band of channels $\Delta k$ centered around mode $k_0$.
By Taylor expansion of the dispersion relation $\omega(k) \approx \omega(k_0) + v_g(k_0)(k - k_0) + \frac{1}{2}\text{GVD}(k_0)(k - k_0)^2$:
$$\Psi_{\text{band}}(\Delta) \approx e^{i \omega(k_0)\Delta} \int_{-\Delta k/2}^{\Delta k/2} A(k_0) e^{i [v_g(k_0) \Delta \cdot \delta k + \frac{1}{2}\text{GVD}(k_0) \Delta \cdot (\delta k)^2]} d(\delta k).$$
- **First-Order Beat Envelope:** The local interference envelope is modulated by $\operatorname{sinc}\left(\frac{v_g(k_0) \Delta \cdot \Delta k}{2}\right)$.
- **Coherence Length ($L_{\text{coh}}$):** The spatial distance $\Delta$ over which adjacent modes maintain phase coherence before destructive interference (dephasing) occurs:
  $$L_{\text{coh}}(k) \sim \frac{2\pi}{|v_g(k)| \Delta k} = \frac{2\pi K}{\ln b \cdot |f'(k/K)| \omega(k) \Delta k}.$$

#### Physical Failure Mode of Extrapolation in Geometric RoPE:
1. **High Frequencies ($\omega \Delta \gg 2\pi$):** For out-of-distribution lags $\Delta > L_{\text{train}}$, $L_{\text{coh}}$ is tiny. High-frequency modes undergo rapid dephasing and severe phase wrapping (aliasing), causing high-frequency attention logits to degrade into pseudorandom noise.
2. **Low Frequencies ($\omega \Delta \ll 1$):** $L_{\text{coh}}$ is immense ($L_{\text{coh}} \to \infty$). Channels fail to dephase at all, remaining mutually collinear and providing zero fine-grained positional distinguishability.
3. **The Non-Linear Solution:** Non-linear warping broadens the active mid-frequency band, increasing the effective density of channels whose coherence lengths match the critical interpolation and extrapolation range $[L_{\text{train}}, L_{\text{target}}]$.

---

## 5. The EVQ-Cosh Variational Construction and Uniqueness Bounds

### 5.1 Formulation of the Convex Surrogate Functional $\mathcal{C}_{\text{app}}[\rho]$

To find an optimal continuous allocation without solving an intractable, non-convex, oscillatory trigonometric eigenvalue problem, EVQ-Cosh introduces the convex quadratic surrogate functional:
$$\mathcal{C}_{\text{app}}[\rho] = \frac{\alpha}{2} \int_0^1 \rho(\phi)^2 d\phi + \frac{\beta}{2} \iint_{[0, 1]^2} \rho(\phi)\rho(\psi)\min(\phi, \psi) d\phi d\psi,$$
subject to:
$$\rho \in C^2([0, 1]), \qquad \rho(\phi) \ge 0, \qquad \int_0^1 \rho(\phi) d\phi = 1,$$
where $\alpha > 0$ represents channel load regularization (variance penalty) and $\beta \ge 0$ represents pairwise slow-end redundancy penalty.

### 5.2 Strict Convexity and Positive Semi-Definiteness

**Lemma 1 (PSD of the Green Kernel; App. \ref{sec:proofs}).**
*The integral operator with kernel $K(\phi, \psi) = \min(\phi, \psi)$ is positive semi-definite on $L^2([0, 1])$.*

*Proof.* Using the identity $\min(\phi, \psi) = \int_0^1 \mathbf{1}\{s \le \phi\} \mathbf{1}\{s \le \psi\} ds$:
$$\iint_{[0, 1]^2} f(\phi) f(\psi) \min(\phi, \psi) d\phi d\psi = \int_0^1 \left( \int_0^1 f(\phi) \mathbf{1}\{s \le \phi\} d\phi \right)^2 ds = \int_0^1 \left( \int_s^1 f(u) du \right)^2 ds \ge 0.$$
Since $\alpha > 0$, the $L^2$ norm $\int \rho^2$ is strictly convex. Therefore, $\mathcal{C}_{\text{app}}[\rho]$ is strictly convex on $L^2([0, 1])$, guaranteeing that any stationary point is the unique global minimizer. $\blacksquare$

### 5.3 Euler-Lagrange Derivation of the Cosh ODE

Define the Green potential function:
$$g(\phi) := \int_0^1 \rho(\psi) \min(\phi, \psi) d\psi = \int_0^\phi \psi \rho(\psi) d\psi + \phi \int_\phi^1 \rho(\psi) d\psi.$$
Differentiating $g(\phi)$:
$$g'(\phi) = \phi \rho(\phi) + \int_\phi^1 \rho(\psi) d\psi - \phi \rho(\phi) = \int_\phi^1 \rho(\psi) d\psi,$$
$$g''(\phi) = -\rho(\phi).$$

The Lagrangian for constrained minimization is:
$$\mathcal{L}[\rho, \nu] = \frac{\alpha}{2}\int_0^1 \rho^2 d\phi + \frac{\beta}{2}\int_0^1 \rho(\phi) g(\phi) d\phi + \nu \left(1 - \int_0^1 \rho d\phi\right).$$
Setting the first variation to zero:
$$\alpha \rho(\phi) + \beta g(\phi) - \nu = 0.$$
1. Differentiating once with respect to $\phi$:
   $$\alpha \rho'(\phi) + \beta g'(\phi) = 0 \implies \alpha \rho'(\phi) + \beta \int_\phi^1 \rho(\psi) d\psi = 0.$$
2. Differentiating a second time:
   $$\alpha \rho''(\phi) - \beta \rho(\phi) = 0 \implies \rho''(\phi) - \tau^2 \rho(\phi) = 0, \qquad \tau = \sqrt{\frac{\beta}{\alpha}}.$$
3. Boundary Conditions:
   - At $\phi = 1$: $g'(1) = \int_1^1 \rho = 0 \implies \rho'(1) = 0$.
   - At $\phi = 0$: $g'(0) = \int_0^1 \rho = 1 \implies \alpha \rho'(0) + \beta(1) = 0 \implies \rho'(0) = -\frac{\beta}{\alpha} = -\tau^2$.
   - Normalization: $\int_0^1 \rho(\phi) d\phi = 1$.

### 5.4 Exact Closed-Form Quantile Solution

The general solution of $\rho'' - \tau^2 \rho = 0$ with $\rho'(1) = 0$ is:
$$\rho(\phi) = C \cosh(\tau(1 - \phi)).$$
Applying the normalization condition $\int_0^1 \rho(\phi) d\phi = 1$:
$$\int_0^1 C \cosh(\tau(1 - \phi)) d\phi = \left[ -\frac{C}{\tau} \sinh(\tau(1 - \phi)) \right]_0^1 = \frac{C}{\tau} \sinh \tau = 1 \implies C = \frac{\tau}{\sinh \tau}.$$
Thus, the unique stationary density is:
$$\rho_\tau(\phi) = \frac{\tau \cosh(\tau(1 - \phi))}{\sinh \tau}.$$

The cumulative distribution function $F_\tau(\phi)$ is:
$$F_\tau(\phi) = \int_0^\phi \rho_\tau(u) du = 1 - \frac{\sinh(\tau(1 - \phi))}{\sinh \tau}.$$
Inverting $F_\tau(\phi) = u \in [0, 1]$:
$$1 - u = \frac{\sinh(\tau(1 - \phi))}{\sinh \tau} \implies \sinh(\tau(1 - \phi)) = (1 - u)\sinh \tau,$$
$$\tau(1 - \phi) = \operatorname{arcsinh}((1 - u)\sinh \tau) \implies \phi(u) = 1 - \frac{1}{\tau}\operatorname{arcsinh}((1 - u)\sinh \tau).$$
For discrete channels $k \in \{0, 1, \dots, K-1\}$, evaluating at midpoints $u_k = \frac{k + 1/2}{K}$ yields the EVQ-Cosh frequency table:
$$\omega_k = b^{-\phi(u_k)} = b^{-\left[ 1 - \frac{1}{\tau}\operatorname{arcsinh}\left(\left(1 - \frac{k+1/2}{K}\right)\sinh \tau\right) \right]}.$$

### 5.5 Precise Claim Ceilings and Uniqueness Boundaries

In compliance with repository rule `AGENTS.md` (§2 Claim Ceilings):
1. **Uniqueness Scope:** EVQ-Cosh is unique **ONLY** for the stated quadratic surrogate $\mathcal{C}_{\text{app}}[\rho]$. It is a closed-form, zero-learned-parameter construction and controlled intervention.
2. **Not a Full-RoPE or LM Optimum:** Real full-RoPE Grassmannian geometry involves pairwise canonical correlations $c_{\omega\nu} = \frac{1}{2}\|S_\omega^{-1/2}H_{\omega\nu}S_\nu^{-1/2}\|_F^2$. Real transformer training involves content-dependent Q/K weights, attention softmax entropy, and downstream cross-entropy loss. EVQ-Cosh is **not** the theoretical or empirical global upper bound for language model performance.
3. **Finite $\tau$ Operating Prior:** The multiplier $\tau = \max(d_{\text{head}}/\sqrt{L}, 1.4)$ is an empirical operating prior, not a universal physical constant or continuous basin bound.

---

## 6. Post-Mortem on Falsified Routes

### 6.1 Falsification of the Arcsine Conjecture (O5)

#### Origin of the Conjecture
In classical potential theory, minimizing the logarithmic Riesz energy $\iint \ln\frac{1}{|x-y|} \rho(x)\rho(y) dx dy$ on a compact interval $[-1, 1]$ yields the equilibrium Chebyshev/arcsine distribution:
$$\rho_{\text{arcsine}}(x) = \frac{1}{\pi \sqrt{1 - x^2}},$$
which diverges at both boundaries (U-shaped profile). It was hypothesized in early drafts that heavy-tailed attention distance priors would drive optimal RoPE allocations into an arcsine-like U-shaped distribution (piling channels at both the extreme fast and extreme slow ends).

#### Why It Fails Mathematically
1. **Non-Green Leading Kernel:** In $\phi$-space, the full-RoPE canonical collision kernel $c_{\omega\nu}$ expanded for separation priors has a leading-order term proportional to $\min(\phi, \psi)^2$.
2. **Differential Operator Breakdown:** The operator $\min(\phi, \psi)^2$ is **not** the Green's function for $\partial_\phi^4$. Specifically, applying $\partial_\phi^4$ yields:
   $$\partial_\phi^4 \left[ \int_0^1 \min(\phi, \psi)^2 \rho(\psi) d\psi \right] = -6\rho'(\phi) - 2\phi \rho''(\phi) \neq C \rho(\phi).$$
   The operator does not invert to a simple local differential equation, breaking the equilibrium measure equivalence.
3. **Numerical Refutation (optimization_notes.md O5):**
   Direct numerical minimization of the full-RoPE collision $\bar{c}$ on a 16-bin piecewise-constant density under an identical $\chi^2$-stiffness constraint ($S_{\chi^2}[\rho] \le 0.1803$) produced the best-found profile:
   $$\vec{\rho} = [3.01, 1.57, 1.14, 1.17, 1.07, 0.92, 0.82, 0.80, 0.80, 0.77, 0.71, 0.67, 0.65, 0.62, 0.65, 0.63].$$
   This shape is **strictly monotonically decreasing** with a fast-end spike and a flat slow-end tail. It exhibits zero slow-end boundary accumulation ($\rho(0) = 3.01 \gg \rho(1) = 0.63$). The Arcsine conjecture is definitively falsified.

---

### 6.2 Falsification of Naive Collision / Logdet Minimization: The Fourier Comb Collapse

#### Origin of the Failure
Early attempts sought to maximize the static effective rank $r_2(\Gamma) = \frac{2K}{1 + (K-1)\bar{c}}$ or log-determinant $\ln\det(\Gamma)$ by setting frequencies freely over a fixed training window $[0, L]$.

#### The Fourier Comb Collapse Theorem
**Theorem 3 (Harmonic Lattice Orthogonality & Exact Aliasing).**
*Let $\Delta \sim \mathrm{Unif}[0, L]$. The pairwise cross-Gram blocks $H_{\omega_j, \omega_k} = \mathbb{E}[\mathbf{x}_{\omega_j}^\top \mathbf{x}_{\omega_k}]$ vanish identically ($c_{jk} = 0$) for all $j \neq k$, achieving maximum static rank $r_2 = 2K$, if and only if the frequencies form a harmonic Fourier lattice:*
$$\omega_k = \frac{2\pi a_k}{L}, \qquad a_k \in \mathbb{Z}^+.$$
*However, any such harmonic frequency table satisfies the exact periodic translation identity:*
$$\mathcal{R}_\Omega(\Delta + L) = \mathcal{R}_\Omega(\Delta) \quad \forall \Delta \in \mathbb{R}.$$

*Proof.*
For any two frequencies $\omega_j, \omega_k$, the cross-Gram terms are linear combinations of $\frac{\sin((\omega_j \pm \omega_k)L)}{(\omega_j \pm \omega_k)L}$.
For these terms to vanish identically for all pairs, we must have $(\omega_j \pm \omega_k)L \in 2\pi \mathbb{Z}$, which forces $\omega_k = \frac{2\pi a_k}{L}$ with integer $a_k$.
Evaluating the rotation block at lag $\Delta + L$:
$$\cos(\omega_k(\Delta + L)) = \cos\left(\omega_k \Delta + \frac{2\pi a_k}{L} L\right) = \cos(\omega_k \Delta + 2\pi a_k) = \cos(\omega_k \Delta),$$
$$\sin(\omega_k(\Delta + L)) = \sin\left(\omega_k \Delta + \frac{2\pi a_k}{L} L\right) = \sin(\omega_k \Delta + 2\pi a_k) = \sin(\omega_k \Delta).$$
Therefore, $\mathcal{R}_\Omega(\Delta + L) = \mathcal{R}_\Omega(\Delta)$. $\blacksquare$

```
Fourier Comb Exact Aliasing:
  Lag Delta:      0 ---- L/2 ---- L ---- 3L/2 ---- 2L
  Pos Embed R:    R_0 -- R_mid -- R_0 -- R_mid --- R_0
                  [  Period 1  ]  [  Period 2 (Identical!) ]
```

#### Catastrophic Extrapolation Consequences:
1. **Total Distinguishability Collapse:** Because $\mathcal{R}(\Delta + L) = \mathcal{R}(\Delta)$, the model is fundamentally incapable of distinguishing token lag $\Delta$ from $\Delta + L$, $\Delta + 2L$, etc.
2. **Failure of Static Rank as an Extrapolation Predictor:** The Fourier comb achieves the absolute theoretical ceiling of static effective rank ($r_2 = 2K$), yet exhibits catastrophic failure under sequence length extrapolation.
3. **Scientific Value:** Multi-scale geometric and warped RoPE tables deliberately employ non-harmonic, incommensurate frequencies to prevent global periodicity, accepting small in-window pairwise correlations to preserve out-of-distribution phase distinguishability.

---

## 7. Synthesis and Cross-Validation with Repository Evidence

The theoretical conclusions derived above are strictly corroborated by canonical repository empirical evidence:

1. **Exact-Range Causal Identification (`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`):**
   - Fixed sampled endpoints and log-span ($a, R$), altering only the 30 interior frequencies via anchored EVQ-Cosh.
   - Result across 3 paired training seeds: $+0.026$ NLL in-window ($L=256$), and $-0.281 / -0.176 / -0.146$ NLL out-of-distribution ($L = 512 / 1024 / 2048$).
   - Proves empirically that interior allocation $z$ is a causally active design coordinate independent of scalar base or support dilation.
2. **Table-Weight Co-Adaptation (`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`):**
   - In the 50M $2 \times 2$ table $\times$ weight crossing:
     - Matched systems: $\text{Geo}_{\text{w}} \times \text{Geo}_{\text{tab}} = 7.14$ PPL; $\text{EVQ}_{\text{w}} \times \text{EVQ}_{\text{tab}} = 7.16$ PPL.
     - Post-hoc table swap: $\text{Geo}_{\text{w}} \times \text{EVQ}_{\text{tab}} = 76.20$ PPL.
   - Direct evidence that static geometric rank does not dictate trained loss; trained Q/K projection weights strongly co-adapt to the specific frequency basis present during training.
3. **Exact Post-Hoc Transplant Obstruction (`OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`):**
   - Proves Theorem \ref{thm:obstruction}: exact function equivalence between two distinct frequency multisets $\Omega \neq \Omega'$ cannot be achieved via position-independent linear Q/K transformations $A^\top \mathcal{R}_{\Omega'}(\Delta) B = \mathcal{R}_\Omega(\Delta)$.

---

## 8. Summary of Mathematical Formulas

| Mathematical Concept | Exact Formula |
| :--- | :--- |
| **Linear Scaling / Base Isomorphism** | $\omega_b(cz) = (b^c)^z = \omega_{b^c}(z)$ |
| **Continuous Spectral Density** | $\rho_f(\omega) = \frac{1}{\omega \ln b \cdot \|f'(f^{-1}(-\log_b \omega))\|}$ |
| **EVQ-Cosh Density** | $\rho_\tau(\phi) = \frac{\tau \cosh(\tau(1 - \phi))}{\sinh \tau}, \quad \phi \in [0, 1]$ |
| **EVQ-Cosh Quantiles** | $\phi_k(\tau) = 1 - \frac{1}{\tau}\operatorname{arcsinh}\left(\left(1 - \frac{k+1/2}{K}\right)\sinh \tau\right)$ |
| **Single-Crossing Budget Shift** | $\phi_c(\tau) = 1 - \frac{1}{\tau}\operatorname{arcosh}\left(\frac{\sinh \tau}{\tau}\right) \le 1 - \frac{1}{\sqrt{3}}$ |
| **Phase Velocity** | $v_p(k) = \frac{\omega(k)}{k}$ |
| **Group Velocity** | $v_g(k) = \frac{d\omega}{dk} = -\frac{\ln b}{K} f'(k/K) \omega(k)$ |
| **Coherence Length** | $L_{\text{coh}}(k) \sim \frac{2\pi}{\|v_g(k)\| \Delta k}$ |
| **Rényi-2 Spectral Budget Identity** | $r_2(\Gamma) = \frac{2K}{1 + (K-1)\bar{c}}$ |

