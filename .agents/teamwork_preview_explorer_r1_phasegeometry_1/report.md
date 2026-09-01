# Phase Code $\Phi(\Delta)$ Geometry, Gram Matrix & Extrapolation Breakdown (R1 Axis B)

- **Agent:** `explorer_r1_phasegeometry_1`
- **Date:** 2026-09-01
- **Focus:** R1 Axis B — Joint Phase Code $\Phi(\Delta)$ Geometry, Gram Matrix Properties, Stable Rank Identity, and Dual-End Extrapolation Breakdown
- **Status:** Complete Analytical Investigation & Synthesis

---

## 1. Executive Summary

This report establishes the rigorous mathematical foundation for **Rotary Position Embedding (RoPE) as a finite spectral basis**, characterizing its joint phase code geometry on the $K$-dimensional torus $\mathbb{T}^K = (S^1)^K$, the structure of its shift-invariant Gram kernel $G(\delta)$ and phase-space Euclidean metric $D^2(\delta)$, the exact block-whitened stable rank identity $r_2(R) = \frac{2K}{1 + (K-1)\bar{c}}$, and the dual-end geometric breakdown of position distinguishability when extrapolating to $\Delta > L_{\text{train}}$.

### Key Theoretical Findings
1. **Joint Phase Code as a Torus Embedding:**
   The positional mapping $\Phi(\Delta) = [\cos(\omega_0 \Delta), \sin(\omega_0 \Delta), \dots, \cos(\omega_{K-1} \Delta), \sin(\omega_{K-1} \Delta)]^\top \in \mathbb{R}^{2K}$ (with $K = d/2$) embeds 1D relative displacement $\Delta \in \mathbb{R}$ into a $K$-dimensional flat torus $\mathbb{T}^K \subset \mathbb{R}^{2K}$. Its trajectory $\boldsymbol{\theta}(\Delta) = (\omega_0 \Delta \pmod{2\pi}, \dots, \omega_{K-1} \Delta \pmod{2\pi})$ constitutes a multi-frequency linear winding flow.
2. **Stationary Shift-Invariant Gram Kernel & Metric:**
   The pairwise inner product between embedded positions is strictly shift-invariant:
   $$G(\Delta, \Delta') = \langle \Phi(\Delta), \Phi(\Delta') \rangle = \sum_{j=0}^{K-1} \cos(\omega_j(\Delta - \Delta')) = G(\delta), \quad \text{where } \delta = \Delta - \Delta'.$$
   The associated Euclidean distance in phase space is:
   $$D^2(\delta) = \|\Phi(\Delta) - \Phi(\Delta')\|_2^2 = 2K - 2\sum_{j=0}^{K-1} \cos(\omega_j \delta) = 4\sum_{j=0}^{K-1} \sin^2\left(\frac{\omega_j \delta}{2}\right).$$
3. **Exact Stable Rank Identity on Block-Whitened Correlation Matrix:**
   Under any relative distance prior measure $p(\Delta)$, the full 2D subspace Gram matrix $R \in \mathbb{R}^{2K \times 2K}$, with $K$ diagonal blocks $I_2$ and off-diagonal blocks given by the whitened cross-Gram $Q_{jk} = S_j^{-1/2} H_{jk} S_k^{-1/2}$, satisfies the exact algebraic identity:
   $$r_2(R) = \frac{(\operatorname{tr}R)^2}{\operatorname{tr}(R^2)} = \frac{2K}{1 + (K-1)\bar{c}},$$
   where $\bar{c} = \frac{1}{K(K-1)}\sum_{j \neq k} c_{jk}$ is the mean pairwise canonical collision metric $c_{jk} = \frac{1}{2}\|Q_{jk}\|_F^2 = \frac{\sigma_1^2 + \sigma_2^2}{2} \in [0, 1]$.
4. **Dual-End Geometry of Extrapolation Breakdown ($\Delta > L_{\text{train}}$):**
   - **High-Frequency Phase Aliasing:** For fast bands ($\omega_j \gg 2\pi / L_{\text{train}}$), single channels wrap around $S^1$ $N_j = L_{\text{train}} / \lambda_j \gg 1$ times. Beyond $L_{\text{train}}$, fast frequencies produce pseudo-random phase hash collisions and high-frequency noise variance $\operatorname{Var}[G_{\text{fast}}(\delta)] \sim K_{\text{fast}}/2$.
   - **Low-Frequency Spectral Collapse & Non-Linear Turnover:** For slow bands ($\omega_j L_{\text{train}} \ll 1$), in-distribution Taylor expansion shows that unwhitened subspaces collapse to $V_{\omega_j} \to \operatorname{span}\{1, \Delta\}$ in $L_2$, and centered attention softmax quotient geometry ($F_{\text{sm}}\mathbf{1} = 0$) collapses to $\operatorname{span}\{\Delta - \mathbb{E}_p\Delta, \Delta^2 - \mathbb{E}_p\Delta^2\}$. When $\Delta > L_{\text{train}}$, slow bands exit their trained linear regime into untrained sinusoidal turnover ($\omega_j \Delta \sim \pi/2, \pi$), inverting the macroscopic distance coordinate.
5. **Destruction of Distance Distinguishability:**
   The multi-scale interference that yields smooth monotonic distance decay in-window is destroyed out-of-window, resulting in spurious quasi-periodic revivals, loss of metric injectivity, and collapse of the distance signal-to-noise ratio $\operatorname{SNR}_{\text{dist}}(\delta)$.
6. **Strict Claim Ceiling Adherence:**
   Static basis geometry ($r_2(R)$, collision $c_{jk}$, logdet) measures positional basis redundancy, **not** LM quality or extrapolation performance. Collision minimization does not imply extrapolation (e.g., Fourier harmonic combs achieve $r_2 = 2K$ on $[0, L]$ but suffer $100\%$ periodic aliasing $\Phi(\Delta + L) = \Phi(\Delta)$).

---

## 2. Mathematical Definition & Torus Geometry of Joint Phase Code $\Phi(\Delta)$

### 2.1 Formal Definition of Joint Phase Code
In Rotary Position Embedding (RoPE), hidden query and key vectors $q, k \in \mathbb{R}^d$ are partitioned into $K = d/2$ orthogonal 2D subspaces. For a relative position displacement $\Delta \in \mathbb{R}$, each 2D subspace $j \in \{0, 1, \dots, K-1\}$ is modulated by a rotation matrix $\mathbf{R}_{\omega_j}(\Delta) \in \mathrm{SO}(2)$:
$$\mathbf{R}_{\omega_j}(\Delta) = \begin{bmatrix} \cos(\omega_j \Delta) & -\sin(\omega_j \Delta) \\ \sin(\omega_j \Delta) & \cos(\omega_j \Delta) \end{bmatrix}.$$

The fundamental geometric carrier of relative position information across all $K$ channels is the **joint phase code vector** $\Phi(\Delta) \in \mathbb{R}^{2K}$:
$$\Phi(\Delta) \triangleq \begin{bmatrix} \cos(\omega_0 \Delta) \\ \sin(\omega_0 \Delta) \\ \cos(\omega_1 \Delta) \\ \sin(\omega_1 \Delta) \\ \vdots \\ \cos(\omega_{K-1} \Delta) \\ \sin(\omega_{K-1} \Delta) \end{bmatrix} = \begin{bmatrix} x_0(\Delta) \\ x_1(\Delta) \\ \vdots \\ x_{K-1}(\Delta) \end{bmatrix} \in \mathbb{R}^{2K},$$
where each 2D sub-vector is defined as $x_j(\Delta) \triangleq [\cos(\omega_j \Delta), \sin(\omega_j \Delta)]^\top \in \mathbb{R}^2$.

### 2.2 Torus Embedding Geometry
Because $\|x_j(\Delta)\|_2^2 = \cos^2(\omega_j \Delta) + \sin^2(\omega_j \Delta) = 1$ for all $j \in \{0, \dots, K-1\}$ and all $\Delta \in \mathbb{R}$, each sub-vector $x_j(\Delta)$ traces the unit circle $S^1 \subset \mathbb{R}^2$ with constant angular frequency $\omega_j$.

Consequently, the mapping $\Phi: \mathbb{R} \to \mathbb{R}^{2K}$ embeds the 1D real line of relative displacements $\Delta$ into a **$K$-dimensional flat torus**:
$$\mathbb{T}^K \triangleq \underbrace{S^1 \times S^1 \times \dots \times S^1}_{K \text{ times}} \subset \mathbb{R}^{2K}.$$

The total Euclidean norm of the phase code vector is strictly constant:
$$\|\Phi(\Delta)\|_2 = \sqrt{\sum_{j=0}^{K-1} \|x_j(\Delta)\|_2^2} = \sqrt{K}, \quad \forall \Delta \in \mathbb{R}.$$
Thus, $\Phi(\Delta)$ lies on the intersection of the flat torus $\mathbb{T}^K$ and the sphere $\mathbb{S}^{2K-1}(\sqrt{K})$.

### 2.3 Phase Coordinates and Linear Winding Flow
Let $\boldsymbol{\theta}(\Delta) = (\theta_0(\Delta), \dots, \theta_{K-1}(\Delta)) \in [0, 2\pi)^K$ denote the angular phase coordinates on $\mathbb{T}^K$:
$$\theta_j(\Delta) \equiv \omega_j \Delta \pmod{2\pi}, \quad j \in \{0, \dots, K-1\}.$$

The trajectory $\Delta \mapsto \boldsymbol{\theta}(\Delta)$ defines a **constant-velocity linear flow** on the torus $\mathbb{T}^K$:
$$\frac{\mathrm{d}\boldsymbol{\theta}}{\mathrm{d}\Delta} = \boldsymbol{\omega} = [\omega_0, \omega_1, \dots, \omega_{K-1}]^\top \in \mathbb{R}^K.$$

If the frequency ratios $\{\omega_j / \omega_k\}_{j \neq k}$ are rationally incommensurate (which is true for standard geometric RoPE bases with irrational or algebraic exponents), the trajectory $\boldsymbol{\theta}(\Delta)$ forms a dense, non-periodic Kronecker-Weyl winding on $\mathbb{T}^K$.

### 2.4 Standard Geometric Spectrum Allocation
In standard RoPE architectures (e.g. LLaMA, Mistral, OLMo), the frequency multiset $\Omega = \{\omega_0, \dots, \omega_{K-1}\}$ is defined by a geometric sequence with scalar base $b > 1$ (typically $b = 10\,000$ or $b = 500\,000$):
$$\omega_j = b^{-2j/d} = b^{-j/K}, \quad j \in \{0, 1, \dots, K-1\}.$$

In logarithmic coordinates, the negative log-frequencies $x_j \triangleq -\log \omega_j$ are uniformly spaced:
$$x_j = \frac{j}{K} \log b = a + R z_j, \quad \text{with } a = 0, \; R = \log b, \; z_j = \frac{j}{K} \in [0, 1).$$
Here, $(a, R)$ defines the **sampled spectral support** and $z \in [0, 1]^K$ is the **normalized interior allocation**.

---

## 3. Gram Matrix Properties, Euclidean Metric & Stable Rank Identity

### 3.1 Pointwise Shift-Invariant Gram Kernel $G(\Delta, \Delta')$
The fundamental quantity governing positional overlap and attention logit formation is the inner product between two phase vectors at positions $\Delta$ and $\Delta'$:

$$\begin{aligned}
G(\Delta, \Delta') &\triangleq \langle \Phi(\Delta), \Phi(\Delta') \rangle = \sum_{j=0}^{K-1} x_j(\Delta)^\top x_j(\Delta') \\
&= \sum_{j=0}^{K-1} \Big( \cos(\omega_j \Delta)\cos(\omega_j \Delta') + \sin(\omega_j \Delta)\sin(\omega_j \Delta') \Big) \\
&= \sum_{j=0}^{K-1} \cos(\omega_j(\Delta - \Delta')).
\end{aligned}$$

**Theorem 3.1 (Shift-Invariance of Pointwise Phase Gram Kernel):**
*The pointwise phase Gram kernel $G(\Delta, \Delta')$ is strictly stationary and depends solely on the relative displacement $\delta = \Delta - \Delta'$:*
$$G(\Delta, \Delta') = G(\Delta - \Delta') = G(\delta) \triangleq \sum_{j=0}^{K-1} \cos(\omega_j \delta).$$

#### Properties of $G(\delta)$:
1. **Maximum at origin:** $G(0) = \sum_{j=0}^{K-1} 1 = K$.
2. **Symmetry:** $G(-\delta) = G(\delta)$.
3. **Bounded range:** $-K \le G(\delta) \le K$ for all $\delta \in \mathbb{R}$.
4. **Spectral interpretation:** $G(\delta)$ is the inverse Fourier transform of the discrete frequency measure $\mu = \sum_{j=0}^{K-1} \frac{1}{2}(\delta_{\omega_j} + \delta_{-\omega_j})$.

---

### 3.2 Phase-Space Euclidean Distance Metric $D^2(\Delta, \Delta')$
The Euclidean distance between two phase code vectors in $\mathbb{R}^{2K}$ directly reflects their distinguishability:

$$\begin{aligned}
D^2(\Delta, \Delta') &\triangleq \|\Phi(\Delta) - \Phi(\Delta')\|_2^2 \\
&= \|\Phi(\Delta)\|_2^2 + \|\Phi(\Delta')\|_2^2 - 2 \langle \Phi(\Delta), \Phi(\Delta') \rangle \\
&= K + K - 2 G(\Delta - \Delta') \\
&= 2K - 2\sum_{j=0}^{K-1} \cos(\omega_j(\Delta - \Delta')).
\end{aligned}$$

Using the trigonometric identity $1 - \cos(\theta) = 2\sin^2(\theta/2)$:
$$D^2(\delta) = 4 \sum_{j=0}^{K-1} \sin^2\left(\frac{\omega_j \delta}{2}\right).$$

#### Local Curvature and Asymptotic Bounds:
- **Near-field / Local Taylor Expansion ($\delta \to 0$):**
  $$\sin^2\left(\frac{\omega_j \delta}{2}\right) = \frac{\omega_j^2 \delta^2}{4} - \frac{\omega_j^4 \delta^4}{48} + O(\omega_j^6 \delta^6),$$
  $$D^2(\delta) = \left( \sum_{j=0}^{K-1} \omega_j^2 \right) \delta^2 - \frac{1}{12}\left( \sum_{j=0}^{K-1} \omega_j^4 \right) \delta^4 + O(\delta^6) = \Omega_{\text{tot}}^2 \delta^2 - \frac{1}{12}\Omega_4 \delta^4 + O(\delta^6),$$
  where $\Omega_{\text{tot}}^2 \triangleq \sum_{j=0}^{K-1} \omega_j^2$ is the total spectral stiffness (dominated by the highest frequency bands $\omega_0, \omega_1$).
- **Global Bounds:**
  $$0 \le D^2(\delta) \le 4K, \quad \forall \delta \in \mathbb{R}.$$
  The maximum distance $D^2 = 4K$ is achieved when all channels are in exact anti-phase ($\omega_j \delta \equiv \pi \pmod{2\pi}$ for all $j$).

---

### 3.3 Full-Subspace Gram Matrix & Canonical Correlations
In full-RoPE geometry, each frequency band $\omega_j$ contributes a full 2D subspace:
$$V_{\omega_j} = \operatorname{span}\{\cos(\omega_j \Delta), \sin(\omega_j \Delta)\} \subset L_2(p),$$
where $p(\Delta)$ is a declared relative-distance prior measure on $[0, L]$ (e.g. uniform $p(\Delta) = 1/L$ or causal $p(\Delta) = 2(L-\Delta)/L^2$).

Let $x_j(\Delta) = [\cos(\omega_j \Delta), \sin(\omega_j \Delta)]$. For any two bands $j, k \in \{0, \dots, K-1\}$, define:
- **Self-Gram Matrix:** $S_j \triangleq \mathbb{E}_{\Delta \sim p}[x_j(\Delta)^\top x_j(\Delta)] \in \mathbb{R}^{2\times 2}$.
- **Cross-Gram Matrix:** $H_{jk} \triangleq \mathbb{E}_{\Delta \sim p}[x_j(\Delta)^\top x_k(\Delta)] \in \mathbb{R}^{2\times 2}$.

#### Analytical Closed Form for Uniform Prior $p(\Delta) = \mathcal{U}[0, L]$:
Let $d = (\omega_j - \omega_k)L$, $s = (\omega_j + \omega_k)L$, and define the basic integrals:
$$a(t) \triangleq \operatorname{sinc}(t) = \frac{\sin t}{t}, \qquad b(t) \triangleq j(t) = \frac{1 - \cos t}{t}.$$

Then the exact 2x2 cross-Gram block is:
$$H_{jk} = \frac{1}{2} \begin{bmatrix} a(d) + a(s) & b(s) - b(d) \\ b(s) + b(d) & a(d) - a(s) \end{bmatrix}.$$

For the self-Gram ($j = k \implies d = 0, a(0) = 1, b(0) = 0, s = 2\omega_j L$):
$$S_j = \frac{1}{2} \begin{bmatrix} 1 + a(2\omega_j L) & b(2\omega_j L) \\ b(2\omega_j L) & 1 - a(2\omega_j L) \end{bmatrix}.$$

#### Whitened Cross-Gram and Canonical Collision Metric:
To measure the true geometric overlap between the subspaces $V_{\omega_j}$ and $V_{\omega_k}$ independent of coordinate rotations and scaling, we apply **block whitening**:
$$Q_{jk} \triangleq S_j^{-1/2} H_{jk} S_k^{-1/2} \in \mathbb{R}^{2\times 2}.$$

The singular values $\sigma_1(Q_{jk}), \sigma_2(Q_{jk}) \in [0, 1]$ are the **canonical correlations** (principal cosines) between subspaces $V_{\omega_j}$ and $V_{\omega_k}$.

**Definition 3.2 (Canonical Pairwise Collision Metric):**
$$c_{jk} \triangleq \frac{1}{2} \|Q_{jk}\|_F^2 = \frac{\sigma_1^2(Q_{jk}) + \sigma_2^2(Q_{jk})}{2} \in [0, 1].$$

*Invariance Property:* $c_{jk}$ is invariant under arbitrary intra-pair phase rotations (coordinate frame shifts $x_j \mapsto \mathbf{R}_{\phi_j} x_j$) and any invertible linear transformation within each 2D channel subspace.

---

### 3.4 Global Block-Whitened Correlation Matrix & Exact Stable Rank Identity

Assemble the $K \times K$ blocks of 2x2 matrices into the global block-whitened correlation Gram matrix $R \in \mathbb{R}^{2K \times 2K}$:
$$R \triangleq \begin{bmatrix} 
I_2 & Q_{0,1} & \dots & Q_{0,K-1} \\
Q_{1,0} & I_2 & \dots & Q_{1,K-1} \\
\vdots & \vdots & \ddots & \vdots \\
Q_{K-1,0} & Q_{K-1,1} & \dots & I_2
\end{bmatrix}.$$

**Theorem 3.3 (Exact Block-Whitened Stable Rank Identity):**
*Let $R \in \mathbb{R}^{2K \times 2K}$ be the global block-whitened correlation matrix for $K$ frequency channels. Then the stable rank (Rényi-2 effective rank) of $R$ satisfies the exact identity:*
$$r_2(R) \triangleq \frac{(\operatorname{tr}R)^2}{\operatorname{tr}(R^2)} = \frac{2K}{1 + (K-1)\bar{c}},$$
*where $\bar{c} \triangleq \frac{1}{K(K-1)} \sum_{j \neq k} c_{jk}$ is the mean off-diagonal canonical collision.*

#### Proof:
1. **Trace Calculation:**
   Since each of the $K$ diagonal blocks is $I_2$, we have:
   $$\operatorname{tr}(R) = \sum_{j=0}^{K-1} \operatorname{tr}(I_2) = \sum_{j=0}^{K-1} 2 = 2K.$$
2. **Frobenius Norm / Square-Trace Calculation:**
   The trace of $R^2$ is the sum of squared entries of $R$ (since $R$ is symmetric):
   $$\operatorname{tr}(R^2) = \|R\|_F^2 = \sum_{j=0}^{K-1} \|I_2\|_F^2 + \sum_{j \neq k} \|Q_{jk}\|_F^2.$$
   For each diagonal block, $\|I_2\|_F^2 = 1^2 + 0^2 + 0^2 + 1^2 = 2$.
   For off-diagonal blocks, by Definition 3.2, $\|Q_{jk}\|_F^2 = 2 c_{jk}$.
   Summing all entries:
   $$\operatorname{tr}(R^2) = 2K + \sum_{j \neq k} 2 c_{jk} = 2K + 2 \sum_{j \neq k} c_{jk}.$$
   Substituting $\sum_{j \neq k} c_{jk} = K(K-1)\bar{c}$:
   $$\operatorname{tr}(R^2) = 2K + 2K(K-1)\bar{c} = 2K\bigl[1 + (K-1)\bar{c}\bigr].$$
3. **Stable Rank Ratio:**
   $$r_2(R) = \frac{(\operatorname{tr}R)^2}{\operatorname{tr}(R^2)} = \frac{(2K)^2}{2K[1 + (K-1)\bar{c}]} = \frac{2K}{1 + (K-1)\bar{c}}.$$
   $\blacksquare$

#### Boundary Cases:
- **Case 1: Mutually Orthogonal Subspaces ($\bar{c} = 0$):**
  $$r_2(R) = 2K.$$
  The basis achieves maximal algebraic dimensionality (full rank $2K$).
- **Case 2: Complete Subspace Redundancy ($\bar{c} = 1$):**
  $$r_2(R) = \frac{2K}{1 + K - 1} = 2.$$
  All $K$ channels span the exact same 2D subspace, collapsing the effective dimension from $2K$ down to $2$.

---

## 4. Geometry of Extrapolation Breakdown at $\Delta > L_{\text{train}}$

When a Transformer trained on context length $L_{\text{train}}$ is evaluated at relative displacements $\Delta > L_{\text{train}}$, the positional representation suffers a **dual-end geometric catastrophe**: high frequencies undergo phase aliasing on $\mathbb{T}^K$, while low frequencies exit their trained linear regime into sinusoidal turnover.

```
       HIGH FREQUENCIES (Fast Bands)             LOW FREQUENCIES (Slow Bands)
       omega_j >> 2*pi / L_train                 omega_j * L_train << 1
  ┌────────────────────────────────────────┐ ┌────────────────────────────────────────┐
  │ In-distribution:                       │ │ In-distribution:                       │
  │ • N_j >> 1 windings around S^1         │ │ • omega_j * Delta << 1                 │
  │ • Periodic: lambda_j = 2*pi/omega_j    │ │ • Taylor approx: cos ~ 1, sin ~ omega*D│
  │ • Positional uniqueness requires joint │ │ • V_w collapses to span{1, Delta}      │
  │   multi-scale non-rational winding.    │ │ • Softmax collapses to centered poly   │
  │                                        │ │   span{Delta - E[Delta], Delta^2 - ...}│
  ├────────────────────────────────────────┤ ├────────────────────────────────────────┤
  │ Out-of-distribution (Delta > L_train): │ │ Out-of-distribution (Delta > L_train): │
  │ • Rapid wrap-around on torus T^K       │ │ • omega_j * Delta ~ pi/2, pi           │
  │ • Lands on untrained phase coordinates │ │ • Exits linear Taylor regime           │
  │ • Act as pseudo-random phase hashes    │ │ • Sinusoidal turnover: sin peaks & drops│
  │ • Destructive interference / noise.    │ │ • Monotonic distance metric inverts!   │
  └────────────────────────────────────────┘ └────────────────────────────────────────┘
```

---

### 4.1 High-Frequency Phase Aliasing & Torus Packing Distortion

#### 4.1.1 Fast Band Characterization
For high-frequency channels where $\omega_j \gg 2\pi / L_{\text{train}}$, the spatial wavelength $\lambda_j = \frac{2\pi}{\omega_j}$ is much shorter than the training context:
$$\lambda_j \ll L_{\text{train}}.$$

During training ($\Delta \in [0, L_{\text{train}}]$), the phase angle $\theta_j(\Delta) = \omega_j \Delta \pmod{2\pi}$ wraps around the unit circle $S^1$ a large number of complete turns:
$$N_j = \left\lfloor \frac{L_{\text{train}}}{\lambda_j} \right\rfloor = \left\lfloor \frac{\omega_j L_{\text{train}}}{2\pi} \right\rfloor \gg 1.$$

#### 4.1.2 Single-Channel Periodic Ambiguity
For any single fast channel $j$, the 2D basis vector is strictly periodic with period $\lambda_j$:
$$x_j(\Delta + m \lambda_j) = x_j(\Delta), \quad \forall m \in \mathbb{Z}.$$
A single fast channel cannot distinguish position $\Delta$ from $\Delta + \lambda_j, \Delta + 2\lambda_j, \dots$.

#### 4.1.3 Torus Trajectory in Extrapolation Regime
In-distribution uniqueness was guaranteed solely by the joint embedding across all $K$ incommensurate frequencies (the multi-scale Chinese Remainder Theorem on $\mathbb{T}^K$).

However, when extrapolating to $\Delta > L_{\text{train}}$:
1. **Untrained Torus Coordinates:** The joint phase vector $\boldsymbol{\theta}(\Delta) = (\omega_0 \Delta \pmod{2\pi}, \dots, \omega_{K-1} \Delta \pmod{2\pi})$ travels into combinations of phases that were never visited during training.
2. **Pseudo-Random Phase Hashing:** Because the fast frequencies oscillate extremely rapidly relative to the macroscopic context scale, their pairwise inner products $\cos(\omega_j \delta)$ act as pseudo-random hash functions. For displacements $\delta > \lambda_{\text{fast}}$, the expectation is $\mathbb{E}[\cos(\omega_j \delta)] \approx 0$, but the variance is non-zero:
   $$\operatorname{Var}\left[ \sum_{j \in \text{fast}} \cos(\omega_j \delta) \right] \approx \frac{K_{\text{fast}}}{2}.$$
3. This creates high-frequency background noise in attention logits that drowns out meaningful query-key correlations.

---

### 4.2 Low-Frequency Spectral Collapse & Non-Linear Turnover Catastrophe

#### 4.2.1 In-Distribution Polynomial Degeneracy ($L_2$ Metric)
For slow bands where $\omega_j L_{\text{train}} \ll 1$, the spatial wavelength $\lambda_j \gg 2\pi L_{\text{train}}$ spans far beyond the training window.
For all in-distribution displacements $\Delta \in [0, L_{\text{train}}]$, $\omega_j \Delta \ll 1$. Taylor expanding the basis functions:
$$\cos(\omega_j \Delta) = 1 - \frac{1}{2}\omega_j^2 \Delta^2 + O(\omega_j^4 \Delta^4),$$
$$\sin(\omega_j \Delta) = \omega_j \Delta - \frac{1}{6}\omega_j^3 \Delta^3 + O(\omega_j^5 \Delta^5).$$

Dividing the sine component by $\omega_j$:
$$\frac{\sin(\omega_j \Delta)}{\omega_j} = \Delta - \frac{1}{6}\omega_j^2 \Delta^3 + O(\omega_j^4 \Delta^5) \xrightarrow{\omega_j \to 0} \Delta.$$

**Theorem 4.1 (Low-Frequency $L_2$ Spectral Collapse):**
*As $\omega_j \to 0$, the 2D subspace $V_{\omega_j} = \operatorname{span}\{\cos(\omega_j \Delta), \sin(\omega_j \Delta)\}$ converges in $L_2([0, L_{\text{train}}])$ topology to the affine linear subspace:*
$$V_{\omega_j} \xrightarrow{\omega_j \to 0} \operatorname{span}\{1, \Delta\}.$$

Furthermore, the cross-subspace projection deficit between any two slow frequencies $x = \omega_j L_{\text{train}}$ and $y = \omega_k L_{\text{train}}$ vanishes at fourth order:
$$2 - \|Q_{xy}\|_F^2 = \frac{19}{12600}(x^2 - y^2)^2 + O(\epsilon^6).$$

#### 4.2.2 Softmax Attention Quotient Geometry ($F_{\text{sm}}\mathbf{1} = 0$)
In Transformer self-attention, relative logits pass through the categorical softmax operator. The local Hessian / Fisher metric of the softmax log-partition is the categorical Fisher matrix:
$$F_{\text{sm}} = \operatorname{diag}(p) - p p^\top, \quad \text{where } p_i = \frac{\exp(z_i)}{\sum_k \exp(z_k)}.$$

Because $F_{\text{sm}} \mathbf{1} = \operatorname{diag}(p)\mathbf{1} - p (p^\top \mathbf{1}) = p - p(1) = 0$, **softmax attention is strictly invariant to additive constant shifts**.

Centering any function $f(\Delta)$ against the attention distribution $p$: $\bar{f}(\Delta) \triangleq f(\Delta) - \mathbb{E}_p[f]$.
Applying this centering to the Taylor expansion of slow bands:
$$\frac{\overline{\sin(\omega_j \Delta)}}{\omega_j} = \frac{\sin(\omega_j \Delta) - \mathbb{E}_p[\sin(\omega_j \Delta)]}{\omega_j} \xrightarrow{\omega_j \to 0} \Delta - \mathbb{E}_p[\Delta],$$
$$-\frac{2\,\overline{\cos(\omega_j \Delta)}}{\omega_j^2} = -\frac{2\big(\cos(\omega_j \Delta) - \mathbb{E}_p[\cos(\omega_j \Delta)]\big)}{\omega_j^2} \xrightarrow{\omega_j \to 0} \Delta^2 - \mathbb{E}_p[\Delta^2].$$

**Theorem 4.2 (Softmax Centered Quadratic Quotient Limit):**
*In the attention softmax quotient geometry, all slow frequency bands ($\omega_j L_{\text{train}} \ll 1$) collapse onto the identical 2D centered polynomial subspace:*
$$V_{\omega_j}^{\text{sm}} \xrightarrow{\omega_j \to 0} \operatorname{span}\left\{ \Delta - \mathbb{E}_p[\Delta], \; \Delta^2 - \mathbb{E}_p[\Delta^2] \right\}.$$

#### 4.2.3 Quantification of Dimensionality Loss in Standard RoPE
Consider standard geometric RoPE with $L_{\text{train}} = 4096$, $b = 500\,000$, and head dimension $d = 128$ ($K = 64$ pairs):
- There are **23 slow frequency pairs** (46 nominal dimensions) satisfying $\omega_j L_{\text{train}} \le 1$.
- Together, these 23 pairs have a block-whitened stable rank of:
  $$r_2 \approx 2.00013 \quad (\text{out of } 46 \text{ nominal dimensions}).$$
- Their raw entropy rank is $\approx 1.079$.
- **Stable Dimensionality Loss:** $\frac{46 - 2.00013}{46} \times 100\% = \mathbf{95.65\%}$.

*Physical Interpretation:* Despite dedicating 46 hidden dimensions to slow frequencies, standard geometric RoPE provides only $\approx 2$ effective positional dimensions across the entire lower half of its spectrum during in-distribution training.

#### 4.2.4 The Extrapolation Turnover Catastrophe
During training on $[0, L_{\text{train}}]$, the attention projection weights $W_q, W_k$ learn linear functional readouts against the polynomial slope $\omega_j \Delta \approx \text{linear}$.

When evaluated at $\Delta > L_{\text{train}}$, as $\Delta$ grows to $\Delta \sim \frac{\pi}{2\omega_j}$ or $\frac{\pi}{\omega_j}$:
1. **Departure from Linear Regime:** $\sin(\omega_j \Delta)$ reaches its global maximum at $\omega_j \Delta = \pi/2$ and begins **decreasing**.
2. **Distance Inversion:** The function $\sin(\omega_j \Delta)$, which the network learned to read as a monotonically increasing indicator of distance $\Delta$, turns around and decreases. A larger distance $\Delta_2 > \Delta_1$ produces a *smaller* phase value $\sin(\omega_j \Delta_2) < \sin(\omega_j \Delta_1)$.
3. **Catastrophic Attention Distortion:** Query-key logits $\ell(\Delta) = \sum_j A_j \cos(\omega_j \Delta + \psi_j)$ suffer severe destructive interference, causing attention entropy explosion, loss of focus on relevant tokens, and perplexity explosion.

---

## 5. Destruction of Relative Distance Distinguishability Across Distance Domains

We can formally partition relative distance $\delta = \Delta - \Delta'$ into three geometric regimes:

```
Distance Regime Partitioning:
0 ──────────── lambda_min ──────────────────── L_train ───────────────────────> delta
  [ Near-Field ]           [ Mid-Field (In-Dist) ]          [ Far-Field (Extrap) ]
  Local resolution         Multi-scale interference         Phase aliasing &
  D^2 ~ Omega^2 * d^2      Smooth monotonic decay           Sinusoidal turnover
  Quadratic contrast       G(d) -> 0                        Sidelobes & SNR collapse
```

### 5.1 Domain Characterization

| Domain | Distance Range $\delta$ | Geometric Behavior | Distinguishability Mechanism |
| :--- | :--- | :--- | :--- |
| **Near-Field (Local)** | $\delta \in [0, \lambda_{\min}]$ ($\lambda_{\min} = 2\pi$) | High frequencies dominate; phase angles $\omega_j \delta \ll \pi$. Quadratic curvature $D^2(\delta) \approx \Omega_{\text{tot}}^2 \delta^2$. | **Sharp local resolution:** Adjacent tokens have distinct phase codes; high contrast. |
| **Mid-Field (In-Dist)** | $\delta \in [\lambda_{\min}, L_{\text{train}}]$ | Log-uniform geometric progression $\omega_j = b^{-j/K}$ creates smooth multi-scale wave packet dispersion. Fast bands average out; intermediate bands provide monotonic contrast. | **Monotonic decay:** $G(\delta)$ smoothly decays from $K$ toward 0; low pairwise collision. |
| **Far-Field (Extrapolation)** | $\delta > L_{\text{train}}$ | Fast bands alias and generate pseudo-random hash noise ($\operatorname{Var} \sim K_{\text{fast}}/2$); slow bands exit linear regime and undergo sinusoidal turnover. | **Catastrophic breakdown:** Spurious revivals, loss of injectivity, distance SNR collapse. |

### 5.2 Distance Signal-to-Noise Ratio Collapse
Define the **Distance Distinguishability Signal-to-Noise Ratio** $\operatorname{SNR}_{\text{dist}}(\delta)$:
$$\operatorname{SNR}_{\text{dist}}(\delta) \triangleq \frac{G(0) - \mathbb{E}[G(\delta)]}{\sqrt{\operatorname{Var}[G(\delta)]}} = \frac{K - \sum_{j=0}^{K-1} \cos(\omega_j \delta)}{\sqrt{\operatorname{Var}\left[\sum_{j=0}^{K-1} \cos(\omega_j \delta)\right]}}.$$

- In the **Near-Field**, $\operatorname{SNR}_{\text{dist}}(\delta) \propto \delta^2 \Omega_{\text{tot}}^2 \gg 1$.
- In the **Mid-Field**, wave-packet destructive interference keeps $\operatorname{Var}[G(\delta)]$ small while maintaining $G(0) - G(\delta) \approx K$.
- In the **Far-Field ($\delta > L_{\text{train}}$)**:
  - The mean signal $G(0) - \mathbb{E}[G(\delta)]$ is corrupted by slow-band sinusoidal oscillation.
  - The noise variance $\operatorname{Var}[G(\delta)]$ is inflated by fast-band aliasing.
  - $\operatorname{SNR}_{\text{dist}}(\delta)$ collapses, destroying the model's ability to discriminate distinct out-of-distribution positions.

---

## 6. Synthesis with Repository Evidence, Counterexamples & Claim Ceilings

### 6.1 Strict Claim Ceiling Adherence
In accordance with `AGENTS.md` and repository standards:

1. **Full-RoPE Geometry Ceiling:**
   *Full-RoPE geometry ($r_2(R)$, canonical collision $c_{jk}$, logdet) is a static, phase-invariant property of the positional basis under a declared distance measure. It measures positional-basis redundancy and effective dimensionality, NOT LM quality, perplexity, or extrapolation performance.*
2. **Low-Frequency Collapse Ceiling:**
   *Slow bands are redundant in the stated static metric; this does not mean they are unused or reclaimable.* Trained checkpoint attention probes (e.g. 50M probe) demonstrate that these 22–23 slow channels form a $\approx 1.05$ logit recency kernel that the network actively exploits.
3. **EVQ-Cosh Placement:**
   *EVQ-Cosh is a closed-form, zero-learned-parameter construction and controlled intervention that uniquely optimizes the stated convex surrogate functional $\mathcal{J}[\rho]$. It is not a unique or universal optimum for language modeling or general extrapolation.*

---

### 6.2 Three Decisive Counterexamples to Naive Extrapolation Optimization
Repository audits (`analysis/full_rope_audit/` and `paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`) rigorously falsified the hypothesis that "minimizing static collision or maximizing static effective rank guarantees improved extrapolation":

#### Counterexample 1: Cosine-Only Collision Selects Sub-Optimal Rank
At $K = 64$:
$$C_{\cos}(A) = 2.62 \times 10^{-9} < 1.44 \times 10^{-8} = C_{\cos}(B),$$
yet the full block-whitened stable rank is inverted:
$$r_2(A) = 64.48 < 128.00 = r_2(B).$$
Cosine-only collision measures only the $\cos\text{--}\cos$ block at zero content phase, failing to capture full 2D subspace geometry.

#### Counterexample 2: Length Ranking Inversion Across $L, 2L, 4L$
For two fixed-endpoint tables at $K = 16$:
$$C_L(A) = 0.45837 < 0.61695 = C_L(B),$$
but at $2L$ and $4L$:
$$C_{2L}(A) = 0.45834 > 0.41132 = C_{2L}(B), \qquad C_{4L}(A) = 0.45830 > 0.24763 = C_{4L}(B).$$
Static collision rankings on $[0, L]$ invert on extended domains $[0, 2L]$ and $[0, 4L]$.

#### Counterexample 3: Exact Fourier Harmonic Comb (The 100% Aliasing Catastrophe)
Optimizing static full-subspace rank $r_2(R)$ on $[0, L]$ yields an exact Fourier harmonic comb $\omega_k = \frac{2\pi k}{L}$.
- On $[0, L]$, this table achieves **perfect orthogonality** ($c_{jk} = 0, r_2(R) = 2K$).
- However, at any extrapolation distance $\Delta > L$:
  $$\Phi(\Delta + L) = \Phi(\Delta), \quad \forall \Delta.$$
  The table exhibits **100% exact periodic aliasing**, completely destroying position uniqueness outside $[0, L]$.

---

### 6.3 Separation of Causal Coordinates
To ensure scientific clarity, the RoPE frequency tensor must be factored into two distinct causal coordinates:
$$x_k = -\log \omega_k = a + R z_k, \quad \text{with } z_0 = 0, \; z_{K-1} = 1.$$
- **$(a, R)$ — Sampled Spectral Support:** Governs the global base $b$ and physical endpoint frequencies $[\omega_{\min}, \omega_{\max}]$.
- **$z \in [0, 1]^K$ — Normalized Interior Allocation:** Governs the allocation profile and spectral density within the fixed support (e.g. geometric $z_k = k/(K-1)$ vs. EVQ-Cosh $z_k$).

Causal identification of pure interior allocation ($z$) is owned by the **Exact-Range 151M 3-seed experiment** (`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`), while checkpoint retrofit compatibility is governed by the **50M 2x2 table-weight crossing**.

---

## 7. Implications for Downstream Research Axes

1. **For R2 (Frozen Checkpoint Q/K Readout & Co-adaptation):**
   The attention logit $\ell(\Delta) = q^\top \mathbf{R}(\Delta) k = \sum_{j=0}^{K-1} A_j \cos(\omega_j \Delta + \psi_j)$ directly couples frozen projection weights $W_q, W_k$ (which set amplitudes $A_j$ and phases $\psi_j$) with the phase spectrum $\omega_j$. When $\Delta > L_{\text{train}}$, slow-band turnover and fast-band aliasing cause severe mismatch with frozen $A_j, \psi_j$, explaining attention entropy collapse.
2. **For R3 (Non-Linear $f(z) \neq cz$ vs. Linear Base Scaling $cz$):**
   Linear scaling $f(z) = cz$ merely dilates the support $(a, R) \to (a, cR)$, shifting the entire spectrum without altering relative channel density. In contrast, non-linear $f(z)$ redistributes channel density $\rho(\omega) = \frac{\mathrm{d}z}{\mathrm{d}\omega}$, selectively thinning redundant slow bands and reinforcing critical mid-frequency bands.
3. **For R4 (Empirical Synthesis & Practical Value):**
   Non-linear spectrum allocation provides structured, zero-parameter inductive bias that mitigates low-frequency collapse while preserving multi-scale wave-packet dispersion.

---
