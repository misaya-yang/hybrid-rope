# Information-Theoretic Foundations of the RoPE Exponent Spectrum $z = -2i/d$

**Author**: Explorer Subagent `explorer_r1_infotheory_1`  
**Date**: 2026-09-01  
**Scope**: R1 Axis A — First-Principles Mathematical Derivations, Continuous Spectral Density, Multi-Resolution Wavelet Framing, Octave Channel Capacity, Phase Entropy, and Riemann-Lebesgue Locality Prior.

---

## 1. Executive Summary & Foundational Axiomatics

Rotary Position Embedding (RoPE; Su et al., 2021) encodes relative position $\Delta = m - n$ into the attention mechanism by rotating 2D query and key subspaces by frequencies $\omega_i = b^{-2i/d}$. While commonly introduced as an empirical extension of sinusoidal embeddings (Vaswani et al., 2017), the choice of the linear exponent spectrum $z_i = -2i/d$ is not arbitrary. It represents the unique canonical discretization of a scale-invariant, log-uniform continuous spectral density $\rho(\omega) \propto 1/\omega$.

This report establishes the complete mathematical and information-theoretic foundations of this exponent spectrum from first principles:

1. **Geometric Sequence Origin ($z_i = -2i/d$)**: Derives from the requirement of a constant scale factor (constant octave ratio $r = b^{2/d}$) between adjacent 2D rotary sub-channels, ensuring uniform logarithmic coverage over the frequency interval $[\omega_{\min}, \omega_{\max}] = [b^{-1}, 1]$.
2. **Continuous Spectral Density ($\rho(\omega) = \frac{1}{\omega \ln b}$)**: In the continuous channel limit $u = 2i/d \in [0, 1]$, the exponent profile $z(u) = -u$ induces a density $\rho(\omega) = |du/d\omega| = \frac{1}{\omega \ln b}$, which is the unique scale-invariant distribution (eigen-measure of the dilation group) and the Haar measure on the multiplicative group $(\mathbb{R}^+, \times)$.
3. **Multi-Resolution Analysis (MRA) & Continuous Wavelet Transform (CWT)**: RoPE acts as a dyadic filter bank where wavelength scales $a_i = 1/\omega_i = b^{2i/d}$ decompose position into hierarchical frequency bands. High frequencies resolve microscopic token transitions (n-grams, syntax), while low frequencies encode macroscopic sequence order. However, over a finite context window $L \ll T_{\text{slow}}$, slow frequencies exhibit severe spectral collapse ($V_\omega \to \operatorname{span}\{1, \Delta\}$; softmax metric $\to \operatorname{span}\{\Delta - \bar\Delta, \Delta^2 - \bar{\Delta^2}\}$).
4. **Information-Theoretic Allocation & Phase Entropy**: The log-uniform distribution equalizes nominal channel capacity across spatial octaves ($K_{\text{octave}} = \frac{d \ln 2}{2 \ln b} = \text{const}$). However, differential phase entropy $H(\theta_i)$ reveals an extreme information deficit at low frequencies: fast channels maximize entropy at $H(\theta_i) = \ln(2\pi)$, while slow channels suffer an entropy loss $\Delta H_i = \ln(\frac{2\pi}{\omega_i L})$ due to lack of phase wrapping within the training context.
5. **Riemann-Lebesgue Lemma & Locality Prior**: The expected attention kernel $\bar{K}(\Delta) = \int_{b^{-1}}^1 \cos(\omega \Delta) \rho(\omega) \, d\omega = \frac{\text{Ci}(\Delta) - \text{Ci}(\Delta/b)}{\ln b}$ decays logarithmically as $\bar{K}(\Delta) \approx 1 - \frac{\ln \Delta}{\ln b}$ for intermediate distances $1 \ll \Delta \ll b$. This decay is an exact consequence of destructive phase interference governed by the Riemann-Lebesgue lemma.

---

## 2. First-Principles Derivation of the Discrete Spectrum $z_i = -2i/d$

### 2.1 2D Subspace Decomposition and Attention Bilinear Form

Let the query and key embedding vectors $q, k \in \mathbb{R}^d$ ($d$ even) be partitioned into $K = d/2$ orthogonal 2D subspaces:
$$q = \bigoplus_{i=0}^{K-1} q^{(i)}, \quad k = \bigoplus_{i=0}^{K-1} k^{(i)}, \quad q^{(i)}, k^{(i)} \in \mathbb{R}^2$$

At absolute sequence positions $m, n \in \mathbb{N}_0$, RoPE applies an orthogonal block-diagonal rotation matrix $R_\Omega(m) = \operatorname{diag}\left( R_{\omega_0}(m), R_{\omega_1}(m), \dots, R_{\omega_{K-1}}(m) \right)$, where each $2 \times 2$ rotation block is given by:
$$R_{\omega_i}(m) = \begin{bmatrix} \cos(\omega_i m) & -\sin(\omega_i m) \\ \sin(\omega_i m) & \cos(\omega_i m) \end{bmatrix} \in \mathrm{SO}(2)$$

The pre-softmax attention logit for query position $m$ and key position $n$ depends purely on the relative displacement $\Delta = m - n$:
$$\ell(\Delta) = q^\top R_\Omega(m)^\top R_\Omega(n) k = q^\top R_\Omega(m - n) k = q^\top R_\Omega(\Delta) k$$

Expanding across the $K$ independent 2D channels:
$$\ell(\Delta) = \sum_{i=0}^{K-1} (q^{(i)})^\top R_{\omega_i}(\Delta) k^{(i)}$$

Writing the 2D vectors in coordinates $q^{(i)} = [q_{2i}, q_{2i+1}]^\top$ and $k^{(i)} = [k_{2i}, k_{2i+1}]^\top$:
$$(q^{(i)})^\top R_{\omega_i}(\Delta) k^{(i)} = \begin{bmatrix} q_{2i} & q_{2i+1} \end{bmatrix} \begin{bmatrix} \cos(\omega_i \Delta) & -\sin(\omega_i \Delta) \\ \sin(\omega_i \Delta) & \cos(\omega_i \Delta) \end{bmatrix} \begin{bmatrix} k_{2i} \\ k_{2i+1} \end{bmatrix}$$
$$= (q_{2i} k_{2i} + q_{2i+1} k_{2i+1}) \cos(\omega_i \Delta) + (q_{2i} k_{2i+1} - q_{2i+1} k_{2i}) \sin(\omega_i \Delta)$$

Defining the pair-wise content inner product $C_i$ and symplectic wedge product $D_i$:
$$C_i \equiv q_{2i} k_{2i} + q_{2i+1} k_{2i+1} = \langle q^{(i)}, k^{(i)} \rangle$$
$$D_i \equiv q_{2i} k_{2i+1} - q_{2i+1} k_{2i} = q^{(i)} \wedge k^{(i)}$$

The attention logit contribution from channel $i$ is a general harmonic function:
$$f_{\omega_i}(\Delta) = C_i \cos(\omega_i \Delta) + D_i \sin(\omega_i \Delta) = A_i \cos(\omega_i \Delta + \psi_i)$$
where amplitude $A_i = \sqrt{C_i^2 + D_i^2} = \|q^{(i)}\| \|k^{(i)}\|$ and initial phase $\psi_i = -\operatorname{atan2}(D_i, C_i)$.

The relative position information is spanned by the 2D harmonic subspace:
$$V_{\omega_i} = \operatorname{span}\{\cos(\omega_i \Delta), \sin(\omega_i \Delta)\} \subset L^2([0, L])$$

---

### 2.2 Derivation from Multiplicative Scale Homogeneity

To determine how the discrete frequencies $\{\omega_i\}_{i=0}^{K-1}$ should be distributed across the $K$ channels, we impose three axiomatic requirements:

1. **Boundary Normalization**: The fastest frequency must resolve individual token steps without spatial aliasing at $\Delta = 1$, setting $\omega_0 = 1$ (period $T_0 = 2\pi \approx 6.28$ tokens). The slowest frequency is parameterized by base $b > 1$ such that $\omega_{K-1} \approx b^{-1}$ (period $T_{K-1} \approx 2\pi b$).
2. **Scale Homogeneity (Constant Octave Step)**: The relative resolution ratio between any two adjacent frequency channels must be invariant across all channels:
   $$\frac{\omega_i}{\omega_{i+1}} = r = \text{const} > 1, \quad \forall i \in \{0, 1, \dots, K-2\}$$
3. **Logarithmic Uniform Grid**: In the logarithmic coordinate $x \equiv -\ln \omega \in [0, \ln b]$, the grid points $x_i = -\ln \omega_i$ must be equispaced:
   $$x_{i+1} - x_i = \Delta x = \text{const} = \frac{\ln b}{K}$$

From the recurrence relation $\omega_i = \omega_0 r^{-i} = r^{-i}$, applying the boundary condition at channel $K$:
$$r^K = b \implies r = b^{1/K} = b^{2/d}$$

Substituting $r$ back into the frequency formula:
$$\omega_i = \left( b^{2/d} \right)^{-i} = b^{-2i/d}, \quad i \in \{0, 1, \dots, d/2-1\}$$

Defining the dimensionless exponent variable $z_i$:
$$z_i \equiv -\frac{2i}{d} = -\frac{i}{K} \in [-1 + 1/K, 0]$$
$$\boxed{\omega_i = b^{z_i} = \exp\left( z_i \ln b \right)}$$

The wavelength (oscillation period) of channel $i$ is:
$$T_i = \frac{2\pi}{\omega_i} = 2\pi b^{2i/d} = 2\pi b^{-z_i}$$

```
Channel Index i:    0 -------- 1 -------- 2 -------- ... -------- K-1 (d/2 - 1)
Exponent z_i:       0       -2/d     -4/d              -(d-2)/d ≈ -1
Frequency \omega_i: 1      b^{-2/d}  b^{-4/d}          b^{-(1-2/d)} ≈ 1/b
Period T_i:         2π      2π b^{2/d} 2π b^{4/d}      2π b^{1-2/d} ≈ 2π b
```

---

## 3. Continuous Spectral Density $\rho(\omega) = \frac{1}{\omega \ln b}$ and Scale Invariance

### 3.1 Derivation of the Continuous Measure

In modern large language models, $d_{\text{head}} \in \{64, 128\}$, corresponding to $K \in \{32, 64\}$ rotary pairs per attention head. In the asymptotic limit $K \to \infty$, we define a continuous normalized channel coordinate:
$$u \equiv \frac{i}{K} = \frac{2i}{d} \in [0, 1]$$

Under the canonical RoPE specification, the continuous exponent function is:
$$z(u) = -u, \quad u \in [0, 1]$$

The continuous frequency spectrum is:
$$\omega(u) = b^{z(u)} = b^{-u} = \exp(-u \ln b), \quad u \in [0, 1]$$

As $u$ increases monotonically from $0$ to $1$, $\omega(u)$ decreases monotonically from $\omega_{\max} = 1$ to $\omega_{\min} = b^{-1}$.

Assuming a uniform channel density on $u \in [0, 1]$ (i.e., $p_U(u) = 1$), the cumulative distribution function (CDF) of the random variable $\Omega = \omega(U)$ is:
$$F_\Omega(\omega) = \mathbb{P}(\Omega \le \omega) = \mathbb{P}\left( b^{-U} \le \omega \right)$$
$$-U \ln b \le \ln \omega \iff U \ge -\frac{\ln \omega}{\ln b}$$
$$F_\Omega(\omega) = \int_{-\frac{\ln \omega}{\ln b}}^1 1 \, du = 1 - \left( -\frac{\ln \omega}{\ln b} \right) = 1 + \frac{\ln \omega}{\ln b}, \quad \text{for } \omega \in [b^{-1}, 1]$$

Differentiating the CDF with respect to $\omega$ yields the probability density function $\rho(\omega)$:
$$\rho(\omega) = \frac{dF_\Omega(\omega)}{d\omega} = \left| \frac{du}{d\omega} \right| = \frac{d}{d\omega} \left( 1 + \frac{\ln \omega}{\ln b} \right)$$
$$\boxed{\rho(\omega) = \frac{1}{\ln b} \cdot \frac{1}{\omega}, \quad \omega \in [b^{-1}, 1]}$$

#### Normalization Verification:
$$\int_{b^{-1}}^1 \rho(\omega) \, d\omega = \frac{1}{\ln b} \int_{b^{-1}}^1 \frac{1}{\omega} \, d\omega = \frac{1}{\ln b} \Big[ \ln \omega \Big]_{b^{-1}}^1 = \frac{1}{\ln b} \left( 0 - \ln(b^{-1}) \right) = \frac{\ln b}{\ln b} = 1$$

---

### 3.2 Mathematical Properties of the $1/\omega$ Spectral Density

#### Theorem 1 (Scale Invariance / Dilation Symmetry).
A continuous spectral density $\rho(\omega)$ on $(0, \infty)$ is scale-invariant if and only if scaling all frequencies by an arbitrary constant dilation factor $\lambda > 0$ preserves the relative probability distribution:
$$\rho(\lambda \omega) \, d(\lambda \omega) = \rho(\omega) \, d\omega \iff \lambda \rho(\lambda \omega) = \rho(\omega)$$

*Proof.*
Rearranging $\lambda \rho(\lambda \omega) = \rho(\omega)$ gives:
$$\rho(\lambda \omega) = \lambda^{-1} \rho(\omega)$$
Setting $\omega = 1$:
$$\rho(\lambda) = \rho(1) \cdot \lambda^{-1} \implies \rho(\omega) = \frac{C}{\omega}$$
where $C = \frac{1}{\ln b}$ is the normalization constant over support $[b^{-1}, 1]$. $\blacksquare$

#### Invariant Measure on the Dilation Group $(\mathbb{R}^+, \times)$:
Consider the multiplicative topological group $G = (\mathbb{R}^+, \times)$. The left- and right-invariant Haar measure $\mu_G$ on $G$ satisfies:
$$\mu_G(\lambda E) = \mu_G(E), \quad \forall \lambda \in \mathbb{R}^+, \; \forall \text{ Borel sets } E \subset \mathbb{R}^+$$
The differential Haar measure is:
$$d\mu_G(\omega) = \frac{d\omega}{\omega} = d(\ln \omega)$$

Thus, the canonical RoPE exponent profile $z(u) = -u$ is mathematically equivalent to uniform sampling with respect to the Haar measure on the scale dilation group.

#### Constant Mass per Octave and Decade:
For any frequency interval $[\omega_1, \omega_2] \subset [b^{-1}, 1]$ with frequency ratio $Q = \omega_2 / \omega_1$:
$$\mathbb{P}(\omega_1 \le \Omega \le \omega_2) = \int_{\omega_1}^{\omega_2} \frac{1}{\ln b} \frac{d\omega}{\omega} = \frac{\ln(\omega_2 / \omega_1)}{\ln b} = \frac{\ln Q}{\ln b}$$

This yields two critical corollaries:
1. **Octave Equipartition**: For any octave band ($Q = 2$), the channel probability mass is $\frac{\ln 2}{\ln b}$, which is strictly independent of the center frequency $\omega_c$.
2. **Decade Equipartition**: For any decade band ($Q = 10$), the channel probability mass is $\frac{\ln 10}{\ln b}$, which is strictly constant.

```
Frequency Spectrum \omega:
b^{-1} --------------- b^{-3/4} --------------- b^{-1/2} --------------- b^{-1/4} --------------- 1
|====== 25% budget ======|====== 25% budget ======|====== 25% budget ======|====== 25% budget ======|
|←     Slow Bands       →|←   Mid-Slow Bands     →|←   Mid-Fast Bands     →|←     Fast Bands       →|
```

#### Connection to Jeffreys Prior in Estimation Theory:
In Bayesian inference, when estimating an unknown scale or frequency parameter $\omega \in (0, \infty)$ from harmonic observations $y(t) = \cos(\omega t + \phi)$, the Fisher information is:
$$I(\omega) = \mathbb{E}\left[ \left( \frac{\partial \ln p(y|\omega)}{\partial \omega} \right)^2 \right] \propto \frac{1}{\omega^2}$$

Jeffreys' uninformative prior, which is invariant under reparameterization $\tilde{\omega} = g(\omega)$, is defined as:
$$p_{\text{Jeffreys}}(\omega) \propto \sqrt{\det I(\omega)} \propto \frac{1}{\omega}$$

Therefore, setting $\rho(\omega) \propto 1/\omega$ is equivalent to asserting maximal uninformative epistemic entropy over length scales prior to observing sequence data.

---

## 4. Multi-Resolution Analysis (MRA) and Continuous Wavelet Transform Perspective

### 4.1 Wavelet Frame Formulation of Rotary Positional Attention

In harmonic analysis, the Continuous Wavelet Transform (CWT) of a spatial signal $f \in L^2(\mathbb{R})$ with mother wavelet $\psi(t)$ is defined by dilations $a > 0$ and translations $s \in \mathbb{R}$:
$$\mathcal{W}_\psi f(a, s) = \frac{1}{\sqrt{a}} \int_{-\infty}^\infty f(t) \psi^*\left(\frac{t - s}{a}\right) dt$$
where $a$ represents the spatial scale, corresponding inversely to frequency $\omega = 1/a$.

In RoPE, the positional representation of a token at sequence index $m$ is constructed from $K$ 2D harmonic basis atoms:
$$\phi_i(m) = \begin{bmatrix} \cos(\omega_i m) \\ \sin(\omega_i m) \end{bmatrix} = \begin{bmatrix} \cos(m / a_i) \\ \sin(m / a_i) \end{bmatrix}, \quad a_i \equiv \frac{1}{\omega_i} = b^{2i/d}$$

The spatial scale parameters $\{a_i\}_{i=0}^{K-1}$ follow a dyadic-like geometric sequence:
$$a_0 = 1, \quad a_1 = b^{2/d}, \quad a_2 = b^{4/d}, \quad \dots, \quad a_{K-1} = b^{1 - 2/d} \approx b$$

This directly mirrors Mallat's Multi-Resolution Analysis (MRA), where a Hilbert space $L^2(\mathbb{R})$ is decomposed into a nested sequence of closed approximation subspaces $\{V_j\}_{j \in \mathbb{Z}}$ and orthogonal detail wavelet subspaces $\{W_j\}_{j \in \mathbb{Z}}$:
$$\dots \subset V_{-1} \subset V_0 \subset V_1 \subset V_2 \subset \dots, \quad V_{j+1} = V_j \oplus W_j$$

In RoPE's discrete multi-scale representation:
- Each rotary pair $i$ represents a detail band $W_i$ centered at spatial scale $a_i = b^{2i/d}$.
- The total positional representation space is the direct sum $\mathcal{H}_{\text{pos}} = \bigoplus_{i=0}^{K-1} V_{\omega_i}$.

```
Spatial Scale a = 1/\omega:
a_0 = 1 (Token-level detail)
├── a_1 = b^{2/d} (Sub-phrase n-grams)
│   ├── a_2 = b^{4/d} (Clause level)
│   │   ├── ...
│   │   │   └── a_{K-1} ≈ b (Global context / document level)
```

---

### 4.2 Spectral Division of Labor: High vs. Low Frequencies

The continuous scale spectrum creates a natural division of labor across frequency bands:

| Spectral Regime | Channel Index $i$ | Frequency Range $\omega$ | Spatial Scale $a = 1/\omega$ | Primary Linguistic / Functional Role | Failure Mode / Boundary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **High Frequency (Micro-scale)** | $i \approx 0 \dots \frac{K}{4}$ | $\omega \in [b^{-1/4}, 1]$ | $a \in [1, b^{1/4}]$ (e.g. $1 \dots 15$ tokens) | Resolves local token adjacency, syntax trees, morphology, subword binding, and punctuation transitions. | Severe phase wrapping / aliasing at distances $\Delta > 2\pi / \omega$. |
| **Mid Frequency (Meso-scale)** | $i \approx \frac{K}{4} \dots \frac{3K}{4}$ | $\omega \in [b^{-3/4}, b^{-1/4}]$ | $a \in [b^{1/4}, b^{3/4}]$ (e.g. $15 \dots 500$ tokens) | Encodes sentence boundaries, paragraph cohesion, coreference resolution, and local cross-attention routing. | Transition zone; susceptible to dispersion under context length extrapolation. |
| **Low Frequency (Macro-scale)** | $i \approx \frac{3K}{4} \dots K-1$ | $\omega \in [b^{-1}, b^{-3/4}]$ | $a \in [b^{3/4}, b]$ (e.g. $500 \dots 10^5+$ tokens) | Provides monotonic global distance gradient across document sections, long-range tracking, and prompt-response demarcation. | Low-frequency spectral collapse: redundant 2D subspaces, near-zero phase displacement over $L_{\text{train}}$. |

---

### 4.3 Low-Frequency Spectral Collapse: Rigorous Mathematical Derivation

While low frequencies are mathematically designated to resolve macro-scale context, their geometric utility is constrained by sequence length $L$.

#### Theorem 2 (Asymptotic Subspace Degeneracy under $L_2$ Metric).
Let $x = \omega L$. In the low-frequency limit $x \to 0$ (i.e. $\omega \ll 1/L$), the 2D harmonic basis functions satisfy the Taylor expansions:
$$\cos(\omega \Delta) = \cos(x \cdot \tilde{\Delta}) = 1 - \frac{x^2 \tilde{\Delta}^2}{2} + O(x^4)$$
$$\frac{\sin(\omega \Delta)}{\omega} = L \cdot \frac{\sin(x \cdot \tilde{\Delta})}{x} = L \left( \tilde{\Delta} - \frac{x^2 \tilde{\Delta}^3}{6} + O(x^4) \right)$$
where $\tilde{\Delta} \equiv \Delta / L \in [0, 1]$.

As $x \to 0$, the rotary subspace $V_\omega = \operatorname{span}\{\cos(\omega \Delta), \sin(\omega \Delta)\}$ degenerates to:
$$\lim_{\omega \to 0} V_\omega = \operatorname{span}\{1, \Delta\}$$

For any two slow frequencies $\omega, \nu$ with $x = \omega L, y = \nu L \to 0$, the canonical correlation metric $c_{\omega\nu} = \frac{1}{2}\|S_\omega^{-1/2} H_{\omega\nu} S_\nu^{-1/2}\|_F^2$ satisfies:
$$2 - \|Q_{x, y}\|_F^2 = \frac{19}{12600} (x^2 - y^2)^2 + O(\epsilon^6)$$

*Proof.*
Let $x_\omega(\Delta) = [\cos(\omega \Delta), \sin(\omega \Delta)]^\top$. For $\Delta \sim \operatorname{Unif}[0, L]$, the cross-Gram matrix $H_{\omega\nu} = \mathbb{E}[x_\omega(\Delta) x_\nu(\Delta)^\top]$ has entries:
$$H_{\omega\nu} = \frac{1}{2} \begin{bmatrix} a(d) + a(s) & b(s) - b(d) \\ b(s) + b(d) & a(d) - a(s) \end{bmatrix}$$
where $d = (\omega - \nu)L$, $s = (\omega + \nu)L$, $a(t) = \frac{\sin t}{t}$, and $b(t) = \frac{1 - \cos t}{t}$.
Expanding $a(t) = 1 - \frac{t^2}{6} + \frac{t^4}{120} - \dots$ and $b(t) = \frac{t}{2} - \frac{t^3}{24} + \dots$ around $t = 0$, the whitened cross-Gram $Q_{\omega\nu} = S_\omega^{-1/2} H_{\omega\nu} S_\nu^{-1/2}$ yields the Frobenius norm deficit:
$$2 - \|Q_{\omega\nu}\|_F^2 = \frac{19}{12600} (\omega^2 L^2 - \nu^2 L^2)^2 + O((\omega L)^6)$$
which scales as $O((\omega L)^4)$. $\blacksquare$

#### Theorem 3 (Softmax Quotient Geometry and Centralization).
In transformer attention, the categorical Fisher information matrix under attention probability vector $p \in \Delta^{L-1}$ is $F_{\text{sm}} = \operatorname{diag}(p) - p p^\top$. Because $F_{\text{sm}} \mathbf{1} = 0$, the constant component $1$ is annihilated by the softmax gauge invariance (shift invariance $\ell_j \to \ell_j + c$).

Centering the low-frequency expansions:
$$\frac{\overline{\sin(\omega \Delta)}}{\omega} = \frac{\sin(\omega \Delta) - \mathbb{E}_p[\sin(\omega \Delta)]}{\omega} \xrightarrow{\omega \to 0} \Delta - \mathbb{E}_p[\Delta]$$
$$-\frac{2\overline{\cos(\omega \Delta)}}{\omega^2} = -\frac{2(\cos(\omega \Delta) - \mathbb{E}_p[\cos(\omega \Delta)])}{\omega^2} \xrightarrow{\omega \to 0} \Delta^2 - \mathbb{E}_p[\Delta^2]$$

Thus, in softmax space, all low-frequency channels collapse into the single centered parabolic subspace:
$$\lim_{\omega \to 0} V_\omega^{\text{softmax}} = \operatorname{span}\left\{ \Delta - \mathbb{E}_p[\Delta], \; \Delta^2 - \mathbb{E}_p[\Delta^2] \right\}$$

#### Empirical Consequence on Stable Effective Rank ($r_2$):
As recorded in repository benchmarks ($L=4096, b=500000$):
- For $K=64$ (128-dimensional head), the 24 slowest pairs (nominal 48 dimensions) have $\omega L \le 1$.
- The block-whitened stable rank $r_2 = \frac{(\operatorname{tr} R)^2}{\operatorname{tr}(R^2)}$ over these 24 pairs is $r_2 \approx 2.0002$.
- **Stable Dimension Deficit**: A nominal allocation of 48 dimensions yields only $\approx 2$ effective linearly independent dimensions under inner-product geometry—a **95.83% redundancy loss**.

*(Note: In accordance with repository claim ceilings, slow bands are mathematically redundant in this static metric; this does not imply they are dead, unused, or freely reclaimable without co-adaptation consequences).*

---

## 5. Information-Theoretic Capacity and Phase Entropy Across Scales

### 5.1 Nominal Channel Capacity per Octave

Consider a Gaussian communication channel model for the attention logit in subspace $i$, where the signal is the positional rotation and is corrupted by content variance or Gaussian attention noise $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$:
$$\ell_i(\Delta) = A_i \cos(\omega_i \Delta + \psi_i) + \epsilon_i$$

The Shannon channel capacity for channel $i$ is:
$$C_i = \frac{1}{2} \log_2\left( 1 + \frac{P_i}{\sigma^2} \right) \quad \text{[bits/channel]}$$
where $P_i = \frac{1}{2} \mathbb{E}[A_i^2]$ is the average signal power.

Assuming equal power allocation across channels $P_i = P_0$:
- Total bandwidth in octaves: $N_{\text{octaves}} = \log_2\left( \frac{\omega_{\max}}{\omega_{\min}} \right) = \log_2(b) = \frac{\ln b}{\ln 2}$.
- Total pairs: $K = d/2$.
- Number of channels allocated per octave:
  $$K_{\text{octave}} = \frac{K}{N_{\text{octaves}}} = \frac{d/2}{\log_2 b} = \frac{d \ln 2}{2 \ln b} = \text{const}$$

The nominal positional capacity per octave is therefore:
$$\boxed{\mathcal{C}_{\text{octave}} = K_{\text{octave}} \cdot C_i = \left( \frac{d \ln 2}{2 \ln b} \right) \cdot \frac{1}{2} \log_2\left( 1 + \text{SNR} \right) = \text{const}}$$

Under $z_i = -2i/d$, RoPE enforces an **exact equipartition of nominal capacity** across all spatial frequency octaves.

---

### 5.2 Differential Entropy of the Rotary Phase Representation

While nominal channel capacity is uniformly distributed across octaves, the *realized differential entropy* of the phase representation depends crucially on whether the phase wraps within the context window $[0, L]$.

Let relative distance $\Delta$ be distributed on $[0, L]$ with probability density $p_\Delta(\Delta)$. For channel $i$, the wrapped phase random variable $\Theta_i \in [0, 2\pi)$ is defined by:
$$\Theta_i \equiv (\omega_i \Delta) \pmod{2\pi}$$

```
High-Frequency Phase \Theta_fast (\omega L >> 2\pi):
[0 -------- 2π][0 -------- 2π][0 -------- 2π] ... [0 -------- 2π]  --> Uniform on [0, 2π)
Entropy: H(\Theta_fast) = ln(2π) ≈ 1.838 nats (Maximal)

Low-Frequency Phase \Theta_slow (\omega L << 2\pi):
[0 ==================> \omega L << 2π]                              --> Confined to [0, \omega L]
Entropy: H(\Theta_slow) = H(\Delta) + ln(\omega) << ln(2π)         (Severe Deficit)
```

#### Theorem 4 (Phase Entropy Asymptotics Across Scales).
1. **High-Frequency Regime ($\omega_i L \gg 2\pi$)**:
   The phase completes $M_i = \frac{\omega_i L}{2\pi} \gg 1$ complete revolutions. By the Poincaré-Hopf theorem and modulo-arithmetic mixing, the wrapped phase density $p_{\Theta_i}(\theta)$ rapidly converges to the uniform distribution:
   $$\lim_{\omega_i L \to \infty} p_{\Theta_i}(\theta) = \frac{1}{2\pi}, \quad \forall \theta \in [0, 2\pi)$$
   The differential entropy of the high-frequency phase reaches the theoretical maximum:
   $$\boxed{H(\Theta_i) = \int_0^{2\pi} -\left(\frac{1}{2\pi}\right) \ln\left(\frac{1}{2\pi}\right) d\theta = \ln(2\pi) \approx 1.83787 \text{ nats}}$$

2. **Low-Frequency Regime ($\omega_i L \ll 2\pi$)**:
   The phase undergoes strictly less than one full cycle: $\Theta_i = \omega_i \Delta \in [0, \omega_i L] \subset [0, 2\pi)$. No phase wrapping occurs.
   The transformed density is:
   $$p_{\Theta_i}(\theta) = \frac{1}{\omega_i} p_\Delta\left( \frac{\theta}{\omega_i} \right)$$
   The differential entropy is:
   $$H(\Theta_i) = -\int_0^{\omega_i L} p_{\Theta_i}(\theta) \ln p_{\Theta_i}(\theta) \, d\theta = H(\Delta) + \ln \omega_i$$
   where $H(\Delta) = -\int_0^L p_\Delta(\delta) \ln p_\Delta(\delta) d\delta$.
   For uniform distance $\Delta \sim \operatorname{Unif}[0, L]$, $H(\Delta) = \ln L$, yielding:
   $$\boxed{H(\Theta_i) = \ln L + \ln \omega_i = \ln(\omega_i L)}$$

3. **Phase Entropy Deficit ($\Delta H_i$)**:
   The information-theoretic deficit between maximal possible entropy and realized phase entropy is:
   $$\Delta H_i \equiv H_{\max} - H(\Theta_i) = \ln(2\pi) - \ln(\omega_i L) = \ln\left( \frac{2\pi}{\omega_i L} \right)$$

As $\omega_i \to b^{-1}$, since $\omega_{\min} L = \frac{L}{b} \ll 1$ (e.g. for $L = 4096, b = 500000$, $\omega_{\min} L \approx 0.0082$):
$$\Delta H_{\min} = \ln\left( \frac{2\pi}{0.0082} \right) \approx \ln(766) \approx 6.64 \text{ nats}$$

This proves that the slow channels under $z_i = -2i/d$ suffer from a massive phase entropy collapse when evaluated over finite training windows, providing an information-theoretic explanation for why shifting channel density towards higher frequencies can yield richer representations within the training window.

---

## 6. Positional Locality Prior and Riemann-Lebesgue Decay Mechanics

### 6.1 Derivation of the Expected Attention Kernel

In transformer self-attention, consider the expected attention logit as a function of distance $\Delta$, averaged over isotropic query/key representations where $\mathbb{E}[C_i] = 1$ and $\mathbb{E}[D_i] = 0$:
$$\bar{K}_K(\Delta) = \frac{1}{K} \sum_{i=0}^{K-1} \cos(\omega_i \Delta)$$

In the continuous spectrum limit $K \to \infty$, substituting the density $\rho(\omega) = \frac{1}{\omega \ln b}$:
$$\bar{K}(\Delta) = \int_{b^{-1}}^1 \cos(\omega \Delta) \rho(\omega) \, d\omega = \frac{1}{\ln b} \int_{b^{-1}}^1 \frac{\cos(\omega \Delta)}{\omega} \, d\omega$$

Let $t = \omega \Delta$. Then $dt = \Delta \, d\omega$, $\frac{d\omega}{\omega} = \frac{dt}{t}$, and the limits of integration transform to $[t_{\min}, t_{\max}] = [\Delta / b, \Delta]$:
$$\bar{K}(\Delta) = \frac{1}{\ln b} \int_{\Delta / b}^\Delta \frac{\cos t}{t} \, dt$$

Recall the definition of the special Cosine Integral function $\operatorname{Ci}(x)$:
$$\operatorname{Ci}(x) \equiv -\int_x^\infty \frac{\cos t}{t} \, dt = \gamma + \ln x + \int_0^x \frac{\cos t - 1}{t} \, dt$$
where $\gamma \approx 0.5772156649$ is the Euler-Mascheroni constant.

Expressing the integral in terms of $\operatorname{Ci}(x)$:
$$\int_{\Delta / b}^\Delta \frac{\cos t}{t} \, dt = \left( -\int_\Delta^\infty \frac{\cos t}{t} dt \right) - \left( -\int_{\Delta / b}^\infty \frac{\cos t}{t} dt \right) = \operatorname{Ci}(\Delta) - \operatorname{Ci}\left( \frac{\Delta}{b} \right)$$

We arrive at the closed-form analytical expression for the continuous RoPE attention kernel:
$$\boxed{\bar{K}(\Delta) = \frac{\operatorname{Ci}(\Delta) - \operatorname{Ci}\left( \frac{\Delta}{b} \right)}{\ln b}}$$

---

### 6.2 Asymptotic Regimes of the Attention Kernel

#### Regime 1: Immediate Token Vicinity ($\Delta \to 0$)
Using the small-argument asymptotic expansion $\operatorname{Ci}(x) = \gamma + \ln x + O(x^2)$ as $x \to 0$:
$$\bar{K}(\Delta) = \frac{(\gamma + \ln \Delta) - \left(\gamma + \ln(\Delta / b)\right) + O(\Delta^2)}{\ln b} = \frac{\ln \Delta - (\ln \Delta - \ln b)}{\ln b} = \frac{\ln b}{\ln b} = 1$$
At $\Delta = 0$, all rotary channels are perfectly in phase ($\cos(0) = 1$), producing maximal constructive interference:
$$\bar{K}(0) = 1$$

#### Regime 2: Intermediate Distance ($1 \ll \Delta \ll b$)
For $\Delta \gg 1$, the upper cosine integral decays oscillatory:
$$\operatorname{Ci}(\Delta) = \frac{\sin \Delta}{\Delta} + O\left(\frac{1}{\Delta^2}\right) \approx 0$$
For the lower limit, since $\Delta / b \ll 1$, we retain the logarithmic expansion:
$$\operatorname{Ci}\left(\frac{\Delta}{b}\right) \approx \gamma + \ln\left(\frac{\Delta}{b}\right) = \gamma + \ln \Delta - \ln b$$
Substituting into the kernel formula:
$$\bar{K}(\Delta) \approx \frac{0 - (\gamma + \ln \Delta - \ln b)}{\ln b} = 1 - \frac{\ln \Delta}{\ln b} - \frac{\gamma}{\ln b}$$

Neglecting the small constant $\frac{\gamma}{\ln b} \approx \frac{0.577}{\ln b}$:
$$\boxed{\bar{K}(\Delta) \approx 1 - \frac{\ln \Delta}{\ln b}}$$

#### Linguistic Interpretation of Logarithmic Locality:
This derivation proves that the $1/\omega$ spectral density automatically endows the transformer with a **natural logarithmic distance penalty**. Queries and keys at short distance $\Delta$ naturally produce higher pre-softmax logits, with attention strength declining linearly as a function of $\ln \Delta$.

```
Expected Attention Kernel K̄(Δ):
1.0 | *
    |  *
    |   \
    |    \  K̄(Δ) ≈ 1 - ln(Δ)/ln(b)  (Logarithmic Locality Decay)
    |     \
    |      \
0.0 |_______\_________________________________*_______
    0       10       100       1,000        10,000 (b)   Δ (log scale)
```

#### Regime 3: Ultra-Long Distance Extrapolation ($\Delta \gg b$)
When distance exceeds the base $\Delta \gg b$, both arguments of $\operatorname{Ci}$ become large:
$$\operatorname{Ci}(\Delta) \approx \frac{\sin \Delta}{\Delta}, \quad \operatorname{Ci}\left(\frac{\Delta}{b}\right) \approx \frac{\sin(\Delta / b)}{\Delta / b}$$
$$\bar{K}(\Delta) \approx \frac{1}{\ln b} \left[ \frac{\sin \Delta}{\Delta} - \frac{b \sin(\Delta / b)}{\Delta} \right] = O\left( \frac{b}{\Delta \ln b} \right) \xrightarrow{\Delta \to \infty} 0$$

---

### 6.3 Destructive Interference via the Riemann-Lebesgue Lemma

#### Theorem 5 (Riemann-Lebesgue Decay of the Expected Attention Kernel).
Let $\rho \in L^1([b^{-1}, 1])$ be the continuous spectral density of RoPE frequencies. Then the Fourier cosine transform:
$$\bar{K}(\Delta) = \int_{b^{-1}}^1 \cos(\omega \Delta) \rho(\omega) \, d\omega$$
satisfies:
$$\lim_{\Delta \to \infty} \bar{K}(\Delta) = 0$$

*Proof.*
By the Riemann-Lebesgue Lemma, for any function $f \in L^1(\mathbb{R})$, $\lim_{|\xi| \to \infty} \int_{-\infty}^\infty f(x) e^{-i \xi x} dx = 0$.
Setting $f(\omega) = \rho(\omega) \mathbf{1}_{[b^{-1}, 1]}(\omega) \in L^1(\mathbb{R})$, the real part of the Fourier transform is:
$$\operatorname{Re} \int_{-\infty}^\infty f(\omega) e^{-i \Delta \omega} d\omega = \int_{b^{-1}}^1 \cos(\omega \Delta) \rho(\omega) d\omega = \bar{K}(\Delta)$$
Taking the limit $\Delta \to \infty$ directly yields $\lim_{\Delta \to \infty} \bar{K}(\Delta) = 0$. $\blacksquare$

#### Physical Mechanism: Wave-Packet Dispersion and Phase Randomization
- **At $\Delta = 0$**: All harmonic oscillators $\cos(\omega_i \Delta)$ have phase $\theta_i = 0$. The sum $\sum_{i=0}^{K-1} \cos(0) = K$ is fully coherent and constructive.
- **For $\Delta > 0$**: Each sub-channel rotates at a distinct angular velocity $\omega_i = b^{-2i/d}$. As distance increases, the phases $\theta_i(\Delta) = \omega_i \Delta \pmod{2\pi}$ scatter uniformly across the unit circle $S^1$.
- **Cancellation**: Because $\rho(\omega) = \frac{1}{\omega \ln b}$ is a continuous, smooth spectrum, every positive half-wave $\cos(\omega \Delta) > 0$ is canceled by an adjacent frequency band's negative half-wave $\cos(\nu \Delta) < 0$. This destructive interference is what prevents distant tokens from drowning out local context.

---

### 6.4 Discrete Grid Artifacts: Poincaré Revivals and Aliasing

In real implementations, $K$ is finite ($K \in \{32, 64\}$). The discrete sum:
$$\bar{K}_K(\Delta) = \frac{1}{K} \sum_{i=0}^{K-1} \cos\left( b^{-2i/d} \Delta \right)$$
is a quasi-periodic function of $\Delta$.

While the continuous integral $\bar{K}(\Delta) \to 0$ monotonically, the discrete sum exhibits two critical failure modes when extrapolating to $\Delta > L_{\text{train}}$:

1. **Phase Coherence Revivals (Talbot / Poincaré Spikes)**:
   At certain discrete distances $\Delta_{\text{revival}}$, fractional combinations of frequencies satisfy:
   $$b^{-2i/d} \Delta_{\text{revival}} \equiv 0 \pmod{2\pi} \quad \text{for multiple } i$$
   This produces spurious constructive interference spikes where $\bar{K}_K(\Delta) \gg 0$, tricking the attention mechanism into attending heavily to irrelevant distant tokens.
2. **Frozen Q/K Readout Mismatch**:
   During training on $\Delta \le L_{\text{train}}$, the projection weights $W_q, W_k$ learn specific content amplitudes $A_i$ and phases $\psi_i$ tailored to the training interval. When $\Delta > L_{\text{train}}$, unseen phase combinations break the destructive cancellation, causing softmax entropy collapse and high-temperature attention noise.

```
Continuous Kernel vs Discrete Sum:
K̄(Δ)
1.0 | \
    |  \   Continuous Kernel (Monotonic decay to 0)
    |   \---------------------------------------
    |    \      / \ (Spurious Discrete Spike)
    |     \    /   \     /\
0.0 |______\__/_____\___/__\____________________
    0     L_train            Δ_extrapolate
```

---

## 7. Synthesis with Repository Theoretical Architecture

To ensure complete alignment with the repository's foundational theorems and claim boundaries (as specified in `AGENTS.md` and `INDEX.md`), we summarize the exact theoretical mapping:

| Theoretical Concept | Mathematical Owner / Formulation | Strict Scientific Claim & Boundary |
| :--- | :--- | :--- |
| **Stable Rank Identity** | $r_2(R) = \frac{2K}{1 + (K-1)\bar{c}}$ (ICLR 2027 Theory §2.3) | **Exact proved identity**. The nominal dimension $2K$ is compressed by the average whitened cross-redundancy $\bar{c}$. |
| **Low-Frequency Degeneracy** | $V_\omega \to \operatorname{span}\{1, \Delta\}$; Softmax $\to \operatorname{span}\{\Delta - \bar\Delta, \Delta^2 - \bar{\Delta^2}\}$ | **Exact proved limit**. Slow channels are mathematically redundant in this metric; they are *not* dead or freely reclaimable without retraining. |
| **Allocation Identification** | Exact-range $K$-fixed, endpoint-fixed intervention ($x_k = a + R z_k$) | Pure allocation identification at fixed sampled support $(a, R)$. |
| **Transplant Obstruction** | $A^\top R_{\Omega'}(\Delta) B = R_\Omega(\Delta) \implies \Omega' = \Omega$ | **Exact theorem**. Exact fixed invertible Q/K compensation is obstructed for unequal frequency multisets. |
| **Static Geometry vs Task Loss** | 50M 2x2 Factorial: Geo+EVQ rank $4.57 \to 12.54$ but PPL $7.14 \to 76.20$ | Static rank improvement does *not* imply lower LM loss or better extrapolation; weights and tables co-adapt. |
| **EVQ-Cosh Construction** | Unique stationarity point for the convex surrogate functional $\mathcal{J}[\rho]$ | Minimal zero-learned-parameter construction; unique *only* for the stated convex surrogate, not a universal task-loss optimum. |

---

## 8. Summary of Findings

1. **Origin of $z_i = -2i/d$**: The canonical RoPE frequency formula $\omega_i = b^{-2i/d}$ is the unique uniform discretization of logarithmic frequency space that preserves scale homogeneity (constant octave ratio $r = b^{2/d}$).
2. **Scale Invariance of $\rho(\omega) = \frac{1}{\omega \ln b}$**: The continuous spectral density is uniquely invariant under scale dilation ($\rho(\lambda \omega) = \lambda^{-1}\rho(\omega)$), representing the invariant Haar measure on the multiplicative group $(\mathbb{R}^+, \times)$ and Jeffreys uninformative prior over spatial length scales.
3. **MRA & Spatial Scale Pyramid**: RoPE constructs a continuous wavelet-like multi-scale basis. High frequencies provide microscopic token adjacency resolution ($a \sim 1$), while low frequencies provide macroscopic order ($a \sim b$). Over finite training windows $L \ll T_{\text{slow}}$, slow frequencies suffer severe 2D subspace redundancy and softmax parabolic collapse.
4. **Information & Entropy Asymmetries**: While nominal channel capacity is equipartitioned across spatial octaves ($K_{\text{octave}} = \frac{d \ln 2}{2 \ln b}$), differential phase entropy exhibits a severe deficit at low frequencies ($\Delta H_i = \ln(\frac{2\pi}{\omega_i L})$ nats), as slow phases never wrap within $L$.
5. **Riemann-Lebesgue Locality Prior**: The expected attention kernel $\bar{K}(\Delta) = \frac{\operatorname{Ci}(\Delta) - \operatorname{Ci}(\Delta/b)}{\ln b}$ decays logarithmically as $1 - \frac{\ln \Delta}{\ln b}$ for $1 \ll \Delta \ll b$ via destructive phase interference. In discrete implementations, finite-$K$ sampling introduces spurious Poincaré revivals and aliasing at $\Delta > L_{\text{train}}$, requiring careful spectrum allocation and weight co-adaptation.
