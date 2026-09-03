# First-Principles Theoretical Derivation and Audit of Zero-Training RoPE Retrofit

**Author:** First-Principles Theorist (R4)  
**Date:** 2026-09-02  
**Working Directory:** `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_critic_r4_firstprinciples_1`  
**Formatting Contract:** All substantive assertions are strictly tagged with `[OBSERVED]`, `[DERIVED]`, `[HYPOTHESIS]`, or `[UNKNOWN]`. For every `[DERIVED]` statement, explicit assumptions and the shortest necessary algebraic steps are shown. For every `[OBSERVED]` statement, exact repository file paths and numbers are cited.

---

## 1. Executive Summary & Mathematical Setup

### 1.1 Notation and Mathematical Frame
Let $h_i, h_j \in \mathbb{R}^{d_{model}}$ be activation vectors at sequence positions $i, j \in \mathbb{N}_0$, with non-negative lag $\Delta = i - j \ge 0$.
Let $W_q, W_k \in \mathbb{R}^{d \times d_{model}}$ be learned query and key projection matrices for an attention head of dimension $d = 2K$, where $K \in \{16, 32, 64, 128\}$.
The un-rotated query and key vectors are:
$$q_i = W_q h_i \in \mathbb{R}^d, \quad k_j = W_k h_j \in \mathbb{R}^d.$$

The Rotary Position Embedding (RoPE) operator $R(\Delta) \in \mathrm{SO}(d)$ is a block-diagonal orthogonal matrix:
$$R(\Delta) = \bigoplus_{k=0}^{K-1} R_k(\omega_k \Delta), \quad R_k(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix},$$
parameterized by a frequency vector $\Omega_0 = (\omega_0, \omega_1, \dots, \omega_{K-1})^\top \in \mathbb{R}^K_{>0}$.
In standard transformers, $\omega_k = b^{-k/K}$ with base $b > 1$ (e.g., $b = 10^4, 5\times 10^5, 10^6$).

The pre-softmax attention logit (score) between query $i$ and key $j$ at relative displacement $\Delta$ is:
$$s_{ij}(\Delta) = q_i^\top R(\Delta) k_j.$$

---

## 2. Independent First-Principles Derivation of Attention Logit $s_{ij}(\Delta)$

### 2.1 Block-wise Complex Factorization
`[DERIVED]`  
*Assumptions:* Standard RoPE definition where $R(\Delta)$ is block-diagonal with $2 \times 2$ rotation blocks $R_k(\omega_k \Delta)$ acting on pairs of coordinates $(2k, 2k+1)$ of $q_i, k_j$.  
*Steps:*
1. Decompose the head vectors into $K$ two-dimensional coordinate pairs:
   $$q_i = \bigoplus_{k=0}^{K-1} \begin{pmatrix} q_{i, 2k} \\ q_{i, 2k+1} \end{pmatrix}, \quad k_j = \bigoplus_{k=0}^{K-1} \begin{pmatrix} k_{j, 2k} \\ k_{j, 2k+1} \end{pmatrix}.$$
2. The inner product decomposes as the sum over orthogonal 2D subspaces:
   $$s_{ij}(\Delta) = q_i^\top R(\Delta) k_j = \sum_{k=0}^{K-1} \begin{pmatrix} q_{i, 2k} & q_{i, 2k+1} \end{pmatrix} \begin{pmatrix} \cos(\omega_k \Delta) & -\sin(\omega_k \Delta) \\ \sin(\omega_k \Delta) & \cos(\omega_k \Delta) \end{pmatrix} \begin{pmatrix} k_{j, 2k} \\ k_{j, 2k+1} \end{pmatrix}.$$
3. Expanding each 2D term:
   $$\begin{aligned}
   s_{ij}^{(k)}(\Delta) &= (q_{i, 2k} k_{j, 2k} + q_{i, 2k+1} k_{j, 2k+1}) \cos(\omega_k \Delta) + (q_{i, 2k} k_{j, 2k+1} - q_{i, 2k+1} k_{j, 2k}) \sin(\omega_k \Delta).
   \end{aligned}$$
4. Define complex scalar representations of the 2D projected query and key:
   $$z_k^q \equiv q_{i, 2k} + i q_{i, 2k+1} \in \mathbb{C}, \quad z_k^k \equiv k_{j, 2k} + i k_{j, 2k+1} \in \mathbb{C}.$$
5. Under rotation by angle $\theta_k = \omega_k \Delta$, the complex key rotates as $z_k^k \mapsto z_k^k e^{i \omega_k \Delta}$.
6. The real Euclidean inner product between the rotated 2D vectors is identically the real part of the Hermitian product:
   $$\begin{aligned}
   \operatorname{Re}\left[ (z_k^q)^* \cdot (z_k^k e^{i \omega_k \Delta}) \right] &= \operatorname{Re}\left[ (q_{i, 2k} - i q_{i, 2k+1}) (k_{j, 2k} + i k_{j, 2k+1}) e^{i \omega_k \Delta} \right] \\
   &= \operatorname{Re}\left[ c_k e^{i \omega_k \Delta} \right],
   \end{aligned}$$
   where the complex coefficient $c_k \in \mathbb{C}$ is defined as:
   $$c_k \equiv (z_k^q)^* z_k^k = (q_{i, 2k} - i q_{i, 2k+1})(k_{j, 2k} + i k_{j, 2k+1}).$$
7. Summing over all $K$ rotary pairs yields the exact score decomposition:
   $$s_{ij}(\Delta) = \sum_{k=0}^{K-1} \operatorname{Re}\left[ c_k e^{i \omega_k \Delta} \right]. \quad \blacksquare$$

### 2.2 Polar Representation and Wave Interference Form
`[DERIVED]`  
*Assumptions:* Representation of $c_k$ in polar coordinates $c_k = A_k e^{i \psi_k}$ with amplitude $A_k \ge 0$ and phase $\psi_k \in (-\pi, \pi]$.  
*Steps:*
1. Express $z_k^q = |z_k^q| e^{i \phi_k^q}$ and $z_k^k = |z_k^k| e^{i \phi_k^k}$.
2. Then $c_k = |z_k^q| |z_k^k| e^{i (\phi_k^k - \phi_k^q)}$.
3. Thus $A_k = |z_k^q| |z_k^k| = \sqrt{q_{i, 2k}^2 + q_{i, 2k+1}^2} \sqrt{k_{j, 2k}^2 + k_{j, 2k+1}^2}$, and $\psi_k = \phi_k^k - \phi_k^q$.
4. Substituting into $s_{ij}(\Delta)$:
   $$s_{ij}(\Delta) = \sum_{k=0}^{K-1} A_k \cos(\omega_k \Delta + \psi_k). \quad \blacksquare$$
*Physical interpretation:* The attention score is a finite sum of $K$ discrete harmonic carriers $\cos(\omega_k \Delta)$ with spatial frequencies $\omega_k$, modulated by content-dependent amplitudes $A_k$ and phase offsets $\psi_k$ generated by the upstream network.

### 2.3 Exact Algebraic Invariants of $s_{ij}(\Delta)$
`[DERIVED]`  
*Assumptions:* Exact score formula $s_{ij}(\Delta) = \sum_{k=0}^{K-1} \operatorname{Re}[c_k e^{i \omega_k \Delta}]$.  
*Steps:*
1. **Zero-Lag Invariant:** At $\Delta = 0$:
   $$s_{ij}(0) = \sum_{k=0}^{K-1} \operatorname{Re}[c_k] = \sum_{k=0}^{K-1} (q_{i, 2k} k_{j, 2k} + q_{i, 2k+1} k_{j, 2k+1}) = q_i^\top k_j.$$
   $s_{ij}(0)$ is strictly invariant under any modification of the frequency table $\Omega$.
2. **Mean-Power Identity:** For pairwise distinct frequencies $\omega_k \ne \omega_l$ ($\forall k \ne l$):
   $$\lim_{T \to \infty} \frac{1}{T} \int_0^T |s_{ij}(\Delta)|^2 d\Delta = \sum_{k, l} \operatorname{Re}\left[ c_k c_l^* \lim_{T \to \infty} \frac{1}{T} \int_0^T e^{i (\omega_k - \omega_l) \Delta} d\Delta \right] = \frac{1}{2} \sum_{k=0}^{K-1} |c_k|^2.$$
   The asymptotic mean power depends strictly on the channel magnitudes $\{|c_k|\}$ and is independent of the frequencies $\{\omega_k\}$ and slot permutations, provided all frequencies remain distinct.
   *Slot Merging Cavity:* If two slots merge ($\omega_k = \omega_l = \omega^*$), the power on that harmonic becomes $\frac{1}{2}|c_k + c_l|^2 = \frac{1}{2}(|c_k|^2 + |c_l|^2 + 2\operatorname{Re}[c_k c_l^*])$, which shifts mean power by $\operatorname{Re}[c_k c_l^*] \in [-|c_k||c_l|, +|c_k||c_l|]$.
3. **Analytic Real-Injectivity:** The function $\Delta \mapsto s_{ij}(\Delta)$ is an entire real-analytic function on $\mathbb{R}$. By the identity theorem for holomorphic functions, knowledge of $s_{ij}(\Delta)$ on any open non-empty interval $I \subset \mathbb{R}$ uniquely determines $s_{ij}(\Delta)$ on all of $\mathbb{R}$.
4. **Fourier Uniqueness:** For distinct positive frequencies $\{\omega_k\}$, the mapping from the discrete complex spectral measure $\nu = \sum_{k=0}^{K-1} c_k \delta_{\omega_k}$ to the function $s_{ij}(\cdot)$ is injective. Given $s_{ij}(\cdot)$, the pairs $(c_k, \omega_k)$ are uniquely recoverable (via Prony's method or harmonic analysis). $\blacksquare$

---

## 3. Incorporation of Realistic Model Properties

### 3.1 Learned Projections and Checkpoint Co-adaptation
`[DERIVED]`  
*Assumptions:* During pretraining on inputs of length $L \le L_{\text{train}}$, weights $W_q, W_k$ are optimized via stochastic gradient descent against task loss with a fixed native frequency table $\Omega_0 = \{\omega_k\}$.  
*Deduction:*
1. The complex coefficients $c_k = (W_q^{(k)} h_i)^* (W_k^{(k)} h_j)$ are outputs of learned linear maps $W_q^{(k)}, W_k^{(k)} \in \mathbb{R}^{2 \times d_{model}}$.
2. The gradient of the loss $\mathcal{L}$ with respect to the $k$-th subspace projection depends explicitly on $\omega_k$:
   $$\frac{\partial \mathcal{L}}{\partial W_q^{(k)}} = \sum_j \frac{\partial \mathcal{L}}{\partial s_{ij}} \left[ \cos(\omega_k \Delta) W_k^{(k)} h_j + \sin(\omega_k \Delta) J_2 W_k^{(k)} h_j \right] h_i^\top, \quad J_2 = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}.$$
3. Consequently, the subspace weights $W_q^{(k)}, W_k^{(k)}$ are strongly co-adapted to the specific carrier frequency $\omega_k$. They learn representations $h$ whose phases $\psi_k$ and magnitudes $A_k$ constructively interfere at semantically relevant lags $\Delta$ and destructively interfere at distractor lags.
4. `[OBSERVED]` In `paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` §5.2–5.3, a $2 \times 2$ factorial crossing on a 50M model (`GG`, `GE`, `EG`, `EE` across Geometric and EVQ tables/weights) proves that the table $\times$ weights interaction effect on LM loss is $I_{T \times W} = -3.5367$ ($95\%$ CI $[-5.165, -3.039]$), which is $5.9\times$ larger in magnitude than the table main effect ($+0.5991$) and weights main effect ($-0.5965$). Post-hoc mismatch causes perplexity to explode from $7.14$ (`GG`) to $76.20$ (`GE`).

### 3.2 Ordered Rotational Subspaces vs. Unordered Spectrum
`[DERIVED]`  
*Assumptions:* Let $\pi \in S_K$ be a non-trivial permutation of the $K$ rotary slots. Apply $\pi$ to the frequency table such that slot $k$ receives $\omega_{\pi(k)}$. Weights $W_q, W_k$ remain frozen.  
*Deduction:*
1. The new score is $s'_{ij}(\Delta) = \sum_{k=0}^{K-1} \operatorname{Re}[c_k e^{i \omega_{\pi(k)} \Delta}]$.
2. Although the un-ordered multiset of frequencies is identical ($\{\omega_{\pi(k)}\} = \{\omega_k\}$), the coefficient $c_k$ (trained to pair with $\omega_k$) is now paired with $\omega_{\pi(k)}$.
3. For $k \ne \pi(k)$, the phase error at lag $\Delta$ is $\delta\theta_k(\Delta) = (\omega_{\pi(k)} - \omega_k)\Delta$.
4. For high or mid-frequency channels where $|\omega_{\pi(k)} - \omega_k| \gg 2\pi/L$, the term $\cos(\omega_{\pi(k)} \Delta + \psi_k)$ becomes completely de-correlated from the intended target $\cos(\omega_k \Delta + \psi_k)$, replacing constructive attention with quasi-random noise of variance $\frac{1}{2}\sum_k |c_k|^2$.
5. `[OBSERVED]` In `paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` §2 and `INDEX.md` §0, applying a same-multiset permutation to frozen mature checkpoints collapses functional behavior:
   - OLMo 1x PG-19 NLL degrades from $3.10423 \to 6.86493$.
   - Qwen 64K core-4 score drops catastrophically from $0.7000 \to 0.0000$.
   - Hence, the learned positional object is the **ordered sequence of tuples** $((W_q^{(k)}, W_k^{(k)}), \omega_k)_{k=0}^{K-1}$, not the unordered spectrum.

### 3.3 Finite Head Dimension and Discrete Frequency Lattice
`[DERIVED]`  
*Assumptions:* Finite head dimension $d = 2K \in \{32, 64, 128, 256\}$, meaning $K \in \{16, 32, 64, 128\}$. Frequencies follow $\omega_k = b^{-k/K}$ with $k \in \{0, \dots, K-1\}$.  
*Deduction:*
1. The number of cycles completed by channel $k$ in the native window $[0, L]$ is:
   $$C_k = \frac{\omega_k L}{2\pi} = \frac{L}{2\pi} b^{-k/K}.$$
2. Fast channels ($k \approx 0$): $C_0 = L / (2\pi) \gg 1$. For $L = 4096$, $C_0 \approx 652$ cycles. Phase wraps rapidly.
3. Slow channels ($k \approx K-1$): For $b = 5\times 10^5$ and $L = 4096$, $\omega_{K-1} = 2\times 10^{-6}$, so $C_{K-1} \approx 0.0013$ cycles. Phase does not complete even $1\%$ of a single cycle across the entire native window.
4. Low-Frequency Subspace Collapse:
   `[OBSERVED]` In `paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` §3.1–3.2, as $\omega_k \Delta \to 0$:
   $$\cos(\omega_k \Delta) = 1 - \frac{\omega_k^2 \Delta^2}{2} + O(\omega_k^4 \Delta^4), \quad \frac{\sin(\omega_k \Delta)}{\omega_k} = \Delta - \frac{\omega_k^2 \Delta^3}{6} + O(\omega_k^4 \Delta^4).$$
   All slow channels converge to the identical two-dimensional subspace $\operatorname{span}\{1, \Delta\}$.
   For $K=64$ ($L=4096, b=5\times 10^5$), the 24 slowest pairs (nominal dimension 48) have a block-whitened stable rank of only $r_2 = 2.0002$ (a $95.83\%$ stable dimension loss).
   In the attention softmax Fisher metric ($F_{\mathrm{sm}} = \operatorname{diag}(p) - p p^\top$), the constant direction $\mathbf{1}$ is eliminated, collapsing the slow bands into centered polynomials $\operatorname{span}\{\Delta - \mathbb{E}_p\Delta, \Delta^2 - \mathbb{E}_p\Delta^2\}$.

---

## 4. The Fundamental Question: What Object is Controlled vs. Disrupted?

When altering the frequency table $\Omega \to \Omega'$ ($\omega_k \mapsto \omega'_k = \rho_k \omega_k$) in a frozen mature checkpoint:

### 4.1 What Physical, Spectral, and Geometric Object is Actually Being Controlled?
`[DERIVED]`  
*Assumptions:* Frequency transformation $\omega_k \to \omega'_k = \rho_k \omega_k$ with dilation factors $\rho_k \in (0, 1]$.  
*Deduction:*
1. **Phase Velocity Vector:** The velocity of rotation in each 2D rotary plane with respect to token index is $d\theta_k/d\Delta = \omega'_k$. Controlling $\Omega'$ directly sets the speed at which the joint phase vector $\Phi(\Delta) = (\omega'_0 \Delta, \dots, \omega'_{K-1} \Delta)^\top$ winds around the $K$-dimensional torus $\mathbb{T}^K$.
2. **Channel-Specific Lag Distortion (Warping):**
   $$s'_{ij}(\Delta) = \sum_{k=0}^{K-1} \operatorname{Re}\left[ c_k e^{i \omega'_k \Delta} \right] = \sum_{k=0}^{K-1} \operatorname{Re}\left[ c_k e^{i \omega_k (\rho_k \Delta)} \right].$$
   The retrofit controls an independent coordinate compression $\Delta_k^{\mathrm{eff}} = \rho_k \Delta$ per channel.
3. **Novelty vs. Blur Waterbed Trade-off:**
   `[DERIVED]` (from definition $\omega'_k = \omega_k S^{-m_k}$ where $S = L_{\mathrm{ext}}/L$):
   The deployed phase range relative to the native training range is:
   $$r_k(S) \equiv \frac{\omega'_k S L}{\omega_k L} = \rho_k S = S^{1 - m_k}.$$
   - If $m_k = 1$ ($\rho_k = 1/S$, Position Interpolation): $r_k(S) = 1$. The maximum phase reached at $S L$ is exactly the native phase at $L$. Channel $k$ encounters **zero phase novelty**. However, inside the native window $\Delta \in [0, L]$, the phase is compressed by $S$, causing **resolution blur** (high frequencies shifted to lower bands).
   - If $m_k = 0$ ($\rho_k = 1$, No Interpolation / Extrapolation): $r_k(S) = S$. Native resolution is perfectly preserved ($s'_{ij}(\Delta) = s_{ij}(\Delta)$ for all $\Delta$), but at $\Delta > L$ the channel encounters **unseen phase novelty** up to $S \times$ the native maximum.
   - For non-linear profiles ($m_k \in (0, 1)$), one controls the **spectral boundary** separating which channels act as pure interpolators vs. pure extrapolators.

### 4.2 What is Inevitably Disrupted?
`[DERIVED]`  
*Deduction:*
1. **The Co-adapted Fourier Reconstruction of the Attention Kernel (Compatibility Modulus):**
   The attention logit perturbation for any content realization $c$ is:
   $$|s'_{ij}(\Delta) - s_{ij}(\Delta)| = \left| \sum_{k=0}^{K-1} \operatorname{Re}\left[ c_k (e^{i \omega'_k \Delta} - e^{i \omega_k \Delta}) \right] \right| \le \sum_{k=0}^{K-1} |c_k| \min\left(2, |\omega'_k - \omega_k| \Delta\right).$$
   For fast channels (where $\omega_k$ is large), shifting $\omega_k \to \rho_k \omega_k$ causes $|\omega'_k - \omega_k|\Delta = (1 - \rho_k)\omega_k \Delta$ to reach $\pi$ at extremely short distances:
   $$\Delta_k^{\mathrm{flip}} = \frac{\pi}{(1 - \rho_k) \omega_k}.$$
   For $k=0$ ($\omega_0 \approx 1$), if $\rho_0 = 0.25$ ($S=4$ PI), $\Delta_0^{\mathrm{flip}} \approx \pi / 0.75 \approx 4$ tokens!
   At lag $\Delta = 4$, the fast channel's contribution completely inverts sign ($\cos \to -\cos$), destroying local syntax, n-gram matching, and induction heads.
2. **Off-Arc Torus Exposure (The Non-Uniform Torus Geometric Disruption):**
   `[DERIVED]`  
   Let $\gamma(\Delta) = (\omega_0 \Delta, \dots, \omega_{K-1} \Delta) \pmod{2\pi\mathbb{Z}^K}$ be the native phase trajectory in $\mathbb{T}^K$ for $\Delta \in [0, L]$.  
   Let $\gamma'(\Delta) = (\omega'_0 \Delta, \dots, \omega'_{K-1} \Delta) \pmod{2\pi\mathbb{Z}^K}$ be the deployed phase trajectory for $\Delta \in [0, SL]$.  
   - Under uniform PI ($\rho_k \equiv 1/S$): $\gamma'(\Delta) = \gamma(\Delta / S)$. The image $\gamma'([0, SL]) = \gamma([0, L])$. No new points on $\mathbb{T}^K$ are ever visited.
   - Under any non-uniform retrofit (where $\rho_k \ne \rho_l$ for some $k \ne l$): the direction vector $\Omega'$ is not collinear with $\Omega_0$. In $\mathbb{R}^K / 2\pi\mathbb{Z}^K$, the two one-parameter subgroups $\mathbb{R}\Omega_0$ and $\mathbb{R}\Omega'$ intersect only at the origin $\mathbf{0}$ (assuming incommensurate frequency ratios).  
   Therefore, for almost all $\Delta \in (L, SL]$, the joint phase configuration $\Phi'(\Delta)$ lies **completely outside the native manifold** $\gamma([0, L])$!  
   The frozen downstream layers receive attention weights formed by query-key phase interactions that never existed in the training data.
3. **Linear Q/K Compensability (Transplant Rigidity):**
   No position-independent linear transformation of queries and keys ($q \mapsto A q$, $k \mapsto B k$) can undo a non-trivial frequency modification (see Theorem 1 below).
4. **Conditioning Stability (Error Amplification by Factor $S$):**
   Any small mismatch or tolerance $\epsilon$ on the native range $[0, L]$ is amplified to at least $S \epsilon$ at the deployment horizon $SL$ (see Theorem 4 below).

---

## 5. Five Structural and Impossibility Theorems

### Theorem 1 (Transplant Rigidity: Impossibility of Fixed Linear Compensation)
`[DERIVED]`  
*Assumptions:* Let $R(\Omega\Delta)$ and $R(\Omega'\Delta)$ be the $d \times d$ block-diagonal RoPE rotation matrices for frequencies $\Omega$ and $\Omega'$. Suppose there exist content-independent, position-independent matrices $A, B \in \mathbb{R}^{d \times d}$ such that for all content vectors $q, k \in \mathbb{R}^d$ and all $\Delta \in \mathbb{R}$:
$$q^\top A^\top R(\Omega'\Delta) B k = q^\top R(\Omega\Delta) k.$$
*Claim:* The frequency multisets must be identical up to sign: $\{|\omega'_k|\}_{k=0}^{K-1} = \{|\omega_k|\}_{k=0}^{K-1}$.  
*Proof:*
1. The condition implies $A^\top R(\Omega'\Delta) B = R(\Omega\Delta)$ for all $\Delta \in \mathbb{R}$.
2. Evaluating at $\Delta = 0$: since $R(0) = I_d$, we have $A^\top B = I_d$, which implies $B = (A^\top)^{-1} = A^{-\top}$.
3. Substituting back: $A^\top R(\Omega'\Delta) A^{-\top} = R(\Omega\Delta)$ for all $\Delta \in \mathbb{R}$.
4. Differentiating with respect to $\Delta$ at $\Delta = 0$:
   $$A^\top \left( \left. \frac{d R(\Omega'\Delta)}{d\Delta} \right|_{\Delta=0} \right) A^{-\top} = \left. \frac{d R(\Omega\Delta)}{d\Delta} \right|_{\Delta=0}.$$
5. The derivative at zero is the block-diagonal skew-symmetric Lie algebra generator:
   $$J(\Omega) = \bigoplus_{k=0}^{K-1} \begin{pmatrix} 0 & -\omega_k \\ \omega_k & 0 \end{pmatrix}, \quad J(\Omega') = \bigoplus_{k=0}^{K-1} \begin{pmatrix} 0 & -\omega'_k \\ \omega'_k & 0 \end{pmatrix}.$$
6. Since $J(\Omega)$ and $J(\Omega')$ are similar ($J(\Omega) = A^\top J(\Omega') A^{-\top}$), they must share the exact same characteristic polynomial and eigenvalues.
7. The eigenvalues of $J(\Omega)$ are $\{\pm i \omega_k\}_{k=0}^{K-1}$, and those of $J(\Omega')$ are $\{\pm i \omega'_k\}_{k=0}^{K-1}$.
8. Therefore, the multisets of absolute frequencies must coincide: $\{|\omega'_k|\} = \{|\omega_k|\}$. $\blacksquare$  
*Implication:* `[OBSERVED]` In `rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`, this rigorously proves that post-hoc frequency modification cannot be compensated for by any linear adapter (e.g. LoRA on Q/K) across all positions.

---

### Theorem 2 (PI Uniqueness and Arc Containment)
`[DERIVED]`  
*Assumptions:* Let $\Omega = (\omega_0, \dots, \omega_{K-1})^\top$ with at least one pair of frequencies having an irrational ratio. The deployed frequencies are $\omega'_k = \rho_k \omega_k$ with $\rho_k \in (0, 1]$. Let $\gamma([0, L]) = \{ (\omega_k \Delta \bmod 2\pi)_{k=0}^{K-1} : \Delta \in [0, L] \} \subset \mathbb{T}^K$, and $\gamma'([0, SL]) = \{ (\rho_k \omega_k \Delta \bmod 2\pi)_{k=0}^{K-1} : \Delta \in [0, SL] \} \subset \mathbb{T}^K$.  
*Claim:* $\gamma'([0, SL]) \subseteq \gamma([0, L])$ as subsets of $\mathbb{T}^K$ if and only if $\rho_k \equiv \rho \le 1/S$ for all $k \in \{0, \dots, K-1\}$. Set equality $\gamma'([0, SL]) = \gamma([0, L])$ holds if and only if $\rho \equiv 1/S$ (pure Position Interpolation).  
*Proof:*
1. At $\Delta = 0$, both arcs start at $\mathbf{0}$. For small $\Delta > 0$, the tangent vectors in the covering space $\mathbb{R}^K$ are $\mathbf{v}' = (\rho_0 \omega_0, \dots, \rho_{K-1} \omega_{K-1})^\top$ and $\mathbf{v} = (\omega_0, \dots, \omega_{K-1})^\top$.
2. For the straight ray $\Delta \mathbf{v}'$ to be contained in the ray $\Delta \mathbf{v} \pmod{2\pi\mathbb{Z}^K}$ on a finite neighborhood of zero, the tangent ray must be collinear: $\mathbf{v}' = c \mathbf{v}$ for some scalar $c > 0$.
3. Collinearity forces $\rho_k \omega_k = c \omega_k \implies \rho_k = c$ for all $k$. Thus $\rho$ must be uniform across all channels.
4. The Euclidean arc length of $\gamma'([0, SL])$ is $\int_0^{SL} \|\mathbf{v}'\|_2 d\Delta = SL \rho \|\Omega\|_2$.
5. The Euclidean arc length of $\gamma([0, L])$ is $L \|\Omega\|_2$.
6. For $\gamma'([0, SL]) \subseteq \gamma([0, L])$, the length cannot exceed the container: $S L \rho \|\Omega\|_2 \le L \|\Omega\|_2 \implies \rho \le 1/S$.
7. For set equality $\gamma'([0, SL]) = \gamma([0, L])$, the lengths must match, forcing $\rho = 1/S$. $\blacksquare$  
*Implication:* Any non-uniform frequency scaling ($\rho_k \ne \rho_l$) **necessarily escapes the native phase manifold** and exposes the model to off-arc phase configurations.

---

### Theorem 3 (Compatibility Modulus and Weight-Blind Vacuity)
`[DERIVED]`  
*Assumptions:* Let $s(\Delta) = \sum_{k=0}^{K-1} \operatorname{Re}[c_k e^{i \omega_k \Delta}]$ and $s'(\Delta) = \sum_{k=0}^{K-1} \operatorname{Re}[c_k e^{i \omega'_k \Delta}]$ for fixed content coefficients $c_k$.  
*Claim:*
1. The score difference is tightly bounded by:
   $$|s'(\Delta) - s(\Delta)| \le \sum_{k=0}^{K-1} |c_k| \min\left(2, |\omega'_k - \omega_k| \Delta\right).$$
2. If $|s'(\Delta) - s(\Delta)| \le \epsilon$ uniformly across all keys in an attention row, the $L_1$ variation of attention probabilities satisfies:
   $$\|p' - p\|_1 \le e^{2\epsilon} - 1.$$
3. Any diagnostic functional relying solely on diagonal energies $\{|c_k|\}$ or $\{|c_k|^2\}$ is vacuous for bounding functional change.  
*Proof:*
1. By linearity: $s'(\Delta) - s(\Delta) = \sum_k \operatorname{Re}[c_k (e^{i \omega'_k \Delta} - e^{i \omega_k \Delta})]$.
   $|\operatorname{Re}[c_k (e^{i \omega'_k \Delta} - e^{i \omega_k \Delta})]| \le |c_k| |e^{i \omega'_k \Delta} - e^{i \omega_k \Delta}|$.
   We have $|e^{i\theta'} - e^{i\theta}| = |e^{i(\theta'-\theta)} - 1| = 2|\sin((\theta'-\theta)/2)| \le \min(2, |\theta' - \theta|)$.
   Setting $\theta' = \omega'_k \Delta$ and $\theta = \omega_k \Delta$ yields $|e^{i\omega'_k\Delta} - e^{i\omega_k\Delta}| \le \min(2, |\omega'_k - \omega_k|\Delta)$. Summing yields the modulus bound.
2. For softmax with logits shifted by $|\delta s_j| \le \epsilon$:
   $$\frac{p'_j}{p_j} = \frac{e^{s'_j}}{\sum_l e^{s'_l}} \frac{\sum_l e^{s_l}}{e^{s_j}} \le \frac{e^{s_j + \epsilon}}{\sum_l e^{s_l - \epsilon}} \frac{\sum_l e^{s_l}}{e^{s_j}} = e^{2\epsilon}.$$
   Similarly, $p'_j / p_j \ge e^{-2\epsilon}$.
   Then $\|p' - p\|_1 = \sum_j |p'_j - p_j| = \sum_j p_j |p'_j/p_j - 1| \le \max(e^{2\epsilon} - 1, 1 - e^{-2\epsilon}) = e^{2\epsilon} - 1$.
3. *Counterexample for Weight-Blind Vacuity:* Consider $K = 2$, $\Omega = (1, 2)^\top$.
   Let pair (a) have $c = (2, 1)$, and pair (b) have $c = (1, 2)$.
   Both share identical slot energies $\{1, 2\}$ and the identical frequency multiset $\{1, 2\}$.
   Yet at $\Delta = \pi$:
   - For (a): $s(\pi) = 2\cos(\pi) + 1\cos(2\pi) = -2 + 1 = -1$.
   - For (b): $s(\pi) = 1\cos(\pi) + 2\cos(2\pi) = -1 + 2 = +1$.
   The logit flips sign entirely! Thus, knowing $\{|c_k|\}$ without knowing the paired phase alignment $c_k \leftrightarrow \omega_k$ cannot even determine the sign of the attention logit. $\blacksquare$

---

### Theorem 4 (Conditioning and Ill-Posedness of Extrapolation)
`[DERIVED]`  
*Assumptions:* Let $\Omega'$ be a deployed table on $\Delta \in [0, SL]$. Consider an arbitrary perturbation $\delta\Omega' = (\delta\omega'_0, \dots, \delta\omega'_{K-1})^\top$. Let $\| \delta s \|_{[0, T]} \equiv \sup_{c : \|c\|_1 \le 1} \sup_{\Delta \in [0, T]} |\delta s(\Delta)|$.  
*Claim:* The condition number of score extrapolation from the native window $[0, L]$ to the deployment horizon $[0, SL]$ is exactly $S$:
$$\frac{\| \delta s \|_{[0, SL]}}{\| \delta s \|_{[0, L]}} = S.$$
*Proof:*
1. To first order in $\delta\omega'_k$:
   $$\delta s(\Delta) = \sum_{k=0}^{K-1} \operatorname{Re}\left[ c_k (i \delta\omega'_k \Delta) e^{i \omega'_k \Delta} \right] = -\Delta \sum_{k=0}^{K-1} \delta\omega'_k \operatorname{Im}[c_k e^{i \omega'_k \Delta}].$$
2. Taking the supremum over all content coefficients with $\sum_k |c_k| \le 1$:
   For any given $\Delta$, we can choose the phase of $c_k$ such that $\operatorname{Im}[c_k e^{i \omega'_k \Delta}] = -\operatorname{sgn}(\delta\omega'_k) |c_k|$.
   Then:
   $$\sup_{\|c\|_1 \le 1} |\delta s(\Delta)| = \Delta \max_{k} |\delta\omega'_k| = \Delta \|\delta\Omega'\|_\infty.$$
3. Taking the supremum over $\Delta \in [0, T]$:
   $$\| \delta s \|_{[0, T]} = \sup_{\Delta \in [0, T]} \Delta \|\delta\Omega'\|_\infty = T \|\delta\Omega'\|_\infty.$$
4. For $T = SL$ versus $T = L$:
   $$\frac{\| \delta s \|_{[0, SL]}}{\| \delta s \|_{[0, L]}} = \frac{SL \|\delta\Omega'\|_\infty}{L \|\delta\Omega'\|_\infty} = S. \quad \blacksquare$$
*Implication:* Extrapolation is an ill-posed problem whose error conditioning degrades linearly with scale factor $S$. Any approximation error on native length is magnified by $S$ at the target horizon.

---

### Theorem 5 (Multi-Level Non-Identifiability from Native Data)
`[DERIVED]`  
*Assumptions:* Admissible native design rules are functionals $F(\theta) = G(O_L(\theta), S)$, where $O_L(\theta)$ represents observable behavioral data of model $\theta$ on inputs of length $\le L$.  
*Claim:*
1. *(Table Level)* Exact native agreement of scores $s(\Delta)$ for all content across $[0, L]$ forces identical complex spectral measures $\nu$, and therefore identical tables $\Omega' = \Omega$. No non-trivial table change can be completely silent on $[0, L]$.
2. *(Checkpoint Level / Dormant Circuit)* For every $\epsilon > 0$ and $S > 1$, there exist checkpoint pairs $(\theta_1, \theta_2)$ whose output probability distributions differ by at most $\epsilon$ on all inputs of length $\le L$, but differ by $\Omega(1)$ on inputs of length $SL$.
3. *(Retrofit Non-Identifiability)* For any non-uniform retrofit ($\rho_k \ne \rho_l$), the network is queried on joint phase configurations $\Phi'(\Delta) \in \mathbb{T}^K$ for $\Delta \in (L, SL]$ that are completely disjoint from the native manifold $\gamma([0, L])$. The response of the frozen network to these off-arc queries is mathematically unconstrained by native-range observations.  
*Proof:*
1. Follows directly from Section 2.3: $s(\Delta)$ is real-analytic. Agreement on $[0, L]$ implies agreement on $\mathbb{R}$, which by Fourier uniqueness forces $\sum c_k \delta_{\omega'_k} = \sum c_k \delta_{\omega_k} \implies \Omega' = \Omega$.
2. *Constructive sketch:* Add an attention circuit whose score contribution is $h(\Delta) = \alpha [1 - \cos(\omega^* \Delta)]$, where $\omega^* = \pi / (SL)$.
   On the native range $\Delta \in [0, L]$, the maximum phase is $\omega^* L = \pi / S$.
   Using $1 - \cos\theta \le \theta^2 / 2$:
   $$\sup_{\Delta \in [0, L]} h(\Delta) \le \frac{\alpha \pi^2}{2 S^2}.$$
   Choosing $\alpha = O(1)$, this contribution is bounded by $O(S^{-2})$, which can be made arbitrarily smaller than $\epsilon$ by scaling down or absorbing into layer norms.
   However, at the deployment horizon $\Delta = SL$:
   $$h(SL) = \alpha [1 - \cos(\pi)] = 2\alpha = O(1).$$
   The circuit is dormant ($O(S^{-2})$) on native text, but fully active ($O(1)$) at scale $SL$.
3. For non-uniform $\Omega'$, $\gamma'((L, SL]) \cap \gamma([0, L]) = \emptyset$ (from Theorem 2). The network's activations and attention weights at off-arc points depend on the multi-dimensional interpolation properties of the feed-forward networks and value projections in regions of $\mathbb{R}^d$ never visited during pretraining. No functional evaluated solely on $[0, L]$ can identify this response. $\blacksquare$  
*Implication:* `[OBSERVED]` In `paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` §0.D and §H, this explains why two Native-derived tables (`log_s4` and `YaRN-4`) yield inverted performance on long tasks: `log_s4` wins RULER-16K ($0.49859$ vs $0.1056$) while `YaRN-4` wins HotpotQA-16K ($0.500$ vs $0.153$ F1). Native metrics cannot identify long-range task tolerance.

---

## 6. Falsifiable Mapping to Repository Empirical Evidence

The following table connects each theoretical derivation directly to canonical repository evidence:

| Theoretical Result | Mathematical Mechanism | Repository Empirical Fact | Canonical Owner |
| :--- | :--- | :--- | :--- |
| **Ordered Coupling** | Subspace weights $W_q^{(k)}, W_k^{(k)}$ co-adapt to specific $\omega_k$. | Permuting same frequency multiset across slots collapses OLMo ($3.10 \to 6.86$ NLL) and Qwen ($0.70 \to 0$ RULER). | `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md` §2 |
| **Transplant Rigidity** | Lie generator similarity forces $\{|\omega'_k|\} = \{|\omega_k|\}$; no fixed linear map can alter frequencies. | LoRA Q/K adaptation fails to preserve held-out native retrieval ($55\% \to 0\%$) when frequencies are altered. | `OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` §1–2 |
| **Table $\times$ Weights Interaction** | Mismatched tables destroy co-adapted destructive/constructive interference. | 50M $2\times 2$ crossing shows interaction effect $-3.5367$ is $5.9\times$ larger than main effects; PPL jumps $7.14 \to 76.20$. | `FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` §5.2 |
| **Compatibility Modulus Breakdown** | High-frequency compression causes $(1-\rho_k)\omega_k\Delta \approx \pi$ at $\Delta \le 4$, flipping attention signs. | Compressing fast channels damages 1x PG-19 retention gate; pure PI fails retention gate ($0.6588 < 0.875$). | `HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md` line ~175 |
| **Off-Arc Torus Exposure & Non-Identifiability** | Non-uniform $\Omega'$ generates phase tuples on $\mathbb{T}^K$ never seen in training; dormant circuits activate. | Native-only tables (`log_s4` vs `YaRN-4`) invert rankings on RULER-16K ($0.499$ vs $0.106$) vs HotpotQA-16K ($0.153$ vs $0.500$). | `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` §0.D |
| **Scale Super-Doubling** | Novelty volume $N_k(2S) > 2 N_k(S)$ diverges for fixed profile $m_k \in (0, 1)$. | Frozen `log_s4` table continued to $8\times$ fails to maintain $4\times$ capability ceiling. | `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` §D.6, §T7 |
| **Softmax Gain Thresholding** | Scalar gain $g$ alters sharpness, not argmax rank; below $g=1$, attention mass diffuses catastrophically. | Gain sweep shows $16\times$ collapse at $g=0.9$ (F1 $.588 \to .036$); free gain drops loss but destroys EOS generation. | `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` §0.E, §T8 |

---

## 7. Direct Answers to the Fundamental Audit Questions

### 7.1 What is the exact mathematical object being controlled when $\omega \to \omega'$?
`[DERIVED]`  
In a frozen checkpoint, modifying $\omega_k \to \omega'_k = \rho_k \omega_k$ controls:
1. The **phase velocity vector** $\Omega' \in \mathbb{R}^K$ governing the winding rates of the 1D trajectory on the torus $\mathbb{T}^K$.
2. The **per-channel effective coordinate scaling** $\Delta_k^{\mathrm{eff}} = \rho_k \Delta$, which reshapes the dispersion relation of the attention wave-packet.
3. The **allocation of novelty vs. blur budget**: $r_k(S) = S^{1 - m_k}$, dictating which channels preserve native resolution ($m_k = 0$) versus which channels avoid out-of-distribution phase angles ($m_k = 1$).

### 7.2 What is inevitably disrupted?
`[DERIVED]`  
1. **In-window constructive interference of fast channels:** If fast channels are compressed ($\rho_k < 1$), their phase error exceeds $\pi$ within a few tokens ($\Delta \sim 4$), destroying short-range syntax, local retrieval, and in-context learning.
2. **Joint phase distribution coherence (Off-arc exposure):** If channels are scaled non-uniformly, the deployed trajectory $\gamma'([0, SL])$ is strictly off the native manifold $\gamma([0, L])$ on $\mathbb{T}^K$. Downstream layers are forced to evaluate feed-forward and attention circuits on un-trained phase combinations.
3. **Linear Q/K compensability:** Theorem 1 proves that no fixed linear transformation can repair this frequency shift.
4. **Softmax decisiveness:** Un-adapted phase interference raises the floor of distractor logits, causing softmax entropy to collapse and triggering hallucination or premature EOS.

### 7.3 Why is there an inherent tension between Native Retention and Long Extension?
`[DERIVED]`  
- Native retention requires $|s'(\Delta) - s(\Delta)| \le \epsilon$ for all $\Delta \in [0, L]$. By Theorem 3, this forces $\rho_k \approx 1$ ($m_k \approx 0$) for all channels with non-negligible $|c_k|$, especially fast and mid channels.
- Long extension requires that at $\Delta = SL$, the phase does not wrap destructively or exceed the network's tolerance: $\omega'_k SL \le \omega_k L \implies \rho_k \le 1/S$ ($m_k \ge 1$).
- These two conditions are mathematically contradictory for any fixed static table:
  $$\rho_k \approx 1 \quad (\text{Native preservation}) \quad \Longleftrightarrow \quad \rho_k \le \frac{1}{S} \quad (\text{Long de-aliasing}).$$
- Non-uniform retrofits (e.g. `log_s4`, YaRN) attempt to resolve this by partitioning the channels: fast channels stay at $\rho_k \approx 1$ while slow channels move to $\rho_k \approx 1/S$.
- However, as proven in Theorem 2 and Theorem 5, partitioning channels breaks collinearity, generating off-arc phase vectors on $\mathbb{T}^K$. Whether a task survives this off-arc exposure depends on its circuit tolerance (e.g. RULER survives coarse phase shifts; multi-hop QA does not).
- `[OBSERVED]` This explains why no tested static table simultaneously solves strict Native retention and natural multi-hop QA across all benchmarks (`INDEX.md` §0).

---

## 8. Synthesis and Verdict

1. `[DERIVED]` Zero-training RoPE retrofit is fundamentally constrained by **Transplant Rigidity (Theorem 1)** and **Conditioning Amplification (Theorem 4)**. It cannot be treated as a harmless coordinate re-parameterization.
2. `[DERIVED]` Non-uniform frequency reallocation is not a continuous deformation on the native data manifold; it is an **off-arc projection into un-trained regions of the phase torus $\mathbb{T}^K$ (Theorem 2 & 5)**.
3. `[OBSERVED]` The tension between Native retention and long-context extension is an empirical reality backed by multi-seed causal studies and factorial crosses (`INDEX.md` §0, `EXACT_RANGE_151M_3SEED_RESULT_20260820.md`, `FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`).
4. `[DERIVED]` Any claim that a single static table can achieve zero degradation on native text while simultaneously providing arbitrary long extrapolation without weight adaptation violates the compatibility modulus (Theorem 3) and the conditioning theorem (Theorem 4).
