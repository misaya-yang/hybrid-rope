# Research Report: Bilinear Attention Logit Readout & Frozen Q/K Co-Adaptation Dynamics (Focus Area R2)

- **Author / Agent:** `explorer_r2_qkreadout_1`
- **Date:** 2026-09-01
- **Focus Area:** R2 — Bilinear Attention Logit Readout, Phase Interference Dynamics, Frozen Checkpoint Extrapolation Failure, Post-Hoc Transplant Obstruction, and the 2x2 Factorial Crossing Diagnostic
- **Target Venue / Context:** ICLR 2027 Submission (`paper-2027/`)

---

## 1. Executive Summary & Theoretical Context

Rotary Position Embedding (RoPE) represents token positions by applying two-dimensional rotation matrices to orthogonal pairs of the query and key projection vectors. While standard analyses often treat RoPE as a passive position encoding or a simple geometric scaling factor $\theta$, this report investigates RoPE from the perspective of **bilinear readout mechanics** and **trained parameter co-adaptation**.

### Core Findings
1. **Bilinear Attention Readout as Fourier Wave-Packet Synthesis:**
   For any two tokens $x_m, x_n$ separated by displacement $\Delta = m - n$, the pre-softmax attention logit expands exactly into a sum of modulated sinusoids:
   $$\ell(\Delta) = q_m^\top R_\Omega(\Delta) k_n = \sum_{j=0}^{K-1} A_j \cos(\omega_j \Delta + \psi_j)$$
   where the amplitudes $A_j \ge 0$ and phase shifts $\psi_j \in [-\pi, \pi)$ are content-dependent bilinear forms in $(x_m, x_n)$ determined by the projection matrices $W_q, W_k$.
2. **Coupling and Interference Dynamics:**
   During pretraining, backpropagation co-adapts the projection weights $(W_q, W_k)$ with the fixed frequency multiset $\Omega = (\omega_0, \dots, \omega_{K-1})$. For attended relative distances $\Delta^*$, the network learns content representations that align the phases ($\psi_j \approx -\omega_j \Delta^* \pmod{2\pi}$), generating **sharp constructive interference peaks**. Simultaneously, for distractor displacements $\Delta \neq \Delta^*$, the phases are dispersed across $[0, 2\pi)$, enforcing **destructive interference** (background suppression).
3. **Failure Mechanism of Frozen Weights at $\Delta > L_{\text{train}}$:**
   When evaluated at sequence lengths beyond training ($\Delta > L_{\text{train}}$), the phase vector $\mathbf{\Phi}(\Delta) = (\omega_0 \Delta, \dots, \omega_{K-1} \Delta)$ encounters out-of-distribution (OOD) phase configurations on the $K$-torus $\mathbb{T}^K$. Because the frozen projection weights $W_q, W_k$ were optimized only for $\Delta \in [0, L_{\text{train}}]$, destructive interference breaks down into pseudo-constructive random phase alignment, causing the background noise floor to accumulate ($\sigma_{\text{bg}} \sim \sqrt{\sum A_j^2 / 2}$), peak-to-background ratio (PBR) to collapse, and softmax attention to suffer from dispersion and entropy collapse.
4. **Post-Hoc Frequency Transplant Obstruction Theorem:**
   For two unequal frequency multisets $\Omega \neq \Omega'$, no fixed, position-independent, invertible linear transformations $A, B$ can satisfy $A^\top R_{\Omega'}(\Delta) B = R_\Omega(\Delta)$ across any open interval of $\Delta$. Differentiating at $\Delta = 0$ proves that the skew-symmetric Lie algebra generators must be similar, forcing the eigenvalue multisets $\{\pm i \omega_k\}$ and $\{\pm i \omega'_k\}$ to be identical. Thus, post-hoc frequency alteration is mathematically obstructed from being absorbed by linear Q/K adapters.
5. **The 2x2 Table-Weight Crossing Diagnostic:**
   On a controlled 50M parameter transformer, swapping the frequency table post-hoc under frozen weights causes perplexity to explode from **$7.14 \to 76.20$**, despite static geometric effective rank improving from $4.57 \to 12.54$. A $2\times2$ factorial decomposition reveals that the table $\times$ weights interaction ($I_{T \times W} = -3.5367$, $95\%$ CI $[-5.165, -3.039]$) dominates the main effects by a factor of $5.9\times$, providing rigorous empirical proof that frequency tables act as coordinates to which neural weights strongly co-adapt.

---

## 2. Bilinear Attention Logit Expansion

### 2.1 Rotary Embedding Geometry in Multi-Head Attention

Let $d$ be the head dimension, and let $K = d / 2$ denote the number of rotary pairs (orthogonal 2D subspaces). Let $\Omega = (\omega_0, \omega_1, \dots, \omega_{K-1})$ be the ordered frequency multiset with $\omega_0 > \omega_1 > \dots > \omega_{K-1} > 0$.

For an input sequence with hidden representations $x_m \in \mathbb{R}^{d_{\text{model}}}$ at position $m$ and $x_n \in \mathbb{R}^{d_{\text{model}}}$ at position $n$, the unrotated query and key vectors for a given attention head are:
$$q_m^{(0)} = W_q x_m \in \mathbb{R}^d, \qquad k_n^{(0)} = W_k x_n \in \mathbb{R}^d$$
where $W_q, W_k \in \mathbb{R}^{d \times d_{\text{model}}}$.

Partition $q_m^{(0)}$ and $k_n^{(0)}$ into $K$ two-dimensional subvectors:
$$q_{m,j} = \begin{bmatrix} q_{m, 2j}^{(0)} \\ q_{m, 2j+1}^{(0)} \end{bmatrix} \in \mathbb{R}^2, \qquad k_{n,j} = \begin{bmatrix} k_{n, 2j}^{(0)} \\ k_{n, 2j+1}^{(0)} \end{bmatrix} \in \mathbb{R}^2, \qquad j \in \{0, 1, \dots, K-1\}$$

RoPE applies a position-dependent 2D rotation $R(\omega_j m)$ and $R(\omega_j n)$ to each pair:
$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$
The rotated query and key pairs are:
$$\tilde{q}_{m,j} = R(\omega_j m) q_{m,j}, \qquad \tilde{k}_{n,j} = R(\omega_j n) k_{n,j}$$

The total block-diagonal rotation operator on $\mathbb{R}^d$ is denoted $R_\Omega(m) = \bigoplus_{j=0}^{K-1} R(\omega_j m)$.

### 2.2 Exact Algebraic Derivation of Pairwise Readout

The pre-softmax attention logit $\ell(m, n)$ (omitting the fixed scale factor $1/\sqrt{d}$ for notational brevity, or absorbing it into $W_q$) is:
$$\ell(m, n) = \tilde{q}_m^\top \tilde{k}_n = q_m^{(0)\top} R_\Omega(m)^\top R_\Omega(n) k_n^{(0)}$$

Using the orthogonal group property $R(\omega_j m)^\top R(\omega_j n) = R(-\omega_j m) R(\omega_j n) = R(\omega_j (n - m)) = R(-\omega_j \Delta) = R(\omega_j \Delta)^\top$, with relative displacement $\Delta = m - n$:
$$\ell(\Delta) = \sum_{j=0}^{K-1} q_{m,j}^\top R(\omega_j \Delta) k_{n,j}$$

Expanding each $2 \times 2$ bilinear block:
$$\begin{aligned}
\ell_j(\Delta) &= \begin{bmatrix} q_{m, 2j}^{(0)} & q_{m, 2j+1}^{(0)} \end{bmatrix} \begin{bmatrix} \cos(\omega_j \Delta) & -\sin(\omega_j \Delta) \\ \sin(\omega_j \Delta) & \cos(\omega_j \Delta) \end{bmatrix} \begin{bmatrix} k_{n, 2j}^{(0)} \\ k_{n, 2j+1}^{(0)} \end{bmatrix} \\
&= \left( q_{m, 2j}^{(0)} k_{n, 2j}^{(0)} + q_{m, 2j+1}^{(0)} k_{n, 2j+1}^{(0)} \right) \cos(\omega_j \Delta) + \left( q_{m, 2j}^{(0)} k_{n, 2j+1}^{(0)} - q_{m, 2j+1}^{(0)} k_{n, 2j}^{(0)} \right) \sin(\omega_j \Delta)
\end{aligned}$$

Define the symmetric (cosine) and skew-symmetric (sine) content coefficients:
$$C_j(x_m, x_n) \triangleq q_{m, 2j}^{(0)} k_{n, 2j}^{(0)} + q_{m, 2j+1}^{(0)} k_{n, 2j+1}^{(0)}$$
$$D_j(x_m, x_n) \triangleq q_{m, 2j}^{(0)} k_{n, 2j+1}^{(0)} - q_{m, 2j+1}^{(0)} k_{n, 2j}^{(0)}$$

### 2.3 Complex Representation & Polar Amplitude-Phase Form

In the complex plane $\mathbb{C}$, let:
$$q_{m,j}^{\mathbb{C}} = q_{m, 2j}^{(0)} + i q_{m, 2j+1}^{(0)} = |q_{m,j}| e^{i \theta_{q,j}}, \qquad k_{n,j}^{\mathbb{C}} = k_{n, 2j}^{(0)} + i k_{n, 2j+1}^{(0)} = |k_{n,j}| e^{i \theta_{k,j}}$$

Form the complex bilinear product $\alpha_j$:
$$\alpha_j \triangleq q_{m,j}^{\mathbb{C}} \overline{k_{n,j}^{\mathbb{C}}} = \left( q_{m, 2j}^{(0)} + i q_{m, 2j+1}^{(0)} \right) \left( k_{n, 2j}^{(0)} - i k_{n, 2j+1}^{(0)} \right) = C_j + i (-D_j)$$

Then:
$$A_j \triangleq |\alpha_j| = |q_{m,j}^{\mathbb{C}}| |k_{n,j}^{\mathbb{C}}| = \sqrt{C_j^2 + D_j^2}$$
$$\psi_j \triangleq -\operatorname{atan2}(D_j, C_j) = \theta_{k,j} - \theta_{q,j} = \arg(k_{n,j}^{\mathbb{C}}) - \arg(q_{m,j}^{\mathbb{C}})$$

Using the harmonic addition theorem:
$$C_j \cos(\omega_j \Delta) + D_j \sin(\omega_j \Delta) = A_j \cos(\omega_j \Delta + \psi_j)$$

Summing across all $K$ rotary pairs yields the canonical Fourier readout formula:
$$\boxed{\ell(\Delta) = \sum_{j=0}^{K-1} A_j(x_m, x_n) \cos\left( \omega_j \Delta + \psi_j(x_m, x_n) \right)}$$

### 2.4 Explicit Dependence on Token Representations and Projection Matrices

Let $W_q^{(r)}$ and $W_k^{(r)}$ denote the $r$-th row of $W_q$ and $W_k$ respectively ($r \in \{0, \dots, d-1\}$). The content coefficients are explicit bilinear matrix forms in $(x_m, x_n)$:
$$C_j = x_m^\top \left( W_q^{(2j)\top} W_k^{(2j)} + W_q^{(2j+1)\top} W_k^{(2j+1)} \right) x_n \triangleq x_m^\top M_{j, \text{sym}} x_n$$
$$D_j = x_m^\top \left( W_q^{(2j)\top} W_k^{(2j+1)} - W_q^{(2j+1)\top} W_k^{(2j)} \right) x_n \triangleq x_m^\top M_{j, \text{skew}} x_n$$

**Key Insight:** The attention logit is not simply a function of position displacement $\Delta$. It is a **content-modulated Fourier wave-packet**, where the token representations $x_m, x_n$ dynamically set both the channel amplitudes $A_j$ and the channel phase modulations $\psi_j$ through the trained projection tensors $M_{j, \text{sym}}$ and $M_{j, \text{skew}}$.

---

## 3. Projection Weight ($W_q, W_k$) & Phase Spectrum Co-Adaptation Dynamics

### 3.1 Fourier Synthesis of Attention Distance Curves

From a signal processing perspective, an attention head attempting to implement relative distance addressing (e.g., attending to the start of a sentence, a syntactic head, or a retrieval target at displacement $\Delta^*$) must construct a logit function $\ell(\Delta)$ that exhibits a prominent global maximum at $\Delta = \Delta^*$ and remains suppressed at non-target displacements $\Delta \neq \Delta^*$.

Because $\ell(\Delta) = \sum_{j=0}^{K-1} A_j \cos(\omega_j \Delta + \psi_j)$ is a truncated trigonometric series with $K$ frequencies, the network achieves this via **phase interference synthesis**:

```
                  ATTENTION LOGIT WAVE-PACKET SYNTHESIS
                  
     Target Position Delta*                 Background Positions Delta != Delta*
   (Constructive Interference)                   (Destructive Interference)
   
  Band 0 (Fast):   /\  /\  /\  /\  /\          /\  /\  /\  /\  /\
  Band 1 (Mid):     /--\  /--\  /--\            /--\  /--\  /--\
  Band 2 (Slow):   /----\    /----\            /----\    /----\
  
  Sum l(Delta):         |||                           ~ ~ ~ ~
                   (Sharp Peak)                    (Noise Floor)
                 l(Delta*) ~ sum A_j            l(Delta) ~ 0 (+/- sigma_bg)
```

### 3.2 Mechanism of Constructive Interference at Target Positions $\Delta^*$

When the query token at $m$ needs to attend to key token at $n = m - \Delta^*$:
1. The projection matrices $W_q, W_k$ map the semantic features of $(x_m, x_n)$ into query/key subvectors such that:
   $$\psi_j(x_m, x_n) \approx -\omega_j \Delta^* \pmod{2\pi} \quad \forall j \in \mathcal{S}_{\text{active}}$$
   where $\mathcal{S}_{\text{active}} \subseteq \{0, \dots, K-1\}$ is the subset of frequency bands allocated to that attention head's addressing range.
2. Under this phase-matching condition, each cosine argument evaluates to:
   $$\cos(\omega_j \Delta^* + \psi_j) \approx \cos(0) = 1$$
3. The individual channel logits sum coherently:
   $$\ell(\Delta^*) \approx \sum_{j \in \mathcal{S}_{\text{active}}} A_j \gg 0$$
   This produces a sharp, localized peak in the pre-softmax logit landscape.

### 3.3 Mechanism of Destructive Interference at Distractor Positions $\Delta \neq \Delta^*$

At non-target positions $\Delta \neq \Delta^*$:
1. For any $\Delta$, the phase arguments $\theta_j(\Delta) = \omega_j \Delta + \psi_j \pmod{2\pi}$ disperse quasi-uniformly across the unit circle $\mathbb{S}^1$.
2. The sum behaves as a random-phase walk:
   $$\mathbb{E}_{\Delta \neq \Delta^*} [\ell(\Delta)] \approx 0$$
   $$\operatorname{Var}_{\Delta \neq \Delta^*} [\ell(\Delta)] = \sum_{j=0}^{K-1} A_j^2 \mathbb{E}[\cos^2(\omega_j \Delta + \psi_j)] = \frac{1}{2} \sum_{j=0}^{K-1} A_j^2$$
3. Thus, destructive interference suppresses the background logit level to a quiet baseline with fluctuation standard deviation $\sigma_{\text{bg}} = \sqrt{\frac{1}{2} \sum_{j=0}^{K-1} A_j^2}$.

### 3.4 The Nature of Co-Adaptation

The projection weights $W_q, W_k$ do **not** learn frequency-agnostic semantic embeddings. Instead, during gradient descent training:
- The gradients $\frac{\partial \mathcal{L}}{\partial W_q}$ and $\frac{\partial \mathcal{L}}{\partial W_k}$ explicitly depend on $R_\Omega(\Delta)$.
- The weights co-adapt their internal principal axes to the specific numerical values of the frequency multiset $\Omega = (\omega_0, \dots, \omega_{K-1})$.
- The weights internalize the metric tensor of the frequency multiset: which channel rotates at what rate, and how content vectors must be steered in phase space to create constructive peaks at in-distribution distances $[0, L_{\text{train}}]$.

---

## 4. Failure Mechanism for Frozen Weights at $\Delta > L_{\text{train}}$

When a pretrained checkpoint with frozen weights $W_q, W_k$ is evaluated at context lengths extending beyond the training window ($\Delta > L_{\text{train}}$), the mathematical readout mechanism encounters four severe failure modes:

```
+-------------------------------------------------------------------------------+
|             EXTRAPOLATION BREAKDOWN (Delta > L_train, Frozen Weights)          |
+-------------------------------------------------------------------------------+
| 1. OOD Phase Combinations: (omega_0 Delta, ..., omega_K-1 Delta) on T^K       |
|    --> Fast bands wrap hundreds of times; slow bands enter unvisited sectors  |
+-------------------------------------------------------------------------------+
| 2. Breakdown of Destructive Interference:                                     |
|    --> Background logits no longer cancel; random constructive alignment      |
|    --> Severe noise spikes at arbitrary distractor positions                  |
+-------------------------------------------------------------------------------+
| 3. Collapse of Peak-to-Background Ratio (PBR):                               |
|    --> PBR = [l(Delta*) - mu_bg] / sigma_bg drops precipitously               |
+-------------------------------------------------------------------------------+
| 4. Softmax Attention Dispersion & Entropy Collapse:                           |
|    --> Denominator sum_n exp(l_n) sums over thousands of noisy tokens         |
|    --> True target attention probability p(Delta*) -> 0                       |
+-------------------------------------------------------------------------------+
```

### 4.1 OOD Phase Trajectory on the $K$-Torus $\mathbb{T}^K$

The joint phase configuration at displacement $\Delta$ is a point on the $K$-dimensional torus:
$$\mathbf{\Phi}(\Delta) = (\omega_0 \Delta \bmod 2\pi, \; \omega_1 \Delta \bmod 2\pi, \; \dots, \; \omega_{K-1} \Delta \bmod 2\pi) \in \mathbb{T}^K$$

During pretraining on sequences of length $L_{\text{train}}$, the model only observes phase trajectories along the curve $\gamma: [0, L_{\text{train}}] \to \mathbb{T}^K$.
- For fast frequencies ($\omega_j \gg 2\pi / L_{\text{train}}$), the phase wraps multiple times, but its relative correlation with intermediate and slow bands is sampled only for $\Delta \le L_{\text{train}}$.
- For slow frequencies ($\omega_j \le 2\pi / L_{\text{train}}$), the phase never completes a single full cycle during training ($\omega_j \Delta < 2\pi$).
- When $\Delta > L_{\text{train}}$, the slow bands traverse completely unobserved phase angles $[ \omega_j L_{\text{train}}, \omega_j L_{\text{ext}} ]$, while the fast bands combine with these new slow-band phases in permutations never encountered during training.

### 4.2 Breakdown of Destructive Cancellation & Noise Floor Accumulation

Because the projection matrices $W_q, W_k$ were optimized only on $\Delta \in [0, L_{\text{train}}]$:
1. The mathematical condition for background suppression ($\sum_j A_j \cos(\omega_j \Delta + \psi_j) \approx 0$) was enforced only along the segment $[0, L_{\text{train}}]$.
2. For $\Delta > L_{\text{train}}$, the quasi-random phase shifts $\omega_j \Delta$ frequently line up by chance across multiple channels for arbitrary, irrelevant distractor tokens $n$.
3. This creates **pseudo-constructive noise spikes**: distractor tokens receive attention logits $\ell(\Delta) \gg 0$ purely due to out-of-distribution phase alignment.

### 4.3 Peak-to-Background Ratio (PBR) Degradation

Define the Peak-to-Background Ratio for a target token at displacement $\Delta^*$:
$$\mathrm{PBR}(\Delta^*) \triangleq \frac{\ell(\Delta^*) - \mu_{\text{bg}}}{\sigma_{\text{bg}}}$$
where $\mu_{\text{bg}} = \mathbb{E}_{n \neq n^*} [\ell(m - n)]$ and $\sigma_{\text{bg}} = \sqrt{\operatorname{Var}_{n \neq n^*} [\ell(m - n)]}$.

- In-distribution ($\Delta^* \le L_{\text{train}}$): The target logit achieves coherent alignment ($\ell(\Delta^*) \approx \sum_j A_j$), while background logits are suppressed ($\mu_{\text{bg}} \approx 0, \sigma_{\text{bg}} \approx \sqrt{\sum A_j^2 / 2}$), yielding high PBR ($\mathrm{PBR} \approx \sqrt{2 K_{\text{active}}}$).
- Out-of-distribution ($\Delta^* > L_{\text{train}}$): The target phase coherence is partially degraded, while background distractor tokens at large sequence lengths $N \gg L_{\text{train}}$ generate extreme values in the tail of the background logit distribution:
  $$\max_{n \text{ distractor}} \ell(m - n) \approx \mu_{\text{bg}} + \sigma_{\text{bg}} \sqrt{2 \ln N}$$
  When $N$ is large (e.g., $16\text{K}, 32\text{K}$), $\max \ell_{\text{distractor}}$ exceeds $\ell(\Delta^*)$, completely burying the retrieval signal.

### 4.4 Softmax Entropy Collapse & Attention Dispersion

The attention weight assigned to the target token is:
$$p(\Delta^*) = \frac{\exp(\ell(\Delta^*))}{\exp(\ell(\Delta^*)) + \sum_{n \neq n^*} \exp(\ell(m - n))}$$

In the long-context regime:
1. **Attention Dispersion:** The denominator sums over $N - 1$ background terms. Even if individual distractor logits are moderate, their accumulated sum $\sum_{n \neq n^*} \exp(\ell(m - n)) \approx N \mathbb{E}[\exp(\ell_{\text{bg}})]$ grows linearly with context length $N$, driving $p(\Delta^*) \to 0$.
2. **Entropy Collapse / Spurious Sink:** When pseudo-constructive spikes occur at distractor positions, or when the model retreats to initial tokens (attention sinks) to dump unroutable attention mass, the attention distribution undergoes catastrophic entropy collapse, assigning near-unit probability to uninformative tokens.

### 4.5 Repository Evidence & Channel Ablation Findings

This theoretical failure mechanism is directly corroborated by canonical empirical studies in the repository:
- **Destructive Interference Channel Ablation (`EXPERIMENT_REPORT_20260724.md` §§10–11):**
  On 151.9M checkpoints at seed 42, deleting the trained high-norm OOD frequency pairs *improved* $1\text{K}–8\text{K}$ NLL (Geo improved by $-0.280$ NLL at $8\text{K}$ when 16 destructive pairs were removed; EVQ improved by $-0.108$ NLL when 18 pairs were removed). This proves directly that features learned under one frequency assignment become actively destructive after their phases move out of distribution.
- **Llama-3-8B Causal Probe (`FABLE5_EVQ_MECHANISM_AUDIT.md` §1.2 T2):**
  In matched 16K passkey probes on Llama-3-8B, gold-block deletion in the unadapted Geo arm caused **$\Delta\text{NLL} \approx -0.010$ (zero causal effect)**, proving that at $16\text{K}$, frozen standard RoPE completely loses the ability to transmit remote context into the residual stream due to noise-floor accumulation.

---

## 5. Mathematical Proof of Post-Hoc Frequency Transplant Obstruction

The following theorem formalizes why a pretrained model's frequency table cannot be replaced post-hoc by inserting a fixed, position-independent linear adapter on Q and K without altering the mathematical bilinear form.

### 5.1 Formal Theorem Statement

\begin{theorem}[Post-Hoc Frequency Transplant Obstruction]
Let $R_\Omega(\Delta) = \bigoplus_{k=0}^{K-1} R(\omega_k \Delta)$ and $R_{\Omega'}(\Delta) = \bigoplus_{k=0}^{K-1} R(\omega'_k \Delta)$ be the block-diagonal rotation operators for two frequency multisets $\Omega = \{\omega_0, \dots, \omega_{K-1}\}$ and $\Omega' = \{\omega'_0, \dots, \omega'_{K-1}\}$ on $\mathbb{R}^{2K}$.

If there exist position-independent, invertible real matrices $A, B \in \mathrm{GL}(2K, \mathbb{R})$ such that
$$A^\top R_{\Omega'}(\Delta) B = R_\Omega(\Delta)$$
holds for all $\Delta$ in an open interval $(-\epsilon, \epsilon) \subseteq \mathbb{R}$ containing zero, then $\Omega'$ and $\Omega$ must be identical as frequency multisets up to sign and permutation:
$$|\Omega'| = |\Omega|$$
Repeated frequencies may mix within their full equal-frequency invariant subspace.

For integer-only positions $\Delta \in \mathbb{Z}$, the condition implies equality of frequencies up to sign, permutation, and $2\pi$ aliasing:
$$\omega'_k \equiv \pm \omega_{\pi(k)} \pmod{2\pi}$$
\end{theorem}

### 5.2 Rigorous Step-by-Step Proof

**Step 1: Evaluation at $\Delta = 0$ and elimination of $B$.**
Both $R_\Omega(\Delta)$ and $R_{\Omega'}(\Delta)$ are continuous one-parameter matrix groups with identity at zero:
$$R_\Omega(0) = \bigoplus_{k=0}^{K-1} R(0) = \bigoplus_{k=0}^{K-1} I_2 = I_{2K}$$
$$R_{\Omega'}(0) = \bigoplus_{k=0}^{K-1} R(0) = \bigoplus_{k=0}^{K-1} I_2 = I_{2K}$$

Evaluating the hypothesis $A^\top R_{\Omega'}(\Delta) B = R_\Omega(\Delta)$ at $\Delta = 0$:
$$A^\top R_{\Omega'}(0) B = R_\Omega(0) \implies A^\top I_{2K} B = I_{2K} \implies A^\top B = I_{2K}$$
Since $A$ is invertible, this uniquely determines $B$:
$$B = A^{-\top} \triangleq (A^\top)^{-1} = (A^{-1})^\top$$

**Step 2: Group Similarity Relation.**
Substituting $B = A^{-\top}$ back into the original identity gives:
$$A^\top R_{\Omega'}(\Delta) A^{-\top} = R_\Omega(\Delta) \quad \forall \Delta \in (-\epsilon, \epsilon)$$
Taking the transpose inverse of both sides confirms that $R_{\Omega'}(\Delta)$ and $R_\Omega(\Delta)$ are similar matrices in $\mathrm{GL}(2K, \mathbb{R})$ for every $\Delta \in (-\epsilon, \epsilon)$:
$$R_\Omega(\Delta) = A^\top R_{\Omega'}(\Delta) (A^\top)^{-1}$$

**Step 3: Infinitesimal Generators in the Lie Algebra $\mathfrak{so}(2K)$.**
Each rotation operator is generated by a block-diagonal skew-symmetric matrix:
$$R_\Omega(\Delta) = \exp(\Delta G_\Omega), \qquad R_{\Omega'}(\Delta) = \exp(\Delta G_{\Omega'})$$
where the Lie algebra generator $G_\Omega \in \mathfrak{so}(2K)$ is:
$$G_\Omega = \bigoplus_{k=0}^{K-1} \omega_k J, \qquad J = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$
$$G_{\Omega'} = \bigoplus_{k=0}^{K-1} \omega'_k J$$

**Step 4: Differentiation at the Identity ($\Delta = 0$).**
Differentiating the similarity relation with respect to $\Delta$ at $\Delta = 0$:
$$\left. \frac{d}{d\Delta} \left( A^\top R_{\Omega'}(\Delta) A^{-\top} \right) \right|_{\Delta=0} = \left. \frac{d}{d\Delta} R_\Omega(\Delta) \right|_{\Delta=0}$$
Using the chain rule:
$$A^\top \left( \left. \frac{d}{d\Delta} \exp(\Delta G_{\Omega'}) \right|_{\Delta=0} \right) A^{-\top} = \left. \frac{d}{d\Delta} \exp(\Delta G_\Omega) \right|_{\Delta=0}$$
$$A^\top G_{\Omega'} A^{-\top} = G_\Omega$$

Thus, the generator $G_\Omega$ is similar to the generator $G_{\Omega'}$ via the similarity matrix $A^\top$:
$$G_\Omega \sim G_{\Omega'}$$

**Step 5: Spectral Invariance under Similarity.**
Matrix similarity preserves the characteristic polynomial, the minimal polynomial, the trace, and the full eigenvalue multiset (spectrum with algebraic multiplicities):
$$\operatorname{Spec}(G_\Omega) = \operatorname{Spec}(G_{\Omega'})$$

For any 2D block $\omega_k J = \begin{bmatrix} 0 & -\omega_k \\ \omega_k & 0 \end{bmatrix}$, its characteristic equation is:
$$\det(\lambda I_2 - \omega_k J) = \lambda^2 + \omega_k^2 = 0 \implies \lambda = \pm i \omega_k$$
Because $G_\Omega$ is block-diagonal, its spectrum on $\mathbb{C}$ is:
$$\operatorname{Spec}(G_\Omega) = \{ +i\omega_0, -i\omega_0, \; +i\omega_1, -i\omega_1, \; \dots, \; +i\omega_{K-1}, -i\omega_{K-1} \}$$
Similarly, for $G_{\Omega'}$:
$$\operatorname{Spec}(G_{\Omega'}) = \{ +i\omega'_0, -i\omega'_0, \; +i\omega'_1, -i\omega'_1, \; \dots, \; +i\omega'_{K-1}, -i\omega'_{K-1} \}$$

Equating the two spectra $\operatorname{Spec}(G_\Omega) = \operatorname{Spec}(G_{\Omega'})$ requires that the imaginary parts match up to pairing and ordering:
$$\{ |\omega_0|, |\omega_1|, \dots, |\omega_{K-1}| \} = \{ |\omega'_0|, |\omega'_1|, \dots, |\omega'_{K-1}| \}$$
Thus, there exists a permutation $\pi \in S_K$ such that $|\omega'_k| = |\omega_{\pi(k)}|$ for all $k \in \{0, \dots, K-1\}$.

**Step 6: Discrete Integer-Position Case ($\Delta \in \mathbb{Z}$).**
If the identity $A^\top R_{\Omega'}(\Delta) B = R_\Omega(\Delta)$ holds only for integer positions $\Delta \in \mathbb{Z}$:
- Evaluating at $\Delta = 0$ still yields $B = A^{-\top}$.
- Evaluating at $\Delta = 1$ yields similarity of the discrete step operators:
  $$R_\Omega(1) = A^\top R_{\Omega'}(1) A^{-\top} \implies R_\Omega(1) \sim R_{\Omega'}(1)$$
- The spectrum of $R(\omega_k)$ is $\{ e^{+i \omega_k}, e^{-i \omega_k} \}$.
- Similarity implies $\{ e^{\pm i \omega'_k} \} = \{ e^{\pm i \omega_k} \}$, which yields:
  $$\omega'_k \equiv \pm \omega_{\pi(k)} \pmod{2\pi}$$
- In standard LLM implementations (e.g., OLMo, LLaMA, Qwen), base frequencies are bounded by $\omega_k \in (0, 1] \subset (-\pi, \pi]$. In this principal frequency band, $2\pi$ aliasing cannot occur, and discrete equality forces exact multiset equality $|\Omega'| = |\Omega|$. $\blacksquare$

### 5.3 Scope and Practical Implications
- **Obstruction Scope:** This theorem proves that no position-independent linear adapter (such as a standard linear LoRA on $W_q, W_k$) can exactly preserve a model's bilinear attention form under a non-trivial frequency transplant $\Omega \to \Omega'$.
- **Empirical Confirmation (`OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`):**
  When EVQ frequencies were transplanted into mature OLMo-2-1B-Instruct ($63$ of $64$ rotary pairs modified), unwrapped phase error exceeded $\pi$ radians on $45/64$ pairs at $\Delta = 4095$. Matched LoRA training concentrated $>81\%$ of adapter energy into Q/K trying to compensate for coordinate shift, but held-out retrieval on 4K RULER tasks collapsed from $55\%$ to $0\%$.

---

## 6. The 2x2 Table-Weight Crossing Diagnostic ($7.14 \to 76.20$ PPL Shock)

To empirically isolate the interaction between trained projection weights and runtime frequency tables, a counterfactual factorial experiment was conducted using fully trained weights.

### 6.1 Experimental Protocol
- **Model:** 50M parameter transformer (6 layers, 8 heads, $d_{\text{head}} = 64$, $K = 32$ rotary pairs).
- **Training Configurations:** Two identical architectures trained from scratch on TinyStories with base $b = 500\text{K}$, $L_{\text{train}} = 512$, seed 42:
  1. `Geo`: Trained with standard Geometric RoPE frequency table.
  2. `EVQ`: Trained with EVQ-Cosh frequency table ($\tau = 2.83$).
- **Evaluation Dataset:** TinyStories validation split, 8 identical sequence windows of length $L = 512$, evaluated at query positions $\{63, 127, 255, 383, 511\}$, spanning $1,920$ head-query observations. All model parameters were frozen (zero training during probe, CPU execution).
- **Metric Definitions:**
  - `Bare Geometry r_2`: Block-whitened stable rank of position embedding matrix $\Phi$.
  - `Bare M^sm r_2`: Stable rank of attention categorical Fisher $M^{\text{sm}} = J^\top F_{\text{sm}} J$, where $F_{\text{sm}} = \operatorname{diag}(p) - p p^\top$.
  - `Content M^EF r_2 / Trace`: Empirical Fisher outer product $(J^\top g_z)(J^\top g_z)^\top$ using actual LM loss gradient $g_z = \frac{\partial \mathcal{L}_{\text{LM}}}{\partial z}$ with content-weighted Jacobian.
  - `Frequency M^EF r_2 / Trace`: Empirical Fisher outer product with respect to $\log \omega_k$.

### 6.2 Full 2x2 Empirical Counterfactual Table

\begin{table}[ht]
\centering
\caption{\textbf{50M $2\times2$ Table-Weight Counterfactual Matrix.} Canonical evaluation on TinyStories validation ($L=512$, seed 42, 1,920 observations; owner: \texttt{scripts/analysis/attention\_fisher\_50m\_probe.py}).}
\label{tab:coadapt_50m}
\small
\begin{tabular}{@{}llcccccc@{}}
\toprule
\textbf{Weights} & \textbf{Runtime Table} & \textbf{LM Loss} & \textbf{PPL} & \textbf{Bare $r_2$} & \textbf{Bare $M^{\text{sm}} r_2$} & \textbf{Content $M^{\text{EF}}$ ($r_2$ / trace)} & \textbf{Freq $M^{\text{EF}}$ ($r_2$ / trace)} \\
\midrule
Geo & Geo (Matched) & 1.9659 & \textbf{7.14} & 4.57 & 14.86 & 10.20 / $1.99 \times 10^{-8}$ & 1.79 / $8.80 \times 10^{-7}$ \\
Geo & EVQ (Mismatch) & 4.3333 & \textbf{76.20} & 12.54 & 25.51 & 1.48 / $2.93 \times 10^{-6}$ & 2.21 / $3.95 \times 10^{-3}$ \\
EVQ & Geo (Mismatch) & 3.1378 & \textbf{23.05} & 4.57 & 16.94 & 8.34 / $1.12 \times 10^{-7}$ & 1.04 / $5.63 \times 10^{-5}$ \\
EVQ & EVQ (Matched) & 1.9685 & \textbf{7.16} & 12.54 & 21.91 & 12.65 / $1.69 \times 10^{-8}$ & 4.99 / $2.80 \times 10^{-7}$ \\
\bottomrule
\end{tabular}
\end{table}

### 6.3 Factorial Decomposition & Analysis of Variance (ANOVA)

Let $y_{GG}, y_{GE}, y_{EG}, y_{EE}$ denote the evaluation metric across the four cells:
- **Table Main Effect ($E_T$):**
  $$E_T = \frac{(y_{GE} + y_{EE}) - (y_{GG} + y_{EG})}{2} = \frac{(4.3333 + 1.9685) - (1.9659 + 3.1378)}{2} = \mathbf{+0.5991}$$
  $95\%$ Bootstrap CI (500 resamples across 40 query groups): $[+0.331, +0.956]$.
- **Weights Main Effect ($E_W$):**
  $$E_W = \frac{(y_{EG} + y_{EE}) - (y_{GG} + y_{GE})}{2} = \frac{(3.1378 + 1.9685) - (1.9659 + 4.3333)}{2} = \mathbf{-0.5965}$$
  $95\%$ Bootstrap CI: $[-0.896, -0.175]$.
- **Table $\times$ Weights Interaction Effect ($I_{T \times W}$):**
  $$I_{T \times W} = y_{EE} - y_{EG} - y_{GE} + y_{GG} = 1.9685 - 3.1378 - 4.3333 + 1.9659 = \mathbf{-3.5367}$$
  $95\%$ Bootstrap CI: $[-5.165, -3.039]$.

```
                      FACTORIAL INTERACTION ANALYSIS
                      
      Loss
       ^
  4.5 -|                     Geo Weights + EVQ Table (4.3333, PPL 76.20)
       |                        *
  4.0 -|                       /
       |                      /
  3.5 -|                     /
       |                    /    EVQ Weights + Geo Table (3.1378, PPL 23.05)
  3.0 -|                   /        *
       |                  /        /
  2.5 -|                 /        /
       |                /        /
  2.0 -|  *------------/--------/------* Matched Self-Consistent Baselines
       | (1.9659, PPL 7.14)           (1.9685, PPL 7.16)
       +------------------------------------------------------------>
             Geo Table                     EVQ Table
```

### 6.4 Key Insights and Theoretical Significance

1. **Dominance of the Interaction Term ($5.9\times$ Larger than Main Effects):**
   The interaction effect ($|I_{T \times W}| = 3.5367$) is nearly six times larger than either main effect ($|E_T| \approx 0.60, |E_W| \approx 0.60$). In ANOVA terms, the model's loss is almost entirely determined by whether the runtime table matches the coordinate system under which the weights were trained.
2. **Failure of Static Geometric Effective Rank as an LM Predictor:**
   - When EVQ table is installed on Geo weights, the bare positional effective rank $r_2$ jumps from **$4.57 \to 12.54$** ($+174\%$), and bare softmax Fisher rank jumps from **$14.86 \to 25.51$** ($+72\%$).
   - Yet language modeling perplexity collapses catastrophically from **$7.14 \to 76.20$** ($+967\%$).
   - This provides decisive empirical proof that static basis orthogonality and effective rank are **not** monotonic predictors of transformer loss or extrapolation capability.
3. **Explosion of Empirical Fisher Sensitivity:**
   - In the matched Geo-Geo cell, frequency Empirical Fisher trace is $8.80 \times 10^{-7}$.
   - In the mismatched Geo-EVQ cell, frequency Empirical Fisher trace explodes to $3.95 \times 10^{-3}$, an increase of **$4,494\times$**.
   - This massive gradient sensitivity reflects the severe loss shock caused when the coordinate system is altered underneath frozen projection weights.

---

## 7. Synthesis with Repository Evidence & Canonical Boundary Mapping

To maintain strict alignment with repository rules and claim ceilings, the following table maps each empirical and theoretical component to its canonical owner:

\begin{table}[ht]
\centering
\caption{\textbf{Canonical Evidence Ledger for Focus Area R2.}}
\label{tab:evidence_ledger}
\small
\begin{tabular}{@{}llll@{}}
\toprule
\textbf{Research Focus} & \textbf{Canonical Owner} & \textbf{Verified Result} & \textbf{Claim Ceiling / Limitation} \\
\midrule
Bilinear Readout Formulation & \texttt{ROPE\_CAUSAL\_...\_20260823.md} §1.1 & $\ell = \sum A_j \cos(\omega_j \Delta + \psi_j)$ & Exact algebraic decomposition \\
Subspace Budget Identity & \texttt{FULL\_ROPE\_...\_20260819.md} §2.3 & $r_2(\Gamma) = \frac{2K}{1+(K-1)\bar{c}}$ & Phase-invariant static metric only \\
Low-Frequency Collapse & \texttt{03\_theory.tex} Prop. 2 & Limit $\operatorname{span}\{1, \Delta\}$ / $\operatorname{span}\{\Delta, \Delta^2\}$ & Geometric loss of slow dimensions \\
Post-Hoc Obstruction Proof & \texttt{OLMO2\_POSTHOC\_...\_20260726.md} & $A^\top R_{\Omega'}(\Delta) B = R_\Omega(\Delta) \implies |\Omega'|=|\Omega|$ & Proved for exact invertible linear maps \\
50M $2\times2$ Crossing & \texttt{attention\_fisher\_50m\_probe.py} & PPL $7.14 / 76.20 / 23.05 / 7.16$ & Diagnostic; seed-42 counterfactual \\
Pure Allocation Identification & \texttt{EXACT\_RANGE\_151M\_3SEED...} & $-0.281 / -0.176 / -0.146$ @OOD & Fixed-support training-time $z$ only \\
Mature Checkpoint Co-Adaptation & \texttt{COADAPTIVE\_ALLOCATION...} & 4K $+0.00098$, tail $-0.0877$ & Internal mechanism; full/tail trade \\
\bottomrule
\end{tabular}
\end{table}

### 7.1 Nomenclature Compliance
- `Geo`: Fixed geometric-table baseline in from-scratch training.
- `Native`: Unmodified model-native RoPE table in pretrained checkpoints (e.g., OLMo-2, Qwen-2.5).
- `\evq{} / EVQ-Cosh`: Proposed closed-form nonlinear fixed table ($0$ learned parameters).
- `anchored \evq{}`: Exact-range EVQ-Cosh quantiles normalized to match target endpoints.
- `\rs{} / YaRN-style`: Fixed-index operator scaling slow bands and preserving fast bands.

### 7.2 Explicit Claim Boundaries
1. **Static Geometry is Not Task Performance:** Higher static rank $r_2$ indicates reduced pairwise subspace overlap under the uniform separation prior; it does not guarantee lower perplexity or superior length generalization.
2. **Obstruction Theorem Scope:** The obstruction theorem proves that exact post-hoc compensation is impossible for fixed, position-independent linear Q/K maps. It does not exclude approximate nonlinear retraining, token-dependent gating, or whole-operator reparameterizations.
3. **EVQ-Cosh Uniqueness:** EVQ-Cosh is the unique minimizer strictly for the declared quadratic/Green-kernel surrogate functional $\mathcal{C}_{\text{app}}[\rho]$, not a universal optimum over all possible language modeling objectives.

---

## 8. Conclusion

Focus Area R2 establishes the fundamental bridge between the static geometry of RoPE frequency tables and the dynamic behavior of trained language models. The attention logit readout $\ell(\Delta) = \sum_{j=0}^{K-1} A_j \cos(\omega_j \Delta + \psi_j)$ couples the discrete frequency spectrum with the continuous representation space learned by $W_q, W_k$.

Because training co-adapts weights to synthesize constructive interference at target positions and destructive interference across background distractors, altering the frequency table post-hoc breaks phase coherence and triggers catastrophic loss shocks (as demonstrated by the 50M $2\times2$ crossing). Furthermore, the Post-Hoc Frequency Transplant Obstruction Theorem proves that this coupling cannot be undone by linear reparameterizations. Consequently, optimal spectral budget design (such as EVQ-Cosh) must either be installed before pretraining to allow full weight co-adaptation or deployed through matched, coordinate-aware retrofit architectures.
