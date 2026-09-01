# R4 Axis B: Engineering Boundaries & Native-Support Pure-$z$ Paradigm
**Comprehensive Technical Report on RoPE Exponent Allocation, Co-Adaptation Barriers, and Deployment Realities**

- **Author / Agent:** `explorer_r4_engineering_1`
- **Date:** 2026-09-01
- **Status:** Complete Synthesis Report (Read-Only Investigation)
- **Target Repository:** `hybrid-rope` (ICLR 2027 Submission & Research Frontier)

---

## 1. Executive Summary & Core Architectural Thesis

The fundamental inquiry of RoPE length extrapolation centers on the spectral allocation of rotary channels $z_k \in [0, 1]$ and their structural coupling with learned query-key projection weights $(W_q, W_k)$. 

This report investigates the engineering boundaries and theoretical foundations of RoPE frequency re-allocation, synthesizing repository canonical evidence from 151.9M exact-range identification, 50M $2\times 2$ table-weight crossings, and mature 1.485B/1.5B checkpoint retrofit trials.

### Core Architectural Findings:
1. **Zero-Training Frozen Retrofit Ceiling:** Static frequency table replacement under frozen model weights is fundamentally bounded by *table-shock*. Because attention logits represent a coherent summation over rotary pairs whose phases are entangled with projection weights, replacing the frequency multiset under frozen weights destroys the learned content interference pattern (e.g., shifting 50M PPL from $7.14 \to 76.20$). Exact linear compensation is mathematically obstructed for unequal frequency multisets.
2. **The Native-Support Pure-$z$ Paradigm:** Rather than dilating the base $b_{\text{native}}$ or stretching the log-support span $[e_0, e_0 + R]$, the correct causal formulation isolates interior reallocation $z_k = F(k, s, \dots) \in [0, 1]$ while strictly inheriting $b_{\text{native}}, e_0, R$. Resolving the co-adaptation barrier requires freezing $z_{\text{new}}$ prior to training and updating only weights via matched low-rank adaptation (LoRA), controlled against a matched Native-$z$ LoRA baseline.
3. **Operational Serving Invariants:** Real-world serving systems (vLLM, SGLang, FlashAttention) strictly demand a **single static table** and a **single model instance** across all context lengths ($1\times, 2\times, 4\times$). Session routing hacks and dynamic per-request temperature gains break uniform serving and merely mask in-window degradation. Furthermore, absolute-position-dependent piecewise slope operators break translation invariance ($\theta(q) - \theta(k) \neq f(q-k)$) across boundary crossings, causing catastrophic task failure ($0.0000$ RULER macro).
4. **Post-Mortem of 12 Falsified Routes:** An in-depth audit of the 12 closed routes in `INDEX.md` §3.4 demonstrates that all scalar table selectors evaluated under assumed isotropic content or decoupled geometric metrics fail to rank downstream LM performance because they ignore the dominant table $\times$ weights interaction.

---

## 2. Fundamental Engineering Ceiling of Zero-Training Frozen Retrofit

### 2.1 The Table-Shock Mechanism & Attention Logit Decomposition
For a single attention head with $K = d_{\text{head}} / 2$ rotary pairs and relative token displacement $\Delta = i - j$, the pre-softmax attention logit is expressed as:
$$\ell_{ij} = \mathbf{q}_i^\top R_\Omega(\Delta) \mathbf{k}_j = \sum_{m=0}^{K-1} A_m \cos(\omega_m \Delta + \psi_m)$$
where:
- $\Omega = (\omega_0, \dots, \omega_{K-1})$ is the physical rotary frequency table;
- $A_m = \|\mathbf{q}_{i, (2m:2m+1)}\| \|\mathbf{k}_{j, (2m:2m+1)}\|$ is the pair-wise content amplitude;
- $\psi_m = \angle \mathbf{q}_{i, (2m:2m+1)} - \angle \mathbf{k}_{j, (2m:2m+1)}$ is the pair-wise learned content phase angle.

In a pretrained model, $(W_q, W_k)$ have undergone hundreds of billions of gradient steps to co-adapt $(A_m, \psi_m)$ to the specific geometric spectrum $\omega_m^{\text{native}} = b_{\text{native}}^{-2m/d}$. The attention logit relies on precise constructive interference at salient syntactic/semantic relative lags $\Delta$ and destructive interference elsewhere.

When an engineer swaps $\Omega^{\text{native}} \to \Omega^{\text{new}}$ under frozen weights:
1. **Phase Distorsion:** Every cosine term shifts phase by $(\omega_m^{\text{new}} - \omega_m^{\text{native}})\Delta$.
2. **Logit Collapse (Table-Shock):** The constructive alignment breaks across all tokens, de-concentrating attention entropy and causing massive logit deformation.
3. **Empirical Magnitude:** In the canonical 50M co-adaptation probe (`paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`), swapping the table under frozen weights explodes TinyStories PPL from **$7.1413 \to 76.1955$** (a $>10\times$ degradation). In 151.9M OLMo models (`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`), FMRoPE-trained weights on a Cosh table jump from $3.426 \to 5.776$ tail NLL.

```
+-----------------------------------------------------------------------------+
|                         The 2x2 Co-Adaptation Matrix                         |
|                                                                             |
|                           Runtime Table: Geo      Runtime Table: EVQ        |
|  Weights: Geo-Trained         PPL = 7.14             PPL = 76.20            |
|                           (Self-Consistent)         (Table Shock)           |
|                                                                             |
|  Weights: EVQ-Trained         PPL = 23.05             PPL = 7.16            |
|                           (Cross-Mismatch)       (Self-Consistent)          |
+-----------------------------------------------------------------------------+
```

### 2.2 The Frozen Transplant Obstruction Theorem
The exact obstruction theorem (`OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`) establishes a rigorous mathematical boundary:

> **Theorem (Post-Hoc Frequency Transplant Obstruction):**
> Let $\Omega = \{\omega_m\}_{m=0}^{K-1}$ and $\Omega' = \{\omega'_m\}_{m=0}^{K-1}$ be two distinct frequency multisets ($\Omega \neq \Omega'$). There exist **no fixed invertible linear projection maps** $M_Q, M_K \in \mathbb{R}^{d \times d}$ such that for all content vectors $\mathbf{q}, \mathbf{k} \in \mathbb{R}^d$ and all relative positions $\Delta \in \mathbb{Z}$:
> $$\mathbf{q}^\top M_Q^\top R_{\Omega'}(\Delta) M_K \mathbf{k} = \mathbf{q}^\top R_\Omega(\Delta) \mathbf{k}$$

*Implication for Adaptation:* Exact zero-shot post-hoc linear compensation is impossible. Any attempt to retrofit an existing checkpoint without retraining must accept either:
- Significant in-window approximation error (unrepairable logit energy $D^* > 0$), or
- Non-linear, length-dependent heuristics that break serving constraints.

### 2.3 The In-Window Preservation vs. Out-of-Distribution (OOD) Extrapolation Dilemma
Under zero training, any static table modification faces a zero-sum trade-off:
- **In-Window ($1\times, \Delta \le L_{\text{train}}$):** Demands $\omega_m^{\text{new}} \approx \omega_m^{\text{native}}$ to preserve learned $\psi_m$ alignment and keep short-context NLL and generation intact.
- **Extrapolation ($4\times, \Delta > L_{\text{train}}$):** Unmodified frequencies fail due to dual spectral pathologies:
  - *Slow-band failure ($\omega_m L_{\text{train}} \le 1$):* Bands that never completed a full cycle during pretraining enter unexplored phase angles $\omega_m \Delta \in (1, 4]$, presenting out-of-distribution phase configurations to frozen weights.
  - *Fast-band aliasing ($\omega_m L_{\text{train}} \gg 2\pi$):* High frequencies wrap rapidly, creating severe positional ambiguities where $\Delta$ and $\Delta + 2\pi/\omega_m$ are indistinguishable.
- **The Empirical Impossibility of Analytic Zero-Parameter Single Tables:** As shown in `ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md`, zero-parameter single tables (such as anchored EVQ-Cosh or protected-band Cosh) successfully improve $2\times$ extrapolation NLL ($-0.25$ and $-0.18$) but inflict catastrophic damage on $1\times$ in-window NLL (**$+3.98$** and **$+0.69$** NLL degradation).

---

## 3. The Native-Support Pure-$z$ Adaptation Paradigm

### 3.1 Formal Definition & Support-Invariance Algebra
The repository's authoritative causal formulation (`INDEX.md` §6.2, `ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`) parameterizes rotary frequencies via normalized log-coordinates:

$$x_k = -\log \omega_k = a + R z_k, \qquad \omega_k = b_{\text{native}}^{-e_k}$$
where:
- $b_{\text{native}}$ is the checkpoint's native RoPE base (e.g., $10000$ or $500000$);
- $e_k = e_0 + R z_k$ is the per-pair exponent, spanning support $[e_0, e_0 + R]$;
- $e_0 = 0$ (fastest band, $\omega_0 = 1$) and $e_{K-1} = (d-2)/d$ (slowest band);
- $R = e_{K-1} - e_0$ is the fixed logarithmic span;
- $z_k \in [0, 1]$ is the **pure normalized interior allocation** ($z_0 = 0, z_{K-1} = 1$).

```
Physical Frequency Spectrum:
k = 0 (Fastest)                                             k = K-1 (Slowest)
+---------------------------------------------------------------------------+
| z_0 = 0.0                      z_k in [0, 1]                  z_{K-1} = 1.0|
| e_0 (Pinned)           <-- Pure Interior Allocation -->          e_{K-1}  |
| omega_0 = 1.0                                                omega_{K-1}  |
+---------------------------------------------------------------------------+
  [============= Fixed Native Log-Support Span [e_0, e_0 + R] =============]
```

Under the **Native-support pure-$z$ paradigm**:
1. $b_{\text{native}}$, $e_0$, and $R$ are **strictly inherited** from the pretrained checkpoint.
2. The target extrapolation factor $s$ controls **only** the interior distribution function $z = F(k, s, \dots)$.
3. No endpoint shift, no scalar base dilation ($b \to b \cdot s^\gamma$), and no range expansion is permitted.

### 3.2 Resolving the $2\times 2$ Co-Adaptation Barrier via Matched LoRA
Because static table swapping causes table-shock under frozen weights, the network must be permitted to adapt its projection weights to the new coordinate system $z_{\text{new}}$.

The key insight is:
- $z_{\text{new}}$ is derived from first-principles spectral theory (e.g., EVQ-Cosh or phase-chord allocation) and **hash-frozen prior to adaptation**.
- The adaptation optimizer updates **only model weights** (via low-rank adapters on $W_q, W_k$).
- The optimizer adjusts $(A_m, \psi_m)$ to realign with the new static basis $z_{\text{new}}$, resolving the table-shock within a minimal token budget (e.g., 300 steps of dense natural LM replay).

### 3.3 The 4-Arm Matched Control Protocol ($z \times \text{adaptation}$)
To guarantee scientific validity and prevent confounding allocation improvements with generic fine-tuning capacity, `INDEX.md` §6.3 enforces a strict $2\times 2$ factorial experimental matrix:

| Arm | Frequency Allocation $z$ | Weight State | Scientific / Diagnostic Role |
| :--- | :--- | :--- | :--- |
| **Arm 1** | Native $z_{\text{geo}}$ | Frozen Native Weights | Unmodified Pretrained Reference |
| **Arm 2** | Proposed $z_{\text{new}}$ | Frozen Native Weights | Zero-Training Table-Shock Diagnostic |
| **Arm 3** | Native $z_{\text{geo}}$ | Matched Q/K LoRA | Adaptation-Budget Control Baseline |
| **Arm 4** | Proposed $z_{\text{new}}$ | Matched Q/K LoRA | **Target Method Hypothesis** |

#### Strict Protocol Invariants:
1. **Identical Hyperparameters:** Arms 3 and 4 must share identical adapter rank $r$, target projection layers, initialization seeds, optimizer (AdamW), learning rate, schedule, token order, batch geometry, attention scaling, and loss objective.
2. **Acceptance Rule:** Arm 4 ($z_{\text{new}} + \text{LoRA}$) must strictly outperform Arm 3 ($z_{\text{geo}} + \text{LoRA}$) at extrapolation lengths ($2\times, 4\times$) while matching Arm 1 at $1\times$. If Arm 4 equals Arm 3, the improvement is entirely attributable to LoRA capacity, falsifying the utility of $z_{\text{new}}$.

---

## 4. Operational Deployment Constraints in Real-World Systems

Industrial inference engines (e.g., vLLM, SGLang, TensorRT-LLM) place stringent architectural constraints on positional embeddings. Any method that violates these constraints cannot be deployed in production.

### 4.1 Single Static Table & KV-Cache Coordinate Uniformity
- **Requirement:** A single model checkpoint and a single static rotary frequency table $\Omega$ must simultaneously serve sequences from length $1$ to $L_{\text{max}} = 4 L_{\text{native}}$.
- **Failure of Length-Dependent Multi-Table Serving:** Maintaining multiple frequency tables for different context buckets ($1\times, 2\times, 4\times$) requires:
  - Multi-instance model replication (multiplying VRAM footprint), or
  - Re-computing rotary embeddings dynamically per request, which breaks KV-cache sharing and paged memory layouts.

### 4.2 Rejection of Dynamic Gain Scaling & Session Routing Hacks
- **Session Routing Hacks:** A router that chooses between Native RoPE for $L \le L_{\text{native}}$ and a modified table for $L > L_{\text{native}}$ requires knowing the total request length ($\text{prefill} + \text{max\_gen}$) upfront. It cannot support dynamic multi-turn conversations where length grows past $L_{\text{native}}$ mid-session, and it merely hides $1\times$ degradation behind conditional execution.
- **Dynamic Gain Scaling:** Applying a length-dependent attention temperature or amplitude multiplier $\gamma(L) = 1 + c \ln(L / L_{\text{native}})$ breaks chunked prefill and speculative decoding, where tokens in different positions within the same batch or KV cache would require conflicting scaling factors.

### 4.3 Preservation of Shift Invariance: Why Piecewise Coordinates Fail
A critical finding from repository experiments (`ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md` §2, `SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md` §2) is that **rotary position codes must strictly preserve relative translation invariance**:

$$\theta_m(q) - \theta_m(k) = \omega_m (q - k)$$

#### The Stateless Boundary-Slope Counterexample:
To avoid setting a target length $s$, a continuous-boundary-slope operator was constructed:
$$\theta_m(\text{pos}) = \begin{cases} \omega_m \cdot \text{pos}, & \text{pos} \le L_{\text{native}} \\ \omega_m L_{\text{native}} + \omega'_m (\text{pos} - L_{\text{native}}), & \text{pos} > L_{\text{native}} \end{cases}$$

When a query $q > L_{\text{native}}$ attends to a cached key $k \le L_{\text{native}}$:
$$\theta_m(q) - \theta_m(k) = \omega_m L_{\text{native}} + \omega'_m (q - L_{\text{native}}) - \omega_m k \neq \omega'_m (q - k)$$

The key coordinate remains scaled by $-\omega_m$, whereas the query operates in $-\omega'_m$. This cross-boundary relative-phase distortion completely destroyed long-range retrieval, yielding **$0.0000$ core-4 RULER macro score at both 8K and 16K**.

```
Query q > L_native attending to Key k <= L_native:

Key k:   [======================|======================]  Phase: omega * k
Query q: [======================|========]                Phase: omega * L_native + omega' * (q - L_native)
                                ^ Boundary
Resulting Relative Phase: theta(q) - theta(k) != f(q - k)  ==> DESTROYS RELATIVE CODE ==> RULER = 0.0000
```

---

## 5. Bounded Degradation Realities & Empirical Discrepancies

### 5.1 NLL vs. Needle Retrieval Dissociation
A central empirical phenomenon across all evaluated checkpoints is the decoupling between next-token likelihood (NLL) and downstream retrieval capability (RULER, NIAH, LongBench 2Wiki):
- **NLL is Token-Averaged:** Over 90% of cross-entropy loss is driven by local bigram statistics, short-range syntax, and common tokens that depend only on nearby context ($<512$ tokens).
- **Retrieval Depends on Sparse Circuits:** Multi-hop reasoning and long-range associative retrieval depend on a small subset of specialized attention heads maintaining sharp, high-entropy focus over thousands of tokens.
- **Evidence:** In `COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md`, an adapted table improved 16K tail NLL by $-0.08767$ while showing virtually zero gain on 200-row 2Wiki ($+0.00089$ token F1). Conversely, an operator can exhibit acceptable NLL but score $0.0000$ on needle retrieval.

### 5.2 Position-Dependent Non-Monotonicity (Penalty Bands vs. Far-Tail Gains)
Detailed per-token logit decomposition across 128 documents (`ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md` §5, `ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md`) reveals that the effect of spectrum reallocation is non-monotonic across sequence lags:

| Context Window Segment | Relative Position Range | Measured Delta NLL vs. Native | Physical / Mechanism Effect |
| :--- | :--- | :--- | :--- |
| **In-Window Segment** | Positions $0 - 4095$ | $+0.001$ NLL | Clean short-context preservation |
| **Transition Penalty Band** | Positions $4096 - 8192$ | **$+0.092$ NLL (8K)** / **$+0.034$ NLL (16K)** | Intermediate phase distortion band |
| **Far-Tail Gain Segment** | Positions $> 8192$ (Final 512) | **$-0.039$ NLL (8K)** / **$-0.088$ NLL (16K)** | High-entropy long-range resolution gain |

Rather than a uniform gain, spectrum compression introduces an intermediate "penalty band" just beyond the training window before delivering substantial gains in the deep tail.

### 5.3 Task Saturation vs. True Structural Resolution Headroom
Aggregate benchmarks (e.g., RULER 4-task macro) can severely mislead optimization decisions due to task saturation:
- **Single-Key Retrieval:** Saturated at **$1.00$** accuracy across 8K and 16K lengths in 1.485B models, providing zero discriminatory power.
- **Multikey-3 & Variable Tracking:** Multikey-3 collapses to **$0.00$** at 16K, while variable tracking sits at **$0.03 - 0.05$** at 8K/16K.
- **Diagnostic Lesson:** An operator that dramatically improves multikey discrimination can produce a negligible shift in aggregate macro score if the macro is dominated by saturated single-needle tasks.

---

## 6. Comprehensive Post-Mortem of the 12 Falsified / Closed Routes

`INDEX.md` §3.4 codifies the repository's anti-duplication mechanism. Below is an in-depth breakdown of the 12 closed routes, their failure mechanisms, and structural lessons learned.

```
+-------------------------------------------------------------------------------------------------------+
|                                12 Falsified & Closed Routes Taxonomy                                  |
|                                                                                                       |
|  [Static Geometry Selectors]       [Attention/Fisher Heuristics]      [Underdetermined / Ad-hoc Maps] |
|  - Route 1: Cosine collision       - Route 3: Kappa-att ordering      - Route 6: Direct distance map  |
|  - Route 2: Logdet / Comb          - Route 4: LeRoPE w^(1/3) oracle   - Route 10: 2-doc direct-z      |
|  - Route 5: Arcsine conjecture     - Route 7: D* unrepairable energy  - Route 11: Static single table |
|                                    - Route 8: Coverage residual       - Route 12: Piecewise slope     |
|                                    - Route 9: Phase risk floor                                        |
+-------------------------------------------------------------------------------------------------------+
```

### 6.1 Systematic Enumeration & Mechanism Analysis

#### 1. Cosine-Only Collision Kernel ($C_{\cos}$)
- **Hypothesis:** Positional basis quality can be optimized by minimizing pairwise cosine collision $C_{\cos}(\omega, \nu) = \frac{1}{L}\int_0^L \cos((\omega-\nu)\Delta) d\Delta$.
- **Falsification:** A configuration with lower cosine collision $C_{\cos}(A) < C_{\cos}(B)$ produced a lower block-whitened Renyi-2 rank $r_2(A) < r_2(B)$ (`FULL_ROPE_...20260819.md` §4.3).
- **Failure Mechanism:** Neglects the orthogonal sine components and the full $2K \times 2K$ Gram matrix structure, which requires joint phase-invariant block-whitening.

#### 2. Collision / Logdet Minimization as Extrapolation Objective
- **Hypothesis:** Maximizing log-determinant or minimizing joint collision across all frequencies produces optimal extrapolation.
- **Falsification:** Free unconstrained optimization degenerates into an exact **Fourier comb** ($\omega_k = 2\pi k / L$), where $\Phi(\Delta + L) = \Phi(\Delta)$ produces perfect periodic aliasing.
- **Failure Mechanism:** Ignores the continuous aperiodic distinguishability required outside the window.

#### 3. $\kappa_{\text{att}}$ Attention-Fisher Ordering
- **Hypothesis:** First-order attention-Fisher sensitivity $\kappa_{\text{att}}$ predicts checkpoint susceptibility to table swaps.
- **Falsification:** First-order formula predicted Spearman $\rho = +1.0$ against log PPL, but exact finite-swap subtraction yielded Spearman $\rho = -0.20$ (Branch C failure, `KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md`).
- **Failure Mechanism:** First-order Taylor approximations fail completely under the large finite coordinate displacements ($\Delta \omega$) induced by table swaps.

#### 4. LeRoPE $w^{1/3}$ Structural Curvature Oracle
- **Hypothesis:** High-rate optimal density $\rho^*(\phi) \propto w(\phi)^{1/3}$ under structural softmax logit curvature $w_k = \mathbb{E}[J_k^\top (\text{diag}(p) - pp^\top) J_k]$ predicts the empirically learned LeRoPE profile.
- **Falsification:** Measured structural utility $w(\phi)$ drops by **12 orders of magnitude** from fast to slow bands, driving the predicted profile to concentrate exclusively on fast frequencies (RMSE to LeRoPE = $6.244$, `LEROPE_PROFILE_ORACLE_AUDIT_20260820.md`).
- **Failure Mechanism:** Unsigned local logit sensitivity rewards high-frequency oscillation, whereas actual LM training benefits from suppressing slow-band rotation and creating stable dominant-band trajectories.

#### 5. Arcsine Conjecture
- **Hypothesis:** Heavy-tailed attention distance distributions dictate a U-shaped (arcsine-like) optimal frequency density.
- **Falsification:** Numerical optimization under equal stiffness constraints yielded a strictly monotonic decreasing profile with a fast-band spike and flat tail, not a U-shape (`optimization_notes.md` O5).
- **Failure Mechanism:** The logarithmic kernel in frequency space does not map to a logarithmic Riesz potential in normalized coordinates ($\partial_\phi^4(K\rho) \neq \delta$).

#### 6. Direct Attention-Distance Map (Without Rotary Phase Kernel)
- **Hypothesis:** Frequency demand can be mapped directly from attention distance histograms $D_{\text{att}}(\Delta)$ without convolving with the rotary operator.
- **Falsification:** Direct distance mapping increased extrapolation NLL ($+0.0735$ at 512, $+0.0900$ at 1024), while phase-chord mapping $m_{\text{chord}}(\phi) = \mathbb{E}[1 - \cos(\omega(\phi)\Delta)]$ reduced NLL ($-0.237$ and $-0.152$, `EXPERIMENT_REPORT_20260821.md`).
- **Failure Mechanism:** Attention consumes physical distance solely through rotary phase angles.

#### 7. $D^*$ (Unrepairable In-Window Logit Energy) as Retrofit Target
- **Hypothesis:** Minimizing in-window logit energy $D^*$ that cannot be restored by any linear Q/K map predicts downstream RULER performance.
- **Falsification:** Spearman rank correlation between RULER and $-D^*$ was **$-0.550$**—the sign was completely inverted (`RETROFIT_AXIS_FALSIFICATION_20260822.md`).
- **Failure Mechanism:** $D^*$ is a valid theoretical upper bound on linear emulation, but not an optimization objective for out-of-distribution retrieval.

#### 8. Coverage Residual
- **Hypothesis:** Requiring a single continuous displacement $D'$ to match all channel trajectories simultaneously predicts retrofit quality.
- **Falsification:** Spearman rank correlation against RULER was **$-0.250$** (`RETROFIT_AXIS_FALSIFICATION_20260822.md`).
- **Failure Mechanism:** Fails to account for non-uniform channel weighting in downstream retrieval tasks.

#### 9. Phase Risk ("Already-Wrapped Channels are Safe")
- **Hypothesis:** Channels that completed $\ge 1$ full rotation during pretraining ($\omega_m L_{\text{train}} \ge 2\pi$) can be left unscaled.
- **Falsification:** The `one_turn_floor_s2` operator, engineered specifically to satisfy this criterion with near-optimal $D^* = 0.0192$, scored **$0.0000$ RULER macro** (`RETROFIT_AXIS_FALSIFICATION_20260822.md`).
- **Failure Mechanism:** Slow-to-medium channels that have wrapped once still alias heavily when extended to $2\times - 4\times$ contexts.

#### 10. Direct-$z$ Two-Document Calibration
- **Hypothesis:** Directly optimizing 62 simplex degrees of freedom in $z$ over 2 training documents can calibrate a frozen checkpoint.
- **Falsification:** Optimization reduced loss on design rows but caused significant regression on held-out test rows ($+0.08229$ NLL), violating robustness gates (`DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md`).
- **Failure Mechanism:** 62 continuous degrees of freedom severely overfit when calibrated on minimal document samples.

#### 11. Analytic Zero-Parameter Single-Table Candidates
- **Hypothesis:** A single analytic reallocation (anchored EVQ-Cosh or protected-band Cosh) can serve both $1\times$ and $2\times$ without weight adaptation.
- **Falsification:** Anchored EVQ-Cosh regressed $1\times$ tail NLL by $+3.9780$; protected-band Cosh regressed $1\times$ by $+0.6922$ (`ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md`).
- **Failure Mechanism:** Confirms the universal reality of frozen table-shock; no static single table can bypass co-adaptation.

#### 12. Target-Free Continuous-Boundary-Slope Implementation
- **Hypothesis:** Preserving native phase up to $L_{\text{native}}$ and continuing at reduced slope beyond $L_{\text{native}}$ creates a target-free operator.
- **Falsification:** Scored **$0.0000$ core-4 RULER macro** at both 8K and 16K (`ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md`).
- **Failure Mechanism:** Breaks query-key relative phase shift invariance across the $L_{\text{native}}$ boundary.

---

### 6.2 The Unifying Structural Lesson: The Pitfall of Decoupled Static Selectors
The overarching theoretical lesson uniting all 12 falsifications is summarized in the following structural principle:

> **The Decoupled Functional Fallacy:**
> Every attempt (Routes 1–11) to discover an optimal frequency spectrum by evaluating a scalar functional over a 1D frequency table—decoupled from model weights and assuming isotropic content—inevitably fails because **language model performance is strictly dominated by the bilinear interaction between the positional spectrum $\Omega$ and the learned projection weights $(W_q, W_k)$**.

A mathematical metric that cannot see the learned content phases $\psi_m$ and amplitudes $A_m$ is optimizing a non-dominant term. This is why:
1. Pure table replacement without adaptation fails ($2\times 2$ barrier).
2. Theoretical bounds ($D^*$) invert empirical correlation.
3. The only viable path forward is the **Native-support pure-$z$ matched adaptation paradigm**.

---

## 7. Engineering Recommendations & Preflight Specifications

Based on the synthesis of theoretical limits, operational constraints, and empirical evidence, future research and experimental execution should adhere to the following strict guidelines:

1. **Cease Zero-Training Search:** Terminate all further GPU compute allocated to searching for zero-training frozen table retrofits or scalar table selectors.
2. **Execute the Matched $z \times \text{adaptation}$ $2\times 2$ Matrix:**
   - Pre-freeze candidate $z_{\text{new}}$ using closed-form EVQ-Cosh or phase-chord profiles.
   - Run the 4-arm matched design (Native frozen, $z_{\text{new}}$ frozen, Native + LoRA, $z_{\text{new}}$ + LoRA).
   - Require $z_{\text{new}} + \text{LoRA}$ to strictly beat $\text{Native} + \text{LoRA}$ across $1\times, 2\times, 4\times$ on both NLL and multi-needle RULER tasks.
3. **Enforce Single Static Table Serving:** Validate all models under a single static rotary embedding tensor without runtime routing, temperature gain switching, or coordinate transformation.
4. **Evaluate Multi-Key & Chained Tasks:** Replace single-key NIAH with multi-key retrieval (multikey-2, multikey-3) and variable tracking to ensure non-saturated, high-resolution diagnostic signals.
