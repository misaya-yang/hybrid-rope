# Canonical Empirical Synthesis & Causal Grounding (R4 Axis A)

**Author:** `explorer_r4_empirical_1`  
**Date:** 2026-09-01  
**Status:** Complete / Authoritative Empirical Synthesis  
**Scope:** Rigorous synthesis of repository empirical evidence, causal identification of RoPE exponent allocation $z$, co-adaptation diagnostics, systems scale breadth, zero-training retrofit mechanics, and strict enforcement of claim ceilings and locked nomenclature.

---

## 1. Executive Summary & The Empirical Evidence Pyramid

The central empirical thesis of the RoPE spectral budget research is that **RoPE's frequency multiset is an active coordinate system during training rather than a transparent hyperparameter or post-hoc plug-in**. Under the causal variable decomposition:
$$
x_k = -\log \omega_k = a + R z_k, \quad k = 0, \dots, K-1, \quad z_k \in [0, 1]
$$
where $a = -\log \omega_0$ sets the fast frequency anchor, $R = \log(\omega_0/\omega_{K-1}) = \frac{K-1}{K}\log B$ defines the total log-frequency span, and $z_k$ represents the normalized interior allocation curve, empirical results establish a rigorous hierarchy of evidence:

```
                  ┌─────────────────────────────────────────┐
                  │       Mature Checkpoint Retrofit        │
                  │   OLMo-2 1B (0.6047 vs 0.0056 @16K)    │
                  │   Qwen 1.5B (0.6650 vs 0.5775 @64K)     │
                  │   RULER core-4: 0.5825@8K / 0.4000@16K  │
                  └────────────────────┬────────────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  │       Scale & Systems Evidence          │
                  │   432M MLA (3-seed, 8K→32K flagship)   │
                  │   750M Continue (16K PPL -45.9%, AR 77%)│
                  │   1.485B OLMo-2 scratch (2.1B tokens)   │
                  │   8B LLaMA-3 LoRA (32K NLL -2.048)      │
                  │   129.6M Video-DiT (3D RoPE breadth)    │
                  └────────────────────┬────────────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  │  Causal Identification & Co-Adaptation  │
                  │   151.9M 3-Seed Exact-Range (3/3 seeds) │
                  │   50.9M M4 Factorial (192 runs, 12 cfg) │
                  │   50M 2x2 Crossing (PPL 7.14 → 76.20)   │
                  │   151.9M 2-Seed Crossover (Δ 3.400 NLL) │
                  └─────────────────────────────────────────┘
```

1. **Causal Identification:** Pure interior allocation $z$ is causally consequential at fixed support $(a, R)$ across independent training seeds (151.9M 3-seed log-loss delta $+0.026 / -0.281 / -0.176 / -0.146$ at $256/512/1\text{K}/2\text{K}$; M4 50.9M 192-run factorial).
2. **Co-adaptation Diagnostic:** Attention weights $W_q, W_k$ co-adapt to the frequency coordinate system during training. Swapping tables post-hoc without weight adaptation causes catastrophic loss collapse ($7.14 \to 76.20$ PPL at 50M; $3.400 / 3.251$ NLL crossover at 151.9M), decoupling static basis rank $r_2$ ($4.57 \to 12.54$) from LM task performance.
3. **Systems & Scale Breadth:** Non-geometric reallocation provides massive long-range extrapolation benefits across 432M MLA scarce-channel attention, 750M continued pretraining, 1.485B from-scratch training, 8B LoRA adaptation, and 129.6M Video-DiT 3D spatio-temporal attention.
4. **Retrofit & Readout Mechanics:** In mature frozen checkpoints, non-geometric allocation restores long-range attention ordering when coupled with length-conditioned amplitude scaling, while a label-free linear movement-profile ramp achieves equivalent macro performance ($0.6104$ vs $0.6047$ on OLMo-2 16K), demonstrating that the primary practical mechanism is the identification of the model-relative split point rather than a bespoke non-linear operator.

---

## 2. Causal Identification of Pure Exponent Allocation $z$

### 2.1 The Fixed-Support Causal Decomposition
In standard RoPE with base $B$ and rotary pairs $K = d_{\text{head}}/2$:
$$
\omega_k = B^{-2k/d_{\text{head}}} = \exp\left(-\frac{k}{K}\log B\right), \quad k = 0, \dots, K-1.
$$
The sampled extrema are $\omega_0 = 1$ and $\omega_{K-1} = B^{-(K-1)/K}$. Standard geometric spacing fixes $z_k = \frac{k}{K-1}$.

To determine whether the interior allocation curve $z_k$ has an independent causal effect beyond scalar base dilation (which shifts $\omega_{K-1}$ and $R$) or endpoint truncation, the **Exact-Range protocol** strictly fixes:
- $\omega_0 = 1$ (fastest frequency),
- $\omega_{K-1} = B^{-(K-1)/K}$ (slowest frequency),
- $R = \log(\omega_0 / \omega_{K-1}) = \frac{K-1}{K} \log B$ (log-frequency span),
- Architecture, weight initialization, batch row order, optimizer state, token budget, and evaluation anchors.

Only the positions of the $K-2$ interior frequencies are modulated via normalized quantiles:
$$
s_k = \frac{\phi_\tau\left(\frac{k+1/2}{K}\right) - \phi_{\tau,0}}{\phi_{\tau,K-1} - \phi_{\tau,0}}, \quad \omega_k = \exp(-R s_k).
$$

---

### 2.2 151.9M 3-Seed Primary Replication (`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`)

- **Model Specification:** 151,898,880 parameters, 12 layers, hidden dimension 768, 12 heads, $d_{\text{head}}=64$, $K=32$ rotary pairs ($K-2=30$ interior frequencies).
- **Training Setup:** Training length $L_{\text{train}} = 256$ tokens; budget 499,974,144 tokens (7,629 optimizer steps); dataset FineWeb-Edu; batch size 256.
- **Seeds:** 42, 137, 256.
- **Intervention:** Anchored EVQ-Cosh at $\tau=4$ vs paper-faithful FMRoPE at base 256.
- **Metric:** Paired final-128-token teacher-forced NLL across 32 frozen evaluation anchors at lengths $256, 512, 1024, 2048$.

#### Fixed-Range Primary Contrasts (Anchored EVQ-Cosh minus FMRoPE NLL)
*Negative values favor Anchored EVQ-Cosh.*

| Length | Seed 42 | Seed 137 | Seed 256 | Mean Delta | SD (seeds) | 95% Student-$t$ CI ($df=2$) | EVQ Wins | $\exp(\text{Mean})-1$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **256 ($1\times$)** | $+0.03276$ | $+0.02903$ | $+0.01680$ | **$+0.02619$** | $0.00835$ | $[+0.00545, +0.04693]$ | 0/3 | $+2.65\%$ |
| **512 ($2\times$)** | $-0.47750$ | $-0.27050$ | $-0.09418$ | **$-0.28073$** | $0.19186$ | $[-0.75735, +0.19589]$ | **3/3** | **$-24.48\%$** |
| **1,024 ($4\times$)** | $-0.20499$ | $-0.19088$ | $-0.13211$ | **$-0.17599$** | $0.03865$ | $[-0.27202, -0.07997]$ | **3/3** | **$-16.14\%$** |
| **2,048 ($8\times$)** | $-0.11284$ | $-0.18083$ | $-0.14347$ | **$-0.14571$** | $0.03405$ | $[-0.23029, -0.06114]$ | **3/3** | **$-13.56\%$** |

#### Anchor-Level Consistency (Fraction of 32 anchors favoring Anchored EVQ-Cosh)
| Length | Seed 42 | Seed 137 | Seed 256 |
|:---:|:---:|:---:|:---:|
| 256 | 31.25% | 34.38% | 40.62% |
| 512 | 100.00% | 93.75% | 65.62% |
| 1,024 | 84.38% | 78.12% | 75.00% |
| 2,048 | 68.75% | 84.38% | 78.12% |

**Empirical Finding:**
1. At identical sampled support and log-span, modulating only the 30 interior frequencies shifts trained-model behavior systematically: a small in-domain tax ($+0.026$ NLL / $+2.65\%$) yields consistent out-of-domain gains across all three independent seeds ($3/3$ win rate at $2\times, 4\times, 8\times$).
2. The effect exhibits length heterogeneity: large, variable gain at $2\times$ ($-0.281$ NLL) stabilizing to $-0.176$ at $4\times$ and $-0.146$ at $8\times$.

---

### 2.3 M4 50.9M Pure-Shape Factorial (`M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md`)

To verify whether this effect generalizes across architectural hyperparameters and whether Cosh is uniquely optimal, a 192-run factorial was executed on Apple M4 Max across:
- 2 Base frequencies ($500\text{K}, 1\text{M}$),
- 2 Training lengths ($L_{\text{train}} = 256, 1024$),
- 3 Head dimensions ($d_{\text{head}} = 32, 64, 128 \implies K=16, 32, 64$),
- 3 Seeds ($42, 137, 256$),
- 5 Arms: Native Geo, Anchored Cosh $0.75\times$, Anchored Cosh $1.00\times$ (rule), Anchored Cosh $1.25\times$, and Anchored Exponential (RMS deformation-matched).

#### Weighted OOD NLL ($\bar{N}_{\text{OOD}} = \frac{\sum_{r \in \{2,4,8\}} \log_2(r+1) N_{rL}}{\sum \log_2(r+1)}$)
| Training Schedule | Mean Weighted OOD NLL | Equivalent PPL | Delta vs Geo | Configs Beating Geo |
|---|:---:|:---:|:---:|:---:|
| Native Geo | 6.211973 | 498.68 | 0.000000 | — |
| Anchored Cosh $0.75\times$ | 6.202858 | 494.16 | **$-0.009115$** | 8/12 |
| Anchored Cosh $1.00\times$ (Rule) | 6.202094 | 493.78 | **$-0.009879$** | 7/12 |
| Anchored Cosh $1.25\times$ | **6.199873** | **492.69** | **$-0.012100$** | **10/12** |
| Anchored Exponential (RMS-matched) | 6.201354 | 493.42 | **$-0.010619$** | 9/12 |

#### Pairwise Contrasts & Exact Permutation Tests (12 Configs as Statistical Unit)
| Contrast | Mean Delta NLL | 95% Bootstrap CI | Negative Configs | Exact Sign-Flip $p$-value |
|---|:---:|:---:|:---:|:---:|
| Cosh $0.75\times$ vs Geo | $-0.009115$ | $[-0.017780, -0.001006]$ | 8/12 | $p = 0.0815$ |
| Cosh Rule vs Geo | $-0.009879$ | $[-0.021040, +0.001540]$ | 7/12 | $p = 0.1250$ |
| Cosh $1.25\times$ vs Geo | **$-0.012100$** | **$[-0.020828, -0.002945]$** | **10/12** | **$p = 0.0273$** |
| Exponential vs Geo | $-0.010619$ | $[-0.020808, -0.000696]$ | 9/12 | $p = 0.0708$ |
| Cosh Rule vs Exponential | **$+0.000740$** | **$[-0.005484, +0.007487]$** | 7/12 | **$p = 0.8364$** |

**Decisive Mechanistic Inferences:**
1. **Interior Allocation is an Independent Design Variable:** All non-uniform schedules beat Geo on average under exact endpoint and log-span pinning.
2. **Cosh is Not Uniquely Optimal:** Rule Cosh vs deformation-matched Exponential yields $+0.00074$ NLL ($p=0.8364$, CI spans zero). Any smooth reallocation concentrating channel density toward slow modes captures the benefit.
3. **The Static $\tau = d_{\text{head}}/\sqrt{L_{\text{train}}}$ Rule is a Basin Prior, Not a Global Optimum:** The rule point won only 4/12 configurations ($1.25\times$ won 6/12, $0.75\times$ won 2/12) with average regret $0.01144$ NLL. Boundary sweeps ($\tau=1 \to 1.5\times$ better; $\tau=8 \to 0.75\times$ better) confirm that the heuristic formula over-expands $\tau$ at the extremes.

---

## 3. Target-Matched Deployment Boundaries & Non-Linear Support Interaction

When the same 151.9M models are evaluated under **Target-Matched FMRoPE** (where the frequency base is dynamically adjusted to match the target evaluation context $L_{\text{target}}$):

#### Target-Matched Contrasts (Anchored EVQ-Cosh minus Target-Matched FMRoPE NLL)
*Positive values favor Target-Matched FMRoPE.*

| Length | Seed 42 | Seed 137 | Seed 256 | Mean Delta | SD (seeds) | 95% Student-$t$ CI ($df=2$) | EVQ Wins | $\exp(\text{Mean})-1$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **256 ($1\times$)** | $+0.03276$ | $+0.02903$ | $+0.01680$ | $+0.02619$ | $0.00835$ | $[+0.00545, +0.04693]$ | 0/3 | $+2.65\%$ |
| **512 ($2\times$)** | $+0.06109$ | $+0.05257$ | $+0.06729$ | **$+0.06032$** | $0.00739$ | $[+0.04196, +0.07867]$ | **0/3** | $+6.22\%$ |
| **1,024 ($4\times$)** | $+0.18182$ | $+0.15560$ | $+0.34417$ | **$+0.22720$** | $0.10214$ | $[-0.02654, +0.48094]$ | **0/3** | $+25.51\%$ |
| **2,048 ($8\times$)** | $+0.27857$ | $+0.39924$ | $+0.70096$ | **$+0.45959$** | $0.21757$ | $[-0.08087, +1.00006]$ | **0/3** | $+58.34\%$ |

### Causal Synthesis on Support vs Allocation
1. **Complete Reversal of Order:** While fixed-support interior allocation $z$ wins $3/3$ seeds under fixed range, target-matched support expansion beats fixed-support allocation in $3/3$ seeds at every OOD length ($+0.060 / +0.227 / +0.460$).
2. **Support $(a, R)$ and Interior Allocation $z$ are Interacting Coordinates:** Support displacement is the primary coarse lever for context expansion. Fixed allocation $z$ optimizes basis utilization within a given support but does not substitute for dynamic support scaling when external target context lengths are supplied.
3. **No Additive Synergy Claim:** The data does not support an additive or monotonic gain when splicing interior reallocation onto dynamic target-matched support expansion.

---

## 4. Co-adaptation Diagnostics & Readout Mechanics

### 4.1 The 50M $2\times 2$ Table-Weight Crossing (`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`)

A fundamental question in position coding is whether RoPE frequencies act as an independent geometric prior or whether Transformer projection weights $W_q, W_k$ tightly co-adapt to the frequency coordinates during training.

- **Setup:** 50M decoder-only Transformer, 6 layers, 8 heads, $d_{\text{head}}=64$, $L=512$, Base 500K, seed 42. Evaluated on TinyStories validation across 1,920 head-query observations. Parameters completely frozen (CPU deterministic evaluation).

#### Full $2\times 2$ Crossing Matrix
| Trained Weights | Runtime Frequency Table | LM Loss | Perplexity (PPL) | Static Geometry $r_2$ | Bare Softmax $r_2$ | Content Jacobian $r_2 / \text{trace}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Geo** | **Geo** | **1.9659** | **7.14** | 4.57 | 14.86 | 10.20 / $1.99 \times 10^{-8}$ |
| **Geo** | **EVQ** | **4.3333** | **76.20** | **12.54** | **25.51** | 1.48 / $2.93 \times 10^{-6}$ |
| **EVQ** | **Geo** | **3.1378** | **23.05** | 4.57 | 16.94 | 8.34 / $1.12 \times 10^{-7}$ |
| **EVQ** | **EVQ** | **1.9685** | **7.16** | **12.54** | 21.91 | 12.65 / $1.69 \times 10^{-8}$ |

#### Factorial ANOVA Decomposition on LM Loss
Let cell indices be $GG, GE, EG, EE$:
$$
E_T = \frac{y_{GE} + y_{EE} - y_{GG} - y_{EG}}{2} = +0.5991 \quad (95\% \text{ CI: } [+0.331, +0.956])
$$
$$
E_W = \frac{y_{EG} + y_{EE} - y_{GG} - y_{GE}}{2} = -0.5965 \quad (95\% \text{ CI: } [-0.896, -0.175])
$$
$$
I_{T \times W} = y_{EE} - y_{EG} - y_{GE} + y_{GG} = \mathbf{-3.5367} \quad (95\% \text{ CI: } [-5.165, -3.039])
$$

**Critical Diagnostic Insights:**
1. **The Interaction Dominates ($|I_{T \times W}| \approx 5.9 \times \text{Main Effects}$):** Both self-consistent systems perform identically (PPL $7.14$ vs $7.16$). Mismatching the table at runtime causes catastrophic perplexity degradation ($7.14 \to 76.20$ and $7.16 \to 23.05$).
2. **Falsification of Static Subspace Geometry as an LM Predictor:** In the worst-performing cell ($\text{Geo weights} + \text{EVQ table}$, PPL $76.20$), the bare static basis rank $r_2$ jumps from $4.57$ to $\mathbf{12.54}$. Static basis rank measures phase diversity in isolation; it does not guarantee LM perplexity if trained readout projections cannot resolve the scrambled phase combinations.
3. **Severe Readout Disruption:** Swapping the table blows up the frequency empirical Fisher trace by a factor of $\mathbf{4,494\times}$ ($8.80 \times 10^{-7} \to 3.95 \times 10^{-3}$), demonstrating that frozen weights experience extreme gradient shock under unadapted coordinate shifts.

---

### 4.2 151.9M Weights-by-Table Replication (`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`)

The co-adaptation interaction was independently replicated at 151.9M across two training seeds (137 and 256) at length 1024:

| Frozen Trained Weights | FMRoPE-derived Table | Cosh-derived Table | Same-Support Geometric |
|:---:|:---:|:---:|:---:|
| **FMRoPE-trained** | **3.426** | 5.776 | 3.429 |
| **Anchored-Cosh-trained**| 4.455 | **3.479** | 4.177 |

- **Crossover Interaction:** $[L(W_F, T_C) - L(W_F, T_F)] - [L(W_C, T_C) - L(W_C, T_F)] = \mathbf{3.400}$ NLL (Seed 137) and $\mathbf{3.251}$ NLL (Seed 256) at 1K.
- **Mechanism:** FMRoPE weights demand FMRoPE tables; Cosh weights demand Cosh tables. This proves that from-scratch training and post-hoc frozen retrofit represent fundamentally distinct causal estimands.

---

## 5. Scale & Systems Evidence (Breadth & Persistence)

The repository provides comprehensive multi-scale, multi-system evidence verifying the behavior of spectrum reallocation across architectures, scales, and modalities.

### Summary of Systems & Scale Evidence
| Protocol | Scale / Architecture | Hardware / Stack | Key Results | Causal Scope / Ceilings |
|---|---|---|---|---|
| **432M MLA Flagship** | 432M Multi-Head Latent Attention, $d_{\text{eff}}=128$, $K=16$, 3 seeds (42, 43, 88) | GPU Cluster | 8K train $\to$ 32K eval.<br>16K PPL: Geo $138.81 \to$ EVQ $\mathbf{95.59}$ ($-31.1\%$).<br>+YaRN(s=4) @32K: Geo $278.50 \to$ EVQ $\mathbf{236.59}$ | Scarce-channel systems flagship; $\tau=1.414$ is operating convention, not global optimum |
| **750M Continuation** | 750M Decoder-only, 2K $\to$ 4K continuation, 500M tokens, Seed 42 | Server R6000 | 2K/4K PPL cost ($+0.9\% / +1.5\%$).<br>8K PPL: $23.39 \to 19.61$ ($-16.2\%$).<br>16K PPL: $45.14 \to \mathbf{24.41}$ ($\mathbf{-45.9\%}$).<br>8K Passkey AR Exact: $0\% \to \mathbf{77.5\%}$ | Single-seed training persistence; retrieval saturates (100%), AR exact match resolves |
| **1.485B OLMo-2 Scratch** | 1.485B OLMo-2 architecture, 4K context, 2.097B tokens (Step 1,000) | Single-GPU HF loop vs AI2 distributed | Step-1,000 128-doc PG-19 PPL:<br>2K: $+7.51\%$, 4K: $+3.88\%$,<br>8K: $\mathbf{-4.28\%}$, 16K: $\mathbf{-12.64\%}$ ($182.73 \to \mathbf{159.64}$).<br>16K Tail PPL: $214.63 \to \mathbf{172.60}$ ($128/128$ docs favor EVQ) | Same-initialization/same-recipe from scratch; trainer stack confound prevents bitwise paired claim |
| **8B Mature Adaptation** | LLaMA-3-8B-Instruct, 300 steps rank-64 LoRA, Seed 42 | Server GPU | Temporal NLL (24 packs):<br>8K: $+0.390$, 16K: $\mathbf{-1.510}$, 32K: $\mathbf{-2.048}$.<br>16K Gold Block Ablation: NLL shock $+1.5055$ (Native $-0.0095$).<br>16K Hit@16: $18.75\% \to \mathbf{64.06\%}$ | Mature model probability & causal routing evidence; top-1 generation conversion remains open (0% exact) |
| **129.6M Video-DiT** | 129.6M 3D Spatio-Temporal DiT, Oscillating Moving MNIST, Seed 42 | MPS / Float32 | Denoising MSE:<br>Train frames (32): $-20.95\%$.<br>All extrapolated (128): $-16.29\%$.<br>Far extrapolated: $\mathbf{-35.42\%}$ ($0.00989 \to \mathbf{0.00639}$) | Cross-modal supporting breadth; single seed-42, no training-seed uncertainty |

---

## 6. Mature Checkpoint Retrofit & Length-Conditioned Mechanics

### 6.1 Same-Support Control in Frozen Checkpoints (`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`)

When testing zero-training retrofit on mature models (OLMo-2-1B and Qwen2.5-1.5B), holding support $(a, R)$, amplitude $a(s)$, and KV cache coordinates constant reveals the exact causal contribution of interior $z$:

#### Macro Performance Under Frozen Interventions
| Checkpoint & Benchmark | Installed Frequency Table | Macro Score | Contrast vs Geometric | 95% Paired Row Bootstrap CI |
|---|---|:---:|:---:|:---:|
| **OLMo-2-1B-Instruct**<br>Native 4K $\to$ Unseen-9 RULER @16K | Official YaRN factor 4<br>Same-support Geometric<br>**Derived Allocation**<br>Nearest Movement-Profile Ramp | $0.0794$<br>$0.0056$<br>$\mathbf{0.6047}$<br>$0.6104$ | —<br>—<br>**$+0.5992$**<br>$+0.6048$ | —<br>—<br>$[+0.5488, +0.6480]$<br>$[+0.5559, +0.6542]$ |
| **Qwen2.5-1.5B**<br>Native 32K $\to$ Core-4 RULER @64K | Native<br>Official YaRN factor 4<br>Same-support Geometric<br>Nearest Movement-Profile Ramp<br>**Corrected Derived Allocation** | $0.5450$<br>$0.6025$<br>$0.5775$<br>$0.6400$<br>$\mathbf{0.6650}$ | —<br>—<br>—<br>$+0.0625$<br>**$+0.0875$** | —<br>—<br>—<br>$[-0.0325, +0.1075]$<br>$[+0.0025, +0.1750]$ |
| **Qwen2.5-1.5B**<br>Native 32K $\to$ Benchmark @128K | Native<br>Official YaRN factor 4<br>Old Aliased Derived Table<br>Same-support Geometric<br>**Corrected Derived Allocation** | $0.4350$<br>$0.4650$<br>$0.6175$<br>$0.4550$<br>$\mathbf{0.5400}$ | —<br>—<br>*(superseded)*<br>—<br>**$+0.0850$** | —<br>—<br>*(numerical artifact)*<br>—<br>$[-0.0100, +0.1800]$ |

**Key Findings:**
1. **Geometric Collapse on OLMo:** Standard geometric interpolation at matched support fails completely ($0.0056$), whereas non-geometric derived allocation achieves $0.6047$.
2. **Profile Uniqueness is Not Necessary:** The label-free nearest movement-profile ramp achieves $0.6104$ on OLMo and $0.6400$ on Qwen. The derived-minus-ramp CI spans zero on both models (OLMo: $[-0.0464, +0.0345]$; Qwen: $[-0.0525, +0.1000]$).
3. **The Real Value of Spectral Redundancy:** The scientific contribution is deriving the **model-relative split boundary** from spectral redundancy, not claiming that a specific mathematical curve defines a novel operator family.
4. **Correction of Qwen 128K Aliasing:** The stride-16 aliasing artifact that inflated Qwen 128K to $0.6175$ was eliminated; the true corrected result is $\mathbf{0.5400}$.

---

### 6.2 Length-Conditioned Budgeted Retrofit (`LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md`)

- **Architecture:** Zero learned parameters; Native path for $L \le 4096$; for $s = L_{\text{target}}/L_{\text{train}} > 1$, uses:
$$
\text{move}_k = (1 - \text{uniqueness}_k)^2, \quad \omega'_k = \omega_k(1 - \text{move}_k) + \frac{\omega_k}{s} \text{move}_k
$$
combined with matched length-only attention amplitude $a(s) = 1 + 0.1 \log s$.

#### Core-4 RULER Results on OLMo-2-1B-Instruct
| Operator | 8K ($s=2$) | 16K ($s=4$) |
|---|:---:|:---:|
| Native | 0.0000 | 0.0000 |
| Native + Matched Amplitude | 0.0000 | 0.0000 |
| Budgeted Frequency Only | 0.4000 | 0.1150 |
| Official Transformers YaRN | 0.5375 | 0.0125 |
| **Budgeted Frequency + Matched Amplitude** | **0.5825** | **0.4000** |

#### Two-Factor Non-Additive Interaction Decomposition
$$
\text{Interaction} = \text{Joint} - \text{Freq\_Only} - \text{Amp\_Only} + \text{Native}
$$
- **At 8K:** $0.5825 - 0.4000 - 0.0000 + 0.0000 = \mathbf{+0.1825}$
- **At 16K:** $0.4000 - 0.1150 - 0.0000 + 0.0000 = \mathbf{+0.2850}$

**Mechanistic Grounding:**
Frequency reallocation restores the distinguishability of positional ordering across long contexts, and amplitude scaling counters attention dilution once that ordering is restored. Neither component functions in isolation.

---

## 7. Strict Verification of Claim Ceilings & Epistemic Boundaries

To prevent over-claiming and maintain absolute scientific rigor for ICLR 2027, all empirical claims are audited against `AGENTS.md`:

| Domain | Claim Ceiling Rule | Verified Evidence Status | Strictly Prohibited Language |
|---|---|---|---|
| **Full-RoPE Geometry** | Static, phase-invariant positional-basis redundancy / effective dimension; not an LM-quality or extrapolation predictor | $r_2$ reaches $12.54$ in the worst PPL cell ($76.20$), decoupling geometry from loss | Do not claim "higher $r_2$ implies better language modeling" |
| **Low-Frequency Collapse** | Slow bands are redundant in stated metric; not necessarily unused or reclaimable | Slow modes collapse to $\text{span}\{1, \Delta\}$ in $L_2$ and $\text{span}\{\Delta, \Delta^2\}$ in softmax | Do not claim slow frequencies are completely dead or waste capacity |
| **Frozen Retrofit** | Exact fixed invertible Q/K compensation is obstructed for unequal frequency multisets | Proved in Thm 3; empirical 50M/151.9M crossings show coordinate co-adaptation | Do not claim zero-training retrofit is mathematically lossless across all tasks |
| **Cosh Function** | Unique only for the stated convex surrogate | M4 factorial shows Cosh $\approx$ Exponential ($p=0.8364$) | Do not claim Cosh is universally or uniquely optimal across all tasks |
| **Finite Tau ($\tau$)** | Fallible zero-search operating prior relative to tested baselines; discrete grids do not establish continuous basin or global optimum | M4 sweep shows rule $\tau$ won only 4/12 configs; regret $0.0114$ NLL | Do not claim $\tau = d_{\text{head}}/\sqrt{L_{\text{train}}}$ is an optimal universal scaling law |
| **Exact-Range** | Pure allocation identification at fixed sampled support | 151.9M 3-seed shows $+0.026 / -0.281 / -0.176 / -0.146$ ($3/3$ seeds) | Do not claim additive synergy with target-matched support expansion |
| **Mature Studies** | Protocol-specific persistence / capability evidence; no pooled effects or cross-protocol control splicing | Evaluated on specific checkpoints (OLMo-2, Qwen2.5, LLaMA-3) under declared protocols | Do not pool results across distinct evaluation contracts or claim generic SOTA |
| **Passkey** | Teacher-forced NLL gap unless an owner explicitly states autoregressive exact match | 750M continuation reports AR exact match ($0\% \to 77.5\%$); 8B LoRA is NLL + routing only | Do not report passkey retrieval rate as autoregressive generation capability |
| **RULER / NIAH** | Task-family adaptation, not unseen-task transfer | OLMo-2 selective QK adaptation improves RULER family macro ($11.09\% \to 42.44\%$) | Do not claim out-of-distribution reasoning transfer to arbitrary tasks |
| **LeRoPE Oracle** | Unsigned structural-curvature $w^{1/3}$ profile failed as a predictor of published LeRoPE shape | Internal audit confirmed negative result (`LEROPE_PROFILE_ORACLE_AUDIT_20260820.md`) | Never promote $w^{1/3}$ curvature formula as a valid predictive oracle |

---

### Locked Nomenclature Compliance Table
| Symbol / Term | Permitted Repository Usage | Prohibited Aliases / Misnomers |
|---|---|---|
| `Geo` | Geometric-table baseline in from-scratch or continued training | Do not call `Native` when training from scratch |
| `Native` | Unmodified model-native RoPE/checkpoint in pretrained adaptation or retrofit | Never interchangeable with `Geo` |
| `FMRoPE` | Exact-range paper-faithful FMRoPE arm (keep exponent rule as protocol detail) | Never create prefixed aliases or call it `Native` |
| `anchored \evq{}` / `anchored EVQ-Cosh` | Exact-range EVQ-Cosh quantiles normalized to FMRoPE endpoints | Do not use ambiguous variants |
| `\rs{}` / `YaRN-style` | Repository fixed-index operator preserving fast bands & scaling slow bands | Do not claim exact/tuned reproduction of cited YaRN |
| `MLA wavelength-blend operator` | Run-specific MLA operator | Never call `YaRN-style`, `RAMP`, or `legacy scaler` |
| `\evq{}` / `EVQ-Cosh` | Proposed fixed table | Do not present as learned or dynamic |

---

### Falsified Routes Catalog (Anti-Repetition Lock)
1. **Cosine-Only Collision Kernel:** Falsified because $C_{\cos}(A) < C_{\cos}(B)$ but $r_2(A) < r_2(B)$ (`full_rope_audit`).
2. **Collision / Log-Det Minimization as Extrapolation Objective:** Degenerates into a Fourier comb with $\Phi(\Delta + L) = \Phi(\Delta)$ exact aliasing.
3. **$\kappa_{\text{att}}$ Attention-Fisher Ordering:** Produced conflicting rankings (Branch C audit).
4. **LeRoPE $w^{1/3}$ Curvature Oracle:** Falsified as a predictor of published LeRoPE shapes.
5. **Arcsine Conjecture:** Equal-stiffness free optimization is non-U-shaped (optimization notes O5).
6. **Direct Attention-Distance Mapping (Without Phase Kernel):** Closed with `STOP_DIRECT_DISTANCE_MAP`.
7. **$D^*$ Metric as Retrofit Design Objective:** Anti-correlated with downstream performance (Spearman $-0.55$).
8. **Coverage Residual & Phase Risk:** Failed with Spearman $-0.25$ and $0.000$; decisive counterexample in `one_turn_floor_s2`.

---

## 8. Actionable Guidance for Manuscript & Revision Architecture

1. **Unify the Causal Narrative:**
   - In Section 2 / Figure 1, present the **Exact-Range 151.9M 3-seed result** as the definitive proof of pure interior allocation $z$ as a training-time design axis.
   - Contrast it with the **Target-Matched boundary** in the discussion/appendix to clearly demarcate the boundary between interior reallocation and support dilation.
2. **Ground Co-Adaptation Rigorously:**
   - Include the **50M $2\times 2$ table-weight crossing** and the **151.9M 2-seed replication** directly alongside the post-hoc transplant obstruction theorem.
   - Emphasize the decoupling of static basis rank ($r_2 = 12.54$) from task perplexity ($\text{PPL} = 76.20$) to preempt reviewer questions regarding pure geometric optimization.
3. **Present Retrofit as Model-Relative Split Identification:**
   - Report the OLMo-2 ($0.6047$ vs $0.0056$) and Qwen2.5 ($0.6650$ vs $0.5775$) results as practical validations that spectral redundancy identifies useful split boundaries in standard interpolation families, rather than claiming a bespoke non-linear operator family.
   - Maintain the corrected Qwen 128K baseline ($0.5400$) and explain the removal of numerical stride aliasing.
4. **Maintain Strict Claim Ceilings:**
   - State all results within their verified boundaries (e.g., 3-seed descriptive uncertainty, single-seed 8B adaptation, single-seed Video-DiT cross-modal breadth).
   - Strictly separate teacher-forced NLL, routing/attention hit rates, and autoregressive exact generation match.
