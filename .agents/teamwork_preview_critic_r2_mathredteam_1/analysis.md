# Comprehensive Mathematical Red Team Audit: Zero-Training RoPE Retrofit Theory

- **Author:** Mathematical Red Team (R2)
- **Date:** 2026-09-03
- **Status:** Complete Adversarial Mathematical Audit
- **Working Directory:** `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_critic_r2_mathredteam_1/`
- **Formatting Contract:** All substantive assertions are strictly tagged with `[OBSERVED]`, `[DERIVED]`, `[HYPOTHESIS]`, or `[UNKNOWN]`.
- **Target Corpus:**
  - `paper-2027/sections/03_theory.tex` & `paper-2027/appendix/a1_proofs.tex`
  - `paper-2027/research/foundations/`
  - `paper-2027/research/attention-aware-retrofit/theory/`
  - `rebuttal/rebuttal_0723/theory_results/`

---

## 1. Executive Summary & Verdict

`[DERIVED]` The theoretical corpus supporting RoPE frequency re-allocation and zero-training retrofit contains a sharp dichotomy between:
1. **Rigorous linear-algebraic and variational identities** on idealized geometric objects (e.g., block-whitened Gram traces, Lie generator similarity, convex-hull projections, and Euler-Lagrange minimizers of quadratic surrogates); and
2. **Behavioral extrapolations and causal interpretations** that repeatedly commit category errors by conflating static positional basis geometry with trained neural network readout dynamics.

`[OBSERVED]` The foundational documents (`paper-2027/sections/03_theory.tex`, `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md`, `COMMON_DIRECTION_FEASIBILITY_AND_BASIN_BARRIER_THEORY_20260902.md`) make significant mathematical progress by eliminating discredited routes (such as cosine-only collision kernels, naive collision minimization, the arcsine conjecture, and smooth two-parameter gain models). 

`[DERIVED]` However, the mathematical red team audit reveals five fundamental theoretical vulnerabilities, hidden assumptions, and over-claimed bounds:
1. **The Slot Permutation Paradox (Theorem 3 vs. Reality):** The Post-Hoc Transplant Obstruction Theorem (`thm:obstruction`, Memo T2) proves that exact position-independent linear compensation requires multiset frequency equality *up to permutation*. Yet in a mature frozen checkpoint, permuting frequencies across slots while preserving the exact multiset causes total behavioral collapse (Qwen 64K RULER drops from 0.7000 to 0.0000; OLMo NLL explodes from 3.10 to 6.86). The theorem's necessary condition is dangerously misleading if interpreted as granting permutation invariance, because frozen weights fix the linear map $A = B = I$ and couple specific semantic features to specific rotary slot indices.
2. **The Multi-Layer Argmax Fallacy (Memo T8 & Gain Theory):** The claim that "scalar gain $g > 0$ preserves per-decision argmax and cannot fix retrieval ranking errors" is an exact algebraic identity only for an isolated single-step softmax. In a multi-layer Transformer, attention outputs feed through residual connections into subsequent layers; varying $g$ changes the mixture vector, which non-linearly alters downstream queries and keys. We construct an explicit minimal 2-layer counterexample where scalar attention gain at Layer 1 reverses the argmax ranking at Layer 2.
3. **Vacuity of Compatibility Modulus and Softmax Error Bounds (Memo T3):** The compatibility modulus bound $|s'(\Delta) - s(\Delta)| \le \sum_k |c_k| \min(2, |\omega'_k - \omega_k|\Delta)$ and the associated softmax total variation bound $\|p' - p\|_1 \le e^{2\epsilon} - 1$ are mathematically valid for microscopic local deviations ($\epsilon \ll 0.1$). However, for actual context extrapolation ($S=4, 8$), $|\omega'_k - \omega_k|\Delta \ge 2$ across most channels, yielding logit deviations $\epsilon \sim 15\text{--}20$. The resulting softmax bound yields $\|p' - p\|_1 \le e^{30} - 1 \approx 10^{13}$, which is exponentially vacuous compared to the trivial probability bound $\|p' - p\|_1 \le 2$.
4. **Surrogate Circularity of EVQ-Cosh:** The Cosh density $\rho_\tau(\phi)$ is the unique minimizer of the surrogate functional $\mathcal{C}_{\mathrm{app}}[\rho]$. However, the surrogate kernel $\min(\phi, \psi)$ is not derived from the oscillatory RoPE cross-Gram kernel; it is the 1D Laplacian Green's function chosen specifically to yield an analytically solvable second-order linear ODE with hyperbolic eigenfunctions. Discrete transport bounds for EVQ-Cosh explode as $O(\sinh\tau / (4K\tau))$ and become vacuous for $\tau \ge 5$.
5. **Heuristic Nature of the "Basin Barrier" and "Waterbed Effect":** Neither the "basin barrier" nor the "waterbed effect" is an algebraic impossibility theorem. The "waterbed effect" is a simple definition ($r_k \cdot S^{m_k} = S$). The "bimodal basin" is a model-dependent conjecture predicated on assuming a step-function threshold for long-context capability and a convex bowl for native retention.

---

## 2. Exhaustive Claim Classification & Boundary Taxonomy

`[DERIVED]` Every substantive mathematical claim in the surveyed literature is classified into one of five rigorous epistemic categories:
- **Algebraic Identity:** Exact equality holding by algebraic manipulation without approximation or behavioral assumptions.
- **Local Approximation:** Asymptotic or truncated series valid only in a restricted limit (e.g., $\omega L \to 0$ or small $\tau$).
- **Upper/Lower Bound:** Mathematical inequality with provable direction, subject to explicit tightness conditions.
- **Empirical Regularity / Hypothesis:** Phenomenological observation or model-based conjecture without general mathematical proof.
- **Theorem:** Deductive mathematical result rigorously proven from explicitly stated axioms/conditions.

| Claim ID | Source Document | Stated Mathematical Object | Rigorous Classification | Epistemic Status & Tightness |
| :--- | :--- | :--- | :--- | :--- |
| **C1.1** | Paper §3.1, Thm 1 | $r_2(\Gamma) = \frac{2K}{1 + (K-1)\bar{c}}$ | **Algebraic Identity** | Exact for any block-whitened Gram with $I_2$ diagonal blocks. Static basis only; zero predictive power for LM loss. |
| **C1.2** | Paper §3.2, Prop 1 | $2 - \|Q_{x,y}\|_F^2 = \frac{19}{12600}(x^2 - y^2)^2 + O(\epsilon^6)$ | **Local Approximation** | Asymptotic Taylor expansion as $\epsilon = \max(\omega L, \nu L) \to 0$. Vacuous for fast bands ($\omega L \gg 1$). |
| **C1.3** | Paper §3.2, Prop 1 | Softmax centered limit $\operatorname{span}\{\Delta - \mathbb{E}\Delta, \Delta^2 - \mathbb{E}\Delta^2\}$ | **Local Approximation** | Holds for fixed exogenous distribution $p$. Ignores endogenous coupling where logits depend on $\omega$. |
| **C1.4** | Paper §A.3, Prop 2 | Parity lattice $\omega_k = \pi a_k / L \implies \Gamma = I_{2K}, r_2 = 2K$ | **Theorem** | Exact mutual orthogonality on uniform measure $\mathrm{Unif}[0,L]$. Fails on causal triangular measure. |
| **C2.1** | Paper §3.3, Thm 2 | Unique minimizer of $\mathcal{C}_{\mathrm{app}}[\rho]$ is $\rho_\tau(\phi) = \frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau}$ | **Theorem** | Exact calculus of variations theorem conditional on surrogate $\mathcal{C}_{\mathrm{app}}$. Surrogate kernel $\min(\phi,\psi)$ is an ad-hoc choice. |
| **C2.2** | Paper §A.6, Thm 4 | Surrogate self-consistency: $\tau^2 T_2(\tau) + T_1(\tau) = \tau\coth\tau$ | **Algebraic Identity** | Exact identity connecting energy integrals of the Euler-Lagrange solution. Internal check only. |
| **C2.3** | Paper §A.7, Eq 525 | Scaling balance $\tau_* = c \cdot d_{\mathrm{head}} / \sqrt{L}$ | **Local Approximation / Heuristic** | Relies on small-$\tau$ expansion, diffuse attention prior $p_0 = 1/L$, and arbitrary scalar tradeoff $\lambda$. |
| **C2.4** | Paper §A.8, Eq 571 | Discrete transport bound $W_1(\mu_{K,\tau}, \mu_{\rho_\tau}) \le \frac{\sinh\tau}{4K\tau}$ | **Upper Bound** | Valid for bounded densities, but bound grows exponentially in $\tau$; vacuous for $\tau \ge 5$. |
| **C3.1** | Paper §3.4, Thm 3 | $A^\top \mathcal{R}_{\Omega'}(\Delta) B = \mathcal{R}_\Omega(\Delta) \implies \{|\omega'_k|\} = \{|\omega_k|\}$ | **Theorem** | Exact Lie generator spectral similarity theorem. Covers only constant linear maps; ignores frozen weight-slot binding. |
| **C4.1** | Memo T3 | $|s'(\Delta) - s(\Delta)| \le \sum_k |c_k| \min(2, |\omega'_k - \omega_k|\Delta)$ | **Upper Bound** | Provably tight envelope for worst-case phase alignment. Exponentially loose for large $\Delta$. |
| **C4.2** | Memo T3 | Softmax TV bound $\|p' - p\|_1 \le e^{2\epsilon} - 1$ | **Upper Bound** | Mathematically valid, but becomes vacuous ($> 2$) whenever logit perturbation $\epsilon > \frac{1}{2}\ln 3 \approx 0.55$. |
| **C4.3** | Memo T4 | Score conditioning: $\frac{\sup_{\Delta \le SL} \|\delta s\|}{\sup_{\Delta \le L} \|\delta s\|} \le S$ | **Upper Bound** | First-order linear perturbation bound. Fails when phase displacement wraps modulo $2\pi$. |
| **C4.4** | Memo T5 | Checkpoint non-identifiability via dormant circuits ($O(S^{-2})$ on $L$, $O(1)$ on $SL$) | **Theorem** | Constructive model-class existence theorem. Proves Native-window measurements cannot uniformly bound horizon behavior. |
| **C5.1** | Memo A6 | Per-slot waterbed identity $r_k(S) \cdot S^{m_k} = S$ | **Algebraic Identity** | Trivial definition ($S^{1-m_k} \cdot S^{m_k} = S$). Not a physical conservation law. |
| **C5.2** | Memo T6 | Semigroup composition $F(S_1 S_2) = F(S_2)\circ F(S_1) \implies \omega' = \omega S^{-m}$ | **Algebraic Identity** | Solution of Cauchy power functional equation. Bookkeeping artifact of power-law parameterization; falsified behaviorally. |
| **C5.3** | Memo T7 | Novelty doubling $N_k(2S)/N_k(S) = 2 + \frac{1}{4^{1-m_k}-1} > 2$ | **Algebraic Identity** | Exact arithmetic for frozen $4^{-m_k}$ continuation. Coordinate-count descriptive; does not predict loss. |
| **C6.1** | Common Dir Thm | $\max \gamma - \frac{\lambda}{2}\|d\|^2 \text{ s.t. } g_j^\top d + \gamma \le 0 \implies d^* = -\bar{g}^*/\lambda$ | **Theorem** | Exact QP duality theorem for convex hull minimum norm point. Infinitesimal local gradient scope only. |
| **C6.2** | Common Dir §4 | Bimodal basin structure along dilation coordinate | **Hypothesis** | Model-based conjecture assuming thresholded long-context benefit and smooth convex native tax. |
| **C7.1** | Memo T8 | Scalar gain $g > 0$ preserves argmax $\arg\max_j (g s_{ij}) = \arg\max_j s_{ij}$ | **Algebraic Identity** | Strictly valid for single softmax. Fails in multi-layer networks via hidden state mixture changes. |
| **C8.1** | Target-Free §2 | Eigenvalues $\lambda_\pm(S_\omega) = \frac{1 \pm |\chi_L(2\omega)|}{2}$ under causal prior $p_L$ | **Algebraic Identity** | Exact matrix algebra for rank-2 symmetric block under finite discrete Fourier transform. |
| **C8.2** | MaxEnt §4-5 | Quantiles $r_i(\lambda) = [1 + q_i(s^\lambda - 1)]^{1/\lambda}$ minimize $\sum \omega_i(1 - 1/r_i)$ | **Theorem** | Follows rigorously from Gibbs entropy maximization and the Hardy-Littlewood-Pólya rearrangement inequality. |

---

## 3. Deep-Dive Mathematical Red Team Audits

### Topic 1: Full-RoPE Subspace Geometry, Whitened Cross-Gram & Spectral Budget Identity

#### Mathematical Formulations and Definitions
`[OBSERVED]` In `paper-2027/sections/03_theory.tex` (lines 33–69) and `appendix/a1_proofs.tex` (lines 7–47):
- For frequency $\omega$, the positional subspace is $V_\omega = \operatorname{span}\{\cos(\omega\Delta), \sin(\omega\Delta)\}$.
- For separation measure $\mu$, let $x_\omega(\Delta) = [\cos(\omega\Delta), \sin(\omega\Delta)] \in \mathbb{R}^{1\times 2}$.
- $S_\omega = \mathbb{E}_\mu[x_\omega^\top x_\omega] \in \mathbb{R}^{2\times 2}$, $H_{\omega\nu} = \mathbb{E}_\mu[x_\omega^\top x_\nu] \in \mathbb{R}^{2\times 2}$.
- Whitened cross-Gram: $Q_{\omega\nu} = S_\omega^{-1/2} H_{\omega\nu} S_\nu^{-1/2}$.
- Pairwise collision: $c_{\omega\nu} = \frac{1}{2}\|Q_{\omega\nu}\|_F^2 = \frac{\sigma_1^2 + \sigma_2^2}{2} \in [0,1]$.
- Global correlation matrix $\Gamma \in \mathbb{R}^{2K \times 2K}$ has block entries $\Gamma_{ij} = Q_{\omega_i \omega_j}$, with $\Gamma_{ii} = I_2$.
- Spectral budget identity (Theorem 1):
  $$\operatorname{tr}\Gamma = 2K, \qquad \operatorname{tr}(\Gamma^2) = 2K[1 + (K-1)\bar{c}], \qquad r_2(\Gamma) = \frac{2K}{1 + (K-1)\bar{c}}$$

#### Quantifier and Condition Verification
`[DERIVED]`
- **Quantifiers:** For all $K \in \mathbb{N}$, for any frequency set $\{\omega_1, \ldots, \omega_K\} \subset \mathbb{R}^+$, and for any probability measure $\mu$ on $\mathbb{R}$ such that $S_{\omega_i} \succ 0$ for all $i$.
- **Proof Rigor:** The proof is algebraically exact:
  $$\operatorname{tr}\Gamma = \sum_{i=1}^K \operatorname{tr}(I_2) = 2K$$
  $$\operatorname{tr}(\Gamma^2) = \sum_{i,j} \operatorname{tr}(\Gamma_{ij}\Gamma_{ji}) = \sum_i \operatorname{tr}(I_2) + \sum_{i\ne j} \|Q_{\omega_i\omega_j}\|_F^2 = 2K + \sum_{i\ne j} 2 c_{ij} = 2K + 2K(K-1)\bar{c}$$
  Dividing $(\operatorname{tr}\Gamma)^2 = 4K^2$ by $\operatorname{tr}(\Gamma^2)$ yields $r_2(\Gamma)$.

#### Adversarial Red Team Critique
1. `[DERIVED]` **Identity, Not a Law of Behavior:** Theorem 1 is a property of block-partitioned positive semidefinite matrices with identity diagonal blocks. It does not incorporate learned query-key projections $W_q, W_k$, attention softmax, or sequence loss.
2. `[OBSERVED]` **Empirical Failure of Rank as a Quality Metric:** The paper's own 50M weights-by-table crossing (Table 1) demonstrates that increasing $r_2(\Gamma)$ from $4.57$ to $12.54$ by substituting EVQ under Geo weights causes PPL to degrade catastrophically from $7.14$ to $76.20$.
3. `[DERIVED]` **Measure Sensitivity:** The closed-form entries of $H_{\omega\nu}$ in Eq. (13) assume $\Delta \sim \mathrm{Unif}[0,L]$. In autoregressive Transformer decoders, the true distribution of token separations is causal triangular: $p_L(\Delta) = \frac{2(L-\Delta)}{L^2}$. Changing the measure breaks the closed form in Eq. (13) and changes the numerical value of $r_2$.

---

### Topic 2: Variational Surrogate Functional, Euler-Lagrange Cosh Optimum & Operating Rule

#### Mathematical Formulations and Definitions
`[OBSERVED]` In `paper-2027/sections/03_theory.tex` (lines 112–147) and `appendix/a1_proofs.tex` (lines 304–356):
- Functional: $\mathcal{C}_{\mathrm{app}}[\rho] = \frac{\alpha}{2}\int_0^1 \rho(\phi)^2 d\phi + \frac{\beta}{2}\iint_{[0,1]^2} \rho(\phi)\rho(\psi)\min(\phi,\psi) d\phi d\psi$, with $\alpha > 0, \beta \ge 0$.
- Constraint: $\rho \in L^2([0,1]), \rho \ge 0 \text{ a.e.}, \int_0^1 \rho(\phi) d\phi = 1$.
- Theorem 2 states that the unique minimizer is $\rho_\tau(\phi) = \frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau}$, where $\tau = \sqrt{\beta/\alpha}$.
- Operating rule (Eq. 489): $\tau = c \cdot d_{\mathrm{head}} / \sqrt{L_{\mathrm{train}}}$.

#### Quantifier and Condition Verification
`[DERIVED]`
- **Euler-Lagrange Equation:** With Lagrange multiplier $\nu$ for the mass constraint, the stationarity condition is:
  $$\alpha \rho(\phi) + \beta g(\phi) + \nu = 0, \qquad g(\phi) = \int_0^1 \rho(\psi)\min(\phi,\psi) d\psi$$
- Since $g(\phi) = \int_0^\phi \psi \rho(\psi) d\psi + \phi \int_\phi^1 \rho(\psi) d\psi$, Leibniz rule yields:
  $$g'(\phi) = \int_\phi^1 \rho(\psi) d\psi, \qquad g''(\phi) = -\rho(\phi)$$
  Boundary conditions on $g$: $g(0) = 0$, $g'(0) = \int_0^1 \rho = 1$, $g'(1) = 0$.
- Differentiating twice: $\alpha \rho''(\phi) - \beta \rho(\phi) = 0 \implies \rho''(\phi) - \tau^2 \rho(\phi) = 0$.
- Differentiating once at $\phi=1$: $\alpha \rho'(1) + \beta g'(1) = 0 \implies \rho'(1) = 0$.
- Differentiating once at $\phi=0$: $\alpha \rho'(0) + \beta g'(0) = 0 \implies \alpha \rho'(0) + \beta(1) = 0 \implies \rho'(0) = -\tau^2$.
- The unique solution satisfying $\int_0^1 \rho = 1$ is $\rho_\tau(\phi) = \frac{\tau\cosh(\tau(1-\phi))}{\sinh\tau}$.

#### Adversarial Red Team Critique
1. `[DERIVED]` **Surrogate Kernel is the Laplacian Green's Function, Not RoPE:** Why $\min(\phi,\psi)$? In real RoPE, cross-subspace collision involves sinc-type terms: $\frac{\sin((\omega-\nu)L)}{(\omega-\nu)L}$. This is non-local and oscillatory. The kernel $\min(\phi,\psi)$ is specifically the Green's function for the differential operator $-\frac{d^2}{d\phi^2}$ on $[0,1]$ with boundary conditions $u(0)=0, u'(1)=0$. The hyperbolic cosine solution is therefore a direct consequence of choosing an invertible differential operator, rather than reflecting true positional self-attention geometry.
2. `[DERIVED]` **Exploding Transport Bounds on Discrete Grids:** In `appendix/a1_proofs.tex` (line 571), the Wasserstein-1 transport bound is:
   $$W_1(\mu_{K,\tau}, \mu_{\rho_\tau}) \le \frac{\sinh\tau}{4K\tau}$$
   For $\tau = 6$ and $K = 32$: $\frac{\sinh 6}{4 \cdot 32 \cdot 6} \approx \frac{201.7}{768} \approx 0.2627$.
   Because the entire domain is $[0,1]$, a transport error bound of $26.3\%$ is completely loose. At larger $\tau$ (e.g. $\tau \ge 8$), $\sinh\tau$ grows exponentially, rendering the continuous-to-discrete transport guarantees mathematically vacuous.
3. `[OBSERVED]` **Operating Rule $\tau = d_{\mathrm{head}}/\sqrt{L}$ is a Fragile Heuristic:** The paper derives $\tau_*$ from balancing a small-$\tau$ $\chi^2$ expansion ($S_{\chi^2} \approx \tau^4 / (45 d_{\mathrm{head}})$) against a linearized diffuse-attention phase-variance utility. As documented in `TAU_TRUE_ROLE_AND_OPERATING_RULE_AUDIT.md`, this formula fails across different base frequencies $b$ and contexts; setting $c=1$ is an uncalibrated convention.

---

### Topic 3: Post-Hoc Transplant Obstruction & The Slot Permutation Breakdown

#### Mathematical Formulations and Definitions
`[OBSERVED]` In `paper-2027/sections/03_theory.tex` (lines 183–199) and `appendix/a1_proofs.tex` (lines 278–296):
- Theorem 3 states: If position-independent invertible linear maps $A, B \in \mathrm{GL}(2K, \mathbb{R})$ satisfy $A^\top \mathcal{R}_{\Omega'}(\Delta) B = \mathcal{R}_\Omega(\Delta)$ for all $\Delta \in (-\epsilon, \epsilon)$, then $\Omega'$ and $\Omega$ have the same frequency multiset, up to sign and permutation.

#### Quantifier and Condition Verification
`[DERIVED]`
- At $\Delta = 0$: $\mathcal{R}_\Omega(0) = \mathcal{R}_{\Omega'}(0) = I_{2K}$. Thus $A^\top B = I_{2K} \implies B = A^{-\top}$.
- The condition becomes: $A^\top \mathcal{R}_{\Omega'}(\Delta) A^{-\top} = \mathcal{R}_\Omega(\Delta)$.
- Differentiating at $\Delta = 0$: $A^\top G_{\Omega'} A^{-\top} = G_\Omega$, where $G_\Omega = \bigoplus_{k=1}^K \omega_k \begin{bmatrix}0 & -1 \\ 1 & 0\end{bmatrix}$.
- Similar matrices have identical eigenvalues: $\operatorname{spec}(G_\Omega) = \{\pm i\omega_k\}_{k=1}^K$, $\operatorname{spec}(G_{\Omega'}) = \{\pm i\omega'_k\}_{k=1}^K$.
- Equality of spectrum multisets forces $\{|\omega'_k|\} = \{|\omega_k|\}$.
- For integer positions $\Delta \in \mathbb{Z}$, evaluating at $\Delta=1$ forces $\{e^{\pm i\omega'_k}\} = \{e^{\pm i\omega_k}\}$, which implies $\omega'_k \equiv \pm \omega_k \pmod{2\pi}$.

#### Adversarial Red Team Critique & Minimal Counterexample
1. `[DERIVED]` **The Permutation Blind Spot:** Theorem 3 proves that frequency multisets must match up to permutation. Algebraically, if $\Omega' = \pi(\Omega)$ is a non-trivial permutation, there EXISTS an orthogonal permutation matrix $P$ such that $P^\top \mathcal{R}_{\pi(\Omega)}(\Delta) P = \mathcal{R}_\Omega(\Delta)$. Thus, under the premises of Theorem 3, frequency permutation is completely admissible and solvable via $A = B = P$.
2. `[OBSERVED]` **Empirical Catastrophe of Permutation:** In actual checkpoints, however, setting $\Omega' = \pi(\Omega)$ while holding weights frozen ($A = B = I$) causes total collapse.
   - `SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`: Permuting frequency slots in Qwen-2.5-7B drops 64K RULER from $0.7000$ to $0.0000$.
   - OLMo-2 1x PG-19 NLL explodes from $3.10423$ to $6.86493$.
3. `[DERIVED]` **Why the Theorem Fails to Capture the True Obstruction:** In zero-training retrofit, $A$ and $B$ are NOT free variables; they are rigidly fixed to $A = B = I$. The projection matrices $W_q, W_k$ have learned column-subspace associations: columns $(2k-1, 2k)$ are tuned to rotary frequency $\omega_k$. Theorem 3 establishes an obstruction against a massive class of linear adapters, but obscures the far more restrictive reality: frozen weights cannot even tolerate an orthogonal permutation $P$.

---

### Topic 4: Compatibility Modulus, Softmax Propagation & Weight-Blind Vacuity

#### Mathematical Formulations and Definitions
`[OBSERVED]` In `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` (§B, lines 214–243):
- Score function: $s(\Delta) = \sum_{k=1}^K \operatorname{Re}[c_k e^{i\omega_k \Delta}]$, with $c_k = z^q_k \bar{z}^\kappa_k \in \mathbb{C}$.
- Theorem T3 (Compatibility Modulus):
  $$|s'(\Delta) - s(\Delta)| \le \sum_{k=1}^K |c_k| \min(2, |\omega'_k - \omega_k|\Delta)$$
- Softmax error propagation: $\|p' - p\|_1 \le e^{2\epsilon} - 1$, where $\epsilon = \sup_\Delta |s'(\Delta) - s(\Delta)|$.
- Theorem T4 (Conditioning Theorem): Score deviation amplification is bounded by $S$:
  $$\frac{\sup_{\Delta \le SL} |\delta s(\Delta)|}{\sup_{\Delta \le L} |\delta s(\Delta)|} \le S$$

#### Quantifier and Condition Verification
`[DERIVED]`
- For T3:
  $$|s'(\Delta) - s(\Delta)| = \left|\sum_k \operatorname{Re}[c_k (e^{i\omega'_k\Delta} - e^{i\omega_k\Delta})]\right| \le \sum_k |c_k| |e^{i(\omega'_k - \omega_k)\Delta} - 1|$$
  Since $|e^{i\theta} - 1| = 2|\sin(\theta/2)| \le \min(2, |\theta|)$, the bound is mathematically exact.
- For Softmax Propagation: Let $p_i = e^{s_i} / \sum_j e^{s_j}$ and $p'_i = e^{s'_i} / \sum_j e^{s'_j}$ with $|s'_i - s_i| \le \epsilon$.
  Then $p'_i / p_i = \frac{e^{s'_i - s_i}}{\sum_j p_j e^{s'_j - s_j}} \in [e^{-2\epsilon}, e^{2\epsilon}]$.
  Hence $|p'_i - p_i| \le p_i \max(1 - e^{-2\epsilon}, e^{2\epsilon} - 1) = p_i (e^{2\epsilon} - 1)$.
  Summing over all $i$: $\|p' - p\|_1 \le e^{2\epsilon} - 1$.

#### Adversarial Red Team Critique
1. `[DERIVED]` **Exponential Vacuity of the Softmax Bound:** The bound $\|p' - p\|_1 \le e^{2\epsilon} - 1$ requires $\epsilon < \frac{1}{2}\ln 3 \approx 0.549$ to yield a bound $< 2$ (since the total variation between any two probability vectors cannot exceed 2).
   In reality, when retrofitting for context scale $S=4$ at $\Delta = 16{,}384$:
   For $K=32$ rotary pairs with typical query/key norms $\|c_k\| \sim 0.5$, $\sum_k |c_k| \approx 16$.
   Since $|\omega'_k - \omega_k|\Delta \gg 2$ for shifted channels, $\min(2, |\omega'_k - \omega_k|\Delta) = 2$.
   Then $\epsilon \approx 2 \times 16 = 32$.
   Plugging $\epsilon = 32$ into the softmax bound gives:
   $$\|p' - p\|_1 \le e^{64} - 1 \approx 6.2 \times 10^{27}$$
   This bound is $10^{27}$ times larger than the trivial ceiling ($2.0$). It provides zero mathematical constraint on actual model behavior under extrapolation.
2. `[DERIVED]` **Breakdown of Score Conditioning T4 under Large Steps:** The conditioning number $S$ holds strictly for first-order linear perturbations where $|\delta\omega_k| SL \ll 1$. When frequencies are scaled by $S^{-m_k}$, the phase displacement $(\omega_k - \omega'_k)SL = \omega_k SL(1 - S^{-m_k})$ exceeds $2\pi$ for multiple channels. The linear envelope collapses into circular phase wrapping, rendering first-order conditioning inapplicable.

---

### Topic 5: The "Waterbed" Effect, Novelty Calculus & Semigroup Vacuity

#### Mathematical Formulations and Definitions
`[OBSERVED]` In `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md`:
- Identity A6: For $\omega'_k = \omega_k S^{-m_k}$, novelty ratio $r_k(S) = \frac{\omega'_k SL}{\omega_k L} = S^{1-m_k}$.
- Identity A6 Claim: $r_k(S) \cdot S^{m_k} = S$ is termed the "per-slot waterbed identity".
- Theorem T6: If $F(S_1 S_2) = F(S_2) \circ F(S_1)$ with $f_k(S)$ multiplicative, then $f_k(S) = S^{-m_k}$.
- Theorem T7: Novelty volume $N_k(S) = \omega_k L(S \cdot 4^{-m_k} - 1)$ satisfies $N_k(2S)/N_k(S) = 2 + \frac{1}{4^{1-m_k}-1} > 2$.

#### Adversarial Red Team Critique
1. `[DERIVED]` **The "Waterbed" is a Tautology:** In classical control theory, Bode's waterbed effect is a conservation law arising from Cauchy's integral theorem applied to analytic transfer functions. Here, $r_k(S) \cdot S^{m_k} = S^{1-m_k} \cdot S^{m_k} = S^{1-m_k + m_k} = S^1 = S$. Calling this elementary cancellation of exponents a "waterbed theorem" inflates a simple definition into a purported physical principle.
2. `[DERIVED]` **Semigroup Structure is a Bookkeeping Artifact:** Theorem T6 shows that if a per-slot scaling is closed under multiplication, it must be a power law $S^{-m_k}$. This is the standard Cauchy power functional equation $f(xy) = f(x)f(y)$. However, Transformer checkpoints do not form a representation of the multiplicative semigroup $(\mathbb{R}^+, \times)$. In reality, scaling from $4\times$ to $8\times$ using the same formula fails empirically (the s8 ceiling), demonstrating that the underlying behavioral system does not obey semigroup dynamics.
3. `[DERIVED]` **Novelty Volume Does Not Predict Failure:** Theorem T7 proves that $N_k(2S) > 2N_k(S)$ for $m_k \in (0,1)$. This is pure arithmetic on the function $g(S) = a S - b$. While the novelty phase volume super-doubles, whether the network's downstream attention heads tolerate this phase volume depends entirely on the content-dependent projection coefficients $c_k$, which are unconstrained by the formula.

---

### Topic 6: Common-Direction Feasibility, Convex-Hull Geometry & The "Basin Barrier"

#### Mathematical Formulations and Definitions
`[OBSERVED]` In `COMMON_DIRECTION_FEASIBILITY_AND_BASIN_BARRIER_THEORY_20260902.md`:
- Robust common-direction problem:
  $$\max_{d \in \mathbb{R}^K, \gamma \in \mathbb{R}} \gamma - \frac{\lambda}{2}\|d\|^2 \quad \text{s.t.} \quad g_j^\top d + \gamma \le 0, \quad j=1,\ldots,J$$
- Theorem (Pillar 1): Unique optimum is $d^* = -\bar{g}^*/\lambda$, $\gamma^* = \|\bar{g}^*\|^2/\lambda$, where $\bar{g}^* = \arg\min_{c \in \operatorname{conv}\{g_1,\ldots,g_J\}} \|c\|$. A strictly improving direction exists iff $0 \notin \operatorname{conv}\{g_j\}$.
- Hypothesis (Pillar 3): Bimodal basin structure along dilation coordinate.

#### Quantifier and Condition Verification
`[DERIVED]`
- Pillar 1 is an application of Wolfe's dual formulation for minimum-norm point finding in a polytope (Désidéri's Multiple-Gradient Descent Algorithm, MGDA):
  $$\mathcal{L}(d,\gamma,\mu) = \gamma - \frac{\lambda}{2}\|d\|^2 - \sum_j \mu_j (g_j^\top d + \gamma)$$
  $\nabla_\gamma \mathcal{L} = 1 - \sum_j \mu_j = 0 \implies \sum_j \mu_j = 1, \mu_j \ge 0 \implies \mu \in \Delta_J$.
  $\nabla_d \mathcal{L} = -\lambda d - \sum_j \mu_j g_j = 0 \implies d = -\frac{1}{\lambda}\bar{g}(\mu)$, where $\bar{g}(\mu) = \sum_j \mu_j g_j \in \operatorname{conv}\{g_j\}$.
  Dual objective: $q(\mu) = \frac{1}{2\lambda}\|\bar{g}(\mu)\|^2$. Maximizing the margin $\gamma$ corresponds to minimizing $\|\bar{g}(\mu)\|$.
  Thus $0 \notin \operatorname{conv}\{g_j\} \iff \gamma^* > 0$.

#### Adversarial Red Team Critique
1. `[DERIVED]` **Infinitesimal Scope vs. Finite Step Breakdown:** Pillar 1 applies strictly to first-order directional derivatives at $d=0$. A strictly improving direction $d^*$ guarantees descent only for step size $\alpha \to 0$. In deep non-convex landscapes, second-order terms $d^\top \nabla^2 \mathcal{L}_j d$ dominate almost immediately.
2. `[OBSERVED]` **Empirical Overfitting of the 18-Sample Route:** The repo attempted to exploit this theorem by computing behavioral gradients on 18 samples across 64 frequency dimensions (`FIRST_PRINCIPLES...` Fact F). Optimizing 64 parameters on 18 samples produced a common descent direction on the calibration set that catastrophically failed on the holdout set.
3. `[DERIVED]` **The "Basin Barrier" is a Built-in Assumption:** Pillar 3 models the loss landscape as:
   - A smooth quadratic tax for native retention: $R(\delta) \approx R_0 - a \delta^2$.
   - A thresholded step function for long-context capability: $C(\delta) \approx \sum_k \sigma(\beta(\delta - \delta_k))$.
   If an objective is defined by subtracting a convex bowl from a series of step functions, the resulting landscape is trivially bimodal by construction. Labeled as an "established basin barrier", it masquerades as an architectural theorem when it is actually an empirical conjecture based on assumed functional forms.

---

### Topic 7: Attention Gain, Softmax Decisiveness & The Multi-Layer Argmax Fallacy

#### Mathematical Formulations and Definitions
`[OBSERVED]` In `FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md` (lines 332–357) and `COMMON_DIRECTION...` (lines 238–247):
- "Multiplying a head's scores by $g > 0$ preserves every per-decision argmax (same sign structure) — exact; gain cannot repair a ranking error, it can only amplify margins and sharpen softmax."
- "Attention gain scales attention scores $s \mapsto cs$. For any $c>0$, $\arg\max$ of the key softmax is unchanged: gain provably cannot alter which key is attended, only the concentration of the mixture around the existing ranking."

#### Adversarial Red Team Attack & Minimal Counterexample
`[DERIVED]` The assertion that "gain provably cannot alter which key is attended" is mathematically FALSE for any multi-layer Transformer. While the scalar map $x \mapsto gx$ is strictly monotonic on $\mathbb{R}$ (preserving the argmax of a single vector), an attention layer outputs an expectation:
$$z_i = \sum_{j=1}^N p_{ij}(g) V x_j, \qquad p_{ij}(g) = \frac{e^{g s_{ij}}}{\sum_k e^{g s_{ik}}}$$
The vector $z_i(g)$ depends continuously on $g$. When $z_i(g)$ passes into subsequent Transformer layers, it alters the query and key representations at Layer 2:
$$q^{(2)} = W_q^{(2)}(x + z(g)), \qquad k^{(2)} = W_k^{(2)}(x + z(g))$$
This directly changes the relative order of attention logits at Layer 2!

**Explicit Minimal Counterexample:**
Let sequence length $N=2$. At Layer 1, query attends to keys with logits $s_1 = 1, s_2 = 0$.
The softmax distribution is $p(g) = \left[\frac{e^g}{e^g + 1}, \frac{1}{e^g + 1}\right]$.
Let value vectors be $v_1 = \begin{bmatrix}1 \\ 0\end{bmatrix}, v_2 = \begin{bmatrix}0 \\ 1\end{bmatrix}$.
The output of Layer 1 is $z(g) = \begin{bmatrix}\frac{e^g}{e^g + 1} \\ \frac{1}{e^g + 1}\end{bmatrix}$.
Now consider Layer 2. Let the Layer 2 query be fixed: $q^{(2)} = \begin{bmatrix}1 \\ -2\end{bmatrix}$.
Let Layer 2 key 1 be driven by Layer 1 output: $k_1^{(2)}(g) = z(g)$.
Let Layer 2 key 2 be a fixed distractor: $k_2^{(2)} = \begin{bmatrix}0.5 \\ 0.5\end{bmatrix}$.
Compute the Layer 2 attention logits:
- For Key 2: $s_2^{(2)} = [q^{(2)}]^\top k_2^{(2)} = 1(0.5) - 2(0.5) = -0.5$.
- For Key 1: $s_1^{(2)}(g) = [q^{(2)}]^\top k_1^{(2)}(g) = 1\left(\frac{e^g}{e^g + 1}\right) - 2\left(\frac{1}{e^g + 1}\right) = \frac{e^g - 2}{e^g + 1}$.

Now evaluate as a function of Layer 1 gain $g$:
- Case A ($g \to -\infty$, uniform attention at Layer 1): $s_1^{(2)} \to -2$. Since $-2 < -0.5$, **Key 2 wins** ($\arg\max = 2$).
- Case B ($g = 0$, unscaled logits): $s_1^{(2)}(0) = \frac{1 - 2}{2} = -0.5$. Tie.
- Case C ($g = 2$, sharpened attention at Layer 1): $e^2 \approx 7.389 \implies s_1^{(2)}(2) = \frac{5.389}{8.389} \approx +0.642$. Since $+0.642 > -0.5$, **Key 1 wins** ($\arg\max = 1$).

`[DERIVED]` **Conclusion:** Changing the scalar attention gain $g$ in Layer 1 directly flips the attended key argmax in Layer 2 from Key 2 to Key 1! The claim that gain cannot alter attended keys is invalid in multi-layer architectures.

---

### Topic 8: Alternative Allocation Theoretic Formulations

#### 1. Phase Isotropy & Pair Volume (`TARGET_FREE_PHASE_ISOTROPY_ALLOCATION_THEORY_20260824.md`)
`[OBSERVED]` The self-Gram under causal prior $p_L(d) = \frac{2(L-d)}{L(L+1)}$ has eigenvalues:
$$\lambda_\pm(S_\omega) = \frac{1 \pm |\chi_L(2\omega)|}{2}, \qquad \chi_L(t) = \sum_{d=0}^{L-1} p_L(d) e^{itd}$$
- Pair volume: $v_L(\omega) = 4\det S_\omega = 1 - |\chi_L(2\omega)|^2$.
- Pair condition ratio (isotropy): $i_L(\omega) = \frac{\lambda_-}{\lambda_+} = \frac{1 - |\chi_L(2\omega)|}{1 + |\chi_L(2\omega)|}$.
- Proposed allocation distortion: $\mathcal{D}_K[\rho] = \frac{1}{12K^2}\int_0^1 \frac{i_L(\omega(\phi))}{\rho(\phi)^2} d\phi$.

`[DERIVED]` **Red Team Assessment:**
1. The eigenvalue derivation is mathematically exact for $2\times 2$ matrices with circular symmetry.
2. The objective $\mathcal{D}_K[\rho]$ is borrowed from Bennett's integral for scalar quantization of continuous random variables. It assumes high-resolution quantization where reconstruction error scales as $K^{-2}$.
3. RoPE frequency allocation is NOT scalar quantization of a continuous signal. It configures the phase velocities of a discrete set of orthogonal rotary planes. There is no proof that minimizing scalar reconstruction distortion $\mathcal{D}_K$ minimizes cross-entropy loss.
4. In repository benchmarks, phase isotropy failed to provide consistent gains over standard geometric or heuristic tables (`INDEX.md` §3.2: `SCREEN_UNRESOLVED`).

#### 2. MaxEnt Dilation Allocation (`MAXENT_DILATION_ALLOCATION_20260901.md`)
`[OBSERVED]`
- Maximizing differential entropy on $[0, \log s]$ relative to Haar measure $dr/r$ subject to fixed mean log-dilation yields:
  $$p_\lambda(\tau) = \frac{e^{\lambda\tau}}{\int_0^{\log s}e^{\lambda u}du}, \qquad r_i(\lambda) = [1 + q_i(s^\lambda - 1)]^{1/\lambda}$$
- Monotone channel coupling pairs larger dilation factors with smaller native frequencies $\omega_i$.
- Claim: The monotone pairing minimizes summed native phase displacement $L \sum_i \omega_i(1 - 1/r_i)$ via the rearrangement inequality.

`[DERIVED]` **Red Team Assessment:**
1. The derivation of $r_i(\lambda)$ is an exact application of the maximum entropy principle and quantile inversion.
2. The pairing argument follows rigorously from the Hardy-Littlewood-Pólya rearrangement inequality: since $\omega_i$ decreases with $i$ and $(1 - 1/r_i)$ increases with $r_i$, the anti-monotone product sum is strictly minimal.
3. However, the objective $L \sum_i \omega_i(1 - 1/r_i)$ treats phase errors as unweighted Euclidean quantities on $\mathbb{R}$, ignoring that phases live on the torus $\mathbb{T}^1 = \mathbb{R}/2\pi\mathbb{Z}$ and that content weights $c_k$ vary by orders of magnitude across heads and channels.

---

## 4. Minimal Counterexamples & Structural Failure Modes

### Counterexample 1: Same-Multiset Permutation Collapse
`[DERIVED]`
- **Claim Challenged:** Frequency table quality is governed by the unordered frequency multiset $\{\omega_k\}$ (implicit in multiset obstruction theorems and scalar spectral integrals).
- **Construction:** Let $\Omega = \{\omega_0, \ldots, \omega_{K-1}\}$ be a valid RoPE table. Let $\pi$ be the permutation reversing order: $\pi(k) = K - 1 - k$. Then $\Omega' = \pi(\Omega)$ has the identical frequency multiset.
- **Result:** Under Theorem 3, there exists an orthogonal matrix $P$ such that $P^\top \mathcal{R}_{\Omega'}(\Delta) P = \mathcal{R}_\Omega(\Delta)$. But in a frozen model with $A = B = I$, the attention logits become:
  $$\ell'(\Delta) = \sum_{k=0}^{K-1} q_k^\top R(\omega_{K-1-k}\Delta) k_k \ne \sum_{k=0}^{K-1} q_k^\top R(\omega_k\Delta) k_k$$
- **Empirical Confirmation:** In Qwen-2.5-7B (`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`), multiset permutation drops 64K RULER from $0.7000$ to $0.0000$, and OLMo 1x PG-19 NLL explodes from $3.10$ to $6.86$.

### Counterexample 2: Multi-Layer Gain Reversal of Argmax Ranking
`[DERIVED]`
- **Claim Challenged:** Scalar attention gain $g > 0$ preserves argmax ranking across the model (Memo T8).
- **Construction:** In Section 3 (Topic 7), we constructed an explicit 2-layer model where:
  - At Layer 1, logits are $s = [1, 0]^\top$. Gain $g$ sharpens attention from uniform ($g \to -\infty$) to one-hot ($g \to +\infty$).
  - Layer 1 output $z(g) = \left[\frac{e^g}{e^g+1}, \frac{1}{e^g+1}\right]^\top$ passes into Layer 2.
  - At Layer 2, $s_1^{(2)}(g) = \frac{e^g-2}{e^g+1}$ and $s_2^{(2)} = -0.5$.
- **Result:** For $g < 0$, Key 2 wins ($s_2^{(2)} > s_1^{(2)}$). For $g > 0$, Key 1 wins ($s_1^{(2)} > s_2^{(2)}$). Gain at Layer 1 directly reverses the attended token ranking at Layer 2.

### Counterexample 3: Compatibility Modulus Exponential Vacuity
`[DERIVED]`
- **Claim Challenged:** The compatibility modulus $|s' - s| \le \sum_k |c_k| \min(2, |\Delta\omega_k|\Delta)$ and $\|p' - p\|_1 \le e^{2\epsilon} - 1$ provide a meaningful mathematical bound for extrapolation.
- **Construction:** Let $K=64$, $\|c_k\| = 0.5$ for all $k$, and scale $S=4$. At context lag $\Delta = 16{,}384$, for all shifted channels $|\Delta\omega_k|\Delta \ge 2$, so $\min(2, |\Delta\omega_k|\Delta) = 2$.
- **Result:** $\epsilon = \sum_{k=1}^{64} 0.5 \times 2 = 64$.
  The softmax bound yields $\|p' - p\|_1 \le e^{128} - 1 \approx 3.9 \times 10^{55}$.
  Since the maximum $L_1$ distance between any two probability vectors is 2.0, the bound exceeds the analytical ceiling by 55 orders of magnitude.

### Counterexample 4: Static Collision Metric Inversion Across Horizons
`[OBSERVED]`
- **Claim Challenged:** Minimizing static subspace collision at training length $L$ guarantees superior collision or effective rank under extrapolation ($2L, 4L$).
- **Evidence:** In `paper-2027/appendix/a1_proofs.tex` (lines 241–254), two $K=16$ schedules $A$ and $B$ satisfy:
  $$C_L(A) = 0.45837 < 0.61695 = C_L(B) \quad \text{($A$ is superior at $L$)}$$
  $$C_{2L}(A) = 0.45834 > 0.41132 = C_{2L}(B) \quad \text{($B$ is superior at $2L$)}$$
  $$C_{4L}(A) = 0.45830 > 0.24763 = C_{4L}(B) \quad \text{($B$ is superior at $4L$)}$$
- **Result:** Optimizing static collision at $L$ inverts and degrades collision properties at $2L$ and $4L$.

### Counterexample 5: Linearized Common-Direction Holdout Collapse
`[OBSERVED]`
- **Claim Challenged:** First-order gradient feasibility $0 \notin \operatorname{conv}\{g_j\}$ guarantees the existence of a viable common direction for finite-step retrofit.
- **Evidence:** In `FIRST_PRINCIPLES...` Fact F and `COMMON_DIRECTION...` §1:
  Behavioral gradients $g_j \in \mathbb{R}^{64}$ were measured across $J=18$ calibration samples.
  Because $J < K$ ($18 < 64$), the vectors $g_j$ are linearly independent, and $0 \notin \operatorname{conv}\{g_j\}$ holds trivially.
- **Result:** The computed descent direction $d^* = -\bar{g}^*/\lambda$ achieved positive margins on the 18 calibration samples, but suffered catastrophic collapse on the unopened holdout set. First-order linear feasibility in an overparameterized regime ($K > J$) is uninformative about true multi-task Pareto feasibility.

---

## 5. Quantifier Scoping, Tightness, and Hidden Assumptions Audit

| Theory / Claim Component | Purported Scope | Actual Mathematical Scope | Missing / Hidden Assumptions | Failure Modes |
| :--- | :--- | :--- | :--- | :--- |
| **Spectral Budget Identity ($r_2$)** | Universal basis capacity metric | Static block-whitened Gram matrix trace ratio | Uniform separation prior $\mathrm{Unif}[0,L]$; ignores $W_q, W_k$ projections and softmax | Rank can increase while PPL degrades by 10x (Table 1 co-adaptation failure) |
| **Variational Cosh Optimum** | Optimal frequency allocation | Unique minimizer of surrogate $\mathcal{C}_{\mathrm{app}}$ | Surrogate Laplacian Green's kernel $\min(\phi,\psi)$; $L^2$ continuum limit | Fails on finite grids for large $\tau$; does not reflect oscillatory RoPE cross-Gram |
| **Post-Hoc Transplant Obstruction** | Explains why post-hoc RoPE replacement is hard | Linear similarity of Lie algebra generators under $A^\top \mathcal{R} B$ | Assumes constant linear maps $A, B$; ignores frozen weight-slot coupling | Suggests multiset permutations are permissible, whereas empirical permutations totally collapse |
| **Compatibility Modulus** | Bounds logit and softmax perturbation | Worst-case phase-aligned Cauchy-Schwarz bound | Assumes microscopic perturbation $\epsilon \ll 0.1$ | Yields $\|p' - p\|_1 \le 10^{27}$ at extrapolation distances ($S=4, 8$) |
| **Conditioning Theorem (T4)** | Score deviation amplification $\le S$ | First-order linear Taylor expansion of sinusoids | Assumes $|\delta\omega_k|SL \ll 1$ (no phase wrapping) | Fails completely once phase shifts wrap around the unit circle |
| **Common-Direction Gate** | Identifies viable multi-task retrofit direction | QP dual minimum-norm point on gradient polytope | Linearized local gradient at $d=0$; assumes convex landscape | Overfits trivially when $K > J$; fails across finite steps and holdout tasks |
| **Attention Gain Invariance** | Gain cannot alter attended keys | Invariance of $\arg\max_j(g s_j)$ for a single vector | Assumes 1-layer isolated softmax | In multi-layer models, gain alters hidden state mixtures and flips downstream argmax rankings |

---

## 6. Synthesis & Strategic Guidance for Theoretical Architecture

`[DERIVED]` Based on the rigorous mathematical audit, we establish the following strategic recommendations for the research program:

1. **Retire Static Basis Optimality Claims:** Static rank $r_2(\Gamma)$, log-determinants, and surrogate energies must never be presented as predictors of language model perplexity, downstream accuracy, or extrapolation capability. They are diagnostic descriptors of the unweighted positional basis, nothing more.
2. **Re-frame Theorem 3 to Highlight the Subspace-Coupling Paradox:** Theorem 3 must be explicitly stated with its full boundary conditions. Rather than merely claiming that "frequencies are rigid up to permutation", the paper must emphasize the **stronger negative result**: even exact-multiset permutations collapse frozen models, proving that learned $W_q, W_k$ weights are rigidly bound to specific rotary slot indices.
3. **Acknowledge the True Multi-Layer Role of Gain:** Do not claim that attention gain cannot change attended keys. Acknowledge that while gain preserves single-layer argmax rankings, it acts as an entropy/temperature regulator whose primary multi-layer effect is altering the mixture vector passed to subsequent layers.
4. **Treat the Basin Barrier as an Empirical Conjecture, Not an Algebraic Theorem:** State plainly that bimodal basin behavior and gradient starvation are model-dependent hypotheses consistent with observation, not proven mathematical impossibilities.
5. **Abandon Underdetermined Behavioral Gradient Searches:** Do not attempt to compute common descent directions in $\mathbb{R}^K$ using $J < K$ sample cells. Optimization in underdetermined regimes without strong structural regularizers guarantees holdout generalization failure.
