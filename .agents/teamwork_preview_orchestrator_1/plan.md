# Master Plan: Multi-Agent Deep Synthesis on RoPE Exponent Allocation and Readout Dynamics

## Objective
Execute a rigorous multi-agent deep synthesis answering the user request on RoPE exponent allocation $z = -2i/d$, out-of-distribution phase code distinguishability, frozen Q/K co-adaptation readout dynamics, mathematical and physical consequences of non-linear $f(z) \neq cz$, and practical engineering boundaries grounded in canonical repository evidence.

## Research Architecture & Decomposition (6 Parallel Axes)

### Axis 1: Information-Theoretic Foundations of Log-Uniform $z = -2i/d$ (R1.1)
- Agent: `explorer_r1_infotheory_1`
- Directory: `.agents/teamwork_preview_explorer_r1_infotheory_1/`
- Questions to answer:
  1. Why did original RoPE (Su et al., 2021) choose the geometric progression $\omega_i = b^{-2i/d}$ ($z = -2i/d$)?
  2. How does log-uniform frequency allocation relate to continuous wavelet transforms, multi-scale resolution, and self-similarity?
  3. Information-theoretic and functional analysis: spectral density $d\rho(\omega)/d\omega \propto 1/\omega$, scale-invariance under relative positional shifts, and uniform capacity allocation across octaves.

### Axis 2: Phase Code $\Phi(\Delta)$ Geometry, Gram Matrix & Extrapolation Breakdown (R1.2)
- Agent: `explorer_r1_phasegeometry_1`
- Directory: `.agents/teamwork_preview_explorer_r1_phasegeometry_1/`
- Questions to answer:
  1. Mathematical derivation of the joint phase code $\Phi(\Delta) = [\cos(\omega_0 \Delta), \sin(\omega_0 \Delta), \dots, \cos(\omega_{K-1} \Delta), \sin(\omega_{K-1} \Delta)]^\top \in \mathbb{R}^{2K}$.
  2. Gram matrix properties $G(\Delta, \Delta') = \langle \Phi(\Delta), \Phi(\Delta') \rangle = \sum_{j=0}^{K-1} \cos(\omega_j(\Delta - \Delta'))$ and stable rank $r_2(R) = \frac{2K}{1 + (K-1)\bar{c}}$.
  3. Geometry of extrapolation failure at $\Delta > L_{\text{train}}$:
     - High-frequency phase aliasing / torus wrapping (non-unique phase coordinates across distant intervals).
     - Low-frequency spectral collapse ($V_\omega \to \mathrm{span}\{1, \Delta\}$ in $L_2$ or $\mathrm{span}\{\Delta, \Delta^2\}$ in softmax metric).
     - Joint phase code degradation: destruction of distinguishable metric embedding across distance domains.

### Axis 3: Bilinear Attention Logit Readout & Frozen Q/K Co-adaptation Dynamics (R2)
- Agent: `explorer_r2_qkreadout_1`
- Directory: `.agents/teamwork_preview_explorer_r2_qkreadout_1/`
- Questions to answer:
  1. Explicit derivation of attention logit $\ell(\Delta) = q^\top R(\Delta) k = \sum_{j=0}^{K-1} A_j \cos(\omega_j \Delta + \psi_j)$. How projection matrices $W_q, W_k$ assign amplitude $A_j$ and phase offset $\psi_j$ to each 2D rotary sub-channel.
  2. Mechanism of failure for frozen weights at $\Delta > L_{\text{train}}$: why weights trained on in-distribution phase interference produce softmax entropy collapse, noisy cross-talk, and loss of sharp attention peaks.
  3. Mathematical diagnosis of the 2x2 table-weight crossing (PPL $7.14 \to 76.20$ upon table swap without weight adaptation) and post-hoc frequency transplant obstruction (no invertible linear compensation for unequal frequency multisets).

### Axis 4: Mathematical & Physical Impact of Non-Linear $z \to f(z)$ vs. Linear Scaling $cz$ (R3)
- Agent: `explorer_r3_nonlinear_1`
- Directory: `.agents/teamwork_preview_explorer_r3_nonlinear_1/`
- Questions to answer:
  1. Rigorous distinction between linear base scaling $f(z) = cz$ (isomorphic to scalar base change $b \to b^c$) and genuine non-linear allocation $f(z) \neq cz$.
  2. Physical and mathematical implications: spectral channel density $\rho(\omega) = |dz/d\omega|$, wave-packet group velocity $v_g = d\omega/dk$ vs phase velocity $v_p = \omega/k$, dispersion relations, and spectral compression/dilation.
  3. Derivation of EVQ-Cosh construction (convex surrogate minimization under fixed support) and analysis of falsified routes (why arcsine conjecture and naive collision minimization fail by collapsing to Fourier combs / exact aliasing).

### Axis 5: Empirical Synthesis & Canonical Evidence Grounding (R4.1)
- Agent: `explorer_r4_empirical_1`
- Directory: `.agents/teamwork_preview_explorer_r4_empirical_1/`
- Questions to answer:
  1. Deep synthesis of canonical empirical results in `INDEX.md` §3 and `paper-2027/research/`:
     - 151M exact-range 3-seed causal identification ($+0.026 / -0.281 / -0.176 / -0.146$).
     - Target-matched allocation vs support dilation boundary ($+0.026 / +0.060 / +0.227 / +0.460$).
     - 50M Fisher probe / 2x2 table-weight crossing.
     - Systems breadth: 432M MLA 3-seed, 750M continuation, 1.485B OLMo-2 baseline, 8B adaptation, Video-DiT seed-42.
     - Retrofit evidence: length-conditioned budgeted retrofit (RULER core-4 0.5825@8K / 0.4000@16K vs YaRN 0.5375 / 0.0125).
  2. Compliance with AGENTS.md claim ceilings and locked nomenclature.

### Axis 6: Engineering Boundaries & Native-Support Pure-$z$ Paradigm (R4.2)
- Agent: `explorer_r4_engineering_1`
- Directory: `.agents/teamwork_preview_explorer_r4_engineering_1/`
- Questions to answer:
  1. Real-world engineering realities: why zero-training frozen retrofit hits fundamental ceiling and why matched weight co-adaptation (lightweight LoRA / Q-K adaptation) is strictly required for non-linear $z$.
  2. Single static table / single model across $1\times, 2\times, 4\times$ contexts without KV-cache coordinate destruction or routing hacks.
  3. Bounded degradation reality, practical trade-offs, and critical review of the 12 falsified routes in INDEX.md §3.4.

## Execution Sequence
1. Dispatch all 6 explorers simultaneously.
2. Monitor progress via heartbeat cron.
3. Collect all 6 handoff reports upon completion.
4. Perform unified synthesis across all 4 requirements (R1, R2, R3, R4) and acceptance criteria.
5. Produce final synthesis report and handoff to Sentinel.
