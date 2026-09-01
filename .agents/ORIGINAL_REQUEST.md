# Original User Request

## 2026-09-01T03:26:47Z

# Teamwork Project Prompt

> Status: Launched
> Goal: Multi-agent deep synthesis on RoPE exponent allocation $z = -2i/d$, out-of-distribution phase code distinguishability, frozen Q/K co-adaptation readout, and scientific/practical value of non-linear $f(z)$
> Requested team: 6 parallel research agents exploring complementary theoretical, geometrical, reading-mechanism, and empirical axes

## Project Description
Deep theoretical and empirical investigation into the fundamental role of RoPE's exponent spectrum $z = -2i/d$, why original RoPE fails under extrapolation ($\Delta > L_{\text{train}}$) from the perspective of joint phase code distinguishability and frozen Q/K readout dynamics, what non-linear spectrum reallocations $f(z) \neq cz$ actually change physically, and the practical value and design boundaries of spectrum-aware RoPE adaptation.

Working directory: `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope`
Integrity mode: development

## Requirements

### R1. First-Principles Analysis of Original $z = -2i/d$ and Extrapolation Failure
- Analyze the information-theoretic role of log-uniform frequency distribution $\omega_i = b^{-2i/d}$.
- Explain the joint phase code $\Phi(\Delta)$ geometry when $\Delta > L_{\text{train}}$: why high-frequency aliasing and low-frequency spectral collapse destroy phase distinguishability across different $\Delta$.

### R2. Frozen Checkpoint Q/K Readout and Co-adaptation Dynamics
- Explain how attention logit $\ell(\Delta) = q^\top R(\Delta) k = \sum_j A_j \cos(\omega_j \Delta + \psi_j)$ couples Q/K projection weights $W_q, W_k$ with the phase spectrum.
- Diagnose why frozen weights fail to read unseen phase combinations at $\Delta > L_{\text{train}}$ (softmax entropy collapse / attention noise).

### R3. Mathematical and Physical Impact of Non-Linear $f(z) \neq cz$
- Characterize the exact effect of warping $z \to f(z)$: spectral channel density, wave-packet dispersion, and phase velocity redistribution.
- Distinguish non-linear spectrum reallocation from trivial scalar base dilation $f(z) = cz$.

### R4. Alignment with Empirical Evidence and Practical Scientific Value
- Synthesize findings with repository evidence (151M exact-range causal identification, 50M 2x2 table-weight crossing, frozen transplant obstruction).
- Articulate the real-world scientific and engineering value of non-linear spectrum allocation under bounded degradation realities.

## Acceptance Criteria

### Theoretical Rigor
- [x] Explicit mathematical derivation of joint phase code $\Phi(\Delta)$ distinguishability and Gram matrix properties.
- [x] Clear explanation of Q/K readout mechanics under frozen vs. adapted regimes.
- [x] Rigorous characterization of non-linear $f(z)$ vs. linear base scaling $cz$.

### Empirical Grounding
- [x] Direct consistency with repository canonical evidence in `INDEX.md` and `paper-2027/research/`.
- [x] Explicit avoidance of falsified routes (arcsine conjecture, naive collision minimization).
