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

## 2026-09-02T19:27:57Z

利用 `/Users/yang/projects/hybrid-rope` 中已完成的历史实验，建立一个包含 12–20 个 episode 的 RoPE 理论伪证基准（Theory Falsification Benchmark）。全流程不使用 GPU、不提出任何新 RoPE 理论或方法、不运行或推荐新实验、不进行自我预测，完成后直接交付并停止。

Working directory: /Users/yang/projects/hybrid-rope/falsification_benchmark
Integrity mode: demo

Reference materials:
- Context repository: `/Users/yang/projects/hybrid-rope` (contains experimental logs in `results/`, `paper-2027/research/`, `experiments/`, and authoritative timelines in `INDEX.md`)

## Requirements

### R1. Chronological Episode Selection & Registry
Select 12–20 distinct historical experiments from `/Users/yang/projects/hybrid-rope`, strictly ordered by their verified chronological timestamps. The episodes must span diverse models (e.g., Qwen, OLMo, Gemma, LLaMA), scales (context length, parameter sizes), task domains (loss/NLL, retrieval/NIAH, QA/LongBench, synthetic proofs), and qualitative failure modes (such as same-multiset permutation collapse, table vs. gain divergence, and headwise specialization conflicts). Each episode must represent a non-obvious outcome that discriminates between competing theoretical hypotheses. Register these in a structured `experiment_registry` (e.g., JSON and markdown summary table).

### R2. Decoupled Visible Packets and Hidden Ground Truth
For each selected episode $E$:
- `visible packet`: Includes strictly all prior established facts, baseline observations, theoretical priors, and exact experimental protocols (model architecture, data split, prompt setup, metrics to measure) known *before* $E$ was executed. Contains zero hints or indicators of the actual outcome.
- `hidden answer`: Encapsulates the ground truth empirical result of $E$ (both quantitative metrics like delta NLL, F1 scores, retention ratios, and qualitative observations like failure or collapse patterns).

### R3. Automated Zero-Leakage Audit
Provide an automated audit mechanism and a formal audit report verifying that no visible packet leaks any information about its corresponding hidden answer. The audit must programmatically verify the absence of numerical leakage, directional outcome terminology, post-hoc timestamps, or leaked qualitative artifacts.

### R4. Deterministic Evaluator
Implement a CPU-only deterministic evaluation script (Python CLI) that takes blind predictions from any candidate theory or agent and scores them against hidden answers across four explicit dimensions:
1. Directional correctness (binary or sign match);
2. Calibrated probability distribution (Brier score or negative log loss);
3. Effect magnitude accuracy (mean absolute error or bounded numerical tolerance);
4. Qualitative pattern alignment (categorical or structured pattern match).

### R5. Fresh-Theorist Instructions & Hard Stop Guardrails
Provide standalone, comprehensive instructions (`fresh_theorist_guide.md`) specifying the exact prediction schema, submission protocol, and blind evaluation flow for future unseen theorist agents. The current team must not participate in making predictions on these episodes. Do not propose or recommend any new RoPE theories, architectures, or GPU experiments. Stop immediately upon benchmark delivery.

## Acceptance Criteria

### Deliverable Completeness
- [ ] Working directory `/Users/yang/projects/hybrid-rope/falsification_benchmark` contains all six required deliverables:
  1. `experiment_registry.json` (and `experiment_registry.md`)
  2. `visible_packets/`
  3. `hidden_answers/`
  4. `leakage_audit/` (audit script and generated report)
  5. `evaluator/` (evaluator script, mock test suite, README)
  6. `fresh_theorist_guide.md`
- [ ] Number of registered episodes is between 12 and 20 ($12 \le N \le 20$).
- [ ] Episodes are strictly sorted in ascending chronological order with source commit or timestamp references.

### Leakage Audit Verification
- [ ] The automated leakage audit script runs with a clean exit code (0 violations / 0 warnings detected).
- [ ] Audit report explicitly itemizes checked tokens, dates, and cross-references for every episode.

### Evaluator Verification
- [ ] The deterministic evaluator runs locally on CPU without errors (`python -m evaluator --predictions ... --answers ...`).
- [ ] Evaluator unit test suite passes cleanly, confirming expected scoring on known synthetic cases (e.g., perfect prediction = 1.0, inverted prediction = penalty, uniform prior baseline).

### Guardrails
- [ ] No GPU resources or commands are invoked throughout execution.
- [ ] No new RoPE variants, architectures, theoretical hypotheses, or GPU follow-up proposals are included in any deliverable.
- [ ] No self-prediction by current session agents.
