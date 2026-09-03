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

## 2026-09-03T02:34:13Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: 多角色独立研究审计团队（包含 Evidence Archivist, Mathematical Red Team, Experimental Auditor, First-Principles Theorist, Judge/Synthesizer 5个独立角色）

对成熟 checkpoint 的 zero-training RoPE retrofit 难题（为什么无法同时保持 Native 基础能力与 Long 扩展能力，以及是否存在高成功率解决方向）开展严格、可审计、只读的多角色联合研究与最终判定。

Working directory: /Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope
Integrity mode: development

## Requirements

### R1. 原始证据链重建与审计 (Evidence Archivist)
- 严格只读遍历仓库内核心索引与证据所有者（包括 `INDEX.md`、`paper-2027/HANDOFF.md`、`paper-2027/research/attention-aware-retrofit/`、`paper-2027/research/foundations/`、`rebuttal/rebuttal_0723/` 等）。
- 重建与成熟 checkpoint retrofit 相关的实验时间线与因果链。
- 提取真实建立的事实（区分 positive, negative, null results），找出历史中被重复引用但缺乏原始数据支撑的 unsupported claims。
- 产出 Claim → Evidence → Evidence strength 证据表，所有事实必须附带文件路径和关键数字。

### R2. 理论与数学严密性红队检验 (Mathematical Red Team)
- 对仓库内现存的定理、恒等式、界（upper/lower bounds）、兼容性论证、Pareto 论证、渐近论证及频域/相角解释逐项红队审查。
- 明确数学定义、量词与成立条件，构造最简反例（如全频 multiset 相同但槽位顺序改变的退化反例、局部线性区向全局外推的破缺等）。
- 严禁替现有理论辩护；严格区分 algebraic identity, local approximation, upper/lower bound, empirical regularity 与 theorem。

### R3. 实验效度与混淆变量审计 (Experimental Auditor)
- 深入核实核心实验证据（如 S=2/4/8 failure、frequency multiset 排序耦合、Native retention 门禁、long NLL、RULER-13、HotpotQA、gain 调节、head/layer 专门化、LoRA/co-adaptation 交叉等）。
- 检查 intervention 究竟改变了什么、是否存在混淆变量（confounds）、对照是否匹配、指标是否足以支撑结论。
- 判别 S=2/4/8 的失败本质是族系选择缺陷（family-specific）、优化失败、权重与表的不兼容，还是真实的结构性约束。

### R4. 第一性原理理论重构 (First-Principles Theorist)
- 独立于现有经验解释，从 RoPE 注意力基础公式 $s_{ij}(\Delta) = \sum_k \text{Re}[c_k e^{i \omega_k \Delta}]$ 出发，结合已学习的 Q/K 投影、有序旋转子空间、checkpoint co-adaptation、有限头维度与部署视界重新推导。
- 不预设 YaRN、log profile、protected ramp、headwise specialization 或 Pareto ceiling 必然成立。
- 回答在冻结成熟权重后，改变 RoPE 频率表真正控制功能行为的数学/几何物理对象到底是什么，并给出与已有实验的可证伪联系。

### R5. 综合判定与核心决策输出 (Judge / Synthesizer)
- 汇总四方独立审查结果，标记理论与实验冲突，对冲突重新查证原始数据，剔除缺乏证据的故事与推论。
- 严格执行 Section 4 的 9 项“防止过早停止”核查清单（包含主动推翻最佳解释、检查数学反例、识别 unsupported claims、分析历史失败最小解释等）。
- 最终输出必须严格且仅回答 Section 5 的六大核心问题（A: 真正问题定义, B: 已经确定事实, C: 历史认知误区, D: 历史失败最小机制解释/竞争假说, E: 是否存在高置信解决方向, F: 下一步唯一最高信息增益动作）。

## Acceptance Criteria

### 0. 行为边界与运行安全
- [ ] 严格只读（READ-ONLY）：仓库内无任何代码、配置、论文修改（`git status --porcelain` 保持完全干净）。
- [ ] 零 GPU 计算：禁止启动任何 GPU 训练、推理或评测脚本；不产生未经授权的外部计算开销。
- [ ] 零虚构：严禁虚构不存在的定理、数字、实验结果或代码文件；缺失证据明确标注 `UNSUPPORTED BY REPOSITORY EVIDENCE`。

### 1. 证据分类与格式契约
- [ ] 核心陈述全部使用 `[OBSERVED]`、`[DERIVED]`、`[HYPOTHESIS]`、`[UNKNOWN]` 四元标签严格归类。
- [ ] 每一个 `[OBSERVED]` 陈述均附有精确的仓库相对文件路径（如 `paper-2027/research/...`）、实验名称及具体数字。
- [ ] 每一个 `[DERIVED]` 陈述均展示最短必要推导步骤及成立的前置假设条件。

### 2. 六大核心问题交付质量
- [ ] 问题 A：在 3 句话内精确定义尚未解决的技术难题（禁止宽泛愿景）。
- [ ] 问题 B：仅包含最强的已确认事实，逐项附带原始证据所有者。
- [ ] 问题 C：明确指出项目中最重要的无证据假设、错误等价与误导性抽象。
- [ ] 问题 D：给出能同时涵盖多个历史失败实验的最小机制解释（或并列竞争假说）。
- [ ] 问题 E：对解决方向提出极严格判断；若证据不足，明确承认“当前证据不足以给出高成功率解决方案”，严禁强造新曲线方案。
- [ ] 问题 F：仅提出单一最高信息增益动作，并附带明确的成功/失败判据及停机（stop condition）标准。

---
*Next: when approved → delegate via invoke_subagent (see Delegation Protocol)*

