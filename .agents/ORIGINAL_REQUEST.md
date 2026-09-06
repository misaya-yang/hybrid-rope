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

## 2026-09-06T12:15:02Z

<USER_REQUEST>
多代理并行完成 Hybrid-RoPE 论文与实验的全局深度整合与审计：
1. 深入分析 `~/Downloads` 目录下最新生成且具长文件名的各大模型分析与实验规划 MD 文件，提取新颖理论视角并对照已知实证；
2. 深度梳理 `paper-2027` 核心目录及 Claude Code 近两日的工作区实验与交接记录，标注真实证据等级；
3. 只读连接远程服务器 `ssh -p 27741 root@connect.westc.seetacloud.com` 巡检当前任务与运行状态、检查输出与报告；
4. 综合撰写一份不超过 5000 字、高度凝练、真实客观的《理论分析与实验综合报告》MD，直接保存至桌面。

Working directory: /Users/yang/projects/hybrid-rope
Integrity mode: development

## Requirements

### R1. 高级模型规划与理论整合 (~/Downloads 目录)
检索并分析 `/Users/yang/Downloads` 中近期由高级大模型生成的长文件名 Markdown (`.md`) 文件。
1. 提取各模型对本仓库架构、坐标分解 ($x_k = -\log(\omega_k) = a + R z_k$)、几何约束与外推能力的分析与假设；
2. 对照当前仓库已执行的实验（如固定支持、EVQ-Cosh、几何基线、对偶与波长混合算子），明确：
   - 哪些规划建议已被实际实验验证或证伪；
   - 哪些新理论视角可被吸纳整合进当前统一几何框架；
   - 哪些建议存在假设缺陷或已被实证淘汰。

### R2. 近两日核心工作区与 Claude Code 实验审计 (paper-2027 深度梳理)
系统审计 `paper-2027`（特别是 `research/attention-aware-retrofit/` 下的 preflights、results、analysis 及 Claude Code 工作区）：
1. 重点复盘近两日的关键实验：OLMo 候选门控表现、Qwen N128 独立 Native 1756 行验证结果、format/indexing 回归、紧凑任务与多源对比；
2. 遵循 `AGENTS.md` 规范，对所有关键实验发现严格标注证据标签（Observation, Derived result, Working hypothesis, Negative result, Unresolved, Superseded）；
3. 明确目前阻碍 4x 到 8x/16x 推广的核心瓶颈与已知边界。

### R3. 远程实验服务器只读巡检 (connect.westc.seetacloud.com:27741)
通过本机免密 SSH 访问 `root@connect.westc.seetacloud.com:27741` 执行严格只读巡检（严禁任何写入、删除、中断或启动未授权任务操作）：
1. 检查当前服务器硬件状态、GPU 利用率与活跃进程（`nvidia-smi`, `ps aux | grep python`）；
2. 巡检实验目录下的已有成果，检查最近生成的结果文件、`review.json`、`execution.json` 及执行日志；
3. 核对远程最新实验产出与本地记录的一致性，识别是否有尚未同步的有效运行数据。

### R4. 理论分析与实验总结综合 MD 报告
综合 R1、R2、R3 的所有信息，在 `/Users/yang/Desktop/THEORY_EXPERIMENT_SYNTHESIS.md` 生成最终报告：
1. **结构与内容**：包含项目核心几何理论闭环、近期实验全景复盘与证据分类、远程服务器运行现状与数据对齐、大模型理论输入之取舍与整合、下一步最高决策价值行动路线；
2. **严谨与真实**：严禁外部模型主观推测当作既成事实，每个核心数字和结论必须追溯至具体运行日志、哈希或实验报告；
3. **篇幅约束**：总字数严格控制在 5000 字以内，确保语言高信息密度、高度简明扼要，拒绝泛泛套话。

## Acceptance Criteria

### 报告完整性与位置
- [ ] 最终报告成功写入 `/Users/yang/Desktop/THEORY_EXPERIMENT_SYNTHESIS.md`。
- [ ] 报告涵盖 R1（下载区模型理论）、R2（paper-2027 近期实验复盘）、R3（远程服务器巡检数据）、R4（整合结论与行动指南）。
- [ ] 报告总字数严格在 5000 字以内（包含中英文字数与核心表格）。

### 科学纪律与证据溯源 (AGENTS.md)
- [ ] 明确标注各类结论的证据标签（Observation / Negative result / Unresolved 等）。
- [ ] 清晰指出 Downloads 模型建议中哪些可行、哪些已被证伪（例如已被证明未能通过 Native 门控的方案）。

### 基础设施与只读安全
- [ ] 远程服务器所有指令均为无害只读查询（如 `ls`, `cat`, `head`, `tail`, `nvidia-smi`, `ps` 等），未变更任何远程或本地只读状态。
</USER_REQUEST>

## 2026-09-06T12:48:17Z

<USER_REQUEST>
全面审查并深度打磨位于 `/Users/yang/Desktop/THEORY_EXPERIMENT_SYNTHESIS.md` 的综合报告，直击更新该文件。严格核对事实准确性、查缺补漏关键实验、彻底剔除过渡套话与冗余清单，交付一份信息密度极高、文字极度简明扼要（严格控制在 5000 字以内）的终审版报告。

Working directory: /Users/yang/projects/hybrid-rope
Integrity mode: development

## Requirements

### R1. 事实准确性与数学严谨性核验 (Accuracy & Fact-Checking)
对照仓库原始凭证（`paper-2027/`、`README.md`、`INDEX.md`、各 preflights 与运行 receipts）逐行核对：
1. **数学公式与符号**：坐标分解 ($x_k = -\log(\omega_k) = a + R z_k$)、双线性相角核展开、贪心保持证书确界 $\kappa(p)$、规范联合置换对称性等推导是否数学严密、无符号倒置；
2. **实测数据与置信区间**：核验 OLMo-1B Native 数据（PPL 3.319650、任务 80.85%、EOS 70.30%）、Qwen N128 1756 行 Native 数据（宏任务 98.06%、格式 77.05%、CI 区间）、Round 11 ZF vs ON 命中数（15/27 vs 0/27 等）；
3. **证据标签合规**：严格依据 `AGENTS.md`，每个事实或结论必须带上精准证据标签（`[Observation]`, `[Derived result]`, `[Negative result]` 等），杜绝主观外推。

### R2. 全面性与完整性覆盖 (Comprehensiveness)
确保核心科学闭环无遗漏：
1. 涵盖下载区高级模型规划之得失（采纳坐标分解因果性，证伪增益过门禁、李群轨道数、连续 MAE 等 4 大缺陷）；
2. 涵盖本地 paper-2027 近两日核心实证（OLMo 静态单表关停、Qwen N128 宏指标过关但局部格式回归、首分歧 KL 翻转证明、物理暴露必要性二分）；
3. 涵盖远程 RTX 4080 SUPER 32GB 实时运行（98% 计算利用率、32GB 显存实测、PID 41974 任务、Round 11 换表必要性证明、Round 12 注意力与表示漂移分解、9.9GB 磁盘红线警报）。

### R3. 极致简明扼要与信息密度 (Extreme Conciseness)
大刀阔斧重构文本，追求极致信息密度：
1. **彻底剔除泛泛空话**：删除所有过渡套话、修饰性形容词与重复说明；
2. **直奔结论与机制**：以高度凝练的因果句式、公式和紧凑对比表呈现，最大化每句话的学术与决策信息量；
3. **不附加冗余清单**：正文直接展现最精纯的终审内容，不附加繁琐的修订对比列表；
4. **字数严格受限**：全文严格控制在 5000 字以内（中英文+字符统一核算）。

### R4. 直接原地更新 (Direct In-Place Desktop Update)
直接覆写更新 `/Users/yang/Desktop/THEORY_EXPERIMENT_SYNTHESIS.md` 为终审精炼版本。

## Acceptance Criteria

### 终审质量与篇幅
- [ ] `/Users/yang/Desktop/THEORY_EXPERIMENT_SYNTHESIS.md` 成功原地更新为精炼终审版。
- [ ] 全文总字数严格控制在 5000 字以内（信息密度极高，无啰嗦冗余）。
- [ ] 不包含多余的修订对比附录，正文直击核心。

### 事实准确度 100% 闭环
- [ ] 报告中的每个数值、置信区间、显存数字、进程号均与原始 log/receipt bitwise 吻合。
- [ ] 证据标签（`AGENTS.md`）100% 严谨匹配。
</USER_REQUEST>

