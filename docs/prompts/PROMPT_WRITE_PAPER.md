# 论文写作 Prompt 模板

## 使用方法
复制下方 prompt，替换 `{变量}` 后直接发给 Claude/Gemini。

---

## Prompt

```
你是一名顶会论文写作专家。请严格按照以下指令写作论文。

## 输入文件
- 写作规划：{PAPER_PLAN_V9.md 的路径}
- 核心数据：{CORE_THEORY.md 的路径}
- 实验数据：{passkey_mix_results.md 等的路径}
- 基础模板：{v8 tex 的路径}（仅用作 LaTeX 框架和 style 定义，内容全部重写）

## 输出要求
- 输出完整 .tex 文件，可直接 pdflatex 编译
- 同时编译输出 .pdf

## 硬性约束（违反任何一条 = 失败）
1. **正文 ≤ 9 页**（从 \section{Introduction} 到 \section{Conclusion} 末尾）
2. **Abstract ≤ 150 words**（写完后 word count 验证）
3. **严格遵循写作规划中的页面预算分配**
4. **References 和 Appendix 不计入 9 页**
5. **NeurIPS 2025 格式**（neurips_2025.sty，10pt Times，single column）

## 写作风格规则
- 段落散文，禁止在正文实验分析中使用 bullet list
- 禁止以下词汇："striking", "remarkably", "interestingly", "notably"
- 禁止自评式语句（如 "This is one of the paper's strongest results"）
- 禁止 "In this subsection we describe..."，直接写内容
- Table caption 必须自包含（只读 caption 就能理解表的含义）
- 每段 3-5 句，绝不超过 7 句
- Bold 标注仅用于最关键数字，每段不超过 2 处
- Contributions 每条以动词开头（We derive / We prove / We validate）

## 写作流程（必须按此顺序）
1. 先读完写作规划和核心数据文件
2. 按规划中的页面预算，逐 section 写作
3. 每写完一个 section，估算已用页数
4. Section 5 (Experiments) 最后写，用剩余页数精确控制
5. 全部写完后编译 PDF
6. 验证：(a) abstract word count ≤ 150  (b) 正文 ≤ 9 页  (c) 无编译错误
7. 如超页，按规划中的优先级砍内容（优先移到 Appendix）

## 内容优先级（空间不够时从底部开始砍）
P0（必须在正文）：Abstract, Intro, Framework (Thm 1+2), EVQ warp, Experiments §5.3 (PE baselines + EVQ+YaRN 组合表)
P1（强烈建议）：τ* scaling law, Passkey mix +40pp, 750M retrieval divergence (Fig 1)
P2（可移到 Appendix）：Collision-block dead zone, r-sweep Pareto, Phase collision scores
P3（直接删除）：Physical interpretation 长段, Implicit priors section, 350M 3-seed detail table
```

---

## 变体：仅重写某个 Section

```
请严格按照 {PAPER_PLAN_V9.md} 中 Section {X} 的规划，重写以下 section。
页面预算：{N} 页。当前 tex 文件：{路径}。
替换从 \section{X} 到 \section{Y} 之间的全部内容。
遵循写作风格规则（无 bullet list 分析，无自评语句，段落散文）。
数据来源：{CORE_THEORY.md} §{对应章节}。
```

---

## 变体：验证已写好的论文

```
请检查 {v9.tex} 是否满足以下所有条件：
1. Abstract word count ≤ 150
2. 正文（\section{Introduction} 到 \section{Conclusion} 末尾）≤ 9 页
3. 无 "striking/remarkably/interestingly" 等禁用词
4. 正文实验段无 bullet list（\begin{itemize} 仅允许在 Contributions 和 Limitations）
5. 所有 Table caption 自包含
6. 所有数字与 {CORE_THEORY.md} 一致（逐表核对）
7. 无未定义引用或缺失 citation
输出：逐条 pass/fail + 具体位置和修复建议。
```
