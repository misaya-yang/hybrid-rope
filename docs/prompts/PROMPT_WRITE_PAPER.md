# 论文写作 Prompt 模板

## 使用方法
复制下方 prompt 直接发给 Claude。路径已写死，无需替换。

---

## Prompt（完整论文写作）

```
你是一名顶会论文写作专家。请严格按照以下指令写作论文。

## 输入文件（先全部读完再动笔）
1. 写作规划：docs/paperdraft/PAPER_PLAN_V9.md
2. 核心理论+数据：docs/paperdraft/CORE_THEORY.md
3. PE baselines + 组合实验：docs/exp/2026-03-03_passkey_mix_results.md
4. LaTeX 框架模板：paper_exports/neurips_v5/hybrid_rope_neurips_v8.tex（仅复用 style 定义和 appendix 证明，正文内容全部按规划重写）

## 输出
- 输出到 paper_exports/neurips_v5/hybrid_rope_neurips_v9.tex
- pdflatex 编译输出 hybrid_rope_neurips_v9.pdf
- 编译后验证页数和 abstract word count

## 硬性约束（违反任何一条 = 失败）
1. 正文 ≤ 9 页（\section{Introduction} 到 \section{Conclusion} 末尾）
2. Abstract ≤ 150 words（写完后 wc 验证）
3. 严格遵循 PAPER_PLAN_V9.md 中的页面预算分配
4. References 和 Appendix 不计入 9 页
5. NeurIPS 2025 格式（neurips_2025.sty，10pt Times，single column）

## 写作风格规则
- 段落散文，禁止在正文实验分析中使用 bullet list
- 禁止词汇："striking", "remarkably", "interestingly", "notably"
- 禁止自评："This is one of the paper's strongest results" 等
- 禁止元叙述："In this subsection we describe..."，直接写内容
- Table caption 必须自包含（只读 caption 就能理解）
- 每段 3-5 句，不超过 7 句
- Bold 仅标注最关键数字，每段 ≤ 2 处
- Contributions 以动词开头：We derive / We prove / We validate

## 写作流程（严格按序）
1. 读完 PAPER_PLAN_V9.md + CORE_THEORY.md + passkey_mix_results.md
2. 从 v8.tex 复制 style 定义 + appendix，正文全部重写
3. 按页面预算逐 section 写，每写完一个估算已用页数
4. Section 5 (Experiments) 最后写，用剩余页数控制
5. 编译 PDF
6. 验证：(a) wc abstract ≤ 150  (b) 正文 ≤ 9 页  (c) 零编译错误
7. 超页则按优先级砍（P2→Appendix，P3→删除）

## 内容优先级（空间不够从底部砍）
P0（必须正文）：Abstract, Intro, Framework (Thm 1+2), EVQ warp, §5.3 PE baselines + EVQ+YaRN 组合表
P1（强烈建议）：τ* scaling law, Passkey mix +40pp, 750M retrieval divergence (Fig 1)
P2（可移 Appendix）：Collision-block dead zone, r-sweep Pareto, Phase collision scores
P3（直接删除）：Physical interpretation 长段, Implicit priors section, 350M 3-seed detail table
```

---

## Prompt（重写单个 Section）

```
严格按照 docs/paperdraft/PAPER_PLAN_V9.md 中 Section {X} 的规划，重写该 section。
页面预算：{N} 页。
当前 tex：paper_exports/neurips_v5/hybrid_rope_neurips_v9.tex
替换从 \section{X} 到 \section{Y} 之间的全部内容。
数据来源：docs/paperdraft/CORE_THEORY.md §{对应章节}。
风格：段落散文，无 bullet list，无自评语句，无禁用词。
```

---

## Prompt（验证已写论文）

```
检查 paper_exports/neurips_v5/hybrid_rope_neurips_v9.tex 是否满足：
1. Abstract word count ≤ 150
2. 正文 ≤ 9 页
3. 无禁用词（striking/remarkably/interestingly/notably）
4. 实验段无 bullet list（itemize 仅允许 Contributions 和 Limitations）
5. 所有 Table caption 自包含
6. 所有数字与 docs/paperdraft/CORE_THEORY.md 一致（逐表核对）
7. 无未定义引用或缺失 citation
逐条输出 pass/fail + 位置 + 修复建议。
```
