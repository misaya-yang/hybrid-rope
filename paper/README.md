# Paper — EVQ-Cosh NeurIPS 2026 Submission

本目录包含论文的完整 LaTeX 源码、图表和参考文献。

---

## 编译

推荐使用本仓库已验证的 `aidemo` wrapper：

```bash
cd paper
bash compile_aidemo.sh
```

输出: `paper/build_aidemo/main.pdf`。如需同时更新提交用的 `paper/main.pdf`，运行:

```bash
COPY_MAIN=1 bash compile_aidemo.sh
```

该脚本会通过 `conda run --no-capture-output -n aidemo` 执行 `pdflatex -> bibtex -> pdflatex x3`，并把中间产物隔离在 `paper/build_aidemo/`。NeurIPS 2026 样式文件已包含在本目录。

---

## 目录结构

```
paper/
├── main.tex              论文入口 (NeurIPS submission mode)
├── sections/             正文各章节
│   ├── 01_intro.tex
│   ├── 02_related.tex
│   ├── 03_theory.tex     EVQ-Cosh 推导
│   ├── 04_predictions.tex historical; not included by main.tex
│   ├── 05_experiments.tex 实验结果
│   ├── 06_limitations.tex
│   └── 07_conclusion.tex (historical; not included by main.tex)
├── appendix/             附录
│   ├── a1_proofs.tex
│   ├── a2_experiment_details.tex
│   ├── a3_supporting_results.tex
│   └── a4_supporting_experiments.tex
├── tables/               论文表格 (.tex)
│   ├── table1_multiscale_raw_ppl.tex
│   ├── table2_evq_yarn_main.tex
│   ├── table3_capability_passkey.tex
│   ├── table4_pe_dominant.tex
│   ├── table5_phase11_leverage.tex
│   ├── table6_750m_continue_supporting.tex
│   ├── table_evidence_tier.tex
│   └── table_method_comparison.tex
├── figs/                 所有图表 (PDF + PNG)
│   ├── fig1_frequency_dynamics.pdf
│   ├── fig2_evq_yarn_synergy.pdf
│   ├── fig3_pe_dominant_scaling.pdf
│   ├── fig4_phase17c_flagship.pdf
│   ├── fig5_downstream_qa.pdf
│   ├── fig6_tau_formula_validation.pdf
│   ├── fig7_multiscale_waterbed.pdf
│   └── attn_*.pdf        (attention visualization)
├── refs/
│   └── references.bib    参考文献
└── neurips_2026.sty      样式文件
```

---

## 论文 ↔ 代码追溯

每个 Figure/Table 的生成脚本和数据来源详见 **`docs/overview/PAPER_CLAIMS_MAP.md`**。

### 快速对照

| Stable asset/source | 生成脚本 |
|---------------------|---------|
| Frequency dynamics / multiscale waterbed figures | `scripts/figures/fig1_neurips.py` |
| EVQ × YaRN synergy figure | `scripts/figures/fig2_evq_yarn_orthogonality.py` |
| PE-dominant scaling figure | `scripts/figures/fig3_pe_dominant_scaling.py` |
| 454M supporting progressive figure | `scripts/core_text_phases/phase17c_*.py` |
| Downstream QA figure | `scripts/core_text_phases/phase21b_quality_eval_clean.py` |
| τ* validation figure | `scripts/core_text_phases/phase16_formula_optimality_sweep.py` |
| Table `.tex` sources | 见 `docs/overview/PAPER_CLAIMS_MAP.md` |

---

## 图表重新生成

所有主论文图表均可从已有结果数据重新生成:

```bash
python scripts/figures/fig1_neurips.py
python scripts/figures/fig2_evq_yarn_orthogonality.py
python scripts/figures/fig3_pe_dominant_scaling.py
```

输出保存至 `paper/figs/`。

---

## 编辑守则

- 正文控制在 9 页内 (NeurIPS 2026 限制)
- 图表优先使用 PDF 格式 (矢量)
- 保持 anonymous submission mode (`\usepackage[nonatbib]{neurips_2026}` + anonymous author block); public non-anonymous preprints should use the official `preprint` option instead.
- 新增图表同步更新 `docs/overview/PAPER_CLAIMS_MAP.md` 的映射
