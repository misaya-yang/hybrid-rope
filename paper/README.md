# Paper — EVQ-Cosh NeurIPS 2026 Submission

本目录只保留当前投稿所需的 LaTeX 源码、图表、参考文献，以及唯一的
投稿 PDF：`main.pdf`。

- `main.tex`：唯一 LaTeX 入口；
- `main.pdf`：唯一需要保留和提交的编译 PDF；
- `build_tectonic/`、`build_aidemo/`：可随时删除的本地构建目录，已由
  `.gitignore` 排除；
- rebuttal、citation audit、review prompt 和历史稿不属于 paper source，
  不应重新放回本目录。

---

## 编译

首选 Tectonic，将所有中间文件隔离到构建目录：

```bash
cd paper
tectonic -X compile main.tex --outdir build_tectonic
cp build_tectonic/main.pdf main.pdf
```

也可以使用仓库已验证的 `aidemo` wrapper：

```bash
cd paper
bash compile_aidemo.sh
```

默认输出 `paper/build_aidemo/main.pdf`。如需更新唯一投稿 PDF，运行：

```bash
COPY_MAIN=1 bash compile_aidemo.sh
```

该脚本会通过 `conda run --no-capture-output -n aidemo` 执行
`pdflatex -> bibtex -> pdflatex x3`，并把中间产物隔离在
`paper/build_aidemo/`。完成核验后可以直接删除两个 `build_*` 目录；不要把
构建副本另存为新的根目录 PDF。NeurIPS 2026 样式文件已包含在本目录。

---

## 目录结构

```
paper/
├── main.tex              唯一论文入口 (NeurIPS submission mode)
├── main.pdf              唯一投稿 PDF
├── compile_aidemo.sh     备用 pdfTeX 编译 wrapper
├── README.md             本说明
├── neurips_2026.sty      当前样式文件
├── sections/             正文各章节
│   ├── 01_intro.tex
│   ├── 02_related.tex
│   ├── 03_theory.tex     EVQ-Cosh 推导
│   ├── 05_experiments.tex 实验结果
│   └── 06_limitations.tex
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
│   ├── table_epistemic_map.tex
│   ├── table_evidence_tier.tex
│   ├── table_lambda_cv.tex
│   └── table_method_comparison.tex
├── figs/                 图表资产；当前编译引用 PDF，PNG 用于预览/复现
│   ├── fig_method_overview.pdf
│   ├── fig1_frequency_dynamics.pdf
│   ├── fig2_evq_yarn_synergy.pdf
│   ├── fig3_pe_dominant_scaling.pdf
│   ├── fig4_phase17c_flagship.pdf
│   ├── fig5_downstream_qa.pdf
│   ├── fig6_tau_rank_readable.pdf
│   ├── fig7_multiscale_waterbed.pdf
│   ├── fig_unification_orthogonal.pdf
│   └── attn_*.pdf        attention visualization
├── refs/
│   └── references.bib    参考文献
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
