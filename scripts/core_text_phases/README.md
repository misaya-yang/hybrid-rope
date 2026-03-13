# Core Text Phases

Phase 8–21 核心文本实验链，直接支撑论文所有 claims。

完整的 Figure/Table → Script 映射见 `docs/overview/PAPER_CLAIMS_MAP.md`。

## Phase Map (含论文追溯)

| Phase | Question | Paper Role | Main Scripts | → Paper |
|---|---|---|---|---|
| 8 | EVQ raw scaling law | Theory foundation | `run_evq_sweep.py`, `phase8d_scaling_law.py`, `phase8f_multi_seed.py` | Table 1, Fig 6 |
| 11 | PE-dominant regime + YaRN interaction | **Primary anchor** | `phase11_L256_extrap.py`, `phase11_yarn_eval.py`, `phase11b_125m_dape.py`, `phase11c_454m_scaling.py` | **Fig 3, Tables 4-5** |
| 13 | Downstream NLL probe | Supporting | `phase13a_longbench_nll.py` | Appendix |
| 14 | EVQ + YaRN >> Geo + YaRN | **Primary anchor** | `phase14c_multiscale_evq_yarn.py`, `phase14d_125m_tinystories_10pct.py` | **Fig 2, Tables 2-3** |
| 15 | Larger-scale continued pretrain | Supporting scale-up | `phase15_750m_2k_to_4k_continue_ckpt_eval.py`, `phase11e_continued_pretrain.py` | Table 6 |
| 16 | τ* formula validation | Theory confirmation | `phase16_formula_optimality_sweep.py` | **Fig 6** (99-run) |
| 17b | 454M L=512→1024 continue | Stage 2 | `phase17b_454m_512_to_1024_continue_ckpt_eval.py` | Fig 4 (middle) |
| 17c | 454M L=1024→2048 continue | Flagship demo | `phase17c_454m_1024_to_2048_continue.py`, `phase17c_extended_eval.py` | **Fig 4** (final) |
| 21b | QuALITY downstream eval | Downstream evidence | `phase21b_quality_eval_clean.py`, `phase21b_scrolls_finetune.py` | **Fig 5** |

## Script → Paper Quick Reference

| Script | Produces |
|--------|---------|
| `run_evq_sweep.py` | Training + τ-sweep data for Table 1 |
| `phase11b_125m_dape.py` | DAPE comparison data for Fig 3 panel (a) |
| `phase11_L256_extrap.py` | L=256 scaling data for Fig 3 panels (b,c) |
| `phase14c_multiscale_evq_yarn.py` | Passkey + PPL data for Fig 2, Tables 2-3 |
| `phase16_formula_optimality_sweep.py` | 99-run τ* data for Fig 6 |
| `phase17c_454m_1024_to_2048_continue.py` | 454M Stage 3 data for Fig 4 |
| `phase21b_quality_eval_clean.py` | QuALITY NLL data for Fig 5 |
| `visualize_attention_distance.py` | Attention viz for Appendix |
| `evq_analysis.py` | τ-sweep analysis figures for Appendix |

## Naming Rules

- `phaseXX_*.py`: phase-specific experiment
- `eval_*.py`: reusable evaluation helper
- `visualize_*.py`: visualization script

## Shared Evaluation Helpers

- `eval_passkey.py` — Passkey retrieval accuracy
- `eval_multi_needle.py` — Multi-needle NIAH
- `eval_longbench_nll.py` — LongBench NLL evaluation
- `eval_pe_baselines.py` — Baseline PE comparison
- `eval_super_extrap.py` — Extreme extrapolation test
- `eval_dsr.py` — Distance sensitivity ratio

## Practical Reading Order

1. `run_evq_sweep.py` — 理解模型架构 + EVQ 实现 + 训练循环
2. `phase11_L256_extrap.py` — PE-dominant regime 核心逻辑
3. `phase14c_multiscale_evq_yarn.py` — YaRN 集成 + passkey 评估
4. `phase17c_454m_1024_to_2048_continue.py` — 续训流程
5. `phase21b_quality_eval_clean.py` — 下游评估
