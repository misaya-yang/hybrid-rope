# Fable5 Review Brief — Rebuttal-Only, No New Experiments

日期：2026-07-10

## 1. Review target

请先读 `rebuttal/FABLE5_GPT56_ASSET_RESPONSE_MATRIX.md`，再复核 `rebuttal/AUTHOR_RESPONSE_PATH_B_READY_DRAFT.md` 和 `rebuttal/AUTHOR_RESPONSE_PATH_B_COMPACT.md`。目标是用仓库现有资产完成 NeurIPS rebuttal，而不是改已提交 PDF、扩写新论文或虚构补实验。

本轮明确边界：

- 没有修改 `paper/*.tex`、figure assets 或 `paper/main.pdf`；
- 所有 “will revise/correct” 都指后续修订，不表示 reviewer-facing PDF 已替换；
- 不新增或改写任何实验指标；
- 只使用可追溯的 surviving aggregate/manifest；
- 部分历史 checkpoint/raw JSON 未留存不等于不可复现；复现依据是公开数据、代码、配置和 rerun entrypoint。不要从均值/方差反推 per-seed 数字，但也不要因此撤回 aggregate conclusion。

## 2. Core rebuttal position

必须保住的核心贡献：

> EVQ-Cosh treats the fixed RoPE frequency table as a finite spectral budget and identifies training-time frequency allocation as a third PE design axis. The cosh family is derived for the stated surrogate; the deployed tau is a semi-analytic operating rule. EVQ changes the substrate on which inference-time scaling acts.

不能扩大成：

- universal long-context SOTA;
- tuned YaRN/LongRoPE replacement;
- globally optimal tau;
- production-identical DeepSeek validation;
- measured compute/memory/latency saving.

同时不能无故撤回：

- Primary I matched-scale 3-seed EVQ×YaRN conclusion；
- Primary II 在明确 seed-42 protocol 下的 Geo/DAPE/EVQ comparison；
- Primary III 432M 3-seed MLA scarce-channel result；
- 99-run 对 tau operating basin 的支持。

## 3. High-risk issues that the draft now concedes

### R1 — Theory

1. Shape/scale are not derived from one unified objective.
2. Constant-diagonal surrogate is tractable but not the exact RoPE kernel.
3. Proposition 1’s exponent is conditional on diffuse baseline and Pearson stiffness; Table 8 shows sensitivity.
4. Pure-tether is not quantitatively controlled at tau=4; trained `R_F`/forced-branch ablation is missing.
5. “Waterbed” is a divergence bound, not a theorem about PPL tradeoffs.
6. MLA `d_eff=d_head` is an empirical convention; `d_rope/sqrt(L)` remains untested.

### R2 — Empirics and reproducibility

1. No tuned geometric-base text control or Geo/EVQ YaRN-scale sweep.
2. Primary II Geo/DAPE/EVQ rows are seed 42 only.
3. Figure 8 uses an old n=200 accuracy pilot under an NLL caption.
4. Submitted Table 21 8K-raw Geo accuracy is an erratum: `513/2086 = 24.59%`, so 24.6%, not 26.6%; NLL values are unchanged.
5. Figure 9 mixes an approximately -81% progressive-training value with Table 20’s three-seed 454M FineWeb-Edu -13.3% row.
6. Table 20 is heterogeneous supporting evidence, not a controlled scaling law.
7. PK is teacher-forced NLL-gap unless explicitly labeled AR exact.
8. 1B MLA reversal is a single-seed schedule-sensitivity limitation, not saturation robustness.
9. Base vs EVQ-LoRA does not isolate EVQ from LoRA/LongAlign.

### R3 — Application and impact

1. QuALITY accuracy is near random at this scale.
2. Reported 8B LoRA does not improve RULER.
3. EVQ should not be enabled unconditionally; video base sweep and 1B MLA reversal define negative boundaries.
4. No measured system-efficiency benefit exists.

## 4. Evidence boundary

Safe, current artifacts:

| Artifact | What it supports | Boundary |
| --- | --- | --- |
| `data/curated/quality_454m_full_eval.json` | QuALITY n=2086 aggregate and 26.6→24.6 erratum | single-seed aggregate; not downstream SOTA |
| `data/curated/phase16_99run_manifest.csv` | 99 planned operating-rule runs: 45 pilot + 54 confirmation | empirical basin only; not global optimum |
| `data/curated/table18_mla_3seed_aggregate.json` | printed MLA mean/std values | aggregate only; no paired seeds/checkpoints |
| `data/curated/table2_evq_yarn_454m_passkey_10pct.json` | Primary I matched-scale aggregate/protocol | not a tuned-scale comparison |
| `data/curated/text_base_10k_500k_pilot.json` | text effect at base 10K and 500K | single-seed supporting pilot, not tuned sweep |
| `data/curated/mla_channel_count_125m_pilot.json` | within-MLA d_rope 32/16 direction | single-seed; d_rope=16 baseline weak |
| `data/curated/learnable_tau_128tok_evidence.json` | three-seed tau convergence and objective mismatch | supports mechanism, not fixed-EVQ multi-seed row |

Rerunnable upgrades, but not rebuttal blockers:

- full tuned-base and tuned-YaRN sweeps;
- Primary-II additional seeds;
- MLA `d_rope/sqrt(L)` direct tau ablation;
- trained `R_F` / `L_eff^J` measurements;
- Primary-I AR exact;
- matched Geo+LoRA table.

## 5. Questions for Fable5

Please answer with P0/P1/P2 severity and exact draft line numbers:

1. Does any sentence still imply the complete deployed method is derived from one objective?
2. Does any sentence turn matched-scale YaRN evidence into best-tuned dominance?
3. Is Primary II consistently called single-seed diagnostic evidence?
4. Are both Figure 8 and Figure 9 admitted as submitted-version errors, not reviewer misunderstandings?
5. Is the 26.6%→24.6% erratum explicit while keeping NLL unchanged?
6. Is aggregate-only MLA provenance clearly separated from per-seed/checkpoint evidence?
7. Does the draft accidentally promise or imply an experiment that is unavailable?
8. Does any response sound evasive, overly defensive, or more damaging than a concise concession?
9. Which paragraphs can be shortened without losing the answer to a reviewer veto?
10. Does any sentence incorrectly equate missing historical raw artifacts with non-reproducibility or withdraw a conclusion already supported by rerunnable code and curated aggregates?

## 6. Requested output

```markdown
## P0: must fix before sending
- [file:line] issue, smallest replacement, reason

## P1: clarity or persuasion
- ...

## Claims correctly scoped
- claim + supporting artifact

## Unsupported promises or evidence upgrades
- sentence + missing evidence

## Recommended compact final response
- a clean English version with no placeholders
```

Do not propose manuscript/PDF edits as the main deliverable. The main deliverable is a concise, truthful rebuttal using only current evidence.
