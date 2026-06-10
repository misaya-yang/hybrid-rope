# Primary Evidence Provenance Note

日期：2026-06-10

用途：解决 rebuttal 中最容易被 R2/AC 追击的 token budget、seed scope、training length provenance 问题。本文只使用当前可公开追溯的 repo 文件，不读取 `results/`、`internal/` 或 audit 目录。

## 0. Bottom Line

Rebuttal 里可以安全报告：

| Anchor | Model | Train length | Train tokens | Seeds | Data | Metric scope |
| --- | --- | ---: | ---: | --- | --- | --- |
| Primary I EVQ x YaRN | 454M decoder-only | 2048 | 100M | 42, 123, 7 | FineWeb-Edu + 10% synthetic passkey mix | PK is teacher-forced NLL-gap retrieval; PPL is full-sequence PPL in Table 2 |
| Primary II PE-dominant Table 4 | 125M decoder-only | 128 | 15M | Geo/DAPE/EVQ seed 42; learnable tau seeds 42/137/256 | FineWeb-Edu | PE-dominant 128->8K diagnostic |
| Primary II supporting Phase 11B | 125M decoder-only | 256 | 100M by script default | script header says 42/123/7; CLI default is 42/137/256 | FineWeb-Edu | Supporting L=256 curves, not the Table 4 primary contrast |
| Primary III MLA | 432M MLA | 8192 | 500M | 42, 43, 88 | FineWeb-Edu / MLA setting | Scarce-channel PPL stress test |

The important reconciliation:

- Table 4 / Primary II uses the 128-token DAPE-style protocol. It should be described as 15M tokens if token budget is reported.
- Phase 11B is a separate 256-token / 100M-token supporting protocol. It should not be used to claim Table 4 had 100M tokens.
- The seed scope is intentionally mixed in Table 4: Geo/DAPE/EVQ are retained seed 42; learnable tau is 3-seed mean/std.

## 1. Primary I: EVQ x YaRN

Authoritative files:

- `data/curated/table2_evq_yarn_454m_passkey_10pct.json`
- `paper/tables/table2_evq_yarn_main.tex`
- `paper/sections/05_experiments.tex`

Confirmed protocol:

| Field | Value | Evidence |
| --- | --- | --- |
| model | 454M decoder-only transformer | curated JSON protocol |
| train length | 2048 | curated JSON protocol; table caption |
| train tokens | 100M | curated JSON protocol |
| data | FineWeb-Edu + 10% synthetic passkey mix | curated JSON protocol; table caption |
| base | 500000 | curated JSON protocol |
| head_dim | 64 | curated JSON protocol |
| seeds | 42, 123, 7 | curated JSON protocol |
| YaRN scale | fixed s=8 on Geo and EVQ | curated JSON protocol; table caption |
| PK metric | teacher-forced NLL-gap retrieval | curated JSON protocol; main text metric definition |
| Table 2 PPL | full-sequence PPL scoring | curated JSON `ppl_note` |

Safe response:

> For Primary I, we will add token and seed scope explicitly: the 454M EVQ x YaRN matched-scale substrate test uses 100M training tokens, \(L_{\mathrm{train}}=2048\), seeds 42/123/7, and the same fixed YaRN scale \(s=8\) on Geo and EVQ. PK denotes teacher-forced NLL-gap retrieval, not autoregressive exact match.

Do not say:

- “This is a tuned YaRN leaderboard.”
- “PK is exact retrieval.”
- “The same result proves LongRoPE/Dynamic-NTK dominance.”

## 2. Primary II: PE-Dominant Table 4

Authoritative files:

- `data/curated/fig3_extreme_128.json`
- `docs/exp/2026-02-24_128tok_baseline_report.md`
- `paper/tables/table4_pe_dominant.tex`
- `paper/sections/05_experiments.tex`

Confirmed protocol:

| Field | Value | Evidence |
| --- | --- | --- |
| model | 125M decoder-only | historical report |
| train length | 128 | curated JSON protocol; historical report; table caption |
| train tokens | 15M | historical report |
| data | FineWeb-Edu | curated JSON protocol; historical report |
| base | 500000 | historical report |
| eval lengths | 128 through 8192 | historical report |
| Geo seed | 42 | curated JSON row |
| DAPE seed | 42 | curated JSON row |
| EVQ seed | 42 | curated JSON row |
| learnable tau seeds | 42/137/256 | curated JSON row; table caption |

Safe response:

> The PE-dominant Table 4 row is intentionally a seed-scoped diagnostic: Geo, DAPE, and EVQ use the retained seed-42 128-token protocol, while the learnable-tau row reports a 3-seed mean/std over 42/137/256. We will report the token budget as 15M for this Table 4 protocol and avoid mixing it with the separate Phase 11B \(L_{\mathrm{train}}=256\), 100M-token supporting curves.

Do not say:

- “Primary II is 3-seed for Geo/DAPE/EVQ.”
- “Table 4 used 100M tokens.”
- “This row proves comprehensive DAPE dominance.”

## 3. Phase 11B Is A Different Supporting Protocol

Authoritative file:

- `scripts/core_text_phases/phase11b_125m_dape.py`

Confirmed protocol from script header/defaults:

| Field | Value |
| --- | --- |
| model | 125M |
| train length | 256 |
| train tokens | 100M by default |
| purpose | scaling-law and DAPE compatibility curves |
| output | `results/core_text/phase11b/` |

Important caveat:

- The script header says seeds 42/123/7, while the CLI default in the same file is `42,137,256`. Rebuttal should not cite a Phase 11B seed set unless the actual run command or result manifest is inspected.

Safe response:

> The \(L_{\mathrm{train}}=256\) Phase 11B curves are separate supporting evidence. They should not be used to change the Table 4 token budget or seed scope.

Do not say:

- “Phase 11B proves Table 4 had 100M tokens.”
- “The Phase 11B seed set is confirmed” without an actual run manifest.

## 4. Primary III: MLA

Authoritative files:

- `paper/sections/05_experiments.tex`
- `paper/appendix/a3_supporting_results.tex`
- `paper/appendix/a2_experiment_details.tex`

Confirmed protocol:

| Field | Value | Evidence |
| --- | --- | --- |
| model | 432M MLA | main text / appendix |
| train length | 8192 | main text / appendix |
| train tokens | 500M | main text / appendix hyperparameter table |
| seeds | 42, 43, 88 | appendix MLA section |
| d_rope | 32, 16 RoPE channels | main text / appendix |
| base | 500K | appendix |

Safe response:

> The primary MLA row is a 3-seed, 500M-token scarce-channel stress test at \(L_{\mathrm{train}}=8192\). It is production-relevant as a compressed-RoPE regime, but not a production-identical DeepSeek configuration.

Do not say:

- “This is the DeepSeek production configuration.”
- “The 1B supporting row proves saturation robustness.”

## 5. Recommended Paper/Rebuttal Patch

Already applied in the working tree:

- `paper/appendix/a2_experiment_details.tex`: add token budget to the compact reproducibility table.
- `paper/tables/table_evidence_tier.tex`: rename the 1B row to schedule-sensitivity check.
- `paper/figs/fig5_downstream_qa.pdf/png`: regenerate QuALITY figure as Gold-NLL.

Still needed before final response:

- Geo+LoRA exact numbers and seed scope.
- Optional Primary I AR exact or explicit statement that PK remains diagnostic.
- Optional Geo+YaRN scale sweep or explicit matched-scale limitation.
