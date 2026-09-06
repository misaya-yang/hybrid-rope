# RoPE Has a Spectral Budget — anonymous supplement

> **2026-09-06 repo slim:** the working branch `main_0726_09_06` keeps only
> `paper-2027/`, `docs/`, `scripts/`, `tests/`, and the root routing files.
> Build-source paths cited below outside that set (`data/curated/`,
> `rebuttal/rebuttal_0723/`, `experiments/`) now exist only on branch
> `main_0726` (full pre-slim state, unchanged). The supplement zip itself is
> self-contained for reviewers; this note concerns repository provenance only.
> The zip now lives at
> `paper-2027/rope-spectral-budget-iclr2027-supplement.zip`.

This archive contains the ICLR 2027 paper source, the EVQ-Cosh frequency
implementation, the primary fixed-support training entrypoint, frozen figure
generators, CPU diagnostics, tests, and sanitized machine-readable evidence.

## Quick checks

Install the pinned Python dependencies in an isolated Python 3.10 environment:

```bash
python -m pip install -r requirements-lock.txt
```

Build the paper:

```bash
cd paper-2027
./compile.sh
cd ..
```

Regenerate every figure bundled with this paper:

```bash
python paper-2027/figs/make_fig_8b_causal_source_use.py
python paper-2027/figs/make_fig_evidence_overview.py
python paper-2027/figs/make_fig_frozen_fixed_support.py
python paper-2027/figs/make_fig_frequency_geometry.py
python paper-2027/figs/make_fig_exact_range_control.py
python paper-2027/figs/make_fig_olmo_scale_crossover.py
python paper-2027/figs/make_fig_spectral_budget_scaling.py
```

Run the complete packaged CPU suite and inspect the exact-range entrypoint
without starting training:

```bash
python -m pytest -q
python scripts/core_text_phases/phase16_exact_range_factorial_m4.py --help
```

## Evidence map

| Paper role | Reviewer artifact |
| --- | --- |
| 151.9M three-seed fixed-support result | `data/curated/exact_range_151m_3seed_result.json` |
| 50.9M exact-range factorial | `rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json` |
| 99-run zero-search operating-prior study | `data/curated/phase16_99run_manifest.csv`; `scripts/core_text_phases/phase16_formula_optimality_sweep.py` |
| 432M three-seed MLA result | `data/curated/table18_mla_3seed_aggregate.json` |
| 750M continued-training result | `data/curated/phase15_750m_continue_result_20260306.json` |
| Frozen OLMo/Qwen fixed-support controls and Qasper policy endpoint | `data/curated/frozen_fixed_support_mature_20260823.json` |
| 1.485B adapted endpoints and exact Q/K protocol | `rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/metrics.json` |
| 8B probability and causal remote-source use | `data/curated/llama8b_causal_source_use_s42_20260714.json` |
| 8B adapted RULER endpoint | `rebuttal/rebuttal_0723/theory_results/llama8b_matched_ruler_mix_20260726.json` |
| 129.6M video-DiT head-to-head result | `data/curated/video_dit_seed42_head_to_head_20260826.json` |
| Full sin/cos geometry and co-adaptation diagnostics | `scripts/analysis/` |

The appendix also reports two completed extensions whose full raw payloads are
not copied into this anonymous archive. The 151.9M two-seed weights-by-table
crossing is reproduced as an aggregate table and protocol in
`paper-2027/appendix/a5_identification.tex`; its canonical raw receipt has
SHA-256 `9304752d885d73af38262958f033843c642f394a448e2306fbfb57a4e1996e22`.
The fresh FineWeb-Edu holdout-512 allocation/routing controls and the compact
PG-19/RULER-13/Qasper/2Wiki policy endpoints are reproduced in
`paper-2027/appendix/a6_mature_scale.tex`. These aggregates preserve the
allocation-only versus bundled-policy distinction; the policy gains are not
relabeled as pure-`z` effects.

Checkpoint bytes, raw generations, token arrays, caches, machine logs, and raw
files containing private absolute paths are intentionally excluded. Their
sanitized owners retain checkpoint, data, table, code, result, and example
hashes. A reported aggregate or executable entrypoint is therefore not a claim
that the corresponding private raw artifact is packaged here.

The exact-range data preparation, protocol, and training scripts are in
`rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/`; the model
definition it trains is `experiments/native_rope_evq_150m/model.py`, also
bundled here. The figure scripts
contain frozen owner values and consistency assertions.
