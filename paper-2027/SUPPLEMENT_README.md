# RoPE Has a Spectral Budget — anonymous supplement

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

Regenerate all main-paper figures:

```bash
python paper-2027/figs/make_fig_evidence_overview.py
python paper-2027/figs/make_fig_method_overview.py
python paper-2027/figs/make_fig_frequency_geometry.py
```

Run the packaged CPU tests and inspect the exact-range entrypoint without
starting training:

```bash
python -m pytest tests/test_rope_core.py tests/test_fmrope_125m_l256_500m.py -q
python scripts/core_text_phases/phase16_exact_range_factorial_m4.py --help
```

## Evidence map

| Paper role | Reviewer artifact |
| --- | --- |
| 151.9M three-seed fixed-support result | `data/curated/exact_range_151m_3seed_result.json` |
| 50.9M exact-range factorial | `rebuttal/rebuttal_0723/theory_results/m4_exact_range_factorial_evidence_20260726.json` |
| 432M three-seed MLA result | `data/curated/table18_mla_3seed_aggregate.json` |
| 1.485B adapted endpoints | `rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729/` |
| 8B probability and causal remote-source use | `data/curated/llama8b_causal_source_use_s42_20260714.json` |
| 8B adapted RULER endpoint | `rebuttal/rebuttal_0723/theory_results/llama8b_matched_ruler_mix_20260726.json` |
| Full sin/cos geometry and co-adaptation diagnostics | `scripts/analysis/` |

The exact-range data preparation, protocol, model, and training scripts are in
`rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/`. The figure scripts
contain frozen owner values and consistency assertions for Figures 1--3.
