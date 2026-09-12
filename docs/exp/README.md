# `docs/exp/` — historical experiment reports by month

This is the NeurIPS-era report archive. It is chronological, not authoritative:
current claim ownership and evidence ceilings are in [`../../INDEX.md`](../../index.md).
Nearby scripts/checkpoints do not upgrade a report's provenance tier.

## Month folders

| Folder | Main work recorded | Current reading rule |
| --- | --- | --- |
| [`2026-02/`](2026-02/) | 128-token baselines, early EVQ runs, finite tau sweeps | foundation/history; finite grids do not prove continuous optima |
| [`2026-03/`](2026-03/) | passkey/PPL composition, L=256, 750M continuation, formula sweep, QuALITY, Video-DiT, GQA/MLA | systems breadth; keep endpoint/operator identities separate |
| [`2026-04/`](2026-04/) | thinking-token streaming plan | plan only |
| [`2026-07/`](2026-07/) | rebuttal-era operator, LoRA, MLA, readout, and exact-range plans/results | historical inputs; July canonical owners often live under `rebuttal/` |

Every dated file remains in its execution month and keeps its original
`YYYY-MM-DD_slug.md` name so Git history and citations stay recoverable.

## High-value historical routes

| Topic | Report | Current boundary |
| --- | --- | --- |
| finite tau landscape | [`2026-02/2026-02-27_evq_tau_sweep_results.md`](2026-02/2026-02-27_evq_tau_sweep_results.md) | non-monotone historical grid; not an optimum theorem |
| 454M passkey/PPL composition | [`2026-03/2026-03-03_passkey_mix_results.md`](2026-03/2026-03-03_passkey_mix_results.md) | teacher-forced NLL-gap, repository YaRN-style scope |
| L=256 scale evidence | [`2026-03/2026-03-04_phase11_L256_results.md`](2026-03/2026-03-04_phase11_L256_results.md) | supporting historical protocol |
| Kerple+MLP compatibility | [`2026-03/2026-03-05_phase11b_125m_results.md`](2026-03/2026-03-05_phase11b_125m_results.md) | Zheng-inspired DAPE-ish, not official DAPE reproduction |
| 750M continuation | [`2026-03/2026-03-06_phase15_750m_2k_to_4k_continue_results.md`](2026-03/2026-03-06_phase15_750m_2k_to_4k_continue_results.md) | supporting scale/persistence owner |
| 99-run formula sweep | [`2026-03/2026-03-09_phase16_formula_optimality_sweep_results.md`](2026-03/2026-03-09_phase16_formula_optimality_sweep_results.md) | tested finite basin only |
| QuALITY diagnostics | [`2026-03/2026-03-12_phase21_quality_downstream_report.md`](2026-03/2026-03-12_phase21_quality_downstream_report.md) | pilot/protocol diagnosis, not clean downstream confirmation |
| GQA/MLA scarcity | [`2026-03/2026-03-20_gqa_mla_125m_compression_ablation.md`](2026-03/2026-03-20_gqa_mla_125m_compression_ablation.md) | single-seed architecture-confounded pilot |
| M4 exact-range plan | [`2026-07/2026-07-24_m4_exact_range_factorial_plan.md`](2026-07/2026-07-24_m4_exact_range_factorial_plan.md) | result owner is in `rebuttal/rebuttal_0723/theory_results/` |

For the full chronology and how interpretations changed, read
[`paper-2027/research/history/TIMELINE.md`](../../paper-2027/research/history/TIMELINE.md).

## Maintenance rule

New historical reports use `docs/exp/YYYY-MM/YYYY-MM-DD_slug.md` and must link
to a canonical owner or state that raw evidence is missing. New active ICLR
result owners belong under `paper-2027/research/evidence/` or the appropriate
`attention-aware-retrofit/` subfolder, not here.
