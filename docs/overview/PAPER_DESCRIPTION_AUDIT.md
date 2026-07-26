# Paper Description Audit

Purpose: audit whether the paper text says only what the experiment code and
JSON/result artifacts can support. This is the paper-description layer on top of
`rebuttal/rebuttal_0723/theory_results/EVQ_COSH_REBUTTAL_PRINCIPLES.md` and
`EXPERIMENT_CODE_RESULT_AUDIT.md`.

## Summary

The main body is mostly reviewer-safe after the current cleanup: PK is defined
as teacher-forced NLL-gap retrieval, EVQ x YaRN is framed as matched-scale
leverage rather than tuned-scaler dominance, PE-dominant rows state the seed
scope, and MLA `tau=1.414` is described as an empirical `d_eff=128`
operating convention distinct from the code fields `head_dim=64` and
`d_rope=32`.

The main remaining risks are appendix/supporting rows where the prose can sound
broader than the artifact chain:

- 1B/4K MLA has code support and a Markdown report, but no exact compact-branch
  JSON. Keep it as limitation/supporting only.
- Fig. 3(b,c) have code and archival JSON, but the compact branch does not
  package the expected Phase 11 JSONs.
- Supporting LoRA/video/progressive rows are useful mechanism checks but should
  not carry the main claim.

## Paper Text Checks

| Location | Paper wording risk | Code/result evidence | Current status |
| --- | --- | --- | --- |
| `paper/sections/05_experiments.tex`, evaluation scope | PK could be confused with autoregressive exact match. | `data/curated/table2_evq_yarn_454m_passkey_10pct.json` states teacher-forced NLL-gap; `scripts/supporting_eval/eval_passkey_scratch.py` implements NLL-gap eval. | Safe: section defines PK as teacher-forced NLL-gap unless explicitly marked AR exact. |
| `paper/tables/table2_evq_yarn_main.tex` | Table uses `PK@8K` shorthand. | Same curated JSON and PK helper. | Acceptable because the surrounding section defines PK; keep caption concise. |
| `paper/tables/table3_capability_passkey.tex` | Capability table said retrieval robustness without restating teacher-forced scope. | Same curated JSON; rows are not AR exact. | Fixed: caption now says teacher-forced retrieval robustness. |
| `paper/sections/05_experiments.tex`, Primary II | Risk of implying DAPE-wide dominance. | `data/curated/fig3_extreme_128.json` is seed-42 Geo/DAPE/EVQ and 3-seed learnable tau. | Safe: text says tested protocol, not comprehensive dominance. |
| `paper/appendix/a4_supporting_experiments.tex`, Fig. 3 caption/table | Phase 11 curves require JSON not present in compact branch. | `scripts/figures/fig3_pe_dominant_scaling.py` expects `results/core_text/phase11/results_phase11_raw.json` and `results_phase11_yarn.json`; visible archival branch has them. | Code-supported but not compact-JSON-supported; restore sanitized JSON or add curated fallbacks before claiming one-command regeneration. |
| `paper/sections/05_experiments.tex`, Primary III MLA | `d_eff=d_head=128` was not aligned with the released code field names. | `run_gqa_evq_experiment.py`, `eval_extended_3seeds.py`, and `yarn_finetune_eval.py` set code `head_dim=64`, `d_rope=32`, and use `tau=1.414`. | Fixed wording: paper now calls `d_eff=128` an empirical MLA operating convention, not a derivation from code `head_dim` or `d_rope`. Keep direct `tau=d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` ablations as open reviewer gaps. |
| `paper/appendix/a3_supporting_results.tex`, MLA table caption | Caption said EVQ+YaRN was best at every tested length, but 8K in-distribution PPL is slightly lower for Geo. | `results/eval_3seeds_full_results.json`: at 8K, Geo mean is 35.44 and EVQ+YaRN(s=4) mean is 35.83; at extrapolated 16K/24K/32K, EVQ+YaRN(s=4) is best. | Fixed: caption now says best at every extrapolated length and notes small 8K in-distribution differences. |
| `paper/appendix/a3_supporting_results.tex`, MLA progression prose | "Intrinsic property" could overstate beyond the 8K/500M stress test. | `results/eval_3seeds_full_results.json` supports monotonic 16K EVQ advantage across 50/75/100% checkpoints in this setup. | Fixed: prose now says tied to frequency allocation in this 8K/500M stress test, not only a late-training artifact. |
| `paper/appendix/a4_supporting_experiments.tex`, 1B/4K MLA paragraph | Could sound like strong evidence despite missing exact JSON in compact branch. | The paper retains the scoped row; `run_gqa_evq_experiment.py` and `yarn_finetune_eval.py` can produce the JSONs, but the historical report and exact baseline/YaRN+FT JSON are not tracked. | Leave supporting-only. Do not cite in rebuttal as JSON-backed until exact JSON is recovered/regenerated. |
| `paper/appendix/a4_supporting_experiments.tex`, signal-gradient paragraph | Passkey layer did not specify teacher-forced. | Curated Table 2/3 PK rows are teacher-forced NLL-gap. | Fixed: paragraph now says teacher-forced passkey. |
| `paper/appendix/a4_supporting_experiments.tex`, LoRA | Single-seed and missing matched Geo+LoRA control. | Paper already names supporting post-hoc evidence and natural follow-up. | Keep supporting only. |
| `paper/appendix/a2_experiment_details.tex`, 750M | Large single-seed numbers can be overused in rebuttal. | Current report is Markdown-backed; table marks single seed/supporting. | Safe if kept as supporting consistency check, not scale proof. |

## Required Paper-Side Rules

- Keep "matched-scale YaRN leverage" as the Table 2 claim. Do not say tuned
  Geo+YaRN, LongRoPE, or LongRoPE2 was beaten.
- Keep PK wording as teacher-forced NLL-gap unless a row explicitly reports AR
  exact match.
- Keep Table 4 seed scope explicit.
- Keep MLA `d_eff=128` as an empirical convention; code `head_dim=64` and
  `d_rope=32` should not be conflated.
- Keep 1B/4K MLA as a limitation/root-cause target unless exact JSON artifacts
  are recovered.
- Keep LoRA/video/progressive/750M as supporting evidence.

## Reviewer-Safe Rebuttal Wording

Use:

> The current paper text and code support a mechanism claim: training-time
> frequency allocation changes the substrate on which matched inference-time
> scaling acts. The evidence is strongest for the 454M EVQ x YaRN stress test,
> the seed-scoped PE-dominant diagnostic, and the 8K/500M 3-seed MLA stress
> test. The 1B/4K MLA row is code-backed and report-backed but not
> compact-JSON-backed, so we treat it as a limitation and a diagnostic of sparse
> frequency-window failure rather than as primary support.

Do not use:

> EVQ+YaRN is universally best, EVQ replaces YaRN/LongRoPE, or the 1B run proves
> training durability.

## Historical Report Cleanup

Two historical progressive reports kept useful raw observations but used wording
that was too broad for rebuttal reuse:

- `docs/exp/2026-03-09_phase17_evq_yarn_overlay_results.md` now has an audit
  note scoping "practical recipe" language to that single supporting overlay
  run.
- `docs/exp/2026-03-11_phase17c_2048_continue_results.md` now has an audit
  note scoping the 48K and AR-exact passkey observations to single-seed
  supporting evidence.

These reports should remain historical/supporting; the primary rebuttal should
cite the curated Table 2, Table 4, and 3-seed MLA artifacts instead.
