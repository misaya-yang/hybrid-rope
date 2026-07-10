# Result Provenance Manifest

Purpose: reviewer-facing provenance ledger for the main EVQ-Cosh result claims.
This file records what the compact repository can currently prove, which files
hold the claim, and which artifact-level hashes are still missing. It does not
create new experimental evidence.

Important distinction: `Missing` below means missing from the current compact
review branch, not that the experiment was not run. Several raw/archival result
artifacts exist on `backup/2026-03-06` or in the external training environment.
For public reviewer materials, promote only sanitized, repo-relative artifacts
from those sources.

Historical script policy: `docs/overview/HISTORICAL_SCRIPT_STATUS.md` lists
server launch wrappers and patch scripts that are provenance clues rather than
current reviewer-facing commands.

Code/result policy: `docs/overview/EXPERIMENT_CODE_RESULT_AUDIT.md` records
whether each major row has code support, implementation support, and JSON/result
artifacts. Use that file before interpreting missing checkpoints as missing
experiments.

External artifact policy: when checkpoints/data/logs are recovered from an
external training machine or archival branch, first run
`scripts/core_text_phases/make_artifact_manifest.py` there. By default it emits
sanitized path hints, SHA256 hashes, tensor shape/dtype metadata, and optional
RoPE `inv_freq` audit summaries without absolute paths.

Hash convention: SHA256 values below are file hashes from the current working
tree at the time of this audit. If any listed file changes, recompute the hash
before using this manifest as a release artifact.

Status labels:

- `Curated`: values are available in a reviewer-facing curated artifact.
- `Packaged result`: values are available in a result JSON/report in this repo.
- `Supporting only`: do not use as a primary claim.
- `Missing`: exact raw artifact/checkpoint/data hash is not present in the
  compact repo.

## M1: Table 2 EVQ x YaRN

Claim scope:

- Primary I matched-scale substrate/range complementarity.
- 454M decoder-only transformer.
- `L_train=2048`.
- FineWeb-Edu with 10% synthetic passkey mix.
- Fixed YaRN scale `s=8` for Geo and EVQ.
- Seeds: 42, 123, 7.
- PK metric: teacher-forced NLL-gap retrieval rate, not autoregressive exact
  match.

Reviewer-safe statement:

> Under the same fixed YaRN scale in the 454M passkey-mix setting, YaRN has
> higher leverage on the EVQ-trained frequency substrate than on Geo.

Do not state:

- Tuned Geo+YaRN or LongRoPE-style baselines were beaten.
- PK is autoregressive exact match.
- `phase14c_multiscale_evq_yarn.py` is the full Table 2 reproduction.

Packaged evidence:

| Artifact | Role | SHA256 |
| --- | --- | --- |
| `paper/tables/table2_evq_yarn_main.tex` | Paper table | `9b07276a1036cadf0f48f863f5c6f7927a491bd497a78d25b92daef888dccca9` |
| `data/curated/table2_evq_yarn_454m_passkey_10pct.json` | Curated values/protocol | `d2c37769a0a166830d022778bf517c0973eb2ea80a9a87aea3bbadf597cee4ac` |
| `scripts/supporting_eval/eval_passkey_scratch.py` | PK sample/eval helpers | Recompute before release if cited |
| `scripts/core_text_phases/run_evq_sweep.py` | Core sweep entrypoint | Recompute before release if cited |
| `scripts/core_text_phases/phase14c_multiscale_evq_yarn.py` | Supporting multiscale check only | Recompute before release if cited |

Current compact-repo gaps:

- Original per-seed checkpoint hashes are not packaged.
- Original data artifact hash is not packaged.
- Original run log is intentionally replaced by curated JSON.
- The curated JSON is sufficient to inspect reported values, but not to verify
  checkpoint-level frequency provenance.

Branch audit note:

- `backup/2026-03-06` contains archival Table-2-adjacent raw artifacts such as
  `data/results_5090b/evq_yarn_10pct_allseeds.json` and
  `docs/exp/2026-03-03_passkey_mix_results.md`.
- Those branch artifacts confirm that the curated JSON was distilled from a
  larger run record, but they are not currently part of the compact main-branch
  reviewer path.
- If reused, sanitize old "6 seed" shorthand: the primary 10% Table 2 row is 3
  seeds per method; the 5% plus 10% EVQ+YaRN observation is supporting context.

Closure action:

- Add per-seed result manifests if checkpoints/logs are recovered.
- Use `scripts/core_text_phases/make_artifact_manifest.py` to import only
  sanitized external checkpoint/data metadata.
- Run `scripts/core_text_phases/audit_rope_checkpoint.py` on any recovered
  checkpoints and record `inv_freq` hashes.

## M2: Table 4 PE-Dominant Diagnostic

Claim scope:

- Primary II PE-dominant diagnostic, not broad downstream evidence.
- 125M FineWeb-Edu.
- `L_train=128`, evaluated at 8K.
- Geo, DAPE, and EVQ rows are seed 42.
- Learnable tau row is mean/std over seeds 42, 137, 256.

Reviewer-safe statement:

> In an extreme PE-dominant diagnostic, EVQ has lower seed-42 extrapolation PPL
> than Geo and DAPE without learned PE parameters.

Do not state:

- Geo/DAPE/EVQ rows are 3-seed validated.
- The 128-to-8K diagnostic is ordinary long-context downstream evidence.

Packaged evidence:

| Artifact | Role | SHA256 |
| --- | --- | --- |
| `paper/tables/table4_pe_dominant.tex` | Paper table | `6075b6f6f5ae39925f030293a145f06b29c689309116aca558463452e6f29331` |
| `data/curated/fig3_extreme_128.json` | Curated panel/table fallback | `3cbf44eb7166b037ed70b96546c302ab941e7214bdcb20b9477e999c1b9d09ee` |

Current compact-repo gaps:

- Geo/DAPE/EVQ additional seeds are not packaged because those rows are not
  multi-seed in the current table.
- Figure 3 panels (b,c) still depend on regenerated Phase 11 result JSONs unless
  additional curated fallbacks are added.

Branch audit note:

- `backup/2026-03-06` includes earlier raw PE-dominant artifacts under
  `data/evq_128tok_results/`, including some `inv_freq.npy` and result JSON
  files that were removed from the compact main branch.
- Those artifacts are useful for internal audit, but the current paper table
  should remain seed-scope explicit unless new curated multi-seed fallbacks are
  promoted.

Closure action:

- Either run two more seeds for Geo/DAPE/EVQ or keep every mention explicitly
  seed-42-scoped.
- Add curated fallbacks for Figure 3 panels (b,c) if reviewer supplement should
  regenerate figures without local Phase 11 outputs.

## M3: Primary MLA Scarce-Channel Stress Test

Claim scope:

- Primary III MLA scarce-channel stress test.
- 432M/350M-class MLA model.
- `L_train=8192`, 500M tokens.
- `d_rope=32`, 16 rotary frequency channels, `base=500K`.
- Seeds: 42, 43, 88.
- Matched-scale `+YaRN(s=4)` comparison.
- `tau=1.414` is an empirical `d_eff=128` operating convention; it is not
  derived from the released code fields `head_dim=64` or `d_rope=32`.

Reviewer-safe statement:

> In the 8K/500M 3-seed MLA scarce-channel stress test, EVQ and EVQ+YaRN improve
> extrapolation PPL under the tested matched-scale setting.

Do not state:

- This is production-identical DeepSeek MLA.
- The empirical `d_eff=128` convention is theoretically forced.
- The 1B/4K supporting row proves durability.

Packaged evidence:

| Artifact | Role | SHA256 |
| --- | --- | --- |
| `data/curated/table18_mla_3seed_aggregate.json` | Aggregate-only recovery of printed mean/std | `17fe0e104d2438a8315c2a441fb3fa45a7f59e1484d7ba0adbd5490b7c3719bb` |
| `paper/appendix/a3_supporting_results.tex` | Submitted paper MLA appendix table/prose | `7eb32f89fefc0be49ed42c5bf87543a1dfa5be9d51eb739df07aa3ab9200cd53` |
| `paper/sections/05_experiments.tex` | Submitted main experiment prose | `20daca808842fdc17b3cb8796dcdbac8b7a11aaba0e527c27bce899387a408b1` |
| `scripts/core_text_phases/run_gqa_evq_experiment.py` | Training entrypoint | `51ad863e3cc8193b5345423ca4c197282b317977060e6716db52529362bd94b0` |
| `scripts/core_text_phases/eval_extended_3seeds.py` | 3-seed eval; explicit checkpoint `inv_freq` audit logging and current/historical run-id resolution | `d0712bf243149ea63e0cc8ddbe3c4bde8d2fb4aadb1a39d85a123e0cab3dd3f0` |
| `scripts/core_text_phases/yarn_finetune_eval.py` | YaRN+FT supporting eval; explicit checkpoint `inv_freq` audit logging and current/historical run-id resolution | `ae2ae0536a43db4e9b15ffba3b54978b080837dedfe7fbf3918391af362dd471` |
| `scripts/core_text_phases/audit_rope_checkpoint.py` | Offline checkpoint frequency audit helper | `9d974b58b44f8b664d9f250cbf6a4d5ec505f5077cb55d1120b4f638f9e53ad9` |
| `scripts/core_text_phases/audit_training_artifacts.py` | Offline train-cache/token-count audit helper | `bb4a89d522557cc1878cfa37b16de9a7a97930228402a49e5fae6c1df5e6edd5` |
| `scripts/core_text_phases/make_artifact_manifest.py` | Sanitized external artifact manifest helper | `38cb1f687cc08aa23e518a4da2399cece69754446292bd92cb6dee153c0f4099` |
| `docs/overview/README.md` | Overview docs entrypoint and audit-stack navigation | `4fbd556bcc908e8705e4a88ebdaa5ce5732bf90afbd71ad1cddbbb1b3cb62f0b` |
| `docs/overview/OPUS48_REBUTTAL_MASTER_BRIEF.md` | Single advisor-facing rebuttal master brief | `61210f3de05535c8d5114025c9bced323713dac9b9db69eff5bce8a5dcab67c8` |
| `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md` | One-stop Opus 4.8 audit index and P0 checklist | `f84f22056513902888c87f9c52abb07713ac538039406ed7343108b777db5f20` |
| `docs/overview/OPUS48_ARTIFACT_RECOVERY_RUNBOOK.md` | External artifact recovery commands and acceptance criteria | `4857a2d859f07d0df9c80955dfb6995bf5412a7b4c167cfb0f54dd79ed6faad6` |
| `docs/overview/OPUS48_COMPLETION_AUDIT.md` | Requirement-level completion and remaining-gap audit | `9ba18985aaf8e50ef459b75bbe9999d12fc584d8d049d26a1bfdf309b56119b4` |
| `docs/overview/OPUS48_ISSUE_RESOLUTION_LEDGER.md` | Per-issue resolution state, evidence, and remaining gates | `f2bf3f142b3511b2e1da8160293156e8f4ecebd938218fd0204548dfa8b02ef2` |
| `docs/overview/OPUS48_REBUTTAL_RESPONSE_MATRIX.md` | Rebuttal-safe answer strategy and forbidden-claim matrix | `75bae5a7c4ef05dcfd93379f38d1ea2cb869c870e3b533f7560298301ac2dae4` |
| `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` | Prompt-structured forensic audit report | `8654e4a2007e132f45f8046480e0404f5048706357ee54d51b8f78e91fcdc44c` |
| `docs/overview/PAPER_CLAIMS_MAP.md` | Paper-to-experiment traceability map with explicit artifact gates | `0971ec6f7e96cbcf3a6cef16652ac18d03e21cb805a5869a697112fc776b1dc6` |
| `paper/REBUTTAL_PLAYBOOK.md` | Scoped rebuttal draft; must not override audit stack | `9526f6923bf73039444e0ecee7f0abd9384cc56254b7fb63c00f1cb0aceec18a` |
| `tests/test_opus48_audit_docs.py` | O48 coverage/link/stale-phrase regression test | `e7e9c1b5cbd82535bfffd0c788c01e9c46bfc6d212468ed15f77a145fb171ecb` |

Current compact-repo gaps:

- The original per-seed evaluation JSON and checkpoints are unavailable; the curated artifact preserves only the printed aggregate.
- Exact checkpoint hashes are not packaged.
- Exact data artifact hashes are not packaged.
- Some historical launch/eval wrappers use stale run directory names or
  non-repo-relative entrypoints.
- Direct `tau=d_rope/sqrt(L)` ablation is not reported.

Branch audit note:

- The visible `backup/2026-03-06` branch predates the final MLA 8K/500M and
  1B/4K reports found on main, so it is not a complete source for the MLA primary
  table.
- Current code is stronger than old provenance because the MLA eval scripts now
  explicitly hash checkpoint-loaded `inv_freq` before applying YaRN.

Closure action:

- Run `audit_rope_checkpoint.py` on exact 8K/500M checkpoints.
- Record checkpoint SHA256, `inv_freq` SHA256, inferred schedule family, and
  inferred tau for all seeds/arms.
- Add direct `tau=d_rope/sqrt(L)` MLA ablation or remove any implication that it
  has been resolved.

## M4: 1B/4K MLA Supporting Row

Claim scope:

- Supporting only.
- 4K train length, 1B tokens, seed 42 reported.
- Different data mixture from the primary 8K/500M MLA run.
- Old MLA-32/K16/base500K sparse frequency substrate.
- Raw EVQ improves 4K PPL but is worse at 8K/16K.
- EVQ+YaRN+FT is mildly better at target length, while some beyond-target
  lengths favor Geo+YaRN+FT.

Reviewer-safe statement:

> The 1B/4K MLA row is a limitation and root-cause target. It shows that EVQ can
> fail in a sparse 4K frequency-window regime, while still suggesting that the
> trained substrate can affect target-length YaRN+FT.

Do not state:

- The 1B row proves training durability.
- EVQ+YaRN wins at every length.
- This row is a same-config token-scaling continuation of the primary MLA run.

Packaged evidence:

| Artifact | Role | SHA256 |
| --- | --- | --- |
| `scripts/core_text_phases/run_350m_4k_1b.sh` | Historical launch script | `0f59e5fa97ddcd4b3ef925e4e2097779e5e425e9a501257b7b11d5a5e3279c05` |
| `scripts/core_text_phases/yarn_finetune_eval.py` | YaRN+FT eval script | `ae2ae0536a43db4e9b15ffba3b54978b080837dedfe7fbf3918391af362dd471` |
| `scripts/core_text_phases/audit_rope_checkpoint.py` | Required artifact-audit helper | `9d974b58b44f8b664d9f250cbf6a4d5ec505f5077cb55d1120b4f638f9e53ad9` |
| `scripts/core_text_phases/audit_training_artifacts.py` | Required train-cache/token-count audit helper | `bb4a89d522557cc1878cfa37b16de9a7a97930228402a49e5fae6c1df5e6edd5` |
| `scripts/core_text_phases/make_artifact_manifest.py` | Sanitized external artifact manifest helper | `38cb1f687cc08aa23e518a4da2399cece69754446292bd92cb6dee153c0f4099` |

Current compact-repo gaps:

- The original Phase 18 Markdown report is unavailable; only summary-level values in the paper/archive survive.
- Exact checkpoint hashes are not packaged.
- Exact training data hash is not packaged.
- Exact baseline and YaRN+FT JSON files are not packaged; the compact branch
  currently has Markdown reports for this row.
- Seeds 43 and 88 are not reported for the 1B/4K row.
- Historical launch script provenance is weaker than primary-table provenance.

Branch audit note:

- The currently visible archival branch does not close the 1B/4K gap. Treat the
  external training environment as the likely source of missing checkpoints/logs
  and import only sanitized manifests into the public repo.

Closure action:

- Treat as limitation unless checkpoint/data hashes and additional seeds are
  recovered.
- Use frequency-window analysis and checkpoint `inv_freq` audit to decide
  whether the reversal is a schedule/window failure or an artifact.

## M5: Phase 16 Operating-Rule Sweep Recovery

Claim scope:

- Nine `(L, H, d_head)` configurations.
- Forty-five pilot runs at seed 42 and 54 confirmation runs at seeds 137/256.
- Ninety-nine planned runs total; two extra local rerun artifacts are excluded by exporting from the plans.
- Supports a near-optimal empirical basin, not a globally optimal tau theorem.

Packaged evidence:

| Artifact | Role | SHA256 |
| --- | --- | --- |
| `data/curated/phase16_99run_manifest.csv` | Sanitized one-row-per-run manifest | `39ce676ca26967434c0091e09d36824cd16d1a1a204ad464dad0a33aef7b18d5` |
| `scripts/core_text_phases/export_phase16_manifest.py` | Torch-free deterministic exporter | `cdbe0c01682989b712cafb98d604bb12ad9534efda505c00ec1680c2033eade5` |
| `docs/exp/2026-03-09_phase16_formula_optimality_sweep_results.md` | Human-readable result summary | Recompute before release if cited |

Recovery boundary:

- The flat manifest preserves planned configuration, seed, tau, training-token, PPL, passkey-summary, and frequency-hash fields.
- It intentionally excludes absolute paths, host names, private environment fields, and checkpoint binaries.
- The raw ignored result tree remains local evidence and is not itself a reviewer artifact.

## M6: QuALITY Full-Evaluation Rebuttal Source

Claim scope:

- 454M, single seed, `n=2086` full evaluation.
- Accuracy remains near the 25% random baseline; Gold-answer NLL is the supporting signal.
- The earlier `n=200` pilot is superseded and must not be used as rebuttal evidence.
- The submitted Figure 8 remains stale/mislabeled; this rebuttal pass does not modify the PDF.

Packaged evidence:

| Artifact | Role | SHA256 |
| --- | --- | --- |
| `data/curated/quality_454m_full_eval.json` | Curated full-eval aggregate | `e09aec916b856f75413e3abc5e9b0c67d1888a69069d462bc297cf03fda70da5` |
| `rebuttal/FIGURE_TABLE_AUDIT.md` | Submitted Figure 8/9 error audit and safe response wording | `6db79faca4fdffb9195cb3513785a6bcf5155f55edd9f1e8708ec7cb9171d8af` |

## M7: Existing Supporting Assets Used In Rebuttal

These assets answer audit questions without changing the submitted PDF or
claiming that a supporting pilot is a new primary result.

| Artifact | Role | SHA256 |
| --- | --- | --- |
| `data/curated/text_base_10k_500k_pilot.json` | Shows the text direction at base 10K and 500K | `f5a8e8d8c8a658d8ae91614b7c4ffb53ef14fee19fa172b3feef51241454b553` |
| `data/curated/mla_channel_count_125m_pilot.json` | Within-MLA d_rope 32/16 channel-scarcity pilot | `ed9f99e27d4369b6bb3461964e0b4c3c64b6d1f82ae4dea628974944cef1705c` |
| `data/curated/learnable_tau_128tok_evidence.json` | Three-seed tau convergence and in-range/OOD objective split | `b357dea87f5caee2a3440bf99586daa3e0dae88f9414d70260d8daf9af5a7c5a` |
| `rebuttal/FABLE5_GPT56_ASSET_RESPONSE_MATRIX.md` | Maps Fable5/GPT-5.6 issues to current assets and rerun paths | `5d0167120a297cb599eba38a9a21265bebd13349d698873009cba40aef2beb09` |

Claim boundaries:

- The base and MLA channel-count artifacts are single-seed supporting pilots,
  not tuned-baseline or multi-seed primary replacements.
- The learnable-tau artifact explains the training-objective mismatch and does
  not turn fixed EVQ in Table 4 into a multi-seed row.
- Independent reproduction remains available through the public scripts,
  configurations, and data-preparation path even when an original historical
  checkpoint is not packaged.

## Release Checklist

- [x] Recompute file hashes after final edits in this audit pass.
- [ ] Run `make_artifact_manifest.py` on recovered external artifacts before
  promoting them into reviewer-facing docs.
- [ ] Add checkpoint/data hashes for any recovered primary artifacts.
- [ ] Keep Table 2 PK wording as teacher-forced NLL-gap.
- [ ] Keep Table 4 seed scope explicit.
- [x] Keep MLA `d_eff=128` wording as an empirical convention distinct from
  code `head_dim=64` and `d_rope=32`.
- [ ] Do not use 1B/4K as primary support.
- [ ] Do not cite stale launch scripts as authoritative reproduction entrypoints
  unless they are cleaned or marked historical.
