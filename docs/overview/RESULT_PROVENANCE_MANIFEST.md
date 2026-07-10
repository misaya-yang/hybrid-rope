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
- `Raw JSON backed`: the tracked artifact embeds values from an exact recovered
  source JSON and records that source file's SHA256.
- `Report backed`: values are transcribed from a tracked narrative report; the
  exact raw/full-evaluation JSON is unavailable and must not be implied.
- `Sanitized run manifest`: per-run configuration/metric rows and hashes are
  portable, but the original source directory/checkpoints are not packaged.
- `Trace only`: a historical summary survives without enough source provenance
  for rebuttal use; do not cite its numbers.
- `Packaged result`: values are available in a result JSON/report in this repo.
- `Supporting only`: do not use as a primary claim.
- `Missing`: exact raw artifact/checkpoint/data hash is not present in the
  compact repo.

Portable July reconciliation:

| Artifact | Tier | Artifact SHA256 | Reviewer-use boundary |
| --- | --- | --- | --- |
| `data/curated/table18_mla_3seed_aggregate.json` | Raw JSON backed | `ad751a26fee43939f644a8ee12e5e50b0003d0d490f790ce1cd85971902ffe8d` | Primary III seedwise/aggregate metrics; no d_eff-convention claim. |
| `data/curated/phase11_l256_3seed_recovered.json` | Raw JSON backed | `8af8bce33e96f70542d943745bebbbaaa7cc65117587950b75c584f06a2f68db` | L=256 archival Geo/EVQ/scaling records; not L=128 Primary II replication. |
| `data/curated/phase16_99run_manifest.csv` | Sanitized run manifest | `39ce676ca26967434c0091e09d36824cd16d1a1a204ad464dad0a33aef7b18d5` | Supports run coverage and basin/rank audit; not checkpoint reproduction. |
| `data/curated/learnable_tau_128tok_evidence.json` | Report backed | `3873c1bd6dfe2b70eb6eb7ed770946ccd271256174bdee9babd8760df7d7f1cd` | Final tau endpoints, not a per-step trajectory. |
| `data/curated/mla_channel_count_125m_pilot.json` | Report backed | `03c690f4f69ce64285ac1015addda402944c5e7bf7ef9490d8a4b39b6ae16ca7` | Single-seed qualitative support, not a d_eff/tau ablation. |
| `data/curated/quality_454m_full_eval.json` | Raw JSON backed | `648442141fc94c06db5143283ea95eb46133dcb2ceda39bbffafa17b738cdb84` | Correct n=2,086 table/figure values; accuracy remains inconclusive. |
| `data/curated/text_base_10k_500k_pilot.json` | Raw JSON backed | `fbd4c04abdfe13adf8578bc49e40f084942aab8b0e18d207a026208e51ebd6c4` | Single-seed 151.9M supporting pilot; not a tuned-base sweep or `c_pred` control. |

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
| `paper/tables/table2_evq_yarn_main.tex` | Paper table | `372ac2365ae316a885e556d754268c95e7831aa3df7c81e7decc27d02e560206` |
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
| `data/curated/learnable_tau_128tok_evidence.json` | Report-backed seedwise final-tau endpoints | `3873c1bd6dfe2b70eb6eb7ed770946ccd271256174bdee9babd8760df7d7f1cd` |
| `data/curated/phase11_l256_3seed_recovered.json` | Raw-backed L=256 archival payload; distinct protocol | `8af8bce33e96f70542d943745bebbbaaa7cc65117587950b75c584f06a2f68db` |

Current compact-repo gaps:

- Geo/DAPE/EVQ additional seeds are not packaged because those rows are not
  multi-seed in the current table.
- Figure 3 panels (b,c) values are preserved in a portable Phase11 snapshot,
  but the existing figure generator is not yet wired to that consolidated
  schema.

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
| `data/curated/table18_mla_3seed_aggregate.json` | Raw-backed portable result JSON; embeds ignored-source hash `1e44d30...30953` | `ad751a26fee43939f644a8ee12e5e50b0003d0d490f790ce1cd85971902ffe8d` |
| `data/curated/mla_channel_count_125m_pilot.json` | Report-backed single-seed channel-count pilot | `03c690f4f69ce64285ac1015addda402944c5e7bf7ef9490d8a4b39b6ae16ca7` |
| `paper/appendix/a3_supporting_results.tex` | Paper MLA appendix table/prose | `b1f8380d6a23b7f109d78ed0606cee7a69649ee4a0413510d05b117e97613715` |
| `paper/sections/05_experiments.tex` | Main experiment prose | `f2b2294b20769c4bf7b672cd65b4e393e2d36ee8a369d0c9e522612a1501391e` |
| `scripts/core_text_phases/run_gqa_evq_experiment.py` | Training entrypoint | `51ad863e3cc8193b5345423ca4c197282b317977060e6716db52529362bd94b0` |
| `scripts/core_text_phases/eval_extended_3seeds.py` | 3-seed eval; explicit checkpoint `inv_freq` audit logging and current/historical run-id resolution | `d0712bf243149ea63e0cc8ddbe3c4bde8d2fb4aadb1a39d85a123e0cab3dd3f0` |
| `scripts/core_text_phases/yarn_finetune_eval.py` | YaRN+FT supporting eval; explicit checkpoint `inv_freq` audit logging and current/historical run-id resolution | `ae2ae0536a43db4e9b15ffba3b54978b080837dedfe7fbf3918391af362dd471` |
| `scripts/core_text_phases/audit_rope_checkpoint.py` | Offline checkpoint frequency audit helper | `9d974b58b44f8b664d9f250cbf6a4d5ec505f5077cb55d1120b4f638f9e53ad9` |
| `scripts/core_text_phases/audit_training_artifacts.py` | Offline train-cache/token-count audit helper | `bb4a89d522557cc1878cfa37b16de9a7a97930228402a49e5fae6c1df5e6edd5` |
| `scripts/core_text_phases/make_artifact_manifest.py` | Sanitized external artifact manifest helper | `38cb1f687cc08aa23e518a4da2399cece69754446292bd92cb6dee153c0f4099` |
| `docs/overview/README.md` | Overview docs entrypoint and audit-stack navigation | `4fbd556bcc908e8705e4a88ebdaa5ce5732bf90afbd71ad1cddbbb1b3cb62f0b` |
| `docs/overview/OPUS48_REBUTTAL_MASTER_BRIEF.md` | Single advisor-facing rebuttal master brief | `6c4c5c3fde341fa4fc235c5c44d3490debb5cbd9746b926cd6c235aa27554470` |
| `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md` | One-stop Opus 4.8 audit index and P0 checklist | `f84f22056513902888c87f9c52abb07713ac538039406ed7343108b777db5f20` |
| `docs/overview/OPUS48_ARTIFACT_RECOVERY_RUNBOOK.md` | External artifact recovery commands and acceptance criteria | `4857a2d859f07d0df9c80955dfb6995bf5412a7b4c167cfb0f54dd79ed6faad6` |
| `docs/overview/OPUS48_COMPLETION_AUDIT.md` | Requirement-level completion and remaining-gap audit | `9ba18985aaf8e50ef459b75bbe9999d12fc584d8d049d26a1bfdf309b56119b4` |
| `docs/overview/OPUS48_ISSUE_RESOLUTION_LEDGER.md` | Per-issue resolution state, evidence, and remaining gates | `f2bf3f142b3511b2e1da8160293156e8f4ecebd938218fd0204548dfa8b02ef2` |
| `docs/overview/OPUS48_REBUTTAL_RESPONSE_MATRIX.md` | Rebuttal-safe answer strategy and forbidden-claim matrix | `75bae5a7c4ef05dcfd93379f38d1ea2cb869c870e3b533f7560298301ac2dae4` |
| `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` | Prompt-structured forensic audit report | `ec51bb535eb9badc97a0e0d55aa1ccc5e2486e830c813f437403370e71c71a5b` |
| `docs/overview/PAPER_CLAIMS_MAP.md` | Paper-to-experiment traceability map with explicit artifact gates | `cf9ed462b193a79f1096f73ff9f52153e0ca9d228ca09a37d4da25ebc7f0b293` |
| `paper/REBUTTAL_PLAYBOOK.md` | Scoped rebuttal draft; must not override audit stack | `ef94296fb39c9b6b2e7b22f16c23851d94b6c90d25e9f7cc218d1fd337e6dbe4` |
| `tests/test_opus48_audit_docs.py` | O48 coverage/link/stale-phrase regression test | `e7e9c1b5cbd82535bfffd0c788c01e9c46bfc6d212468ed15f77a145fb171ecb` |

Current compact-repo gaps:

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

- Exact checkpoint hashes are not packaged.
- Exact training data hash is not packaged.
- Exact baseline and YaRN+FT JSON files are not packaged, and the compact branch
  does not contain a reviewer-grade result report for this row.
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
