# Historical Script Status

Purpose: distinguish current reviewer-facing reproduction entrypoints from
historical external/server helper scripts. Many experiments were run outside the
compact review branch. This file prevents old launch wrappers from being
mistaken for authoritative reproduction commands.

Rule: do not cite a historical launcher as the canonical reviewer path unless it
has been cleaned, made repo-relative, and paired with a result manifest.

## Current Reviewer-Facing Entrypoints

Use these as the preferred public paths when describing reproduction or audit
logic.

| Scope | Current path | Status |
| --- | --- | --- |
| EVQ-Cosh schedule API | `scripts/lib/rope/schedules.py` | Canonical |
| Core text sweeps | `scripts/core_text_phases/run_evq_sweep.py` | Canonical code path |
| EVQ x YaRN supporting rerun | `scripts/core_text_phases/phase14c_multiscale_evq_yarn.py` | Supporting only; not full Table 2 reproduction |
| Passkey helpers | `scripts/supporting_eval/eval_passkey_scratch.py` | Canonical metric/sample helper |
| PE-dominant diagnostic | `scripts/core_text_phases/phase11b_125m_dape.py` | Current script, but Table 4 seed scope remains row-specific |
| MLA/GQA training | `scripts/core_text_phases/run_gqa_evq_experiment.py` | Current training code; needs per-run manifest for primary MLA rows |
| MLA 3-seed eval | `scripts/core_text_phases/eval_extended_3seeds.py` | Current eval code; now logs checkpoint-loaded `inv_freq` hashes |
| MLA YaRN+FT eval | `scripts/core_text_phases/yarn_finetune_eval.py` | Current eval code; now logs checkpoint-loaded `inv_freq` hashes |
| Checkpoint frequency audit | `scripts/core_text_phases/audit_rope_checkpoint.py` | New artifact gate |
| External artifact manifest | `scripts/core_text_phases/make_artifact_manifest.py` | New sanitized manifest gate for recovered external checkpoints/data |

## Historical Launch Wrappers

These scripts encode real server workflows or debugging sessions, but they are
not reviewer-facing commands in their current form.

| Pattern/examples | Why historical | Reviewer-safe handling |
| --- | --- | --- |
| `scripts/core_text_phases/run_350m_mla32_500m.sh`, `run_350m_seeds.sh` | Server launch wrappers with fixed external run roots and older run-name assumptions. | Use as provenance clues only. Pair the result JSON with a sanitized manifest. |
| `scripts/core_text_phases/run_350m_4k_1b.sh`, `run_350m_4k_v2_1b.sh`, `run_350m_4k_v2_continue.sh` | 1B/4K exploratory MLA launchers. They change train length/data relative to the primary MLA run. | Supporting-only. Do not use as same-config token-scaling evidence. |
| `scripts/core_text_phases/run_125m_4k_1b_v1.sh`, `run_125m_4k_continue_v2.sh`, `run_125m_4k_1b5.sh` | 125M exploratory continuation scripts with external data/run roots. | Historical. Use only if paired with recovered sanitized artifacts. |
| `scripts/core_text_phases/run_50m_4k_tau_sweep.sh`, `run_50m_mla_v2_tau_sweep.sh`, `run_125m_mla_v2_500m.sh` | Tau/window exploration scripts. Useful for mechanism diagnosis, not current paper tables. | Cite only as exploratory unless promoted with a manifest. |
| `scripts/core_text_phases/run_phases2to5.sh`, `run_phases4to5.sh`, `run_phase6_gqa2_tau1p5.sh` | Early GQA/MLA compression sweeps. | Historical/supporting; do not use as primary evidence. |
| `scripts/core_text_phases/run_quality_454m.sh`, `run_quality_454m_eval_only.sh` | Downstream QuALITY workflow wrappers with external paths. | Supporting downstream check; keep capacity caveats. |
| `scripts/core_text_phases/run_750m_full_eval.sh` | 750M single-seed/supporting evaluation wrapper. | Supporting only; do not use as primary scale proof. |
| `scripts/core_text_phases/START_NOW.sh` | Convenience orchestration wrapper. | Do not cite. |

## Patch/Fix Scripts

Patch scripts document how the server copy evolved. They are not reproduction
entrypoints and should not be included in reviewer commands.

| Pattern/examples | Status |
| --- | --- |
| `gqa_patch.py`, `mla_patch.py`, `patch_continue_pretrain.py` | Historical patch records for server-side script evolution. |
| `fix_compile_mode.sh`, `fix_mla_assert.py` | Debug/fix records. |

Reviewer-safe handling:

- Keep them in the repository only as historical context.
- Do not use them as evidence that the current code path implements a result.
- If a result depends on a patched server version, recover the exact patched
  script or write a manifest that names the divergence.

## Known Drift Points

These are the drift patterns that triggered the Opus 4.8 audit.

- Some launch wrappers assume old run directory names that do not match the
  current `run_gqa_evq_experiment.py` run-id format.
- Some reports mention intermediate checkpoints that are not produced by the
  current committed training script.
- Some launch wrappers mention compile settings while the exact current code
  path may differ from the server-patched path.
- Some scripts encode external data locations and symlink conventions that are
  not part of the compact reviewer branch.
- Some historical 1B data-preparation variants used different tensor shapes or
  data mixtures; provenance must name the exact variant.

## Required Promotion Checklist

Before any historical script supports a paper or rebuttal claim:

- [ ] Replace external machine paths with repo-relative/configurable arguments.
- [ ] Record exact script commit or file SHA256.
- [ ] Record data artifact hash and tensor shape.
- [ ] Record checkpoint SHA256.
- [ ] Generate a sanitized manifest with
  `scripts/core_text_phases/make_artifact_manifest.py` before copying any
  external artifact metadata into public docs.
- [ ] Run `scripts/core_text_phases/audit_rope_checkpoint.py` and record
  `inv_freq` SHA256 plus schedule/tau classification.
- [ ] Record metric definition, seed list, and eval script hash.
- [ ] State whether the row is primary, supporting, exploratory, or negative.

## Branch Notes

- `backup/2026-03-06` contains archival raw artifacts that were removed from the
  compact main branch for submission hygiene.
- Those branch artifacts are useful for internal audit and recovery, but should
  be sanitized before becoming public reviewer evidence.
- Missing files on the compact main branch should be read as "not packaged
  here," not as "not run."
