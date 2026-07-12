# Results

Experiment outputs are classified by evidentiary role.

Reviewer-facing portable artifacts live in `data/curated/`, not here. A file in `results/` is only usable at the tier assigned by `docs/overview/RESULT_PROVENANCE_MANIFEST.md`.

## Buckets

- `core_text/`: primary text experiments that directly support the current paper narrative
- `theory/`: theory validation and mechanism artifacts
- `supporting_cross_model/`: supporting non-core model families and adaptation runs
- `supporting_video/`: cross-modal temporal transfer results
- `legacy/`: preserved historical outputs that are no longer part of the core submission package

## Tracked Core/Historical Result Sets Present

- `core_text/50m_yarn_compare_v2/`
- `core_text/phase11b/`
- `core_text/phase14_yarn_passkey/`
- `core_text/350m_final/`
- `core_text/phase9f_750m_2k_1b/`
- `core_text/phase21b/`

Some paper rows are raw-JSON-backed through `data/curated/`; others remain report-backed or missing-artifact even if a nearby directory exists here.

## Rule

New result bundles should be placed into the narrowest bucket that matches their evidentiary role. New outputs are ignored by default; do not force-add large runs, checkpoints or caches. Avoid adding flat, unclassified directories at `results/` root.
