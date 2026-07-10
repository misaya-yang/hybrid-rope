# No-server fix log

## Scientific reporting and paper clarity

- Replaced the contradictory QuALITY accuracy plot with a gold-answer-NLL figure aligned to the `n=2086` table and preserved the recovered aggregate as a raw-JSON-backed sanitized snapshot.
- Removed the unsupported single-number QA-accuracy narrative and retained all four rounded accuracy deltas as a capacity-floor/non-result.
- Renamed the primary passkey metric prominently as teacher-forced NLL-gap retrieval.
- Added min-kernel motivation, advance-prediction versus calibration language, realistic-prior and unmeasured-`L_eff^J` limitations, MLA-convention boundaries, and a substantive compute/energy impact paragraph.
- Promoted the NTK anti-composition counterexample and stated that composition evidence is YaRN-specific.
- Added a worked EVQ schedule example and a conditional practitioner guide.
- Removed internal “Habitable Zone”/phase jargon and replaced the illegible tau-rank panel with a print-readable vector figure.
- Downgraded the video correction and LoRA rank threshold to their supported status.

## Protocol and implementation repairs

- LoRA training now exposes an explicit `native_geo` method whose standard endpoint schedule is distinct from the EVQ midpoint limit at tau zero.
- Positional-PPL evaluation resolves and verifies the exact saved training-time inverse-frequency tensor instead of rebuilding a potentially different schedule.
- All LoRA evaluation entrypoints now fail on missing or method-mismatched frequency artifacts instead of silently evaluating the model-default schedule.
- Frequency injection is verified across every rotary module, and result JSONs record only path-safe artifact names, schedule metadata, and SHA-256 provenance.
- LoRA training infers `head_dim` and `rope_theta` from the model configuration unless explicitly overridden, and selects the installed Transformers evaluation-strategy keyword (`eval_strategy` or legacy `evaluation_strategy`).
- Existing comparison checkpoints are validated against `experiment_meta.json` and `custom_inv_freq.pt` before a launcher is allowed to skip training; affected legacy Geo controls are rejected rather than reused.
- RULER evaluation requires a variant label and writes provenance-rich, non-colliding result filenames.
- All LoRA launch wrappers now pass explicit method/variant labels.
- Generic LLaMA continued pretraining infers head geometry and RoPE base from model configuration, accepts explicit training length, validates packed sequence shape and frequency count, and records the resolved geometry. This prevents silent 32/64-channel and 1B/8B configuration mismatches.
- Added regression tests covering all of the above failure modes.

## Provenance and repository hygiene

- Public claim maps now identify the recovered QuALITY aggregate as raw-JSON-backed and explicitly exclude the old `n=200` pilot.
- Rebuttal audit documents reference the tracked TikZ/build sources rather than a nonexistent figure generator.
- Figure build intermediates are temporary; only the source scripts and reviewer-facing PDF/PNG assets are retained.

## Verification record

- Python syntax gate: all touched/core entrypoints passed `python3 -m py_compile`.
- Shell syntax gate: all touched LoRA launchers and both figure builders passed `bash -n`.
- Full test suite: 171/171 passed under the repository `.venv`; this includes 23 focused rebuttal protocol regressions covering fail-closed frequency reuse, provenance, API compatibility, geometry, packed-data length, and stale-checkpoint rejection.
- LaTeX: bundled Tectonic compiled `paper/main.tex` successfully to a 41-page PDF. The main body ends on page 9 and References starts on page 10. Warnings are known underfull-box/hyperref warnings; no overfull box was reported.
- PDF: `pdfinfo` reports no custom metadata and the reviewed pages render correctly. A raw `/Type3` marker scan found none; `pdffonts` is unavailable on this machine.
- Packaging: `python3 scripts/package_supplement.py` produced the curated supplement successfully.
- Authoritative review copy: 224 lines, 37,871 bytes, SHA-256 `520ff82bb04c4d552f1d36a5573ef613fc17d9bb15838b780c8f16de1d751864`.

## 2026-07-10 ignored-asset reconciliation

- Promoted the ignored MLA three-seed and Phase11 archival JSONs into hash-identified, anonymous portable snapshots.
- Added the 99-run sanitized CSV plus a provenance sidecar and tested the exporter against a reconstructed 45-pilot/54-confirm source tree.
- Promoted the recovered QuALITY n=2,086 aggregate and four base=10K/500K result JSONs into source-hashed, anonymous curated snapshots.
- Kept learnable-tau and MLA channel-count evidence report-backed; no raw/full JSON is implied for those rows.
- Added `IGNORED_ASSET_RECONCILIATION.md` with all 18 Fable5 responses, experiment priority, and company-computer checkout commands.

Fresh gates after the second-checkout raw promotion:

- Full repository suite: `185 passed` under the `aidemo` conda environment with locked `pytest==9.0.2`.
- Python syntax: core entrypoints plus all new builder/validator/exporter/test files passed `python3 -m py_compile`.
- Portable bundle: `python3 scripts/validate_rebuttal_evidence_bundle.py --skip-tracked-check` passed before staging.
- Curated supplement: `/tmp/evq-cosh-raw-backed-supplement.zip` was created, passed `unzip -t`, includes the raw-backed QuALITY/base assets, and excludes the internal bundle contract test.
- Paper source was not changed in this reconciliation, so the prior PDF was not regenerated.
