# No-server fix log

## Scientific reporting and paper clarity

- Replaced the contradictory QuALITY accuracy plot with a gold-answer-NLL figure aligned to the `n=2086` table and documented the report-backed provenance boundary.
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
- RULER evaluation requires a variant label and writes provenance-rich, non-colliding result filenames.
- All LoRA launch wrappers now pass explicit method/variant labels.
- Generic LLaMA continued pretraining infers head geometry and RoPE base from model configuration, accepts explicit training length, validates packed sequence shape and frequency count, and records the resolved geometry. This prevents silent 32/64-channel and 1B/8B configuration mismatches.
- Added regression tests covering all of the above failure modes.

## Provenance and repository hygiene

- Public claim maps now distinguish report-backed QuALITY values from raw-JSON-backed artifacts and explicitly exclude the old `n=200` pilot.
- Rebuttal audit documents reference the tracked TikZ/build sources rather than a nonexistent figure generator.
- Figure build intermediates are temporary; only the source scripts and reviewer-facing PDF/PNG assets are retained.

## Verification record

- Python syntax gate: all touched/core entrypoints passed `python3 -m py_compile`.
- Shell syntax gate: all touched LoRA launchers and both figure builders passed `bash -n`.
- Unit tests: 23 non-pytest repository tests passed, including 11 new rebuttal protocol regressions. The separate `tests/test_rope_core.py` pytest gate could not run because this machine has no `pytest` module; `torch` is available.
- LaTeX: bundled Tectonic compiled `paper/main.tex` successfully to a 41-page PDF. The main body ends on page 9 and References starts on page 10. Warnings are known underfull-box/hyperref warnings; no overfull box was reported.
- PDF: `pdfinfo` reports no custom metadata and the reviewed pages render correctly. A raw `/Type3` marker scan found none; `pdffonts` is unavailable on this machine.
- Packaging: `python3 scripts/package_supplement.py` produced the curated supplement successfully.
- Authoritative review copy: 224 lines, 37,871 bytes, SHA-256 `520ff82bb04c4d552f1d36a5573ef613fc17d9bb15838b780c8f16de1d751864`.
