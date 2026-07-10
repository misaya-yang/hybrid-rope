# Promote Local Rebuttal Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the verified local QuALITY and base-sweep raw JSONs into anonymous, reproducible reviewer assets, repair the portable evidence gate, and merge the validated feature branch into local `main`.

**Architecture:** Extend the existing evidence builder rather than copying ignored files. Each source is accepted only when its SHA256 matches a fixed expected identity; the builder emits minimal reviewer-safe JSON without machine fields. The validator, supplement packager, provenance ledger, and rebuttal handoff share the same evidence-tier contract.

**Tech Stack:** Python 3.13 standard library, `unittest`/`pytest`, Git, JSON, Markdown.

## Global Constraints

- Do not commit ignored raw results, checkpoints, caches, local paths, server names, or identity markers.
- Keep QuALITY as supporting probability-space evidence; do not claim a stable accuracy gain.
- Keep the base 10K/500K result single-seed and supporting; do not call it a tuned-base sweep or a `c_pred` control.
- Do not change paper experimental metrics or regenerate the PDF.
- Do not push directly to `main`.

---

### Task 1: Define the raw-backed evidence contract

**Files:**
- Modify: `tests/test_rebuttal_evidence_bundle.py`
- Modify: `scripts/build_rebuttal_evidence_bundle.py`
- Modify: `scripts/validate_rebuttal_evidence_bundle.py`

**Interfaces:**
- Consumes: ignored QuALITY aggregate JSON and four ignored Phase18 result JSONs.
- Produces: `build_quality_snapshot(source)`, `build_base_snapshot(sources)`, and component-selective CLI rebuilding.

- [ ] **Step 1: Write failing tests** asserting QuALITY/base are `raw-json-backed`, source hashes are preserved, raw server/checkpoint fields are omitted, local Phase11 paths resolve, and individual builder components can run without the missing MLA source.
- [ ] **Step 2: Run** `python3 -m unittest tests.test_rebuttal_evidence_bundle -v`; expected result is failure on the old report-backed/trace-only contract.
- [ ] **Step 3: Implement minimal builders** that verify fixed SHA256 values and emit only protocol, numeric rows, source identities, and claim boundaries.
- [ ] **Step 4: Run the focused test suite** and require all evidence-bundle tests to pass.

### Task 2: Regenerate portable assets and synchronize reviewer documentation

**Files:**
- Modify: `data/curated/quality_454m_full_eval.json`
- Create: `data/curated/text_base_10k_500k_pilot.json`
- Delete: `rebuttal_7/trace_only/text_base_10k_500k_pilot.json`
- Modify: `scripts/package_supplement.py`
- Modify: `docs/overview/RESULT_PROVENANCE_MANIFEST.md`
- Modify: `docs/overview/OPUS48_REBUTTAL_MASTER_BRIEF.md`
- Modify: `rebuttal_7/IGNORED_ASSET_RECONCILIATION.md`
- Modify: `rebuttal_7/FABLE5_RESPONSE_AND_FIX_LEDGER.md`
- Modify: `rebuttal_7/NO_SERVER_FIX_LOG.md`

**Interfaces:**
- Consumes: Task 1 builders and the verified local ignored sources.
- Produces: tracked anonymous assets, current hashes, and reviewer-safe scope language.

- [ ] **Step 1: Regenerate** only `phase11`, `quality`, and `base` components from local raw sources; compare Phase11 output with the existing tracked snapshot.
- [ ] **Step 2: Update docs and packaging** so base is included as raw-backed supporting evidence, QuALITY is raw-backed, and no tracked document requires the missing ignored Phase18 report.
- [ ] **Step 3: Run** the validator, focused tests, supplement build, ZIP integrity check, and leak scan.

### Task 3: Verify and merge into local main

**Files:**
- Verify all staged files; no paper files may change.

**Interfaces:**
- Consumes: validated feature-branch commit.
- Produces: local `main` fast-forwarded through the portable evidence commit(s).

- [ ] **Step 1: Run** `conda run --no-capture-output -n aidemo python -m pytest -q`; expected result is 185 tests passing after the four new evidence-builder tests are added.
- [ ] **Step 2: Run** `git diff --check`, supplement packaging, anonymous staged-content scan, and confirm `git diff --name-only -- paper` is empty for this follow-up.
- [ ] **Step 3: Commit** the evidence promotion with a terse factual message.
- [ ] **Step 4: Fast-forward local `main`** from `origin/main`, merge the feature branch locally, and rerun the full test suite on merged `main`.
- [ ] **Step 5: Do not push `main`;** report the exact local commit and remaining remote action.
