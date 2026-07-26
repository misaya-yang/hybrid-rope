# AGENTS.md — EVQ-Cosh (NeurIPS 2026)

Paper-and-reproducibility repo for **Submission 11628 / EVQ-Cosh**.
Not a general software project. Priorities: scientific correctness, reviewer
trust, anonymous hygiene, reproducible evidence paths.

This file is the **only** project-level agent instruction. Do not add a second
root `Agent.md`. Long operational detail lives in the index links below—do not
duplicate it here.

---

## 0. Core principles (non-negotiable)

### P1 — Rebuttal answers only what reviewers asked

- Respond **only** to retained reviewer/AC concerns in
  `rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`.
- **No over-extension:** no new research program, no EVQ-v2, no extra benchmarks,
  no side narratives that do not map to a stable concern ID.
- If evidence is not needed to answer a named concern, do not run it, do not
  lead with it, and do not pad the response with it.
- Simulated reviews and internal audits are not new official concerns.

### P2 — Prepare fully offline before any server / GPU job

- **Zero idle GPU:** code, data, hashes, configs, entry command, output paths,
  env, stop condition, and free-space check must be ready **before** the job
  starts. Validate off-GPU first (READY / preflight for paid GPUs).
- Do not use the machine to debug missing scripts, missing data, or undefined
  protocols. If anything is missing → stop, prepare offline, then launch.
- Cost-first: minimize GPU-seconds to answer the scientific question; no
  symmetry reruns or “look busy” probes.

---

## 1. Hard rules

### Paper is read-only

Do **not** modify, create, delete, move, format, compile, or regenerate anything
under `paper/` (sources, tables, figures, bib, build artifacts, `main.pdf`).
Reading/auditing is allowed.

- `paper/` is the restored submitted-paper tree from `994f618` (identical to
  submission baseline `cb7d83e`). Repository cleanup, branch reorganization,
  and rebuttal work must preserve the directory byte-for-byte.
- Never move, rename, replace, flatten, or partially copy `paper/`; treat the
  whole directory as one immutable root.
- Need a paper change → **stop and ask**.
- Override only if the user **explicitly names** the paper file/scope **and**
  overrides this ban.
- Review / rebuttal / playbook / narrative / evidence work does **not** authorize
  paper edits.

**Submitted PDF and sources:** `paper/main.pdf` and the rest of `paper/`.

### Do not invent science

- No fabricated numbers, experiments, or metric edits.
- No silent upgrade of single-seed / supporting / diagnostic rows to primary or SOTA.
- Negative results and protocol limits stay visible when they bound a claim.
- Filenames and labels ≠ method identity; use tensors, hashes, raw/curated owners.

### Do not touch without explicit ask

`internal/`, `results/`, `audit_v3/`, `audit_v4/`, `.codex/`, `.claude/`.
Do not commit secrets, private paths, checkpoints, caches, or bytecode.

---

## 2. Core paper identity

> RoPE is not only a positional operator or a range-scaling target; it is also a
> **finite spectral budget**. EVQ-Cosh is a **closed-form, zero-learned-parameter**
> **training-time frequency-grid allocation**—a third PE design axis, complementary
> to operator design and inference-time range scaling.

**Do not** reframe as: universal long-context SOTA; YaRN/LongRoPE/FMRoPE/DAPE
replacement; or learned-PE replacement.

**Primary tiers (submitted):**
I EVQ×YaRN matched-scale · II PE-dominant (Geo/DAPE/EVQ seed-42 diagnostic) ·
III MLA scarce-channel. Supporting stays supporting (DiT, LoRA-8B, progressive,
750M, QuALITY, …).

**Hard claim guardrails**

| Topic | Rule |
| --- | --- |
| YaRN (submitted) | Repo fixed-index smooth-ramp scaler unless artifact says otherwise |
| DAPE (submitted row) | Layer-shared learnable inv-freq; not pure shape attribution |
| Cosh uniqueness | Only for stated convex surrogate \(C_{\mathrm{app}}\) |
| \(\tau=d/\sqrt{L}\) | Operating default / basin prior—not global optimum |
| Passkey (PK) | Teacher-forced **NLL-gap** unless marked AR exact |
| FMRoPE direct control | Fixed-range: allocation identifiable; retargeted: range methods can win; do not claim replacement or universal dominance |
| Novelty bar | Related theme ≠ same method (same standard as AdamW / YaRN vs NTK) |

---

## 3. Rebuttal routing (mandatory order)

Workspace: `rebuttal/rebuttal_0723/`.

| Step | File | Role |
| ---: | --- | --- |
| 1 | `00_REVIEWER_SCORES_AND_AC_METAREVIEW.md` | Official panel text + stable IDs |
| 2 | `01_REBUTTAL_PLAYBOOK.md` | Reviewer-facing evidence, status, safe wording |
| 3 | `theory_results/EVQ_COSH_REBUTTAL_PRINCIPLES.md` | Method / theory / stop rules |
| 4 | `README.md` | Package layout + experiment index |
| 5 | Standalone report + plan under `theory_results/` / `experiments/` | Protocol, numbers, provenance |

**Panel (stable IDs in `00_…`):**
Dz6s (`RDz6s.*`) · zWsa (`RzWsa.*`) · 27bE (`R27bE.*`) · AC XLtL (`AC.*`).
Do not cite pre-rebuttal simulated Reviewer 1/2/3 as OpenReview sources.
AC/`Dz6s`/`zWsa` text may be author-pasted exports; only use wording in `00_…`.

**Every experiment proposal must open with:**
(1) concern ID · (2) existing evidence · (3) smallest missing · (4) smallest plan ·
(5) stop condition.
If (1) is empty → **do not run** (P1).
If (4) is not fully prepared offline → **do not launch GPU** (P2).

**Evidence sync:** anything cited in a response needs
(1) standalone owner under `theory_results/` **and**
(2) matching playbook row. Both must agree.

**Layout:** root routing files are `README` + `00_` + `01_` + `02_`; results in `theory_results/`;
code in `experiments/`; tests stay in repo `tests/`.

---

## 4. Work hygiene (short)

- Facts = code, configs, logs, raw/curated artifacts, hashes.
- Before in-place frequency patches: clone reference tensors; hash hybrids.
- Separate: reviewer request · repo fact · proposal · completed evidence.
- Preserve unrelated worktree changes.
- Verify at claim strength (transcript/tooling for agent claims).
- Completion report: passed / failed / skipped / unverified.
- User language: usually Chinese; lead with concrete status.
- Prefer minimal edits; `main_0726` is the active rebuttal source of truth.

---

## 5. Index — details live elsewhere

| Need | Go here |
| --- | --- |
| Claim map / provenance | `docs/overview/PAPER_CLAIMS_MAP.md`, `docs/overview/RESULT_PROVENANCE_MANIFEST.md` |
| Reproduce / data | `docs/overview/REPRODUCE.md`, `DATA_PREPARATION.md` |
| GPU / Blackwell / cost-first runs | `docs/overview/RTX5090_BLACKWELL_PROFILE.md` |
| Positive/negative evidence ledgers | `rebuttal/rebuttal_0723/theory_results/REVIEWER_USABLE_EVIDENCE_LEDGER_*.md`, `INTERNAL_NEGATIVE_AND_DIAGNOSTIC_LEDGER_*.md` |
| Numeric rebuttal entry | `rebuttal/rebuttal_0723/theory_results/EXPERIMENT_REPORT_20260724.md` |
| Frequency identity | `theory_results/FREQUENCY_DEFINITION_MANIFEST.json`, `experiments/geo_rope_contract.py` |
| Canonical schedule API | `scripts/lib/rope/schedules.py` |
| Core sweep / passkey helpers | `scripts/core_text_phases/run_evq_sweep.py`, `scripts/supporting_eval/eval_passkey_scratch.py` |
| Reviewer supplement zip | `scripts/package_supplement.py` (never zip repo root) |
| Pre-rebuttal (history only) | `rebuttal/pre_rebuttal/` — not the action queue |
| Repo map / handoff | `REPO_MAP.md`, `README.md` |
| LaTeX build (only if paper ban overridden) | `paper/compile_aidemo.sh`; preferred tectonic from `paper/` |
| Smoke tests | `tests/test_rope_core.py`; py_compile schedules + core entrypoints |

### GPU (one paragraph; enforces P2)

Experiment-only. **Prepare everything offline first—no idle GPU.** Paid GPU ≠
5090: READY receipt + offline preflight first. RTX 5090: scoped diagnostic
exception (named concern, bound command, stop, free space, verify)—see
`RTX5090_BLACKWELL_PROFILE.md`. No silent math-attention fallback. Cost-first:
minimal GPU-seconds to answer the concern; runtime micro-details need not match
across arms.

### Git (one paragraph)

Work on `main_0726`; sync `origin/main_0726` before serious edits. `main` and
`backup/main-restored-paper-20260726` are preserved pre-cleanup baselines, not
the active rebuttal queue. Push only on explicit request. Never reset/checkout
away user work. Temporary `codex/*` is not an evidence baseline until merged.

---

## 6. Default agent behavior

1. Read this file → apply **P1/P2** → route via §3 before any rebuttal/GPU work.
2. Prefer playbook-selected evidence that answers a **named** concern; never hide
   a claim-limiting bound; never over-extend the rebuttal story.
3. Do not invent score changes or “reviewer missed X” attacks.
4. If blocked (paper ban, missing provenance, unclear concern, incomplete
   offline prep), **ask** once—do not invent detours that burn GPU or rewrite
   the paper.
