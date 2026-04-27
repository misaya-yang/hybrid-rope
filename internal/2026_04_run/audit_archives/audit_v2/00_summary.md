# Round-2 Cross-Validation — Synthesis

**Generated**: 2026-04-27 (cont.)
**Source files**: `audit_v2/01_diff_verification.md` … `audit_v2/06_open_issues_tracking.md`
**Repo state**: HEAD = `616edd7` (Stage A committed); Stage B + C uncommitted in working tree.
**Diff baseline for verification**: `git diff a358d6d -- paper/` (committed + uncommitted).

---

## Per-claim verification table

| stage_id | claim | verified by | status | notes |
|---|---|---|:-:|---|
| A1 | P0-1 Table 5 caption + abstract/intro/§5/tier propagation (5 sites) | aud1 (diff), aud2 (source), aud4 (xref), aud6 (consistency) | ✓ | all 5 propagation sites consistent; Learnable τ std source-grounded |
| A2 | P0-2 Table 4 caption L=2048/s=8/4-8× + tier sync | aud1, aud2, aud6 | ✓ | corrected to actual source data |
| A3 | P1-A1 Table 4 PK@8K std (41±5/61±3/53±8/100±0) | aud2 rerun | ✓ | from per-seed retrieval JSON |
| A4 | P1-A3 Table phase11 6-cell std | aud2 rerun via verify_phase11_leverage.py | ✓ | one P2 rounding nit (-3.04 → -3.1) |
| A5 | P1-A4 Table 1 row 14 +0.9% → +1.2% | aud2 (= 25.9→26.2 = +1.158%) | ✓ | matches Table 6 |
| A6-A11 | P1-B1..B4 + D1-D2 (theory consistency + new \ref) | aud1 EXACT MATCH | ✓ | K=d_rot/2; α multi-arch; S_χ² parenthetical; β=0; tab:pe-yarn-l256; fig:attn-mechanism |
| A12 | C1-C9 caption verb softenings (9 sites) | aud1 9/9 EXACT MATCH | ✓ | abstract/Q1/Q3/a2-153/a4-57/a3-55,108/a4-36/a3-69/a3-103 |
| A13 | F2 paper/ → internal/2026_04_run/chat_scaffolding/ move | aud1 (R100 rename), aud5 (paper/ now clean) | ✓ | submission unit anonymity intact |
| B1 | §3.7 KL ½ parenthetical (absorbed into λ) | aud1 EXACT MATCH | ✓ | working tree |
| B2 | §3.7 Proposition Q_1>0 verification + range [0.008,0.032] | aud2 recompute (8-config grid → [0.0083, 0.0317]) | ✓ | working tree; range correctly bounds tested grid |
| B3 | §3.2 D ∈ W^{m,1} bounded-support qualifier | aud1 EXACT MATCH | ✓ | working tree |
| B4 | §A.1.1 KKT inactive-constraint sentence | aud1 EXACT MATCH | ✓ | working tree; auditor 4 confirms ρ_τ ≥ τ/sinh τ holds |
| B5 | Action 5 skip | no diff expected, none found | ✓ | |
| C1 | §3.2 K_app reframe (discrete-grid surrogate) | aud1 EXACT MATCH | ✓ | working tree |
| C2 | §A.5 NEW \subsection sec:why-constant-alpha | aud3 SymPy independent verification: Claims 1, 2, 4(i)(ii)(iii) all PASS; Claim 3 BC system not in paper text — no paper bug | ✓ | label at a1_proofs.tex:100; Bessel substitution genuinely yields x²y''+xy'-x²y=0 with r=√(β/γ); 24-92% range verified across 12 configs |

**19/19 claims verified.** No P0. All Stage A items in commit `616edd7`; all Stage B+C items in working tree only.

---

## Issues identified this round

### Submission-relevant — needs decision before commit/push

| # | severity | finding | fix | time |
|---|:-:|---|---|:-:|
| 1 | ⚠ confirm | Stage B silently removes §5.4 "Takeaway:" sentence (auditor 1 + 4 both flagged as unclaimed extra hunk). Net: paper body has zero conclusion text since §6 was already a comment stub. **In prior conversation you said this was intentional**; auditors flag it because it wasn't in your formal claim list. | confirm intent (likely intentional) | — |
| 2 | P1 | New label `sec:why-constant-alpha` is orphan from body — §3.2 (sections/03_theory.tex:32) cites umbrella `\ref{sec:proofs}` plus natural-language "Why constant α" pointer instead of `\ref{sec:why-constant-alpha}` directly. The new defensive content is reachable only via section-traversal, not via xref. | 1-line edit at sections/03_theory.tex:32 | 2 min |
| 3 | P1 | Symbol collision in §A.5: paper redefines `λ := ln b` inside the new Bessel subsection — collides with §3.7 transport-multiplier `λ` (calibrated unit constant). Disclosed in the paragraph but reader confusion likely. | rename to `\lambda_b` or `\Lambda` inside §A.5 only | 5 min |

### Bessel-section P1 hand-waves (optional polish, auditor 3)

| # | finding | severity |
|---|---|:-:|
| 4 | F0 — §A.5 asserts stationary-phase α_sp(φ) = πD₀/(L·b^φ) without local derivation. Standard, but uncited. | P1 |
| 5 | F2 — §A.5 asserts the bound is `O(K⁻¹ + (L ln b)⁻¹·b)` without numerically disclosing that the second term ≈ 9 at b=500K, L=4096 (i.e., the constant-α surrogate's deviation from stationary-phase is non-negligible there). A reviewer could press. | P1 |
| 6 | F3 — Bessel reduction step is declared but not shown (4 lines of chain-rule algebra). One-line gloss closes it. | P1 |

These are exposition-level. Math content verified PASS; paper survives reviewer attack on the constant-α choice.

### Persistent open from prior round (auditor 6)

| # | issue | status | next step |
|---|---|:-:|---|
| 7 | P0-3 MLA 432M entry-point: `run_gqa_evq_experiment.py` doesn't exist; Q5 still promises | **FAIL** | (A) recreate runner 2-4h, OR (B) soften Q5 wording 5 min — **submission-blocker** |
| 8 | F1 SSH passwords still in internal/team/archive/recent_handoffs/ | **FAIL** | rotate today (security; independent of submission) |
| 9 | F7 cluster paths in to-be-shipped scripts: 35 → **64 lines** (new scripts/2026-04/01a-01e_lora_train_*.sh added with `/root/autodl-tmp/`) | DECIDE | sed-replace at packaging (audit/08 §8) |
| 10 | A5 (a2:42 750M LR), A6 (a3:55,62 attn-viz), A7 (454M label footnote), E1 (li2025hope/hua2025fope authors) | FAIL but tagged camera-ready | defer |

---

## P2 polish (default reject, listed for completeness)

- Auditor 2: 2 rounding nits (Learnable τ PPL@8K=437.97 truncated to 437.9; Phase11 Geo @8K=-3.04 rounded up to -3.1).
- Auditor 5: 17 underfull \hbox/\vbox; +08:00 timezone fingerprint in PDF metadata; exiftool not installed.
- Auditor 1: 3 over-delivery hunks (Table 4 caption per-seed disclosure clause, \tabcolsep tweak, "750M single seed" qualifier in fig:attn-mechanism caption).
- Auditor 4: 33 prior dead equation labels still inactive (no regression, optional cleanup).

---

## Implementation order

| Stage | Items | Time | Risk |
|---|---|:-:|---|
| 1. Confirm Takeaway intent | issue #1 | 0 min (just confirm) | must answer before commit |
| 2. Apply 2 P1 fixes | issues #2 + #3 | 7 min | low |
| 3. Optional polish | F0/F2/F3 (3 hand-waves) | 10 min | optional |
| 4. Commit + push Stage B+C | git workflow | 5 min | clean |
| 5. Decide P0-3 | issue #7 | (A) 2-4h or (B) 5 min | submission-critical |
| 6. Pre-submission anonymity hygiene | F1 + F7 + tarball excludes per audit/08 §8 | 30 min | independent |

**Total quick wins (1-4)**: ~12 min, 0 GPU. Submission-critical decision (5) separate.

---

## Files referenced

- Detailed reports: `audit_v2/01-06_*.md`
- Bessel verification script: `audit_v2/scripts/verify_bessel_substitution.py` (~3s runtime, sympy + scipy)
- Prior-round audit history: `audit/00_action_list.md`, `audit/01-08_*.md`
- Diff baseline: commit `a358d6d`
