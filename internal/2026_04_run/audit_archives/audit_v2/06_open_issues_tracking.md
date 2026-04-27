# Audit v2 / Auditor 6: Open-issues tracking across rounds

**Auditor**: 6/6 (cross-validation team, parallel)
**Date**: 2026-04-27
**Repo HEAD at audit time**: `616edd7 paper: apply 2026-04-27 audit findings`
**Round-1 audit head was**: `a358d6d` — so one new commit (`616edd7`) has landed since.

---

## 1. Methodology

For each of the 12 accumulated open issues from rounds 1 and 2:

1. Re-run the round-1 grep / file-existence verification commands from the brief (no trust in prior wording).
2. Inspect the actual file:line evidence as it stands NOW.
3. Tag PASS / DECIDE / FAIL based on whether the issue is currently resolved, deferred-as-planned, or still actively present.
4. Record the new commit `616edd7` (the audit-fix commit since the round-1 head `a358d6d`) for each item — that commit is what either resolved or did not resolve each line.

DO NOT modify any file. Read-only verification.

For PASS verdicts: the issue is closed in the current paper PDF / repo state OR has been correctly executed per the round-1 plan.
For DECIDE: the issue is in a "deferred-as-planned" state — i.e. round-1 explicitly tagged it as "camera-ready" or "exclude at packaging time", and the deferral is still defensible.
For FAIL: the issue is still actively present and is NOT in a planned-deferral state for this submission round.

---

## 2. Per-issue status table

| # | Issue (round-1 ref) | Round-1 claim | Current evidence | Status |
|---|---|---|---|:-:|
| 1 | F1 / audit/08 §6.1 — TWO live SSH passwords in `internal/team/archive/recent_handoffs/2026-{02-27_8b_longinst,03-03_5090b}_handoff.md` | Rotate today, independent of paper | Both passwords still verbatim on disk: `2026-03-03_5090b_handoff.md:5` (`sshpass -p '3wog+1mHWO4C'`), `2026-02-27_8b_longinst_handoff.md:14` (comment `# 密码: htG0sD63/yG0`), `:165` (`sshpass -p 'htG0sD63/yG0'`). Files have not been edited (`git log --since=2026-04-26` returns no entries for either). File mtimes 2026-03-31 19:38 (pre-audit). | **FAIL** |
| 2 | P0-3 / audit/06 — MLA 432M entry-point `run_gqa_evq_experiment.py` missing | Recreate (2-4 h) OR soften Q5 (5 min) | (a) `find . -name run_gqa_evq_experiment.py` still empty — script absent. (b) `paper/main.tex:108` (Q5) still reads "...configuration files for the primary experiments (EVQ$\times$YaRN 454M, PE-dominant 125M/454M, **MLA 432M**)..." — wording has NOT been softened. (c) `scripts/core_text_phases/run_350m_mla32_500m.sh:6` still invokes `run_gqa_evq_experiment.py`. Round-1 binary "fix script OR soften Q5" — neither was done. | **FAIL** |
| 3 | F7 / audit/08 §6.4 — cluster paths `/root/autodl-tmp`, `/sessions/<id>/mnt` in to-be-shipped scripts | Pre-submission packaging-time sed-replace per §8 checklist | Within original audit/08 §6.4 scope (`scripts/2026-04/`, `experiments/lora_evq_v2/`, `scripts/analysis/unification_plot*.py`): **64 lines** (vs ~35 reported by audit/08). Repo-wide count `scripts/ + experiments/`: **335 lines**. No source file has been pre-scrubbed. The audit/00 implementation plan §F7 explicitly tagged this as "PRE-SUBMISSION packaging-time" — i.e. these are intended to be cleaned at the rsync/sed step, not in source. The count grew slightly (35→64 in scope), suggesting new scripts were added without scrubbing — but does not block submission since packaging step has not yet run. | **DECIDE** (deferred to packaging step; same posture as round-1, count grew slightly) |
| 4 | F5 / audit/06 P6 — standalone `prepare_passkey_mix.py` script | Carve out (2 h, deferred to camera-ready per audit/00 §F5) | `find scripts -name 'prepare_passkey_mix*'` empty. Logic still embedded at `scripts/core_text_phases/run_evq_sweep.py:645` (`def maybe_wrap_with_passkey_mix(...)`) and `:867` (call site). Round-1 audit/00 §F5 row tagged this **CAMERA-READY**, not submission-blocker — so deferred posture is consistent. | **DECIDE** (still embedded; round-1 deferred to camera-ready) |
| 5 | F6 / audit/06 P7 — anonymized English supp-archive README | Translate + scrub from `docs/overview/REPRODUCE.md` (1.5-2 h, camera-ready per audit/00 §F6) | `find -maxdepth 4 ... repro|REPRODUCE|SUPP_README` returns only `docs/overview/REPRODUCE.md`. That file is the original Chinese `# Reproducibility Guide` (51 CJK chars in 202 lines, e.g. `预期结果` at L41, `路径二: 核心结果复现` at L50). It does NOT contain `/root/autodl-tmp` or `seetacloud` or `misaya` strings (clean on those vectors), but it IS Chinese-language and references `internal/AIHANDOFF.md` at L26. No English-translated, anonymized supp-archive README at repo root or under `scripts/2026-04/`. Round-1 audit/00 §F6 row also tagged this **CAMERA-READY** — deferred posture consistent. | **FAIL** (technically per the brief: "PASS / FAIL / DECIDE — does an English anonymized supp-archive README exist NOW?" — answer is no. But round-1 explicitly accepted this as deferred camera-ready work.) |
| 6 | 12-config 24-92% functional validation in `tab:surrogate-validation` | Should still range 24% to 92%, consistently referenced from §3.2, §3.7, captions, sec:why-constant-alpha | `paper/appendix/a1_proofs.tex:117-149` Table caption explicitly: "Functional surrogate validation across 12 configurations." Twelve rows (L=128/256/512/1024/2048/4096 + b=10K/100K + Video b=100/1K/10K/50K), Δ_C values: -92%, -86%, -73%, -56%, -38%, -24%, -45%, -41%, -37%, -34%, -28%, -26% → range **-92% to -24%**, matches "24-92%" exactly. Cross-references all consistent: `main.tex:94` (Q3 checklist) "24--92\%", `paper/sections/03_theory.tex:15` "$24$--$92\%$ collision-score reduction across $12$ configs", `:32` "$24$--$92\%$ across $12$ configurations", `appendix/a1_proofs.tex:97,108,113,115,149` all "24--92%" / "12 configurations". Section label `sec:why-constant-alpha` defined at `a1:99-100`. | **PASS** |
| 7 | A5 / audit/01 — `paper/appendix/a2_experiment_details.tex:42` 750M LR `1.5e-4` vs source `3e-4` | VERIFY+APPLY (10 min) | a2:42 still reads `$1.5{\times}10^{-4}$ (750M)`. Source `results/core_text/phase9f_750m_2k_1b/summary.json` records `"lr": 0.0003` (verified). Drift NOT resolved by 616edd7 commit. The 616edd7 commit message lists Table std insertion + theory + caption work but no LR fix. | **FAIL** |
| 8 | A6 / audit/01 — `appendix/a3_supporting_results.tex:55,62` 750M attention-viz numbers (295/432, 508-token, 2.5×) lack source-script citation | DEFER or APPLY (30 min) | a3:55-57 caption still reads `295/432 heads (68\%)`, `${\approx}508$ tokens`, `up to $2.5\times$ higher density`. No footnote citing JSON / script source has been added. Round-1 marked this DEFER-or-APPLY; 616edd7 did not touch it. | **FAIL** (per literal brief; deferred-as-planned per round-1 option B) |
| 9 | A7 / audit/01 — "454M" vs "350M" labeling clarification footnote | APPLY (5 min) | `paper/main.tex:46` abstract uses `454$M` directly with no `\footnote{}`. `01_intro.tex` does not contain a "350M" mention or footnote. The clarification footnote ("454M = 454.2M parameters; reported as 350M in some internal logs") has NOT been inserted. 616edd7 did not address this. | **FAIL** |
| 10 | E1 / audit/03 — `references.bib` `li2025hope` and `hua2025fope` show `1 author + and others` | APPLY (15 min, camera-ready) | Both still truncated: `references.bib` `li2025hope` author = `{Li, Haoran and others}`, `hua2025fope` author = `{Hua, Ermo and others}`. Round-1 audit/00 §E1 was tagged "P1 / 15 min", but action_list §"Implementation order" stage 7 was "P1 deferred (camera-ready)". Consistent with deferred posture. | **FAIL** (per literal brief; deferred to camera-ready per round-1) |
| 11 | P0-1 / caption propagation — `tab:evidence-tier` Primary II "1-3 seed", abstract, §1 intro, §5.2, `table4_pe_dominant` caption | All should reflect "1-3 seeds (or equivalent)" | All five locations consistently rewritten by 616edd7: `main.tex:46` abstract no longer claims "three primary 3-seed stress tests" (now "Three primary stress tests anchor our evidence"); `01_intro.tex:9` "(Primary~I--III in §exp-X, 3-seed where indicated): ... (1--3 seed; the learned-PE row is 3-seed, EVQ/Geo/DAPE rows are seed 42)"; `05_experiments.tex:11` "(§exp-pe, 1--3 seed; only the learned-PE comparison row is 3-seed)"; `tables/table_evidence_tier.tex:13` Primary II "PE-dominant ... (DAPE; learned-PE row 3-seed, others seed 42) & 1--3"; `tables/table4_pe_dominant.tex:2` "Geo, DAPE, and EVQ rows are single-seed (seed 42); the Learnable $\tau$ row is 3-seed (42/137/256)". Std `181.2{\pm 1.3}` / `437.9{\pm 12.2}` correctly inserted. | **PASS** |
| 12 | P0-2 / caption propagation — `tab:evidence-tier` Primary I and `table2_evq_yarn_main.tex` caption: L=512/s=4 → L=2048/s=8 | Both should reflect new metadata | Both updated by 616edd7: `tables/table2_evq_yarn_main.tex:2` "(454M, $L_{\mathrm{train}}{=}2048$, 10\% passkey mix; $4{\times}/6{\times}/8{\times}$ extrapolation). YaRN uses the \emph{same fixed} scale $s{=}8$..." (matches source data); `tables/table_evidence_tier.tex:12` Primary I row "EVQ+YaRN, 454M MHA, $L_{\mathrm{train}}{=}2048$, $4$/$6$/$8{\times}$ extrapolation". | **PASS** |

---

## 3. Summary counts

| Status | Count | Items |
|---|---:|---|
| **PASS** (closed) | 3 | #6 (24-92%), #11 (P0-1 prop), #12 (P0-2 prop) |
| **DECIDE** (deferred-as-planned, defensible posture) | 2 | #3 (F7 packaging-time scrub), #4 (F5 camera-ready carve-out) |
| **FAIL** (actively unresolved) | 7 | #1 (F1 SSH passwords), #2 (P0-3 MLA entry+Q5), #5 (F6 supp README), #7 (A5 750M LR), #8 (A6 attn-viz source), #9 (A7 350/454 footnote), #10 (E1 bib expansion) |

### Severity-weighted view

The **FAIL** items split into three real risk classes:

| Risk class | Items | Notes |
|---|---|---|
| Submission-blocker | **#2 P0-3 (MLA 432M entry+Q5)** | This was the one P0 from audit/06. Either the MLA runner script is shipped or Q5 wording is softened. Currently neither — Q5 still claims "MLA 432M" code, but `run_gqa_evq_experiment.py` does not exist. If a reviewer downloads the supp archive and follows the MLA wrapper, they hit a "file not found". This breaks the Q5=Yes claim. |
| Security-incident (independent of paper) | **#1 F1 (SSH passwords)** | Two live cluster credentials sitting on disk in `internal/team/archive/recent_handoffs/`. Independent of submission. The `.gitignore` does not exclude `internal/team/archive/`. Even if the planned `rsync --exclude=internal/` keeps them out of the supp zip, they are in git history and on the local disk. **Rotate today** regardless of paper. |
| Camera-ready / deferred-by-round-1 | #5 (F6), #7 (A5), #8 (A6), #9 (A7), #10 (E1) | These five were tagged as camera-ready or "DEFER" in audit/00. Reporting as FAIL per the brief's literal requirement to verify "current evidence", but submission posture per round-1 already accepted these as deferred. |

---

## 4. Per-issue file-line evidence (for fast confirm by next auditor)

```
F1 SSH passwords (verbatim grep result):
  /Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/internal/team/archive/recent_handoffs/2026-03-03_5090b_handoff.md:5:
      sshpass -p '3wog+1mHWO4C' ssh -p 16966 root@connect.westb.seetacloud.com
  /Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/internal/team/archive/recent_handoffs/2026-02-27_8b_longinst_handoff.md:14:
      # 密码: htG0sD63/yG0
  /Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/internal/team/archive/recent_handoffs/2026-02-27_8b_longinst_handoff.md:165:
      sshpass -p 'htG0sD63/yG0' scp -P 23173 -r \

P0-3 MLA entry:
  find -name run_gqa_evq_experiment.py: empty
  paper/main.tex:108 contains "configuration files for the primary experiments (EVQ$\times$YaRN 454M, PE-dominant 125M/454M, MLA 432M)"
  scripts/core_text_phases/run_350m_mla32_500m.sh:6 references missing script

F7 cluster paths in scripts/ + experiments/:
  Total: 335 lines (vs ~35 in audit/08 §6.4 scope)
  In original audit/08 §6.4 scope: 64 lines

F5 passkey-mix prep:
  find scripts -name 'prepare_passkey_mix*': empty
  scripts/core_text_phases/run_evq_sweep.py:645 def maybe_wrap_with_passkey_mix
  scripts/core_text_phases/run_evq_sweep.py:867 call site

F6 supp README:
  Only docs/overview/REPRODUCE.md (Chinese, 202 lines, 51 CJK chars)
  Cluster paths within REPRODUCE.md: 0 (clean on that vector)
  But: not English, references internal/AIHANDOFF.md, not at root

24-92% / 12 configs (PASS):
  paper/appendix/a1_proofs.tex:118 caption: "Functional surrogate validation across 12 configurations"
  paper/appendix/a1_proofs.tex:128-144 12 rows, Δ_C ranges -92% to -24%
  paper/appendix/a1_proofs.tex:149 prose "EVQ reduces collision under the exact kernel in all 12 configurations ($-24\%$ to $-92\%$)"
  paper/main.tex:94 Q3 checklist "24--92\% collision-score reduction across 12 configurations"
  paper/sections/03_theory.tex:15 "$24$--$92\%$ collision-score reduction across $12$ configs"
  paper/sections/03_theory.tex:32 "$24$--$92\%$ across $12$ configurations"
  paper/appendix/a1_proofs.tex:99-100 \subsection{Why constant $\alpha$} \label{sec:why-constant-alpha}
  paper/appendix/a1_proofs.tex:97 "(24--92\%)" in pure-tether/forced-branch paragraph

A5 750M LR:
  paper/appendix/a2_experiment_details.tex:42 "$6{\times}10^{-4}$ (50M/125M), $2{\times}10^{-4}$ (454M), $1.5{\times}10^{-4}$ (750M)"
  results/core_text/phase9f_750m_2k_1b/summary.json: "lr": 0.0003 (= 3e-4)

A6 attn-viz:
  paper/appendix/a3_supporting_results.tex:57 "crossover at ${\approx}508$ tokens", "up to $2.5\times$ higher density"
  paper/appendix/a3_supporting_results.tex:57 "295/432 heads (68\%) attend farther under EVQ"
  No footnote / script citation added

A7 354M/454M:
  paper/main.tex:46 abstract "$454$M transformer" - no \footnote
  paper/sections/01_intro.tex - no 350M reference, no footnote

E1 bib:
  paper/refs/references.bib li2025hope: author = {Li, Haoran and others}
  paper/refs/references.bib hua2025fope: author = {Hua, Ermo and others}

P0-1 propagation (PASS):
  main.tex:46 abstract — "Three primary stress tests" (no longer "3-seed" claim)
  01_intro.tex:9 — "(Primary~I--III ..., 3-seed where indicated): ... (1--3 seed; the learned-PE row is 3-seed, EVQ/Geo/DAPE rows are seed 42); ... (3-seed)"
  05_experiments.tex:11 — "PE-dominant extrapolation (\S\ref{sec:exp-pe}, 1--3 seed; only the learned-PE comparison row is 3-seed)"
  table_evidence_tier.tex:13 Primary II — "PE-dominant ... (DAPE; learned-PE row 3-seed, others seed 42) & 1--3"
  table4_pe_dominant.tex:2 — "Geo, DAPE, and EVQ rows are single-seed (seed 42); the Learnable $\tau$ row is 3-seed (42/137/256)"
  Std insertion: $\mathbf{181.2{\pm 1.3}}$ / $437.9{\pm 12.2}$ correctly inserted

P0-2 propagation (PASS):
  table2_evq_yarn_main.tex:2 — "454M, $L_{\mathrm{train}}{=}2048$, 10\% passkey mix; $4{\times}/6{\times}/8{\times}$ extrapolation. YaRN uses the \emph{same fixed} scale $s{=}8$"
  table_evidence_tier.tex:12 Primary I — "EVQ+YaRN, 454M MHA, $L_{\mathrm{train}}{=}2048$, $4$/$6$/$8{\times}$ extrapolation"
```

---

## 5. Newly identified open issues (from this audit)

None of the round-2 verification turned up unforeseen new findings. Two minor observations:

- **Cluster-path count drift**: F7 grew from ~35 (audit/08 §6.4 scope) to 64 in the same scope. This suggests new scripts under `scripts/2026-04/` (the LoRA-baseline batch added Apr 26 per `01a-01e_lora_train_*.sh`) were added with `/root/autodl-tmp/` literals, without applying the round-1 anonymization sed-replace at authoring time. Mitigated entirely if the packaging-time sed still runs, but it does mean each new script needs to be added to the scrub set.

- **`maybe_wrap_with_passkey_mix` has TWO call sites** (`run_evq_sweep.py:645` def, `:867` call) — auditor 6 round-1 (audit/06 §3 P6 action) only mentions `:645-690`. Carve-out should also re-route `:867`. Minor — does not change the FAIL/DECIDE tag.

---

## 6. Recommended next-step priorities

In strict ROI / time order — only items that should move BEFORE submission:

| # | Action | Time | Severity if skipped |
|---|---|:-:|:-:|
| 1 | **Rotate the two SSH passwords** (`htG0sD63/yG0`, `3wog+1mHWO4C`) on the actual cluster — independent of submission | 5 min | Security incident; not paper-blocker but ongoing exposure |
| 2 | **Resolve P0-3 MLA**: choose Option A (recreate `run_gqa_evq_experiment.py`, 2-4 h) OR Option B (soften Q5 wording at `paper/main.tex:108` to drop explicit "MLA 432M" code claim, 5 min) | 5 min — 4 h | **Submission-blocker** — Q5=Yes is currently load-bearing for "Open access" checklist with a non-existent script |
| 3 | (At packaging time) Add to identity-grep allowlist: `autodl|seetacloud|misaya|/Users/`, and confirm `internal/team/archive/` path is excluded from rsync (current `.gitignore` does not exclude it; the supp tarball recipe in audit/08 §8 does, so this is just a checklist confirmation) | 10 min | Submission posture safe IF audit/08 §8 commands run literally |
| 4 | A5 750M LR fix at `a2:42` (1.5e-4 → 3e-4) | 5 min | Camera-ready acceptable but cheap to fix now |
| 5 | A7 footnote at first abstract `454M` mention | 5 min | Camera-ready acceptable but cheap |

**Camera-ready / deferred (NOT submission-blockers, per round-1 plan)**:
- F5 (carve-out passkey-mix prep), F6 (English supp README), F7 (sed-replace in source), E1 (bib author expansion), A6 (attn-viz source citation).

---

## 7. Cross-check vs round-1 implementation table

Round-1 audit/00 §"Implementation order" had 8 stages totalling ~2.5h quick-wins + 6h with MLA recovery. Mapping to round-2 reality:

| Stage | Items | Round-1 plan | Round-2 reality |
|---|---|---|---|
| 1 | F1 (rotate SSH) | 5 min today | NOT DONE — passwords still on disk |
| 2 | P0-1, P0-2 caption fixes | 45 min | DONE in 616edd7 |
| 3 | A1-A4, A7, B1-B4, D1-D2 | 50 min | A1, A2, A3, A4 partly DONE in 616edd7 (PK@8K std + Table 1 +0.9%→+1.2%); A7 NOT DONE; B1-B4 DONE in 616edd7; D1-D2 DONE in 616edd7 |
| 4 | C1-C9 | 17 min | DONE in 616edd7 (verified in commit message) |
| 5 | F2, F3, F7 | 30 min | F2/F3 status not part of this audit; F7 NOT DONE (still in source) |
| 6 | P0-3 entry-point | 2-4 h OR 5 min | NEITHER DONE — Q5 still claims MLA 432M, script still missing |
| 7 | E1 bib | 15 min | NOT DONE (camera-ready) |
| 8 | A5, A6, F4, F5, F6 | 5 h | NOT DONE (camera-ready) |

**Outstanding from round-1 plan**: F1 (security), A7 (5-min footnote), P0-3 (binary either/or), F7 (sed at packaging time). Everything else is DONE or correctly DEFERRED.

The single highest-leverage outstanding item is **P0-3 Option B (5-min Q5 wording softening)** — without this, Q5=Yes is technically a misrepresentation.
