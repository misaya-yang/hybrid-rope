# Audit 08 — Anonymity (NeurIPS 2026 Double-Blind)

Auditor: 8/8 (parallel) Date: 2026-04-27

Severity legend
- **P0** — paper-visible leak in the *submitted PDF*, .tex, or .bib. Breaks double-blind on submission.
- **P1** — code-archive leak. Breaks double-blind only when the supplementary code zip ships.
- **P2** — git-history leak. Only breaks double-blind if `.git` is included in the tarball.

TL;DR — **The submission PDF and all paper-source files (`paper/`) are clean and submittable as-is.** The risk surface is entirely in the auxiliary repo content (`internal/`, `experiments/lora_evq_v2/`, `scripts/2026-04/`, `docs/`, the `tong-jincheng-skill/` sibling tree, and the git history). Every issue below is contained to *non-paper* artifacts; the user's existing `.gitignore` and the internal anonymisation guide at `internal/2026_04_run/docs/24_anonymous_code_release_0424.md` already plan around most of them, but the actual tarball-build has not yet been performed — until it is, **do not run `tar czf code.tar.gz .` from the repo root**.

---

## 1. Methodology

Greps run (all read-only):

```
# Paper-visible (1.a–h)
grep -rE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'  paper/                  # email
grep -rE '\d{4}-\d{4}-\d{4}-\d{4}'                          paper/                  # ORCID
grep -rEi '(Tsinghua|Stanford|MIT|Anthropic|OpenAI|...)'     paper/                  # institutions
grep -rE 'github\.com|gitlab\.com|huggingface\.co|wandb\.ai' paper/                  # repo / wandb URLs
grep -rE '\\(thanks|affil|email|acksection)'                 paper/                  # acknowledgments
grep -rEi 'we (previously|earlier) showed|our (prior|earlier) work'  paper/         # self-reference
grep -rni 'misaya|yanghej|hejaz|tong-jincheng|童锦程'        paper/                  # author leak

# Code archive (2.a–g)
grep -rE '/Users/[^/]+/|/home/[^/]+/|/mnt/[^/]+/|/shared/[^/]+/'   .  --inc=*.py *.sh *.md *.yaml *.json
grep -rE 'wandb\.ai/|WANDB_ENTITY|wandb\.init'                     .  --inc=*.py *.sh *.md
grep -rE 'connect\.bjb1\.seetacloud|seetacloud|sshpass'            .  --inc=*.py *.sh *.md
grep -rE 'AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|ghp_[A-Za-z0-9]{36}'  .

# Git history (3)
git log --format='%an <%ae>' | sort -u

# PDF metadata (4)
pdfinfo paper/main.pdf
strings paper/main.pdf | grep -iE 'Author|Username|/Users'

# EXIF (5)   exiftool not installed → fallback
strings paper/figs/*.pdf paper/figs/*.png | grep -iE 'Author|Username|Creator|Producer|/Users|misaya'

# Bib self-citation (6)
grep -nEoH 'author = \{[^}]*\}' paper/refs/references.bib | <count first authors>

# Cluster-path scan in to-be-packaged dirs (7)
grep -rE '/root/|/mnt/|/home/' scripts/ experiments/
```

All commands run from `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope`. Reading-only — no files modified.

---

## 2. Paper-level findings (P0)

> Scope: `paper/main.tex`, `paper/sections/*`, `paper/appendix/*`, `paper/tables/*`, `paper/refs/references.bib`, and the rendered `paper/main.pdf`. Files inside `paper/` that are *not* part of the LaTeX build (`paper/REVIEW_PROMPT.md`, `paper/REBUTTAL_PLAYBOOK.md`, `paper/COWORK_FINAL_PROMPT.md`, `paper/EDIT_CHANGELOG.md`, `paper/CITATION_AUDIT_REPORT.md`, `paper/_build_to_delete/`) are out of submission scope and treated as P1 below.

**Result: NONE. The submission PDF is fully anonymous.** Per-vector breakdown:

| Vector | Result | Evidence |
|---|---|---|
| 1.a Real names not in citations | Clean | All person names appear inside `\citet{}` / `\citealp{}` macros only. Spot-checked `paper/sections/01_intro.tex:5,9`, `02_related.tex:6,9,12,17`, `03_theory.tex` (no names), appendix files. |
| 1.b Real email addresses | Clean | Zero matches in `paper/main.tex`, `sections/`, `appendix/`, `tables/`, `refs/references.bib`. (Matches in `paper/neurips_2025.sty:16` / `neurips_2026.sty:16` are upstream style-file boilerplate `garnett@wustl.edu`, not authored.) |
| 1.c ORCID IDs (`\d{4}-\d{4}-\d{4}-\d{4}`) | Clean | Zero matches in any paper file. |
| 1.d Institutions outside citations | Clean | Only "OpenAI" appears at `refs/references.bib:141` as the `journal = {OpenAI Technical Report}` field of `radford2019gpt2` — that is the actual venue of GPT-2, not author affiliation. All other institution-name hits are inside `\citet{}` arguments or bib entries (legitimate). |
| 1.e GitHub / external repo URLs | Clean | Zero `github.com / gitlab.com / huggingface.co / wandb.ai / drive.google / dropbox` hits in any paper file. |
| 1.f Slack / Discord / wandb workspace | Clean | Zero hits. |
| 1.g Self-reference ("our prior work") | Clean | Zero hits for `we (previously\|earlier) (showed\|...)`, `our (prior\|earlier\|previous) work`, `as in our`. |
| 1.h Acknowledgments section | Empty | `paper/main.tex` contains no `\section{Acknowledg...}` and no `\acksection`; only Broader Impact at line 67 and the NeurIPS Checklist (lines 73–180), which are required, audience-neutral content. |

**Author block** — `paper/main.tex:39: \author{Anonymous Authors}`. Rendered PDF first page (`pdftotext paper/main.pdf | head`) shows `Anonymous Author(s)` / `Affiliation` / `Address` / `email` placeholders — correct anonymous mode of `neurips_2026.sty`.

**Title** — `paper/main.tex:37: \title{EVQ-Cosh: Variational Frequency Allocation for Rotary Position Embedding}`. No author signature embedded.

**Bib** — `paper/refs/references.bib` has 44 entries, all standard third-party citations of public works.

**Conclusion: paper P0 = 0 leaks. Submittable as-is for double-blind review.**

---

## 3. PDF metadata (P0 — clean)

`pdfinfo paper/main.pdf` returns:

```
Title:           
Subject:         
Keywords:        
Author:          
Creator:         LaTeX with hyperref
Producer:        pdfTeX-1.40.27
CreationDate:    Mon Apr 27 11:30:57 2026 CST
ModDate:         Mon Apr 27 11:30:57 2026 CST
```

- Title / Subject / Keywords / Author: **all blank**. Pass.
- Creator / Producer: standard pdfTeX. Pass.
- `strings paper/main.pdf | grep -iE '/Users/|misaya|hejaz'` → empty. No filesystem path embedded by pdfTeX. Pass.

Soft fingerprint (NOT a P0, mention for completeness)

> `strings paper/main.pdf` reveals `/ModDate (D:20260427113057+08'00')` — the PDF embeds the user's local timezone offset `+08:00`. That is consistent with mainland China, Singapore, Western Australia, Malaysia, etc.; not a strong identifier. NeurIPS reviewers do not typically inspect `/ModDate` strings, but if you want to scrub even this, recompile with `SOURCE_DATE_EPOCH=0 pdflatex main` or set `\pdfinfo{/ModDate (D:20260101000000Z)}`.

---

## 4. EXIF / image metadata (P1, none — clean)

`exiftool` is **not installed** on this Mac, fell back to `strings | grep`. Audited every file in `paper/figs/` and `paper/figs/unused/`:

- All 11 PDF figures + 9 unused PDF figures: Producer = `Matplotlib pdf backend v3.10.8` *or* `GPL Ghostscript 9.55.0` (after a Ghostscript pass), CreatorTool = `Matplotlib v3.10.8, https://matplotlib.org`. **No `Author`, `Username`, `/Users/...`, or filesystem path strings.** Matplotlib does not embed `~/.config` paths into PDFs.
- All `.png` figures: zero `Author` / `Username` / `/Users/` strings.
- `paper/figs/attention_stats.npz`: zero identifying strings.

Conclusion: figures are clean. **However**, since the build was on macOS (`misaya.yanghejazfs.com.au` user dir), if a future regenerated figure ever embeds tempfile paths (a known matplotlib edge case when `savefig` writes through a TmpDir), re-audit before submission. To eliminate this risk preemptively, recommend running before tarball:

```bash
brew install exiftool
exiftool -overwrite_original -all= paper/figs/*.pdf paper/figs/*.png
```

---

## 5. Bib self-reference scan (P0, clean)

First-author distribution across `paper/refs/references.bib` (44 entries):

| First author | Entries | Comment |
|---|---|---|
| Li | 3 | li2024fire (FIRE, Google), li2025hope (HoPE, VLM), li2026copeclipped (CoPE clipped). Three distinct research groups. Generic Chinese surname. |
| Yang, Zheng, DeepSeek-AI, Chen | 2 each | All distinct papers / orgs. |
| All other surnames | 1 each | Fine. |

**No author appears as first author ≥3 times in a way suggestive of self-citation.** "Li" coincidence is generic-surname collision — verified by inspecting the citation contexts, all three are baselines we cite for related-work positioning, not promotion of a single research line. Pass.

---

## 6. Code-archive findings (P1)

> Triggers on tarball release. The user has NOT yet built the supplementary zip. The following are issues to clean BEFORE running `rsync … /tmp/evq-cosh-supp/` from `internal/2026_04_run/docs/24_anonymous_code_release_0424.md`. Listed in priority order.

### 6.1 SSH credentials in repo (P1, also a security incident)

**TWO live cluster passwords are in the repo:**

| File | Line | Leak |
|---|---|---|
| `internal/team/archive/recent_handoffs/2026-02-27_8b_longinst_handoff.md` | 165 | `sshpass -p 'htG0sD63/yG0' scp -P 23173 -r ...` |
| `internal/team/archive/recent_handoffs/2026-03-03_5090b_handoff.md` | 5 | `sshpass -p '3wog+1mHWO4C' ssh -p 16966 root@connect.westb.seetacloud.com` |
| `internal/team/archive/recent_handoffs/2026-02-27_8b_longinst_handoff.md` | (separate) | comment "# 密码: htG0sD63/yG0" |

Beyond anonymity, **these are credentials and should be rotated regardless** — git history retains them. The two `.gitignore` excludes `internal/tools/` and `internal/reviews/` but **does not** exclude `internal/team/archive/`. If the tarball follows `24_anonymous_code_release_0424.md` (which excludes the entire `internal/`), this is contained. If the tarball is built any other way, they ship.

Recommendation: rotate both passwords today (independent of paper submission) and document the rotation outside the repo.

### 6.2 Cluster SSH host strings (P1)

```
internal/2026_03_run/docs/11_AIHANDOFF_0323.md:           ssh -p 23173 root@connect.bjb1.seetacloud.com
internal/2026_03_run/docs/10_TD_FIX_REPORT_0322.md:        ssh -p 30402 root@connect.westc.seetacloud.com
internal/team/archive/legacy_handoffs/2026-02-23/plan.md:  connect.bjb1.seetacloud.com:52592
internal/team/archive/recent_handoffs/2026-02-27_8b_longinst_handoff.md:165  root@connect.bjb1.seetacloud.com:/root/autodl-tmp/dfrope/hybrid-rope/...
internal/team/archive/recent_handoffs/2026-02-27_8b_longinst_handoff.md     ssh -p 23173 root@connect.bjb1.seetacloud.com
docs/exp/2026-02-24_128tok_baseline_report.md:                              ssh -p 12205 root@connect.bjb1.seetacloud.com
```

8 occurrences total. AutoDL/SeetaCloud is a Chinese cloud-GPU provider; identifying the user's cloud-provider account is not directly de-anonymizing but narrows the geographic / institutional surface considerably. Most are inside `internal/` (excluded by the planned rsync), but **`docs/exp/2026-02-24_128tok_baseline_report.md` is in `docs/` which the planning doc only excludes line 84** — verify when building the archive.

### 6.3 `/Users/<user>/` absolute paths (P1)

22 path-leak lines, two distinct usernames:

| Username | Files | Count |
|---|---|---|
| `misaya.yanghejazfs.com.au` (this Mac) | `internal/paper_plans/figs_README.md` (4), `internal/paper_plans/PAPER_PLAN_V9.md` (1), `internal/team/archive/legacy_handoffs/2026-02-23/v7_data_coverage_audit_2026-02-24.md` (1), `internal/team/archive/legacy_handoffs/2026-02-25/01_IMPLEMENTED_SCOPE.md` (1), `internal/team/archive/recent_handoffs/2026-02-27_8b_longinst_handoff.md` (1), `internal/brief/README.md` (1), `docs/exp/2026-03-10_phase17b_1024_continue_vs_512_baseline.md` (1) | 10 |
| `yang` (separate workstation, embedded in `/Users/yang/projects/hybrid-rope/...`) | `paper/COWORK_FINAL_PROMPT.md:7`, `paper/EDIT_CHANGELOG.md:7`, `docs/exp/2026-03-14_video_temporal_blackwell_plan.md` (10 occurrences) | 12 |

Important: **the two `paper/COWORK_FINAL_PROMPT.md` and `paper/EDIT_CHANGELOG.md` files are inside the `paper/` directory.** They are NOT compiled into the PDF (LaTeX never `\input`s them), so they do not break the *submission*. But if the user submits `paper/` as the supplementary "paper source" tarball that some venues require, they would ship. **Recommendation: move both files out of `paper/` to `internal/`** or delete before any zip of `paper/`.

```
paper/COWORK_FINAL_PROMPT.md:7: 你是NeurIPS论文修改专家。对 `/Users/yang/projects/hybrid-rope/paper/` 下的论文进行最终一轮精确修复。
paper/EDIT_CHANGELOG.md:7:     你是一位NeurIPS论文修改专家。对 `/Users/yang/projects/hybrid-rope/paper/` 下的论文进行一轮系统性完善。
```

### 6.4 Cluster paths embedded in scripts that WILL be packaged (P1)

These are inside `scripts/2026-04/` (named in the handover doc as a folder to ship) and `experiments/lora_evq_v2/` (also referenced from the README's repository structure as user-facing). They will go out unless explicitly cleaned.

| Pattern | Files | Counts |
|---|---|---|
| `/root/autodl-tmp/...` | `experiments/lora_evq_v2/run.sh`, `run_pe_comparison.sh`, `run_all_eval.sh`, `run_stage2.sh`, `server_setup.sh`, `download_model_data.py`, `eval_*.py`, `train_*.py`, `test_*.py`, `debug_generation.py`, `diagnose_inv_freq.py`, `eval_ruler_logprob.py`, plus `scripts/2026-04/00_preflight.sh`, `01a..01e_lora_train_*.sh`, `internal/2026_04_run/docs/12_LoRA_PE_Baseline_Comparison_实验计划.md` | ~30 lines |
| `/sessions/<harness-id>/mnt/hybrid-rope/...` (Anthropic Claude harness session paths) | `scripts/analysis/unification_plot.py`, `unification_plot_v2.py`, `unification_plot_v3.py`, `unification_plot_final.py`, `internal/draft_scripts/PHASE21_README.md` | 5 lines |

The `/sessions/eloquent-exciting-babbage/` and `/sessions/vibrant-practical-hawking/` paths embedded in the analysis plotters are particularly conspicuous: they are Claude Code container session IDs, which de-anonymize that the analysis was generated by an LLM-assisted workflow on a remote sandbox. Replace with `paper/figs` (relative).

`/root/autodl-tmp/...` is generic enough that it does not directly identify the user, but combined with the SSH host `connect.bjb1.seetacloud.com` and the AutoDL provider footprint, it narrows substantially.

### 6.5 Wandb (P1, low risk)

```
scripts/train.py:464  ap.add_argument("--wandb_project", type=str, default="hybrid-rope-neurips")
scripts/train.py:465  ap.add_argument("--wandb_entity",  type=str, default="")
scripts/train.py:494  wandb.init(project=args.wandb_project, entity=(args.wandb_entity or None), …)
```

`entity=""` (default empty) — no wandb workspace name leaks via code. Project name `hybrid-rope-neurips` is paper-internal but reveals NeurIPS targeting (still anonymous-safe). The internal release doc plans `WANDB_MODE = disabled` for the supp zip. Pass once that is enforced.

### 6.6 GitHub URLs in repo (P1, all third-party)

```
internal/team/plans/phase23_video_temporal_blackwell.md   https://github.com/Wiselnn570/VideoRoPE
docs/exp/2026-03-01_video_temporal_transfer.md            https://github.com/Wiselnn570/VideoRoPE
docs/exp/2026-03-14_video_temporal_blackwell_plan.md      https://github.com/Wiselnn570/VideoRoPE
```

`Wiselnn570` is the upstream VideoRoPE author, not us — third-party citation in markdown only. Acceptable, no action needed.

### 6.7 The `tong-jincheng-skill/` sibling tree (P1)

`tong-jincheng-skill/` is at the repo root and contains a personal Claude Code skill packaging the user's collection of materials around 童锦程 (Tong Jincheng). Already gitignored (`.gitignore` line 90: `tong-jincheng-skill/`), so it does not enter the git history, but it **does** exist on disk and would be captured by `tar czf code.tar.gz .`. **Confirm exclusion explicitly in the rsync filter** when building the supp zip.

### 6.8 `.claude/`, `.codex/` directories (P1)

Both gitignored. Same caveat as 6.7 — verify explicit exclusion in any tarball build.

### 6.9 `audit/` directory (P1, generated this run)

The auditor reports being written this session contain user-identifying paths in their reports. They are gitignored only if matched (currently `.gitignore` does not explicitly exclude `audit/`). Check `audit/` is excluded from the supp zip — it is not part of the deliverable.

### 6.10 The handover doc explicitly flagged in the brief (P1, EXCLUDE EXPLICITLY)

**`scripts/2026-04/PAPER_HANDOVER_2026-04-27.md`** is the file the audit brief specifically called out. Re-checking its content:

- It does **NOT** contain SSH host strings or passwords. (`grep -nE 'connect.bjb1|seetacloud|sshpass|misaya|/Users/|/root/'` → 0 matches.)
- It **DOES** contain paper-strategic information: rebuttal-stage attack vectors, internal commit hash references (`2b33a59`), unfinished GPU experiments, `~15h GPU budget`, and review-score predictions ("7.4–7.7 / 10"). This is reviewer-game-theory disclosure that should not ship to the public archive even though it's anonymity-clean.

**Recommendation: exclude `scripts/2026-04/PAPER_HANDOVER_2026-04-27.md` from the supp tarball explicitly.** The README in the same directory (`scripts/2026-04/README.md`) is a runbook and is fine to ship after a one-line strip of the `cd /root/autodl-tmp/hybrid-rope` example (line 24).

### 6.11 Internal strategic content (P1, exclude wholesale)

The entire `internal/` tree (and especially `internal/paper_plans/`, `internal/2026_03_run/`, `internal/2026_04_run/`, `internal/team/archive/`, `internal/reviews/`) contains attack-vector planning, peer-review-game-theory notes, rebuttal playbooks, and audit reports. Do not ship. The author's own `24_anonymous_code_release_0424.md` already excludes `internal/` — confirm.

Same for `paper/REBUTTAL_PLAYBOOK.md`, `paper/REVIEW_PROMPT.md`, `paper/CITATION_AUDIT_REPORT.md`, `paper/EDIT_CHANGELOG.md`, `paper/COWORK_FINAL_PROMPT.md`, `paper/_build_to_delete/`. These are inside `paper/` but are not compiled into the PDF — they should be excluded from any "paper-source" archive.

---

## 7. Git history (P2)

`git log --format='%an <%ae>' | sort -u`:

```
Misaya       <nk82cj44cb@privaterelay.appleid.com>     ← Apple privaterelay
Misaya Yang  <misaya.yang@hejazfs.com.au>              ← real corporate email
misaya       <yang@192.168.0.155>                       ← LAN IP
misaya       <yang@misayadeMacBook-Air.local>           ← local hostname
```

Per-author commit counts: 128 + 53 + 23 = 204 commits. **Three of the four committer identities are non-anonymous** (the `@hejazfs.com.au` is a hard reveal of corporate domain, and `misayadeMacBook-Air` reveals the user's first name).

Severity:
- **If the supplementary code archive is built via `git archive` or `git bundle`, this is P0** (history travels with `.git`).
- **If built via `rsync … --exclude='.git'`** (per `24_anonymous_code_release_0424.md` line 70), this stays P2 — git history doesn't ship.

Recommendation: stick with the rsync-based pack the user already planned. If for any reason a `.git`-bearing archive is needed (e.g., reviewers asking for full provenance), run a `git filter-repo` author rewrite first:

```bash
# After installing git-filter-repo:
git filter-repo \
    --commit-callback '
commit.author_name  = b"Anonymous"
commit.author_email = b"anonymous@example.com"
commit.committer_name  = b"Anonymous"
commit.committer_email = b"anonymous@example.com"
'
```

(Note: filter-repo will also rewrite the `Co-Authored-By:` trailers — verify that nothing else in commit messages reveals identity. Quick check: `git log --all --format='%B' | grep -iE 'misaya|yanghej|hejaz|tong'` → should be empty. I checked the public commit messages: clean. Only the `%an / %ae` slots are leaks.)

---

## 8. Pre-submission anonymisation checklist (concrete commands)

The user's existing guide at `internal/2026_04_run/docs/24_anonymous_code_release_0424.md` is mostly correct. The extras / corrections from this audit:

```bash
# 0. SECURITY FIRST (do this today, independent of submission):
#    Rotate both leaked SSH passwords noted in §6.1.

# 1. From the repo root.
cd /Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope

# 2. Build the supp zip *outside* the repo (no .git, no /Users/<u>/ leak path).
mkdir -p /tmp/evq-cosh-supp

rsync -av \
  --exclude='.git/' \
  --exclude='.claude/' \
  --exclude='.codex/' \
  --exclude='internal/' \
  --exclude='paper/REBUTTAL_PLAYBOOK.md' \
  --exclude='paper/REVIEW_PROMPT.md' \
  --exclude='paper/CITATION_AUDIT_REPORT.md' \
  --exclude='paper/EDIT_CHANGELOG.md' \
  --exclude='paper/COWORK_FINAL_PROMPT.md' \
  --exclude='paper/_build_to_delete/' \
  --exclude='audit/' \
  --exclude='tong-jincheng-skill/' \
  --exclude='results/' \
  --exclude='scripts/2026-04/PAPER_HANDOVER_2026-04-27.md' \
  --exclude='*.pyc' --exclude='__pycache__/' \
  --exclude='.DS_Store' --exclude='**/.DS_Store' \
  --exclude='*.safetensors' --exclude='*.pt' --exclude='*.pth' \
  --exclude='*.bin' --exclude='*.ckpt' --exclude='*.npz' \
  ./ /tmp/evq-cosh-supp/

# 3. Generic-replace cluster paths with relative ones:
cd /tmp/evq-cosh-supp
find . -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.yaml' -o -name '*.yml' \) \
  -exec sed -i.bak \
    -e 's|/root/autodl-tmp|./data|g' \
    -e 's|/sessions/[a-z-]*/mnt/hybrid-rope|.|g' \
    -e 's|/Users/[a-zA-Z0-9._-]*/projects/hybrid-rope|.|g' \
    -e 's|/Users/misaya[^"`'\'' ]*|./|g' \
    {} \;
find . -name '*.bak' -delete

# 4. Strip wandb (already disabled by default in scripts/train.py since entity=""):
grep -rEln 'wandb\.init|WANDB_PROJECT|WANDB_ENTITY' . | xargs -I{} echo "MANUAL CHECK: {}"

# 5. EXIF-strip figure metadata (defensive, even though current scan is clean):
brew install exiftool   # if not present
exiftool -overwrite_original -all= paper/figs/*.pdf paper/figs/*.png

# 6. PDF metadata: re-emit with reproducible timestamp + null /ModDate timezone:
cd paper && SOURCE_DATE_EPOCH=0 pdflatex -output-directory=. main && cd ..

# 7. Final identity sweep — must return ZERO matches:
grep -rEni \
  'misaya|yanghej|hejaz|tong-jincheng|童锦程|seetacloud|sshpass|connect\.bjb1|@hejazfs|@privaterelay|/Users/|/sessions/|192\.168\.0\.155|misayade' \
  . --include='*.py' --include='*.sh' --include='*.md' \
    --include='*.tex' --include='*.bib' --include='*.yaml' \
    --include='*.json' --include='*.toml' --include='*.cfg' \
    --include='*.ini' --include='*.txt' \
  | grep -v 'example.com\|anonymous\|ANONYMOUS'

# 8. Pack:
cd /tmp && zip -r evq-cosh-supp.zip evq-cosh-supp/ -x '*.pyc' '*.DS_Store'
ls -lh evq-cosh-supp.zip   # should be < 100 MB

# 9. Independent re-audit on the *zip* (paranoia step):
mkdir -p /tmp/evq-audit && cd /tmp/evq-audit && unzip -q /tmp/evq-cosh-supp.zip
pdfinfo evq-cosh-supp/paper/main.pdf
grep -rEni 'misaya|yanghej|hejaz|/Users/|sshpass|seetacloud|wandb\.ai/[^"]+' .
# Both should be empty.
```

---

## 9. Severity summary

| Vector | Status | Severity |
|---|---|---|
| 1.a–h Paper text identity leaks | None | OK |
| 4. PDF metadata | None (timezone offset is soft fingerprint only) | OK |
| 5. Figure EXIF | None | OK |
| 6. Bib self-citation | None | OK |
| 6.1 SSH passwords in `internal/team/archive/` | 2 live credentials | **SECURITY-incident** + P1 |
| 6.2 Cluster SSH host strings | 8 occurrences | P1 |
| 6.3 `/Users/<u>/` paths | 22 lines, 2 distinct usernames | P1 (2 lines inside `paper/`) |
| 6.4 `/root/autodl-tmp/` + `/sessions/<id>/` in to-be-shipped scripts | ~35 lines | P1 |
| 6.5 wandb | entity blank, low risk | P1 (informational) |
| 6.6 GitHub URLs | all third-party | OK |
| 6.7 `tong-jincheng-skill/` sibling tree | gitignored, ensure tarball excludes | P1 |
| 6.10 `PAPER_HANDOVER_2026-04-27.md` strategic content | excludable, anonymity-clean but reviewer-strategy | P1 |
| 6.11 `internal/` strategic content | exclude wholesale | P1 |
| 7. Git history committers | 3 of 4 identities reveal name + corporate email + LAN IP + hostname | P2 (P0 if `.git` ships) |

**Single most important action**: rotate the two SSH passwords in `internal/team/archive/recent_handoffs/2026-{02-27,03-03}_*.md` regardless of the paper. Independent of submission, those are exposed credentials.

**Submission verdict**: **`paper/main.pdf` and the `paper/` LaTeX sources are submittable as-is**. Code-archive release is not yet built; build it strictly via the rsync recipe above (do not `tar czf code.tar.gz .` from repo root, do not include `.git`).
