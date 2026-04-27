# Audit 06 - Reproducibility narrative truthfulness

Auditor: 6/8 (parallel deep audit)
Date: 2026-04-27
Scope: Does paper Q5 (and supporting Q4/Q6) over-promise relative to what the repo can actually deliver in an anonymous supplementary archive?

Sources read:
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/scripts/2026-04/PAPER_HANDOVER_2026-04-27.md`
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/internal/2026_04_run/docs/24_anonymous_code_release_0424.md`
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/paper/main.tex` (lines 90-180, esp. 104-116)
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/paper/appendix/a2_experiment_details.tex`
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/docs/overview/REPRODUCE.md`
- `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/docs/overview/PAPER_CLAIMS_MAP.md`
- repo tree under `scripts/`, `experiments/`, `tests/`, `internal/`

---

## 1. Q5 promise enumeration (verbatim)

`paper/main.tex:108`:

> "We include an anonymous supplementary code archive containing the EVQ-Cosh inverse-frequency initializer, the collision-surrogate and \(\tau^*\)-rule analyses, evaluation scripts, and configuration files for the primary experiments (EVQ\(\times\)YaRN 454M, PE-dominant 125M/454M, MLA 432M). Upstream datasets (FineWeb-Edu, TinyStories, QuALITY, RULER passkey/NIAH) are available from their respective public releases; our passkey-mix composition and data-prep scripts are in the archive. Model checkpoints and weights are not included to preserve double-blind review and to keep the archive within size limits. A reproducibility README lists the commands to regenerate each primary-tier table and figure from a fresh checkout."

Numbered components (P = promised; numbering used in audit table):

| # | Promised artifact |
|---|-------------------|
| P1 | EVQ-Cosh inverse-frequency initializer |
| P2 | Collision-surrogate analysis |
| P3 | \(\tau^*\)-rule analysis |
| P4 | Evaluation scripts |
| P5a | Configuration file: EVQ\(\times\)YaRN 454M |
| P5b | Configuration file: PE-dominant 125M |
| P5c | Configuration file: PE-dominant 454M |
| P5d | Configuration file: MLA 432M |
| P6 | Passkey-mix composition + data-prep scripts |
| P7 | Reproducibility README mapping primary tables/figures to commands |

Note: "configuration files" is interpreted as Q5 promises — the natural reading is YAML/JSON config a reviewer can run, which is what `internal/2026_04_run/docs/24_anonymous_code_release_0424.md` (the official packaging plan) targets at lines 28-31. If interpreted permissively as "the runnable training script with hard-coded hyperparameters", every primary except MLA is reachable; MLA is not (see audit table).

---

## 2. Component audit table

| # | Promise | Search result (absolute path) | Tag | Severity |
|---|---------|-------------------------------|-----|----------|
| P1 | EVQ-Cosh initializer | `scripts/lib/rope/schedules.py:98` (canonical `build_inv_freq` with `evq_cosh` branch) + 30+ duplicates incl. `scripts/core_text_phases/run_evq_sweep.py:141`, `scripts/core_text_phases/evq_analysis.py:74`, `tests/test_rope_core.py:63`. Method aliases at `schedules.py:14-34`. | Exists clearly | OK |
| P2 | Collision-surrogate analysis | `scripts/analysis/verify_c_coll.py` (Q1(L,b) + c_pred = sqrt(45 Q1)); `scripts/analysis/exp_tau_theory_verify.py` (verify_softmax_transport family); `scripts/analysis/verify_softmax_transport.py`, `scripts/analysis/verify_softmax_v2.py`, `scripts/analysis/verify_stiffness_and_regime.py`. | Exists clearly | OK |
| P3 | \(\tau^*\)-rule analysis | `scripts/core_text_phases/phase16_formula_optimality_sweep.py` (99-run sweep producing Fig 6); `scripts/analysis/tau_exact_derivation.py`; `scripts/analysis/tau_refined_formula.py`; `scripts/analysis/tau_scaling_analysis.py`; `scripts/verify_tau_unified.py`; `scripts/analysis/compute_eta_vp.py` (DiT \(\eta\) cross-check). | Exists clearly | OK |
| P4 | Evaluation scripts | `scripts/core_text_phases/eval_passkey.py`, `eval_multi_needle.py`, `eval_dsr.py`, `eval_super_extrap.py`, `eval_extended_3seeds.py`, `eval_pe_baselines.py`, `eval_phase17h_yarn_strict.py`, `eval_phase17h_fineweb_strict.py`; `scripts/supporting_eval/eval_passkey_scratch.py`, `eval_niah_recall.py`, `eval_niah_heatmap.py`, `eval_longbench.py`, `eval_multi_needle.py`; `scripts/text_eval/eval_454m_multilength.py`. | Exists clearly | OK |
| P5a | Config: EVQ\(\times\)YaRN 454M | No `*.yaml` or stand-alone config exists (`find scripts/ experiments/ -name '*.yaml'` returns 0 hits). The 454M EVQ\(\times\)YaRN run is reconstructible by running `scripts/core_text_phases/phase17b_454m_512_to_1024_continue_ckpt_eval.py` then `scripts/core_text_phases/phase17c_454m_1024_to_2048_continue.py` (CFG inline at `phase17c_*.py:164`), but no externalized config matches the packaging-guide layout `experiments/configs/mha_454M_L512.yaml` (`internal/2026_04_run/docs/24_anonymous_code_release_0424.md:30`). | Partial | P1 |
| P5b | Config: PE-dominant 125M | `scripts/core_text_phases/phase11b_125m_dape.py` is the runnable script; CFG inline. No `mha_125M_L128.yaml` per packaging plan. | Partial | P1 |
| P5c | Config: PE-dominant 454M | `scripts/core_text_phases/phase11c_454m_scaling.py` is runnable (Phase 11C 454M token-scaling), CFG inline. No externalized YAML. | Partial | P1 |
| P5d | Config: MLA 432M | The shell wrapper `scripts/core_text_phases/run_350m_mla32_500m.sh:6` invokes `run_gqa_evq_experiment.py`, which **does not exist anywhere in the repo** (verified with `find` and `grep`). Paper says "432M MLA, 500M tokens, L_train=8192, 3-seed" but the only MLA-aware shell wrapper points at a missing script. The `run_evq_sweep.py` `TIER_CONFIGS` only covers 50m/125m/350m/500m MHA tiers; there is no `--attn_type mla` branch in that file. Other MLA-named files: `mla_tau_optimization.py`, `mla_tau_optimization_v2.py`, `mla_patch.py`, `fix_mla_assert.py` — these are sweep/patch helpers, not the entry point that produced the 432M MLA Table 19 numbers. **The runnable entry point that produced the 432M MLA primary results is not in the repo.** | Missing entirely | **P0** |
| P6 | Passkey-mix composition + data-prep | Composition: `scripts/core_text_phases/run_evq_sweep.py:645` (`maybe_wrap_with_passkey_mix`) + `scripts/supporting_eval/eval_passkey_scratch.py` (`MixedDataset`); fixed-ratio gate via `PASSKEY_MIX_RATIO` env at `run_evq_sweep.py:623`. Data-prep: `scripts/data_prep/prepare_mixed_prior_dataset_v1.py` (FineWeb-Edu prior mix, NOT the passkey-mix used by the paper's primary), `prepare_8k_mixed_500m.py` (SlimPajama tokenizer), `tokenize_synth.py`. There is **no stand-alone "prepare passkey-mix" script** as Q5 promises and as the packaging guide expects at `data/prepare_passkey_mix.py` (`internal/2026_04_run/docs/24_anonymous_code_release_0424.md:135`); the composition is implicit inside `run_evq_sweep.py` and triggered at training time, not as a pre-prep step. The `scripts/data_prep/README.md:9` even claims `prepare_mixed_prior_dataset_v1.py` does "FineWeb-Edu 预标记化 + Passkey 混合" but that file's docstring talks about `power_law_base / bimodal_reasoning / uniform_scaffold` priors, not passkey insertion (verified by reading the file head-30). | Partial | **P1** (close to P0 — Q5 reads a reviewer can find a freestanding passkey-mix prep script; in fact it is fused into the trainer) |
| P7 | Reproducibility README mapping tables/figures to commands | `docs/overview/REPRODUCE.md` (table-1 path through table-6 path, hardware tiers, 50M-quick-validate route). `docs/overview/PAPER_CLAIMS_MAP.md` (every Figure 1-7 + Table 1-6 → script + data + report). `README.md:108-118` summary mapping. Together they satisfy "lists the commands to regenerate each primary-tier table and figure". The packaging plan `internal/2026_04_run/docs/24_anonymous_code_release_0424.md:37` refers to a separate `repro.md` placeholder that does not yet exist in the repo. | Partial — substantively present in `docs/overview/`, but those files use Chinese narrative + reference internal absolute paths (`/root/autodl-tmp/...`) and would need an English, anonymized rewrite into the supp archive | P1 |

### 2.1 Severity summary

- **P0 (true gap)**: 1 — MLA 432M entry point is missing entirely (`run_gqa_evq_experiment.py`). Reviewers cannot reproduce the third primary anchor from the supplied code.
- **P1 (partial)**: 5 — externalized configs (P5a-c), packaged passkey-mix prep (P6), supp-archive-grade README (P7).
- **P2**: 0 within the Q5 promise scope.

---

## 3. Gap-closing plan (per ⚠️/❌ row)

### P5d / 432M MLA (severity P0)

**Action**: Recover or re-author `run_gqa_evq_experiment.py` (or rename and patch the existing MLA-aware script) and ship it under `scripts/core_text_phases/`.

Candidates to inspect for cannibalization:
- `scripts/core_text_phases/mla_tau_optimization_v2.py` (likely contains the MLA forward + RoPE-decoupled paths)
- `scripts/core_text_phases/mla_patch.py`, `fix_mla_assert.py` (apparent monkey-patches to GQA-MLA forward)
- `internal/2026_04_run/` (the actual training scripts may be under `internal/` per `.gitignore`-by-policy and need to be promoted to `scripts/`)

Effort: 2-4 person-hours to rebuild a clean `scripts/core_text_phases/run_mla_432m_evq.py` that exposes `--tier 432m_mla --d_rope 32 --taus 0,X --seeds 42,43,44 --train_tokens 5e8 --seq_len 8192`. Re-run is **NOT** required if the original 432M MLA training results JSONs are already in `results/`; only a "smoke-test runnable" entry point is.

Blocker risks:
- MEDIUM: if the actual training code was a fork of HuggingFace `DeepseekV2` modeling, a clean MIT-licensable single-file may need a careful re-derivation.
- LOW: paths like `/root/autodl-tmp/...` need scrubbing.

### P5a / P5b / P5c — externalized YAML configs

**Action (option A, minimal)**: Add three small YAML files under `scripts/core_text_phases/configs/` (or `experiments/configs/`) that codify the inline `CFG_*` dicts in the corresponding `phase17{b,c}_*.py`, `phase11b_*.py`, `phase11c_*.py`. Each YAML is ~40 lines (model arch + optim + data + seed). Loader can be one tiny `--config X.yaml` flag added to each runner.

Effort: ~3 person-hours total (1h per primary × 3, plus loader plumbing).

**Action (option B, defer interpretation)**: Soften the Q5 wording at `paper/main.tex:108` from "configuration files for the primary experiments" to "training scripts with hard-coded configurations for each primary experiment". 0 person-hours but is a **promise rewrite** (Q5 stays Yes; just the prose tightens).

Recommendation: **Option B** — the runnable training scripts are already in the repo; calling them "configuration files" is defensible if a reviewer reads `phase17c_454m_1024_to_2048_continue.py` lines 162-185 (the `CFG_454M_2048` dict).

Blocker: NONE.

### P6 — passkey-mix prep packaging

**Action**: Carve out a stand-alone `scripts/data_prep/prepare_passkey_mix.py` that wraps `MixedDataset` (currently in `scripts/supporting_eval/eval_passkey_scratch.py`) and `maybe_wrap_with_passkey_mix` (currently in `scripts/core_text_phases/run_evq_sweep.py:645-690`) so the prep step is reachable independently of the trainer. ~80-120 LOC.

Effort: 2 person-hours.

Blocker: LOW — code already exists, just needs to be assembled into one self-contained CLI.

Side fix: `scripts/data_prep/README.md:9` mis-describes `prepare_mixed_prior_dataset_v1.py` as the passkey-mix prep; correct it (or replace the row).

### P7 — supp-archive-grade reproduction README

**Action**: Author `repro.md` per packaging plan. Translate `docs/overview/REPRODUCE.md` (Chinese, internal paths) and `docs/overview/PAPER_CLAIMS_MAP.md` (table → script map) into one English, anonymized README placed at the repo root or under `scripts/2026-04/`. Replace `/root/autodl-tmp/...` with `./data` and `./checkpoints`.

Effort: 1.5-2 person-hours (existing material is already substantive; this is a translation + scrubbing pass).

Blocker: NONE — content already in `docs/overview/`.

### Total effort

If all gaps are closed: **8.5-12 person-hours** (well under the 1.5-2 day window between now and submission). The dominant item is **P0 MLA 432M entry-point recovery (2-4 h)**; everything else is packaging/wording.

If only P0 is closed and the prose is softened: **3-5 person-hours**.

---

## 4. Packaging-guide cross-check (`internal/2026_04_run/docs/24_anonymous_code_release_0424.md`)

### 4.1 Path validity

The guide describes a TARGET supplementary structure; the paths in section 0 are **all aspirational and not yet present in the live repo**:

| Guide path | Exists? | Note |
|------------|---------|------|
| `evq/inverse_freq.py` | NO | EVQ-Cosh code lives in `scripts/lib/rope/schedules.py` and ~30 duplicate inline copies — needs consolidation. |
| `evq/warp.py`, `evq/surrogate.py` | NO | C_app/K_app surrogate code is split across `scripts/analysis/verify_c_coll.py`, `verify_softmax_transport.py`, `verify_stiffness_and_regime.py` — needs unifying file. |
| `analyses/tau_formula_validation.py` | NO file with that name. The 99-run sweep is in `scripts/core_text_phases/phase16_formula_optimality_sweep.py`. |
| `analyses/lambda_curvature.py` | NO. Function is in `scripts/analysis/verify_c_coll.py` (different name). |
| `analyses/surrogate_validation.py` | NO. Function is in `scripts/analysis/verify_softmax_transport.py` (different name). |
| `analyses/stiffness_sweep.py` | NO. Function is in `scripts/analysis/verify_stiffness_and_regime.py`. |
| `experiments/configs/{mha_454M_L512.yaml, mha_125M_L128.yaml, mla_432M_L8K.yaml}` | NO yaml configs exist anywhere (`find` confirmed). |
| `experiments/run.py` | NO. Each primary has its own `phase{NN}_*.py` runner. |
| `eval/{passkey.py, niah.py, ruler.py, ppl.py}` | NO. Eval scripts are at `scripts/core_text_phases/eval_*.py` and `scripts/supporting_eval/eval_*.py` with longer names. |
| `data/prepare_passkey_mix.py` | NO. (See P6 above.) |
| `repro.md` | NO. (See P7 above.) |

So the guide is a **plan, not a description of the current state**. It bills 1.5 hours of work as if everything just needs to be `rsync`'d, but the repo's current naming, structure, and consolidation level mean it actually requires the 8.5-12 person-hours estimated in §3.

### 4.2 Scope vs Q5 promise consistency

The guide and Q5 promise the SAME components, with two divergences:

- **MLA scope**: Both Q5 (paper/main.tex:108) and the guide (line 30, `mla_432M_L8K.yaml`) promise an MLA primary config. Q5 says "MLA 432M"; the guide says `mla_432M_L8K.yaml`. **Consistent.** But neither is currently shippable (P0).
- **Reproducibility README**: Q5 says "A reproducibility README lists the commands to regenerate each primary-tier table and figure"; the guide's section 3 README template (lines 110-138) is shorter and just gives 4 quick-start one-liners + a single torchrun example. The guide's `repro.md` (line 37, 122-126) is the deeper "per primary" mapping — but no draft `repro.md` is in the repo. **Consistent in promise, both unfulfilled.**

In one place the guide is **stricter** than Q5: line 4 mandates "**必须不包含**任何能识别作者...的信息" (zero leak tolerance), which is the right standard for double-blind. Q5 just claims "anonymous supplementary".

### 4.3 Anti-leak measures in the guide

The guide does have explicit anti-leak passes:
- `internal/.../24_anonymous_code_release_0424.md:42-56` — table of removal/replacement rules (wandb keys, emails, cluster paths, internal slurm/submitit, `internal/`, raw `results/`).
- Lines 60-105 — `rsync` exclude list (`.git`, `internal/`, `paper/`, `wandb/`, `*.ckpt`, etc.) and an identity grep with allowlist.
- Lines 142-149 — "final identity grep" for `your_name|@gmail|@anthropic|@openai|@meta|@google|slack|jira|wandb|/mnt/|/shared/|your_org`.

These are **adequate but not sufficient**:
1. The grep at line 146 is not actually run as part of the `git archive` pipeline; it is a manual checklist item. There is no CI hook, no `.github/workflows/anon_check.yml`, no `Makefile target`. Risk: HIGH if the human packager runs the rsync but skips the grep.
2. The cluster-path scrubbing relies on a manual sed; common offenders such as `/root/autodl-tmp/`, `connect.bjb1.seetacloud.com` (cited in `MEMORY.md`), and `/Users/misaya...` are not in the guide's grep list. Recommend adding `autodl|seetacloud|misaya|/Users/` to line 146.
3. There is no `git filter-repo`/`git filter-branch` pass; `git archive` strips the `.git/config` but not in-file commit hashes the assistant may have left as comments. Recommend a final `grep -rEn "git[a-z]*\.com|commit hash" supp/`.

These are P1 polish items. The supplied skeleton is good enough to ship if the human follows the manual checklist.

---

## 5. Q4 / Q6 vs `a2_experiment_details.tex` consistency

### Q4 (Experimental result reproducibility, paper/main.tex:97-102)

> "The body states the principal settings and the appendix records datasets, seeds, evaluation grouping, and the status of larger supporting runs."

Verification against `paper/appendix/a2_experiment_details.tex`:

| Q4 sub-claim | a2 location | Verdict |
|--------------|-------------|---------|
| "datasets" | a2:24-26 (FineWeb-Edu + 10% passkey mix; FineWeb-Edu / TinyStories; FineWeb-Edu) and a2:189 (LLaMA-2 reference set), §a2 video-DiT block lists OscMNIST | YES (datasets recorded for the three primary anchors and DiT) |
| "seeds" | a2:23-27 reproducibility table includes a "Seeds" column ("3+3", "1-3", "3"); a2:71 says "12 runs: 3 seeds × ..."; a2:121, a2:131 list seeds 42 and 137 for DiT | YES |
| "evaluation grouping" | a2:11-12 ("These anchors were selected because they are either multi-seed or directly tied to the central theoretical prediction") plus the body §5 evidence-tier paragraph at `05_experiments.tex:11` | YES |
| "status of larger supporting runs" | a2:65-72 ("§5.3 Larger-scale supporting evidence" — 750M is "supporting evidence only, single-seed, LongBench download failed"); a2:74-104 progressive 454M is "single-seed run separate from the primary"; a2:145 "preliminary scale-up to 382M DiT (single seed) ... reported as supporting evidence only" | YES |

Q4 justification is **truthful and adequately backed by a2**. No gap.

### Q6 (Experimental setting/details, paper/main.tex:111-116)

> "Main settings are given in the body and additional details are placed in the appendix."

Verification:

- `a2:33-61` Table `tab:hyperparams` documents optimizer, LR per tier (50M / 125M / 454M / 750M), schedule, warmup, batch sizes, weight decay, gradient clipping, dropout, sequence packing. **Covered.**
- `a2:51-58` adds video-DiT and MLA per-tier deviations. **Covered.**
- `a2:62-63` the prose disclosure of `b=500K` primary base. **Covered.**

Q6 justification is **truthful**.

### One minor cross-Q snag

Q4 paper text says the body states the principal settings, but §5 body actually defers PE-dominant 125M training-token count, etc., entirely to the appendix. Reviewer might tag this as Q6/Q4 borderline; in practice both Q4 and Q6 answer "Yes" with a deferred-to-appendix justification, and a2 actually covers what was deferred. **No correction needed.**

---

## 6. Summary

**Q5 truthfulness verdict**: The paper's Q5 justification is **mostly truthful but has one P0-severity gap** that should be closed before submission:

1. **P0 — MLA 432M entry point missing**. `scripts/core_text_phases/run_350m_mla32_500m.sh:6` references `run_gqa_evq_experiment.py`, which does not exist in `scripts/`, `experiments/`, or `tests/`. The 432M MLA primary cannot be reproduced from the public code as currently committed. Either recover/re-author this file (2-4 person-hours) or remove the explicit "MLA 432M" claim from Q5 (5 minutes; rewrite `paper/main.tex:108` to drop "MLA 432M" or label it "code release pending camera-ready"). Recommend the former — Table 19 (`paper/main.aux:345-346`) and §exp-mla in `paper/sections/05_experiments.tex:48` both load-bearing on this primary.

2. **P1 — Five packaging deltas**: externalized YAML configs (P5a-c), stand-alone passkey-mix prep (P6), and an English/anonymized supp-archive README (P7). These are the dominant time cost of the packaging plan but are pure restructuring; no scientific work needed.

The `internal/.../24_anonymous_code_release_0424.md` packaging guide is a **forward-looking plan**, not a description of the current repo. Its 1.5-hour budget is **optimistic by ~6-10 hours** because it assumes all the components already sit at the right paths. Anti-leak measures in the guide are adequate (manual identity grep) but lack automation; recommend extending the grep list to include `autodl|seetacloud|misaya|/Users/`.

`a2_experiment_details.tex` adequately backs both Q4 and Q6 — no truth-in-advertising issue at those checklist items.

**Bottom line**: P0 is real and time-bounded (2-4 h). Q5 itself can stay `[Yes]` if the MLA gap is closed. If the team chooses not to close P0 before submission, the only correct action is to soften Q5 to drop the explicit MLA 432M mention and let the §exp-mla data + REBUTTAL_PLAYBOOK carry the burden.
