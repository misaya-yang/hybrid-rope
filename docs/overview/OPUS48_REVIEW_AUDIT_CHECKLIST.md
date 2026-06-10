# Opus 4.8 Review Audit Checklist

Purpose: turn the harsh Opus 4.8 review into a repository-grounded rebuttal and
self-rescue checklist. This is an evidence audit, not a new experiment report.
Do not change numerical claims from this document alone.

Scope:

- Paper claims in `paper/`.
- Reproduction and audit code in `scripts/`.
- Curated and result artifacts in `data/curated/`, `results/`, and `docs/`.
- No private run roots, private machines, or unpackaged local paths should be
  used as public evidence.

Status labels:

- `Closed`: current paper already states the limitation clearly enough.
- `Partly closed`: evidence exists, but wording/provenance still needs cleanup.
- `Open`: rebuttal should not defend this as solved.
- `Do not defend`: concede and scope down.

Severity labels:

- `P0`: must fix before any rebuttal/submission text leans on this point.
- `P1`: high-value wording, provenance, or experiment gap.
- `P2`: supporting cleanup or optional strengthening.

## Executive Verdict

The core mechanism claim is still defensible if kept narrow: EVQ-Cosh is a
closed-form, zero-parameter training-time frequency allocation that exposes
RoPE's finite spectral budget, and it is complementary to inference-time range
scaling. It is not a universal long-context recipe, a replacement for YaRN or
LongRoPE, or a frontier-scale SOTA claim.

The strongest Opus 4.8 attack is not that EVQ "does not work"; it is that the
paper can overstate several diagnostics as if they were broad, robust,
production-scale evidence. The most urgent fixes are to narrow claim language,
make provenance explicit, and stop using supporting single-seed rows as
rebuttal weapons.

The 1B MLA reversal should be treated as a real limitation signal, not ignored.
Repository evidence suggests it is not a same-configuration token-scaling
counterexample: it changes train length from 8K to 4K, changes data, is only
seed 42 in the reported row, and uses the old MLA-32/K16/base500K spectral
substrate where the 4K to 8K extrapolation window is poorly populated by EVQ.
That explains why raw EVQ can improve in-distribution PPL yet lose at 8K/16K.
However, this does not close the issue without checkpoint-level artifact audits.

The earlier suspicion that `+YaRN` evaluation used a fresh local frequency table
instead of checkpoint-loaded frequencies is likely false. In the MLA eval code,
the registered `inv_freq` buffer aliases the local tensor object; PyTorch
`load_state_dict` copies the checkpoint buffer into that object in place. The
code is still too implicit and should be audited with checkpoint hashes, but the
large EVQ/Geo `+YaRN` difference is not explained by "freq came from nowhere."

## Issue Inventory

| ID | Severity | Status | Opus 4.8 attack | Repository-grounded assessment | Required action |
| --- | --- | --- | --- | --- | --- |
| O48-01 | P1 | Open | Novelty may be just a simple schedule change; missing rebased-Geo/fixed-interpolation control. | `scripts/lib/rope/schedules.py` gives a clean EVQ schedule API, but there is no clearly packaged training-time rebased geometric control that isolates "non-geometric shape" from endpoint/range. | Add a minimal control or concede as a missing baseline. Do not claim EVQ beats all plausible training-time schedules. |
| O48-02 | P1 | Partly closed | Theory is surrogate-fitted, not first-principles attention dynamics. | `paper/sections/03_theory.tex` states the exact kernel is oscillatory and EVQ is exact for a broadband surrogate; appendix gives functional validation. | Keep this wording prominent. Rebuttal should say "surrogate-derived and functionally validated," not "derived from full attention." |
| O48-03 | P2 | Partly closed | Constant alpha is chosen for tractability; variable-alpha/Bessel variants are possible. | `paper/appendix/a1_proofs.tex` already discusses variable-alpha alternatives and why constant alpha is used. | No new claim. If space allows, cite this as design choice, not theorem of nature. |
| O48-04 | P1 | Open | `tau=d_eff/sqrt(L)` depends on diffuse post-softmax transport and lacks direct `L_eff^J` measurement. | `paper/sections/03_theory.tex` and `paper/appendix/a1_proofs.tex` frame tau as an operating default and register `L_eff^J` as a falsifiable future measurement. | Do not over-defend tau scaling. Add measurement if possible; otherwise explicitly scope as basin selector. |
| O48-05 | P1 | Partly closed | Flat tau basin weakens the precise tau rule. | Paper already says tau is a basin/default, not global optimum. | Rebuttal should embrace this: shape is the main claim; tau is a stable operating rule. |
| O48-06 | P0 | Partly closed | PK is teacher-forced NLL-gap, softer than autoregressive exact match. | `paper/sections/05_experiments.tex`, `docs/overview/REPRODUCE.md`, and `data/curated/table2_evq_yarn_454m_passkey_10pct.json` identify PK as teacher-forced NLL-gap. | Never present PK as AR exact match. If challenged, concede metric scope and avoid task-accuracy claims. |
| O48-07 | P1 | Partly closed | EVQ alone is modest in the 454M passkey table; the headline is EVQ+YaRN. | `paper/tables/table2_evq_yarn_main.tex` shows EVQ alone improves PK but the large effect is matched-scale EVQ+YaRN. | Phrase as "YaRN has higher leverage on EVQ substrate," not "EVQ alone solves long context." |
| O48-08 | P1 | Open | Matched-scale YaRN does not prove superiority over tuned Geo+YaRN or LongRoPE-style tuning. | Main table uses a fixed matched YaRN scale. The paper already says this is not dominance over every tuned-scale baseline. | Do not defend as tuned-scaler SOTA. A tuned Geo/YaRN control would materially strengthen rebuttal. |
| O48-09 | P0 | Partly closed | PE-dominant Table 4 Geo/DAPE/EVQ rows are seed 42 only. | `paper/tables/table4_pe_dominant.tex` caption states Geo/DAPE/EVQ are seed 42; only Learnable tau is 3-seed. | Any rebuttal table must keep this seed scope explicit. Do not call it a 3-seed primary result. |
| O48-10 | P1 | Partly closed | PE-dominant 64x extrapolation is diagnostic and extreme. | `data/curated/fig3_extreme_128.json` and Table 4 document the diagnostic setup. | Keep as PE-dominant mechanism stress test, not ordinary downstream long-context evidence. |
| O48-11 | P0 | Partly closed | MLA tau convention is under-specified and had a paper-code naming mismatch. | Current scripts set `head_dim=64`, `d_rope=32`, and run `tau=1.414`; the old paper prose called this `d_eff=d_head=128`. The paper now states `tau=1.414` as an empirical `d_eff=128` operating convention rather than deriving it from code `head_dim` or `d_rope`. | Keep convention explicit and non-theorem. Add direct `tau=d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` ablations if possible. |
| O48-12 | P0 | Open | Direct MLA `tau=d_rope/sqrt(L)` ablation is named but not reported. | No packaged primary result closes this. `results/PHASE22_23_MLA_TAU_SWEEP_REPORT.md` analyzes related config effects but is not the same ablation. | Either run/report it or remove any implication that it is resolved. |
| O48-13 | P0 | Open | 1B MLA raw EVQ reverses: EVQ worse at 8K/16K; EVQ+YaRN+FT only mildly better at target. | `results/PHASE18_YARN_FT_REPORT.md` reports raw EVQ worse at 8K/16K for the 1B 4K seed-42 run and small target-length EVQ+YaRN+FT gains. | Treat as limitation and root-cause target. Do not use this row as broad support. |
| O48-14 | P0 | Open | 1B provenance is not reviewer-grade. | Launch/report artifacts show config drift: 4K train length, data change, seed-42-only reported row, old K16/base500K MLA substrate, and historical script/report mismatches. `EXPERIMENT_CODE_RESULT_AUDIT.md` confirms the code chain exists, but exact 1B baseline and YaRN+FT JSON files plus checkpoint/data hashes are absent from the compact branch. | Import sanitized result/checkpoint/data manifests from the external training environment if available; otherwise keep the row supporting-only. |
| O48-15 | P1 | Partly closed | Old MLA-32/base500K differs from production-like DeepSeek settings. | `paper/appendix/a3_supporting_results.tex` notes production DeepSeek uses `d_rope=64`, `base=10K`; `results/PHASE22_23_MLA_TAU_SWEEP_REPORT.md` says K16/base500K can reverse patterns. | Rebuttal should avoid saying the old config is production-identical. Call it a scarce-channel stress test. |
| O48-16 | P1 | Open | LoRA into LLaMA-3-8B lacks matched Geo+LoRA control and has in-distribution cost. | `paper/appendix/a4_supporting_experiments.tex` frames LoRA as supporting only. | Keep supporting-only. Add Geo+LoRA control before using it in rebuttal. |
| O48-17 | P2 | Partly closed | LoRA phase-transition story is phenomenological. | Supporting LoRA text is not a primary claim. | Avoid theory-heavy rebuttal based on LoRA; present as exploratory. |
| O48-18 | P1 | Partly closed | Video DiT support depends on dead-channel regime; Geo can win when channels are alive. | Appendix video rows are supporting; dead-channel framing is mechanism-consistent but not universal video superiority. | Keep supporting-only and disclose base/channel sensitivity. |
| O48-19 | P1 | Partly closed | Progressive training is single-seed and partly driven by Geo+YaRN degradation. | `paper/appendix/a2_experiment_details.tex` marks progressive as single-seed supporting. | Do not use as primary durability proof. |
| O48-20 | P1 | Partly closed | Scale evidence is limited: small from-scratch models, 8B LoRA only. | Paper limitations already say this is a PE mechanism study, not universal long-context recipe. | Keep scale claims narrow. Do not imply frontier-scale validation. |
| O48-21 | P2 | Partly closed | Downstream task accuracy is weak/limited. | `paper/sections/05_experiments.tex` says downstream accuracy is non-regression, not primary. | Rebuttal should lead with PE diagnostics, not downstream wins. |
| O48-22 | P1 | Open | Missing tuned LongRoPE2/CoPE/tuned-scale YaRN baselines. | No complete packaged baseline suite for these alternatives. | Concede baseline gap or add a small, clearly scoped control. |
| O48-23 | P2 | Partly closed | Appendix-heavy presentation; main text may under-deliver caveats. | The main text has several caveats, but key fragilities live in appendix/results docs. | Promote the most important caveats into main/rebuttal prose. |
| O48-24 | P0 | Partly closed | Rebuttal playbook overclaims. | `paper/REBUTTAL_PLAYBOOK.md` had risky statements: no convergence with Geo, no Geo+YaRN outperforming EVQ+YaRN, and PE-dominant as multi-seed. The highest-risk passages have been rewritten to scope 1B, MLA, composition, and PE-dominant evidence. | Keep reviewing before use; do not paste old aggressive language into rebuttal. |
| O48-25 | P0 | Partly closed | Experimental chain has script/artifact drift. | Current scripts and reports had drift in run IDs, passkey mix defaults, compile behavior, intermediate checkpoint names, and remote eval entrypoints. `eval_extended_3seeds.py` and `yarn_finetune_eval.py` now explicitly hash checkpoint-loaded `inv_freq` and resolve both current and historical MLA run IDs; `RESULT_PROVENANCE_MANIFEST.md`, `EXPERIMENT_CODE_RESULT_AUDIT.md`, and `HISTORICAL_SCRIPT_STATUS.md` separate evidence, code support, and historical wrappers. | Import sanitized manifests for recovered artifacts and avoid citing historical wrappers as canonical reproduction commands. |

## Deep Dive A: 1B MLA Reversal

### What the repository says

Primary MLA table:

- Paper location: `paper/appendix/a3_supporting_results.tex`.
- Result artifact: `results/eval_3seeds_full_results.json`.
- Reported setup: 432M/350M-class MLA stress test, 8K train length, 500M
  tokens, 3 seeds, `d_rope=32`, `base=500K`, matched `+YaRN(s=4)`.
- Main result: at 16K, EVQ improves raw PPL versus Geo and EVQ+YaRN is best.

1B reversal row:

- Launch artifact: `scripts/core_text_phases/run_350m_4k_1b.sh`.
- Report artifact: `results/PHASE18_YARN_FT_REPORT.md`.
- Reported setup: 4K train length, 1B tokens, different data mixture, seed 42
  reported, old MLA-32/K16/base500K substrate.
- Raw result: EVQ is better at 4K but worse at 8K and 16K.
- YaRN+FT result: EVQ+YaRN+FT is mildly better at target length, while some
  beyond-target lengths favor Geo+YaRN+FT.

These are not the same experiment with only token count changed. They differ in
train length, data, seed coverage, and extrapolation window.

### Frequency-window explanation

For MLA-32, there are only 16 rotary frequency channels. With `base=500K`, the
frequency periods are very sparse around the 4K to 16K extrapolation region.

Canonical schedule inspection gives the following qualitative pattern:

| Setup | In-train region | 2x window | 4x window | Interpretation |
| --- | --- | --- | --- | --- |
| 4K Geo, K16/base500K | many channels below 4K | one channel near 4K-8K | one channel near 8K-16K | Geo keeps a bridge into 8K. |
| 4K EVQ, K16/base500K, tau 1.414 | more channels below 4K | near-empty 4K-8K window | one channel near 8K-16K | EVQ can lower 4K PPL but fail early extrapolation. |
| 8K Geo, K16/base500K | many channels below 8K | one channel near 8K-16K | sparse beyond 16K | Baseline is weak but not empty at 2x. |
| 8K EVQ, K16/base500K, tau 1.414 | more useful coverage below 8K | one channel near 8K-16K | additional coverage near 16K-32K | EVQ aligns better with the primary 8K stress test. |

This is consistent with `results/PHASE22_23_MLA_TAU_SWEEP_REPORT.md`, which
identifies K16/base500K as a regime where EVQ patterns can reverse and a
DeepSeek-aligned K32/base10K setup restores a healthier channel distribution.

### Judgment

The 1B result should not be hidden. It is a valid limitation and reviewer attack.
The best current answer is:

1. It is not a same-config token-scaling contradiction.
2. It exposes a real fragility of the old MLA-32/K16/base500K setup at 4K.
3. It supports the finite-spectral-budget thesis more than a universal EVQ win:
   when the budget is too sparse in the wrong window, shape can hurt.
4. It cannot be promoted as supporting evidence without checkpoint-level audit.

### Required closure checks

- Run `scripts/core_text_phases/audit_rope_checkpoint.py` on the exact Geo, EVQ,
  Geo+YaRN, and EVQ+YaRN/FT checkpoints.
- Record checkpoint SHA256, `inv_freq` SHA256, inferred schedule family, and
  inferred tau.
- Record data artifact hashes and whether cached train tensors are 1D flat or
  2D chunked.
- Re-evaluate with the exact eval script used for the paper or mark the row as
  historical/supporting-only.
- If rerunning is possible, run at least seeds 43 and 88 or remove any durability
  language.

## Deep Dive B: Where `+YaRN` Frequencies Come From

### Concern

The concern was that `+YaRN` eval scripts might load a checkpoint but then apply
YaRN to a freshly constructed local `inv_freq`, losing the trained frequency
substrate.

### Code-level finding

The eval scripts construct an `inv_freq` tensor and pass it into the model. The
model registers that tensor as a buffer. PyTorch `load_state_dict` then copies
checkpoint buffer values into the registered buffer in place. Because the local
`inv_freq` object aliases the registered buffer, the local variable is mutated
to the checkpoint value after load.

Therefore, if the checkpoint contains `inv_freq`, the subsequent YaRN scaling is
expected to use the checkpoint-loaded frequency table, not the original locally
constructed Geo/EVQ table.

### Remaining risk

The behavior is implicit and fragile. It depends on aliasing and on the
checkpoint actually containing the buffer. It also makes reviewer-facing
reproduction harder to inspect.

Required cleanup:

- Done in `scripts/core_text_phases/eval_extended_3seeds.py` and
  `scripts/core_text_phases/yarn_finetune_eval.py`: after `load_state_dict`,
  the scripts explicitly clone the checkpoint-loaded RoPE `inv_freq`, print a
  short SHA256 hash, and apply YaRN to that loaded table.
- Done in those same scripts: fail loudly if checkpoint `inv_freq` is absent
  when evaluating frequency allocation experiments.
- Done: `tests/test_yarn_checkpoint_inv_freq.py` verifies the helper behavior
  without requiring pytest.

## Deep Dive C: Provenance Gaps

The following gaps are reviewer-risky because they make reported rows harder to
reproduce from the packaged repository:

- Some launch scripts reference remote run locations rather than repo-relative
  entrypoints.
- Current `run_gqa_evq_experiment.py` run IDs include architecture labels, while
  some eval scripts expect older names.
- Reported passkey mix and current defaults do not always line up across GQA/MLA
  and core text sweep scripts.
- Some reports mention compile or intermediate checkpoints that are not clearly
  produced by the current committed script.
- Early 1B data preparation scripts can write flat token tensors, while training
  loaders expect chunked sequence tensors unless a later preparation version was
  used.
- `backup/2026-03-06` contains archival raw artifacts that were intentionally
  removed from compact main-branch reviewer paths. This means "not packaged" is
  not equivalent to "not run."

Required cleanup:

- Done: `docs/overview/RESULT_PROVENANCE_MANIFEST.md` records compact-branch
  evidence, archival branch pointers, and missing artifact gates.
- Done: `docs/overview/HISTORICAL_SCRIPT_STATUS.md` marks server launch wrappers
  and patch scripts as historical unless promoted with a manifest.
- Done: `scripts/core_text_phases/make_artifact_manifest.py` can be run on an
  external artifact machine to produce a sanitized JSON ledger with file hashes,
  tensor metadata, and optional `inv_freq` audit summaries.
- Done: `docs/overview/EXPERIMENT_CODE_RESULT_AUDIT.md` separates code support,
  implementation health, and JSON/result artifacts for each major row.
- Done: `docs/overview/PAPER_DESCRIPTION_AUDIT.md` records paper wording checks,
  fixed overstatements, and remaining caveats.
- For each primary table row, record script commit, config, data artifact hash,
  checkpoint hash, eval script, seed list, and metric definition.
- Mark stale server helper scripts as historical if they are not intended to
  reproduce current tables.

## Deep Dive D: Rebuttal Language That Must Change

High-risk statements in `paper/REBUTTAL_PLAYBOOK.md` should be rewritten before
any reviewer response is drafted:

- "No evidence of convergence between EVQ and Geo at any training duration" is
  too strong. The 1B 4K raw MLA row and earlier 4K reports show regimes where
  Geo catches or beats EVQ.
- "No experiment has ever shown Geo+YaRN outperforming EVQ+YaRN" is too strong.
  Phase18 target lengths favor EVQ+YaRN+FT, but some beyond-target lengths favor
  Geo+YaRN+FT.
- PE-dominant evidence should not be described as fully 3-seed for Geo/DAPE/EVQ.
  The table caption says those rows are seed 42.
- Old MLA-32/base500K should not be described as production-identical to
  DeepSeek MLA. It is a scarce-channel stress test inspired by compressed RoPE;
  production-like settings use different rotary width/base choices.

Safer rebuttal posture:

- EVQ changes the training-time frequency substrate.
- Matched inference-time scaling can have higher leverage on that substrate.
- The evidence is strongest in mechanism stress tests, not downstream SOTA.
- 1B 4K MLA reveals a budget-window failure mode, which is compatible with the
  finite-spectral-budget thesis but narrows the claim.

## Rebuttal Answer Templates

### Does the 1B reversal invalidate EVQ?

No, but it is a real limitation. The row changes train length, data, seed
coverage, and spectral window. It should be treated as a stress-test failure of
the old 4K MLA-32/K16/base500K substrate, not as same-config token scaling. The
right rebuttal is to scope down and, if possible, show checkpoint frequency
audits and a direct tau/window ablation.

### Is `tau=d_eff/sqrt(L)` derived?

Only in the limited sense stated in the paper. The EVQ-Cosh shape follows from
the surrogate variational problem. The tau scale is an operating default from a
diffuse transport calibration and is supported by an empirical basin. It is not
claimed as a global optimum.

### Why use teacher-forced NLL-gap passkey?

Because the paper is a PE mechanism study, not a final downstream benchmark
paper. PK should always be named as teacher-forced NLL-gap retrieval unless a
row is explicitly marked as autoregressive exact match.

### Why no tuned Geo+YaRN or LongRoPE2 baseline?

This is a fair gap. The main EVQ x YaRN table tests matched-scale leverage, not
tuned-scaler dominance. Rebuttal can add a small control if time permits; absent
that, it should concede scope.

### Why is LoRA supporting only?

Because it is single-seed/adaptation-only and lacks a matched Geo+LoRA control.
It can show plausibility of modifying frequency substrate in a pretrained model,
but it should not carry the main claim.

## Action Plan

### Immediate text/doc fixes, no new experiments

- [x] Rewrite `paper/REBUTTAL_PLAYBOOK.md` overclaims listed above.
- [x] Add a 1B 4K MLA limitation sentence wherever Phase18 is mentioned in
  paper-facing docs.
- [x] Ensure every PK mention says teacher-forced NLL-gap unless AR exact match is
  explicitly reported.
- [x] Ensure the rebuttal quick-reference table no longer describes the
  PE-dominant Geo/DAPE/EVQ rows as fully 3-seed.
- [x] Keep MLA `d_eff=128` as a calibrated operating convention, not a derivation
  from code `head_dim=64` or `d_rope=32`, in paper/rebuttal wording.

### Lightweight code/provenance fixes

- [x] Add `scripts/core_text_phases/audit_rope_checkpoint.py` as an artifact
  gate script.
- [x] Add `scripts/core_text_phases/make_artifact_manifest.py` as a sanitized
  external artifact manifest generator.
- [x] Add `docs/overview/EXPERIMENT_CODE_RESULT_AUDIT.md` for code/result
  support status.
- [x] Add `docs/overview/PAPER_DESCRIPTION_AUDIT.md` for paper wording support
  status.
- [ ] Run `scripts/core_text_phases/audit_rope_checkpoint.py` on exact primary
  and 1B checkpoints.
- [ ] Run `scripts/core_text_phases/make_artifact_manifest.py` on recovered
  external checkpoints/data/logs before promoting them into public docs.
- [x] Make `+YaRN` eval scripts explicitly read `model.inv_freq` after checkpoint
  load before applying scaling.
- [x] Add result manifests for Table 2, Table 4, MLA primary, and 1B supporting
  rows.
- [x] Mark stale or remote-run helper scripts as historical if they are not current
  reproduction entrypoints.

### Minimal experiment priorities

1. P0: MLA direct tau ablation: `tau=d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` versus current convention.
2. P0: Checkpoint frequency audit for all 1B 4K and primary MLA rows.
3. P1: Re-run PE-dominant Geo/DAPE/EVQ with 2 more seeds or stop calling it
   primary evidence without qualification.
4. P1: Add matched Geo+LoRA control if LoRA is used in rebuttal.
5. P1: Add a simple rebased-Geo/fixed-interpolation training-time schedule
   control.
6. P2: Measure `L_eff^J` on trained attention maps to test the tau calibration.
7. P2: Add tuned-scale Geo+YaRN and, if feasible, LongRoPE2/CoPE baselines.

## Final Reviewer-Safe Position

Defend:

- EVQ-Cosh exposes RoPE frequency allocation as a finite spectral budget.
- The schedule is closed-form and zero-parameter.
- Matched YaRN can leverage EVQ's trained substrate better than Geo in the main
  passkey-mix stress test.
- Scarce-channel MLA results support the mechanism in the 8K/500M 3-seed setup.

Concede or scope:

- Tau is a basin selector, not globally optimal.
- PK is teacher-forced NLL-gap.
- PE-dominant and supporting rows have seed/provenance limits.
- 1B 4K MLA reversal is a real failure mode under sparse old-frequency settings.
- Missing tuned scaler and training-time schedule controls limit breadth.

Do not say:

- EVQ replaces YaRN/LongRoPE/LongRoPE2/DAPE/FIRE.
- EVQ universally improves long-context performance.
- The 1B run proves durability.
- The MLA `d_eff=128` convention is theoretically forced by code `head_dim` or `d_rope`.
- PK is autoregressive exact retrieval unless explicitly measured that way.
