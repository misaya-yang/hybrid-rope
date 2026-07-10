# OPUS 4.8 Forensic Audit Report

Last updated: 2026-06-02

This report answers the original forensic prompt at repository level: whether the
experiments have code support, whether the implementation paths are coherent,
whether current JSON/result artifacts exist, and where paper prose exceeds the
available evidence. It complements:

- `OPUS48_REVIEW_AUDIT_CHECKLIST.md`
- `EXPERIMENT_CODE_RESULT_AUDIT.md`
- `RESULT_PROVENANCE_MANIFEST.md`
- `PAPER_DESCRIPTION_AUDIT.md`
- `HISTORICAL_SCRIPT_STATUS.md`

Evidence classes:

- `A`: current compact tree has code and reviewer-facing JSON/result artifact.
- `B`: code path is present, but raw run/checkpoint manifest is incomplete.
- `C`: code and Markdown report exist, but exact JSON/checkpoint artifact is
  missing from the compact tree.
- `D`: not found or insufficient evidence.

## 1. Executive Summary

The paper's core mechanism claim remains defensible only in its scoped form:
EVQ-Cosh changes the training-time RoPE frequency substrate, and matched
inference-time scaling can use that substrate differently. The strongest current
evidence is the 454M EVQ x YaRN table and the 432M/350M-class MLA 8K/500M
three-seed stress test.

The 1B/4K MLA anomaly is not "no code." It is code-backed and report-backed:
`run_350m_4k_1b.sh` / `run_350m_4k_v2_1b.sh` invoke
`run_gqa_evq_experiment.py`, and `yarn_finetune_eval.py` implements baseline,
inference-only YaRN, and YaRN+FT. However, the compact branch does not contain
the exact 1B baseline or YaRN+FT JSONs behind the historical Phase18 report reference.
Therefore the 1B row is not reviewer-grade provenance and should remain
supporting-only or be removed from any evidence chain that requires JSON replay.

The biggest newly confirmed paper-code issue is the MLA tau convention. Current
scripts use `head_dim=64`, `d_rope=32`, and `tau=1.414`; older prose described
the row as `d_eff=d_head=128`. The paper has been tightened to say
`tau=1.414` is an empirical `d_eff=128` MLA operating convention, not a
derivation from the code-level `head_dim` or `d_rope` fields. Direct
`tau=d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` ablations remain open.

## 2. Repository Map

| Area | Key files | Role | Status |
| --- | --- | --- | --- |
| Canonical RoPE schedules | `scripts/lib/rope/schedules.py` | EVQ-Cosh and baseline schedule API | Current |
| Core MHA training | `scripts/core_text_phases/run_evq_sweep.py` | MHA model, data loader, training loop, PPL/passkey helper integration | Current, but old fallback/data-cache behavior needs manifests |
| Core GQA/MLA training | `scripts/core_text_phases/run_gqa_evq_experiment.py` | GQA/MLA model, EVQ inv-freq construction, checkpoint/results persistence | Current consolidated entrypoint |
| Primary MLA eval | `scripts/core_text_phases/eval_extended_3seeds.py` | 8K/500M MLA progression and extended PPL/YaRN eval | Patched to use checkpoint-loaded `inv_freq` explicitly |
| 1B/4K YaRN+FT eval | `scripts/core_text_phases/yarn_finetune_eval.py` | Baseline, inference-only YaRN, YaRN+FT for MLA checkpoints | Patched to use checkpoint-loaded `inv_freq` explicitly |
| Passkey helper | `scripts/supporting_eval/eval_passkey_scratch.py` | Teacher-forced NLL-gap and auxiliary AR exact passkey eval; passkey-mix dataset | Current |
| PE-dominant DAPE diagnostic | `scripts/core_text_phases/phase11b_125m_dape.py` | 128-to-8K Geo/DAPE/EVQ diagnostic | Current code, seed scope limited |
| QuALITY support | `scripts/core_text_phases/phase21b_quality_eval_clean.py` | Gold-answer/option NLL eval with clean YaRN path | Supporting |
| LoRA support | `experiments/lora_evq_v2/` | LLaMA-3-8B LoRA exploratory support | Supporting only; missing Geo+LoRA control |
| Video support | `scripts/video_temporal/`, `results/video_dit/`, `results/supporting_video/` | DiT/video temporal support | Supporting only |
| Paper | `paper/main.tex`, `paper/sections/`, `paper/appendix/`, `paper/tables/` | NeurIPS manuscript | Wording tightened |
| Result artifacts | `data/curated/`, `results/` | Curated and compact result artifacts | Mixed provenance; see manifests |
| New audit helpers | `audit_rope_checkpoint.py`, `audit_training_artifacts.py`, `make_artifact_manifest.py` | Read-only artifact/checkpoint/data audits | Current |

## 3. 1B Anomaly Provenance

Current answer: the 1B anomaly is a real reported supporting result, but not a
same-configuration longer-training ablation and not JSON-backed in the compact
branch.

| Field | 8K/500M primary MLA | 4K/1B supporting MLA | Evidence |
| --- | --- | --- | --- |
| Code entrypoint | `run_gqa_evq_experiment.py` | same shared entrypoint | `run_350m_mla32_500m.sh`, `run_350m_4k_1b.sh` |
| Attention | MLA | MLA | launch wrappers pass `--attn_type mla` |
| Nominal model | 432M/350M-class MLA | 432M/350M-class MLA | paper supporting row; code tier `350m` |
| Train length | 8192 | 4096 | launch wrappers |
| Train tokens | 500M | 1B | launch wrappers/report |
| Seeds | 42, 43, 88 in eval JSON | seed 42 reported; wrapper planned 42,43,88 | current compact branch lacks exact JSON for all 1B seeds |
| Base | 500000 | 500000 | shared default/launch |
| `d_rope` | 32 | 32 | launch wrappers |
| code `head_dim` | 64 | 64 | `run_evq_sweep.py` tier config; eval scripts |
| `kv_lora_rank` | 256 | 256 | eval scripts/report |
| Tau | 1.414 for EVQ | 1.414 for EVQ | launch/report |
| Tau interpretation | empirical `d_eff=128` convention | same numeric convention, not same train length | paper wording now scoped |
| Dataset label | `fineweb-edu` cache label | `fineweb-edu` cache label | code/cache names |
| Actual data source | likely FineWeb-Edu if no external cache substitution; no data hash in compact tree | report/scripts indicate Pile+OpenWebText v1, with v2/v3/v4/v5 variants also present | data-prep scripts and report |
| Eval lengths | 8K,16K,20K,24K,28K,32K | 4K,8K,16K,32K and YaRN target/beyond-target lengths | portable MLA snapshot, paper supporting row |
| Scoring | full-sequence random chunks | same family for PPL; YaRN+FT script also uses random chunks | eval scripts |
| Current JSON | yes: `results/eval_3seeds_full_results.json` | no exact baseline/YaRN+FT JSON found | compact tree scan |

Judgment: do not call the 1B row a longer-training ablation of the primary MLA
experiment. It changes at least train length, data source/provenance, seed
coverage, and artifact quality. It is best treated as a limitation and
root-cause target for the old sparse MLA-32/base500K setting.

## 4. Code Path Audit

Core EVQ/MLA training is code-supported:

- `run_gqa_evq_experiment.py` dispatches MHA/GQA/MLA and applies RoPE only to
  `d_rope` for MLA.
- It constructs `inv_freq` with `rope_dim = d_rope` for MLA and `head_dim` for
  MHA/GQA.
- It saves `model.pt`, `inv_freq.npy`, and per-run `results.json`.
- It writes work-dir `summary.json`, but the compact tree does not package the
  external per-run 1B directory.

Issues found:

- Current run IDs are `350m_mla_tau...`, while historical eval scripts also
  expected `350m_tau...`. This was fixed in `eval_extended_3seeds.py` and
  `yarn_finetune_eval.py` by resolving both current and legacy names.
- Current 1B launch wrappers have machine-specific external roots and should be
  treated as historical launchers unless paired with sanitized artifact
  manifests.
- Result JSON schema from `run_gqa_evq_experiment.py` omits exact token math
  fields such as `actual_tokens_in_cache`, `used_tokens_by_train_loop`, and
  `dropped_tokens_from_batch_floor`. This is now auditable with
  `audit_training_artifacts.py`, but old result JSONs do not contain it.

## 5. RoPE / EVQ Implementation Audit

Current implementation behavior:

- Canonical EVQ schedule lives in `scripts/lib/rope/schedules.py`.
- `run_evq_sweep.py` imports the canonical schedule with midpoint grid enabled.
- `run_gqa_evq_experiment.py` uses `d_rope` for MLA RoPE construction, so only
  `d_rope/2` frequency pairs are quantized.
- `RotaryEmbedding.register_buffer("inv_freq", inv_freq)` means `inv_freq` is
  checkpointed by default.
- `load_state_dict` overwrites the registered buffer if the checkpoint contains
  `attn.rope.inv_freq`.

YaRN frequency-source conclusion:

- The earlier suspicion that YaRN was applied to a freshly rebuilt local
  geometric/EVQ table is likely false for checkpoints that contain
  `attn.rope.inv_freq`.
- The old behavior was implicit because local construction happened before
  `load_state_dict`. This was fragile.
- The eval scripts now require checkpoint `inv_freq`, clone the loaded buffer,
  print a short SHA256 hash, and apply YaRN to the checkpoint-loaded table.

Open issue:

- Exact checkpoint-level `inv_freq` hashes for the primary MLA and 1B rows are
  not packaged. Use `audit_rope_checkpoint.py` on recovered checkpoints.

## 6. Token-Count Audit

For the core from-scratch text scripts, token count is not
`steps x batch_size x seq_len x grad_accum x world_size`. The main training loop
is single-process and has no gradient accumulation/world-size multiplier.

Actual core formula:

```text
chunks = train_tensor.shape[0]
steps = chunks // batch_size
used_tokens = steps * batch_size * seq_len
```

Implications:

- If the cached tensor has exactly `floor(target_tokens / seq_len)` chunks,
  actual cached tokens are at most `target_tokens` and may be smaller by
  `target_tokens % seq_len`.
- The training loop drops up to `(batch_size - 1) * seq_len` cached tokens due to
  `steps = len(data) // batch_size`.
- This drop is small relative to 500M/1B but should be recorded for reviewer
  provenance.
- `results.json` currently records nominal configuration but not actual token
  math. This is an audit gap, not a confirmed bug in the training loop.

Data-prep caveat:

- Some 1B prep scripts save true Pile/OWT or other-source tensors but symlink
  them to `train_fineweb-edu_...pt` so the training script can load them through
  its existing cache-name convention. That makes the cache filename a label, not
  reliable dataset provenance.

Added helper:

```bash
python scripts/core_text_phases/audit_training_artifacts.py \
  --work-dir <recovered-work-dir> \
  --dataset fineweb-edu \
  --seq-len 4096 \
  --batch-size 12 \
  --train-tokens 1000000000
```

This prints train/val tensor metadata, actual/used/dropped tokens, and whether
per-run `results.json`, `model.pt`, and `inv_freq.npy` exist.

## 7. Eval Protocol Audit

PPL eval in the core MLA/MHA scripts is coherent for the metric it reports:

- `model.eval()` is called.
- `torch.no_grad()` is used.
- Label shift is standard causal LM: `model(chunk[:, :-1])` against
  `chunk[:, 1:]`.
- No padding mask is needed for prepacked fixed-length tensor chunks.
- Eval uses random contiguous chunks from flat validation tokens; this is
  full-sequence random-chunk PPL, not per-document PPL.

Known protocol separations:

- Table 2 full-sequence PPL and Table 3 per-document PPL differ by design; the
  table caption now states this.
- Passkey PK is teacher-forced NLL-gap retrieval unless explicitly marked AR
  exact. Captions and appendix wording now state this.
- QuALITY accuracy/NLL is supporting and capacity-limited; it should not lead
  rebuttal.

Open eval provenance gaps:

- Exact 1B JSON outputs from `yarn_finetune_eval.py` are absent.
- Batch size should not affect PPL except via numerical memory behavior, but no
  deterministic replay manifest proves this for 1B.
- Exact validation tensor hashes for 500M and 1B MLA rows are absent.

## 8. Dataset Provenance Audit

| Experiment family | Intended/current data | Evidence | Provenance status |
| --- | --- | --- | --- |
| 454M EVQ x YaRN | FineWeb-Edu with 10 percent synthetic passkey mix | curated JSON protocol | `A` for table values; raw per-seed data hashes missing |
| PE-dominant 128-to-8K | FineWeb-Edu | table caption and script | `A/B`; compact curated fallback for panel/table, not full sweep artifacts |
| 8K/500M primary MLA | FineWeb-Edu cache label, likely FineWeb-Edu if no external substitution | launch wrapper and loader | `A` result JSON; data hash missing |
| 4K/1B MLA anomaly | Pile+OpenWebText in v1 report/script; v2/v3 Pile+OWT variants; v4 FineWeb; v5 C4 scripts exist | prep scripts and report | `C`; exact artifact manifest missing |
| 2B/combined 4K | combines v1+v2 1B caches | prep script only | no cited result JSON found |
| 4B | no reviewer-grade current artifact found | search | `D` |
| LoRA 8B | LongAlign/LongAlpaca-style training and retrieval mix support | `experiments/lora_evq_v2/` | supporting; missing matched Geo+LoRA control |
| Video DiT | Moving-MNIST/UCF/video temporal data depending on row | video scripts/results | supporting-only, mixed artifacts |

High-risk dataset issue: the cache-name convention uses `fineweb-edu` even for
externally prepared non-FineWeb tensors in the 1B path. The report must describe
the true source via manifest, not infer it from filename.

## 9. Paper-Table Provenance Audit

| Paper artifact | Evidence source | JSON/result status | Reviewer-safe use |
| --- | --- | --- | --- |
| Table 2 EVQ x YaRN | `data/curated/table2_evq_yarn_454m_passkey_10pct.json` | `A` curated JSON | matched-scale substrate/range complementarity |
| Table 3 capability/passkey | same curated JSON | `A` curated JSON | teacher-forced retrieval robustness and per-document PPL caveat |
| Table 4 PE-dominant | `data/curated/fig3_extreme_128.json` plus `phase11b_125m_dape.py` | `A` for retained rows, `B` for full regeneration | seed-42 diagnostic for Geo/DAPE/EVQ |
| MLA table in appendix | `results/eval_3seeds_full_results.json` | `A` aggregate JSON | strongest scarce-channel stress test, with tau-convention caveat |
| Phase 11 leverage table | Phase 11 scripts/docs | `B` | supporting only |
| Multiscale raw PPL | mixed historical/current docs/results | mixed | supporting unless row is 454M FineWeb-Edu 3-seed |
| QuALITY Gold NLL | `results/core_text/phase21b/` | `A` supporting | downstream probability signal, not main task win |
| LoRA 8B | `experiments/lora_evq_v2/` and supporting result dirs | `B/C` | exploratory only |
| Video DiT | `results/video_dit/`, `results/supporting_video/` | mixed | supporting only |
| 1B/4K MLA | paper supporting row plus historical external report references | `C` | limitation/root-cause target only |

Manual-copy risks already fixed:

- MLA caption/prose no longer says EVQ+YaRN is best at every tested length; it
  now says every extrapolated length.
- Passkey table now states teacher-forced retrieval robustness.
- Rebuttal playbook overclaims were scoped down.

## 10. Most Likely Explanation of the 1B Anomaly

Ranked causes:

| Rank | Hypothesis | Evidence | Current confidence |
| --- | --- | --- | --- |
| 1 | Not a same-config ablation | 4K train length, different data, seed-42-only report, old MLA-32/base500K window | High |
| 2 | Sparse frequency-window failure in old MLA-32/base500K | Phase 22-23 report shows K=16/base500K is brittle and can reverse patterns | Medium-high |
| 3 | Dataset/source shift | 1B prep scripts use Pile/OWT and other variants under `fineweb-edu` cache labels | Medium-high |
| 4 | Tau convention mismatch | code `head_dim=64`, `d_rope=32`, tau=1.414; old prose called it `d_eff=d_head=128` | Medium-high |
| 5 | Seed outlier | only seed 42 reported for exact 1B row | Medium |
| 6 | Missing exact JSON/checkpoint provenance | no compact exact JSON or checkpoint hashes | Medium |
| 7 | Token-count mouthfeel error | core math likely sane but actual/used tokens not recorded | Low-medium |
| 8 | YaRN used wrong local freq | now unlikely for checkpoints with `inv_freq`; scripts patched to make explicit | Low |
| 9 | Geo/EVQ labels swapped | no direct evidence; checkpoint frequency audit needed | Low-medium until hashes recovered |
| 10 | Eval scoring mismatch | scripts appear same family of full-sequence random-chunk PPL; exact 1B JSON missing | Low-medium |
| 11 | True over-training effect | possible, but confounded by config/data/seed/provenance | Medium-low |

Bottom line: the anomaly is more likely a confounded sparse-window/config/data
result than clean evidence that EVQ generally disappears with more tokens.

## 11. Concrete Bugs or Inconsistencies Found

Fixed:

- Eval scripts now resolve both current and historical MLA run directory names.
- Eval scripts now explicitly require and hash checkpoint-loaded `inv_freq`
  before applying YaRN.
- MLA table/prose no longer overstates EVQ+YaRN at 8K.
- Passkey wording now says teacher-forced NLL-gap where applicable.
- Paper wording no longer describes MLA `tau=1.414` as code
  `d_eff=d_head=128`.

Still open:

- Exact 1B JSON/checkpoint/data hashes are missing.
- Direct MLA tau ablations for `d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` are
  missing.
- Token math/data-source metadata is not present in old `results.json`.
- 1B data cache labels can hide true dataset source without an external
  manifest.
- Tuned Geo+YaRN/LongRoPE2/CoPE/rebased-geometric controls remain absent.

## 12. Minimal Fixes and Scripts Added

Added or updated:

- `scripts/core_text_phases/audit_rope_checkpoint.py`
  - audits checkpoint `inv_freq`, schedule family, hashes, and frequency grid.
- `scripts/core_text_phases/audit_training_artifacts.py`
  - audits cached train/val tensors, token math, and per-run artifacts.
- `scripts/core_text_phases/make_artifact_manifest.py`
  - creates sanitized artifact manifests without absolute paths by default.
- `tests/test_yarn_checkpoint_inv_freq.py`
  - covers checkpoint-loaded frequency helpers and current/legacy run-id
    resolution.
- `tests/test_artifact_manifest.py`
  - covers sanitized manifest generation.
- `tests/test_training_artifact_audit.py`
  - covers token math and artifact listing.

Recommended result-manifest fields for recovered external runs:

- run label, method, seed, train length, eval lengths
- exact command or sanitized argv
- data source, cache filename, tensor shape/dtype/numel, tensor SHA256
- checkpoint filename and SHA256
- `inv_freq` SHA256 and first/last eight frequencies
- actual used tokens from `audit_training_artifacts.py`
- raw `results.json` and YaRN+FT JSON SHA256

## 13. Minimal Experiment Plan

Core question:

> Is EVQ a stable spectral-allocation advantage, or mainly an early-training
> optimization advantage under the old sparse MLA window?

Low-cost plan:

- Architecture: same MLA code path as primary, but log exact tensor/checkpoint
  manifests.
- Model: 50M or 125M MLA, because the question is mechanism/protocol, not scale.
- Train length: fixed 4096 and fixed 8192 as separate arms.
- Data: one locked public source, no fallback, no symlink relabeling.
- Base/d_rope: old `base=500000, d_rope=32` and one production-like
  `base=10000, d_rope=64` diagnostic arm.
- Methods: Geo, EVQ current tau, EVQ `tau=d_rope/sqrt(L)`, EVQ
  `tau=head_dim/sqrt(L)`, Geo+YaRN, EVQ+YaRN.
- Seeds: at least 3 for the old failure setting.
- Checkpoints: 25%, 50%, 75%, 100% token budget.
- Eval: same validation tensor, full-sequence random-chunk PPL at 1x/2x/4x,
  plus checkpoint `inv_freq` audit.

Ideal plan:

- Repeat the primary 432M/350M-class MLA at fixed config across 250M/500M/750M/1B
  tokens with identical data, seeds, and eval.
- Add tuned Geo+YaRN and rebased-geometric training-time baseline.
- Add exact 1B recovered artifact manifest if old runs are used.

Decision thresholds:

- If same-config 8K MLA keeps EVQ raw advantage through 1B with three seeds, the
  current 1B/4K anomaly is a config/data/window failure.
- If same-config 8K MLA raw EVQ converges to Geo but EVQ+YaRN stays better, the
  claim should emphasize substrate/range composition, not raw durability.
- If tuned Geo+YaRN or rebased-geometric closes the gap, the claim must shrink
  to "one effective closed-form allocation" rather than uniqueness of the cosh
  shape.
- If `d_rope/sqrt(L)` or code-`head_dim/sqrt(L)` matches current tau, the MLA
  convention can be simplified; if they fail, the paper must keep `d_eff=128` as
  empirical.

## 14. Current Paper Disposition

Recommended disposition for the 1B row:

- Keep only as a limitation/root-cause target if exact artifacts remain missing.
- Do not use as primary support.
- Do not call it same-config longer-training evidence.
- If a reviewer asks for reproducibility and no JSON/checkpoint manifest can be
  recovered, remove numeric 1B claims from rebuttal and cite it only as an
  internal diagnostic that motivated further ablations.

Recommended claim calibration:

- Keep: finite spectral budget, training-time allocation as a design axis,
  matched-scale EVQ x YaRN complementarity, 3-seed 8K/500M MLA stress-test
  evidence.
- Do not claim: universal long-context SOTA, tuned scaler dominance, LoRA
  conclusiveness, or production-scale durability.

## 15. Original Prompt Coverage

| Requested block | Current coverage |
| --- | --- |
| Repository map | This report section 2 |
| 1B anomaly provenance | Sections 3 and 10 |
| Code path audit | Section 4 |
| RoPE/EVQ implementation audit | Section 5 plus `audit_rope_checkpoint.py` |
| Token-count audit | Section 6 plus `audit_training_artifacts.py` |
| Eval protocol audit | Section 7 |
| Dataset provenance audit | Section 8 |
| Paper-table provenance audit | Section 9 plus `RESULT_PROVENANCE_MANIFEST.md` |
| Most likely explanation | Section 10 |
| Concrete bugs/inconsistencies | Section 11 |
| Minimal fixes/scripts | Section 12 |
| Minimal experiment plan | Section 13 |
| Paper disposition | Section 14 |
