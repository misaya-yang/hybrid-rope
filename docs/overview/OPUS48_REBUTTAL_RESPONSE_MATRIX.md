# Opus 4.8 Rebuttal Response Matrix

Purpose: convert the Opus 4.8 audit into rebuttal-ready response strategy. This
is not a claim-upgrade document. Use it to decide whether to defend, concede,
scope, or request/recover additional evidence.

## Response Principles

- Lead with the finite-spectral-budget mechanism claim, not universal
  long-context SOTA.
- Separate training-time allocation from inference-time range scaling.
- Treat matched-scale EVQ x YaRN as complementarity, not tuned-scaler dominance.
- Name PK as teacher-forced NLL-gap unless a row explicitly reports
  autoregressive exact match.
- Treat the 1B/4K MLA row as a limitation/root-cause target until exact JSON,
  checkpoint, and data manifests are recovered.
- Do not defend supporting LoRA/video/progressive rows as primary evidence.

## Matrix

| Reviewer attack | Response posture | Safe response | Evidence to cite | Do not say | If pressed |
| --- | --- | --- | --- | --- | --- |
| "This is just a simple low-frequency schedule." | Scope + partial defense | The contribution is a closed-form, zero-parameter training-time allocation derived from a stated surrogate and validated as a mechanism axis. It is not claimed to dominate every possible schedule. | `docs/overview/OPUS48_REVIEW_AUDIT_CHECKLIST.md` O48-01; `scripts/lib/rope/schedules.py` | EVQ is uniquely optimal among all schedules. | Concede missing rebased-Geo/fixed-interpolation control or add it. |
| "The variational theory is surrogate-fitted." | Concede + defend honesty | The variational result is exact for the broadband surrogate; the paper uses functional validation rather than claiming full-attention derivation. | `paper/sections/03_theory.tex`; `docs/overview/PAPER_DESCRIPTION_AUDIT.md` | Derived from full transformer attention. | Point to surrogate validation and keep claim mechanistic. |
| "Tau scaling rests on diffuse attention." | Concede/scope | The EVQ-Cosh shape comes from the surrogate; the tau rule is an operating default/basin selector, not a global optimum. | `docs/overview/OPUS48_ISSUE_RESOLUTION_LEDGER.md` O48-04/O48-05 | `tau=d_eff/sqrt(L)` is theoretically forced. | Offer `L_eff^J` measurement as future/additional evidence. |
| "PK is a soft metric." | Concede + clarify | Correct: Table 2/3 PK is teacher-forced NLL-gap retrieval. The paper uses it as a PE diagnostic, not as AR task accuracy. | `paper/sections/05_experiments.tex`; `paper/tables/table3_capability_passkey.tex` | PK is AR exact match. | Add AR exact results only if explicitly measured. |
| "EVQ alone is modest; the headline is EVQ+YaRN." | Defend composition | That is the point of the matched-scale test: EVQ changes the training substrate on which YaRN has higher leverage. | `data/curated/table2_evq_yarn_454m_passkey_10pct.json`; `docs/overview/RESULT_PROVENANCE_MANIFEST.md` M1 | EVQ alone solves long context. | Keep wording to matched-scale substrate/range complementarity. |
| "No tuned Geo+YaRN/LongRoPE2 baseline." | Concede scope | The current table tests fixed matched-scale leverage, not tuned scaler SOTA. | `docs/overview/OPUS48_AUDIT_CONTROL_CENTER.md`; `docs/overview/OPUS48_ISSUE_RESOLUTION_LEDGER.md` O48-08/O48-22 | We beat all tuned range scalers. | Add tuned controls or concede this as a limitation. |
| "Primary II is seed 42 only." | Concede + scope | The Geo/DAPE/EVQ PE-dominant rows are seed-42 diagnostic rows; only the learnable-tau row is multi-seed. | `docs/overview/RESULT_PROVENANCE_MANIFEST.md` M2; `paper/tables/table4_pe_dominant.tex` | The whole PE-dominant table is 3-seed. | Add seeds or keep diagnostic-only. |
| "MLA tau convention is underived." | Concede + narrow | Current code uses `head_dim=64`, `d_rope=32`, and `tau=1.414`; the paper now calls this an empirical `d_eff=128` operating convention. | `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md`; `paper/appendix/a3_supporting_results.tex` | The convention follows from code `head_dim` or `d_rope`. | Run direct `tau=d_rope/sqrt(L)` and code-`head_dim/sqrt(L)` ablations. |
| "The 1B MLA row reverses EVQ." | Treat as limitation | Yes, it is a real limitation signal. It is not same-config longer training: train length, data/provenance, seed coverage, and artifact quality differ from the 8K/500M primary MLA row. | `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` sections 3 and 10; `docs/overview/RESULT_PROVENANCE_MANIFEST.md` M4 | The 1B run proves durability. | If exact artifacts remain missing, remove numeric 1B claims from rebuttal. |
| "Maybe YaRN used the wrong frequency table." | Defend code fix + remaining audit | The old aliasing behavior was implicit; current eval scripts now require checkpoint `inv_freq`, clone the loaded table, print hash, and apply YaRN to that table. | `scripts/core_text_phases/eval_extended_3seeds.py`; `scripts/core_text_phases/yarn_finetune_eval.py`; `tests/test_yarn_checkpoint_inv_freq.py` | Exact 1B checkpoint frequencies have already been audited. | Run `audit_rope_checkpoint.py` on recovered checkpoints. |
| "Token counts may be nominal." | Defend code math + note gap | Core scripts are single-process; used tokens are `floor(chunks/batch_size)*batch_size*seq_len`. Old JSONs do not record this, so recovered runs need audit manifests. | `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` section 6; `scripts/core_text_phases/audit_training_artifacts.py` | Nominal 500M/1B filenames prove exact consumed tokens. | Run artifact audit on recovered run dirs. |
| "Dataset provenance is mixed." | Concede + require manifest | For 1B, cache labels can hide true source; the row stays supporting until a sanitized data manifest is available. | `docs/overview/DATA_PREPARATION.md`; `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` section 8 | 1B used the same data as primary MLA. | Provide tensor hashes and source manifest. |
| "LoRA lacks Geo+LoRA." | Concede | LoRA is supporting/exploratory only and is not used to carry the main claim. | `paper/appendix/a4_supporting_experiments.tex`; `docs/overview/PAPER_DESCRIPTION_AUDIT.md` | LoRA proves EVQ-specific 8B transfer. | Add matched Geo+LoRA control. |
| "Video/DiT is conditional." | Concede/scope | Video evidence is supporting and mechanism-consistent in dead-channel regimes; it is not universal video superiority. | `docs/overview/PAPER_CLAIMS_MAP.md`; `docs/overview/OPUS48_ISSUE_RESOLUTION_LEDGER.md` O48-18 | EVQ universally wins for video. | Promote only if packaged multi-seed evidence is available. |
| "Downstream accuracy is weak." | Scope | Downstream accuracy is a non-regression/capacity check; the primary endpoints are PE diagnostics. | `paper/sections/05_experiments.tex`; `docs/overview/OPUS48_FORENSIC_AUDIT_REPORT.md` | Downstream SOTA. | Lead with PPL/PK/gold-NLL diagnostics. |

## Claim Disposition

| Claim family | Rebuttal disposition |
| --- | --- |
| Finite spectral budget and allocation axis | Defend. |
| EVQ-Cosh closed-form zero-parameter allocation | Defend, with surrogate scope. |
| Matched-scale EVQ x YaRN complementarity | Defend. |
| 8K/500M 3-seed MLA scarce-channel result | Defend with tau-convention caveat. |
| PE-dominant 128-to-8K result | Keep diagnostic and seed-scoped. |
| 1B/4K MLA row | Limitation/root-cause target only. |
| LoRA/video/progressive/750M rows | Supporting only. |
| Tuned-scaler or universal long-context superiority | Do not claim. |

## Minimal Rebuttal Checklist

Before sending any rebuttal paragraph:

- [ ] Does it cite only rows that are code/result-supported at the needed level?
- [ ] Does it name PK as teacher-forced NLL-gap when relevant?
- [ ] Does it avoid saying EVQ replaces YaRN/LongRoPE?
- [ ] Does it avoid using 1B/4K as primary support?
- [ ] Does it keep MLA `d_eff=128` empirical?
- [ ] Does it concede missing tuned-scaler, rebased-Geo, tau-ablation, and
  Geo+LoRA controls when relevant?
