# ICLR 2027 research index

This is the single durable routing layer for the active `paper-2027/`
manuscript. It tells an agent what to read, which file owns each claim, and
which material is only a plan, audit, or external review.

## Start here

1. [`../../AGENTS.md`](../../AGENTS.md) — stable scientific, submission, safety,
   and workspace rules.
2. [`../HANDOFF.md`](../HANDOFF.md) — live manuscript/worktree state and the
   only current action queue.
3. [`ICLR2027_RESEARCH_SYNTHESIS_20260819.md`](ICLR2027_RESEARCH_SYNTHESIS_20260819.md)
   — implemented claim architecture.
4. [`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`](EXACT_RANGE_151M_3SEED_RESULT_20260820.md)
   — raw-hash-receipted three-training-seed fixed-support result.
5. [`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)
   — canonical theory, finite-K counterexamples, and 50M crossing.
6. [`attention-aware-retrofit/README.md`](attention-aware-retrofit/README.md)
   — current mature-checkpoint retrofit results, negative routes, and receipts.

Do not start from the newest date, an external review, or a preflight.

## Current scientific architecture

The manuscript separates sampled support from normalized interior allocation:

\[
x_k=-\log\omega_k=a+Rz_k.
\]

Its central claim is that `z` is a separately identifiable training-time
variable even when `(a,R)` is fixed; it changes full sin/cos subspace geometry
and trained behaviour, while model weights co-adapt to the table used during
training. EVQ-Cosh is one closed-form intervention on this axis, not a universal
optimum.

The new frozen-checkpoint case study is internal and does not change that
claim. It shows that fixed-support `z` remains consequential on OLMo and Qwen,
but a nearest label-free ramp matches the detailed derived profile. The
defensible novelty is fixed-support identification and model-relative split
derivation, not merely producing a non-geometric table or a new YaRN family.

## Directory map

| Path | Role |
| --- | --- |
| research-root dated owners | central synthesis, exact-range owner, canonical theory, and early theory architecture |
| [`attention-aware-retrofit/`](attention-aware-retrofit/) | mature retrofit results, receipts, analyses, theory, and preflights, each in a separate subdirectory |
| [`audits/`](audits/) | internal manuscript/theory/evidence audits and falsified internal measures |
| [`external-reviews/`](external-reviews/) | untrusted independent-model recomputations and proposals |
| [`three_completions/`](three_completions/) | supplementary derivations, verification scripts, and rendered internal note |

Additional durable theory/supporting files at research root:

- [`ICLR2027_THEORY_ARCHITECTURE.md`](ICLR2027_THEORY_ARCHITECTURE.md) — early
  theory design exploration; current synthesis and canonical report win on
  conflict;
- [`three_completions/README.md`](three_completions/README.md) — index for the
  long-form derivation/verification bundle.

## Canonical evidence routing

| Question | Canonical source | Maximum role |
| --- | --- | --- |
| Full sin/cos geometry, collision, stable-rank identity | [`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) | main theory; static basis, not LM-quality predictor |
| Pure fixed-support interior-allocation identification | [`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`](EXACT_RANGE_151M_3SEED_RESULT_20260820.md) plus M4 historical owner | main causal experiment |
| Exact frozen Q/K transplant obstruction | `../../rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` | exact impossibility for fixed static maps, not all approximate adapters |
| Weights/table co-adaptation | canonical full-RoPE report plus `../../scripts/analysis/attention_fisher_50m_probe.py` | diagnostic; keep separate from exact-range estimand |
| Scarce-budget systems flagship | `../../data/curated/table18_mla_3seed_aggregate.json` | three-seed 432M MLA result |
| Training-stage and scale persistence | `../../docs/exp/2026-03-06_phase15_750m_2k_to_4k_continue_results.md` and `../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` | 750M continuation and 1.485B trend |
| Matched mature phase exposure | `../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md` | protocol-specific capability evidence |
| Mature fixed-support `z` controls and 151.9M crossing | [`attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) | internal causal case study; not a new operator-family claim |
| Zero-training practical session policy | [`attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md`](attention-aware-retrofit/results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md) | internal OLMo deployment/capability evidence |
| LeRoPE related-work facts | `../../rebuttal/rebuttal_0723/theory_results/LEROPE_CONCURRENT_WORK_NOTE_20260728.md` plus primary paper | positioning only |
| Failed LeRoPE profile oracle | [`audits/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md`](audits/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md) | internal negative |
| Failed attention-measure ordering gate | [`audits/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md`](audits/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md) | internal negative |

## Protocol boundaries

- Exact-range and M4 own pure interior-allocation identification.
- The 50M and 151.9M crossings own co-adaptation diagnostics.
- Mature frozen-checkpoint RULER is task-family adaptation, not unseen-task
  transfer or from-training evidence.
- NLL/PPL, teacher-forced NLL gap, strict generation, token F1, exact match,
  RULER, and causal source-use are distinct endpoints.
- Row bootstraps condition on a fixed checkpoint and task set; they are not
  model-, task-population-, or training-seed uncertainty.
- A preregistration, script, checkpoint inventory, command, or launch log is not
  a result.

## Current stop list

- Do not sweep another OLMo/Qwen table, gain, beta, rank, step count, or RULER
  cell from the current results.
- Do not resume the CE-only far-pass residual route.
- Do not revive revoked source-selection protocols whose target is absent from
  model input.
- Do not promote the old aliased Qwen `0.6175` as the corrected profile; the
  valid corrected 128K result is `0.5400`.
- Do not describe the mature profile as a new interpolation family or universal
  optimum.

If the frozen case study is promoted, the highest-value missing external test
is a preregistered same-support geometric/ramp/derived comparison on a natural
OLMo task and a natural Qwen task. It is not currently authorized.

## Placement rules

| New material | Destination |
| --- | --- |
| live mutable state | only `../HANDOFF.md` |
| central paper-facing owner | research root only when it joins the main claim architecture |
| mature retrofit result | `attention-aware-retrofit/results/` |
| mechanism/falsification | `attention-aware-retrofit/analysis/` |
| preregistration or revoked protocol | `attention-aware-retrofit/preflights/` |
| compact receipt | the relevant `evidence/` directory |
| manuscript/theory audit | `audits/` |
| external-model review | `external-reviews/<source-date>/` |

Raw checkpoints, evaluation rows, caches, server paths, and secrets stay
outside `paper-2027/`. External reviews never supersede canonical owners.
