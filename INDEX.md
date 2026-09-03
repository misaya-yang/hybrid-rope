# INDEX — theory, evidence, code, and durable agenda

- **Updated:** 2026-09-02
- **Role:** the repository's only durable index.
- **Does not own:** rules (`AGENTS.md`), volatile state
  (`paper-2027/HANDOFF.md`), or historical narrative
  (`paper-2027/research/history/TIMELINE.md`).

Authority order: **rules > index > state > historical summary**. Facts and
numbers always defer to the canonical/raw owner linked below.

## 0. Cold-start snapshot

Current lifecycle:

`PURE_Z_LONG_SIGNAL_ESTABLISHED / NATURAL_QA_AND_NATIVE_LONG_JOINT_UNSOLVED /
NO_SOTA / GPU_METHOD_DEVELOPMENT_STOPPED`

What is established:

1. At fixed sampled support, normalized interior allocation `z` is a causal
   training-time variable in the 151.9M three-seed study.
2. Full-RoPE static geometry supports phase-invariant redundancy/effective
   dimension claims, not LM-quality or extrapolation ranking.
3. Frozen mature checkpoints are sensitive to the ordered pairing between
   learned rotary subspaces and frequencies/dilations; same-multiset
   permutations can collapse.
4. Static pure-`z` interventions improve long NLL, RULER/NIAH, and
   source-conditioned answer likelihood in tested protocols, but no tested arm
   jointly solves Native retention and natural autoregressive QA.
5. Normalized pair index is the best-supported tested cross-`K` transport
   coordinate, not a canonical or universal law.

What is not established: SOTA, a universal/unique optimum, continuous basin
bounds, arbitrary-scale flow, K causality, natural-generation QA conversion,
or a single Native-compatible and long-capable direction.

For chronology, read
[`TIMELINE.md`](paper-2027/research/history/TIMELINE.md). For immediate work and
authorization, read [`HANDOFF.md`](paper-2027/HANDOFF.md).

## 1. Theory map

### 1.1 Paper-level foundations

| Object | Supported conclusion | Owner |
| --- | --- | --- |
| Finite full-RoPE basis | Exact block-whitened stable-rank identity; phase/basis-invariant canonical collision | [`FULL_ROPE...`](paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) |
| Low-frequency collapse | Slow bands become redundant in the stated static/softmax metrics | same owner |
| Causal coordinates | `x_k = a + R z_k` separates support `(a,R)` from normalized allocation `z` | [`ROPE_CAUSAL_VARIABLES...`](paper-2027/research/foundations/ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md) |
| Frozen transplant obstruction | Exact position-independent invertible Q/K compensation requires equal frequency multisets up to sign/permutation | [`OLMO2_POSTHOC...`](rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md) |
| EVQ-Cosh | Closed-form zero-learned-parameter construction; unique only for its stated convex surrogate | [`ICLR2027_RESEARCH_SYNTHESIS...`](paper-2027/research/foundations/ICLR2027_RESEARCH_SYNTHESIS_20260819.md) |

Claim ceilings remain in `AGENTS.md`. In particular, static geometry is not a
behavioural predictor and EVQ-Cosh is not a unique or universal LM optimum.

### 1.2 Mature-checkpoint internal theory

| Object | Current status | Owner |
| --- | --- | --- |
| Ordered rotary coupling | Same frequency multiset with different slot assignment can collapse; unordered `pi(omega)` is insufficient | [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831`](paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md) |
| Cross-`K` coordinate | Normalized index has the strongest tested breadth; physical-coordinate privilege is closed | [`K128 confirmation`](paper-2027/research/attention-aware-retrofit/results/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md) + [`K32 confirmation`](paper-2027/research/attention-aware-retrofit/results/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md) |
| Scale flow | Log-frequency flow is a useful coordinate/problem statement; current data do not identify a canonical vector field or curvature | [`FIRST_PRINCIPLES...`](paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md) |
| Native/long basin barrier | Tested log-start and Native-start arms occupy opposite sides of the joint objective | [`COMMON_DIRECTION...`](paper-2027/research/attention-aware-retrofit/theory/COMMON_DIRECTION_FEASIBILITY_AND_BASIN_BARRIER_THEORY_20260902.md) |

Historical theory plans live in
[`paper-2027/research/foundations/`](paper-2027/research/foundations/) and
[`paper-2027/research/archive/`](paper-2027/research/archive/). They do not own
current scheduling.

## 2. Evidence map

### 2.1 Main causal identification

| Estimand | Result role | Canonical owner |
| --- | --- | --- |
| Fixed-support interior `z` during training | 151.9M, three paired training seeds; OOD direction consistent, small in-window cost | [`EXACT_RANGE_151M_3SEED_RESULT_20260820`](paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) + [JSON](paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json) |
| Target-matched boundary | Same pair reverses ordering when support is retargeted; support and allocation interact | same owner §4 |
| M4 exact-range factorial | Fixed endpoints/span across bases, train lengths, head dimensions, and seeds; allocation active, Cosh not unique | [`M4_EXACT_RANGE_FACTORIAL_RESULT_20260726`](rebuttal/rebuttal_0723/theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md) |
| Weights×table co-adaptation | 50M 2×2 frozen table/weights crossing; diagnostic, not pure-`z` identification | [`FULL_ROPE...`](paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) |

Training seeds, evaluation rows, structural configurations, and single matched
trajectories are different uncertainty units. Never count anchors/tasks as
training seeds.

### 2.2 Scale and systems breadth

| Evidence | Role | Owner |
| --- | --- | --- |
| 432M MLA, three seeds | Scarce-channel system breadth | [`table18_mla_3seed_aggregate.json`](data/curated/table18_mla_3seed_aggregate.json) |
| 750M 2K→4K continuation | Training-stage persistence; PPL/retrieval/AR endpoints remain distinct | [`2026-03-06 report`](docs/exp/2026-03/2026-03-06_phase15_750m_2k_to_4k_continue_results.md) |
| 1.485B same initialisation | Pretraining-scale comparison; same scientific recipe, not bitwise paired execution | [`OLMO2_1B_RELEASED_ROPE_BASELINE_20260725`](rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md) |
| 1.485B phase adaptation | Protocol-specific task-family capability | [`OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729`](rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md) |
| 8B adaptation | Adaptation/capability evidence, not pretraining-scale causality | [`EVQ_8B_ADAPTATION_EVIDENCE_20260724`](rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md) |
| Video-DiT seed 42 | One matched cross-modal comparison | [`VIDEO_DIT...`](paper-2027/research/evidence/VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md) + [JSON](data/curated/video_dit_seed42_head_to_head_20260826.json) |

These owners are co-equal breadth/persistence evidence under their own
protocols. They do not replace the fixed-support causal owner.

### 2.3 Mature-checkpoint retrofit

| Question | Verdict | Owner |
| --- | --- | --- |
| Does same-support `z` matter at frozen weights? | Yes in tested OLMo/Qwen RULER cells; detailed profile often tied with nearest ramp | [`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823`](paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) |
| Does it persist on fresh natural text? | Length-conditional NLL effect; not a universal predictor | [`FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md) |
| Can one static table pass declared OLMo gates? | One s4 table passes tested 1x gates and improves longer endpoints; same-multiset permutation fails | [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831`](paper-2027/research/attention-aware-retrofit/results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md) |
| Does two-parameter coupling transfer? | Long Qwen/OLMo behaviour retained; strict Native gate slightly missed | [`LOW_DIM_COUPLING_GPU_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md) |
| What explains the old Gemma K128 negative? | Reference/request scale correction recovers 8K/16K; physical-vs-index causality unresolved in that owner | [`REFERENCE_CORRECTED_K128_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/REFERENCE_CORRECTED_K128_RESULT_20260901.md) |
| Does index tilt replicate at K128? | Yes for the registered N80 contrast; no universality/K-causality claim | [`K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md) |
| Does K32 index retain breadth? | Full RULER-13 improves at 64K with 32K retention pass; task families remain mixed | [`K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901`](paper-2027/research/attention-aware-retrofit/results/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md) |
| Does long signal convert to natural QA? | Long NLL/source-use improves; natural generation QA and EOS conversion remain unresolved | [`ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902`](paper-2027/research/attention-aware-retrofit/results/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md) |
| Do headwise clocks solve the joint objective? | Improve long-task Pareto; tested starts do not jointly pass Native retention and long QA | [`HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902`](paper-2027/research/attention-aware-retrofit/results/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md) |

The 2026-09-02 Qwen natural/QA aggregate lacks recovered remote raw JSON/JSONL;
it remains internal decision evidence. No silent rerun is implied.

### 2.4 Endpoint and identity boundaries

- `Geo`, `Native`, `FMRoPE`, anchored EVQ-Cosh, EVQ-Cosh, repository
  YaRN-style, and the MLA wavelength-blend operator retain the locked meanings
  in `AGENTS.md`.
- NLL/PPL, answer-token NLL, teacher-forced NLL-gap, strict generation, token
  F1, exact match, RULER/NIAH, QA, causal source use, adaptation, and transfer
  are separate evidence tiers.
- Pure-`z` attribution holds support/base, operator, checkpoint or training
  contract, gain, routing, data/rows, decoder, and endpoint fixed.
- A preflight is protocol history, not evidence that code ran. A result report
  without raw artifacts remains report-backed at most.

## 3. Closed and unresolved routes

### 3.1 Falsified/closed — do not repeat

| # | Closed object | Evidence | Owner |
| ---: | --- | --- | --- |
| 1 | cosine-only collision kernel as full-RoPE ranker | ordering counterexample | [`FULL_ROPE...`](paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) §4.3 |
| 2 | collision/logdet minimization as extrapolation target | Fourier-comb periodic aliasing | same owner §4.2–4.3 |
| 3 | attention-Fisher `kappa_att` ordering | contradictory orderings | [`KAPPA audit`](paper-2027/research/audits/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md) |
| 4 | LeRoPE `w^(1/3)` structural-curvature oracle | failed profile prediction | [`LeRoPE oracle audit`](paper-2027/research/audits/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md) |
| 5 | arcsine conjecture | equal-stiffness free optimum is not U-shaped | [`optimization_notes`](paper-2027/research/three_completions/optimization_notes.md) |
| 6 | direct attention-distance map without phase kernel | registered negative | [`EXPERIMENT_REPORT_20260821`](paper-2027/research/attention-aware-retrofit/results/EXPERIMENT_REPORT_20260821.md) |
| 7 | `D*` as retrofit design target | measured rank sign opposite | [`RETROFIT_AXIS_FALSIFICATION_20260822`](paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md) |
| 8 | coverage residual selector | negative measured rank relation | same owner |
| 9 | phase-risk / “wrapped channels are safe” | zero rank relation + deterministic counterexample | same owner |
| 10 | direct-`z` two-document calibration | per-row gate failure | [`DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md) |
| 11 | two analytic Native-support static tables | improve 2× while damaging 1× | [`ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824`](paper-2027/research/attention-aware-retrofit/results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md) |
| 12 | tested continuous-boundary-slope operator | 8K/16K capability zero in its harness | [`ZERO_TRAINING_MECHANISM_AND_CEILING_20260826`](paper-2027/research/attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md) §2; only this implementation is closed |

Static selectors above share the same failure class: a scalar functional of one
shared table cannot see the dominant table×weights interaction. A future route
must state how it escapes that class before work begins.

### 3.2 Unresolved — not a negative and not a queue

| Object | Status | Owner |
| --- | --- | --- |
| phase-isotropy / pair-volume / min-eigenvalue | `SCREEN_UNRESOLVED` | [`PHASE_ISOTROPY...`](paper-2027/research/attention-aware-retrofit/results/PHASE_ISOTROPY_50M_M4_RESULT_20260824.md), [`PHASE_ALLOCATION...`](paper-2027/research/attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md) |
| Native-compatible/long-capable bridge | no tested common direction | headwise result + basin-barrier theory owners above |
| per-head adapter rank allocation | historical unrun diagnostic | no completed owner; no action |

## 4. Code and directory map

### 4.1 Code entrypoints

| Need | Location | Boundary |
| --- | --- | --- |
| RoPE implementation | [`scripts/lib/rope/`](scripts/lib/rope/) | implementation authority; not result owner |
| Historical from-scratch chain | [`scripts/core_text_phases/`](scripts/core_text_phases/) | completed phase runners; no automatic queue |
| Reusable CPU diagnostics | [`scripts/analysis/`](scripts/analysis/) | computed numbers need explicit owner/conventions |
| Static third-axis rank diagnostic | [`scripts/analysis/third_axis_ceiling.py`](scripts/analysis/third_axis_ceiling.py) | algebraic search only; not an LM-quality owner |
| Current evaluation utilities | [`scripts/eval/`](scripts/eval/) | launchers require live protocol/authorization |
| Data preparation | [`scripts/data/`](scripts/data/), [`scripts/data_prep/`](scripts/data_prep/) | data builders and receipts |
| Supporting/standalone experiments | [`experiments/`](experiments/) | protocol-specific supporting evidence |
| Figures | [`paper-2027/figs/`](paper-2027/figs/), [`scripts/figures/`](scripts/figures/) | active ICLR vs historical NeurIPS figures |
| Tests | [`tests/`](tests/) | source/protocol/navigation gates |
| Supplement packager | [`scripts/package_supplement.py`](scripts/package_supplement.py) | run from root with `--profile iclr2027` |
| Theory falsification benchmark | [`falsification_benchmark/`](falsification_benchmark/) | completed blind evaluator; not a new experiment queue |

### 4.2 Directory ownership

| Path | Owns | Rule |
| --- | --- | --- |
| `paper-2027/` | only active manuscript and live handoff | build/edit only this paper |
| `paper-2027/research/foundations/` | durable theory/causal documents | facts still defer raw owner |
| `paper-2027/research/evidence/` | compact paper-level result owners | no mature-retrofit duplication |
| `paper-2027/research/attention-aware-retrofit/results/` | completed mature-checkpoint owners | result/claim ceilings only |
| `.../evidence/` | compact receipts | no machine paths/secrets |
| `.../analysis/` | mechanism/falsification analysis | never an action queue |
| `.../preflights/` | historical preregistrations/plans | completion does not make them current |
| `.../theory/` | mature-checkpoint theory history | current agenda stays here in INDEX |
| `paper-2027/research/history/` | chronological ledgers | non-authoritative summaries |
| `paper-2027/research/archive/` | retired plans/reviews/logs | frozen historical inputs |
| `analysis/full_rope_audit/` | raw/static full-RoPE audit bundle | historical reproduction; canonical interpretation is in `foundations/` |
| `docs/exp/YYYY-MM/` | NeurIPS-era reports by execution month | historical/report-backed unless promoted |
| `docs/theory/`, `docs/tau_algor/`, `docs/archive/` | early and superseded theory derivations | read-only history; current theory is §1 |
| `docs/overview/` | provenance/reproduction/terms/Blackwell profile | maintained historical infrastructure |
| `experiments/` | standalone supporting model/protocol packages | code and protocol assets, never a result by location |
| `rebuttal/rebuttal_0723/` | official review and July evidence owners | historical layer, not queue |
| `data/curated/` | small portable reviewer assets | tracked with provenance |
| `paper_experiments/` | symlinked historical paper-code workspace | integrity view only; canonical code stays in `scripts/`/`experiments/` |
| `research_notes/` | exploratory plans and analysis bundles | legacy, non-authoritative |
| `nonuniform-alloc/` | protected legacy allocation study | closed historical branch; do not modify or schedule from it |
| `results/` | raw/local/historical result tree | ignored by default; not automatic evidence |
| `research_notes/` | legacy exploratory bundles | not current routing |
| `internal/` | private NeurIPS-era archive | protected; do not modify without request |
| `paper/` | immutable NeurIPS 2026 baseline | never edit, compile, move, or regenerate |

### 4.3 Placement rules

- New paper-level theory: `paper-2027/research/foundations/`.
- New compact paper-level result owner: `paper-2027/research/evidence/`.
- Mature retrofit result/receipt/analysis/preflight/theory: the corresponding
  `attention-aware-retrofit/` subfolder.
- Historical experiment report: `docs/exp/YYYY-MM/YYYY-MM-DD_slug.md`.
- Reusable code: `scripts/analysis/`, `scripts/data/`, or `scripts/eval/` by
  role; do not create a new root tool tree.
- Volatile state: update only `paper-2027/HANDOFF.md`.
- Never create `REPO_MAP.md`, another root handoff, another index, or a second
  action queue.

## 5. Durable agenda

### 5.1 Submission work — active priority

1. Decide which completed mature-checkpoint findings change the manuscript
   claim set; modify reviewer-facing text only when an owner supports it.
2. Recover the missing 2026-09-02 Qwen raw results if a surviving copy exists.
   If not, keep the aggregate internal; do not silently rerun or promote it.
3. Rebuild and validate the curated supplement on the work machine.
4. Complete owner-by-owner number review, author metadata freeze, and current
   venue-validity checks before upload.

Immediate progress and authorization live only in the handoff.

### 5.2 Research — stopped unless explicitly reopened

No GPU method-development experiment is active. If the user separately reopens
research, a candidate must first provide a CPU-identifiable, nontrivial
cross-scale or cross-task prediction for a common Native-compatible and
long-capable direction, explain how it escapes the closed scalar-selector
class, and bind exact owners/stop conditions before any paid run.

Do not restart from another ramp, cutoff, gain, unrestricted per-frequency
field, historical preflight, or parameter sweep. A numerical static search is
only best-found under its stated support/measure/optimizer/restarts and is not a global or behavioural ceiling.

## 6. Historical and machine routing

- Full chronology: [`paper-2027/research/history/TIMELINE.md`](paper-2027/research/history/TIMELINE.md)
- NeurIPS experiment reports: [`docs/exp/`](docs/exp/)
- July rebuttal/evidence history: [`rebuttal/rebuttal_0723/README.md`](rebuttal/rebuttal_0723/README.md)
- August plans/reviews: [`paper-2027/research/archive/`](paper-2027/research/archive/)
- Current state/actions: [`paper-2027/HANDOFF.md`](paper-2027/HANDOFF.md)

The work machine owns `aidemo`, PyTorch/pytest, GPU execution, packaging, and
final release validation. The low-configuration personal PC owns reading,
documentation, planning, lightweight standard-library checks, and local
LaTeX/Tectonic iteration. Never copy credentials, private paths, raw
checkpoints, caches, or ignored evidence into Git to bridge machines.
