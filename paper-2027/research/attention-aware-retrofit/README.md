# Mature-checkpoint allocation research index

This directory stores mature-checkpoint result owners, receipts, analyses, and
historical preflights. It does not own a second research agenda. Current priority
and order live only in [`../../../INDEX.md`](../../../INDEX.md) §0 and §6.

## Current question

`K32_MATCHED_S_PARETO / K128_SCREEN_UNRESOLVED_LONG_NEGATIVE /
P3_REJECTED`

The completed low-dimensional experiment now separates two effects:

- a frozen two-parameter `G_4(x)` preserves OLMo and Qwen long behavior;
- the same table misses the strict Native operating point by a small amount on
  both checkpoints;
- Qwen's self-profile, but not the transported OLMo 64-point residual, restores
  Qwen 32K retention, locating the remaining problem in checkpoint-specific
  Native compatibility rather than the transferable long backbone;
- on the K32 holdout, physical `x` is the best 64K arm even though it fails the
  32K Native gate, while the zero-parameter cell-average hypothesis fails its
  CPU entrance condition;
- matched K32 s2 closes the scale confound: physical `x` wins 64K while
  normalized index passes Native, so neither uniformly dominates;
- two K128 Gemma-1 screens recover nonzero 8K behavior under frozen tables but
  remain zero at 16K. Native 4K and table-only/gain/loader controls prove this
  is a real behavioral boundary rather than the wrong artifact or a broken
  evaluator, while cross-K coordinate identification remains unresolved.

No Native-correction or hierarchical candidate survives the current gates.
The next evidence question is a matched deterministic static-baseline panel,
not another fitted curve. Native/long routing and post-outcome gain rescue
remain outside the target method.

## Evidence to open on demand

| Question | Owner | Boundary |
| --- | --- | --- |
| Does fixed-support `z` affect mature frozen checkpoints? | [`results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) | frozen sensitivity/capability, not a usable Native-support method |
| Does the same axis change fresh natural-text NLL? | [`results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md`](results/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md) | length-conditional fixed-support effect; detailed profile not separated from coarse ramp |
| What did the complete zero-training system establish? | [`results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md`](results/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md) | table + gain + routing system; not pure `z` |
| Why is adaptation expected? | [`results/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md`](results/COADAPTIVE_ALLOCATION_ORACLE_RESULT_20260825.md) and the crossings routed by [`../../../INDEX.md`](../../../INDEX.md) §3 | weights/table co-adaptation; no current method winner |
| Which frozen routes failed? | [`results/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md`](results/ALLOCATION_DOSE_RESPONSE_RESULT_20260826.md), [`results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md`](results/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md), and [`results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`](results/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md) | historical failure/engineering boundaries only |
| Does the two-parameter compression preserve LM behavior? | [`results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md`](results/LOW_DIM_COUPLING_GPU_RESULT_20260901.md) | Qwen/OLMo long behavior yes; strict Native deployment gate no |
| Does K32 failure come from finite-grid point sampling? | [`results/K32_FINITE_K_COUPLING_ANALYSIS_20260901.md`](results/K32_FINITE_K_COUPLING_ANALYSIS_20260901.md) | transition is under-resolved, but exact cell averaging does not match the residual and is not a GPU candidate; physical `x` still transports positively at 64K |
| Does frozen `G(x)` transport at matched scale and K128? | [`results/FROZEN_2D_COUPLING_TRANSPORT_RESULT_20260901.md`](results/FROZEN_2D_COUPLING_TRANSPORT_RESULT_20260901.md) | K32 has a physical-long/index-Native Pareto crossing; K128 is unresolved with replicated zero 16K endpoints, so no universal or SOTA claim |

## Retired work

W0/F1 success-first, F2--F4, `ABSOLUTE/ANCHORED`, protected-ramp,
band-restoration, local-gap, s8/log scaling, per-head frequency, dynamic gain,
spectral flow, and routing-based rescue have no current action or authorization.
Their files remain only because results and preregistrations must preserve
scientific provenance.

The old
[`preflights/ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830.md`](preflights/ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830.md)
is a retired preregistration, not a next protocol. Historical code under
`scripts/eval/` and `scripts/analysis/` must not be launched from this README.

## Directory roles

- [`results/`](results/) — completed result owners and claim ceilings.
- [`evidence/`](evidence/) — compact machine-path-free receipts.
- [`analysis/`](analysis/) — mechanism interpretation and failed hypotheses;
  never an action queue.
- [`preflights/`](preflights/) — what was registered before past execution;
  currently no active preflight.
- [`theory/`](theory/) — historical method/theory work; current method boundary
  is in `AGENTS.md` and `INDEX.md`.

Before future compute, derive and freeze the deterministic `f(z)`, its controls,
and its multi-length gate in the existing authority chain, then obtain explicit
run authorization. A historical script or preflight never supplies missing
protocol fields by implication.
