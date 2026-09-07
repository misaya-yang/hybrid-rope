# INDEX — file and evidence-source index

**Updated:** 2026-09-07; research failure review and final Pro decision request. This file locates source
files and their roles; it does not own numerical results, verdicts or live tasks.
Project details are in `README.md`, constraints in `AGENTS.md`, and live state
in `paper-2027/HANDOFF.md`. Read only the relevant entries.

A local report is not automatically a validated result. Follow its status and
raw receipts; new audit concerns are tracked in the revision brief. Paths marked
`main_0726:<path>` exist on the immutable pre-slim branch, not in this checkout;
inspect them with `git show main_0726:<path>`.

## 1. Project, manuscript and reconstruction

| File | Purpose / status |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Core constraints, plan-before-compute rules and checks |
| [README.md](README.md) | Project question, reconstruction context, intended outcomes and layout |
| [paper-2027/HANDOFF.md](paper-2027/HANDOFF.md) | Current local Git/PDF/authorization state; source-reported remote state is labelled |
| [Research failure review](docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md) | Diagnosis of the guess/fail loop; added phase, spacing and low-frequency coupling derivations around MrPro; no new GPU results |
| [Native phase constrained carrier](docs/research/ROPE_NATIVE_SECTOR_CARRIER_20260907.md) | Core experiment 3 stopped after nine tasks; retained development/raw results, EOS and phase checks; incomplete RULER, no SOTA claim |
| [Low-frequency carrier-removal protocol](docs/research/ROPE_CARRIER_REMOVAL_PILOT_20260907.md), [frozen array](docs/research/ROPE_CARRIER_REMOVAL_CANDIDATE_20260907.json), [supplied Pro source](paper-2027/research/external-reviews/ROPE_LOWFREQ_CARRIER_METHOD_20260907.md) | Author-selected fixed Carrier trial, signed implementation, construction receipts and generated outcomes; 13-task single-arm evaluation; live budget by HANDOFF |
| [Recovered historical Qwen log-p2 table](docs/research/ROPE_RECOVERED_QWEN_P2_20260907.json) | Exact Native/m/deployment arrays recovered and hash-checked from old Qwen 1.5B audit; same Native geometry as current 3B, not a new 3B result; comparison with MrPro |
| [Shared-frequency output response](scripts/analysis/shared_frequency_response.py), [existing-cache analysis](scripts/analysis/analyze_cached_frequency_response.py) | Split-half sine-sign error corrected against independent absolute rotation; real Native-cache output parity checked; retains signed key/head response, not a loss optimizer |
| [Scale-transport assumption calculations](docs/research/ROPE_SCALE_TRANSPORT_ASSUMPTIONS_20260907.json), [CPU analysis](scripts/analysis/diagnose_scale_transport_assumptions.py) | Existing-array changes and exact counterexamples for independent-phase energy versus shared-frequency response, and frequency clipping versus amplitude filtering; no new candidate or GPU run |
| [Final Pro decision request](docs/research/ROPE_PRO_DECISION_REQUEST_20260907.md) | Current self-contained prompt with actual failures; request one implementable method and a discriminating low-cost experiment |
| [Qwen 1.5B minimal mechanism comparison](docs/research/ROPE_QWEN15_MINIMAL_MECHANISM_20260907.md), [fixed tables](docs/research/ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json), [full-lag audit](scripts/analysis/audit_qwen_p2_full_lag.py), [decision replay](scripts/analysis/compare_decision_traces.py) | Current small-model protocol: one numerical repair of the old effective rule, official and matched-gain Mr references, actual generation decisions; supersedes full 3B matrix |
| [Unified frequency research plan](docs/research/ROPE_FREQUENCY_UNIFIED_PLAN_20260907.md) | Current zero-training to LoRA to sparse-attention order; physical 64K/replay implementation; V4 shared-KV rotation identity and its limits; old matrix suggestions superseded |
| [Scale-transport proposal review](docs/research/ROPE_SCALE_TRANSPORT_REVIEW_20260907.md) | External-proposal review at its original scope; subsequent implementation/results routed to pilot owner |
| [Pro scale-transport source](paper-2027/research/external-reviews/ROPE_SCALE_TRANSPORT_METHOD_AND_CODEX_20260907.md) | Exact author-supplied September 7 input; SHA in review; reported code/tests unverified, execution proposals are not authorization |
| [Independent general-allocation derivation](docs/research/ROPE_GENERAL_ALLOCATION_DERIVATION_20260907.md) | Historical mathematical exploration; general-definition/MGDA route withdrawn by author correction; not an experiment prerequisite |
| [General-allocation CPU checks](docs/research/ROPE_GENERAL_ALLOCATION_CPU_20260907.json), [verification program](scripts/analysis/verify_general_rope_allocation.py) | Reproducible standard-library numerical checks; no real model or GPU results |
| [Pro research prompt](docs/research/ROPE_FREQUENCY_PRO_PROMPT_20260907.md) | Historical prompt already sent to Pro; preserve as an input artifact, not current instructions |
| [Luna results and ROI review](docs/research/ROPE_FREQUENCY_LUNA_ROI_20260907.md) | Historical OLMo E0/E1 evidence and dated ROI suggestions; old Z-only training is not the continuation queue |
| [paper-2027/REVISION_BRIEF.md](paper-2027/REVISION_BRIEF.md) | Current reconstruction contract; historical v5/v4 folded below and explicitly non-operative |
| [paper-2027/NARRATIVE_GUIDE.md](paper-2027/NARRATIVE_GUIDE.md) | Historical narrative guide; current reconstruction contract supersedes former priorities |
| [paper-2027/README.md](paper-2027/README.md) | Manuscript package layout and build conventions |
| [paper-2027/main.tex](paper-2027/main.tex) | Actual manuscript entrypoint; wording has not yet been reconstructed |
| [paper-2027/main.pdf](paper-2027/main.pdf) | Current rendered manuscript; verify identity in HANDOFF |
| [paper-2027/sections/](paper-2027/sections/) | Main-text sources |
| [paper-2027/appendix/](paper-2027/appendix/) | Proofs, extended experiments and limitations |
| [paper-2027/tables/](paper-2027/tables/) | Reviewer-facing table inputs |
| [paper-2027/figs/](paper-2027/figs/) | Figures and active generators |
| [paper-2027/refs/](paper-2027/refs/) | Bibliography sources |
| [paper-2027/compile.sh](paper-2027/compile.sh) | Active manuscript build and format gates |
| [paper-2027/SUBMISSION_CHECKLIST.md](paper-2027/SUBMISSION_CHECKLIST.md) | Release checklist; venue details need live verification |
| [paper-2027/research/RESEARCH_PROTOCOL_REFERENCE.md](paper-2027/research/RESEARCH_PROTOCOL_REFERENCE.md) | Metric conventions, evidence labels, method identities and prior root-index claim/correction annotations |
| [paper-2027/research/external-reviews/ROPE_ICLR2027_CROSS_AUDIT_20260906.md](paper-2027/research/external-reviews/ROPE_ICLR2027_CROSS_AUDIT_20260906.md) | Exact supplied audit; planning input, not independently validated theory/model/CPU evidence |
| [paper-2027/research/external-reviews/README.md](paper-2027/research/external-reviews/README.md) | External-source status and provenance boundaries |

## 2. Latest local round and implementation audit

| File | Purpose / status |
| --- | --- |
| [Cross-audit experiment protocol](paper-2027/research/CROSS_AUDIT_EXPERIMENT_PROTOCOL_20260907.md) | Author-selected September 6 source; E0/E1 preparation, full/LoRA contract, CPU findings and conditional stages |
| [Scale-transport two-hour pilot protocol](docs/research/ROPE_SCALE_TRANSPORT_PILOT_20260907.md) | Main current result owner: guard/QA diagnostics plus combination RULER; score interpretation corrections, skipped VT rows and raw identities |
| [Paired P2-middle/Mr experiment](scripts/experiments/scale_transport/paired_ruler_run.py), [state readback](scripts/experiments/scale_transport/capture_paired_states.py), [binding-state analysis](scripts/analysis/analyze_binding_states.py) | Continuation experiment 1 and numerical replay of the same conditions; current Qwen3B result and count owned by pilot/HANDOFF; no automatic candidate search |
| [Pilot follow-up analysis and candidate](docs/research/ROPE_SCALE_TRANSPORT_FOLLOWUP_20260907.json), [analysis program](scripts/analysis/analyze_scale_transport_pilot.py) | Row-level score decomposition and one untested Mr-middle/scale-tail/CoPE composition; explanation and limits in pilot owner |
| [Combination RULER preparation](scripts/experiments/scale_transport/ruler_prepare.py), [runner](scripts/experiments/scale_transport/ruler_run.py) | Executed combination RULER apparatus: 22 generated rows and 8 skipped VT rows; no active process; results in pilot owner |
| [Scale-transport pilot code](scripts/experiments/scale_transport/), [math tests](tests/test_scale_transport_math.py) | Executed own-method pilot apparatus; four CPU tests and real model hook checks passed; guarded versus diagnostic variants remain distinct; results in pilot owner |
| [Cross-audit apparatus](scripts/experiments/cross_audit/) | CPU freeze/rescore/input checks; mature/scratch evaluation, teacher, full/LoRA training, cost proposal, exact-job SSH supervision and receipt report |
| [Contract tests](tests/test_cross_audit_contracts.py), [runtime tests](tests/test_cross_audit_runtime.py), [execution tests](tests/test_cross_audit_execution.py) | Complete answers/groups, cached decoding, CE/KL value/gradient parity, full/LoRA steps and job guards |
| [paper-2027/claude_code_workspace/README.md](paper-2027/claude_code_workspace/README.md) | Local Round10–12 context; current reconstruction amendment takes precedence over old restart plans |
| [paper-2027/claude_code_workspace/INDEX.md](paper-2027/claude_code_workspace/INDEX.md) | Detailed per-round file inventory |
| [paper-2027/claude_code_workspace/LESSONS.md](paper-2027/claude_code_workspace/LESSONS.md) | Recorded implementation/evaluation pitfalls; verify applicability |
| [paper-2027/claude_code_workspace/round12_20260906/REPORT_ROUND12_20260906.md](paper-2027/claude_code_workspace/round12_20260906/REPORT_ROUND12_20260906.md) | Reported results/stop state; baseline fidelity and score interpretation require reconciliation; recovery list is historical |
| [paper-2027/claude_code_workspace/round12_20260906/PREGLUCTION_ROUND12.md](paper-2027/claude_code_workspace/round12_20260906/PREGLUCTION_ROUND12.md) | Frozen original table/task/scoring declaration |
| [paper-2027/claude_code_workspace/round12_20260906/PREGLUCTION_ROUND12_V2_ADDENDUM.md](paper-2027/claude_code_workspace/round12_20260906/PREGLUCTION_ROUND12_V2_ADDENDUM.md) | Historical protocol amendment |
| [paper-2027/claude_code_workspace/round12_20260906/RUNBOOK_ROUND12.md](paper-2027/claude_code_workspace/round12_20260906/RUNBOOK_ROUND12.md) | Historical execution instructions; no restart authority |
| [paper-2027/claude_code_workspace/round12_20260906/PHASE0_INTERPRETIVE_MEMO.md](paper-2027/claude_code_workspace/round12_20260906/PHASE0_INTERPRETIVE_MEMO.md) | Interpretation requiring raw/source checks |
| [paper-2027/claude_code_workspace/reports/ROUND10_LORA_RESULTS_20260905.md](paper-2027/claude_code_workspace/reports/ROUND10_LORA_RESULTS_20260905.md) | Earlier low-rank execution report |
| [paper-2027/claude_code_workspace/reports/ROUND11_OLMO_RESULTS_20260905.md](paper-2027/claude_code_workspace/reports/ROUND11_OLMO_RESULTS_20260905.md) | Earlier OLMo execution report |
| [paper-2027/claude_code_workspace/round12_20260906/code/rope_tables.py](paper-2027/claude_code_workspace/round12_20260906/code/rope_tables.py) | N/Z/Y/M array construction and identity/gain declarations |
| [paper-2027/claude_code_workspace/round12_20260906/code/build_y2_canon.py](paper-2027/claude_code_workspace/round12_20260906/code/build_y2_canon.py) | Y2 builder; smoothstep/square-root convention under fidelity review |
| [paper-2027/claude_code_workspace/round12_20260906/code/scoring.py](paper-2027/claude_code_workspace/round12_20260906/code/scoring.py) | Saved output scoring and row/group interpretation |
| [paper-2027/claude_code_workspace/round12_20260906/code/track_a_eval.py](paper-2027/claude_code_workspace/round12_20260906/code/track_a_eval.py) | Frozen evaluation including chat-template path |
| [paper-2027/claude_code_workspace/round12_20260906/code/track_b_train_v2.py](paper-2027/claude_code_workspace/round12_20260906/code/track_b_train_v2.py) | Existing continuation trainer; inspect actual trainable modules/losses before reuse |
| [paper-2027/claude_code_workspace/round12_20260906/code/round11_harness_ref/](paper-2027/claude_code_workspace/round12_20260906/code/round11_harness_ref/) | Frozen reference code; preserve original execution identity |

## 3. Theory and evidence owners

| File | Locate when checking |
| --- | --- |
| [paper-2027/research/attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md](paper-2027/research/attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md) | Understand or improve the two core problems |
| [paper-2027/research/attention-aware-retrofit/analysis/PRO_REPORT_AUDIT_RECONCILIATION_20260905.md](paper-2027/research/attention-aware-retrofit/analysis/PRO_REPORT_AUDIT_RECONCILIATION_20260905.md) | Review the latest Pro audit and revised experiment order |
| [paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md](paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) | Does allocation matter at fixed support during training? |
| [paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md](paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md) | What does full sin/cos geometry prove? |
| [paper-2027/research/foundations/ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md](paper-2027/research/foundations/ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md) | How are support and allocation separated? |
| [paper-2027/research/foundations/ROPE_OPTIMALITY_IDENTIFIABILITY_AND_CONDITIONAL_EQUATIONS_20260903.md](paper-2027/research/foundations/ROPE_OPTIMALITY_IDENTIFIABILITY_AND_CONDITIONAL_EQUATIONS_20260903.md) | Can RoPE/attention structure alone determine optimal `z`, a frequency system, or mature-checkpoint movement? |
| [paper-2027/research/evidence/VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md](paper-2027/research/evidence/VIDEO_DIT_HEAD_TO_HEAD_SEED42_RESULT_20260826.md) | What breadth supports the paper? |
| [paper-2027/research/attention-aware-retrofit/results/causal-mechanism/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md](paper-2027/research/attention-aware-retrofit/results/causal-mechanism/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md) | What is the strongest practical no-update result? |
| [paper-2027/research/attention-aware-retrofit/results/causal-mechanism/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md](paper-2027/research/attention-aware-retrofit/results/causal-mechanism/FRESH_FINEWEB_S4_GENERALIZATION_RESULT_20260824.md) | Does it persist on fresh natural text? |
| [paper-2027/research/attention-aware-retrofit/results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md](paper-2027/research/attention-aware-retrofit/results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md) | What is the strongest tracked static-table result? |
| [paper-2027/research/attention-aware-retrofit/theory/SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903.md](paper-2027/research/attention-aware-retrofit/theory/SINGLE_STATIC_LOG_P2_SELECTION_AND_LORA_20260903.md) | Under one static table and one path, which completed form is retained, and what same-table LoRA follows? |
| [paper-2027/research/attention-aware-retrofit/results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md](paper-2027/research/attention-aware-retrofit/results/adaptation-coadaptation/LOG_P2_QK_LORA_GAIN_MATCHED_RESULT_20260904.md) | Under one static table and one path, which completed form is retained, and what same-table LoRA follows? |
| [paper-2027/research/attention-aware-retrofit/results/coupling-transfer/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md](paper-2027/research/attention-aware-retrofit/results/coupling-transfer/K32_PAIRED_CROSSING_CONFIRMATION_RESULT_20260901.md) | What is supported across K32/K128? |
| [paper-2027/research/attention-aware-retrofit/results/coupling-transfer/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md](paper-2027/research/attention-aware-retrofit/results/coupling-transfer/K128_COORDINATE_RANKING_CONFIRMATION_RESULT_20260901.md) | What is supported across K32/K128? |
| [paper-2027/research/attention-aware-retrofit/results/coupling-transfer/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md](paper-2027/research/attention-aware-retrofit/results/coupling-transfer/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RESULT_20260901.md) | What is supported across K32/K128? |
| [paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md) | Does long signal convert to natural QA? |
| [paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md) | Do headwise clocks solve the joint objective? |
| [paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RESULT_20260903.md](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RESULT_20260903.md) | Does calibration-frozen attention-displacement Selective-31 beat layer-matched random/reverse masks? |
| [paper-2027/research/attention-aware-retrofit/theory/NATIVE_ONLY_MOVEMENT_PROFILE_IDENTIFIABILITY_20260903.md](paper-2027/research/attention-aware-retrofit/theory/NATIVE_ONLY_MOVEMENT_PROFILE_IDENTIFIABILITY_20260903.md) | Can Native checkpoint structure uniquely determine an ordered movement profile? |
| [paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md) | Can Native checkpoint structure uniquely determine an ordered movement profile? |
| [paper-2027/research/attention-aware-retrofit/theory/STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903.md](paper-2027/research/attention-aware-retrofit/theory/STATIC_NATIVE_NO_HARM_AND_PREFIX_HANDOFF_20260903.md) | Can one non-Native static table guarantee exact Native short behaviour and change long geometry? |
| [paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/SESSION_BINARY_S4_REAL_CONTEXT_RESULT_20260823.md) | Can one non-Native static table guarantee exact Native short behaviour and change long geometry? |
| [paper-2027/research/attention-aware-retrofit/theory/LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903.md](paper-2027/research/attention-aware-retrofit/theory/LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903.md) | Is the Selective-31 calibration score a universal functional sensitivity, and does joint Q/K--frequency relabeling invalidate the ordered-coupling results? |
| [paper-2027/research/attention-aware-retrofit/results/operator-analysis/SCALE_ORBIT_BOUNDARY_VALIDATION_RESULT_20260904.md](paper-2027/research/attention-aware-retrofit/results/operator-analysis/SCALE_ORBIT_BOUNDARY_VALIDATION_RESULT_20260904.md) | Do finite scale-orbit boundary and Fourier-rank quantities predict mature-model behaviour? |
| [paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md](paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md) | Do finite scale-orbit boundary and Fourier-rank quantities predict mature-model behaviour? |
| [paper-2027/research/attention-aware-retrofit/theory/FINITE_SCALE_COVARIANCE_PROOF_NOVELTY_AND_TIGHTNESS_AUDIT_20260904.md](paper-2027/research/attention-aware-retrofit/theory/FINITE_SCALE_COVARIANCE_PROOF_NOVELTY_AND_TIGHTNESS_AUDIT_20260904.md) | What survives a proof, novelty, and tightness audit of the supplied finite scale-covariance derivation? |
| [paper-2027/research/attention-aware-retrofit/results/operator-analysis/SCALE_CONJUGACY_TIGHTNESS_RESULT_20260904.md](paper-2027/research/attention-aware-retrofit/results/operator-analysis/SCALE_CONJUGACY_TIGHTNESS_RESULT_20260904.md) | What survives a proof, novelty, and tightness audit of the supplied finite scale-covariance derivation? |
| [paper-2027/research/attention-aware-retrofit/analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md](paper-2027/research/attention-aware-retrofit/analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md) | Where is the earlier single-table synthesis preserved? |
| [paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md](paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md) | What can the 9/2 first-principles memo support? |
| [paper-2027/research/attention-aware-retrofit/results/coupling-transfer/REFERENCE_CORRECTED_K128_RESULT_20260901.md](paper-2027/research/attention-aware-retrofit/results/coupling-transfer/REFERENCE_CORRECTED_K128_RESULT_20260901.md) | old Gemma 16K zero with 8K reference |
| [paper-2027/research/audits/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md](paper-2027/research/audits/KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md) | attention-Fisher `kappa_att` |
| [paper-2027/research/audits/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md](paper-2027/research/audits/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md) | LeRoPE `w^(1/3)` oracle |
| [paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/DIRECT_Z_FIXED_SUPPORT_PILOT_RESULT_20260824.md) | direct-`z` two-document calibration |
| [paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md](paper-2027/research/attention-aware-retrofit/results/zero-training-deployment/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md) | two analytic Native-support tables |
| [paper-2027/research/attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md](paper-2027/research/attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md) | continuous-boundary-slope operator |
| [paper-2027/research/attention-aware-retrofit/results/PHASE_ISOTROPY_50M_M4_RESULT_20260824.md](paper-2027/research/attention-aware-retrofit/results/PHASE_ISOTROPY_50M_M4_RESULT_20260824.md) | phase-isotropy / pair-volume / min-eigenvalue |
| [paper-2027/research/attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md](paper-2027/research/attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md) | phase-isotropy / pair-volume / min-eigenvalue |
| [paper-2027/research/attention-aware-retrofit/results/SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md](paper-2027/research/attention-aware-retrofit/results/SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md) | execution report |

## 4. Protocols, code and checks

| File | Locate when checking |
| --- | --- |
| [paper-2027/research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md](paper-2027/research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md) | Start the prepared N_compact / Qwen Z/Y round |
| [scripts/experiments/matched_transfer_round.py](scripts/experiments/matched_transfer_round.py) | Earlier N_compact/Z/Y planner; not an implementation of new E0–E5 |
| [scripts/analysis/audit_generation_transitions.py](scripts/analysis/audit_generation_transitions.py) | Start the prepared N_compact / Qwen Z/Y round |
| [scripts/train/run_native_constrained_transfer.sh](scripts/train/run_native_constrained_transfer.sh) | Review FFN learning/forgetting and choose the next task from results |
| [scripts/analysis/review_native_constrained_transfer.py](scripts/analysis/review_native_constrained_transfer.py) | Receipt review and protocol-specific stopping decisions |
| [tests/test_native_constrained_transfer.py](tests/test_native_constrained_transfer.py) | Review FFN learning/forgetting and choose the next task from results |
| [scripts/experiments/single_table_generation.py](scripts/experiments/single_table_generation.py) | Implement or audit the new assay |
| [scripts/analysis/export_single_table_controls.py](scripts/analysis/export_single_table_controls.py) | Implement or audit the new assay |
| [scripts/train/train_single_table_native_constrained.py](scripts/train/train_single_table_native_constrained.py) | Registered Native-constrained training apparatus |
| [scripts/lib/rope/generation_contract.py](scripts/lib/rope/generation_contract.py) | Full-output/EOS, retention and paired uncertainty code |
| [tests/test_single_table_generation.py](tests/test_single_table_generation.py) | Implement or audit the new assay |
| [paper-2027/research/attention-aware-retrofit/preflights/operator-analysis/SCALE_ORBIT_TRANSPORT_RESIDUAL_PREFLIGHT_20260904.md](paper-2027/research/attention-aware-retrofit/preflights/operator-analysis/SCALE_ORBIT_TRANSPORT_RESIDUAL_PREFLIGHT_20260904.md) | Do finite scale-orbit boundary and Fourier-rank quantities predict mature-model behaviour? |
| [paper-2027/research/attention-aware-retrofit/preflights/operator-analysis/SCALE_CONJUGACY_TIGHTNESS_PREFLIGHT_20260904.md](paper-2027/research/attention-aware-retrofit/preflights/operator-analysis/SCALE_CONJUGACY_TIGHTNESS_PREFLIGHT_20260904.md) | What survives a proof, novelty, and tightness audit of the supplied finite scale-covariance derivation? |
| [scripts/lib/rope/](scripts/lib/rope/) | Reusable RoPE code |
| [scripts/analysis/third_axis_ceiling.py](scripts/analysis/third_axis_ceiling.py) | Static rank diagnostic |
| [scripts/eval/](scripts/eval/) | Current evaluation utilities |
| [scripts/experiments/source_only_generation_guard.py](scripts/experiments/source_only_generation_guard.py) | source-only control guard |
| [scripts/experiments/native_window_guard.py](scripts/experiments/native_window_guard.py) | Native-window guard |
| [scripts/experiments/niah_retention_canary.py](scripts/experiments/niah_retention_canary.py) | NIAH retention canary |
| [scripts/experiments/simple_capability_canary.py](scripts/experiments/simple_capability_canary.py) | simple capability canary |
| [scripts/lib/rope/official_yarn.py](scripts/lib/rope/official_yarn.py) | Pinned official-equation operator and non-native-grid generalization boundary |
| [tests/test_official_yarn_parity.py](tests/test_official_yarn_parity.py) | Operator parity checks; work-machine dependencies |
| [tests/test_matched_transfer_round.py](tests/test_matched_transfer_round.py) | Prepared-round contracts; imports PyTorch through package initialization |
| [scripts/package_supplement.py](scripts/package_supplement.py) | Curated ICLR packager; allowlist includes archive-only inputs requiring release reconciliation |
| [scripts/README.md](scripts/README.md) | Implementation directory roles and supporting entrypoints |

## 5. Supporting catalogues and history

| File | Locate when checking |
| --- | --- |
| [docs/overview/SERVER_STORAGE_CLEANUP_20260907.md](docs/overview/SERVER_STORAGE_CLEANUP_20260907.md) | Completed no-GPU storage cleanup, retained assets, local evidence migration and the pre-existing seed42 checkpoint gap |
| [docs/overview/CURRENT_RESEARCH_COMPUTE_AND_MODEL_ASSETS_20260904.md](docs/overview/CURRENT_RESEARCH_COMPUTE_AND_MODEL_ASSETS_20260904.md) | Identify measured compute and available model assets |
| [paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json](paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json) | Does allocation matter at fixed support during training? |
| [paper-2027/sections/03_theory.tex](paper-2027/sections/03_theory.tex) | What is the bounded EVQ-Cosh theorem? |
| [paper-2027/appendix/a1_proofs.tex](paper-2027/appendix/a1_proofs.tex) | What is the bounded EVQ-Cosh theorem? |
| [docs/exp/2026-03/2026-03-06_phase15_750m_2k_to_4k_continue_results.md](docs/exp/2026-03/2026-03-06_phase15_750m_2k_to_4k_continue_results.md) | What breadth supports the paper? |
| [paper-2027/research/history/TIMELINE.md](paper-2027/research/history/TIMELINE.md) | Why the question changed |
| [paper-2027/research/README.md](paper-2027/research/README.md) | Research-layer placement and local catalogue entrypoints |
| [paper-2027/research/attention-aware-retrofit/results/README.md](paper-2027/research/attention-aware-retrofit/results/README.md) | Mature-result catalogue |
| [paper-2027/research/attention-aware-retrofit/theory/README.md](paper-2027/research/attention-aware-retrofit/theory/README.md) | Mature theory status |
| [docs/exp/](docs/exp/) | Historical reports |
| [docs/overview/TERMS_AND_PROTOCOLS.md](docs/overview/TERMS_AND_PROTOCOLS.md) | Early historical terms/metrics; do not apply globally to current assays |
| [docs/overview/METHODOLOGY.md](docs/overview/METHODOLOGY.md) | Early MHA/core-text methodology; protocol-specific historical reference |

## 6. Archive-only sources

| Git locator | Purpose / historical question |
| --- | --- |
| `main_0726:rebuttal/rebuttal_0723/theory_results/M4_EXACT_RANGE_FACTORIAL_RESULT_20260726.md` | Does the effect persist across exact-range configurations? |
| `main_0726:rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` | What is the exact frozen transplant boundary? |
| `main_0726:rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md` | What supports the matched-adaptation route? |
| `main_0726:rebuttal/rebuttal_0723/theory_results/EVQ_8B_ADAPTATION_EVIDENCE_20260724.md` | What supports the matched-adaptation route? |
| `main_0726:data/curated/table18_mla_3seed_aggregate.json` | What breadth supports the paper? |
| `main_0726:rebuttal/rebuttal_0723/theory_results/OLMO2_1B_RELEASED_ROPE_BASELINE_20260725.md` | What breadth supports the paper? |
| `main_0726:rebuttal/rebuttal_0723/README.md` | July review/evidence |

## 7. Referenced inputs not yet matched

| External reference | Local verification status |
| --- | --- |
| Audit P1 manuscript attachment | Current main.pdf has not been byte-matched to the supplied-review attachment |
| Audit P2 synthesis and P3/P4 attachments | Not located under their exact names in this checkout/adjacent Downloads lookup; do not alias them to another report |
| `rope_codex_revision/verify_theory.py`, `select_allocation.py`, `verification_results.json` | Referenced external package not found in the searched locations; CPU checks not reproduced |
| Exact trained weights, raw execution receipts and current server processes | Require work-machine verification; local code/manifests are only leads |

Prior owner summaries, correction labels and scoped negative interpretations
were moved to [research reference — prior routing annotations](paper-2027/research/RESEARCH_PROTOCOL_REFERENCE.md#prior-routing-annotations).
Their raw-backed owners retain authority; the index itself makes no new verdict.
