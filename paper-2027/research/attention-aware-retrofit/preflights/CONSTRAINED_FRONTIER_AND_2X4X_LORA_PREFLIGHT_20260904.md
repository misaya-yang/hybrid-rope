# Fixed-witness confirmation and Native-constrained transfer — v3 reviewed

- **Date/status:** 2026-09-04; historical v3 design, amended during authorized
  execution. Actual assets, runtime and results are in the
  [execution owner](../results/SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md).
  Statements below about unavailable assets and segmented64-step resume are
  superseded for the current Qwen branch; they describe the original preflight.
- **Questions:** Z: what can the existing global static table do, and at what
  Native cost? F: does fixed-table small adaptation transport an already-solvable
  task into full long generation with separate Native retention?
- **Theory and dossier decisions:** [first-principles owner](../theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md), especially §§6–8.
- **Authority:** current user scope remains separate Native retention >=.88;
  historical .875 is marginal. The dossier's .03-NLL/-2-point thresholds and
  72 GPU-h proposal are NOT substituted for this target or compute authorization.
- **Machines:** user-reported AutoDL 5090 32GB and 4080 Super 32GB. Detect actual
  memory, BF16 and Flash. Neither device name nor available VRAM predicts runtime.
- **Superseded before execution:** new frequency/gain sweeps, m^1.25 as next arm,
  QK -> QKVO escalation, source-contrast training, and synthetic replay alone as
  Native protection. No scientific result was produced by those prototypes.

## Execution amendment — read before the historical commands

The active checkpoint is Qwen1.5B with true Native32K; train caps remain2K/8K/16K.
The fixed K64 movement is transported onto its Native basis, with frozen gain;
it is a new checkpoint-specific candidate, not the original OLMo tensor.
All-linear r16 has18,464,768 parameters; attention-only budget match isr68
(+0.31%), conditional on feasibility. OLMo r46 below is not the Qwen setting.

An identity-KL numerical defect invalidated the first N trajectory. Engine008
fixes it with the exact first-order logit gradient. Resume parameter parity
remains unresolved, so corrected runs use uninterrupted32R+96T with all saves,
instead of inheriting the historical segmented-resume commands. This spends
one fixed recipe before checkpoint evaluation; no additional optimization
budget, table search or test-based selection is authorized by that amendment.

Original-Native validation subsequently found only4 compact-correct double-
evidence groups, below the registered8. Z was stopped during restoration at26
completed steps; this is an unresolved assay, not a negative method result.
The completed N128 is evaluated diagnostically. No Y, seeds, attention-only
ablation or farther matrix should run until controls resolve. A same64-group
[source-only development guard](../../../../scripts/experiments/source_only_generation_guard.py)
adds common-width source padding, exact near/far swaps and identical deleted
worlds without changing old training/results. Deleted paired zero is structural;
report single-world outputs and differences. A separate24-report Native guard
covers up to32K. Preserve these companion protocols and their limits explicitly.

## 1. What is now fixed

N = original checkpoint/table/gain. Z = exact full legacy-u p2 log-s4 tensor,
rotary amplitude 1.102585782722872, all lengths and requests. G = geometric
interior with exactly Z's sampled endpoints/K/gain. Y = official Transformers
YaRN factor4, with its own recorded official amplitude. G/Z is the pure-z
contrast; N/Y are practical references. Do not label G as FMRoPE.

The exporter copies the exact Z bytes and validates the Native config/tensor.
No outcome picks a gain, scale, or movement. The old c=.10 and c=.074 RULER
vectors belong to different protocols and remain separate, valid owner entries.

E2's candidate is all-linear LoRA r16/alpha16 across Q/K/V/O and gate/up/down;
base weights, norms, embeddings, output head and table/gain are frozen. This is
12,058,624 parameters in the pinned OLMo shapes, a budget choice, not a theorem.
Attention-only r46 is a parameter-matched explanatory control after feasibility.

Full answer/EOS CE plus .25 worst teacher-capped gold-prefix margin is the task
loss. Each lawful counterfactual world gets its own correct target. Source
contrast is measured diagnostically, not optimized. Four-stratum full-vocabulary
Native teacher KL constrains the ACTUAL deployed student function; student
never switches to Native RoPE during replay. KL target .02 is an optimization
setting, not a replacement for final NLL/task/EOS retention checks.

## 2. E0 tomorrow: qualify a cheap independent instrument

From the repo root on the existing work-machine environment:

```bash
export EVQ_PYTHON=python3
export EVQ_CHECKPOINT=<exact-local-OLMo-checkpoint>
export EVQ_WORK_DIR=<new-private-output-directory>
export EVQ_NATIVE_TABLE=<exact-native-npy>
export EVQ_LOG_P2_S4_TABLE=<exact-full-p2-s4-npy>
export EVQ_RETENTION_MANIFEST=<existing-tokenized-Native-evaluation-manifest>
export EVQ_ASSET_TOKENIZER_ROOT=<original-tokenizer-directory-for-that-manifest>
export EVQ_MAX_SECONDS=3600
bash scripts/eval/run_single_table_diagnosis.sh prepare
bash scripts/eval/run_single_table_diagnosis.sh preflight
```

Replace placeholders. No download or GPU model operation occurs in these two
stages. CPU Torch/Transformers on the work machine are needed for the official
Y export and tokenizer. Do not recreate that environment on the personal PC.

After the author chooses machine/time budget:

```bash
export SINGLE_TABLE_GPU_AUTHORIZED=YES
bash scripts/eval/run_single_table_diagnosis.sh smoke
bash scripts/eval/run_single_table_diagnosis.sh controls
```

The 24-group synthetic instrument has compact (<=2K), full near, full far and
source-deleted conditions. Near/far exchange exact token blocks; the background
multiset, query position, answer and total length are unchanged. An independent
text reader checks unique source-owned answers before model evaluation.

Complete decoded raw output plus actual EOS is primary. No substring/first-number
credit, no special-token stripping, no tokenizer cleanup. Equivalent tokenizations
of an identical full string pass; canonical token equality is diagnostic only.
Raw IDs remain the source of re-scoring. Official natural metrics are separate.

Smoke compares cached versus full-prefix greedy on the actual model. Controls
also record all gold-prefix margins/ranks/logprobs, first canonical token
mismatch, and same-target source intervention effects. Positive gold margins
with a different cached path are an implementation/numerical alarm, not an
exposure-bias diagnosis. Attention matrices are never materialized.

Read `controls_native/summary.json` and raw rows. Compact failure prevents an
existing-capability-transport interpretation. Compact succeeds but full near
fails: background burden matters. Near succeeds but far fails: distance/layout
interaction remains. Output probabilities following the source with wrong
answers is not completed conversion. EOS failure is visible separately.

These synthetic tests qualify infrastructure; they are not the natural E1/E2
training or final benchmark. Do not fix them by teaching a special output format
and then call that recovery of an originally existing Native capability.

## 3. E1: confirm the witness, do not search

```bash
bash scripts/eval/run_single_table_diagnosis.sh native
bash scripts/eval/run_single_table_diagnosis.sh nll-screen
bash scripts/eval/run_single_table_diagnosis.sh retention
bash scripts/eval/run_single_table_diagnosis.sh select
```

Default arm is Z. `EVQ_ARM=N`, `G` or `Y` runs the predeclared controls on the
same data. G is an explanatory negative control, not a candidate to promote as
Native-feasible. Do not open an unlimited long matrix for a failed candidate.

The imported Native pack is independently checked for tokenizer/file identity,
source-group split, complete task cells and physical prompt+generation reserve.
The five natural tasks are Qasper, MultiFieldQA-en, HotpotQA, 2WikiMQA, GovReport.
It reports NLL/PPL, ordinary task macro, EOS-completed task macro and paired
source-group intervals. Historical exposure of imported documents is not erased
by splitting them; fresh confirmation still matters.

A fixed Z that misses .88 can be reported as a costly working point, not a
qualified joint point. This does not prohibit testing whether a separately
registered adapter repairs it. Do not silently choose another c to improve the
paper's number. Final Native confirmation uses
`EVQ_RETENTION_FOLD=confirmation` with the corresponding Native baseline.

A formal E1 natural C/N/F confirmation requires new source-locked natural worlds,
not the synthetic pack. Reuse existing official benchmark ROWS only after
verifying their identities, but independently recompute outputs/scores.

## 4. E2 required assets — a concrete preflight boundary

The primary engine is
[`train_single_table_native_constrained.py`](../../../../scripts/train/train_single_table_native_constrained.py).
It is implemented but cannot execute science without the following assets.

**A. Qualified natural task manifest** (`QUALIFIED_NATURAL_TRANSPORT_V1`):

```json
{
  "status": "QUALIFIED_NATURAL_TRANSPORT_V1",
  "tokenizer_files": {"tokenizer.json": "SHA256", "tokenizer_config.json": "SHA256"},
  "eos_token_id": 100257,
  "views_path": "transport_views.jsonl", "views_sha256": "SHA256",
  "qualification_path": "native_compact_qualification.json",
  "qualification_sha256": "SHA256",
  "evaluation_splits": {
    "validation": {"groups": 64, "lengths": [2048, 16384]},
    "test": {"groups": 256, "lengths": [2048, 16384, 32768, 65536]}
  }
}
```

Include exactly the actual tokenizer fingerprint files; placeholders are not
valid hashes. Each view has `semantic_id`, `source_id`, `template_lineage`,
`split` (train/validation/test), `world` (0/1), `family`
(single_evidence/double_evidence/binding), `layout` (compact/near/far),
`length_cap`, `prompt_ids`, and `target_ids` including the real EOS.

Training is 128 semantic groups (64/32/32 by family), two worlds and physical
caps 2048/8192/16384: 768 views once. Long views must actually occupy their physical
length (within the 64-token reserve), not use virtual positions. Source/template/
semantic lineages cannot cross splits. Near/far correctness and counterfactual
fact consistency need source-backed validation, not a boolean invented by an LLM.

Qualification JSON also requires a real `candidate_pool_path` and matching
`candidate_pool_sha256`, `screened_candidates`
(128–2000), `selection_rule="fixed_order_native_compact_both_worlds"`, and a
`rejections` ledger. Do not fill quotas by Z performance or undocumented resampling.
It identifies the original Native weight hash/table/gain and has
one row per training world with `semantic_id`, `world`, `compact_prompt_ids`,
truth-verified `target_ids`, actual `generated_ids`, `gold_margins`, and
`truth_verified`. The actual Native compact output must match the truth-verified
chosen target including EOS; every canonical gold margin must be positive.
The compact training prompt must equal the qualified prompt exactly. Every
validation/test semantic group has BOTH worlds: compact at 2048 and near/far
at every declared long length. Validation is exactly 64 groups; test is at
least 256 groups, including the predeclared 64 unseen-family examples. Optional
131072 must be registered before training. The engine checks matrix completeness,
token types/vocabulary, disjoint complete-answer aliases across worlds and
generation reserve before GPU loading; the owner must
also audit the unseen-family quota and semantic truth.
The source-backed truth and rejection ledger must accompany this mechanical
contract. Preflight checks the fields; it does not authenticate semantic truth.

**B. Independent Native pool manifest** (`NATIVE_REPLAY_POOL_V1`): tokenizer
fingerprint, `rows_path`, `rows_sha256`. Each row: `id`, `source_id`, `split`
(train/calibration/validation/test), `group` (text/instruction/reasoning/
position_format), `input_ids` <=4096, and fixed `prediction_positions` indexing
hidden states that predict the NEXT token. Each group has exactly 128 train,
32 calibration and 64 validation rows, with independent source ownership.
Instruction trajectories include the verified answer/EOS prediction positions.
Whole source groups stay together and do not overlap task data.
For text validation/test rows, provide `text_domain`; prediction positions must
have observed next tokens in `input_ids`. Validation spans at least two domains.
For other validation/test strata, provide `prompt_ids`, a physical
`generation_budget`, source-verified `accepted_full_answers`, and
`truth_verified=true`. These fields enable independent FULL generation scoring;
cached teacher distributions do not supply correctness labels. Formal Native
test targets two text domains x128 and three generation strata x500; a smaller
predeclared test must report unresolved precision, not relax retention bounds.

These natural/Native assets have not been supplied locally. Missing files,
insufficient qualification/quota, bad hashes, overlap or wrong physical lengths
produce a concrete `BLOCKED_DATA_QUALIFICATION` / `BLOCKED_NATIVE_DATA` before
CUDA loading. No synthetic rows are silently substituted to make preflight pass.

## 5. E2 executable stages, after qualified assets exist

```bash
export TASK_MANIFEST=<qualified-natural-transport-manifest>
export NATIVE_POOL=<independent-Native-pool-manifest>
export TEACHER_CACHE=<new-private-cache-directory>
"$EVQ_PYTHON" scripts/train/train_single_table_native_constrained.py preflight \
  --checkpoint "$EVQ_CHECKPOINT" --tasks "$TASK_MANIFEST" \
  --native-pool "$NATIVE_POOL" --output "$EVQ_WORK_DIR/e2_preflight"
"$EVQ_PYTHON" scripts/train/train_single_table_native_constrained.py cache-native \
  --checkpoint "$EVQ_CHECKPOINT" --native-pool "$NATIVE_POOL" \
  --authorized --max-seconds 3600 --output "$TEACHER_CACHE"
```

Teacher is original Native, never Z or an adapter. Cache is full vocabulary,
FP32 storage of the declared BF16-model forward logits, not top-k. The code
checks cache disk capacity; final test rows are excluded. No teacher and student
are kept resident together during training.

For Z, one smoke exercises restoration plus a length-mixed task batch and Native
constraint; output is explicitly nonscientific:

```bash
"$EVQ_PYTHON" scripts/train/train_single_table_native_constrained.py smoke \
  --checkpoint "$EVQ_CHECKPOINT" --tasks "$TASK_MANIFEST" --native-pool "$NATIVE_POOL" \
  --teacher-cache "$TEACHER_CACHE" --arm Z --table "$EVQ_WORK_DIR/fixed_controls/Z.npy" \
  --gain 1.102585782722872 --authorized --max-seconds 600 --output "$EVQ_WORK_DIR/e2_smoke_Z"
```

The exact same arguments with action `train`, a fresh output directory and the
chosen time cap START the fixed 32-step restoration + 96-step transfer protocol.
The default first segment stops at global step64 (32 restoration +32 transfer).
Review Native and natural generation before resuming to96/128. This is one
frozen trajectory, not separate retuned models; remaining matrix can stop early.
N omits `--table` and uses gain1; Y uses its frozen file and amplitude from the
control manifest. Seed42 order is N/Z/Y, then conditional seeds43/44 follow the
dossier's balanced order. No seed lottery or rank/loss sweep.

Saved steps are 0/32/64/96/128. Optimizer resets between stages; all-linear
r16/alpha16; LR1e-4, batch8 task views and 2 Native replay examples in transfer.
Initial total Native KL is measured on a fixed calibration subset; zero
restoration gradients on N are legitimate. Only observed Native strata update
their dual multiplier. Full target-position LM-head projection avoids full-length
vocabulary tensors. The stored completion receipt proves TRAINING ONLY.

The engine's `native-evaluate` action measures held-out observed-token text NLL
and full-answer/EOS Native generation on the separate pool. The old five-task
evaluation remains a secondary historical comparison, not a substitute for this
fresh gate. Its task macro and the new three-stratum generation macro are
different endpoints and must be labelled separately.

The engine's `evaluate` action reads the same frozen natural-view schema at
`--split validation` or `test`, with `--adapter` pointing to a saved step.
Evaluation rows require a predeclared `generation_budget` fitting their cap and
may declare `accepted_full_answers`; outputs preserve raw IDs and full strings.
It reports both-world exact+EOS by family/layout/length. Test requires
`--frozen-selection` binding `table_sha256`, `gain`, `adapter_sha256`, and
`adapter_config_sha256` and `task_manifest_sha256`; Native test also binds
`native_pool_sha256`. Adapter weight hashes alone do not freeze alpha/scaling.
No checkpoint is selected from test output.

Independent Native validation and checkpoint selection remain separate gates.
The reviewed trainer saves optimizer, Python/Torch/CUDA RNG, dual multipliers,
exposure order and hashes. `--resume <step-directory>` requires the identical
recipe/code/assets and a later `--stop-after-step`; use a fresh segment output.
Cache preparation journals completed shards; `--resume-cache` verifies and
reuses them. Neither retries nor checkpoints are scientific replication.
The one-step smoke now has a nonzero LR; v2 accidentally decayed its only step
to zero. No v2 GPU smoke or training was run. Saved/merged greedy parity is
checked after releasing the training model/optimizer. GPU gradient, exact resume
parity, actual memory and timing still require work-machine qualification.
A training-complete file never authorizes a capability claim.
The optional `--placement attention_matched` and `--compact-only` are restricted
to Z/seed42 explanatory runs after main feasibility, not alternative candidates.

## 6. Blind lengths, stop rules and ROI

Freeze final methods/checkpoints before exposing 32K/64K. The original 128K
(32x) objective is a conditional extension, not a first-batch requirement. A
registered zero primary score or failed Native gate stops that candidate's
remaining matrix. Unresolved controls stop interpretation, not the method class.
Do not convert skipped longer cells into zeros or retune after a reveal.

The E0/E1 driver retains seal/far stages for a qualified frozen witness, but do
not open them while the adaptation design is still being changed. All claims
must disclose which lengths influenced development. Early 16K success remains
valuable even if farther evaluation fails.

Cost priority: small valid diagnostic -> fixed witness confirmation -> one
matched N/Z/Y training seed -> independent retention/task validation -> optional
replication/longer lengths -> explanatory controls/second architecture. No 72
GPU-h batch is authorized by this document. Each stage has a cap; a current
forward may finish after the boundary. Save persistent receipts and configure
the user's intended AutoDL external shutdown limit; the scripts do not power
off the host. Do not consume paid idle time waiting for data construction.

Prefer 5090 for main adaptation, the other measured-32GB device for teacher/
Native evaluation if available. Match scientific conditions, record device and
backend, and use actual throughput to allocate work. CPU syntax/schema tests do
not qualify HF/PEFT, Flash gradients, 64K/128K memory or final publication.

## 7. Reviewed first-boot sequence and result-driven continuation

The [E2 stage driver](../../../../scripts/train/run_native_constrained_transfer.sh)
runs ONE stage and ONE arm. Set the E0 variables above plus `TASK_MANIFEST`,
`NATIVE_POOL`, and `TEACHER_CACHE`. No assets are downloaded or silently replaced.
Use `preflight` while compute is off; missing source truth/qualification is the
current blocker. Build these assets from verified source owners before renting
an idle GPU. Preparing them is the next data task, not a reason to rerun old LoRA.

```bash
export E2_ARM=N E2_SEED=42 E2_LABEL=assets
bash scripts/train/run_native_constrained_transfer.sh preflight
# After qualified assets and chosen GPU/time cap exist:
export SINGLE_TABLE_GPU_AUTHORIZED=YES
unset E2_ADAPTER
export E2_LABEL=native_original
bash scripts/train/run_native_constrained_transfer.sh native
bash scripts/train/run_native_constrained_transfer.sh evaluate
export E2_NATIVE_BASELINE="$EVQ_WORK_DIR/e2_native_original_native"
export E2_TASK_BASELINE="$EVQ_WORK_DIR/e2_native_original_validation"
bash scripts/train/run_native_constrained_transfer.sh cache
export E2_ARM=Z E2_LABEL=Z_s42_smoke EVQ_MAX_SECONDS=600
bash scripts/train/run_native_constrained_transfer.sh smoke
export E2_REFERENCE_RUN="$EVQ_WORK_DIR/e2_Z_s42_smoke"
export E2_RESUME="$E2_REFERENCE_RUN/step_001" E2_LABEL=Z_s42_smoke_resumed
bash scripts/train/run_native_constrained_transfer.sh smoke-resume
export E2_RESUMED_RUN="$EVQ_WORK_DIR/e2_Z_s42_smoke_resumed"
bash scripts/train/run_native_constrained_transfer.sh compare-smoke
# Only after actual nonzero gradients, save/reload parity and measured cost pass:
export E2_ARM=N E2_LABEL=N_s42_first EVQ_MAX_SECONDS=3600
bash scripts/train/run_native_constrained_transfer.sh train
export E2_ADAPTER="$EVQ_WORK_DIR/e2_N_s42_first/step_064"
bash scripts/train/run_native_constrained_transfer.sh native
bash scripts/train/run_native_constrained_transfer.sh evaluate
bash scripts/train/run_native_constrained_transfer.sh review
```

Read `e2_N_s42_first_review.json`, checkpoint receipts and raw errors. The
[CPU reviewer](../../../../scripts/analysis/review_native_constrained_transfer.py)
re-scores raw full outputs, checks paired deployment/data identities, recomputes
Native ratios and source-group intervals, and derives the same Native-compact
cohort for near/far comparisons. It writes a next action; it does not launch,
retune, seal, or promote claims. Retain unfiltered scores alongside the cohort.

For a qualified continuation only: set a fresh `E2_LABEL=N_s42_to96`,
`E2_RESUME` to the prior step064 and `E2_STOP_STEP=96`, then run `resume`.
Repeat Native/evaluate/review with `E2_ADAPTER` set to the new step. Continue
to128 only under the same rules. Z/Y use their frozen manifest amplitudes
automatically; change `E2_ARM` and labels explicitly, never inherit an adapter
into a new arm. Output directories refuse overwrite. Reuse original baselines
and the teacher cache across arms. `cache-resume` repairs an interrupted cache;
it does not regenerate successful shards. A time cap preserves the last complete
optimizer boundary; a mid-step crash may require replay from the preceding
saved boundary. No bitwise resume claim exists before the GPU parity check.
`smoke-resume` repeats the transfer step from the original restoration checkpoint;
`compare-smoke` compares final LoRA tensors on CPU at predeclared atol 1e-6 / rtol 1e-5.
Both runs must independently pass saved/merged greedy parity. This qualifies that
small boundary, not arbitrary future resumptions. Record actual max difference;
drift is a runtime issue to resolve before the main matrix, not a method failure.

| Review result | Next session's bounded task |
| --- | --- |
| Missing/mismatched source, cache, matrix or runtime parity | Repair that asset/implementation, preserve invalid receipts; do not spend on training |
| `STOP_NATIVE_DAMAGE` | Close this checkpoint/candidate under this budget; inspect earlier feasible checkpoints and FFN/attention conflict logs; no farther cells/extra seeds |
| `UNRESOLVED_CONTROLS` | Fix resolving power without selecting by candidate success |
| `UNRESOLVED_LOCAL_OR_BACKGROUND` | Inspect compact/near full outputs: acquisition/format, forgetting or background burden; no distance-only conclusion |
| `STOP_ZERO_LONG_GENERATION` with controls resolving | Close this candidate/protocol and remaining matrix; a class-wide impossibility is unsupported |
| `RETENTION_INTERVAL_UNRESOLVED` | Use the predeclared independent confirmation pool/precision audit; do not call non-significance retention |
| Valid nonzero generation and Native retention at64/96 | Review raw outputs and continue the same recipe to the next planned checkpoint |
| Feasible completed128 | Select among predeclared feasible saves by 16K far task macro, then calibration KL, then earlier step; seal before blind32K/64K |
| N/Z/Y gain similarly | Training recipe worked; no Z-specific advantage. Preserve support/allocation paper core and narrow transfer claim |
| Z retains Native and beats matched N/Y | Replicate43/44, then blind reach; only afterwards run parameter-matched attention-only placement control |

Use step32 as restoration-only evidence, not another training run. Always report
step0/32/selected outcomes so restoration is not credited to long supervision.
Validation observations can stop work; no outcome authorizes new hyperparameters.

## 8. FFN review: learning and forgetting must both be observable

The module hypothesis and derivation are in the [theory owner §9](../theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md#9-ffn-review-learning-under-a-native-constraint).
At transfer steps33/64/96/128 the trainer records attention and FFN task-gradient
norm, weighted-Native-gradient norm, ratio and cosine **before clipping**.
It also records answer+EOS CE, the per-view minimum gold margin and EOS margin,
then four-stratum held-out calibration KL/max-prefix-KL/argmax agreement at saves.
These are diagnostic aggregates, not certificates or actual Adam update vectors.
The smoke checks a real update; task gradients must reach FFN B matrices even
though A gradients can be zero at zero-B initialization. Native zero-gradient
restoration on N is valid. Frozen backbone/norm/head leakage is a runtime error.

Healthy optimization requires task improvement AND Native retention. Large
Native/task ratio with negative cosine and stagnant generation flags a possible
constraint conflict; it does not by itself prove the .02 target infeasible.
Small training loss with bad held-out C/N/F generation flags failed transfer,
overfit or task validity; increasing rank/epochs is not the default response.
Low replay KL with falling held-out Native generation is inadequate coverage,
not successful preservation. No automatic KL relaxation, extra FFN LR, gradient
surgery or regularizer sweep is hidden in this recipe.

The full training recipe is at most 8,650,752 student input tokens before
calibration, generation and runtime probes; the first 64-step segment is at most
3,575,808 training input tokens (about 41% of the full recipe). Cache and baselines are shared. Estimate total cost from
actual mixed-length smoke throughput, including cache, validation and restart
overhead; do not transplant old QKVO tokens/s into an all-linear guarantee.
The ROI is resolving alternatives with bounded spend, not a numeric probability
of acceptance. Data qualification is currently the largest readiness risk.


## 9. YaRN-inspired sampled-prefix companion — execution amendment

This is a new prospective arm, not a rewrite of the completed N recipe. Exact
switch: `--prefix-lm` on the constrained trainer. It fixes prefix CE weight0.1
and at most128 uniformly sampled prompt next-token positions per view; answer
CE/margin, Native KL, rank, data, optimizer and table remain unchanged. Its
recipe stores the switch and sample-set hash; no resume into or out of the arm.

```bash
# Use the already verified private paths from the work-machine run bundle.
# This invokes only the two-step test; it is not a formal training result.
"$EVQ_PYTHON" scripts/train/train_single_table_native_constrained.py smoke   --checkpoint "$EVQ_CHECKPOINT" --checkpoint-contract "$CHECKPOINT_CONTRACT"   --tasks "$TASK_MANIFEST" --native-pool "$NATIVE_POOL"   --teacher-cache "$TEACHER_CACHE" --arm N --seed 42 --prefix-lm   --authorized --max-seconds 900 --output "$PREFIX_SMOKE_OUTPUT"
```

Work-machine CPU tests check sampled-gradient expectation and safe prefix indices,
plus the exact-zero Native KL invariant. A GPU smoke must establish finite joint
loss, actual attention/FFN updates, headroom and saved-model generation parity.
Only after resolving natural controls may a fixed full recipe compare its
complete generation and independent Native endpoints with answer-only training.
If only format or NLL improves, do not label it semantic/RoPE transfer.

The zero-training unit-amplitude companion is a separate two-point **component
diagnostic**: exact Z and Y frequency bytes, gain fixed to Native1, original
Native-selection rows and evaluator,1800s cap each. It is not a gain sweep or a
new frequency fit. Original-amplitude failures remain valid; only a separately
qualified new candidate may proceed to a farther matrix.
