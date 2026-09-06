# Fixed-witness confirmation and Native-constrained transfer — v3 reviewed

> **2026-09-05 decision amendment:** Read the [Pro audit reconciliation](../analysis/PRO_REPORT_AUDIT_RECONCILIATION_20260905.md) before acting on older priorities below. Double qualification limits that family, not all valid single-evidence/restoration comparisons; the old joint unresolved verdict remains. Next proposed training is matched N_compact, then fixed-recipe Qwen Z/Y, with no prefix change. Unit-amplitude diagnostics and N128 aggregate confirmation have completed; the latter passes with a format/indexing regression. Its exposed confirmation pool cannot tune or independently confirm new variants. No new GPU run was launched by this amendment.


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

## 10. 开机后的固定比较轮次 — 2026-09-05 准备版

**状态：** 本地代码与协议准备，GPU 未启动；canonical 资产／运行验证留在工作机。
**问题：** N_compact 是否复制 N128；同模型固定 Z/Y 的旧配方能否产生 Native 可行的生成工作点。
**owner：** 本节负责前瞻协议；[审查复核](../analysis/PRO_REPORT_AUDIT_RECONCILIATION_20260905.md)负责推理边界；旧结果仍由[执行报告](../results/SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md)负责。
**修正：** 本节取代上文默认 step64／resume、全任务族共同停止与立即 prefix 的执行顺序，不改旧结果、scorer 或旧 reviewer 默认判定。

### 理论上需要分开的三个效应

对固定任务／布局分数 S，以下差分回答不同问题：

- `S(N128) − S(N_compact)`：匹配目标曝光与更新时，长输入 exposure 的增量；背景、位置及输入 tokens 同时改变，不是纯相位效应。
- `S(Z128) − S(Z0)`：固定 Z 系统在这份适配配方下的增量；包括 restoration 与 transfer，不是 FFN 独立贡献。
- `[S(Z128) − S(Z0)] − [S(N128) − S(N0)]`：系统与适配的描述性交互；必须同时看绝对分数，避免 floor/ceiling 误读。Z/Y 幅度不同，不能叫纯 allocation 因果效应。

严格分数保持完整答案＋EOS。语义、形式和终止另外标注，不能用 substring 调高主指标。Native 平均 KL 与 argmax 决策保持是不同对象：有限平均值不限制未覆盖 teacher 轨迹上的小 margin 决策。全线性适配可同时改变 evidence transport 与 FFN 的非线性映射，但非零 FFN 梯度不识别其必要性。

### 交付代码与固定参数

- [受限轮次入口](../../../../scripts/experiments/matched_transfer_round.py)：`template / prepare / run`，prepare 不调用 CUDA。
- [审阅器](../../../../scripts/analysis/review_native_constrained_transfer.py)：新增显式 `--protocol single_evidence_v4`；默认 legacy_v3 不变。
- [盲化转换审计](../../../../scripts/analysis/audit_generation_transitions.py)：`export / summarize`，人工／独立审阅者标注实际断言，歧义保留上下界。
- [边界测试](../../../../tests/test_matched_transfer_round.py)：曝光、阶段命令、停止范围、原正确保留率、语义歧义和超时退出。

训练固定 all-linear r16/alpha16，seed42，KL预算 .02，32R+96T，最终 step128，prefix off。N_compact 从原始模型开始；每个 world 的 compact 输入重复三次，保留原三桶顺序键，不把 `length_cap` 标签改写成训练实际长度。实际输入 tokens 在 prepare 中另外计算；同日程不意味着 dual 数值相同。

运行**原 N128 的 release008 引擎**，由原 `run.json` 的代码／runtime／KL hash 和已记录 training SHA 双重约束。新入口与 reviewer 放入新的代码目录，绝不覆盖旧 release。若旧目录丢失，先恢复其字节；不能顺手使用不同训练引擎。原引擎的 token 计数偏差保留在旧 complete.json，新计划准确给出 `P+G−1` 数量。

### 开机操作：不需要重读历史文件

本地准备包的当前源文件需复制到新的工作机代码目录；可从上次完整代码副本复制，再覆盖本轮三个实现文件及测试，旧 release008 保持原样。这里不使用 Git pull 或修改历史运行目录。

工作机使用已有 Conda Python，先显式设置自己的路径：

```bash
# 以下变量均为工作机私有路径；不要把填好的 config 提交到仓库。
export ROUND_PYTHON=/absolute/path/to/conda/bin/python
export ROUND_CODE=/absolute/path/to/new/code/release
export ROUND_CONFIG=/absolute/path/to/new/matched-round-config.json
cd "$ROUND_CODE"
"$ROUND_PYTHON" scripts/experiments/matched_transfer_round.py template
```

把 template 输出填入新的 JSON 文件。字段的含义和已有资产标签：

| 字段 | 应指向的现有对象 |
| --- | --- |
| python / engine_root | Conda Python 可执行文件／原 `code_release_008` 根目录 |
| baseline_run | 已完成的 `qwen_N_s42_fixed`，不是 invalid N 或 Z26 |
| checkpoint / checkpoint_contract | Qwen2.5-1.5B-Instruct 权重目录／`qwen15_contract.json` |
| tasks / native_pool | `qwen_tasks/manifest.json`／`qwen_native_pool_clean/manifest.json` |
| teacher_cache | 原始 Native full-vocabulary cache 目录 |
| controls | checkpoint_config hash 与 Qwen 匹配的固定 N/Z/G/Y manifest 目录，不能用 OLMo controls |
| native_baseline / task_baseline | **validation** 的原始 Native 完成目录；Native 数值为 NLL2.26473431、task25.5208%，task 标签 `qwen_task_baseline` |
| baseline_eval_engine | 对应原 baseline receipt 的 release007 trainer 源文件 |
| output_root | 全新输出目录，至少 8 GiB 空闲；可放系统盘，不复制大权重／cache |

不确定路径时在原 bundle 与 scratch 内按文件名和 receipt 查找；不要另下权重、另建训练集或全盘扫描。真正资产验证由 prepare 完成。模板不是已经填好的、可执行的配置。

```bash
"$ROUND_PYTHON" -m unittest tests.test_matched_transfer_round -v
"$ROUND_PYTHON" scripts/experiments/matched_transfer_round.py prepare --config "$ROUND_CONFIG"
# 将 ROUND_PLAN 指向 prepare 返回的 output_root/plan.json。
export ROUND_PLAN=/absolute/path/to/new/output/plan.json
# 用户开机并恢复实验后，以下命令才启动 GPU；这轮准备没有执行它。
"$ROUND_PYTHON" scripts/experiments/matched_transfer_round.py run \
  --plan "$ROUND_PLAN" --cases N_compact Z Y --authorized
```

可仅运行 `--cases N_compact`，读完其结果后用同一 plan 运行 `--cases Z Y`；已存在的 case 不会被覆盖或自动续训。若需要断开终端，让现有终端管理器保持运行即可；不要复制启动同一 plan 的第二个 GPU 进程。

prepare 做一次完整 CPU 资产/tokenizer/cache 检查并冻结依赖 hashes；核对 768 个目标及曝光顺序、448 条 replay（R256/T192）与原 N128 receipt。它输出 `PREPARED_CPU_ONLY_GPU_PENDING`，不冒充 GPU ready。run 再核对代码／资产／命令未变与空闲 GPU。BF16、Flash-only 和数值检查由原验证过的 runtime 执行，5090 必须遵循 [Blackwell profile](../../../../docs/overview/RTX5090_BLACKWELL_PROFILE.md)。不开 math attention fallback，不重新扫执行参数。

### 自动执行到哪里，以及如何结束

- N_compact：训练128 → Native validation → 全384行自然 validation → CPU review。
- Z/Y：训练128 → Z0/Y0 Native与自然 validation → R32 Native诊断 → final128 Native与自然 validation → CPU review。
- 每个 GPU 训练阶段上限3600秒；Native阶段900秒；任务阶段1800秒。每个进程另有120秒清理余量。CPU review上限600秒。全排期这些是保守硬上限，不是预期总耗时；原N的28.7分钟不构成其他臂时长保证。
- 一次只跑一个GPU进程；不自动启动后续 seed、prefix、8B、32K/64K或确认集；全轮结束后退出进程，不自动关闭主机。
- 数值／数据／加载／hash／进程失败会保存日志及 `execution.json`，停止剩余队列。部分输出禁止当完成结果；已有 checkpoint 不自动 resume。
- Native点值失败或主生成零分会由review标注，并关闭该候选的远端确认；已登记的小比较结果保留，其他候选可继续。Double不足不会让其他候选自动停止。
- 如果磁盘不足，只更改尚未开始的 output_root 到有空间的位置并重新 prepare；不删旧原始证据。prepare失败留下的目录保留为诊断，换新目录重试。

### 内容审计可以与 GPU 训练并行

```bash
"$ROUND_PYTHON" scripts/analysis/audit_generation_transitions.py export \
  --checkpoint /absolute/path/to/qwen --tasks /absolute/path/to/tasks/manifest.json \
  --baseline /absolute/path/to/N0_task_validation \
  --candidate /absolute/path/to/N128_task_validation --output /absolute/path/to/new/audit
```

导出前重算原始token完整输出／EOS并绑定原prompt及truth。审阅者只接收 `rubric.json`、`prompts.json`、`cases.jsonl`；`private_mapping.json` 在标签冻结前不提供。复制 cases 到 annotations 文件，填完 semantic_correct、format_compliant 和理由后：

```bash
"$ROUND_PYTHON" scripts/analysis/audit_generation_transitions.py summarize \
  --export /absolute/path/to/audit --annotations /absolute/path/to/annotations.jsonl \
  --output /absolute/path/to/new/annotation-results
```

明确 exact 成功自动填已有真值标签；其余内容不由 substring 自动判分。该审计不必阻断 N_compact 的固定训练，但解释内容迁移必须等待标签完成。审阅者即使隐藏臂身份，也可能认识旧案例，所以只称回顾性盲化标注，不称独立确认。

### 结果到后续工作的映射

| 结果 | 下一步／论文修改 |
| --- | --- |
| N_compact 复现大部分严格提升，内容差分也无明确增量 | 接受低成本任务适配解释；报告配对差和区间，不将“不显著”写成等价；收缩长训练机制叙事 |
| N128 有额外 far 内容增量且 compact 接近 | 保留长输入适配主张；仍不归因 FFN／RoPE 必要性 |
| Z/Y 至少一臂 Native 与主生成点值可行 | 冻结最终128及新确认协议；先新Native确认，再预先冻结的 farther测试；旧N128 confirmation不能调参 |
| Native CI 不够窄但点值合格 | 未确认；允许准备新独立确认，不再无限检查同一个pool或重选checkpoint |
| 固定Z/Y预算都不满足 Native | 停本配方候选；针对真实teacher决策轨迹覆盖设计一个变化，不自动加训练步或扫KL |
| Native可行而内容迁移仍不足 | 再登记 N/Z × prefix off/on；仅增加固定辅助LM目标，不同时改模型/rank/replay |
| Y 不逊 Z | 如实报告系统对照；allocation理论贡献仍由原fixed-support研究支撑，不包装普通适配为特定Z突破 |

Native逐任务 lost/gained 与原正确保留率必须与总体分数一起读。88%旧总体门槛保留；77.05%格式退化不被平均值抵消。新方法若主张“各项保持”，须在新确认之前另定逐项标准。Qwen64K也只等于配置Native32K的2×；这轮不会闭合原始物理Native2×/4×训练及8×/16×/32×外推路线。

### 离线 review 与验证记录

本轮检查了三条路径：配对曝光能否与原 receipt 核对；失败／中断是否保留产物并终止自有进程；审阅器是否在保留 legacy 判定的同时限制 double 的停止范围。额外检查了 Qwen controls 的 Native basis/config、旧 Native confirmation 不进入启动命令、训练最终128固定以及真实token预算与EOS。

本地新增9项机械边界测试与既有24项训练协议、2项确认审阅测试通过；26项仓库导航检查通过，共61项。文档导航检查最初发现冷启动篇幅与旧入口措辞问题，已缩短HANDOFF并恢复明确的不可变文件／工作机边界，未放宽测试。CLI帮助、标准库导入和 `git diff --check` 通过。

**仍待开机验证：** 原release008、真实模型／cache／manifest是否可访问且hash一致；工作机Python依赖；BF16/Flash、显存与磁盘；新case的实际完整训练及生成结果。语义审计工具已准备，真实标签尚未生成。本轮未安装环境、未SSH运行、未GPU计算、未改训练器或TeX/PDF、未提交推送。以上“准备完成”不表示端到端实验已通过。
