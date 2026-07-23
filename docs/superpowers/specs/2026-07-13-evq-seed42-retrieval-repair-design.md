# EVQ Seed-42 Retrieval Repair Design

状态：`approved design / no new result yet`

## 1. Goal

Continue only the completed EVQ-Cosh seed-42 LongAlpaca LoRA adapter and test
the cheapest credible way to add source-dependent long-range retrieval without
destroying its language-model behavior.

This is a capability-repair experiment, not a replacement for the matched
Geo/EVQ paper lineage and not evidence that EVQ is a universal long-context
method.  No paper number or claim changes during preparation.

## 2. Why a new measurement is required first

The first passkey pilot used raw tokenized prose rather than the LLaMA-3 chat
template and treated normalized whole-generation equality as the only success
metric.  A generation containing the correct key followed by extra text was
therefore recorded as zero.  That score mixes three failures:

1. long-range retrieval;
2. instruction/output-format following;
3. stopping after the answer.

Before training, the baseline is re-scored with prompts whose chat-template
tokens are included inside the registered context budget.  Every generation
reports all of the following, without replacing one metric with another:

- strict normalized whole-output exact match;
- first extracted value exact match;
- gold-value containment;
- generated-token count and EOS termination;
- teacher-forced gold-answer NLL.

Controlled retrieval additionally uses original, source-swapped, and
source-removed triplets.  This makes source dependence, rather than answer
format alone, the primary capability evidence.

## 3. Scope and fixed identities

- Backbone: one locally manifested LLaMA-3-8B-Instruct byte identity.
- Parent: the completed step-300 EVQ-Cosh seed-42 LongAlpaca adapter only.
- Parent substrate: canonical EVQ-Cosh, `tau=1.414`, `base=500000`,
  `head_dim=128`.
- LoRA remains `r=64`, `alpha=128`, `dropout=0.05`, targeting only
  `q_proj,k_proj,v_proj,o_proj`.
- Precision remains BF16; no quantization and no new dependency.
- Optimizer state starts fresh for each registered 32-step segment; this is
  recorded explicitly and no segment is described as a continuous 64-step
  optimizer trajectory.
- No Geo arm, tau sweep, rank sweep, 32K training, LongAlpaca replay, MLP
  adapter, or full-model unfreezing is allowed in this low-cost protocol.

The parent adapter, parent frequency tensor, LongAlpaca manifest, model
manifest, tokenizer, prepared data, source code, and every produced adapter are
identified by SHA-256 in the run protocol.

## 4. YaRN identity

The implementation reuses the equations pinned from
`jquesnelle/yarn@995db5b`, including the official correction range, linear
ramp, interpolation/extrapolation blend, and cosine/sine amplitude
`mscale = 1 + 0.1 ln(s)`.

For the native endpoint geometric grid these are the official YaRN equations.
EVQ is not that grid.  For EVQ, the same pinned equations are evaluated on the
virtual channel coordinate

\[
j_v=-d\log(\omega)/(2\log(b)).
\]

Results must therefore be labeled **YaRN-derived generalization on the EVQ
substrate**, not native-grid official YaRN.  This is still the requested
official implementation path: the formula and `mscale` are retained, while the
non-geometric channel mapping is disclosed.

`original_max_position_embeddings` is fixed at 8192.  R8 uses factor 1
(identity), R16 uses factor 2, and the final 32K zero-shot test uses factor 4.
Every factor is applied afresh to the canonical EVQ substrate; factor 4 is
never applied on top of an already factor-2 frequency tensor.

Each checkpoint stores both:

- `substrate_inv_freq`: canonical unscaled EVQ-Cosh;
- `runtime_inv_freq`: the factor-specific tensor actually used for training.

The saved operator record also contains factor, `mscale`, virtual-coordinate
metadata, and tensor hashes.

## 5. Data contract

CPU preparation uses the already pinned FineWeb-Edu filler and the model's
actual tokenizer.  Training, validation, and test are disjoint in all of these
dimensions:

- nonce-key and nonce-value token pools;
- filler spans (train source versus two non-overlapping validation regions);
- deterministic seeds;
- instruction wording.

Every example is an exact token tensor.  Its registered sequence length
includes the LLaMA-3 user/assistant boundaries, answer tokens, and EOS.  Each
complete message is rendered through `apply_chat_template`; empty-user boundary
tokens and separately tokenized content must not be concatenated as a proxy.
CPU preparation verifies parity between the full rendered string tokenization
and `apply_chat_template(..., tokenize=True)` for every produced template.
Prompt labels are `-100`; only the 12-token value plus EOS is supervised.

Task mix is fixed before seeing results:

- 75% single key-value retrieval;
- 25% last-write-wins retrieval with an obsolete and a current value.

The query is at the end.  R8 samples source-to-answer-predictor distance from
2048 to 6144 tokens; R16 samples 6144 to 14336 tokens.  Each validation split
has 16 groups per length (12 KV and 4 update).  Each frozen test split has 32
groups per length (24 KV and 8 update).  Every group expands to equal-length
original, source-swapped, and source-removed records.

The test split is hashed and frozen during CPU preparation.  Checkpoint choice
uses validation only; test is opened once after the stage budget is selected.

The independent passkey gate uses five deterministic trials at each of depths
10%, 25%, 50%, 75%, and 90%: 25 examples at the stage's target length.  These
examples are not present in retrieval training.

## 6. Training stages and cost ceiling

Each optimizer step processes exactly 32,768 physical sequence tokens.

| Stage | Runtime | Seq. length | Microbatch / accumulation | First budget | Optional rescue | Tokens per segment |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R8 | EVQ factor 1 | 8,192 | 1 / 4 | 32 steps | one additional 32-step segment | 1,048,576 |
| R16 | EVQ + YaRN-derived factor 2 | 16,384 | 1 / 2 | 32 steps | one additional 32-step segment | 1,048,576 |

Both stages use answer-only causal cross-entropy, learning rate `2e-5`, four
warm-up steps per segment, cosine decay, weight decay `0.01`, maximum gradient
norm `1.0`, gradient checkpointing, and no reporting service.

The minimum path is 2,097,152 training tokens.  The absolute ceiling, if both
registered rescue segments are justified, is 4,194,304 tokens.  A rescue
segment starts from the selected 32-step adapter with a fresh optimizer and is
recorded as `segment=2`, never silently merged into segment 1 provenance.

## 7. Metrics and pre-registered gates

For each controlled triplet, evaluation reports extracted-value exact match,
strict whole-output exact match, answer NLL, task type, distance bucket, and
source-removal paired NLL delta.

Primary aggregate definitions:

- `pair_consistency`: fraction of groups for which both original and swapped
  generations return their respective source values;
- `source_removal_positive_fraction`: fraction of groups whose original target
  has higher per-token NLL after its source is removed;
- `passkey_containment`: fraction of the 25 independent passkey generations
  containing the exact gold 8-digit key;
- `temporal_delta_nll`: checkpoint mean NLL minus its parent mean NLL on the
  same frozen temporal subset and the same runtime factor.

R8 passes only if validation has:

- `pair_consistency >= 0.80`;
- `source_removal_positive_fraction >= 0.75`;
- `passkey_containment >= 0.50` at 8K, factor 1;
- `temporal_delta_nll <= 0.20` at 8K;
- finite loss, gradients, frequency tensors, and generation scores for both KV
  and update examples.

R16 passes only if validation has:

- `pair_consistency >= 0.50`;
- `source_removal_positive_fraction >= 0.75`;
- `passkey_containment >= 0.50` at 16K, factor 2;
- `temporal_delta_nll <= 0.20` at 16K;
- the same finite and subgroup checks.

If a 32-step checkpoint fails the strict gate, its one rescue segment is
allowed only when pair consistency improves by at least 0.10 absolute over its
parent, the temporal guardrail still passes, and no finite-value check fails.
Otherwise the stage stops immediately.  Failure after the rescue stops the
experiment.

R16 cannot start unless R8 passes.  The 32K factor-4 test cannot run unless R16
passes.  The launcher never advances automatically; it requires the previous
gate JSON with status `pass`.

## 8. Final evaluation order

After R16 passes and its 32/64-step budget is selected:

1. open the frozen controlled test once at R8/factor 1 and R16/factor 2;
2. run the 25-example passkey grid at 32K/factor 4 without 32K training;
3. run the already prepared official RULER/NIAH and downstream subsets;
4. run the full temporal held-out guardrail;
5. write raw per-example JSON and a compact grouped report.

If a primary capability gate fails, downstream and broad RULER runs are not
used to relabel the experiment as successful.  Their execution can be skipped
to save GPU cost.

## 9. Components and boundaries

New focused package:

- `rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/protocol.py`: immutable phase, budget,
  gate, metric, and artifact contracts;
- `rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/prepare_data.py`: deterministic
  three-way data preparation and manifest validation;
- `rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/train.py`: continuation from a
  validated EVQ parent, factor-specific runtime injection, answer-only loss,
  and atomic checkpoint provenance;
- `rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/evaluate.py`: controlled triplet,
  passkey, temporal guardrail, and gate aggregation;
- `rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/run_seed42.sh`: explicit CPU prepare,
  preflight, baseline, R8, R16, gate, and final commands.

Existing code is reused rather than copied where its contract already fits:

- `scripts/lib/rope/official_yarn.py` owns the pinned YaRN operators;
- `rebuttal/pre_rebuttal/frequency_adaptation_8b` supplies tested answer-mask,
  counterfactual tensor transforms, tail-logit, and gradient-diagnostic helpers;
  its empty-user chat-boundary splicing and token-piece exact-distance builder
  are not reused because they are not token-parity-equivalent to a complete
  LLaMA-3 chat-template render;
- `experiments/lora_evq_v2/eval_official_yarn_capability.py` keeps its existing
  matched `x2,x4` Geo/EVQ contract and is extended only with separated
  strict/extracted/containment metrics; the repair evaluator owns the separate
  one-factor `1/2/4` path and never composes factors;
- `experiments/lora_evq_v2/eval_temporal_holdout_three_arm.py` remains the
  source of the frozen temporal scoring semantics.

Tests live in `tests/test_evq_seed42_retrieval_repair.py` plus the existing
official-YaRN tests.  No existing full matched Geo/EVQ adaptation protocol is
renamed or silently changed into this EVQ-only repair experiment.

## 10. Failure handling and cost safety

- CPU preparation and validation complete before CUDA is touched.
- GPU commands fail before model loading when CUDA, manifests, adapters, data,
  prior gate status, or output paths are invalid.
- A global GPU lock prevents overlapping launchers.
- Existing output directories are never overwritten.  Files are written as
  `.incomplete`, validated, and atomically renamed.
- Each command prints its exact stage, parent hash, data hash, factor, expected
  token budget, and output directory before model loading.
- A dry-run validates phase transitions and operator hashes without loading the
  8B weights.
- Results and checkpoints live outside the repository through an explicit
  `EVQ_REPAIR_WORK_DIR`; no private server path is embedded in tracked files.
- No command downloads datasets or models while a paid GPU is needed.

## 11. Verification before upload

Local/CPU gates:

```bash
python -m pytest \
  tests/test_evq_seed42_retrieval_repair.py \
  tests/test_official_yarn_parity.py \
  tests/test_official_yarn_capability_eval.py -q

python -m py_compile \
  rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/protocol.py \
  rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/prepare_data.py \
  rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/train.py \
  rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/evaluate.py \
  experiments/lora_evq_v2/eval_official_yarn_capability.py

bash -n rebuttal/pre_rebuttal/evq_seed42_retrieval_repair/run_seed42.sh
```

The upload copies only the reviewed files.  It does not synchronize or delete
the server's dirty repository.  Server CPU preparation ends with manifest,
frequency, data-split, shell-syntax, and dry-run checks; it does not start a
training or evaluation process in no-GPU mode.
