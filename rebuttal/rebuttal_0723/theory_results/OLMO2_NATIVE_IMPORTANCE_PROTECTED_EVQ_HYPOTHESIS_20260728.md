# OLMo-2 Native-importance-protected EVQ retrofit

Date: 2026-07-28
Status: `DESIGN_ONLY_OFFLINE_IMPLEMENTED_NOT_GPU_READY_NOT_EXECUTED`
Method scope: mature OLMo-2 1.485B retrofit research, not submitted EVQ-Cosh
Rebuttal role: optional future evidence only; the current sendable response
does not depend on this experiment

## 1. Decision

The original idea—reserve one analytic band at period
\(2.205L_{\mathrm{train}}\) and allocate every other band with Cosh—is **not
accurate enough to launch as the primary experiment**.

LeRoPE supplies a valuable hypothesis: a trained model can concentrate
positional function in a small frequency subset, and preserving a good fixed
table can retain much of the learned-frequency benefit. It does not establish
that:

- OLMo-2 Native RoPE relies on LeRoPE's learned \(2.205L\) band;
- the presence of one nearby period is sufficient;
- one universal analytic constant transfers across model, data, objective,
  base, head dimension, and frequency-grid convention; or
- a mature Native model can be changed to that table without adapting its
  learned Q/K routing.

The canonical midpoint EVQ grid already has a channel near this scale in the
4K setting. Therefore “missing the \(2.205L\) frequency” cannot by itself
explain the observed 4K RULER loss. The stronger question is functional:

> Does the untouched Native model place most of its actual 4K post-RoPE
> attention function in a small, stable rotary-pair subset, and can preserving
> that subset reduce mature-model retrofit loss while EVQ remains active in
> the remaining subspace?

This revised hypothesis is sufficiently precise and falsifiable to test. It
is not established evidence.

## 2. Existing facts that constrain the design

1. Full mature-model frequency replacement is not a harmless coordinate
   change: 63/64 OLMo rotary pairs change, and a static position-independent
   Q/K map cannot exactly conjugate two different RoPE generators at every
   distance.
2. Selective Q/K continuation recovers held-out 2Wiki 4K capability almost to
   the Native level (`22.0%/21.5%` Native/EVQ exact) while retaining an EVQ
   8K advantage, but 13-family RULER remains `72.19%/42.44%` at 4K.
3. Small natural-text NLL changes do not bound sparse routing capability.
4. Historical direct-hybrid receipts were invalid because Native and EVQ
   frequency buffers aliased. Every new tensor must be built from independent
   clones and hash-checked.
5. LeRoPE is concurrent, in-window-dominant learned-frequency work. Its
   dominant-band result motivates measuring concentration; it does not validate
   EVQ extrapolation or this retrofit.

Primary local owners:

- `OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md`
- `OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`
- `LEROPE_CONCURRENT_WORK_NOTE_20260728.md`

## 3. Registered hypothesis

### H-PROTECT

On held-out natural 4K sequences, the untouched Native model's causal
attention response to removing one post-RoPE rotary pair is concentrated in
at most 16 of 64 pairs, stable across two row halves, and those pairs explain
material importance in every layer.

If H-PROTECT passes, a calibrated hybrid can reduce 4K retrofit damage:

- protected pairs retain the exact Native frequencies;
- every unprotected pair receives the exact EVQ frequency;
- Q/K LoRA has exactly zero direct output on both coordinates of every
  protected pair;
- V/O and the backbone stay frozen;
- the student matches the Native teacher's actual causal QK attention and
  `A@V` context at all 16 layers using only contiguous physical 4K natural
  sequences.

The protected frequencies and direct LoRA output coordinates are exact. This
does **not** make deeper protected activations structurally identical to
Native, because earlier-layer changes can alter later hidden states. Broad 4K
capability must still be measured.

### Matched causal control

A full-EVQ control uses the same checkpoint, data rows, row order, objective,
Q/K-only rank-512 LoRA, optimizer, budget, and evaluator. Its protected set is
empty. This distinguishes the effect of protection from the effect of the new
all-layer attention-restoration objective.

To minimize GPU cost, the protected arm is screened first. The full-EVQ
matched control is required before attributing a successful result to
protection, but need not be trained if the protected arm itself fails the 4K
gate.

## 4. Stage D — Native functional-band diagnostic

For a sampled causal query row, let

\[
S=\sum_{p=1}^{64}s_p,\qquad P=\operatorname{softmax}(S).
\]

Removing pair \(p\) gives \(P_{-p}=\operatorname{softmax}(S-s_p)\). The exact
forward KL is computed without another model forward:

\[
\mathrm{KL}(P\|P_{-p})
=\mathbb E_P[s_p]+\log\mathbb E_P[\exp(-s_p)].
\]

The diagnostic uses:

- untouched Native OLMo-2 1.485B;
- 16 held-out full-4K LongAlign rows (`split != 0`);
- 16 deterministic query positions spanning 512–4095;
- every attention head and all 16 layers;
- post-RoPE Q/K values;
- no training, optimizer, downstream rows, or long position IDs.

Native/EVQ pair 0 is already exactly identical and therefore does not consume
the protection budget. Among the 63 actually changed pairs, the smallest set
reaching 80% of the aggregate leave-one-pair-out KL score is selected. The
diagnostic passes only if:

1. no more than 16 pairs are needed for 80% aggregate score;
2. each alternating row half also needs no more than 16 pairs;
3. split-half importance-score cosine similarity is at least 0.98;
4. split-half selected-set Jaccard is at least 0.60; and
5. the selected global set captures at least 50% of the score in every layer.

The normalized sum of individual ablation KL values is a selection score, not
an additive causal decomposition. Thresholds are fixed before observation.
Failure ends the method; the set, threshold, or row split must not be tuned
after looking at the result.

## 5. Stage E0 — immediate no-training screen

After a passed diagnostic, evaluate untouched Native and the immediate
protected hybrid at 4K before creating an adapter:

- 2Wiki: 50 fixed held-out rows;
- RULER: 5 fixed rows for all 13 families;
- natural NLL: 64 fixed held-out rows.

If the immediate hybrid already passes the full registered 4K margins, skip
restoration training and proceed directly to frozen 8K/16K evaluation. If it
does not pass, preserve this zero-adapter baseline and permit the single
restoration run below. The immediate result is not used to change the
protected set.

## 6. Stage G/R — one bounded restoration run

After a passed Stage D receipt:

- Native teacher and hybrid student start from the same immutable checkpoint;
- student frequencies are fixed from step 0;
- trainable scope is masked Q/K LoRA only, all 16 layers;
- rank 512, alpha 1024, dropout 0;
- data are hash-bound natural LongAlign rows, exactly 4K and disjoint from the
  diagnostic rows;
- positions are contiguous `0..4095`;
- objective is the mean over all layers of:
  - `KL(A_native || A_student)` for actual causal post-RoPE QK attention;
  - normalized MSE of per-head `A@V` context;
- no CE, output-logit KL, task supervision, virtual positions, or frequency
  morph;
- 144 optimizer steps, micro-batch 1, accumulation 4;
- fused AdamW, LR `2e-5`, 4 warmup steps, minimum LR ratio 0.9;
- no rank, alpha, LR, loss-weight, layer, or protected-set sweep.

Before training, the discarded GPU smoke must verify:

- active architecture, BF16 and Flash-only attention eligibility;
- pinned LinearARD loss/gradient parity;
- full Native teacher plus student memory fit;
- exact frequency and Q/K mask hashes;
- finite composite loss; and
- finite, non-zero gradients on all 32 Q/K LoRA-B tensors.

Any failure stops before the 144-step run.

## 7. Capability gates

Training loss is never the endpoint. Evaluate the same frozen rows and decoder
contract for:

1. untouched Native;
2. immediate protected hybrid with zero adapter;
3. trained protected hybrid;
4. after a protected-arm pass, the matched full-EVQ restoration control.

The 4K gate is:

- 2Wiki: 200 held-out rows, token-F1 drop at most 5 points versus fresh
  Native; report exact, full generation and terminal EOS;
- RULER: all 13 families, 20 held-out rows per family, macro drop at most
  10 points versus fresh Native, and no Native-positive family collapsing
  near zero;
- natural text: 128 held-out rows, NLL no more than `Native + 0.10`;
- one independent QA/MCQA retention slice with no material capability
  collapse.

Only if every 4K gate passes may the frozen adapter be evaluated at 8K and
then 16K. Long-context success requires autoregressive RULER/QA endpoints;
NLL/PPL cannot substitute.

Possible conclusions:

1. **Diagnostic fails:** OLMo Native importance is not sufficiently
   concentrated/stable; abandon the protected-subspace hypothesis.
2. **Diagnostic passes, 4K fails:** frequency/direct-output protection is
   insufficient because hidden-state and attention interactions remain
   distributed; stop.
3. **Protected passes, matched full-EVQ fails:** evidence that functional
   subspace protection, not only the restoration objective, preserves 4K.
4. **Both pass:** the attention-restoration objective is sufficient and the
   protected set is not necessary.
5. **4K passes, long tasks fail:** retention is solved only; no extrapolation
   capability claim.

## 8. Implemented files

- `experiments/olmo2_lora_maturity/native_protected_evq.py`
  - exact frequency/mask construction;
  - exact pair-ablation KL;
  - split-stability selector;
  - diagnostic attention backend;
  - all-layer restoration loss.
- `preflight_4k_native_band_importance.py`
  - no-GPU asset/code/hash receipt.
- `diagnose_4k_native_band_importance.py`
  - GPU diagnostic only; no optimizer.
- `preflight_4k_native_protected_evq.py`
  - post-diagnostic no-GPU training READY receipt.
- `train_4k_native_protected_evq.py`
  - discarded GPU smoke and one registered restoration run;
  - supports `protected` and matched `full-evq-control` arms.
- `evaluate_native_protected_evq_nll.py`
  - matched Native/immediate/trained natural-NLL evaluation.
- `evaluate_2wiki_phase_adaptation.py`
  - supports the dynamic selected frequency/mask receipt.
- `evaluate_instruct_ruler_transfer.py`
  - supports the dynamic selected frequency/mask receipt.
- `tests/test_olmo2_native_protected_evq.py`
  - exact KL parity, selector, frequency identity, mask and LoRA gates.

No GPU work has been executed. A READY receipt generated on the actual asset
host is required before any diagnostic or training.

## 9. Execution order after the server is available

Set host-local paths once:

```bash
export CHECKPOINT=<olmo2_checkpoint>
export CHECKPOINT_READY=<checkpoint_ready_receipt>
export TRAIN_VIEW=<longalign_paired_L4096>
export LINEARARD_ROOT=<pinned_linearard_checkout>
export RUN_ROOT=<new_empty_run_root>
```

Run the asset/hash preflight without a visible GPU:

```bash
CUDA_VISIBLE_DEVICES=-1 python3 -m \
  rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.preflight_4k_native_band_importance \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-ready-receipt "$CHECKPOINT_READY" \
  --training-view "$TRAIN_VIEW" \
  --diagnostic-output "$RUN_ROOT/native_band_importance.json" \
  --receipt-output "$RUN_ROOT/native_band_importance_prepared.json"
```

Then run the no-optimizer diagnostic:

```bash
python3 -m \
  rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.diagnose_4k_native_band_importance \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-ready-receipt "$CHECKPOINT_READY" \
  --training-view "$TRAIN_VIEW" \
  --prepared-receipt "$RUN_ROOT/native_band_importance_prepared.json" \
  --output "$RUN_ROOT/native_band_importance.json"
```

Only after a passed diagnostic, run the registered immediate 4K screen. If it
does not already pass every full 4K margin, prepare the protected arm offline:

```bash
CUDA_VISIBLE_DEVICES=-1 python3 -m \
  rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.preflight_4k_native_protected_evq \
  --arm protected \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-ready-receipt "$CHECKPOINT_READY" \
  --training-view "$TRAIN_VIEW" \
  --selection-receipt "$RUN_ROOT/native_band_importance.json" \
  --linearard-root "$LINEARARD_ROOT" \
  --gpu-ready-receipt "$RUN_ROOT/protected_gpu_ready.json" \
  --run-output "$RUN_ROOT/protected_run" \
  --receipt-output "$RUN_ROOT/protected_prepared.json"
```

The paid-GPU sequence is smoke first, then exactly one run:

```bash
python3 -m \
  rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_4k_native_protected_evq \
  --mode gpu-smoke --arm protected \
  --checkpoint "$CHECKPOINT" \
  --training-view "$TRAIN_VIEW" \
  --selection-receipt "$RUN_ROOT/native_band_importance.json" \
  --linearard-root "$LINEARARD_ROOT" \
  --receipt "$RUN_ROOT/protected_prepared.json" \
  --output "$RUN_ROOT/protected_gpu_ready.json"

python3 -m \
  rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_4k_native_protected_evq \
  --mode train --arm protected \
  --checkpoint "$CHECKPOINT" \
  --training-view "$TRAIN_VIEW" \
  --selection-receipt "$RUN_ROOT/native_band_importance.json" \
  --linearard-root "$LINEARARD_ROOT" \
  --receipt "$RUN_ROOT/protected_gpu_ready.json" \
  --output "$RUN_ROOT/protected_run"
```

Do not prepare or launch `--arm full-evq-control` unless the protected arm
passes the registered 4K capability screen. It uses the same commands and
hyperparameters with distinct receipt/output paths and only the arm value
changed.

## 10. Claim identity

If successful, call this:

> a Native-importance-protected hybrid EVQ mature-model retrofit.

Do not call it:

- pure EVQ-Cosh;
- a zero-parameter schedule;
- a validation of the \(2.205L\) constant;
- function-preserving by construction;
- unseen-task transfer; or
- evidence for the submitted method until a reviewer-relevant, matched,
  raw-backed capability owner is completed and explicitly promoted.
