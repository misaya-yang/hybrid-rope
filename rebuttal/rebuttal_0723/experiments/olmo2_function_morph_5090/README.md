# OLMo-2 finite function-morph audit

Status: `PREPARED_NO_GPU`; this directory is not experimental evidence.

## Question

Does the OLMo-2 attention-distance-derived phase-chord table preserve the
mature model's 4K function more cheaply, while improving 8K/16K natural-text
NLL more strongly, than (a) a non-attention-aware table with the same sampled
support and RMS log-frequency displacement and (b) anchored EVQ-Cosh?

The audit is deliberately smaller than a full Fisher/Lanczos program.  It
measures the actual finite replacement path for each candidate instead of
using phase activity as a proxy for replacement damage.

## Frozen arms

- `phase_chord_olmo_r0_lambda_0p1`: global OLMo-2 4K R0 attention mass,
  `1-cos(omega*Delta)`, lambda 0.1, cube-root companding.
- `matched_exponential_control`: endpoint-inclusive exponential coordinate
  warp, bent opposite to phase-chord and matched only on RMS log-frequency
  displacement.  It uses no attention measurements.
- `anchored_evq_cosh_tau_2`: endpoint-inclusive EVQ-Cosh with
  `tau=head_dim/sqrt(4096)=2`.

Each target shares the exact Native sampled endpoints.  The finite log-space
morph grid is `t={0,.05,.25,.5,.75,1}`.  The evaluator records per-example
Native-teacher forward KL and NLL delta over the final 64 tokens at 4K, 8K,
and 16K, using four deterministic pure-text rows per length.  This is
teacher-forced evidence only.

## No-GPU preparation

Create the two missing deterministic views with
`scripts/analysis/prepare_attention_demand_text.py`, then run:

```bash
python -m rebuttal.rebuttal_0723.experiments.olmo2_function_morph_5090.dry_run \
  --model-dir "$MODEL_DIR" \
  --r0-collection "$R0_COLLECTION" \
  --view-4096 "$VIEW_4096" \
  --receipt-4096 "$RECEIPT_4096" \
  --view-8192 "$VIEW_8192" \
  --receipt-8192 "$RECEIPT_8192" \
  --view-16384 "$VIEW_16384" \
  --receipt-16384 "$RECEIPT_16384" \
  --output-dir "$OUTPUT_DIR"
```

The dry-run parses configuration and R0 arrays, loads only CPU token tensors,
hashes assets, freezes every target/morph tensor, and writes
`target_manifest.json` plus `dry_run_receipt.json`.  It never deserializes the
checkpoint, creates an optimizer, enables gradients, or initializes CUDA.

## Future GPU entry point

The evaluator fails before importing Torch unless both gates are present:

```bash
OLMO_FUNCTION_MORPH_GPU_AUTHORIZED=1 python -m \
  rebuttal.rebuttal_0723.experiments.olmo2_function_morph_5090.run_audit \
  --preflight "$OUTPUT_DIR/dry_run_receipt.json" \
  --output "$OUTPUT_DIR/raw_audit.json" \
  --authorize
```

That command is for a later explicitly authorized GPU turn.  It is
inference-only, freezes all parameters, forces BF16 Flash-only SDPA, retains
per-example rows, and restores Native `inv_freq` on exit.  No silent math or
memory-efficient attention fallback is allowed.

## Decision boundary

The phase-chord route advances only if its measured 4K cost/far-NLL tradeoff
dominates the displacement-matched control and is competitive with anchored
EVQ-Cosh.  A finite positive result would justify a later sparse adaptation
gate; it would not establish a Fisher optimum, continuous basin, statistical
significance, autoregressive retrieval, or a general retrofit solution.
