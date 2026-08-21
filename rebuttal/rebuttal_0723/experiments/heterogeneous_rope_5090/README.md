# R3-prime: heterogeneous RoPE preflight

Status: `OFFLINE_CODE_COMPLETE / GPU_TRAINING_NOT_STARTED`

This package prepares the high-value heterogeneous-table experiment requested
as R3-prime.  It implements per-layer `tau` or arbitrary per-layer
inverse-frequency tables by cloning the existing `RotaryEmbedding` module once
per transformer block.  The MHA/GQA/MLA attention kernels are unchanged.

The current GPT implementation in
`scripts/core_text_phases/run_gqa_evq_experiment.py` shares one RoPE object
across all blocks and broadcasts a 1-D `inv_freq` table across heads.  This
package therefore gates per-head tables rather than silently pretending that a
shared table is head-specific.  A future per-head implementation must add an
explicit head-axis cos/sin contract and separately handle GQA query/KV groups
and MLA `d_rope` heads.

## R0 input contract

The parser accepts a wrapper named `r0`, `R0`, `allocation`, or
`heterogeneous_rope`, then reads model fields from that object or its `model`
child.  The following forms are supported:

```json
{
  "schema_version": 1,
  "num_layers": 24,
  "attention_type": "mla",
  "head_dim": 64,
  "d_rope": 32,
  "num_heads": 16,
  "base": 500000,
  "train_length": 8192,
  "effective_dim": 32,
  "layers": [
    {"layer": 0, "tau": 1.414},
    {"layer": 1, "m": 4.0}
  ],
  "candidate_profiles": {
    "bimodal_candidate": {"kind": "m", "values": [4.0, 2.0, 1.0, 2.0]}
  }
}
```

`m` uses the existing operating-rule conversion
`tau = m * effective_dim / sqrt(train_length)`.  A compact equivalent is
`"per_layer_tau": [ ... ]`, `"per_layer_m": [ ... ]`, or
`"per_layer_inv_freq": [[...], [...], ...]`.  Direct inverse-frequency rows
must have `rope_dim/2` finite positive values in strictly decreasing order.

## CPU-only preflight

Run from the repository root.  The command below builds the existing 350M MLA
architecture on CPU, checks the parameter/shape contract, emits realized
per-layer hashes, and performs shared-table forward parity when every layer has
the same table.  It does not call a trainer or initialize CUDA.

```bash
PYTHONPATH=. conda run --no-capture-output -n aidemo python -m \
  rebuttal.rebuttal_0723.experiments.heterogeneous_rope_5090.preflight \
  --r0-json /path/to/r0.json \
  --tier 350m --seq-len 8192 --output /tmp/r3prime_preflight.json
```

The receipt is `READY_FOR_AUTHORIZED_GATE` only for code/config/model-shape
preflight.  It is not a training-ready or scientific-result receipt.

## Dry-run matrix and result screening

`dry_run_matrix.py` is used by the preflight and can be imported independently.
It emits only declared R0 profiles plus explicit shared references; no metrics
are invented.  Future authorized result JSON can be screened with:

```bash
PYTHONPATH=. conda run --no-capture-output -n aidemo python -m \
  rebuttal.rebuttal_0723.experiments.heterogeneous_rope_5090.dry_run_matrix \
  --r0-json /path/to/r0.json --output /tmp/r3prime_matrix.json
```

```bash
PYTHONPATH=. conda run --no-capture-output -n aidemo python -m \
  rebuttal.rebuttal_0723.experiments.heterogeneous_rope_5090.analyze_results \
  /path/to/results.json --in-window-key in_window_ppl --ood-key ood_ppl \
  --in-window-direction min --ood-direction min \
  --output /tmp/r3prime_screen.json
```

The Pareto output is a descriptive two-metric screen.  The `bimodal` label is
only a deterministic separated-local-peak classifier over per-layer `tau`/`m`
values.  It is not a significance test, a causal regime claim, or permission
to expand the matrix.

## Stop conditions

- Do not start training from this package.  A separate authorized gate is
  required after the receipt and exact protocol are reviewed.
- Do not implement per-head training until the feasibility gate is replaced by
  a tested head-axis RoPE contract for MHA, GQA, and MLA.
- Do not treat a hash, forward parity, or unchanged parameter count as a model
  result.
