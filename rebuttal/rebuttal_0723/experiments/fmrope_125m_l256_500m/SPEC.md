# Exact-range FMRoPE vs Cosh: L=256 / 500M-token protocol

Status: conditional follow-up to the registered 100M-token diagnostic.

This run preserves the parent experiment's model, three frequency arms, seed,
FineWeb-Edu source shards, optimizer family, batch sizes, evaluation anchors,
and NLL metrics. Only the training horizon changes:

- requested tokens: 500,000,000;
- consumed tokens: 499,974,144 complete global batches;
- optimizer steps: 7,629;
- warmup steps: 762 (the same approximately 10% schedule fraction).

The three arms are Paper-Geo base 500K, paper-faithful local FMRoPE
base 256, and EVQ-Cosh tau 4 / base 500K. The validation source remains the
disjoint pinned shard 004. Results must be reported separately from the 100M
run and remain a single-seed, 3.29-token-per-parameter diagnostic rather than a
training-saturation claim.

## Exact-range add-on

`anchored_cosh_tau4_fmrope_range` is the only new training arm. It preserves
the native geometric sampled endpoints/span used by FMRoPE and replaces only
the interior spacing with endpoint-normalized Cosh at `tau=4`. Training and
target-length inference use the existing FMR range mapping.

The seed-42 FMRoPE and matched-Cosh checkpoints are complete. The registered
confirmation uses seeds `137` and `256`; each seed trains exactly two arms,
`fmrope_base256` and `anchored_cosh_tau4_fmrope_range`, because no matching
500M FMRoPE checkpoints exist for those seeds. Data, evaluation anchors, and
all non-seed protocol fields remain fixed. Inference anchors are not treated as
independent seeds.

This is a post-submission three-seed matched-range diagnostic, not a
replacement for the submitted midpoint EVQ implementation. It supports a
reviewer-facing quantitative claim only after aggregation across training
seeds.

## 350M single-seed scale transfer

The scale-transfer group compares only with the 151.9M exact-range protocol; it
does not reuse or claim continuity with historical runs named “350M.”

- Exact parameter count: `350,112,000`.
- Architecture: the 151.9M width, 12 heads, `d_head=64`, MLP width, vocabulary,
  and \(K=32\) are unchanged; depth alone increases from 12 to 33 layers.
- Seed: `42`.
- Arms: `fmrope_base256` and `anchored_cosh_tau4_fmrope_range`.
- Training: `L=256`, 999,948,288 consumed tokens, global batch 256,
  micro-batch 64, BF16/Flash-only/compiled/fused-AdamW. The stream is two
  contiguous, non-overlapping 499,974,144-token segments:
  `[0,499974144)` and `[499974144,999948288)`.
- Optimizer: both 350M arms use the same registered `3e-4` peak and `3e-5`
  minimum learning rates.
- Evaluation: the same 32 frozen anchors, lengths, tail-128 NLL, and
  fixed/target-retargeted conditions as the 151.9M experiment.

The only within-350M arm difference is the immutable frequency table. This
remains a single-seed supporting scale-transfer result.

## Registered continuation

Prepare both seed receipts before enabling the paid GPU:

```bash
FMR_WORK_DIR="$WORK_DIR" ./run_5090.sh preflight-multiseed
```

The GPU command runs only the four missing trainings and their evaluations:

```bash
FMR_WORK_DIR="$WORK_DIR" ./run_5090.sh run-multiseed
```

Afterward, combine those results with the completed seed-42 comparison:

```bash
FMR_WORK_DIR="$WORK_DIR" \
FMR_SEED42_COMPARISON="$SEED42_COMPARISON" \
  ./run_5090.sh aggregate-multiseed
```

For the 350M group, prepare the registered 1B stream and run the CPU receipt:

```bash
FMR_MODEL_TIER=350m \
FMR_WORK_DIR="$WORK_350M" \
  ./run_5090.sh prepare-350m
FMR_WORK_DIR="$WORK_350M" ./run_5090.sh preflight-350m
```

The paid-GPU command is:

```bash
FMR_WORK_DIR="$WORK_350M" ./run_5090.sh run-350m
```
