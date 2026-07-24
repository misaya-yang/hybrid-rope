# Reviewer 27bE: schedule-shape and held-out-scale experiments

Status: spec/code present; no training result is stored in this repository.
Remote data, preflight, and GPU state must be verified independently.

This package answers Reviewer 27bE's requests without assuming or authorizing
the broader 1.5B/RULER planning track. It is also independent of
`rebuttal/rebuttal_0723/experiments/fmrope_125m_l256/`.

## Scientific contract

- Data: a deterministic FineWeb-Edu prefix from shard `000`, with evaluation
  on separate shard `004`. Selection and test windows are fixed, disjoint, and
  non-overlapping.
- Model: the repository's historical “125M” decoder configuration. Its exact
  tied-embedding count is 151,898,880 parameters.
- Pairing: every arm at a given seed has identical trainable initialization,
  token prefix, row order, optimizer, and evaluation windows. Only
  `inv_freq` changes.
- Scientific identity is enforced through explicit protocol, data-prefix,
  schedule, initialization, and anchor hashes. An execution-file hash is
  recorded for diagnosis but is not an artificial cross-run acceptance gate.
- Checkpoints persist the actual float32 `inv_freq` buffer. Loading is strict:
  every shared-buffer key, the NPY sidecar, and metadata must have the same
  SHA-256; constructor frequencies are never accepted as a silent substitute.
- Metric: teacher-forced causal NLL. The primary extrapolation metric is the
  last-128-target NLL of each fixed window; full-window NLL remains secondary.
- Execution optimizations (`torch.compile` default mode without forced CUDA
  Graph capture, BF16, fused AdamW, SDPA/Flash
  attention, worker count, cache paths) are recorded but are not scientific
  variables.

## Suite A: `shape_l128`

This retains the submitted Primary-II scale: `L_train=128`, 15M requested
tokens (14,974,976 actually consumed after complete global batches),
`base=500K`, `d_head=64`, global batch 256, and seeds 42/137/256.

The seed-42 tau scan is pre-registered over
`{0,1,2,3,4,5,d/sqrt(L),6,7}`. Selection uses only mean tail NLL over
`L={1024,2048,4096,8192}` on the selection windows. Test windows are never
read by `select-tau`.

For \(K=d_{\mathrm{head}}/2\), the three named identities are:

- `Std-Geo`: \(u_k=k/K,\ \omega_k=b^{-k/K}\);
- `Paper-Geo`: \(u_k=(k+\tfrac12)/K,\
  \omega_k=b^{-(k+1/2)/K}\);
- `EVQ-Cosh`: the same Paper-Geo \(u_k\), with
  \(\phi_k=1-\operatorname{asinh}((1-u_k)\sinh\tau)/\tau\) and
  \(\omega_k=b^{-\phi_k}\).

Thus `EVQ-Cosh(tau=0) = Paper-Geo`, not Std-Geo. At `d_head=64, b=500K`,
Paper-Geo multiplies every Std-Geo frequency by `0.8146172338565447`,
equivalently multiplying every wavelength by `1.2275703955658044`.
This is a global half-channel shift, not an exact change to one standard RoPE
base. Exact float32 values and hashes are frozen in
`../../theory_results/FREQUENCY_DEFINITION_MANIFEST.json`.

The three-seed shape attribution compares:

- Paper-Geo (`tau=0`);
- a uniform log grid matched to the rule-EVQ endpoints/span;
- EVQ-Cosh at `tau=d/sqrt(L)`;
- power and exponential warps matched to the same endpoints, span, and RMS
  interior deformation as rule-EVQ.

Std-Geo is one seed-42 small ablation only. The main comparison remains
Paper-Geo versus EVQ-Cosh; Std-Geo is not substituted into historical table
claims. This separation prevents the Paper-Geo half-step from being mislabeled
as a Cosh-shape effect.

## Suite B: `heldout_b1m_d128`

This is held out from the submitted `base=500K, d_head=64` configuration:
`base=1M`, `d_head=128`, six attention heads, `L_train=512`, 50M requested
tokens (49,995,776 consumed), and three seeds. The two arms are Paper-Geo and
EVQ-Cosh at `tau=d/sqrt(L)` on the same midpoint quantizer. Evaluation extends
through 16K; Std-Geo is intentionally confined to the smaller Suite A
ablation.

## Claim boundary

These are fresh matched ablations. They must not be numerically merged with the
historical Primary-II table because the evaluation source is now an explicitly
separate FineWeb-Edu shard. They can support mechanism attribution and
held-out-configuration robustness, not a universal long-context/SOTA claim.

## CPU preparation

After the FMRoPE session has produced its verified manifest:

```bash
export SOURCE_MANIFEST=/path/to/fmrope/data/data_manifest.json
export WORK_DIR=/path/to/reviewer27be_shape_base

python -m rebuttal.rebuttal_0723.experiments.reviewer27be_shape_base.prepare \
  --source_manifest "$SOURCE_MANIFEST" \
  --output_dir "$WORK_DIR/data"

bash rebuttal/rebuttal_0723/experiments/reviewer27be_shape_base/run_5090.sh preflight
```

The preparer creates experiment-owned hardlinks (or copies across filesystems)
of the token tensors, so later cleanup of the FMRoPE work directory cannot
remove the underlying data while this experiment still references it.

When GPU mode is deliberately enabled:

```bash
bash rebuttal/rebuttal_0723/experiments/reviewer27be_shape_base/run_5090.sh shape
bash rebuttal/rebuttal_0723/experiments/reviewer27be_shape_base/run_5090.sh heldout
```

The launcher refuses GPU commands without a visible BF16 GPU with at least
30 GiB, runs one discarded compile/forward/backward probe before creating any
training run, locks the work directory against duplicate launches, resumes only
at completed run boundaries, and shares a persistent TorchInductor cache.
The 2GB CPU-only container does not instantiate a full 151.9M model during
preflight; the actual initialization hashes are compared across arms per seed
when results are summarized.
