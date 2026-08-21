# R4' mature OLMo retrofit preparation

Status: PREPARED_NO_GPU_DEFAULT_DISABLED.

This directory is an independent preparation bundle for a possible 1.485B
OLMo-2 mature-model retrofit.  It is not paper evidence, does not alter the
submitted EVQ-Cosh method, and does not authorize a GPU run.

The bundle was prepared after reading:

- the seven-arm non-RULER negative search;
- the exact Native-to-EVQ transplant obstruction;
- the matched Q/K-only owner;
- the attention-restoration and far-only residual specifications;
- the Native-importance protected-subspace hypothesis; and
- the in-window preservation route audit.

## Hard boundaries

- run_5090.sh accepts only dry-run; every other action stops.
- config.json has training, GPU, download, network, and attention-fallback
  switches disabled.
- No model or dataset is downloaded or loaded.
- The historical linear-morph arms A1-D1 are represented only as a disabled
  reproduce-only contrast.  No morph implementation is added.
- The transplant theorem remains an exact obstruction for changed frequency
  multisets.  No finite slow-band or protected-table design is described as
  making the theorem vacuous.
- Any positive result from a future route would be a new mature-model
  attention operator, not validation of submitted full-table EVQ-Cosh.

## Prepared routes

### A. Historical negative reproduction

NegativeMorphReproduction records A1, A2, B1, B2, C1, C2, and D1 with
enabled=false, no training entrypoint, and a historical-negative-only claim
role.  Validation rejects attempts to turn this field on.

### B. Protected/conditional table adapter

ConditionalTableAdapter preserves the selected Native pair indices exactly
and requires a complete offline conditional candidate for all remaining pairs.
If no candidate is supplied, it inspects the unsafe same-index splice and fails
closed.  The collision report checks:

- finite positive inverse frequencies;
- strict descending table order;
- every pairwise log-frequency gap; and
- a minimum gap equal to 20% of the Native median adjacent log spacing.

This implements the route-audit correction: protected indices cannot be
combined with same-index EVQ values without a collision check.

### C. Slow-band far-only residual

slow_residual.py provides shape and route contracts for a low-dimensional
Native-plus-slow-EVQ residual:

- Native route at total budget <=4096;
- augmented route only above 4096;
- at most 16 EVQ pairs, 32 residual coordinates;
- zero-padded residual values and one augmented cache width; and
- a byte-equality short-route gate.

It does not implement a model wrapper or claim capability.  A future GPU
implementation must satisfy the same shape, Flash-only, cache, and gate
contract before any training is considered.

### D. Matched Q/K-only matrix

QKOnlyProtocol proposes three fresh matched seeds, Native and EVQ arms, 300
steps, rank-64 inherited Q/K LoRA, frozen inherited V/O, and the registered
continuous-4K / target-8K / target-16K phase curriculum.  The lambda_target is
the Native teacher's post-RoPE attention and A@V context; loss weights are
1.0 / 1.0 / 0.25 for attention KL, context MSE, and relation KL.

The 4K gate requires 2Wiki token-F1, all-family RULER, natural NLL, strict
autoregressive decoding, no Native-positive family collapse, and an
independent retention slice.  The 8K gate cannot run unless every 4K gate
passes and uses a strict autoregressive metric rather than NLL/PPL.

## Dry-run

The dry-run checks the pinned source hashes, manifest schema, disabled action
flags, optional asset bindings, host RAM visibility, and optional Torch Flash
capability.  Missing external assets are reported as unbound; no fetch is
attempted.

~~~
bash rebuttal/rebuttal_0723/experiments/olmo2_demand_retrofit_5090/run_5090.sh dry-run
~~~

An output receipt can be requested without touching any model or data:

~~~
python3 -m rebuttal.rebuttal_0723.experiments.olmo2_demand_retrofit_5090.dry_run \
  --output /tmp/r4prime_dry_run.json
~~~

The receipt records training_attempted=false, download_attempted=false,
model_loaded=false, and optimizer_created=false.

## Verification boundary

The included tests cover only contracts, table identity/collision rejection,
slow residual shapes/routes, capability-gate ordering, seed-matrix construction,
and the no-side-effect dry-run.  They do not prove model compatibility, Flash
eligibility, memory fit, or capability.  Those remain unverified until an
explicitly authorized asset-bound GPU smoke.
