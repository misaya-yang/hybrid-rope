# Demand companding R1'/R2' preparation package

Status: **code complete; no training result**.

This package leaves the historical FMRoPE runner unchanged. It builds a
hash-backed schedule manifest, then injects one frozen table into that canonical
151.9M data/model/optimizer pipeline through `train_custom.py`.

## Scientific schedule

R0 supplies a positive demand profile (m(\Delta)). The loader reads `m` from
the R0 JSON without silently rescaling it. `delta` is read from the JSON when
present; otherwise an evenly spaced `[0,1]` grid is recorded as an explicit
inference. The raw coordinate is normalised only by its endpoints.

For each registered \(\lambda\in\{0,0.1,0.3\}\), the density is:

```text
rho_lambda(delta) ∝ ((1 - lambda) * m(delta) + lambda) ** (1/3)
```

The finite table uses endpoint-inclusive inverse-CDF quantiles
`u_k = k/(K-1)`. The mapping is deliberately in the same direction:
increasing `delta` maps to increasing normalised log-frequency `phi`. Every
table has `phi[0]=0`, `phi[-1]=1`, positive frequencies, and the same native
RoPE endpoint support `omega=[1, base^(-(K-1)/K)]`.

## R1' controls

For each demand lambda, two controls are generated:

- `anchored_tail_*`: compand the fast/middle prefix while retaining the
  slowest Geo tail values exactly;
- `mid_only_*`: compand only a middle index interval while retaining both the
  fast prefix and slow tail exactly.

Both controls keep all `K` positive frequencies. No slow frequency is replaced
by zero and no content dimension is deleted. The manifest records retained
tail values and `content_dimensions_removed=0`; tests fail if this changes.

## R2' matrix

R2' prepares these five arms:

```text
Geo, Cosh(tau=4), demand(lambda=0), demand(lambda=0.1), demand(lambda=0.3)
```

Each arm is registered for seeds `42, 137, 256`, reusing the tracked 151.9M
canonical contract:

- 151,898,880 parameters;
- train length 256;
- 499,974,144 consumed tokens and 7,629 optimizer steps per arm;
- global batch 256, 32 evaluation anchors, final-128 teacher-forced NLL;
- evaluation lengths 256/512/1024/2048;
- per-seed micro-batch/accumulation geometry from the canonical owner.

This package only records the 15 intended R2 rows as `NOT_STARTED`.

## R0 JSON shape

The primary input is the output of
`scripts/analysis/attention_demand_companding_r0.py`, specifically
`profiles.endpoint_log__mass__no_self.{phi_centers,demand_density}`.
The compact schema below remains supported for numerical tests.

The minimal input is:

```json
{
  "delta": [0.0, 0.5, 1.0],
  "m": [0.8, 1.0, 1.4],
  "base": 500000,
  "K": 32
}
```

This is only a schema illustration, not an experiment result or evidence
owner. `base`, `K`, and `tau` are optional metadata; CLI flags override them.
`m` must be finite and strictly positive so the lambda-zero density has a
strictly increasing CDF.

## Dry run

From the repository root:

```bash
bash rebuttal/rebuttal_0723/experiments/demand_companding_5090/run_5090.sh \
  dry-run \
  --r0-json /path/to/r0.json \
  --output /path/to/demand_companding_dry_run.json \
  --data-manifest /path/to/canonical/data_manifest.json \
  --checkpoint-root /path/to/empty/checkpoint/root
```

Data and checkpoint paths are optional for schedule-only preparation, but the
manifest stays `DRY_RUN_BLOCKED` until they pass.

After explicit GPU authorization, run one discarded throughput probe before
training. Both `--authorize` and
`DEMAND_COMPANDING_GPU_AUTHORIZED=YES` are required:

~~~
DEMAND_COMPANDING_GPU_AUTHORIZED=YES bash \
  rebuttal/rebuttal_0723/experiments/demand_companding_5090/run_5090.sh \
  probe --authorize --schedule-manifest /path/to/manifest.json \
  --arm demand_lambda_0p1 --seed 42 --micro-batch 128 \
  --data-manifest /path/to/data_manifest.json --work-dir /path/to/output
~~~

Use the same command with `train` only after the probe records finite loss,
Flash-only attention, and expected memory/throughput. The previously validated
5090 profile uses micro-batch 128 and
`max-autotune-no-cudagraphs`; micro-batch 256 remains unverified.

Focused CPU tests:

```bash
python3 -m pytest \
  rebuttal/rebuttal_0723/experiments/demand_companding_5090/test_demand_companding.py -q
```
