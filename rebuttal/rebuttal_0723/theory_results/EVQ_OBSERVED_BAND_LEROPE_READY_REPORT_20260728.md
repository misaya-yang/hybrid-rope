# EVQ Observed-Band LeRoPE: Offline-Ready Experiment Report

Date: 2026-07-28

Status: `OFFLINE_READY_NOT_TRAINED`

## Decision

The next experiment replaces unconstrained full-band EVQ-LeRoPE with an
observability-partitioned frequency update:

\[
\omega_k=\omega_k^{\mathrm{EVQ}}\exp(m_k\alpha_k),\qquad
m_k=\mathbf 1\!\left[
\frac{L_{\mathrm{train}}\omega_k^{\mathrm{EVQ}}}{2\pi}\ge 1
\right].
\]

All bands retain the exact paper-midpoint EVQ-Cosh initialization. Bands that
complete at least one cycle inside the training window may learn a shared
LeRoPE log residual; slower bands remain exactly fixed at EVQ.

For \(L_{\mathrm{train}}=128\), base \(500{,}000\), \(\tau=5\), and 32 RoPE
pairs, this produces:

- learnable bands: `0..21` (22 pairs);
- frozen bands: `22..31` (10 pairs).

This is the smallest structural test of the current failure hypothesis. It
does not introduce collision, moat, NoPE, YaRN, or task supervision.

## Why the full-band experiment was insufficient

The completed full-band EVQ-LeRoPE runs used only \(L=128\) language-modeling
loss while allowing all 32 frequency pairs to move. The two completed seeds
show a consistent extrapolation direction but an unstable in-window result:

| Seed | Length | Native NLL / PPL | Full-band EVQ-LeRoPE NLL / PPL | NLL delta |
| --- | ---: | ---: | ---: | ---: |
| 42 | 128 | 5.39394 / 220.07 | 5.38396 / 217.88 | -0.00998 |
| 42 | 256 | 5.38081 / 217.20 | 5.32531 / 205.47 | -0.05550 |
| 42 | 512 | 5.44418 / 231.41 | 5.32268 / 204.93 | -0.12149 |
| 43 | 128 | 5.40158 / 221.76 | 5.42217 / 226.37 | +0.02059 |
| 43 | 256 | 5.36367 / 213.51 | 5.35287 / 211.21 | -0.01081 |
| 43 | 512 | 5.40085 / 221.59 | 5.32595 / 205.60 | -0.07490 |

The mean NLL deltas over the two completed seeds are `+0.00531`, `-0.03315`,
and `-0.09820` at 128, 256, and 512. Thus the naive arm preserved a strong
length-transfer direction but did not robustly recover Native in-window
modeling.

The learned slow-band residual was material rather than negligible: its
\(\ell_2\) norm was `0.1162` for seed 42 and `0.1213` for seed 43. Those bands
have not completed one cycle in training, so their language-modeling
gradients cannot identify their longer-range periodic behavior.

Seed 44 was interrupted by machine shutdown and is not a result.

## Rejected literal Native/EVQ splice

Directly taking same-index Native high frequencies and EVQ slow frequencies
is invalid for this configuration. At the proposed boundary, Native pair 21
has frequency approximately `0.000182`, whereas EVQ pair 22 has frequency
approximately `0.041335`. The splice reverses frequency order.

The implemented arm therefore preserves the complete monotone EVQ grid and
changes only which bands are permitted to learn.

## Required controls

The runner now exposes five exact arms:

1. `native`: fixed geometric RoPE;
2. `evq_fixed`: fixed paper-midpoint EVQ-Cosh;
3. `native_lerope`: Native initialization, all bands learned;
4. `evq_lerope`: EVQ initialization, all bands learned;
5. `evq_observed_lerope`: EVQ initialization, only bands `0..21` learned.

All arms use the same model initializer, data order, optimizer semantics,
token budget, and evaluation offsets for a given seed. The model contains
50,928,160 parameters before accounting for arm-specific trainability.

## Current matched protocol

- training tokens: 15,000,320 stored tokens (14,991,360 consumed after global
  batch truncation);
- model training length: 128;
- evaluation lengths: 128, 256, 512;
- evaluation chunks: 32 fixed paired offsets;
- global/micro batch: 128/128;
- optimizer steps: 915;
- frequency LR multiplier: 1.0;
- frequency weight decay: 0;
- independent frequency gradient clipping: 1.0;
- final and per-step frequency ordering: hard failure if not strictly
  decreasing.

This 15M-token protocol is retained for direct comparison with the completed
full-band runs. A longer-token experiment is a separate question and has not
been represented as ready here.

## Offline preflight receipt

The server-side CPU preflight passed:

- all five arms have the same ordinary initialization SHA-256:
  `816c06ea434254b4cf96a9d60713e79dbdfb4d25fb0e3e3430b42e86a74c6304`;
- observed-band base grid is strictly decreasing;
- learnable bands are exactly `0..21`;
- frozen bands are exactly `22..31`;
- learned-band gradient \(\ell_2\): `0.0135982`;
- frozen-band maximum absolute gradient: `0.0`;
- training artifact SHA-256:
  `36687fee9c61b817169e80e8b2b4347bb587148d07734ed2f7807058b17a82d1`;
- validation artifact SHA-256:
  `f2536ec9aa17da879f1ecafde3bc11a40d7b3c6c5f596af277c7950d4d108ddf`;
- runner SHA-256:
  `2f7f419ca14cd46d0ae6d64c383ebc1c1b7e7912d00c28c42251450299a01486`;
- prepared receipt SHA-256:
  `88788697e3e39a8d91b01b0d171ae105c768e2391387831672fce67e9353f83e`;
- frequency-preflight receipt SHA-256:
  `8af5a276b874c3245ab3db0dfcdda32e3538aa70be90fd8e894e962332644bc0`.

The preflight is readiness evidence only. No training result exists for
`evq_observed_lerope`.

## Result interpretation contract

The experiment answers one narrow question:

> Does preventing under-observed EVQ bands from following short-window
> language-model gradients improve the Native/EVQ in-window–extrapolation
> trade-off?

It does not establish task capability, large-model adaptation, statistical
significance, or universal frequency optimality. NLL/PPL remains distinct from
autoregressive retrieval and downstream capability.

## Existing result owners

- seed-42 full-band aggregate SHA-256:
  `19c8de718649ac0dcc74d2c058672a3e07948c5bc2c09b7daeed7d35ac624a01`;
- seed-43 full-band aggregate SHA-256:
  `0c366139e12d73d70ffa963786a860bbd5e90bba6324d784c08c0895172faeb2`.
