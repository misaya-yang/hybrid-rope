# Phase 9 (L=2048, 50M) Partial Multi-Seed Report (No Checkpoint)

- Generated at: 2026-03-02 21:37:08
- Source policy: only `results_final.json` + `run.log` + per-run `passkey_nll.json`; `results_checkpoint.json` excluded.
- Experiment dir: `/root/autodl-tmp/evq_phase9_350m_L2048_50M_tau0_tau1.5`

## Completion Status

| Seed | Geo(tau=0.0) | EVQ(tau=1.5) | PPL Source | Status |
|---|---|---|---|---|
| 42 | done | done | Geo:results_final.json / EVQ:results_final.json | complete |
| 137 | done | done | Geo:run.log / EVQ:run.log | complete |
| 256 | done | done | Geo:run.log / EVQ:run.log | complete |
| 314 | missing | missing | - | incomplete |

## Per-Seed Geo vs EVQ

| Seed | Geo PPL@2048 | Geo PPL@4096 | Geo PPL@8192 | Geo PPL@16384 | Geo Ret | Geo Gap | EVQ PPL@2048 | EVQ PPL@4096 | EVQ PPL@8192 | EVQ PPL@16384 | EVQ Ret | EVQ Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 87.511 | 119.355 | 173.281 | 279.354 | 0.5467 | 0.0162 | 89.143 | 115.807 | 155.526 | 236.189 | 0.5667 | 0.0209 |
| 137 | 86.346 | 113.702 | 168.676 | 274.944 | 0.5800 | 0.0443 | 84.334 | 109.840 | 156.664 | 251.192 | 0.6200 | 0.0665 |
| 256 | 85.853 | 119.487 | 182.542 | 300.161 | 0.6067 | 0.0349 | 87.219 | 114.767 | 159.854 | 253.310 | 0.5400 | 0.0351 |
| 314 | - | - | - | - | - | - | - | - | - | - | - | - |

## Delta (EVQ - Geo) by Seed

| Seed | dPPL@2048 | dPPL@4096 | dPPL@8192 | dPPL@16384 | dRet | dGap |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 1.6320 | -3.5480 | -17.7550 | -43.1650 | 0.0200 | 0.0047 |
| 137 | -2.0120 | -3.8620 | -12.0120 | -23.7520 | 0.0400 | 0.0222 |
| 256 | 1.3660 | -4.7200 | -22.6880 | -46.8510 | -0.0667 | 0.0002 |

## Mean ± Std (Completed Seeds Only)

- Completed seeds used in aggregation: `[42, 137, 256]` (n=3)

| Metric | Geo Mean | Geo Std | EVQ Mean | EVQ Std | EVQ-Geo Mean |
|---|---:|---:|---:|---:|---:|
| PPL@2048 | 86.5700 | 0.8514 | 86.8987 | 2.4205 | 0.3287 |
| PPL@4096 | 117.5147 | 3.3025 | 113.4713 | 3.1875 | -4.0433 |
| PPL@8192 | 174.8330 | 7.0621 | 157.3480 | 2.2436 | -17.4850 |
| PPL@16384 | 284.8197 | 13.4677 | 246.8970 | 9.3337 | -37.9227 |
| Retrieval | 0.5778 | 0.0301 | 0.5756 | 0.0407 | -0.0022 |
| Mean NLL Gap | 0.0318 | 0.0143 | 0.0408 | 0.0233 | 0.0090 |

## Quick Read

- PPL: EVQ improves long-context PPL on average (dPPL@16384 = -37.9227).
- Passkey retrieval: near tie / slight EVQ downside (dRet = -0.0022).
- Seed314 is still missing and excluded per request.
