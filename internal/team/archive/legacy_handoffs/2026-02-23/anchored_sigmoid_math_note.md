# Anchored-Sigmoid Mathematical Note (for NeurIPS v6+)

## 1) Continuous form
Let `j in {0, ..., N-1}` be frequency-channel index (`N=d/2`).

- Native geometric inverse frequency:

`omega_geo(j) = theta^{-2j/d}`

- Sigmoid gate:

`sigma(j; s, c) = 1 / (1 + exp(-s * (j - cN)))`

where:
- `s` is slope (sharpness),
- `c in (0,1)` is center ratio.

- Anchored scaling factor:

`A(j; a, s, c) = 1 + (a - 1) * sigma(j; s, c)`

with `a >= 1` (`anchor_factor`).

- Anchored-sigmoid schedule:

`omega_anchor(j) = omega_geo(j) / A(j; a, s, c)`

This preserves high-frequency channels near the anchor region while applying a smooth low-frequency warp.

## 2) Relation to theory optimum
The broadband variational optimum has shape `rho*(phi) propto cosh(1-phi)` with bounded amplitude ratio around `cosh(1) ~= 1.54`.

Anchored-sigmoid approximates this by:
- avoiding overly aggressive concentration (`a` too large),
- keeping a bounded high/low bias through (`a`, `s`, `c`),
- acting as a practical constrained approximation under discrete channel and implementation constraints.

## 3) Parameter semantics (operator-facing)
- `anchor_factor (a)`: overall warp amplitude; too high risks waterbed depletion in mid-band.
- `slope_raw (s_raw)`: transition sharpness; too steep increases brittleness.
- `center_ratio (c)`: where the transition starts; higher values move deformation toward low-frequency tail.

## 4) Current tuned operating point
Current tuned point for next controlled reruns:
- `anchor_factor=4`
- `slope_raw=20`
- `center_ratio=0.70`

Use this as the default unless an experiment explicitly scans schedule hyperparameters.
