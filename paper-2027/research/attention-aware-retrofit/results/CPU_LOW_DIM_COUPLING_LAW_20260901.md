# CPU low-dimensional coupling-law audit (2026-09-01)

## Material Passport

- **Type:** deterministic CPU reconstruction and cross-model geometry holdout
- **Status:** `COMPLETE_CPU_ONLY`; the subsequent GPU test is owned by
  [`LOW_DIM_COUPLING_GPU_RESULT_20260901.md`](LOW_DIM_COUPLING_GPU_RESULT_20260901.md)
- **Verification:** five isolated CPU analyses plus one central deterministic rebuild
- **Primary owner:** this report
- **Machine artifact root:**
  [`../evidence/artifacts/coupling_law_cpu/`](../evidence/artifacts/coupling_law_cpu/)
- **Artifact manifest SHA-256:**
  `1b95bd19e2f14a792a85da158961d2c11f15cefa13608f03d4d70b8471aef9a0`
- **Data boundary:** candidate parameters used only the frozen OLMo movement vector.
  Qwen geometry was opened after parameters and model selection were frozen. No
  OLMo or Qwen long-context score was used to fit or select a shape.

## 1. Decision

The frozen 64-slot OLMo movement vector is accurately compressed by the
two-parameter clipped-affine law

\[
G_4(x)=
\begin{cases}
0, & x\ge x_H,\\
\dfrac{x_H-x}{x_H-x_L}, & x_L<x<x_H,\\
1, & x\le x_L,
\end{cases}
\]

with

\[
x_H=0.7382780681,\qquad x_L=0.3664038351.
\]

It reconstructs OLMo with MAE `0.001223` and RMSE `0.006118`. The same frozen
law, evaluated without refitting on Qwen coordinates, differs from the existing
whole-profile OLMo-to-Qwen transport by MAE `0.002174` and RMSE `0.009075`.
It also matches the Qwen self-profile about as closely as the 64-point transport:
MAE `0.014303` versus `0.013473`.

This establishes a low-dimensional **geometric reconstruction** of the current
coupling. It did not by itself establish language-model performance. The
subsequent GPU owner records a mixed result: Qwen long behavior was preserved,
while the registered Native-retention operating point failed.

The selected primary is `C2_clipped_affine`. There is no selected challenger:
the same-complexity logistic fit is slightly worse, and the adversarial audit did
not justify a third parameter. `C0`, `C1`, and `C3` remain diagnostic controls.

## 2. Current evidence entering this audit

| Evidence | Result | What it establishes |
| --- | --- | --- |
| OLMo log-s4, one table and gain | 1x PPL/task retention `0.875302/0.915103`; 2x/4x NLL `3.083278/3.081946`; RULER-13 `0.71397/0.66705/0.49859` | A useful frozen positive-control coupling, not a unique law |
| Same-gain arithmetic-s4 | 1x PPL retention `0.869584`; natural long worse than log; 16K RULER better by `0.00622` | Exponent-space law is the better joint operating point, not uniformly superior |
| Qwen self-construction | core-4 `0.7000/0.5875` at 64K/128K | Construction capability on a second checkpoint; no Qwen 1x retention owner |
| OLMo-to-Qwen whole-profile transport | `0.6725` at 64K; 128K running during this CPU audit | A no-refit 64-point transport works at 64K; not yet a low-dimensional law |
| Final-frequency multiset permutations | OLMo 1x NLL `3.10423 -> 6.86493`; Qwen 64K `0.7000 -> 0` | Slot-frequency assignment is non-exchangeable; the unordered spectrum is insufficient |
| Q/K readout 2x2 | log-s4 long improvement has the same sign under original weights and one Native-4K Q/K-LoRA | Robustness to one readout perturbation; not readout independence |

The numerical and protocol owner for these rows remains
[`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`](SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md).

## 3. Coordinate and exact identities

For a Native geometric RoPE lattice with `K=d/2`,

\[
\omega_i=b^{-i/K},\qquad
c_i=\frac{L\omega_i}{2\pi},\qquad
c_{\mathrm{orth}}=\frac{1}{1-b^{-1/K}},
\]

and the dimensionless fitting coordinate is

\[
x_i=\ln\frac{c_i}{c_{\mathrm{orth}}}.
\]

Because

\[
\Delta\omega_i=\omega_i(1-b^{-1/K}),
\]

the adjacent-slot phase separation over the Native window, measured in cycles,
is exactly

\[
\frac{L\Delta\omega_i}{2\pi}
=\frac{c_i}{c_{\mathrm{orth}}}
=e^{x_i}.
\]

Consequently

\[
x_i=\ln\frac{L}{2\pi c_{\mathrm{orth}}}-\frac{\ln b}{K}i
\]

is exactly affine in slot index for the ideal geometric lattice. The observed
float32 deviation is below `1e-7`.

The intervention remains

\[
m_i=G_4(x_i),\qquad
\omega'_i=\omega_i4^{-m_i},\qquad
z'_i=z_i-m_i\log_b4.
\]

### Coordinate-audit correction

The already evaluated OLMo-to-Qwen 64-point transport was not piecewise-linear
in `x`. It used piecewise-linear interpolation in

\[
u=e^x=\frac{L(\omega_i-\omega_{i+1})}{2\pi}.
\]

The saved table is reproduced under that definition to maximum movement error
`1.09e-6`. Qwen slots `0--9` lie beyond the OLMo fast-side domain and were
clamped to OLMo `m_0=0`; there is no slow-side extrapolation. Linear interpolation
in `x` instead changes at most `0.007942`, concentrated at slots `27--30`.

The present parametric laws are evaluated directly as `G(x)` and do not inherit
that interpolation convention.

## 4. Assumption ledger

| Level | Statement |
| --- | --- |
| **Exact** | The formulas for `c_i`, `c_orth`, `x_i`, adjacent phase separation, and the log-dilation intervention |
| **Reproduced fact** | `x_i` is affine in `i`; the old transport interpolates in `u=e^x`; its endpoint clamping and saved table were reproduced |
| **Supported on OLMo** | A boundary plus transition width is sufficient to reconstruct the frozen movement vector at the reported error |
| **Supported as geometry holdout** | The frozen two-parameter law remains close to the Qwen self-profile and the 64-point transport without refitting |
| **Conjecture** | `G_4(x)` is a transferable causal law rather than a compact proxy for a shared ordinal/readout basin |
| **Unmeasured** | Native retention and long-context performance of every low-dimensional candidate |

The Qwen self-profile is a useful zero-refit geometry holdout, but it was generated
by a related construction and is not an independent mechanism proof. Also, because
`x` is affine in `i` within either model, this two-checkpoint study does not by
itself prove that `x` is uniquely better than an ordinal slot coordinate.

## 5. Complexity ladder on frozen OLMo movement

All fits minimize unweighted squared movement error across the 64 OLMo slots.
The AIC/BIC entries use one common convention:
`n log(RSS/n) + 2k` and `n log(RSS/n) + k log(n)`, with `k` equal to the number
of shape parameters and the shared additive Gaussian constant omitted.

| Candidate | Parameters | MAE | RMSE | Max error | AIC | BIC | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `C0`: exact `x=0` step | 0 | 0.044746 | 0.196128 | 0.999990 | -208.510 | -208.510 | Reject as reconstruction |
| `C1`: movable step | 1 | 0.008233 | 0.044191 | 0.329082 | -397.261 | -395.102 | Explains plateaus, misses transition |
| `C2`: clipped affine | 2 | **0.001223** | **0.006118** | 0.044109 | **-648.347** | **-644.029** | **Selected primary** |
| `C3`: logistic | 2 | 0.001612 | 0.006211 | **0.035621** | -646.426 | -642.108 | No complexity-adjusted advantage |

For `C1`, 99.95% of squared residual is concentrated in the four transition
slots, so the second width parameter is substantively justified. For `C2`,
81.2% of squared residual is concentrated at slot 19 and the top three slots
account for 99.3%; no adjacent pair exceeds absolute residual `0.05`. Plateau
MAE is `0.000665` on the zero side and `0.000059` on the one side. Contiguous
block-out analysis shows weaker parameter identification when the narrow
transition itself is hidden, but no structural residual requiring a third
parameter.

The selected dimensionless boundaries are

\[
\frac{c_H}{c_{\mathrm{orth}}}=e^{x_H}=2.09232956,
\qquad
\frac{c_L}{c_{\mathrm{orth}}}=e^{x_L}=1.44253767.
\]

Their separation is `0.37187423`, or `1.81369` OLMo `x`-grid spacings. These
are fitted values, not analytic constants; they must not be rounded to a
"pretty" theoretical law after seeing them.

## 6. Qwen zero-refit geometry holdout

Every parameter above and the selection of `C2` were frozen before this table
was computed. No Qwen refit was performed.

| Candidate | MAE vs Qwen self | RMSE vs Qwen self | Max vs self | Spearman | MAE vs 64-point transport | RMSE vs transport |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing 64-point transport | 0.013473 | 0.065727 | 0.364302 | 0.941812 | -- | -- |
| `C0` exact step | 0.061126 | 0.217935 | 0.999999 | 0.900120 | 0.047664 | 0.201807 |
| `C1` movable step | 0.018985 | 0.073177 | 0.364302 | 0.898800 | 0.008729 | 0.042320 |
| `C2` clipped affine | **0.014303** | **0.065573** | 0.364302 | 0.910782 | **0.002174** | **0.009075** |
| `C3` logistic | 0.014835 | 0.065637 | 0.364302 | **0.927759** | 0.002752 | 0.009926 |

`C2` and the existing transport locate the main `m=0.5` crossing in the same
Qwen bracket, slots `29--30`, and both preserve exact slow-side `m=1`. The
maximum error against Qwen self occurs at slot 18, one of the self-profile's
isolated spikes. That maximum is therefore not evidence that an additional
smooth-shape parameter is needed. The law does not reproduce those spikes,
which is precisely what a two-parameter compression is supposed to test.

This holdout supports the statement that the transported 64-point curve is
almost entirely explained by boundary plus width. It does not establish that
the Qwen checkpoint will read the compressed table correctly.

## 7. Frozen candidates and hashes

The combined tensor has shape `[model, candidate, slot] = [2,4,64]`, ordered as
`[olmo,qwen]` and `[C0,C1,C2,C3]`.

| Artifact | SHA-256 |
| --- | --- |
| `candidate_tables.npy` | `b6a45bc644922f97ba72407bcef8289ecbddd7d122a27d71200afb1faef8cd31` |
| `candidate_tables.json` | `52c88b2d370c293f5c766d4685f12bd6905c6d4a8c861a28b029c42e11028af3` |
| OLMo `C2` float32 tensor | `113089f05b1e01e61bd9c78f06151ad96c1237b4602e0c380175c27d30562264` |
| OLMo `C2` `.npy` file | `e9e16faad38091800ca09d457b6df848e5a16cc58ec4048dd73f05df002e3700` |
| Qwen `C2` float32 tensor | `e78a0b6a04e2ae39a6004eedf1c11ef5f2250998b3adf2b8d8166274c2d4f14e` |
| Qwen `C2` `.npy` file | `36c9c1ce69be3bfa2426902ed2ae288f5e45dce6004c92c499b4c3e3542e4700` |
| `qwen_holdout.json` | `6597bcd9174f4e606fb0f85a5dab42c88a31de16598d817d72ab6fb36dbaaeb2` |
| `residuals.csv` | `b9ab6b071a35acfaf6a7f3d77a3853dc4f014b2c1916e625c0a1ab76b09c8acf` |

The manifest is authoritative for every file, including the plots and diagnostic
controls. The reusable builder is
[`scripts/analysis/compile_low_dim_coupling_law.py`](../../../../scripts/analysis/compile_low_dim_coupling_law.py).

## 8. Residual diagnostics

- [`olmo_profiles.png`](../evidence/artifacts/coupling_law_cpu/plots/olmo_profiles.png)
- [`qwen_profiles.png`](../evidence/artifacts/coupling_law_cpu/plots/qwen_profiles.png)
- [`slotwise_residuals.png`](../evidence/artifacts/coupling_law_cpu/plots/slotwise_residuals.png)
- [`residuals.csv`](../evidence/artifacts/coupling_law_cpu/residuals.csv)

## 9. What is proved, supported, and open

### Proved / exact

- The coordinate and adjacent-cycle identities in Section 3.
- The deterministic parameter-to-table mapping and reported hashes.
- The reported reconstruction and geometry-holdout numbers for the frozen
  source vectors.

### Supported empirically

- OLMo's frozen 64-slot movement vector is well described by a boundary and a
  width; a hard boundary alone is insufficient.
- The same two parameters produce a Qwen movement vector close to both the
  Qwen self-profile and the existing whole-profile transport.
- Logistic tails and a third shape parameter are unnecessary at the present
  reconstruction resolution.

### Still conjectural or unmeasured

- That `C2` preserves OLMo or Qwen Native retention and long-context capability.
- That `x` is causal or preferable to shared ordinal slot structure.
- That the fitted boundaries follow from `(L,b,K,s)` rather than from this OLMo
  positive control.
- That the same `G_4` transfers beyond these two `K=64` checkpoints.
- Any law in `s`; this audit concerns `G_4` only.

## 10. Frozen next-GPU pre-registration

No job is launched by this report. If the author authorizes confirmation, only
the selected `C2` table is a method candidate; `C0/C1/C3` are not a sweep.

```text
H1:
The frozen two-parameter C2 table preserves the useful empirical coupling closely
enough to pass Native retention and retain long-context capability on both models.

H2:
Low movement-vector reconstruction error is not sufficient for checkpoint behavior;
C2 fails the Native guard or loses the existing long capability.

Intervention:
Replace only the frozen empirical frequency table with the pre-hashed C2 table.
Keep checkpoint, c=.074 gain, rows, decoding, support semantics, and evaluator fixed.
No parameter or table change is permitted after any GPU endpoint is read.

Primary gate:
First run OLMo 1x PG-19 and five-task retention; both must be >=0.875. On pass,
record OLMo 2x/4x natural and RULER without tuning. Then run Qwen core-4 with
the frozen Qwen C2 table: >=0.6225 at 64K and >=0.5375 at 128K are the registered
within-0.05 capability-retention thresholds relative to the frozen references.

What result permanently closes which branch:
OLMo 1x failure closes C2 as the practical law without opening long endpoints.
Passing OLMo but missing either Qwen threshold closes the claim that this fitted
two-parameter law is transferable. Passing all gates supports a transferable
two-parameter G_4 operating law; it still does not prove uniqueness or derivation
from (L,b,K,s).
```

The expected confirmation cost is one OLMo 1x guard followed conditionally by
the already-established OLMo and Qwen evaluation rows. It adds no gain sweep,
shape sweep, training, per-head table, or third parameter.
