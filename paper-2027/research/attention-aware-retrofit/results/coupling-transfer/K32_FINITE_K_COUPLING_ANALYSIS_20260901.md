# K32 finite-grid coupling analysis (2026-09-01)

## Material passport

- **Status:** `MATCHED_S2_COMPLETE / LONG_BACKBONE_SUPPORTED /
  NATIVE_COMPATIBILITY_FAILED /
  CELL_AVERAGE_GPU_CANDIDATE_REJECTED`
- **Question:** does frozen K64 `G(x)` fail on K32 because the physical
  coordinate does not transport, or because point sampling under-resolves a
  continuous law?
- **Frozen law:** `x_H=0.7382780681078285`,
  `x_L=0.366403835112904`; neither boundary was refit.
- **CPU receipt:**
  [`../evidence/K32_FINITE_K_COUPLING_GEOMETRY_20260901.json`](../../evidence/K32_FINITE_K_COUPLING_GEOMETRY_20260901.json)
- **GPU receipt:**
  [`../evidence/K32_FINITE_K_COUPLING_GPU_RECEIPT_20260901.json`](../../evidence/K32_FINITE_K_COUPLING_GPU_RECEIPT_20260901.json)
- **Reproducer:**
  [`scripts/analysis/audit_finite_k_coupling.py`](../../../../../scripts/analysis/audit_finite_k_coupling.py)

## 1. Decision

K32 genuinely under-resolves the frozen transition: its width falls from about
two native log-frequency intervals at K64 to `0.8613` interval at K32, leaving
only one non-binary point sample. That fact makes finite-grid effects plausible,
but it is **not sufficient to explain the Native failure**. The registered
zero-parameter cell average changes only two boundary slots, does not generate
the OLMo fast-side residual shoulder, and slightly worsens reconstruction of
the OLMo self profile. It therefore fails the CPU entrance condition for a GPU
candidate.

The completed K32 long holdout points elsewhere. At 64K, frozen physical `x`
scores `0.4350`, normalized index `0.4275`, monotone self `0.3900`, and Native
`0.2775`. Thus K32 does not show a general loss of long utility for physical
`x`; the same physical law is the best of the three long-table arms while the
tested coordinate laws damage Native behavior.

However, all three K32 candidate tables were constructed with `s=4`, whereas
64K is only `2x` relative to this checkpoint's 32K Native length. The four-arm
experiment therefore tests whether an **s4 maximum profile** remains useful at
the lower half of `[1,4]`; it does not isolate K transport at matched target
dilation. This initially made H0, scale mismatch, the required discriminator.
The completed matched-s2 rows below reject H0 as a sufficient explanation.
H3 remains weakened by the positive long direction, and the specific
cell-average realization of H1 remains rejected by its independent CPU gate.

The subsequent matched-s2 test resolves the scale confound: reducing s4 to s2
improves 32K only from `.5125` to `.5225`, still below the `.5665625` Native
floor, while 64K improves from `.4350` to `.5050`. Scale mismatch affected both
endpoints but was not sufficient to explain the Native failure. This is the
predicted H4 signature: the frozen long backbone remains useful and becomes
stronger at the matched scale, while Native compatibility remains broken.

No `finite-K corrected C2` table was frozen or launched.

## 2. What the s=2 result establishes

The zero-refit s=2 result reuses the s=4 boundaries and construction

\[
\omega_i'(s)=\omega_i s^{-G(x_i)}
\]

without changing `x_H`, `x_L`, or fitting a scale-specific table. On OLMo it
passes the Native PG-19 PPL gate, and its five-task macro is `0.35936`, above
Native `0.34513`. This is strong behavioral evidence that the frozen K64 law is
not merely an s=4 operating point: it remains Native-compatible when the scale
is reduced from four to two.

The completed long-side confirmation gives 2x PG-19 NLL `2.970433`, six-task
natural macro `0.260341`, and 8K core-4 `0.5150`, versus matched Native
`7.100855`, `0.058883`, and `0.0000`. Thus the same frozen law is useful across
the tested `[1,2]` endpoints. It still does not establish arbitrary-scale or
cross-checkpoint universality, and s2's 8K core-4 remains below the s4 maximum
profile's `0.7150` on the same rows.

## 3. Exact finite-K geometry

For a geometric Native table with `K` rotary pairs, base `b`, and Native length
`L_n`, the coordinate grid is

\[
x_i=\log\!\left(\frac{L_n(1-b^{-1/K})}{2\pi}\right)
    -i\frac{\log b}{K}.
\]

Therefore the spacing is exactly, not approximately,

\[
h=\Delta_x=\frac{\log b}{K},\qquad
W=x_H-x_L=0.371874232995,qquad
\eta=\frac{W}{h}.
\]

`eta` counts transition width in grid intervals. It is not the exact number of
non-binary slots: fractional alignment of `x_H` and `x_L` with the grid also
matters, and `L_n` controls that alignment by translating the grid.

| Grid | K | b | Native L | `Delta_x` | `eta` | `x_H` slot/cell | `x_L` slot/cell | point-sampled non-binary slots |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| OLMo | 64 | 500,000 | 4,096 | `0.205037` | `1.81369` | `19.783`, slots 19/20 | `21.597`, slots 21/22 | 20, 21 |
| Qwen K64 | 64 | 1,000,000 | 32,768 | `0.215867` | `1.72270` | `28.638`, slots 28/29 | `30.361`, slots 30/31 | 29, 30 |
| Qwen K32 | 32 | 1,000,000 | 32,768 | `0.431735` | **`0.86135`** | `15.688`, slots 15/16 | `16.549`, slots 16/17 | **16 only** |

The pure Qwen-geometry K sweep makes the discretization effect visible:

| K | `eta` | point non-binary slots | cell-average non-binary slots | max `|cell-point|` |
| ---: | ---: | --- | --- | ---: |
| 16 | `0.43067` | none | 8, 9 | `0.145595` |
| 32 | `0.86135` | 16 | 16, 17 | `0.020487` |
| 64 | `1.72270` | 29, 30 | 29, 30 | `0.005647` |
| 128 | `3.44540` | 52, 53, 54 | 51--55 | `0.011834` |

This is a real transition-resolution collapse, but not a behavioral diagnosis:
the correction magnitude depends on boundary alignment as well as `eta`, and no
benchmark score was used in this audit.

## 4. Zero-parameter cell projection

Use equal-width half-step cells

\[
C_i=[x_i-h/2,x_i+h/2]
\]

and the exact cell mean

\[
m_i^{\rm cell}=\frac1h\int_{C_i}G(u)\,du.
\]

For clipped-affine `G`, an antiderivative is

\[
F(t)=t-\frac{(t-x_L)_+^2}{2W}
       +\frac{(t-x_H)_+^2}{2W},
\]

so `m_i^cell=[F(x_i+h/2)-F(x_i-h/2)]/h`. This is the orthogonal
finite-volume projection onto cellwise constants under uniform `dx`, or
equivalently box-prefiltering before sampling. It conserves the continuous
cell integral, introduces no parameter, remains monotone, preserves frequency
ordering, and converges to point sampling as `K -> infinity`.

That is a valid numerical first principle, but not a checkpoint mechanism: it
assumes each learned rotary pair represents a surrounding uniform-`x` spectral
cell. A frozen model actually contains discrete, slot-specific Q/K
coefficients, so this projection is not an attention-aliasing theorem.

The decisive numerical checks are negative:

- OLMo K64 changes only slot 20 `0.119557 -> 0.141661` and slot 22
  `1.000000 -> 0.997414`. Slot 19 stays exactly protected, so the projection
  cannot produce the empirical fast-side shoulder where `81.2%` of C2's
  squared OLMo residual lies.
- Relative to the OLMo self movement, cell averaging worsens C2 RMSE from
  `0.006118` to `0.006706` and MAE from `0.001223` to `0.001531`.
- Qwen K32 changes only slot 16 `0.362380 -> 0.382867` and slot 17
  `1.000000 -> 0.998594`; it creates no fast-side shoulder.
- The projection remains smooth and monotone and therefore correctly does not
  try to explain Qwen's isolated self-profile spikes.

Accordingly, cell averaging has a clean discretization interpretation but does
not have the required empirical geometry to justify a K32 GPU confirmation.

## 5. K32 paired outcomes

All rows use the same Qwen2.5-0.5B-Instruct checkpoint, frozen `s=4` table for
the entire request, fixed attention scaling `1.102585782722872`, official RULER
commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`, seed `20260822`, and 20 rows
per task. Neither K32 benchmark was read when the tables were generated.

The construction factor `4` is essential context: the 32K/64K columns are
`1x/2x` evaluation points of an s4 maximum profile, not a matched s2 profile.

| Table | 32K single / mk2 / mk3 / VT | 32K macro | retention vs Native | 64K single / mk2 / mk3 / VT | 64K macro |
| --- | --- | ---: | ---: | --- | ---: |
| Native | `1.00/.80/.25/.54` | **`.6475`** | `1.0000` | `.75/.20/.00/.16` | `.2775` |
| frozen physical `x` | `1.00/.40/.00/.65` | `.5125` | `.7915` | `1.00/.30/.05/.39` | **`.4350`** |
| frozen normalized index | `1.00/.65/.05/.44` | `.5350` | `.8263` | `1.00/.40/.00/.31` | `.4275` |
| monotone self | not opened | not opened | not opened | `1.00/.20/.00/.36` | `.3900` |

The central crossing is not subtle: physical `x` loses `0.1350` macro at the
Native length but gains `0.1575` at 2x. Normalized index shows the same sign
pattern, while physical `x` is `+0.0075` ahead at 64K. The experiment therefore
separates Native compatibility from long utility instead of treating the 32K
failure as a total law failure.

## 6. Competing hypotheses after all four 64K arms

### H0 — construction/evaluation scale mismatch

**Rejected as sufficient.** The K32 checkpoint has Native length 32K, so the
64K holdout is a 2x deployment. Replacing the mismatched s4 table with the
exact frozen s2 law improves 32K by only `.0100`, still below the Native gate,
while improving 64K by `.0700`. Scale matters, but does not explain away the
Native incompatibility.

### H1 — universal `x` plus finite-K discretization

**Not supported by the registered correction.** K32 is under-resolved, but the
exact cell projection neither recovers the known OLMo shoulder nor makes a
large structural K32 change. H1 remains logically possible only through a
different checkpoint-aware projection, which would no longer be the tested
zero-parameter cell-average hypothesis.

### H2 — K-dependent coupling law

**Premature, and not required by the current long rows.** A K32
self-derived monotone table improves 64K over Native but is worse than both
frozen K64 coordinate laws. Thus an explicit `G(x;K)` is not currently needed
to explain long behavior. K may still determine how the checkpoint uses the
coarser slots inside Native.

### H3 — `x` does not transport across K

**Weakened.** Physical `x` is the best K32 64K arm and beats Native by `0.1575`.
The small `0.0075` advantage over normalized index is not enough to establish a
unique physical coordinate, but it contradicts the strong form that frozen
physical `x` loses long utility at K32.

### H4 — Native-only incompatibility

**Currently strongest.** Both s4 coordinate laws fail 32K retention yet improve
64K substantially, and the matched s2 physical law repeats the separation more
strongly: `.5225` fails Native while `.5050` is the best K32 long row. This is
the expected signature of a transferable long backbone plus Native-sensitive
Q/K readout. It remains one checkpoint and does not identify the correction.

## 7. Decision rule and next action

The four-arm update is now:

1. physical `x` being clearly above Native at 64K rejects “Native failure means
   long failure”;
2. physical `x` slightly exceeding normalized index weakens, but does not
   decisively eliminate, the K64-regime-coincidence explanation;
3. monotone self failing to beat either frozen coordinate law on long rows says
   checkpoint-specific detail is not automatically useful outside Native;
4. because cell-average failed its CPU entrance condition, no corrected C2 is
   allowed onto GPU;
5. the matched-s2 follow-up rejects H0 as sufficient and promotes H4 as the
   current mechanism explanation.

It remains **not worth running** the proposed zero-parameter finite-K corrected
C2. The completed discriminator used the same frozen physical law at the
correct scale:

```text
K32 s2 physical-x, fixed c=.074
32K Native gate -> 64K long holdout
```

No boundary, curve, K parameter, or benchmark-fitted correction was introduced.
The 32K floor was frozen as `0.875 * 0.6475 = 0.5665625` before the result. The
observed branch is “Native fails, long improves,” so Native compatibility—not
a K-dependent long law—is the next mechanism question. This report does not
authorize fitting that functional or adding a K-dependent parameter.

### 7.1 Matched-s2 outcome

The exact frozen construction used

\[
\omega_i'=\omega_i2^{-G(x_i)},\qquad
1+0.074\log 2=1.0512928913614359,
\]

with table tensor SHA-256
`b61a58f3e84429e00eaac69a0d9ab43abf89bc193987b2bcbcd3ab3bccd455fb`.
Neither boundary nor the gain coefficient changed.

| K32 table | 32K single / mk2 / mk3 / VT | 32K macro | retention | 64K single / mk2 / mk3 / VT | 64K macro |
| --- | --- | ---: | ---: | --- | ---: |
| Native | `1.00/.80/.25/.54` | `.6475` | `1.0000` | `.75/.20/.00/.16` | `.2775` |
| physical `x`, s4 | `1.00/.40/.00/.65` | `.5125` | `.7915` | `1.00/.30/.05/.39` | `.4350` |
| physical `x`, matched s2 | `1.00/.50/.15/.44` | **`.5225`** | **`.80695`** | `1.00/.50/.05/.47` | **`.5050`** |
| normalized index, matched s2 | `1.00/.60/.25/.46` | **`.5775`** | **`.89189`** | `1.00/.40/.00/.35` | `.4375` |

The matched table fails the Native floor by `.0440625`, but beats Native 64K
by `.2275` and its s4 version by `.0700`. Therefore:

- the current object is not merely `G_4(x)`; changing the analytic scale to s2
  improves behavior in the expected direction;
- scale mismatch is not the cause of the K32 Native failure;
- a K-dependent **long** law is not indicated, because long improves strongly;
- the unresolved object is a checkpoint/K-specific Native compatibility
  correction around the transported long backbone.

The matched normalized-index control exposes a real Pareto crossing rather
than a universal physical-coordinate win. It passes the Native gate and exceeds
physical `x` by `.0550` at 32K; physical `x` exceeds it by `.0675` at 64K.
Thus the long-side result favors physical `x` beyond the predeclared `.05`
coordinate band, but the joint Native-to-long operating point favors neither
arm uniformly. This supports physical `x` as a long-backbone coordinate, not as
a complete deployable law.

The machine-path-free receipt is
[`../evidence/K32_MATCHED_S2_RECEIPT_20260901.json`](../../evidence/K32_MATCHED_S2_RECEIPT_20260901.json).

## Claim ceiling

These results reject the claim that correct scale alone makes the frozen K64
law Native-compatible on K32. They support, but do not prove universally, a
matched-s transferable low-dimensional long-context backbone. They do not
prove `x` unique, establish a universal dependence on K, identify a Native
correction, or show that every finite-grid projection must fail.
