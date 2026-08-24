# Direct fixed-support `z` mature-checkpoint pilot

- **Date:** 2026-08-24
- **Status:** complete internal negative gate; downstream evaluation stopped
- **Decision:** stop this two-design-row, 62-effective-degree calibration
  protocol; do not run its PG-19, RULER, LoRA, or full-task queue
- **Machine-path-free receipt:**
  [`../evidence/DIRECT_Z_FIXED_SUPPORT_PILOT_20260824.json`](../evidence/DIRECT_Z_FIXED_SUPPORT_PILOT_20260824.json)

## Decision first

Directly optimizing interior allocation produced a real but heterogeneous 2x
signal. Both design rows improved, and the two held-out rows improved on
average, but one held-out row regressed beyond the frozen per-row gate. The
candidate therefore failed and the launcher correctly blocked every downstream
stage.

This is not evidence that the third axis is inert. It is evidence that a
62-effective-degree table calibrated from two documents is not a sufficiently
stable mature-checkpoint method, even though its mean 2x direction can be
favorable. Another learning-rate, step-count, or seed sweep of this exact
protocol is not justified.

## Contract

The released OLMo-2-0425-1B-Instruct weights were frozen. Sixty-three positive
gap logits, with 62 effective simplex degrees of freedom, changed only the
interior normalized coordinates `z`; the Native fast/slow endpoints and log
span remained exact. The same realized table and attention scaling `1.0` were
used at 1x and 2x. No length route, attention prior, Cosh prior, or collision
surrogate participated in selection.

Rows 0--1 of a four-document, receipt-bound 8K FineWeb-Edu tensor were design
rows; rows 2--3 were untouched held-out gates. The objective minimized final
64-token teacher-forced 2x tail NLL with a per-row 1x tail-NLL no-harm penalty.
The run used ten AdamW steps at learning rate `0.003`, zero model-weight
updates, and one optimizer seed.

## Results

| Split / row | Delta 1x tail NLL | Delta 2x tail NLL |
| --- | ---: | ---: |
| design 0 | `-0.01517` | `-0.31526` |
| design 1 | `+0.02062` | `-0.18339` |
| held-out 2 | `+0.02032` | `+0.08229` |
| held-out 3 | `+0.00157` | `-0.13068` |
| held-out mean | `+0.01095` | `-0.02420` |

The predeclared gates were:

- candidate differs from Native: pass;
- maximum held-out 1x regression at most `+0.05`: pass (`+0.02032`);
- mean held-out 2x delta below zero: pass (`-0.02420`);
- maximum held-out 2x regression at most `+0.05`: **fail** (`+0.08229`).

The candidate changed normalized coordinates by at most `0.000975`. Thus the
failure is not explained by a gross support change or a wildly distorted
table; it is row heterogeneity under a highly underdetermined calibration.

## Runtime and implementation correction

The valid run had exact Native initialization parity (`0.0` maximum NLL
difference), completed its ten optimization steps in `10.08s`, and peaked at
`17.22 GiB` reserved memory on an RTX 5090 with Flash-only attention. GPU
utilization reached 96% during the short run and returned to zero afterward.

The first smoke exposed an implementation mismatch before optimization: the
replacement rotary path returned BF16 cos/sin, while the installed Transformers
OLMo path uses FP32 batched-matmul phases and returns FP32 tensors. The failed
diagnostic produced no result table. After matching the native path, a direct
8K rotary comparison had exactly zero cos/sin error and nonzero finite `z`
gradient; the full smoke then had exact NLL parity. The failed log is retained
with the ignored raw artifacts and is not part of the scientific result.

## Interpretation and routing correction

The result separates two questions:

1. **Does direct `z` movement affect frozen-checkpoint OOD loss?** Yes in this
   pilot: both design rows and one held-out row moved favorably at 2x.
2. **Is the current calibration a reliable retrofit?** No: the untouched row-2
   regression violated the registered robustness gate.

Do not tune this failed run after observing its held-out rows. More
importantly, this pilot is not the repository's zero-parameter method route.
The completed Native/s4 session policy already owns the practical
zero-training mature-checkpoint result, while EVQ-Cosh remains the closed-form,
zero-learned-positional-parameter training-time construction. This failed
learned-oracle branch changes neither owner and does not block their evidence
or the separate completed LoRA/adaptation studies.

If a future study asks for a single static zero-parameter table that jointly
serves 1x and 2x, the protected-band or pinned-scale analytic candidates require
a new preflight. That is a new method question, not the automatic continuation
of this failed 62-effective-degree calibration.

## Claim boundary

This is a single-checkpoint, one-ratio, one-seed, four-document candidate
screen. It is not long-context capability evidence, not a new causal proof that
`z` matters, not a target-free construction, and not evidence for or against a
universal allocation profile. The exact-range and same-support owners retain
their existing causal roles, and the Native/s4 owner retains its practical
zero-training role.
