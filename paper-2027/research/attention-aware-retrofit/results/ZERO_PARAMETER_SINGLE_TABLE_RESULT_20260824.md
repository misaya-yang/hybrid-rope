# Zero-parameter single-table frozen-checkpoint gate

- **Date:** 2026-08-24
- **Status:** completed negative gate; stopped before RULER/natural-task generation
- **Decision:** reject these two tested candidates; retain Native/long routing
  only as the current verified fallback, not as evidence that one-table joint
  in-window/extrapolation performance is impossible

## Question and contract

Can one analytic, zero-learned-parameter, fixed-Native-support table serve both
the released checkpoint's 1x window and 2x natural-text extrapolation without
routing? Both candidates use one table and attention scaling `1.0` at 1x/2x,
read no task labels or OOD losses, and require no `L_target`:

1. Native-support anchored endpoint EVQ-Cosh at `tau=2`;
2. a protected-band construction retaining Native wavelengths in
   `[L_native,4 L_native]` and applying locally anchored Cosh allocation in the
   complement.

The frozen PG-19 gate uses the same first 20 rows per 1x/2x cell as the existing
Native owner. Promotion required 1x regression at most `+0.05` NLL and strict
2x improvement.

## Result

| Static table | 1x tail NLL | Delta vs Native | 2x tail NLL | Delta vs Native |
| --- | ---: | ---: | ---: | ---: |
| Native | `2.9712` | — | `7.1029` | — |
| anchored EVQ-Cosh | `6.9492` | `+3.9780` | `6.8496` | `-0.2533` |
| protected-band Cosh | `3.6633` | `+0.6922` | `6.9184` | `-0.1845` |

Both analytic tables move 2x in the favorable direction, but both fail the 1x
gate by a wide margin. No `tau`, band, gain, or learning-rate sweep followed.

## Interpretation

The negative is candidate-specific: these two analytic reallocations improve
2x NLL but do not preserve the checkpoint's Native in-window function.
Protecting seven `O(L_native)` pairs reduces the coordinate shock substantially
but does not close it for this construction. This is consistent with the exact
frozen-transplant obstruction and the completed weights-by-table crossings,
which diagnose post-hoc compatibility rather than an intrinsic allocation
trade-off.

It does not reject EVQ-Cosh as a training-time zero-parameter construction and
does not close the single-table direction. From-scratch co-adapted studies and
the two-seed phase-chord pilot already show that a specified allocation can be
near-parity in-window while improving every tested extrapolation length. The
completed session policy is therefore a verified engineering fallback for the
current checkpoint, not the theoretical endpoint or a necessary solution.

## Claim boundary

This is one checkpoint, one analytic constant, one protected-band definition,
and PG-19 teacher-forced NLL. It is not capability evidence or a universal
impossibility result for static tables. It provides no evidence for an inherent
in-window/extrapolation trade-off.
