# Zero-parameter protected-band mature-checkpoint gate (2026-08-24)

Status: frozen while the independent anchored-EVQ PG-19 run was executing and
before any of its metrics were read.

## Candidate

Preserve every Native rotary pair whose wavelength lies in
`[L_native, 4 L_native]`, plus both sampled-support endpoints. In each
complementary rank interval, allocate the remaining pairs with a locally
endpoint-anchored EVQ-Cosh warp at `tau=2`.

The constants are part of this single registered candidate and will not be
swept. Construction reads only the released Native frequency tensor and
`L_native`; it reads no `L_target`, requested length, task label, loss,
attention statistic, collision score, or result from the anchored-EVQ arm.
It has zero learned parameters and zero optimizer steps. One static table and
attention scaling `1.0` are used at both 1x and 2x.

## Gate and selection

Run PG-19 on the same frozen first 20 rows at 1x and 2x. Reuse Native.
The candidate passes only if mean 1x tail-NLL regression is at most `+0.05`
and mean 2x tail NLL is strictly below Native.

If exactly one of anchored EVQ-Cosh and protected-band passes, promote that
candidate. If both pass, choose the smaller 1x regression; use 2x NLL only as
the tie-breaker. If neither passes, stop the single-static-table zero-parameter
route without RULER or natural-task generation.

The promoted candidate receives core-4 RULER at 8K with five rows per task,
then Qasper/2Wiki at 1x/2x with 20 rows per available cell. No parameter or
table choice may change after PG-19.
