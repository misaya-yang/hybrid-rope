# T-C distance x competition behavior experiment

This experiment tests a narrow prediction suggested by the completed equal-
displacement TailSpline (T) versus control C results:

> T's relative benefit should increase when the required binding distance is
> longer and competing key-value records are present.

It does not fit a new frequency table.  Both arms use the existing Llama-3-8B
S=4 TailSpline and exact equal-displacement C tables, with the same checkpoint,
gain, decoder and input in every paired cell.

## Stage 1: existing-output behavior audit

`audit_existing_multikey.py` reads every paired T/C multikey output from the
completed clean 16K and 32K panels.  It decodes the frozen prompt IDs, recovers
the full key-value map and classifies each response as correct, a value bound to
another key, other, ambiguous or empty.  All rows remain in the denominator.

The GPU stage is qualified only if prompt mapping coverage is at least 99%, the
32K aggregate has more confirmed wrong-binding repairs than damages, both
multikey-2 and multikey-3 are non-negative, and at least one is positive.  This
gate is fixed before reading the audit result.

## Stage 2: new frozen intervention

If qualified, use 64 base samples and cross two distance conditions with two
competition conditions for both T and C.  The primary statistic is the
difference-in-differences interaction

`[(T-C)_far,strong - (T-C)_near,strong] - [(T-C)_far,weak - (T-C)_near,weak]`.

The base sample, answer format and within-block order remain fixed across the
four conditions. Strong competition uses 512 fixed distractor records; the
weak cell replaces them with equal-length neutral filler. Competition content
therefore differs between weak and strong cells, but T/C always share identical
token IDs within a cell. A common
position-ID offset parity check must pass before model execution.

This is an end-to-end behavioral intervention, not a head-level mediation test.
Large raw rows remain on the experiment server; Git retains code, the frozen
contract and compact reports.
