# CA-NCP safety follow-up

The completed OLMo five-arm pilot rejected full-strength CA-NCP: `P1-P0` is
`-11.35pp`, while carrier-only `P0-C0` is only `-0.77pp`. The runtime was
correct; the median rank-2 plane angle was about 86 degrees. This directory
contains one final task-blind development diagnostic, not a new confirmed
method and not a parameter sweep.

Three new arms reuse the exact Full-13x10 panel and frozen baselines:

- `N_operator_cap`: Native table plus the capped alignment;
- `P_operator_cap`: carrier-NCP plus the same capped alignment;
- `P_axis_consensus`: carrier-NCP plus a source-consensus two-coordinate map.

The shared cap follows the unique geodesic from identity to each original
plane. Its maximum angle is fixed before task execution by

```text
B = max_{1 <= d < L} |exp(-i d nu_carrier) - exp(-i d nu_ncp_carrier)|
H = max_{1 <= d < L, k in I} |exp(-i d nu_k) - exp(-i d nu_carrier)|
phi = 2 asin(min(1, B / (4 H))).
```

This bounds the added non-diagonal kernel perturbation by the carrier snap it
is intended to exploit. The axis arm only acts when PG19-fit and
ProofPile-fit select the same best existing diagonal coordinate; disagreement
returns identity. Neither rule reads task outputs.

```bash
SOURCE=/root/autodl-tmp/today_rope_plan_20260914/ca_ncp_native_20260917
ROOT=/root/autodl-tmp/today_rope_plan_20260914/ca_ncp_safe_followup_20260917
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
DATA=/root/autodl-tmp/today_rope_plan_20260914/tailspline_olmo_s4_classic/assets/ppl46/manifest.json

python -m experiments.ca_ncp_safe_followup_20260917.build_alignments \
  --source-root "$SOURCE" --out "$ROOT"
python -m experiments.ca_ncp_safe_followup_20260917.run \
  --data "$DATA" --model "$MODEL" --source-root "$SOURCE" --root "$ROOT" \
  --parallel-workers 3 --execute
```

This same-panel follow-up can diagnose the original failure. Any positive arm
must be frozen and independently confirmed before supporting a paper claim.
