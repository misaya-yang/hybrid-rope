# Profiled residual diagnostic (RTX 5090)

Purpose: determine whether a nonlinear RoPE residual still improves held-out
tail NLL after the target-aware geometric base is re-optimized. This is
checkpoint-only, inference-only, seed-42 supporting evidence.

The decisive checkpoint is the completed 500M-token `fmrope_base256` arm. It
recovers its exact training frequencies at `L=256, c=1`. The matched 500M
Paper-Geo and EVQ checkpoints are optional secondary diagnostics. Historical
three-seed `shape_l128` checkpoint files were deleted after evaluation and are
therefore not inputs to this run.

Registered search:

- lengths: 256 / 1K / 2K / 4K / 8K;
- target base: `b=cT`, `c in {0.5,1,2,4}`;
- residual strength: `{-1,-0.5,0,0.5,1,1.5}`;
- residuals: orthogonal Cosh residual and norm-matched frozen two-band residual;
- every residual strength re-selects `c` on 16 calibration anchors;
- final numbers use 32 disjoint test anchors.

Signed negative residuals can cross adjacent channel frequencies. They remain
registered diagnostic perturbations (all frequencies stay finite and positive),
and every candidate records whether its table is strictly monotone; they are not
silently dropped from the preregistered directionality check.

CPU/no-card receipt:

```bash
bash rebuttal/rebuttal_0723/experiments/profiled_residual_5090/run_5090.sh preflight
```

First GPU gate:

```bash
bash rebuttal/rebuttal_0723/experiments/profiled_residual_5090/run_5090.sh run-primary
```

Only if the primary gate says `PROCEED_TO_MATCHED_TRAINING`, optionally profile
the two other frozen checkpoints (the command resumes completed primary rows):

```bash
bash rebuttal/rebuttal_0723/experiments/profiled_residual_5090/run_5090.sh run-full
```

Raw output is written only under
`/root/autodl-tmp/profiled_residual_5090/`. A positive single-seed/window-
bootstrap gate authorizes preparation of the matched Affine-EVQ vs Full-EVQ
training experiment; it is not itself a paper claim or a seed-level CI.
