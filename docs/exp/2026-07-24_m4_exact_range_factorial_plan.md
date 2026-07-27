# M4 exact-range pure-shape factorial

## Question

Does the operating rule \(\tau=d_{\rm head}/\sqrt{L_{\rm train}}\) select a
useful **interior allocation shape** after removing its finite-\(K\)
endpoint/span confound?

This is a supporting/mechanistic 50M experiment. It is not scale or downstream
evidence and does not change paper tables.

## Registered design

- Data/harness: the existing Phase16 local WikiText stream and 1% supervised
  passkey training mix.
- Budget: 8,388,608 tokens per run; 4 fixed natural-text PPL chunks at
  \(L,2L,4L,8L\).
- Grid: base \(\in\{500K,1M\}\), \(L_{\rm train}\in\{256,1024\}\),
  \(d_{\rm head}\in\{128,64,32\}\), seeds \(\{42,137,256\}\).
- Arms: native Geo; anchored Cosh at \(0.75\tau_\star,\tau_\star,1.25\tau_\star\);
  and an anchored exponential matched to the rule-Cosh RMS deformation.
- Total: 180 matched runs.

For every arm, the largest and smallest sampled frequency and the log-frequency
span exactly equal native Geo. Only the interior \(K-2\) channel positions
change. Each arm resets the same seed immediately before model construction;
batch indices and passkey rows are deterministic.

## Falsifiable interpretation

- Rule-Cosh consistently better than exact-range Geo: the rule captures a
  pure-shape basin rather than merely changing range.
- Another preregistered Cosh point wins: allocation remains useful, but the old
  rule is not a reliable pure-shape selector.
- Exponential wins: allocation is an independent axis, but Cosh is not uniquely
  favored.
- No exact-range shape wins: much of the historical raw midpoint-Cosh gain came
  from its implicit endpoint/span change.

Historical Phase16 99-run comparisons are descriptive only because their raw
midpoint-Cosh schedules did not hold sampled range fixed.

## Operation

```bash
bash scripts/2026-07/24_m4_exact_range_factorial.sh start
bash scripts/2026-07/24_m4_exact_range_factorial.sh status
bash scripts/2026-07/24_m4_exact_range_factorial.sh stop
bash scripts/2026-07/24_m4_exact_range_factorial.sh report
```

Raw checkpoints/results are resumable under
`results/theory/phase16_exact_range_factorial_m4_20260724/`. A SIGTERM saves
the current run at the next safe point. Estimated runtime from the historical
M4 Phase16 rate is roughly 35--42 hours.

## Frozen follow-up amendment — 2026-07-25

The 180-run five-arm grid is unchanged. The following follow-up is registered
before its results are available:

- The exponential arm matches formula-Cosh by normalized log-frequency node
  RMS displacement from uniform, while retaining identical sampled extrema and
  log-span. Full vectors and numerical matching error are recorded.
- Only the canonical-base (`500K`) minimum- and maximum-rule-\(\tau\)
  configurations add `0.5x` and `1.5x` Cosh arms: two configs, two arms, three
  seeds, 12 new runs.
- Every final checkpoint is evaluated at `1x/2x/4x/8x` under its fixed training
  range and a target-matched range whose runtime base is the target window
  length.
- Geo-trained and formula-Cosh-trained checkpoints receive a `2x2`
  train-shape/runtime-shape cross-swap under common runtime extrema/span.
- Geo, formula-Cosh, and matched-exponential retain `25/50/75/100%`
  checkpoints and receive fixed-range OOD ranking analysis. Already completed
  runs may be deterministically replayed only to recover missing milestones;
  their final model-state hash must match before the replay is admitted.
- Final evaluation uses four frozen windows from both WikiText validation and
  test splits. It reports full NLL, four equal position-bin NLLs, and final-128
  token NLL.

Aggregation first averages anchors, then pairs the three seeds inside each of
the 12 structural configurations, then averages those 12 configuration means.
Lengths, windows, and the 180 runs are not independent statistical samples.
