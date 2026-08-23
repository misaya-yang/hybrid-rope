# Attention-measure kappa audit

- **Date:** 2026-08-20
- **Status:** CPU-only Tier 1 complete; preregistered Branch C
- **Paper role:** internal falsification result; not a manuscript claim
- **Implementation:** `scripts/analysis/attention_fisher_50m_probe.py`
- **Implementation SHA-256:** `9c5d79b09c22f036eee695142afc70ec783ac972ca2a503a598b8d4bfff3e472`

## Protocol

The audit reuses the four cells, checkpoints, held-out TinyStories tensor, eight
windows, five query positions, six layers, and eight heads of the canonical
50M co-adaptation probe: 1,920 aligned head-query observations per cell.  The
numerical direction `g` is computed once from the Geo-weights/Geo-table cell and
held fixed; each cell supplies only its own attention distribution `p`.
Aggregation is Eq. (37)'s ratio-of-means, not a mean of per-query ratios.

Two directions were registered before execution:

1. the first-order Eq. (35) direction at the Geo table;
2. the realised finite Geo-to-EVQ logit difference of Eq. (36).

No direction, aggregation, layer/head subset, or data subset was selected after
observing the result.  A global shift or scale of `g` cancels numerically, and
uniform attention returns exactly `kappa_att=1/L`.

## Tier 1 result

| weights | runtime table | PPL | static r2 | kappa, Eq. (35) | kappa, finite swap |
| --- | --- | ---: | ---: | ---: | ---: |
| Geo | Geo | 7.1413 | 4.5691 | 0.00046476 | 0.00246857 |
| Geo | EVQ | 76.1955 | 12.5351 | 0.00253131 | 0.00152003 |
| EVQ | Geo | 23.0524 | 4.5691 | 0.00114364 | 0.00455603 |
| EVQ | EVQ | 7.1597 | 12.5351 | 0.00053535 | 0.00252844 |

- First order: Spearman `rho=+1.0` against log PPL; the self-consistent and
  mismatched cells are separated.
- Finite swap: Spearman `rho=-0.2`; the self-consistent and mismatched cells are
  not separated.
- The rankings disagree.  The preregistered finite-tau rule therefore selects
  **Branch C**: Tier 1 fails.

The finite-swap expression agrees with direct rotated-logit subtraction to
maximum absolute error `1.78e-15`; manual attention agrees with SDPA to at most
`1.25e-6`.  A deterministic full rerun reproduced the complete kappa payload
hash.  The attractive first-order ordering is not promoted.

Internal raw receipts are intentionally excluded from the anonymous package:

- full JSON SHA-256:
  `7ffc04412bf2a22bb488cfce2700d50b7d6b14bb756d031e7d95f507210369c1`;
- four-cell CSV SHA-256:
  `b1920a8d38a8b9473f7b232e0471b450e844f0eb57149b205f4d54a7d773fc2c`;
- robustness CSV SHA-256:
  `2b729b4e5bbd1141aae20581db519092ac7ab226553c9daebe3164205b47d287`.

## Tier 2 checkpoint audit

Tier 2 requires Geo and non-uniform checkpoints for every registered M4
structural configuration.  They are absent locally, and the canonical M4 owner
records that weights were cleaned after evidence freezing.  Its surviving
evidence JSON contains metrics and provenance, not attention probabilities.
Weekend-sweep checkpoints have different protocols and cannot substitute.

Tier 2 was therefore not executed.  Given the Tier 1 Branch C outcome, no
training rerun or checkpoint-recovery campaign is justified by this go/no-go
task.

## Manuscript decision

Keep the current paper architecture: static full-RoPE geometry accounts for
finite-basis redundancy, while trained effects are measured directly.  Do not
promote Eq. (37) as the missing predictor, do not infer the proposed crossover
law from it, and do not cherry-pick layer/head subsets.
