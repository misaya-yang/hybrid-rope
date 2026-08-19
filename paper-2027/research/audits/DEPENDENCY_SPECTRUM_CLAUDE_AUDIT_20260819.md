# Independent Claude audit record: dependency-spectrum allocation

- **Source:** `results/dependency_spectrum_audit_20260819/REPORT.md`
- **Source SHA-256:** `1227f0b8dbe2c96c7ee0353128be1f9185d5f045d6556b3544f3c761739b5f4e`
- **Status:** internal falsification and measurement pilot; not a current
  manuscript claim

## Source and code record

| File | SHA-256 |
| --- | --- |
| `results/dependency_spectrum_audit_20260819/summary.json` | `706ad9e7c80676195617d31329fd5c80bacf80b2cf1163ea7aff99a02e3b3354` |
| `results/dependency_spectrum_audit_20260819/kr_pilot.json` | `d028d6f53dbb410b4b1930324c2e8cac9788c59c57390e533f4b722c4a0ca871` |
| `scripts/analysis/dependency_spectrum_audit.py` | `e441c8e2ed46ae3ef69b810cafc2f421a528ea267fc2b79ab9817fe2b88d8626` |
| `scripts/analysis/dependency_spectrum_measurement.py` | `e2fa81cb45c500ff19b83a28b5e45c81c2f4657c3fdf184365bc0007c40609ce` |

The `results/` tree is ignored and volatile. This record preserves the source
identity and conclusions without copying private/raw artifacts into the paper
package.

## Valid contribution

The audit is valuable mainly because it falsifies an attractive but unsupported
route:

1. a distance-demand prior does not by itself determine a finite frequency
   table;
2. additive per-frequency utility places every channel at the same best
   frequency unless a redundancy/interference term is added;
3. candidate rules \(\rho\propto p\), \(p^{1/3}\), and \(\sqrt p\) change rank
   with the chosen kernel and objective;
4. harmonic/resonance structure creates many near-optimal single-frequency
   points, invalidating a universal isolated-optimum Taylor argument;
5. the repository's scale-invariant \(1/r\) distance prior is an assumption,
   not a checkpoint measurement.

This supports the paper decision to characterize full-RoPE geometry without
claiming that a static or demand-weighted objective predicts task-optimal
allocation.

## Checkpoint pilot

The measurement script loads the existing 50M, \(L=512\), seed-42 checkpoints,
replaces SDPA with explicit softmax, and differentiates LM loss with respect to
attention logits. The saved pilot reports:

- explicit-softmax loss matching the fused path to `1e-5`;
- maximum output difference around `2.8e-5`;
- a fitted distance-gradient spectrum near \(r^{-2.4}\) for three \(\tau\)
  checkpoints on \(r\in[2,256]\), with \(R^2\) around `0.8`;
- mean short-distance versus far-distance gradient-square ratios of roughly
  `1000–2400x`.

These numbers describe one batch of two sequences with layers and heads
aggregated. The measured object also contains content, trained attention
probability, downstream Jacobians, and model co-adaptation. It is not yet a
universal dependency demand or an allocation objective.

## Known evidence gap

The report records a corrected multi-start reference such as
`heavy_tail K=32: 1.8975`, but that correction is not present in a separate
saved owner or raw artifact discovered during this audit. Do not quote the
corrected absolute optimum externally. The qualitative falsification of a
universal density rule does not depend on that number.

## Paper use

- Use internally to demote any unmeasured demand-prior or universal density
  story.
- Do not add the \(r^{-2.4}\) pilot, new \(\rho^*\), or a resonance optimizer
  to the ICML paper during the current rewrite.
- If later promoted, first freeze the metric definition and run multiple
  batches with layer/head distributions and a direct relationship to a
  decision-relevant LM endpoint. This is optional future work, not a blocker
  for the current paper.
