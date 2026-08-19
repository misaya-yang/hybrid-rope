# Independent Claude audit record: full-RoPE collision

- **Source:** `analysis/full_rope_audit/draft_report.md`
- **Source SHA-256:** `7a32e8195bb7f9f3b0c2848257a2269cbbe26c62a05cfb7c50eabf6afc82928a`
- **Status:** useful independent derivation with known numerical/reporting
  defects; not the canonical paper source
- **Canonical reconciliation:**
  `../FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`

## Verified contribution

The audit independently establishes the correct geometric object for one RoPE
frequency:

\[
V_\omega=\operatorname{span}\{\cos(\omega\Delta),\sin(\omega\Delta)\}.
\]

Its main valid conclusions are:

1. the paper's cosine-only kernel is the cos-cos block of the full Gram and
   omits sin-sin and cross terms;
2. whitened cross-Gram singular values are phase-invariant canonical
   correlations between frequency subspaces;
3. low frequencies converge to the common subspace
   \(\operatorname{span}\{1,\Delta\}\);
4. cosine-only collision can rank two tables opposite to a full-basis
   effective-rank measure;
5. collision order can reverse between \(L\), \(2L\), and \(4L\);
6. maximizing static rank produces near-harmonic/Fourier tables and therefore
   is not a language-model or extrapolation objective.

The updated v2 source also gives an analytic construction on the one-sided
uniform prior: frequencies \(\omega_k=\pi a_k/L\) with all integer \(a_k\) in
the same parity class have mutually orthogonal 2D subspaces. This attains full
static rank while capacity permits. It is useful as an exact characterization
of the static geometry—and as another reason static identifiability is not an
extrapolation objective—but it is not a proposed training schedule.

These conclusions agree with the independent canonical report.

## Reproduction record

| File | SHA-256 |
| --- | --- |
| `analysis/full_rope_audit/verify_core.py` | `f2a6a705d5ec29d07e6572be78aa84243ba7474186276d3bfe06d39f158ddb9e` |
| `analysis/full_rope_audit/verify_onesided.py` | `0a4dfd55b754f71a628bb19cfba2eb63e1270e6288e532768534bbad7e3698f0` |
| `analysis/full_rope_audit/verify_counterexamples.py` | `3f6ac44061771dfa35a2cd2555636da906d177ff91c97d531376c28a815a9534` |
| `analysis/full_rope_audit/verify_small_models.py` | `8664d20ca38d452131c1d13fc7a1a13153c9170bcac25edc26c907c0da0bc0ac` |

Read-only rerun on 2026-08-19:

- full symmetric Gram checks, canonical-correlation invariance, two-block
  logdet identity, and low-frequency \((\omega L)^4\) rate passed;
- the original coarse one-sided quadrature check produced `6.76e-6` error
  while demanding `1e-9`; after increasing quadrature density from `32L` to
  `128L`, the current check passes at `4.22e-7` under a `5e-7` tolerance;
- `verify_counterexamples.py` rejected one older causal example as corrupted,
  then independently found valid within-range violations in `3215/40000`
  sampled comparisons (`8.04%`). The existence claim survives; the rejected
  example must not be cited.

## Do not inherit verbatim

- The draft's `G_attn` discussion chooses one particular
  \(F_z=\ell\ell^\top\) reading and then concludes that the object is not a
  Fisher information. The canonical analysis instead keeps softmax structural
  Fisher and LM empirical-gradient outer products separate.
- The draft mixes causal and symmetric distance grids and multiple effective
  rank definitions. Preserve those definitions when quoting any number.
- The draft says the training model “uses” the slow subspace based on a
  unit-coefficient recency calculation. That calculation proves available
  geometry, not learned coefficient usage.
- The small-model script exactly verifies same-parity orthogonality and finite
  enumerations. Claims about the full oversampled \(K,L\) phase diagram remain
  an open extension rather than a current manuscript theorem.

## Paper use

Use the independent audit as corroboration, not as a manuscript owner. The
paper-facing formulation should come from the canonical report:

- full self/cross Gram;
- canonical collision;
- exact stable-rank identity;
- low-frequency and softmax-centered collapse;
- explicit counterexamples showing that static geometry does not imply
  extrapolation.
