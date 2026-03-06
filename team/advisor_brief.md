# Advisor Brief

## What We Have Done

- Formulated RoPE frequency allocation as a variational inverse problem.
- Derived a closed-form EVQ-cosh family, with geometric RoPE recovered as the `tau = 0` limit.
- Built a paper package around three anchored claims:
  1. closed-form theory for training-time frequency allocation,
  2. EVQ beats learnable PE in DAPE-style extreme extrapolation,
  3. `EVQ + YaRN >> Geo + YaRN` in the main long-context systems setting.

## Strongest Current Evidence

- `350M` multi-seed raw text results: stable long-range PPL gains with near-zero short-range cost.
- `Phase 11 (L=256)`:
  - direct support for the `tau*` prediction,
  - strong PE-dominant extrapolation gains,
  - EVQ×YaRN leverage becomes much larger than Geo×YaRN.
- `Passkey mix` multi-seed:
  - raw retrieval gains are stable,
  - EVQ+YaRN reaches `100% @ 8K` while Geo+YaRN does not.
- `750M continue@4K` supporting signal:
  - long-range gains persist at larger scale,
  - `16K PPL -45.9%`,
  - `8K AR exact 77.5% vs 0%`.

## New Mechanism Question We Now Want To Test

We now have a more specific scale-dependent hypothesis:

- a sufficiently strong model may absorb part of the short-range cost caused by weaker high-frequency positional detail,
- but true long-range separation still requires strong low-frequency positional structure and cannot be recovered by model capacity alone.

The most suggestive observation is from the recent Geo continued-pretraining run:

- `2K PPL: 107.36 -> 36.45`
- `4K PPL: 172.20 -> 36.49`
- `8K PPL after training: 35.04`
- `16K PPL after training: 45.53`

The surprising point is that `8K PPL` is even slightly lower than `2K PPL`. This is not proof, but it is strong motivation to test whether model scale and training sufficiency can absorb short/mid-range positional burden while leaving far-range low-frequency structure as the irreducible bottleneck.

Related note:

- `team/plans/capacity_compensation_hypothesis.md`

## What Is Still Missing

- A cleaner larger-scale primary anchor on a real downstream benchmark.
- A tighter perturbation-bound story for the broadband surrogate step.
- A more polished theorem/conjecture boundary for the scaling law.
- More mature cross-modal evidence if we want video to raise the paper ceiling rather than stay supporting.

## Where Help From You Matters Most

1. Tightening the surrogate approximation argument.
2. Stress-testing the theorem / proposition / conjecture split.
3. Advising whether the new "capacity compensation" hypothesis is theoretically coherent enough to justify a dedicated scale sweep.
4. Advising whether the next largest budget block should go to a `1.5B text anchor` or a stronger `video temporal theory+experiment package`.

## Suggested Reading

1. `paper_draft/mainstory.md`
2. `paper_draft/CORE_THEORY.md`
3. `paper_draft/figs/README.md`
4. `docs/exp/2026-03-04_phase11_L256_results.md`
5. `docs/exp/2026-03-06_phase15_750m_2k_to_4k_continue_results.md`
6. `team/plans/capacity_compensation_hypothesis.md`
