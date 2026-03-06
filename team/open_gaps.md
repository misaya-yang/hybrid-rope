# Open Gaps

## Primary Gaps

1. **Larger-scale primary anchor**
   - We still need a cleaner large-model result on a harder downstream task family.

2. **Downstream fairness for Phase 15**
   - The EVQ side of the LongBench-style comparison did not complete because the remote dataset download failed.

3. **Theory polish**
   - The broadband surrogate step is still the main approximation bottleneck.
   - The scaling law remains an empirical law / conjecture rather than a theorem.

## Secondary Gaps

4. **Video remains supporting**
   - The temporal video evidence is organized and useful, but not yet strong enough to become a co-primary anchor.

5. **Single-seed larger-scale evidence**
   - Phase 9f and Phase 15 are both useful, but they remain supporting rather than main-paper anchors.

6. **Capacity-compensation hypothesis is motivated but not closed**
   - We now have a plausible mechanism story that stronger models may absorb short-range positional burden while far-range low-frequency structure remains irreducible.
   - This is not yet a paper claim; it still needs a dedicated scale/training sweep.

## Current Practical Priority Order

1. stronger large-scale text anchor,
2. DSR / harder retrieval evaluation,
3. scale-sweep for the capacity-compensation hypothesis,
4. theoretical tightening,
5. stronger video temporal package.
