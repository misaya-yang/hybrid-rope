Your review states four conditions under which your evaluation would increase. With the discussion closing, we record their status in one line each; full answers stand in our response above.

**1. Novelty over FMRoPE.** Answered at the parameterization level: FMRoPE selects the base — the band moves, the exponents stay uniform; EVQ-Cosh holds the base fixed and replaces the exponent map, in closed form before training. What is parameterized, and at what stage, differs. The dead-channel observation is credited to Barbero et al. (2025) in our §2 and in Oka et al.

**2. Direct matched comparison.** Run: with both sampled extrema, the log span and all 32 anchors pinned and only the 30 interior frequencies changed, OOD NLL improves by 0.478/0.205/0.113 at 512/1K/2K. We also report where it loses — under target retargeting the ordering reverses — and that the two rules compose.

**3. RULER.** Included: at 8B under matched 8K supervision, 16K official macro goes 0.295% → 14.03%, with the training-length metric disagreement and the in-window cost reported rather than selected around.

**4. Scale.** Matched 1.485B arms: strict free-running exact 18/100 versus 98/100 at 8K, 0/100 versus 60/100 at 16K; from scratch at the same size the NLL delta flips from +0.0381 at 4K to −0.0437/−0.1351 at 8K/16K; the submitted 8B PPL goes 176.3 → 21.5 at 16K.

The missing Oka et al. citation is ours; the revision will cite and directly compare. We would appreciate knowing if any of the four remains insufficient on its own terms.
