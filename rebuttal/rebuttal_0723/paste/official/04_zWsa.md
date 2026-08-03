Three additions, all checkable against the sources: two on your first and second questions, one on your fourth.

**1. What the FMRoPE rule requires, in its authors' own words.**

Your first question asks for the technical difference. Part of it is functional rather than formal, and Oka et al. state it themselves. Their rule is defined by \\(\\theta_{\\rm train} = L_{\\rm train}\\) and \\(\\theta_{\\rm infer} = L_{\\rm target}\\), so it needs the deployment length declared in advance; their §6.3 calls that requirement "a practical limitation" and names adaptive schemes as future work.

EVQ-Cosh sets the training-time grid and introduces no deployment target. A range method is target-aware by construction; an allocation method is not. This is not a claim that ours is better — under retargeting their rule obtains lower NLL, as we reported. It is that a rule which requires \\(L_{\\rm target}\\) and a rule which does not are not the same rule, whatever their effects have in common.

**2. The control that removes parameterization and tuning from the comparison.**

Your second question asks for a matched comparison. Beyond the exact-range control already given, we ran the complementary one, in which nothing is tuned at all: the positional operator, training protocol and evaluation are fixed and every arm is a zero-parameter analytic schedule, so parameter count and tuning budget are identical by construction rather than by matching effort. EVQ-Cosh minus geometric mean tail NLL is **−0.256/−0.305/−0.223/−0.238** at 1K/2K/4K/8K across three agreeing seeds, with span-matched uniform and deformation-matched arms included.

We also report what that ladder does not show: a deformation-matched exponential is statistically indistinguishable from Cosh (+0.0007 NLL, sign-flip p = 0.836). What survives is that non-geometric allocation beats geometric under identical capacity and effort — the axis, not the particular curve. The surrogate's role is to supply that curve in closed form without a search, and its own validation is functional: allocations derived from it reduce the exact-kernel collision score by **24–92% across 12 configurations**.

**3. On scale, whether the 8B model uses distant content or reconstructs it locally.**

At true 16K, deleting the remote gold block from every head worsens EVQ NLL by **1.506** while leaving the matched Native control essentially unchanged.

The missing Oka et al. citation is real and it is ours; the revision will cite and directly compare. If the distinction above still does not read as sufficient, we would rather know which part of it fails than leave it open.
