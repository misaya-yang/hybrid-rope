Two additions to our response, both checkable against sources rather than against assertions of ours.

1. On your first question, the technical difference: one part of it is functional, and Oka et al. state it in their own §6.3.
2. On your second question, the matched comparison: beyond the exact-range control already given, we ran a complementary control in which nothing is tuned at all.

**1. What the FMRoPE rule requires, in its authors' own words.**

Their rule is defined by \\(\\theta_{\\rm train} = L_{\\rm train}\\) and \\(\\theta_{\\rm infer} = L_{\\rm target}\\), so it needs the deployment length declared in advance. Their §6.3 calls that requirement "a practical limitation" and names adaptive schemes as future work.

EVQ-Cosh sets the training-time grid and introduces no deployment target. A range method is target-aware by construction; an allocation method is not. This is not a claim about which rule performs better, which our response addresses separately. It is that a rule which requires \\(L_{\\rm target}\\) and a rule which does not are not the same rule, whatever their effects have in common.

**2. The control in which parameterization and tuning are removed by construction.**

The positional operator, training protocol and evaluation are fixed, and every arm is a zero-parameter analytic schedule, so parameter count and tuning budget are identical by construction rather than by matched effort. EVQ-Cosh minus geometric mean tail NLL is **−0.256/−0.305/−0.223/−0.238** at 1K/2K/4K/8K across three agreeing seeds, with span-matched uniform and deformation-matched arms included.

We also report what that ladder does not show. A deformation-matched exponential is statistically indistinguishable from Cosh (+0.0007 NLL, sign-flip p = 0.836), so what survives is that non-geometric allocation beats geometric under identical capacity and effort — the axis, not the particular curve. The surrogate's role is to supply that curve in closed form without a search, and its own validation is functional: allocations derived from it reduce the exact-kernel collision score by **24–92% across 12 configurations**.

The missing Oka et al. citation is real and it is ours; the revision will cite and directly compare.

If the distinction above still does not read as sufficient, we can say which part we think does the work.
