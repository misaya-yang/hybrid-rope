"""Zero-training table screen: score many frozen-model frequency tables per hour.

The question this package exists for is the project's original one: **can any
table beat MrRoPE with no training at all?**  Nothing here trains, adapts,
fine-tunes or writes a checkpoint.  Every arm is a table of 64 frequencies plus
a gain, installed into a frozen checkpoint, and scored on two measured numbers:

    native_kl    E_u[ KL( p_native || p_table ) ]   -- what the table costs
    long_nll     mean NLL beyond the native window  -- what the table buys

Those two are the plan's objective and constraint respectively, and the pair per
arm is a point on the (price, benefit) plane.  The screen's output is that plane,
not a winner: a table that is dominated by MrRoPE is a result, and so is one that
is not.

WHY THIS CAN DO TEN ARMS AN HOUR.  The archived chain in `curvature_20260910`
computes its long-range gradient by CENTRAL DIFFERENCES -- 36 forwards at 128K
for a 36-slot sample, 60 for a wider one, 20-34 minutes of card time for ONE
gradient.  The screen does not need a gradient at all: it needs a SCORE per arm,
which is one forward per document, amortised over a base pass that is computed
once for every arm.  At the recorded 4.1 s per 32K forward, eight documents put
an arm at ~33 s and the first hour at ~100 arms; at 128K (33.9 s) it is ~4.5
minutes an arm and ~13 arms, which is still over the ten the campaign asks for.
The gradient path, when it is wanted, is `joint_kkt_20260910.joint_grad`, which
gets all 65 design derivatives from one forward and one backward.

WHAT THIS PACKAGE DOES NOT DO, stated because it is the thing most likely to be
over-read.  It does not measure task ability: `long_nll` is a language-modelling
number on held-out documents, not a RULER score, and a table that wins on it may
lose on retrieval.  Every receipt says so.  The screen ranks ARMS; it does not
establish that the winner is better at anything a user cares about, and the
panel jobs that would establish that are a separate, queued step.
"""
