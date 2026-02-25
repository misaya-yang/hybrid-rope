# Theory Status Registry

This document tracks what we can rigorously claim *now* for a theory-framework-first submission.

## Paper Positioning

This work is positioned as a **theoretical framework paper**, not a pure benchmark paper.
The acceptance core is:

1. Formal problem definition and mathematical closure.
2. Proof-level structural conclusions (validity boundary + failure boundary).
3. Falsifiable predictions that are partially/fully confirmed empirically.

Experiments are used as **consistency and falsification evidence**, not as the sole value source.

## Status Legend

| Symbol | Meaning |
|--------|---------|
| `✅ Proven` | Rigorous mathematical proof provided |
| `🧪 Supported` | Empirically supported under current protocol |
| `🔄 Pending` | Running or not yet fully verified |
| `⚠️ Scoped` | Correct only under explicit assumptions |
| `❌ Rejected` | Proven false or based on invalid assumptions |

---

## Main Theorems (Primary Acceptance Axis)

| Result | Status | Verified By | Notes |
|--------|--------|-------------|-------|
| Theorem 1 (Uniform exactness) | ✅ Proven | Sinc envelope bound | Appendix A |
| Theorem 2 (Linguistic convexity) | ✅ Proven | Ci asymptotics | Appendix B |
| Theorem 3 (Proxy trap) | ✅ Proven | Kronecker density | Appendix C |

---

## Structural Stability (Appendix E)

| Result | Status | Verified By | Notes |
|--------|--------|-------------|-------|
| cosh exact solution (`b -> inf`) | ✅ Proven | Brownian covariance + ODE | Appendix E.1 |
| cosh boundary validity at `phi=0` | ✅ Proven | Direct Ci evaluation at boundary | Appendix E.1.1 |
| Order-of-limits non-commutativity | ✅ Proven | Jacobian + Riemann-Lebesgue | Appendix E.2 |
| Fredholm first-order correction | ✅ Proven | Closed-form `rho_1 = cosh(1-phi)-cosh(1)` | Appendix E.3 |
| Waterbed inequality | ✅ Proven | Jensen + CRLB | Appendix E.4 |

---

## Approximation and Numerical Checks

| Result | Status | Verified By | Notes |
|--------|--------|-------------|-------|
| Diagonal residual `O(1/ln b)` | 🧪 Supported | Spot-check (`b=10000`: ~11%) | Appendix D |

---

## Current Empirical Snapshot (Theory-Consistency Use)

Timestamp: `2026-02-23 20:57 CST`  
Protocol: fair 8B downstream eval (`NIAH + LongBench + Passkey-TF`)

### Confirmed now

| Observation | Status | Interpretation for Theory |
|------------|--------|---------------------------|
| `sigmoid` NIAH finished and saturated (`1.0`) | 🧪 Supported | Consistent with convex-prior branch not harming retrieval |
| `baseline` has NIAH misses at `16K` shallow depths | 🧪 Supported | Consistent with mismatch under non-uniform priors |
| LongBench partial averages: `sigmoid` currently strongest among completed methods | 🧪 Supported | Supports "better trade-off curve", but not yet final global claim |
| Passkey-TF at `16K` all methods accuracy near ceiling | 🧪 Supported | Use margin, not raw accuracy, for discrimination |

### Still running

| Item | Status | Why it matters |
|------|--------|----------------|
| `anchored_sigmoid` downstream full set | 🔄 Pending | Final practical ceiling and reviewer-facing killer evidence |
| Full multi-seed stability table | 🔄 Pending | Needed for stronger significance claims |

---

## Claim Ladder (What We Can Claim Today)

### Level A: Strong theorem claims (safe now)
- "Optimal frequency allocation is prior-dependent, not universal."
- "Uniform prior -> geometric optimality; linguistic power-law prior -> convex decaying optimum."
- "Proxy optimization can induce prior mismatch and false optima."
- "Waterbed-style information trade-off imposes hard structural constraints."

### Level B: Theory-consistent empirical claims (safe now, scoped)
- "`sigmoid` reaches NIAH saturation under current fair 8B protocol."
- "`sigmoid` currently leads completed LongBench averages under the same protocol."
- "Empirical behavior is consistent with the framework's predicted trade-off geometry."

### Level C: Hold until anchored finishes
- "Anchored variant is the definitive best practical instantiation."
- "Final SOTA-style superiority across all downstream metrics."
- "Strong significance statements across multiple seeds/tasks."

---

## Rejected / Forbidden Claims

| Result | Status | Reason | Notes |
|--------|--------|--------|-------|
| Euler-Maclaurin discrete acceleration | ❌ Rejected | Depends on unproven `rho(0)=0` boundary condition | Do NOT use |

### What NOT to Claim

- Do NOT claim exponential truncation `rho(0)=0`; this is proven false (Appendix E.1.1).
- Do NOT claim cosh solution is exact for finite `b`; it is asymptotic (`b -> inf`).
- Do NOT call waterbed inequality a "tight bound"; it is an integrated lower bound.
- Do NOT frame this as only a leaderboard paper; keep theory-first framing explicit.

---

## Last Updated

2026-02-23 20:57 CST
