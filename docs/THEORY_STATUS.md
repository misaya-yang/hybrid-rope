# Theory Status Registry

This document tracks the verification status of all theoretical results in the hybrid-rope paper.

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ Proven | Rigorous mathematical proof provided |
| ❌ Rejected | Proven false or based on incorrect assumptions |
| 🔄 Pending | Requires further verification |
| ⚠️ Warning | Correct but with limited applicability |

---

## Main Theorems

| Result | Status | Verified By | Notes |
|--------|--------|-------------|-------|
| Theorem 1 (Uniform exactness) | ✅ Proven | Sinc envelope bound | Appendix A |
| Theorem 2 (Linguistic convexity) | ✅ Proven | Ci asymptotics | Appendix B |
| Theorem 3 (Proxy trap) | ✅ Proven | Kronecker density | Appendix C |

---

## Approximation Results

| Result | Status | Verified By | Notes |
|--------|--------|-------------|-------|
| Diagonal residual O(1/ln b) | ✅ Numerical | Spot-check b=10000: ~11% | Appendix D |

---

## Structural Stability (Appendix E)

| Result | Status | Verified By | Notes |
|--------|--------|-------------|-------|
| cosh exact solution (b→∞) | ✅ Proven | Brownian covariance + ODE | Appendix E.1 |
| cosh uniform validity at ϕ=0 | ✅ Proven | Direct Ci evaluation at boundary | Appendix E.1.1 |
| Order-of-limits non-commutativity | ✅ Proven | Jacobian + Riemann-Lebesgue | Appendix E.2 |
| Fredholm first-order correction | ✅ Proven | Closed-form ρ₁ = cosh(1-ϕ)-cosh(1) | Appendix E.3 |
| Waterbed inequality | ✅ Proven | Jensen + CRLB | Appendix E.4 |

---

## Rejected Results

| Result | Status | Reason | Notes |
|--------|--------|--------|-------|
| Euler-Maclaurin discrete acceleration | ❌ Rejected | Depends on unproven ρ(0)=0 boundary condition | Do NOT use |

---

## What NOT to Claim

- ❌ **Do NOT** claim exponential truncation ρ(0)=0 — this has been strictly proven FALSE (see Appendix E.1.1)
- ❌ **Do NOT** claim cosh solution is exact for finite b — it is an asymptotic b→∞ result
- ❌ **Do NOT** claim waterbed inequality is a "tight bound" — it is an integrated log-error lower bound
- ❌ **Do NOT** use Euler-Maclaurin acceleration — boundary condition is unproven

---

## Last Updated

2026-02-23
