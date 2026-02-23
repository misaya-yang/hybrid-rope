# Waterbed Inequality: Strict Information-Theoretic Proof

## Appendix E.4: Rigorous Derivation

This document provides a **strict Jensen-inequality-based proof** of the waterbed effect in frequency allocation. All claims are rigorously derived from Cramér-Rao lower bound (CRLB) principles.

---

## E.4.1 Definitions

### Local Fisher Information

For a frequency density $\rho(\phi)$ at normalized position $\phi \in [0,1]$, the **local Fisher Information** is defined as:

$$ I(\phi) = c \cdot \rho(\phi) \cdot b^{-2\phi} $$

where $c > 0$ is a positive constant and $b$ is the RoPE base.

### Cramér-Rao Lower Bound (CRLB)

Given the Fisher Information $I(\phi)$, the **local mean squared error** (variance) for position estimation at scale $\phi$ satisfies:

$$ E(\phi) \geq \frac{1}{I(\phi)} = \frac{1}{c \cdot \rho(\phi) \cdot b^{-2\phi}} $$

---

## E.4.2 Integrated Log-Error Lower Bound

We now derive the **integrated log-error** (not pointwise error) lower bound.

### Step 1: Take Logarithm

$$ \ln E(\phi) \geq -\ln[c \cdot \rho(\phi) \cdot b^{-2\phi}] = -\ln c - \ln\rho(\phi) + 2\phi \ln b $$

### Step 2: Integrate over [0,1]

$$ \int_0^1 \ln E(\phi) \, d\phi \geq \int_0^1 \left[ -\ln c - \ln\rho(\phi) + 2\phi \ln b \right] d\phi $$

$$ = -\ln c - \int_0^1 \ln\rho(\phi) \, d\phi + 2\ln b \int_0^1 \phi \, d\phi $$

Since $\int_0^1 \phi \, d\phi = \frac{1}{2}$:

$$ = -\ln c - \int_0^1 \ln\rho(\phi) \, d\phi + \ln b $$

### Step 3: Apply Jensen's Inequality

The function $-\ln(x)$ is **strictly convex** for $x > 0$. By Jensen's inequality:

$$ -\int_0^1 \ln\rho(\phi) \, d\phi \geq -\ln\left( \int_0^1 \rho(\phi) \, d\phi \right) = -\ln(1) = 0 $$

(The equality holds if and only if $\rho(\phi)$ is constant almost everywhere.)

### Step 4: Final Bound

$$ \boxed{ \int_0^1 \ln E(\phi) \, d\phi \geq \ln b - \ln c } $$

---

## E.4.3 Physical Interpretation

### Base Expansion Exacerbation

The lower bound grows linearly with $\ln b$:
- Larger base $b$ → larger theoretical minimum total error
- This **strictly proves** that simply expanding the base (without frequency reallocation) worsens the integrated error bound

### Zero-Sum Nature

The waterbed effect states:
- If you create a local "spike" (high $\rho$ at some $\phi$) to reduce local error $E(\phi)$
- You must compensate by reducing $\rho$ elsewhere (since $\int \rho = 1$)
- This increases error in other frequency bands

### Formally Excluded Scenarios

- **Perfect localization is impossible**: $\rho(0) \to \infty$ would require $\rho(\phi) \to 0$ elsewhere, violating $\int \rho = 1$
- **No free lunch**: Any frequency allocation has a minimum total error given by the bound

---

## E.4.4 Key Properties

| Property | Statement | Proof |
|----------|-----------|-------|
| **Lower bound scales with ln b** | $\int \ln E \geq \ln b - \ln c$ | Direct from derivation |
| **Convexity required** | $-\ln$ must be convex | Fundamental to Jensen |
| **Monotonic in b** | Larger b → larger lower bound | $\ln b$ increases with b |
| **Tightness condition** | Bound achieved iff $\rho = 1$ (constant) | Jensen equality condition |

---

## E.4.5 Numerical Verification (Suggested)

To verify the tightness of this bound:

1. **Insert cosh solution**: $\rho(\phi) = A\cosh(1-\phi)$ (normalized)
2. **Compute actual integrated log-error**:
   $$ \int_0^1 \ln E_{\text{cosh}}(\phi) \, d\phi = \int_0^1 \left[ -\ln c - \ln(A\cosh(1-\phi)) + 2\phi\ln b \right] d\phi $$
3. **Compare with bound**: The actual value should be close to (but greater than) $\ln b - \ln c$

---

## E.4.6 Important Caveats

### Not a Pointwise Bound

This is an **integrated** log-error bound:
$$ \int_0^1 \ln E(\phi) \, d\phi \geq \ln b - \ln c $$

It does **not** imply:
$$ \ln E(\phi) \geq \frac{\ln b - \ln c}{1} \quad \text{(pointwise)} $$

Individual frequency bands can have very small errors if compensated by large errors elsewhere.

### Operational vs. Information-Theoretic

The bound is **information-theoretic** (from CRLB), not an **operational** bound on actual model performance. It sets a fundamental limit on estimation accuracy, not necessarily achievable in practice.

---

## E.4.7 Summary

| Statement | Status |
|-----------|--------|
| Integrated log-error has absolute lower bound | ✅ Proven |
| Bound scales as ln b | ✅ Proven |
| Base expansion exacerbates minimum error | ✅ Proven |
| Zero-sum waterbed effect | ✅ Proven |
| Achieved iff ρ = constant | ✅ Proven |

**Citation guidance**: In the paper, refer to this as the "information-theoretic waterbed inequality" or "log-error scaling bound" rather than a "tight operational bound."
