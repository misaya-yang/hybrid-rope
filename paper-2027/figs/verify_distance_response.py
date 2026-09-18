"""CPU checks of the selected Pro-report increments, not model validation."""
from fractions import Fraction as F
import json
import math
from pathlib import Path


def profiles(n):
    t = [F(q * (3*n*n + 3*n + 1 - q*q), n*(n+1)*(2*n+1)) for q in range(n+1)]
    p = [F(q*(q+1), n*(n+1)) for q in range(n+1)]
    return t, p


def verify():
    counts = {"exact_grids": 0, "tail_residual_comparisons": 0, "crossover_comparisons": 0}
    max_identity_error = 0.0
    for n in range(1, 129):
        t, p = profiles(n)
        e = [t[q] - t[q-1] for q in range(1, n+1)]
        assert sum(e) == 1 and min(e) > 0
        for q in range(n+1):
            assert t[q]-p[q] == F(q*(n-q)*(3*n+q+1), n*(n+1)*(2*n+1))
        # Independently apply the quadratic objective's tridiagonal matrix.
        he = [F(0) for _ in e]
        for q in range(n-1):
            delta = e[q+1]-e[q]
            he[q] -= delta
            he[q+1] += delta
        he[-1] += e[-1]
        optimum = F(6, n*(n+1)*(2*n+1))
        assert all(v == optimum for v in he)
        assert sum((e[q+1]-e[q])**2 for q in range(n-1)) + e[-1]**2 == optimum
        counts["exact_grids"] += 1
        for j in range(1, n):
            q = n-j
            rt = F(j*(j+1)*(3*n+1-j), n*(n+1)*(2*n+1))
            rp = F(j*(2*n+1-j), n*(n+1))
            assert 1-t[q] == rt and 1-p[q] == rp
            assert rt/rp == F((j+1)*(3*n+1-j), (2*n+1)*(2*n+1-j))
            for s in (2, 4, 8, 16):
                ratio = math.expm1(float(rt)*math.log(s))/math.expm1(float(rp)*math.log(s))
                assert 0 < ratio <= float(rt/rp) + 1e-14
                counts["tail_residual_comparisons"] += 1
                at, ap = s**(-float(t[q])), s**(-float(p[q]))
                threshold = 2/(at+ap)
                assert 1 < threshold < s
                for alpha in (1, (1+threshold)/2, (threshold+s)/2, s):
                    lhs = (alpha*at-1)**2-(alpha*ap-1)**2
                    rhs = alpha*(at-ap)*(alpha*(at+ap)-2)
                    error = abs(lhs-rhs)
                    max_identity_error = max(max_identity_error, error)
                    assert error < 1e-12
                    assert (lhs < 0) == (alpha > threshold)
                    # Choose a small reference phase so both residuals stay in [-pi,pi].
                    phi = 0.1
                    ct, cp = math.cos(phi*(alpha*at-1)), math.cos(phi*(alpha*ap-1))
                    assert (ct-cp)*(alpha-threshold) >= -1e-14
                    counts["crossover_comparisons"] += 1
    grids = []
    for name, k, base, length in (
        ("Llama", 64, 500000, 8192), ("OLMo", 64, 500000, 4096),
        ("Qwen", 64, 1000000, 32768), ("GLM", 32, 10000, 32768),
        ("Kanana", 64, 8000000, 32768),
    ):
        omega = [base**(-q/k) for q in range(k)]
        turns = [length*w/(2*math.pi) for w in omega]
        lo = max(q for q, v in enumerate(turns) if v > 32)
        hi = min(q for q, v in enumerate(turns) if v < 1)
        n = hi-lo
        t, p = profiles(n)
        # z/m relation evaluated against direct log-frequency normalization.
        span = (k-1)*math.log(base)/k
        for s in (2, 4):
            xs = [-math.log(w) + math.log(s)*float(t[min(max(q-lo, 0), n)]) for q, w in enumerate(omega)]
            for q in range(k):
                m = float(t[min(max(q-lo, 0), n)])
                formula = (span*q/(k-1)+math.log(s)*m)/(span+math.log(s))
                assert abs(formula-(xs[q]-xs[0])/(xs[-1]-xs[0])) < 1e-14
        ratios = {}
        for s in (2, 4):
            ratios[str(s)] = math.expm1(float(1-t[-2])*math.log(s))/math.expm1(float(1-p[-2])*math.log(s))
        grids.append({"model": name, "base": base, "L": length, "K": k,
                      "band": [lo, hi], "n": n, "terminal_bound": 3/(2*n+1),
                      "terminal_phase_ratio": ratios})
    return {"status": "PASS", "scope": "Selected finite-grid, unwrapped phase and aligned-reference identities; no model evaluation or FP32 installation replay",
            "counts": counts, "max_crossover_identity_error": max_identity_error,
            "public_float64_grids": grids}


if __name__ == "__main__":
    result = verify()
    Path(__file__).with_name("selected_theory_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
