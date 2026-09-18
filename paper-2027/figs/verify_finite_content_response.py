"""Independent CPU checks for the focused finite-response proposal; no model runs."""
import cmath
from fractions import Fraction as F
import json
import math
from pathlib import Path
import random


def response(a, b, residual, reference_phase):
    coefficient = complex(a, b) * cmath.exp(-1j * reference_phase)
    return (coefficient * cmath.exp(1j * (reference_phase + residual))).real


def verify():
    rng = random.Random(20260918)
    cases = 0
    min_normalized_slack = math.inf
    max_tightness_error = 0.0
    certified_positive = 0
    intervals = [(0.0, math.pi), (0.0, .3), (.1, .3), (1.2, 1.8)]
    intervals += [tuple(sorted((rng.random()*math.pi, rng.random()*math.pi))) for _ in range(2000)]
    for r, t in intervals:
        a = 10**rng.uniform(-2, 2)
        kappa = rng.uniform(0, 2)
        mu, half_gap = (r+t)/2, (t-r)/2
        bound = 2*a*math.sin(half_gap)*(math.sin(mu)-kappa*abs(math.cos(mu)))
        ref = rng.uniform(-5, 5)
        for sign in (-1, 1):
            values = []
            for b in (-kappa*a, kappa*a, rng.uniform(-kappa*a, kappa*a)):
                diff = response(a,b,sign*r,ref)-response(a,b,sign*t,ref)
                slack = (diff-bound)/a
                min_normalized_slack = min(min_normalized_slack, slack)
                assert slack > -1e-12
                if bound > 1e-10*a:
                    assert diff > 0
                    certified_positive += 1
                cases += 1
                values.append(diff)
            # The minimum over the two endpoints of the allowed b interval attains the bound.
            error = abs(min(values[:2])-bound)/a
            max_tightness_error = max(max_tightness_error, error)
            assert error < 1e-12
    counter = response(1,-1,.1,0)-response(1,-1,.3,0)
    assert counter < 0
    qrows = []
    n, s, native_length, lo = 17, 4, 8192, 18
    w = F(3*n, 2*(2*n+1))
    ts = [F(q*(3*n*n+3*n+1-q*q), n*(n+1)*(2*n+1)) for q in range(n+1)]
    pro = [F(q*(q+1), n*(n+1)) for q in range(n+1)]
    control = [(1-w)*F(q,n)+w*(2*F(q,n)-pro[q]) for q in range(n+1)]
    assert sum(ts) == sum(control)
    for q in range(n+1):
        assert ts[q]-control[q] == F(q*(n-q)*(2*q-n), 2*n*(n+1)*(2*n+1))
    for q in (12,16):
        omega = 500000**(-(lo+q)/64)
        d0 = native_length/4
        phase = omega*d0
        far_t = phase*(s*s**(-float(ts[q]))-1)
        far_c = phase*(s*s**(-float(control[q]))-1)
        near_t = phase*(s**(-float(ts[q]))-1)
        near_c = phase*(s**(-float(control[q]))-1)
        mu = (far_t+far_c)/2
        qrows.append({"q":q,"far_delta_T":far_t,"far_delta_C":far_c,
                      "far_unit_signal_T_minus_C":response(1,0,far_t,phase)-response(1,0,far_c,phase),
                      "near_unit_signal_T_minus_C":response(1,0,near_t,phase)-response(1,0,near_c,phase),
                      "sufficient_kappa_upper_bound":math.sin(mu)/abs(math.cos(mu))})
    return {"status":"PASS","scope":"Mathematical finite-response bound, tightness and public-grid examples; no checkpoint phase or task validation",
            "finite_response_cases":cases,"positive_bound_cases":certified_positive,
            "minimum_normalized_slack":min_normalized_slack,
            "max_normalized_tightness_error":max_tightness_error,
            "smaller_residual_can_reduce_signal_counterexample":counter,
            "llama_TC_mathematical_examples":qrows,
            "q12_to_q16_far_gain_ratio":qrows[0]['far_unit_signal_T_minus_C']/qrows[1]['far_unit_signal_T_minus_C']}


if __name__ == '__main__':
    result = verify()
    Path(__file__).with_name('focused_tail_priority_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
