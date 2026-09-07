"""Recompute existing table effects and two exact counterexamples; no model runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def response_counterexample():
    # Two keys, one rotary pair, values (1, 0), equal source logits.
    # These logits are realizable as C*cos(nu*d)+S*sin(nu*d).
    def output(nu, second_slope):
        first = 8 * math.sin(nu - 0.2)
        second = second_slope * math.sin(2 * (nu - 0.2))
        return 1 / (1 + math.exp(second - first))
    rows = []
    for slope in (4, -4):
        jac = [2.0, -slope / 4]
        coherent = jac[0] + 2 * jac[1]
        eps = 1e-6
        measured = (output(0.2 + eps, slope) - output(0.2 - eps, slope)) / (2 * eps)
        if not math.isclose(measured, coherent, abs_tol=1e-7):
            raise ValueError("exact derivative differs from direct attention")
        rows.append(dict(phase_derivatives=jac, independent_phase_energy=[x*x for x in jac],
                         frequency_derivative=coherent, finite_difference=measured))
    return rows


def diagnose(proposal_path):
    raw = proposal_path.read_bytes()
    proposal = json.loads(raw)
    values = proposal['proposal']
    if len(values) != 64 or not all(x > 0 and math.isfinite(x) for x in values):
        raise ValueError('requires the positive 64-slot Qwen proposal')
    bands = {}
    for name, indices in [('middle', range(24, 40)), ('tail', range(40, 64))]:
        ratios = []
        for j in indices:
            t = min(17, max(0, j - 23))
            mr = 1e6 ** (-j/64) / 4 ** (t*(t+1)/306)
            ratios.append(values[j]/mr)
        bands[name] = dict(min_ratio_to_mr=min(ratios), max_ratio_to_mr=max(ratios),
                           raw_beta_at_one=[j for j in indices if proposal['slots'][j]['late']['beta'] == 1.])
    # Hard-clipping one frequency leaves its content dot product; deleting the
    # spectral atom does not. At zero distance every rotary block is identity.
    content, sine, omega, distance = 1., .3, .01, 0.
    frequency_clip = content*math.cos(0.*distance)+sine*math.sin(0.*distance)
    amplitude_clip = 0.*(content*math.cos(omega*distance)+sine*math.sin(omega*distance))
    return dict(status='EXISTING_RECEIPT_AND_CPU_ALGEBRA_ONLY',
                source_proposal_sha256=hashlib.sha256(raw).hexdigest(),
                assumptions='Qwen K64 base1e6 s4 Mr boundaries23/40; analytical Native grid, not runtime tensor identity',
                bands=bands, equal_adjacent_slots=[[j,j+1] for j in range(63) if values[j] == values[j+1]],
                distinct_frequencies=len(set(values)),
                high_middle_log_gap=dict(proposal=math.log(values[23]/values[24]),
                                         mr=math.log(1e6)/64+math.log(4)*2/306),
                same_energy_different_response=response_counterexample(),
                clipping_counterexample=dict(distance=distance, frequency_clipped_score=frequency_clip,
                                             amplitude_clipped_score=amplitude_clip),
                limits='No candidate generation, parameter fitting, model-quality prediction or new GPU evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.proposal)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['status','distinct_frequencies','high_middle_log_gap','same_energy_different_response']}, indent=2))


if __name__ == '__main__':
    main()
