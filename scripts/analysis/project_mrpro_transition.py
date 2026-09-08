"""One user-specified unconstrained projection; no fitting to task scores.

x is the normalized rotary-slot coordinate inside the actual MrPro transition.
Effective exponents are recovered from deployed float32 frequencies, because the
pre-rounding FullLagP2 audit is not present in this checkout. No clipping, sorting,
coefficient adjustment, second candidate, or model execution occurs here.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct


def tensor_sha(values):
    return hashlib.sha256(struct.pack('<' + 'f' * len(values), *values)).hexdigest()


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def read_json(path):
    return json.loads(Path(path).read_text())


def exponent(native, frequency, scale):
    return [math.log(n / f) / math.log(scale) for n, f in zip(native, frequency)]


def basis(size, low, high):
    x = [max(0., min(1., (j-low)/(high-low))) for j in range(size)]
    return x, [t*(1-t)*(2*t-1) for t in x]


def project(residual, psi):
    denominator = math.fsum(v*v for v in psi)
    if denominator == 0:
        raise ValueError('nonzero projection basis required')
    return math.fsum(v*d for v, d in zip(psi, residual)) / denominator


def apply(native, mr, gain, scale, low, high, coefficient, p2=None):
    x, psi = basis(len(native), low, high)
    m_mr = exponent(native, mr, scale)
    m_p2 = exponent(native, p2, scale) if p2 is not None else None
    intended = [m + coefficient*v for m, v in zip(m_mr, psi)]
    values = [fp32(w*scale**(-coefficient*v)) if low < j < high else w
              for j, (w, v) in enumerate(zip(mr, psi))]
    realized = exponent(native, values, scale)
    changes = [j for j, (a, b) in enumerate(zip(mr, values)) if a != b]
    rows = []
    for j in range(len(native)):
        row = dict(slot_zero_based=j, slot_one_based=j+1, x=x[j], psi=psi[j],
                   native_frequency=native[j], mr_frequency=mr[j],
                   candidate_frequency=values[j], mr_exponent=m_mr[j],
                   candidate_exponent_intended=intended[j],
                   candidate_exponent_realized=realized[j],
                   exponent_delta_to_mr=intended[j]-m_mr[j],
                   frequency_ratio_to_mr=values[j]/mr[j],
                   frequency_ratio_to_native=values[j]/native[j])
        if p2 is not None:
            row.update(p2_frequency=p2[j], p2_exponent=m_p2[j],
                       p2_exponent_delta_to_mr=m_p2[j]-m_mr[j],
                       candidate_exponent_delta_to_p2=intended[j]-m_p2[j])
        rows.append(row)
    checks = dict(
        finite_positive=all(math.isfinite(w) and w > 0 for w in values),
        strictly_decreasing=all(a > b for a, b in zip(values, values[1:])),
        nondecreasing_pairs=[[j, j+1] for j in range(len(values)-1)
                             if values[j] <= values[j+1]],
        fast_band_bitwise_equal=all(values[j] == mr[j] for j in range(low+1)),
        slow_band_bitwise_equal=all(values[j] == mr[j] for j in range(high, len(values))),
        support_endpoints_bitwise_equal=(values[0], values[-1]) == (mr[0], mr[-1]),
        exponent_min=min(intended), exponent_max=max(intended),
        exponent_outside_unit_interval=[j for j, m in enumerate(intended) if not 0 <= m <= 1],
        sum_exponent_delta_intended=math.fsum(a-b for a, b in zip(intended, m_mr)),
        sum_exponent_delta_realized=math.fsum(a-b for a, b in zip(realized, m_mr)),
        telescoped_exponent_span=intended[-1]-intended[0],
        max_frequency_ratio_to_native=max(v/n for v, n in zip(values, native)),
        maximum_float32_exponent_rounding=max(abs(a-b) for a, b in zip(realized, intended)),
        changed_slots=changes,
        gain_preserved=gain,
    )
    return dict(low=low, high=high, scale=scale, gain=gain, coefficient=coefficient,
                values_float32=values, exponents_intended=intended,
                exponents_realized=realized, tensor_sha256=tensor_sha(values),
                checks=checks, rows=rows)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    owner = args.root/'docs/research'
    source_path = owner/'ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json'
    native_path = owner/'ROPE_RECOVERED_QWEN_P2_20260907.json'
    olmo_path = owner/'ROPE_OLMO_MRPRO_SOURCE_20260908.json'
    source, history, olmo = map(read_json, (source_path, native_path, olmo_path))
    native = history['native_float32']
    mr, p2 = [source['tables'][name]['values_float32'] for name in ('MrPro', 'FullLagP2')]
    scale = source['scale']
    if tensor_sha(native) != source['native_tensor_sha256']:
        raise ValueError('actual Native tensor identity differs')
    for name in ('MrPro', 'FullLagP2'):
        if tensor_sha(source['tables'][name]['values_float32']) != source['tables'][name]['tensor_sha256']:
            raise ValueError('actual deployed tensor identity differs: '+name)
    turns = [w*source['native_length']/(2*math.pi) for w in native]
    low = max(j for j, v in enumerate(turns) if v > 32)
    high = min(j for j, v in enumerate(turns) if v < 1)
    _, psi = basis(len(native), low, high)
    mr_m, p2_m = exponent(native, mr, scale), exponent(native, p2, scale)
    residual = [a-b for a, b in zip(p2_m, mr_m)]
    coefficient = project(residual, psi)
    # Independent exact accumulation of the floating input numbers.
    f = Fraction.from_float
    rational = sum((f(v)*f(d) for v, d in zip(psi, residual)), Fraction()) / sum(
        (f(v)*f(v) for v in psi), Fraction())
    if abs(float(rational)-coefficient) > 1e-12:
        raise AssertionError('projection disagrees with independent rational sum')
    qwen = apply(native, mr, source['tables']['MrPro']['gain'], scale, low, high,
                 coefficient, p2=p2)
    on, om = [olmo['tables'][key]['values_float32'] for key in ('Native', 'MrPro')]
    entry = olmo['tables']['MrPro']
    target = apply(on, om, entry['gain'], olmo['scale'], entry['construction']['low'],
                   entry['construction']['high'], coefficient)
    interior = list(range(low+1, high))
    energy = math.fsum(residual[j]**2 for j in interior)
    remaining = math.fsum((residual[j]-coefficient*psi[j])**2 for j in interior)
    wrong_direction = [j for j in interior if residual[j]*coefficient*psi[j] < 0]
    report = dict(
        status='C1_FROZEN_UNADJUSTED_REVIEW_NOT_RECOMMENDED_FOR_GPU',
        method='MrPro plus a_star*x*(1-x)*(2*x-1)',
        source_coordinate='x=(zero_based_slot-low)/(high-low), clamped to [0,1]',
        exponent_identity='effective exponents -ln(actual_float32_frequency/native_float32)/ln(scale); not pre-rounding construction exponents',
        projection='unweighted ordinary least squares, zero intercept, actual middle slots; no clipping, sorting, or coefficient adjustment',
        a_star=coefficient, independent_fraction_a_star=float(rational),
        source_model=source['model_repo'], source_revision=source['model_revision'],
        source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (source_path, native_path, olmo_path)},
        source_tensor_sha256={name:source['tables'][name]['tensor_sha256'] for name in ('MrPro','FullLagP2')},
        source_qwen=qwen,
        target_model=olmo['model_id'], target_revision=olmo['revision'],
        target_actual_parameters=olmo['actual_parameters'],
        target_native_length=olmo['native_length'], target_base=olmo['base'],
        target_olmo=target,
        transfer='Only a_star is transferred. OLMo uses its own actual Native/MrPro arrays and 14..32 transition. No OLMo FullLagP2 result or identity is implied.',
        review=dict(
            middle_residual_sum=math.fsum(residual[j] for j in interior),
            middle_residual_energy=energy, projection_residual_energy=remaining,
            fraction_residual_energy_captured=1-remaining/energy,
            opposing_source_direction_slots=wrong_direction,
            main_opposition_slots=[j for j in wrong_direction if abs(residual[j]) > .5],
            decision='Retain the exact requested array; do not auto-enqueue it. It reverses the P2 movement at source slots 30/31, raises some middle frequencies above Native, and removes a component that cannot represent the positive mean source displacement. No task-gain theorem follows from least squares.',
            failure_probability='not identifiable from arrays; no numerical probability assigned',
            scope='Specific candidate review. Negative exponents are legal rotations and are not a class-wide impossibility result.',
        ),
        execution='CPU scalar arithmetic and source readback only; no model inference or GPU',
    )
    out = owner/'ROPE_MRPRO_TRANSITION_PROJECTION_20260908.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    write_csv(owner/'ROPE_MRPRO_TRANSITION_QWEN_SLOTS_20260908.csv', qwen['rows'])
    write_csv(owner/'ROPE_MRPRO_TRANSITION_OLMO_SLOTS_20260908.csv', target['rows'])
    print(json.dumps({key:report[key] for key in ('status','a_star','review')}, indent=2))
    print(json.dumps({'qwen_checks':qwen['checks'],'olmo_checks':target['checks']}, indent=2))


if __name__ == '__main__':
    main()
