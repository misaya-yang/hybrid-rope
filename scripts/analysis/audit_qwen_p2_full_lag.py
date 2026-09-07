"""Audit one historical Qwen p2 rule and its full-lag numerical correction.

No model loading, candidate curve list, parameter scan, or capability inference.
Only lag compression changes; rcond=1e-10, p=2 and normalization are fixed.
"""
import argparse
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np


def digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def design_for(frequencies, support, weight):
    theta = support[:, None]*frequencies[None, :]
    design = np.empty((len(support), 2*len(frequencies)))
    design[:, 0::2], design[:, 1::2] = np.cos(theta), np.sin(theta)
    return design*np.sqrt(weight)[:, None]


def normalized_m(u):
    u = np.clip(u, 0, 1)
    return (1-(u-u.min())/(u.max()-u.min()))**2


def residual_fraction(design, pair):
    target = design[:, [2*pair, 2*pair+1]]
    others = np.delete(design, [2*pair, 2*pair+1], axis=1)
    coefficient, _, rank, _ = np.linalg.lstsq(others, target, rcond=1e-10)
    # Avoid local BLAS status warnings observed with @; this is the same sum.
    residual = target-np.einsum('ij,jk->ik', others, coefficient, optimize=False)
    value = float(np.sum(residual**2)/np.sum(target**2))
    assert math.isfinite(value)
    return min(1., max(0., value)), int(rank)


def two_pair_uniqueness(first, second, support, weight):
    target = [np.cos(first*support), np.sin(first*support)]
    other = [np.cos(second*support), np.sin(second*support)]
    g00 = float(np.sum(weight*other[0]*other[0]))
    g01 = float(np.sum(weight*other[0]*other[1]))
    g11 = float(np.sum(weight*other[1]*other[1]))
    det = g00*g11-g01*g01
    inverse = [[g11/det, -g01/det], [-g01/det, g00/det]]
    residual = marginal = 0.
    for column in target:
        cross = [float(np.sum(weight*x*column)) for x in other]
        total = float(np.sum(weight*column*column))
        residual += total-sum(cross[i]*inverse[i][j]*cross[j] for i in range(2) for j in range(2))
        marginal += total
    return residual/marginal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
        default=Path('docs/research/ROPE_RECOVERED_QWEN_P2_20260907.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    source = args.input
    old = json.loads(source.read_text())
    native, movement, deployed = [np.asarray(old[key], dtype=float) for key in
        ('native_float32', 'legacy_m_float64', 'log_s4_float32')]
    identity = old['source_identity']
    assert digest(native, '<f4') == identity['native_omega_sha256_float32']
    assert digest(movement, '<f8') == identity['movement_sha256_float64']
    assert digest(deployed, '<f4') == identity['log_s4_tensor_sha256_float32']
    assert digest(native*4**(-movement), '<f4') == digest(deployed, '<f4')
    length = 32768
    support = np.arange(length, dtype=float)
    weight = length-support
    chunks, lags = weight.reshape(-1, 16), support.reshape(-1, 16)
    mass = chunks.sum(1)
    centroids = np.round((lags*chunks).sum(1)/mass)
    weights = {'sampled_2048': mass/mass.sum(), 'full_32768': weight/weight.sum()}
    supports = {'sampled_2048': centroids, 'full_32768': support}
    result = dict(status='NUMERICAL_ORIGIN_AUDIT_NO_MODEL_RESULT', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original_identity_verified=True, original_gain=1+.074*math.log(4),
        fixed_rule=dict(native_length=length, rcond=1e-10, p=2, normalization='min-max residual fraction'),
        centroid_exceptions_to_16b_plus_7=[(int(j), int(centroids[j]-16*j)) for j in
            np.flatnonzero(centroids != np.arange(2048)*16+7)], alias_pairs=[], constructions={})
    for i, j, sign, harmonic in [(1, 18, -1, 2), (0, 8, 1, 3)]:
        compressed_u = two_pair_uniqueness(native[i], native[j], centroids, weights['sampled_2048'])
        full_u = two_pair_uniqueness(native[i], native[j], support, weights['full_32768'])
        result['alias_pairs'].append(dict(pair=[i, j], relation='difference' if sign == -1 else 'sum',
            alias_residual=float(native[i]+sign*native[j]-harmonic*2*math.pi/16),
            pair_only_sampled_m=(1-compressed_u)**2, historical_m=float(movement[i]),
            pair_only_full_lag_m=(1-full_u)**2))
    for name in supports:
        design = design_for(native, supports[name], weights[name])
        reduced = np.linalg.qr(design, mode='r')
        observations = [residual_fraction(reduced, pair) for pair in range(64)]
        u = np.asarray([x[0] for x in observations]); m = normalized_m(u)
        result['constructions'][name] = dict(u=u.tolist(), m=m.tolist(), ranks=[x[1] for x in observations],
            maximum_m_difference_to_historical=float(np.max(np.abs(m-movement))))
        if name == 'full_32768':
            result['direct_full_design_checks'] = []
            for pair in (1, 29, 30, 31):
                direct_u, rank = residual_fraction(design, pair)
                result['direct_full_design_checks'].append(dict(pair=pair, direct_u=direct_u, qr_u=float(u[pair]),
                    absolute_difference=abs(direct_u-u[pair]), rank=rank))
    full = np.asarray(result['constructions']['full_32768']['m'])
    table = (native*4**(-full)).astype('<f4')
    minorant = np.minimum.accumulate(movement[::-1])[::-1]
    result['comparison'] = dict(middle_m_max_difference=float(np.max(np.abs(full[24:40]-movement[24:40]))),
        middle_frequency_max_relative_difference=float(np.max(np.abs(table[24:40]/deployed[24:40]-1))),
        full_table_values_float32=table.tolist(), full_table_sha256=digest(table, '<f4'),
        crossings=np.flatnonzero(np.diff(table) >= 0).tolist(),
        mathematical_minorant_m_max_difference=float(np.max(np.abs(full-minorant))),
        mathematical_minorant_frequency_max_relative_difference=float(np.max(np.abs(native*4**(-full)/(native*4**(-minorant))-1))),
        mathematical_minorant_sha256=digest(native*4**(-minorant), '<f4'),
        historical_minorant_asset_hash_verified=False,
        middle_rows=[dict(j=j, native_cycles=float(native[j]*length/(2*math.pi)), old_m=float(movement[j]), new_m=float(full[j])) for j in range(24, 40)])
    result['seconds'] = time.monotonic()-start
    output = args.out
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), seconds=result['seconds'], middle_m_max_difference=result['comparison']['middle_m_max_difference'],
        middle_frequency_max_relative_difference=result['comparison']['middle_frequency_max_relative_difference'],
        sampled_historical_m_max_difference=result['constructions']['sampled_2048']['maximum_m_difference_to_historical']), indent=2))


if __name__ == '__main__':
    main()
