"""One carrier projection into each non-wrapping Native causal phase sector.

Core experiment 3 is distinct from the author's original signed Carrier trial.
Keep the stronger published MrPro substrate outside the original carrier band.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.experiments.scale_transport.carrier import tensor_sha


def construct(original, native):
    native = np.asarray(native, dtype=np.float64)
    scale, length = original['scale'], original['native_length']
    band = np.flatnonzero(native*length <= np.pi/2)
    if band.tolist() != list(range(47, 64)) or scale != 4 or length != 32768:
        raise ValueError('this experiment retains the declared Qwen band and scale')
    reference = np.asarray(original['tables']['MrPro']['values_float32'], dtype=np.float32)
    if tensor_sha(reference) != original['tables']['MrPro']['tensor_sha256']:
        raise ValueError('MrPro substrate identity')
    cap = float(native[band].min())
    # D*c^2 - 2*N*c + A is convex. Each causal non-wrapping slot
    # stays within [0, omega_j*L0] at all delta<=s*L0 iff 0<=c<=min omega.
    carrier = float(np.clip(original['c0'], 0, cap))
    values = reference.copy()
    values[band] = ((native[band]-carrier)/scale).astype(np.float32)
    if not np.isfinite(values).all() or not np.all(np.diff(values)<0):
        raise ValueError('slot ordering or finite-value failure')
    assert np.array_equal(values[:47], reference[:47])
    assert np.all(values[band] >= 0)
    assert np.all(values[band].astype(float)*scale <= native[band]*(1+1e-7))
    assert np.allclose(np.diff(values[band].astype(float)), np.diff(native[band])/scale, rtol=2e-6, atol=1e-13)
    # Independent distance endpoints and analytic operator norm, not a quality metric.
    for distance in (0, 1, 4096, 32768, 65536, 131072):
        phase = values[band].astype(float)*distance
        assert np.all(phase >= 0) and np.all(phase <= native[band]*length+1e-8)
    integral = original['integral']
    objective = lambda c: integral['second_moment']-2*c*integral['numerator']+c*c*integral['denominator']
    if original['c0'] >= cap:
        assert carrier == cap and values[-1] == 0
        assert 2*integral['denominator']*cap-2*integral['numerator'] <= 0
        assert objective(carrier) <= objective(0)
    return dict(status='FROZEN_NEW_EXPERIMENT_3_NO_CAPABILITY_CLAIM',
        method='Native-sector-constrained shared carrier on MrPro',
        experiment_index=3, model_revision=original['model_revision'],
        native_length=length, scale=scale, band_start=int(band[0]),
        c0=original['c0'], c=carrier, cap=cap,
        original_signed_c=original['c'], original_signed_tensor_sha256=original['tables']['Carrier']['tensor_sha256'],
        source_means_sha256=original['source_means_sha256'],
        source_manifest_sha256=original['source_manifest_sha256'],
        changes='Project the same frozen Native-background quadratic into per-slot Native causal phase sectors; preserve MrPro exactly outside j47..63. No answer-dependent fit or parameter sweep.',
        supported='All modified non-wrapping slots remain in their own Native causal phase interval at every target distance; signed pairwise frequency gaps remain Native/4.',
        unsupported='This is not a no-harm theorem or a proof that sector violation caused every original failure. It does not isolate the original YaRN-to-Mr substrate change.',
        objective_before=objective(0), objective_after=objective(carrier),
        max_added_phase_native=carrier*length/scale,
        max_added_phase_target=carrier*length,
        maximum_plane_operator_norm_change_at_target=2*float(np.sin(carrier*length/2)),
        negative_slots=[], tables={'MrPro': original['tables']['MrPro'],
            'Carrier': {'values_float32': values.tolist(), 'tensor_sha256': tensor_sha(values),
                        'gain': original['gain']}},
        self_check={'status': 'NUMPY_NATIVE_SECTOR_AND_UNIQUE_CONVEX_PROJECTION_CHECKED',
            'source_background_reestimated': False, 'generated_answers_used': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--means', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    original = json.loads(args.original.read_text())
    if hashlib.sha256(args.means.read_bytes()).hexdigest() != original['source_means_sha256']:
        raise ValueError('same frozen Native construction means required')
    data = np.load(args.means, allow_pickle=False)
    result = construct(original, data['native'])
    result['source_candidate_file_sha256'] = hashlib.sha256(args.original.read_bytes()).hexdigest()
    result['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'c0', 'c', 'cap', 'max_added_phase_target')}, indent=2))
