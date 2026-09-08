"""Freeze a parameter-free slotwise amplitude experiment on the winning BM table."""
import argparse
import json
import math
from pathlib import Path
import shutil

from .prepare import sha_file, write


def allocations(native, frequencies, scale, gain):
    if scale <= 1 or len(native) != len(frequencies):
        raise ValueError('matching frequencies and scale > 1 required')
    exponent = [-math.log(w/n)/math.log(scale) for n,w in zip(native, frequencies)]
    if min(exponent) < -1e-6 or max(exponent) > 1+1e-6:
        raise ValueError('allocation requires exponents within [0,1]')
    # Clamp only FP32 endpoint rounding. No task outputs or fitted coefficients.
    exponent = [max(0.,min(1.,x)) for x in exponent]
    amplitudes = [math.sqrt(1+(gain*gain-1)*x) for x in exponent]
    rms = math.sqrt(sum(a*a for a in amplitudes)/len(amplitudes))
    return amplitudes, rms, exponent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    old, out = args.prepared.resolve(), args.out.resolve()
    manifest = json.loads((old/'manifest.json').read_text())
    for name, expected in manifest['prepared_files'].items():
        if sha_file(old/name) != expected:
            raise ValueError('source input drift: '+name)
    tables = json.loads((old/'tables.json').read_text())
    bm = tables['MrProBM']
    gains, rms, exponent = allocations(tables['Native']['values_float32'], bm['values_float32'], manifest['static_scale'], bm['gain'])
    tables['BMSelectiveGain'] = dict(bm, gain_by_slot=gains)
    tables['BMUniformMatchedGain'] = dict(bm, gain=rms)
    queue = dict(max_candidates=10, ordered_candidates=[
        dict(id='BMSelectiveGain', eligible=True, review_status='REVIEWED_FOR_GPU',
             definition='a_j=sqrt(1+(a_Mr^2-1)*m_j), m_j=-log(omega_BM/omega_native)/log(S); same BM frequency table',
             hypothesis='Preserve native high-frequency amplitude while retaining enhancement for compressed pairs; may improve exact content binding.',
             failure_rule='No 16K gain over BM or a 4K loss does not promote. Uniform matched control must be compared before attributing gain to slot selectivity.'),
        dict(id='BMUniformMatchedGain', eligible=True, review_status='REVIEWED_FOR_GPU',
             definition='Uniform a=sqrt(mean_j a_selective_j^2); same BM frequency table',
             hypothesis='Control whether any selective-gain improvement is explained by a lower average logit multiplier.',
             failure_rule='Control only; no selective-mechanism claim when it matches or exceeds selective gain.')])
    out.mkdir(parents=True, exist_ok=False)
    for name in manifest['prepared_files']:
        shutil.copyfile(old/name, out/name)
    write(out/'tables.json', tables)
    write(out/'queue.json', queue)
    root = Path(__file__).resolve().parents[3]
    dependencies = set(manifest['code_files']) | {
        'scripts/experiments/olmo_fast_screen/runtime.py',
        'scripts/experiments/olmo_fast_screen/prepare_gain.py'}
    manifest.update(reference_arm='MrProBM', complete_candidate_queue=True,
        source_manifest_sha256=sha_file(old/'manifest.json'),
        status='PREPARED_GAIN_FOLLOWUP_GPU_NOT_RUN',
        intervention=dict(kind='slotwise rotary amplitude', exponents=exponent, uniform_matched_gain=rms,
            scalar_bm_gain=bm['gain'], theory_scope='Per-pair bilinear coefficient only, not preservation of full-network function or proof of score gain.'),
        code_files={name:sha_file(root/name) for name in sorted(dependencies)})
    manifest['prepared_files'] = {name:sha_file(out/name) for name in manifest['prepared_files']}
    write(out/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],rows=manifest['screen_rows'],reference=manifest['reference_arm'],uniform_gain=rms)))


if __name__ == '__main__':
    main()
