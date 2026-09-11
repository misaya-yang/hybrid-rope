"""CPU-only readout of saved signed-frequency and gain controls.

The mirrored frequency interventions are compared on common unique prompts.
Incomplete holdout rows remain explicitly incomplete, not a completed trial.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from scripts.experiments.olmo_fast_screen.ruler_bench import score


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    keyed = {r['row_id']: r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError(f'duplicate row IDs: {path}')
    return keyed


def compare(a, b, identities, cap):
    common = sorted(k for k in a.keys() & b.keys()
                    if identities[k]['length_cap'] == cap)
    unique = {}
    for k in common:
        ident = identities[k]
        for arm in (a, b):
            if arm[k]['task'] != ident['task'] or arm[k]['length_cap'] != cap:
                raise ValueError('task/length identity mismatch')
            if abs(score(ident, arm[k]['output_text']) - arm[k]['correct']) > 1e-12:
                raise ValueError('saved score disagrees with original scorer')
        digest = ident['prompt_sha256']
        if digest in unique:
            prior = unique[digest]
            if any(arm[k]['correct'] != arm[prior]['correct'] for arm in (a, b)):
                raise ValueError('duplicate prompt with inconsistent scores')
        else:
            unique[digest] = k
    keys = list(unique.values())
    cells, boot = {}, []
    rng = np.random.default_rng(20260911)
    for task in sorted({identities[k]['task'] for k in keys}):
        kk = [k for k in keys if identities[k]['task'] == task]
        d = np.array([a[k]['correct'] - b[k]['correct'] for k in kk])
        cells[task] = dict(n=len(d), delta_pp=float(d.mean()*100),
                           wins=int(sum(d > 0)), losses=int(sum(d < 0)),
                           ties=int(sum(d == 0)))
        boot.append(rng.choice(d, (20000, len(d))).mean(axis=1))
    means = np.mean(boot, axis=0)
    return dict(common_rows=len(common), unique_prompts=len(keys), tasks=cells,
                delta_pp=float(np.mean([c['delta_pp'] for c in cells.values()])),
                descriptive_bootstrap95_pp=(100*np.quantile(means, [.025, .975])).tolist())


def uuid_outputs(a, b, identities):
    keys = [k for k in a.keys() & b.keys() if identities[k]['task'] == 'niah_single_3']
    rx = re.compile(r'(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-'
                    r'[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f])', re.I)
    out = {'n': len(keys)}
    for name, arm in [('slower', a), ('faster', b)]:
        refs = {k: identities[k]['references'][0].lower() for k in keys}
        if any(not rx.fullmatch(v) for v in refs.values()):
            raise ValueError('UUID diagnostic requires canonical UUID references')
        found = {k: rx.findall(arm[k]['output_text'].lower()) for k in keys}
        out[name] = dict(
            correct_prefix_counts={str(n): sum(refs[k][:n] in arm[k]['output_text'].lower()
                                              for k in keys) for n in (4, 8, 13, 18, 23, 36)},
            first_complete_uuid_correct=sum(bool(found[k]) and found[k][0] == refs[k]
                                            for k in keys))
    wins = [k for k in keys if a[k]['correct'] > b[k]['correct']]
    out['slower_wins'] = len(wins)
    out['faster_correct_prefix8_in_slower_wins'] = sum(
        identities[k]['references'][0][:8].lower() in b[k]['output_text'].lower() for k in wins)
    out['scope'] = 'Saved output evidence, not localization of an internal attention mechanism.'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    identity_path = args.root/'prompt_identities.json'
    identities = json.loads(identity_path.read_text())
    result = dict(scope='Saved-output analysis only; no new model runs.',
                  comparisons={}, source_sha256={})
    for dirname, panel in [('olmo_lb', 'selection'), ('olmo_lb_h', 'holdout')]:
        a_path = args.root/dirname/'nu_m6p104em05.jsonl'
        b_path = args.root/dirname/'nu_p6p104em05.jsonl'
        a, b = read(a_path), read(b_path)
        result['comparisons'][dirname] = dict(
            complete=(a.keys() == b.keys() == identities[panel].keys()),
            counts=dict(slower=len(a), faster=len(b), expected=len(identities[panel])),
            by_length={str(cap): compare(a, b, identities[panel], cap)
                       for cap in sorted({v['length_cap'] for v in a.values()})})
        for p in (a_path, b_path):
            result['source_sha256'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        if panel == 'selection':
            result['uuid_diagnostic'] = uuid_outputs(a, b, identities[panel])
    result['gain_controls'] = {}
    for p in sorted((args.root/'olmo_gsweep').glob('*.jsonl')):
        rows = read(p)
        for k, r in rows.items():
            if abs(score(identities['selection'][k], r['output_text'])-r['correct']) > 1e-12:
                raise ValueError('gain score mismatch')
        result['gain_controls'][p.stem] = dict(
            n=len(rows), complete=rows.keys() == identities['selection'].keys(),
            observed_mean=float(np.mean([r['correct'] for r in rows.values()])))
    result['limitations'] = [
        'Intervals are descriptive paired within-task bootstrap intervals, not selection-adjusted.',
        'Legacy outputs omit prompt hashes; pairing uses source-panel metadata and row IDs.',
        'The partially completed holdout does not estimate the full planned task macro.',
        'A signed contrast does not cancel candidate selection, task dependence, or model dependence.',
        'OLMo signed effects cannot prove an improvement on the healthy Qwen MrRoPE baseline.']
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
