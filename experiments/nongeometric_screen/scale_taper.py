"""Distance-scale taper of the measured BM allocation toward MrPro."""
import argparse
import json
from pathlib import Path

import numpy as np


def prepare(root, tables_path):
    root = Path(root)
    tables = json.loads(Path(tables_path).read_text())
    reference = np.asarray(tables['MrPro']['values_float32'], dtype=np.float64)
    bm = np.asarray(tables['MrProBM']['values_float32'], dtype=np.float64)
    native_length, scale = 32768, 4
    periods = 2 * np.pi / reference
    weight = np.clip(np.log(native_length / periods) / np.log(scale), 0, 1)
    values = np.exp(np.log(reference) + weight * np.log(bm / reference)).astype(np.float32)
    # Preserve exact reference values at both endpoints of the interpolation.
    values[weight == 0] = reference[weight == 0].astype(np.float32)
    values[weight == 1] = bm[weight == 1].astype(np.float32)
    if not (np.isfinite(values).all() and (values > 0).all() and (np.diff(values) < 0).all()):
        raise ValueError('Scale taper is not a positive ordered frequency table')
    if np.array_equal(values, reference.astype(np.float32)) or np.array_equal(values, bm.astype(np.float32)):
        raise ValueError('Scale taper duplicates an existing reference')
    name = 'BM_ScaleTaper'
    table = dict(values_float32=values.tolist(), gain=tables['MrPro']['gain'])
    job = dict(id=name, spec=dict(operator='static', table=table), panel='full',
               nll_docs=16, nll_lengths=[8192, 16384, 32768])
    path = root / 'queue' / ('044d_' + name + '.json')
    if path.exists() and json.loads(path.read_text()) != job:
        raise ValueError('Cannot replace an existing scale-taper contract')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2) + '\n')
    receipt = dict(
        native_length=native_length, scale=scale,
        rule='T_j=2*pi/Mr_j; w_j=clip(log(W/T_j)/log(S),0,1); nu_j=Mr_j*(BM_j/Mr_j)**w_j',
        rationale='Retain measured BM changes at periods <=W/S; fade them over the log-scale interval [W/S,W]; retain MrPro at periods >=W. This interval is an explicit design hypothesis, not an optimized or proven threshold.',
        weights=weight.tolist(), reference_periods=periods.tolist(), table=table,
        changed_slots_zero_based=np.flatnonzero(values != reference.astype(np.float32)).tolist(),
        preserved_long_slots_zero_based=np.flatnonzero(weight == 0).tolist(),
        comparisons=['MrPro official gain .1', 'Existing BM official gain .1'],
        scope='One new whole-table rule on the current Qwen3B development panel, without fitting new-input outcomes. Preserving slow clocks does not guarantee long behavior because faster slots and upstream states also affect distant interactions. No fixed-budget or causal band-isolation claim.')
    output = root / 'planned_controls' / 'scale_taper.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(name=name, changed_slots=receipt['changed_slots_zero_based'],
                         preserved_long_slots=receipt['preserved_long_slots_zero_based'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--tables', required=True)
    args = parser.parse_args()
    prepare(args.root, args.tables)
