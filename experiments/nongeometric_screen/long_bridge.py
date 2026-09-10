"""Paired long-distance phase shifts of the middle-to-low transition band."""
import argparse
import json
from pathlib import Path

import numpy as np


def prepare(root, tables_path):
    root = Path(root)
    tables = json.loads(Path(tables_path).read_text())
    base = np.asarray(tables['MrPro']['values_float32'], dtype=np.float64)
    native_length, target_length = 32768, 131072
    periods = 2 * np.pi / base
    slots = np.flatnonzero((periods >= native_length) & (periods <= target_length))
    if not len(slots):
        raise ValueError('No clock periods in the requested long transition band')
    # An O(1) long-distance phase intervention, with a 4x smaller short effect.
    # The opposite sign is a magnitude-matched mechanism control, not a search.
    step = 1.0 / target_length
    proposals = {}
    for suffix, name, direction in [('2', 'LongBridgeSlower', -1), ('3', 'LongBridgeFaster', 1)]:
        values = base.copy()
        values[slots] += direction * step
        values = values.astype(np.float32)
        if not (np.isfinite(values).all() and (values > 0).all() and (np.diff(values) < 0).all()):
            raise ValueError('Long bridge intervention breaks frequency order')
        outside = np.ones(len(base), dtype=bool)
        outside[slots] = False
        if not np.array_equal(values[outside], base.astype(np.float32)[outside]):
            raise ValueError('Frequency outside the declared band changed')
        table = dict(values_float32=values.tolist(), gain=tables['MrPro']['gain'])
        proposals[name] = dict(table=table, periods_tokens=(2*np.pi/values[slots]).tolist(),
                               phase_shift_at_native=(native_length*(values[slots]-base[slots])).tolist(),
                               phase_shift_at_target=(target_length*(values[slots]-base[slots])).tolist())
        job = dict(id=name, spec=dict(operator='static', table=table), panel='full',
                   nll_docs=16, nll_lengths=[8192, 16384, 32768])
        path = root / 'queue' / f'044{suffix}_{name}.json'
        if path.exists() and json.loads(path.read_text()) != job:
            raise ValueError('Cannot replace an existing long bridge contract')
        path.write_text(json.dumps(job, indent=2)+'\n')
    receipt = dict(
        native_length=native_length, target_length=target_length,
        selection_rule='MrPro clock period between native and target context length, inclusive',
        slots_zero_based=slots.tolist(), baseline_periods_tokens=periods[slots].tolist(),
        absolute_frequency_step=step, proposals=proposals,
        scope='Frozen Qwen3B diagnostic chosen from physical distance scales, without fitting holdout outcomes. One-radian amplitude is an explicit exploratory scale, not an optimum. A fixed-state phase bound is not a bound on end-to-end behavior. Both directions retain all other frequencies and attention gain.')
    output = root / 'planned_controls' / 'long_bridge.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'proposals'}, indent=2))
    for name, item in proposals.items():
        print(name, item['periods_tokens'], item['phase_shift_at_target'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--tables', required=True)
    args = parser.parse_args()
    prepare(args.root, args.tables)
