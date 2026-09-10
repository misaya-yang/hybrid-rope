"""Transfer a fixed log-frequency gap budget from fast to slower clocks.

This tests the resource-allocation idea behind EVQ, not the literal Cosh
family and not a guarantee about frozen-model task behavior.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def prepare(root, tables_path):
    root = Path(root)
    tables = json.loads(Path(tables_path).read_text())
    reference = np.asarray(tables['MrPro']['values_float32'], dtype=np.float64)
    native = np.asarray(tables['Native']['values_float32'], dtype=np.float64)
    first_changed = np.flatnonzero(reference != native)
    if not len(first_changed) or first_changed[0] < 3:
        raise ValueError('Need an unchanged native high-frequency prefix')
    high_end = int(first_changed[0]) - 1
    donors = np.arange(high_end)
    gaps = np.log(reference[:-1] / reference[1:])
    budget = float(gaps[donors].mean())
    native_length, scale = 32768, 4
    periods = 2 * np.pi / reference
    gap_period = np.sqrt(periods[:-1] * periods[1:])
    receipt = dict(
        native_length=native_length, scale=scale,
        rule='g_j=log(nu_j/nu_{j+1}); remove one mean native high-band gap in total, uniformly across high-band gaps; distribute the same total uniformly among recipient gaps selected by geometric-mean period.',
        donor_gaps_zero_based=donors.tolist(), donated_log_range=budget,
        donated_fraction_of_high_log_range=float(budget / gaps[donors].sum()),
        scope='One explicit budget quantum, not an optimized magnitude. Same donor gaps, gain, endpoints, and total log range in both arms. Actual capability must be evaluated from prefill; more separated slow clocks need not be better for every learned content coefficient.',
        proposals={})
    outputs = {}
    for suffix, name, low, high in [
        ('5', 'HighGapToLong', native_length, native_length * scale),
        ('6', 'HighGapToMid', native_length / scale**2, native_length / scale),
    ]:
        recipients = np.flatnonzero((gap_period >= low) & (gap_period <= high))
        if not len(recipients) or np.intersect1d(donors, recipients).size:
            raise ValueError('Recipient band must exist and be disjoint from donors')
        new_gaps = gaps.copy()
        new_gaps[donors] -= budget / len(donors)
        new_gaps[recipients] += budget / len(recipients)
        if not (new_gaps > 0).all() or not np.isclose(new_gaps.sum(), gaps.sum(), atol=1e-12, rtol=0):
            raise ValueError('Gap positivity or total range conservation failed')
        values = np.exp(np.r_[np.log(reference[0]), np.log(reference[0])-np.cumsum(new_gaps)]).astype(np.float32)
        values[0] = reference[0]
        # The accumulated transfer is exactly zero beyond the final recipient.
        values[recipients[-1]+1:] = reference[recipients[-1]+1:].astype(np.float32)
        if not (np.isfinite(values).all() and (values > 0).all() and (np.diff(values) < 0).all()):
            raise ValueError('Invalid deployed table')
        outputs[name] = values
        table = dict(values_float32=values.tolist(), gain=tables['MrPro']['gain'])
        job = dict(id=name, spec=dict(operator='static', table=table), panel='full',
                   nll_docs=16, nll_lengths=[8192, 16384, 32768])
        path = root / 'queue' / f'044{suffix}_{name}.json'
        if path.exists() and json.loads(path.read_text()) != job:
            raise ValueError('Cannot replace an existing gap-transfer contract')
        receipt['proposals'][name] = dict(
            table=table, recipient_period_interval=[low, high],
            recipient_gaps_zero_based=recipients.tolist(),
            gap_delta_float64=(new_gaps-gaps).tolist(),
            changed_slots_zero_based=np.flatnonzero(values != reference.astype(np.float32)).tolist(),
            maximal_phase_change_at_26_tokens=float(26*np.max(np.abs(values-reference))),
            job_filename=path.name)
    if not np.array_equal(outputs['HighGapToLong'][:high_end+1], outputs['HighGapToMid'][:high_end+1]):
        raise ValueError('Donor-band frequencies differ across the two arms')
    planned = root / 'planned_controls' / 'gap_budget_transfer.json'
    planned.parent.mkdir(parents=True, exist_ok=True)
    planned.write_text(json.dumps(receipt, indent=2)+'\n')
    for name, item in receipt['proposals'].items():
        path = root / 'queue' / item['job_filename']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(id=name, spec=dict(operator='static', table=item['table']),
                                       panel='full', nll_docs=16, nll_lengths=[8192,16384,32768]), indent=2)+'\n')
        print(name, 'donors', donors.tolist(), 'recipients', item['recipient_gaps_zero_based'], 'budget', budget)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--tables', required=True)
    args = parser.parse_args()
    prepare(args.root, args.tables)
