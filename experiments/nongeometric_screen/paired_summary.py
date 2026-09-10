"""Summarize completed independent jobs using matched rows and task strata."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    return {r['row_id']: r for r in
            (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def compare(reference, candidate, row_ids, seed=20260910, draws=10000):
    cells = defaultdict(list)
    for key in row_ids:
        a, b = reference[key], candidate[key]
        for field in ('prompt_sha256', 'task', 'length_cap', 'references'):
            if a[field] != b[field]:
                raise ValueError(f'matched-input mismatch: {key} {field}')
        cells[(a['length_cap'], a['task'])].append((a['correct'], b['correct']))
    rng = np.random.default_rng(seed)
    result = {}
    for length in sorted({length for length, _ in cells}):
        tasks, bootstrap = {}, []
        for cap, task in sorted(cells):
            if cap != length:
                continue
            values = np.asarray(cells[(cap, task)], dtype=float)
            delta = values[:, 1] - values[:, 0]
            samples = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(axis=1)
            bootstrap.append(samples)
            tasks[task] = dict(n=len(delta), reference=float(values[:, 0].mean()),
                              candidate=float(values[:, 1].mean()), delta=float(delta.mean()),
                              wins=int((delta > 0).sum()), losses=int((delta < 0).sum()),
                              paired_bootstrap_95=np.quantile(samples, [.025, .975]).tolist())
        macro = np.mean(bootstrap, axis=0)
        result[str(length)] = dict(
            tasks=tasks, macro_delta=float(np.mean([x['delta'] for x in tasks.values()])),
            paired_stratified_bootstrap_95=np.quantile(macro, [.025, .975]).tolist())
    return result


def main(root, job_path):
    root, job_path = Path(root), Path(job_path)
    job = json.loads(job_path.read_text())
    folder = root / 'holdout_results' / job['cohort']
    receipts = {name: json.loads((folder / (job['id'] + '__' + name + '_summary.json')).read_text())
                for name in job['methods']}
    reference_name = 'MrPro'
    row_ids = receipts[reference_name]['evaluated_row_ids']
    for name, receipt in receipts.items():
        if receipt['status'] != 'COMPLETE' or receipt['evaluated_row_ids'] != row_ids:
            raise ValueError(f'incomplete or unequal job panel: {name}')
    reference = read_rows(folder / (reference_name + '.jsonl'))
    result = dict(job_id=job['id'], cohort=job['cohort'], rows_per_method=len(row_ids),
                  reference=reference_name, bootstrap_seed=20260910, bootstrap_draws=10000,
                  scope='Frozen methods on the job-declared new-input panel. Intervals resample paired rows within task/length; task weights are equal. They are descriptive, unadjusted for multiple candidates, and do not measure source/template generalization.',
                  comparisons={name: compare(reference, read_rows(folder / (name + '.jsonl')), row_ids)
                               for name in job['methods'] if name != reference_name})
    output = folder / (job['id'] + '__paired_summary.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    main(args.root, args.job)
