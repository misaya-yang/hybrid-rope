"""Paired row-level reporting. CIs condition on the trained seed, not over seeds."""
import argparse
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if [r['row'] for r in rows] != list(range(len(rows))) or len(rows)%3:
        raise ValueError('row identity/order drift')
    if any(r['correct'] != (r['prediction'] == r['target']) for r in rows):
        raise ValueError('stored score differs from raw generated token')
    return rows


def compare(a_path, b_path, metadata_path):
    a, b = read_rows(a_path), read_rows(b_path)
    if len(a) != len(b) or any((x['target'], x['pair_id'], x['task'], x['world']) !=
                             (y['target'], y['pair_id'], y['task'], y['world']) for x,y in zip(a,b)):
        raise ValueError('not the same paired evaluation inputs')
    ac = np.array([r['correct'] for r in a]).reshape(-1, 3)
    bc = np.array([r['correct'] for r in b]).reshape(-1, 3)
    ap, bp = ac[:, 0]&ac[:, 1], bc[:, 0]&bc[:, 1]
    d = bp.astype(float)-ap.astype(float)
    rng = np.random.default_rng(20260908)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(5000)])
    meta = json.loads(Path(metadata_path).read_text())['metadata']
    if len(meta) != len(ap): raise ValueError('metadata count differs')
    groups = {}
    for key in sorted({str(m.get('marker_slot', m.get('slots'))) for m in meta}):
        idx = np.array([str(m.get('marker_slot', m.get('slots'))) == key for m in meta])
        groups[key] = dict(n=int(idx.sum()), a_pair=float(ap[idx].mean()), b_pair=float(bp[idx].mean()))
    return dict(pairs=len(ap), a_pair=float(ap.mean()), b_pair=float(bp.mean()),
        delta_pair_pp=float(d.mean()*100), sample_pair_bootstrap95_pp=(100*np.quantile(boot,[.025,.975])).tolist(),
        a_only_pairs=int((ap&~bp).sum()), b_only_pairs=int((bp&~ap).sum()),
        a_content=float(ac[:, 2].mean()), b_content=float(bc[:, 2].mean()),
        content_delta_pp=float((bc[:,2].mean()-ac[:,2].mean())*100), by_marker_layout=groups,
        uncertainty='conditional on this trained seed; streams are sampling units, not independent training replications')


def main():
    p=argparse.ArgumentParser()
    for name in ['a','b','metadata','out']: p.add_argument('--'+name, required=True)
    args=p.parse_args(); result=compare(args.a,args.b,args.metadata)
    Path(args.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result))


if __name__ == '__main__': main()
