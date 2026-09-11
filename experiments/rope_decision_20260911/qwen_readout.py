"""Corrected readout for the legacy Qwen NLL arrays; no GPU work."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def clustered_se(values, groups):
    d = np.asarray(values, dtype=float)
    if len(d) != len(groups) or not len(d) or not np.isfinite(d).all():
        raise ValueError('invalid observations or group alignment')
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError('at least two books are required')
    residual = d - d.mean()
    sums = np.array([residual[groups == g].sum() for g in unique])
    return float(np.sqrt(len(unique)/(len(unique)-1) * np.sum(sums*sums) / len(d)**2))


def summarize(bm, mr, groups):
    bm, mr = np.asarray(bm,float), np.asarray(mr,float)
    if bm.shape != mr.shape or bm.ndim != 1:
        raise ValueError('unmatched complete paired arrays')
    d = bm-mr
    se = clustered_se(d,groups)
    mean = float(d.mean())
    t = mean/se if se else None
    return dict(n=len(d), books=len(set(groups)), bm_minus_mrpro=mean,
                lower_nll_arm='BM' if mean < 0 else 'MrRoPE' if mean > 0 else 'tie',
                piece_weighted_cluster_se=se, t=t,
                criterion_met=bool(abs(mean)>=.004 and (t is not None and abs(t)>=3)),
                scope='Same-gain legacy NLL only; not native-gain rescue or task generalization.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,default=Path('/root/autodl-tmp/phase1_20260910/qwen4x_power'))
    ap.add_argument('--books',type=Path,default=Path('/root/autodl-tmp/longtext/prepared_pg19_4x/rows.json'))
    args = ap.parse_args()
    rows = {}
    for line in (args.run/'rows.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['arm'] in rows:
            raise ValueError('duplicate arm from appended/restarted runs; choose one identified run')
        rows[r['arm']] = r
    needed = {'beta_b1_BM','mrpro'}
    if not needed <= rows.keys():
        print(json.dumps(dict(status='INCOMPLETE',present=sorted(rows))))
        return
    bm, mr = rows['beta_b1_BM']['per_doc'], rows['mrpro']['per_doc']
    books = json.loads(args.books.read_text())
    if len(books) < len(bm):
        raise ValueError('missing book identities')
    result = summarize(bm,mr,[b['book'] for b in books[:len(bm)]])
    result['protocol_limitation'] = 'Legacy far=131073 slices far+1 across chunk boundaries; do not use for a clean within-book verdict.'
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
