"""Split two identified frequency tables into disjoint finite interventions.

This module only constructs CPU arrays. It never loads a model, queues work,
or claims that either hybrid improves task performance.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .tables import band, build_tables


def split_tables(yarn, mrpro):
    """Return Y, max(Y,M), min(Y,M), M with exactly the supplied values.

    A selects MrRoPE only where its frequency is higher than YaRN's.
    B selects MrRoPE only where its frequency is lower. No exponent, boundary,
    interpolation strength, or gain is fitted to outcomes.
    """
    if yarn['gain'] != mrpro['gain']:
        raise ValueError('frequency attribution requires the same gain')
    y = np.asarray(yarn['values_float32'], dtype=np.float32)
    m = np.asarray(mrpro['values_float32'], dtype=np.float32)
    if y.ndim != 1 or y.shape != m.shape or len(y) < 2:
        raise ValueError('matching one-dimensional frequency arrays required')
    for v in (y, m):
        if not np.isfinite(v).all() or np.any(v <= 0) or np.any(np.diff(v) >= 0):
            raise ValueError('positive, finite, strictly decreasing inputs required')
    profiles = dict(yarn=y, only_faster=np.maximum(y, m),
                    only_slower=np.minimum(y, m), mrpro=m)
    tables = {name: dict(values_float32=v.tolist(), gain=yarn['gain'])
              for name, v in profiles.items()}
    return dict(tables=tables, faster_slots=np.flatnonzero(m > y).tolist(),
                slower_slots=np.flatnonzero(m < y).tolist(),
                unchanged_slots=np.flatnonzero(m == y).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = {'status': 'CPU_CONSTRUCTION_ONLY', 'settings': {}}
    for model, theta, window in [('qwen3b', 1e6, 32768),
                                  ('llama3_8b', 500000., 8192),
                                  ('olmo1b', 500000., 4096)]:
        for scale in (4., 16.):
            source = build_tables(theta, window, scale=scale)
            row = split_tables(source['yarn_index'], source['mrpro'])
            row.update(theta=theta, window=window, scale=scale,
                       band=list(band(theta, window)))
            result['settings'][f'{model}_s{scale:g}'] = row
    result['scope'] = ('Finite operator attribution and untested hybrid tables. '
                       'No measured model gains or automatic experiment release.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({name: {k: v for k, v in row.items() if k != 'tables'}
                      for name, row in result['settings'].items()}, indent=2))


if __name__ == '__main__':
    main()
