#!/usr/bin/env python3
"""Read OLMO_Z1 round outputs into a compact summary (server-side, read-only).

Usage: /root/miniconda3/bin/python read_olmo_round.py
Prints JSON: strict both-worlds counts per family/layout/length for task0/z0/task128,
EOS health, and the frozen review verdict.
"""
import json
from collections import Counter
from pathlib import Path

OUT = Path('/root/autodl-tmp/claude_round11_olmo_20260905/out')

def strict_counts(eval_dir):
    """Per (family,layout,length): groups where BOTH worlds exact+EOS."""
    pairs = {}
    with (eval_dir / 'examples.jsonl').open() as handle:
        for line in handle:
            r = json.loads(line)
            key = (r['family'], r['layout'], r['length_cap'], r['semantic_id'])
            pairs.setdefault(key, {})[r['world']] = bool(r['full_exact_eos'])
    tallies = Counter(); eos_fail = Counter(); total = Counter()
    for (family, layout, length, _), worlds in pairs.items():
        cell = (family, layout, length)
        total[cell] += 1
        if len(worlds) != 2:
            continue
        if all(worlds.values()):
            tallies[cell] += 1
        if not any(worlds.values()):
            eos_fail[cell] += 1
    stringify = lambda c: {f'{f}:{la}:{le}': v for (f, la, le), v in sorted(c.items())}
    return {'groups': stringify(total), 'both_worlds_strict': stringify(tallies),
            'both_worlds_fail': stringify(eos_fail)}

def eos_health(eval_dir):
    """Per layout/length: fraction of rows ending with EOS."""
    stats = Counter(); ended = Counter()
    with (eval_dir / 'examples.jsonl').open() as handle:
        for line in handle:
            r = json.loads(line)
            cell = (r['layout'], r['length_cap'])
            stats[cell] += 1
            ended[cell] += bool(r['ended_with_eos'])
    return {f'{l}:{n}': round(ended[(l, n)] / stats[(l, n)], 3) for l, n in sorted(stats)}

summary = {}
for name in ('task0', 'z0', 'task128'):
    d = OUT / name
    if not (d / 'examples.jsonl').exists():
        summary[name] = 'MISSING'
        continue
    summary[name] = {**strict_counts(d), 'eos_rate': eos_health(d)}
for name in ('native0', 'native128'):
    p = OUT / name / 'native_evaluation.json'
    if p.exists():
        receipt = json.loads(p.read_text())
        summary[name] = {'rows': receipt['rows'], 'status': receipt['status']}
review_path = OUT / 'review.json'
if review_path.exists():
    review = json.loads(review_path.read_text())
    summary['review'] = {
        'status': review['status'], 'next_action': review['next_action'],
        'native_retention': review.get('native_retention'),
        'paired_ci95': review.get('paired_retention_uncertainty', {}).get('ci95'),
        'family_single_evidence': review.get('family_comparisons', {}).get('single_evidence', {}).get('all_groups'),
    }
print(json.dumps(summary, indent=2))
