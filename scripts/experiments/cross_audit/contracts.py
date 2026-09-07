"""Small, explicit data/scoring contracts shared by CPU preparation and runs."""
from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def read_rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def group_key(row):
    return (row['family'], row['layout'], int(row['length_cap']), row['group_id'])


def validate_rows(rows):
    if not rows:
        raise ValueError('empty evaluation selection')
    seen, groups = set(), defaultdict(list)
    for r in rows:
        if r['row_id'] in seen:
            raise ValueError('duplicate row_id')
        seen.add(r['row_id'])
        if not r['prompt_ids'] or not all(type(x) is int and x >= 0 for x in r['prompt_ids']):
            raise ValueError('invalid prompt tokens')
        if int(r['generation_budget']) <= 0:
            raise ValueError('generation budget must be positive')
        if not r.get('accepted_full_answers') and not r.get('expected_ruler_answers'):
            raise ValueError('missing scoring targets')
        groups[group_key(r)].append(str(r['world']))
    for key, worlds in groups.items():
        expected = {'0'} if key[0].startswith('ruler_') else {'0', '1'}
        if len(worlds) != len(set(worlds)) or set(worlds) != expected:
            raise ValueError(f'incomplete or duplicate worlds: {key}')
    return groups


def select_groups(rows, groups_per_cell):
    """First complete groups in frozen source order, without outcome selection."""
    validate_rows(rows)
    if groups_per_cell <= 0:
        raise ValueError('groups_per_cell must be positive')
    selected, per_cell = set(), defaultdict(set)
    for r in rows:
        key = group_key(r)
        if len(per_cell[key[:3]]) < groups_per_cell:
            per_cell[key[:3]].add(key)
            selected.add(key)
    result = [r for r in rows if group_key(r) in selected]
    validate_rows(result)
    return result


def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score(row, text, ended_eos):
    golds = row.get('accepted_full_answers', [])
    expected = row.get('expected_ruler_answers', [])
    multi = row['family'] == 'ruler_multi_key'
    # The legacy schema stores separate required values in accepted_full_answers.
    # For this diagnostic family freeze a comma/whitespace-separated value list.
    if multi:
        values = [v for v in re.split(r'[,\s]+', text.strip()) if v]
        exact = bool(expected) and Counter(values) == Counter(expected)
    else:
        exact = bool(golds) and text.strip() in golds
    p = Counter(normalize(text).split())
    f1 = 0.0
    for g in golds:
        q = Counter(normalize(g).split())
        common = sum((p & q).values())
        if common:
            f1 = max(f1, 2 * common / (sum(p.values()) + sum(q.values())))
    return {
        'full_answer_exact': exact,
        'full_answer_exact_eos': exact and bool(ended_eos),
        'ended_with_eos': bool(ended_eos),
        'qa_em': None if multi else any(normalize(text) == normalize(g) for g in golds),
        'qa_f1': None if multi else f1,
        'required_value_recall': sum(v.lower() in text.lower() for v in expected) / len(expected) if expected else None,
        'all_required_contains': bool(expected) and all(v.lower() in text.lower() for v in expected),
    }


def summarize(records, tasks):
    validate_rows(tasks)
    expected = {r['row_id']: r for r in tasks}
    actual = {r['row_id']: r for r in records}
    if len(actual) != len(records) or actual.keys() != expected.keys():
        raise ValueError('missing, duplicate or unexpected output rows')
    cells = defaultdict(list)
    for key, worlds in validate_rows(tasks).items():
        rs = [actual[r['row_id']] for r in tasks if group_key(r) == key]
        cells[key[:3]].append(all(r['full_answer_exact_eos'] for r in rs))
    return {':'.join(map(str, k)): {'groups': len(v), 'complete_group_successes': sum(v)}
            for k, v in sorted(cells.items())}
