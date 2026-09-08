"""Small mixed RULER screen; task scores use the upstream matching definitions."""
from collections import defaultdict
import re

TASKS = ('niah_single_2', 'niah_multikey_2', 'niah_multiquery', 'vt', 'fwe', 'qa_1')
FAMILIES = dict(zip(TASKS, ('retrieval', 'retrieval', 'retrieval', 'tracking', 'aggregation', 'qa')))


def score(row, text):
    # NVIDIA RULER scripts/eval/evaluate.py postprocessing and synthetic/constants.py
    # matching, expressed on [0, 1] without rounding each individual observation.
    text = re.sub(r'[\x00-\x1f]', '\n', text.strip()).strip().lower()
    hits = [float(reference.lower() in text) for reference in row['references']]
    if not hits:
        raise ValueError('RULER row has no reference answers')
    return max(hits) if row['task'].startswith('qa_') else sum(hits) / len(hits)


def summarize(records):
    cells = defaultdict(list)
    for row in records:
        cells[(row['length_cap'], row['task'])].append(row['correct'])
    by_length = {}
    for length in sorted({key[0] for key in cells}):
        tasks = {task: sum(values) / len(values) for (cap, task), values in cells.items() if cap == length}
        by_length[str(length)] = dict(task_accuracy=tasks, macro_accuracy=sum(tasks.values()) / len(tasks))
    return dict(rows=len(records), score_sum=sum(r['correct'] for r in records),
                by_length=by_length, eos_count=sum(r['ended_eos'] for r in records))


def verdict(candidate, baseline):
    crows, brows = ({r['row_id']: r for r in rows} for rows in (candidate, baseline))
    if len(crows) != len(candidate) or len(brows) != len(baseline) or crows.keys() != brows.keys():
        raise ValueError('comparison has duplicate or unmatched rows')
    if not crows:
        raise ValueError('empty comparison')
    for key in crows:
        for field in ('prompt_sha256', 'task', 'length_cap', 'references'):
            if crows[key][field] != brows[key][field]:
                raise ValueError('paired row identity differs: ' + key)
    c, b = summarize(candidate), summarize(baseline)
    caps = sorted(c['by_length'], key=int)
    if len(caps) != 2 or int(caps[0]) <= 0 or any(
            set(cell['task_accuracy']) != set(TASKS) for cell in c['by_length'].values()):
        raise ValueError('incomplete task/length panel')
    deltas = {cap: c['by_length'][cap]['macro_accuracy'] - b['by_length'][cap]['macro_accuracy']
              for cap in c['by_length']}
    status = ('DEVELOPMENT_WIN' if deltas[caps[-1]] > 0 and deltas[caps[0]] >= 0 else
              'TRADEOFF' if deltas[caps[-1]] > 0 else 'NO_LONG_GAIN')
    return dict(status=status, macro_delta_by_length=deltas, candidate=c, baseline=b,
                paired_wins=sum(crows[k]['correct'] > brows[k]['correct'] for k in crows),
                paired_losses=sum(crows[k]['correct'] < brows[k]['correct'] for k in crows),
                baseline_extreme_cells=[f'{cap}/{task}' for cap, cell in b['by_length'].items()
                    for task, value in cell['task_accuracy'].items() if value in (0, 1)],
                evidence_scope='Six-task RULER development subset; not full RULER or independent confirmation')
