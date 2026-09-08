"""Four source-dependent tasks, frozen without observing any model output."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import random
import re

FAMILIES = ('lookup', 'linked_lookup', 'latest_update', 'attribute_binding')
CAPS = (4096, 8192, 16384)
WORDS = ('amber', 'birch', 'cedar', 'coral', 'denim', 'ivory', 'jade', 'lilac',
         'maple', 'olive', 'pearl', 'ruby', 'sable', 'teal', 'umber', 'violet')
MAX_NEW_TOKENS = 16


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def instance(family, length_cap, seed, encode, *, compact=False):
    """Two worlds: same query and distractors, different source-bound answer."""
    rng = random.Random(int(digest([family, length_cap, seed, compact])[:16], 16))
    ids = rng.sample(range(10000, 99999), 7000)
    item, holder = 'R'+str(ids[0]), 'H'+str(ids[1])
    answers = rng.sample(WORDS, 2)
    old = [w for w in WORDS if w not in answers][:2]
    placement = (.2, .8)[rng.randrange(2)]
    backgrounds = []
    for i in range(2, len(ids)):
        key, label = 'R'+str(ids[i]), rng.choice(WORDS)
        if family == 'lookup':
            text = f'Item {key}: label {label}.'
        elif family == 'linked_lookup':
            h = 'H'+str(ids[i])
            text = f'Item {key}: holder {h}. Holder {h}: label {label}.'
        elif family == 'latest_update':
            text = f'Update for item {key}: label {label}.'
        else:
            # Every distractor misses at least one queried attribute.
            group, material = rng.choice((('north','wood'), ('south','glass'), ('east','stone')))
            text = f'Item {key}: group {group}; material {material}; label {label}.'
        backgrounds.append(text)

    if family == 'lookup':
        question = f'What is the label of item {item}?'
        instruction = 'Find the record for the exact requested item.'
    elif family == 'linked_lookup':
        question = f'What is the label of the holder of item {item}?'
        instruction = 'First find the item holder, then find that holder label.'
    elif family == 'latest_update':
        question = f'What is the label in the last update for item {item}?'
        instruction = 'Updates are ordered from oldest to newest. Use the final matching update.'
    else:
        question = 'What is the label of the item whose group is north AND whose material is glass?'
        instruction = 'Both requested attributes must belong to the same record.'

    def render(count, world):
        if family == 'lookup':
            evidence = [(placement, f'Item {item}: label {answers[world]}.')]
        elif family == 'linked_lookup':
            evidence = [(.2, f'Item {item}: holder {holder}.'),
                        (.8, f'Holder {holder}: label {answers[world]}.')]
        elif family == 'latest_update':
            evidence = [(.15, f'Update for item {item}: label {old[0]}.'),
                        (.5, f'Update for item {item}: label {old[1]}.'),
                        (.85, f'Update for item {item}: label {answers[world]}.')]
        else:
            evidence = [(placement, f'Item {item}: group north; material glass; label {answers[world]}.')]
        lines = list(backgrounds[:count])
        for fraction, line in sorted(evidence, reverse=True):
            lines.insert(round(fraction*count), line)
        text = (instruction+' Use only the records below.\n'
                'Answer with exactly one label word, without explanation.\n\nRecords:\n'
                +'\n'.join(lines)+'\n\nQuestion: '+question+'\nAnswer:')
        return text, [line for _, line in evidence]

    target = length_cap-MAX_NEW_TOKENS-32
    left, right, selected = 0, min(len(backgrounds), length_cap//4), None
    while left <= right:
        middle = (left+right)//2
        text, evidence = render(middle, 0)
        ids0 = encode(text)
        if len(ids0) <= target:
            selected = middle
            left = middle+1
        else:
            right = middle-1
    if selected is None:
        raise ValueError('instruction and evidence exceed the length cap')
    rows = []
    for world in (0, 1):
        text, evidence = render(selected, world)
        prompt_ids = encode(text)
        if len(prompt_ids)+MAX_NEW_TOKENS > length_cap:
            raise ValueError('counterfactual prompt exceeds frozen capacity')
        group_id = digest([family, length_cap, seed, compact])
        rows.append(dict(row_id=digest([group_id, world]), group_id=group_id,
                         family=family, length_cap=length_cap, world=world,
                         split='qualification' if compact else 'development',
                         question=question, prompt_text=text, prompt_ids=prompt_ids,
                         prompt_sha256=digest(prompt_ids), input_tokens=len(prompt_ids),
                         answer=answers[world], evidence_lines=evidence,
                         distractor_records=selected, max_new_tokens=MAX_NEW_TOKENS))
    if rows[0]['question'] != rows[1]['question'] or rows[0]['answer'] == rows[1]['answer']:
        raise AssertionError('invalid counterfactual pair')
    return rows


def prepare_rows(encode, seed=20260908):
    screen, qualification = [], []
    for family in FAMILIES:
        qualification.extend(instance(family, 768, seed+1, encode, compact=True))
        for cap in CAPS:
            screen.extend(instance(family, cap, seed, encode))
    return screen, qualification


def normalize_answer(text):
    # Whole-answer normalization; never search a long response for the target.
    value = text.strip().lower()
    value = re.sub(r'^answer\s*:\s*', '', value)
    return value.strip(' \t\r\n.!,;:\"\'`')


def score(row, text):
    return int(normalize_answer(text) == row['answer'])


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['family']].append(row)
    means = {f:sum(r['correct'] for r in rows)/len(rows) for f, rows in groups.items()}
    native = [r for r in records if r['length_cap'] == 4096]
    return dict(rows=len(records), correct=sum(r['correct'] for r in records),
                family_accuracy=means, macro_accuracy=sum(means.values())/len(means),
                native_4k_correct=sum(r['correct'] for r in native), native_4k_rows=len(native),
                eos_count=sum(r['ended_eos'] for r in records))


def verdict(candidate, baseline):
    if {r['row_id'] for r in candidate} != {r['row_id'] for r in baseline}:
        raise ValueError('unmatched or incomplete comparison')
    if len(candidate) != len(baseline) or len({r['row_id'] for r in candidate}) != len(candidate):
        raise ValueError('duplicate or incomplete result rows')
    c, b = summarize(candidate), summarize(baseline)
    gain = c['correct']-b['correct']
    native_delta = c['native_4k_correct']-b['native_4k_correct']
    status = ('DEVELOPMENT_WIN' if gain > 0 and native_delta >= 0 else
              'TRADEOFF' if gain > 0 else 'NO_GAIN')
    return dict(status=status, net_correct=gain, macro_delta=c['macro_accuracy']-b['macro_accuracy'],
                native_4k_delta=native_delta,
                family_delta={f:c['family_accuracy'][f]-b['family_accuracy'][f] for f in FAMILIES},
                evidence_scope='small constructed development screen; not generalization or SOTA')
