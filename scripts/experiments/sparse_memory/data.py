"""Frozen paired event streams; vocabulary tokens are the actual answer strings."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np

PAD, SET, MARK, SEP, QUERY, BEFORE, CONTENT, ANSWER = range(8)
ENTITY = 8
VALUE = 32
ENTITIES, VALUES = 16, 16
LENGTH = 132


def one_pair(rng, pair_id):
    target, unique = rng.sample(range(ENTITIES), 2)
    va, vb, vc = rng.sample(range(VALUES), 3)
    remaining = [e for e in range(ENTITIES) if e not in (target, unique)]
    events = [[SET, ENTITY+rng.choice(remaining), VALUE+rng.randrange(VALUES), SEP]
              for _ in range(32)]
    # Three updates and one marker prevent the shortcut 'always return first'.
    # Marker at slot 1 or 2 changes which update is the nearest predecessor.
    block = rng.randrange(1, 5)
    marker_slot = rng.choice([1, 2])
    slots = [marker_slot-1, marker_slot, marker_slot+1]
    a, mark, b = [4*block+s for s in slots]
    other = 4*block+next(s for s in range(4) if s not in slots)
    events[other] = [SET, ENTITY+target, VALUE+vc, SEP]
    events[a] = [SET, ENTITY+target, VALUE+va, SEP]
    events[mark] = [MARK, SEP, SEP, SEP]
    events[b] = [SET, ENTITY+target, VALUE+vb, SEP]
    free = [i for i in range(4, 20) if i not in (a, mark, b, other)]
    ui = rng.choice(free)
    uv = rng.randrange(VALUES)
    events[ui] = [SET, ENTITY+unique, VALUE+uv, SEP]
    prefix = np.asarray(events, dtype=np.int16).reshape(-1)
    swapped = prefix.copy()
    swapped[4*a+2], swapped[4*b+2] = swapped[4*b+2], swapped[4*a+2]
    query = [QUERY, ENTITY+target, BEFORE, ANSWER]
    content = [QUERY, ENTITY+unique, CONTENT, ANSWER]
    rows = [np.concatenate((prefix, query)), np.concatenate((swapped, query)),
            np.concatenate((prefix, content))]
    labels = [VALUE+va, VALUE+vb, VALUE+uv]
    meta = dict(pair_id=pair_id, block=block, slots=slots, target_positions=[4*a+2, 4*b+2],
                marker_position=4*mark, content_position=4*ui+2)
    return rows, labels, meta


def generate(path, pairs, seed, pair_fn=one_pair):
    rng = random.Random(seed)
    xs, ys, metadata = [], [], []
    for i in range(pairs):
        x, y, meta = pair_fn(rng, i)
        xs.extend(x); ys.extend(y); metadata.append(meta)
    path = Path(path)
    np.savez_compressed(path, x=np.asarray(xs, dtype=np.int16), y=np.asarray(ys, dtype=np.int16))
    path.with_suffix('.json').write_text(json.dumps(dict(seed=seed, pairs=pairs,
        metadata=metadata), separators=(',', ':')))
    return dict(file=path.name, seed=seed, pairs=pairs, rows=3*pairs,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                metadata_sha256=hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest())


def solve(x):
    """Independent event interpreter; label-generation code is not reused."""
    entity, task = int(x[-3]), int(x[-2])
    value = None
    for event in np.asarray(x[:-4]).reshape(-1, 4):
        if event[0] == MARK and task == BEFORE:
            break
        if event[0] == SET and event[1] == entity:
            value = int(event[2])
    if value is None:
        raise ValueError('unanswerable stream')
    return value


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    splits = [generate(out/'train.npz', 40000, 2026090801),
              generate(out/'development.npz', 512, 2026090802),
              generate(out/'test.npz', 2048, 2026090803)]
    (out/'manifest.json').write_text(json.dumps(dict(splits=splits, length=LENGTH,
        vocab=dict(entity_range=[8, 23], value_range=[32, 47], special=list(range(8))),
        semantics='one generated vocabulary token; no candidate masking; pair rows 3i,3i+1'), indent=2))
    print(json.dumps(splits))


if __name__ == '__main__':
    main()
