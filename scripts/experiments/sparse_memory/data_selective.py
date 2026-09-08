"""V2 removes the marker-adjacency shortcut: five independently queried entities.

Every block contains three randomized rounds of updates for five entities. A
single remote marker falls inside the second round. The query entity is sampled
after constructing the full stream, so the immediately preceding event usually
belongs to another entity. Only one block has a marker; all blocks have the same
event density, and the query's correct predecessor always exists inside it.
"""
import json
from pathlib import Path

import numpy as np

from .data import SET, MARK, PAD, SEP, QUERY, BEFORE, CONTENT, ANSWER, ENTITY, VALUE, generate, solve


def one_pair(rng, pair_id):
    unique = rng.randrange(16)
    pool = [e for e in range(16) if e != unique]
    target_block = rng.randrange(1, 5)
    events = []
    block_entities = []
    marker_event = None
    for block in range(8):
        entities = rng.sample(pool, 5)
        block_entities.append(entities)
        values = {e:rng.sample(range(16), 3) for e in entities}
        inner = []
        for round_id in range(3):
            order = rng.sample(entities, len(entities))
            inner.extend([[SET, ENTITY+e, VALUE+values[e][round_id], SEP] for e in order])
        slot = rng.randrange(6, 10)
        special = [MARK, SEP, SEP, SEP] if block == target_block else [PAD, SEP, SEP, SEP]
        inner.insert(slot, special)
        if block == target_block: marker_event = len(events)+slot
        events.extend(inner)
    target = rng.choice(block_entities[target_block])
    before = [i for i,e in enumerate(events) if i < marker_event and e[0] == SET and e[1] == ENTITY+target]
    after = [i for i,e in enumerate(events) if i > marker_event and i < 16*(target_block+1)
             and e[0] == SET and e[1] == ENTITY+target]
    a, b = before[-1], after[0]
    # Content control is remote and never overwrites the marked block.
    free = [i for i in range(16, 80) if i//16 != target_block and events[i][0] == SET]
    ui = rng.choice(free); uv = rng.randrange(16)
    events[ui] = [SET, ENTITY+unique, VALUE+uv, SEP]
    prefix = np.asarray(events, dtype=np.int16).reshape(-1)
    swapped = prefix.copy()
    swapped[4*a+2], swapped[4*b+2] = swapped[4*b+2], swapped[4*a+2]
    query = [QUERY, ENTITY+target, BEFORE, ANSWER]
    rows = [np.concatenate((prefix, query)), np.concatenate((swapped, query)),
            np.concatenate((prefix, [QUERY, ENTITY+unique, CONTENT, ANSWER]))]
    labels = [int(prefix[4*a+2]), int(prefix[4*b+2]), VALUE+uv]
    # A frozen alternate query can directly test query-dependent answer changes.
    candidates = []
    for entity in block_entities[target_block]:
        alt = np.concatenate((prefix, [QUERY, ENTITY+entity, BEFORE, ANSWER]))
        value = solve(alt)
        if entity != target and value != labels[0]: candidates.append((entity, value))
    alternate = rng.choice(candidates) if candidates else None
    meta = dict(pair_id=pair_id, block=target_block, marker_slot=marker_event%16,
        target_positions=[4*a+2, 4*b+2], marker_position=4*marker_event,
        immediately_preceding_is_target=events[marker_event-1][1] == ENTITY+target,
        content_position=4*ui+2, alternate_query=alternate)
    return rows, labels, meta


def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True)
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    splits = [generate(out/'train.npz', 40000, 2026090811, one_pair),
              generate(out/'development.npz', 512, 2026090812, one_pair),
              generate(out/'test.npz', 2048, 2026090813, one_pair)]
    (out/'manifest.json').write_text(json.dumps(dict(splits=splits, length=516,
        semantics='v2 selective query; five entities x three updates per compression block',
        correction='v1 relation task can be solved by returning the value immediately before MARK'), indent=2))
    print(json.dumps(splits))


if __name__ == '__main__': main()
