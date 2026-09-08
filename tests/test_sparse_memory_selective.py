"""The repaired assay requires entity selection, not marker-neighbor copying."""
import random

import numpy as np

from scripts.experiments.sparse_memory.data import solve, ENTITY
from scripts.experiments.sparse_memory.data_selective import one_pair


def test_selective_task_labels_and_shortcut_failure():
    rng = random.Random(19)
    adjacent = 0
    alternates = 0
    for i in range(1000):
        rows, y, m = one_pair(rng, i)
        assert [solve(x) for x in rows] == y
        assert y[0] != y[1]
        assert sorted(rows[0][:-4]) == sorted(rows[1][:-4])
        assert min(515-p for p in m['target_positions']) > 45
        assert all(p//64 == m['block'] for p in m['target_positions'])
        adjacent += m['immediately_preceding_is_target']
        if m['alternate_query']:
            entity, value = m['alternate_query']
            alternate = rows[0].copy(); alternate[-3] = ENTITY+entity
            assert solve(alternate) == value and value != y[0]
            alternates += 1
    assert 100 < adjacent < 300
    assert alternates > 950
