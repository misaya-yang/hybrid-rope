import hashlib
from types import SimpleNamespace

import numpy as np

from scripts.analysis.build_band_evidence_ledger import native_turns, table_hash


def test_band_ledger_hash_and_turn_coordinate():
    values = np.linspace(1.0, 0.1, 64, dtype=np.float32)
    expected = hashlib.sha256(values.astype("<f4").tobytes()).hexdigest()
    assert table_hash({"values_float32": values.tolist()}) == expected
    turns = native_turns("OLMo-2-0425-1B-Instruct", (14, 31))
    assert turns[0] > turns[1] > 0.0
