"""Verify the recorded twelve-profile reconstruction; performs no model execution."""
import hashlib
import json
from pathlib import Path
import numpy as np

data = json.loads(Path(__file__).with_name('profile_diagnostic_inputs.json').read_text())
rows = data['rows']
assert len(rows) == 12
native = data['native_base'] ** (-np.arange(data['K'], dtype=float) / data['K'])
for row in rows:
    frequencies = native * 4.0 ** (-np.asarray(row['m']))
    turns = data['test_distance'] * frequencies / (2 * np.pi)
    lo, hi = data['thresholds']
    assert np.count_nonzero((turns >= lo) & (turns <= hi)) == row['N']
    assert hashlib.sha256(frequencies.astype('<f4').tobytes()).hexdigest() == row['reconstructed_float32_sha256']
    assert abs(row['score_sum'] / 350 - row['score']) < 1e-12
x = np.array([row['N'] for row in rows])
y = np.array([row['score'] for row in rows])
coefficients = np.linalg.lstsq(np.c_[np.ones(len(x)), x], np.log(y / (1-y)), rcond=None)[0]
assert np.allclose(coefficients, data['coefficients'], rtol=0, atol=1e-10)
counts = dict(concordant=0, discordant=0, tied_count=0)
for i, a in enumerate(rows):
    for b in rows[i+1:]:
        delta = (a['N']-b['N']) * (a['score']-b['score'])
        key = 'tied_count' if a['N'] == b['N'] else 'concordant' if delta > 0 else 'discordant'
        counts[key] += 1
assert counts == data['pair_counts']
print('Verified 12 reconstructed tables, score aggregates, fit coefficients, and 66 pair classifications.')
