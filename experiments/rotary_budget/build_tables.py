"""Fixed pair identities; only active frequencies change across six arms."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ARMS = ('G32', 'E32', 'G16', 'E16', 'P16', 'U16')
TAU = math.sqrt(2)
BASE = 500000.0

def active_table(k, evq=False, tau=TAU):
    if k < 2 or k > 32:
        raise ValueError('2 <= K <= 32 required')
    z = np.linspace(0., 1., k)
    if evq and abs(tau) > 1e-10:
        u = (np.arange(k) + .5) / k
        q = 1 - np.arcsinh((1-u)*np.sinh(tau))/tau
        z = (q-q[0])/(q[-1]-q[0])
    return np.exp(-(31/32)*math.log(BASE)*z).astype(np.float32)

def build_table(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    result = np.zeros(32, dtype=np.float32)
    if arm in ('P16','U16'):
        full = active_table(32)
        result[:16] = full[:16] if arm == 'P16' else full[::2]
    else:
        k = int(arm[1:])
        result[:k] = active_table(k, arm.startswith('E'))
    return result

def receipt():
    return {arm: {'inv_freq': build_table(arm).tolist(),
                  'sha256_float32': hashlib.sha256(build_table(arm).tobytes()).hexdigest(),
                  'pair_layout': '(j,j+32)', 'scale': 1/8,
                  'active_pairs': int(np.count_nonzero(build_table(arm)))} for arm in ARMS}

if __name__ == '__main__':
    path = Path(__file__).with_name('tables.json')
    path.write_text(json.dumps(receipt(), indent=2)+'\n')
    print(path)
