#!/usr/bin/env python3
"""
Verify paper/tables/table5_phase11_leverage.tex YaRN gain numbers.

Source: results/core_text/phase11/results_phase11_yarn.json (350M, L=256, 3 seeds)

Paper says:
  Geo:        YaRN gain @4K = -4.5%, @8K = -3.1%
  EVQ tau=2:  YaRN gain @4K = -27.0%, @8K = -28.9%
  EVQ tau=4:  YaRN gain @4K = -32.5%, @8K = -40.7%

Using `yarn_auto[L]` (best-scale) vs `raw[L]` to compute gain = (yarn-raw)/raw.
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

P = Path("/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/results/core_text/phase11/results_phase11_yarn.json")
data = json.loads(P.read_text())

methods = {
    "Geo":     ["350m_geo_seed42",     "350m_geo_seed137",     "350m_geo_seed256"],
    "EVQ2.0":  ["350m_evq2.0_seed42",  "350m_evq2.0_seed137",  "350m_evq2.0_seed256"],
    "EVQ4.0":  ["350m_evq4.0_seed42",  "350m_evq4.0_seed137",  "350m_evq4.0_seed256"],
}

for method, ids in methods.items():
    raws  = [data[i]["raw"] for i in ids]
    yarn_autos = [data[i]["yarn_auto"] for i in ids]
    print(f"\n{method}:")
    for L in [4096, 8192]:
        deltas = []
        for ra, ya in zip(raws, yarn_autos):
            r = ra[str(L)]
            y = ya[str(L)]
            d = (y - r) / r * 100
            deltas.append(d)
        m = statistics.mean(deltas)
        s = statistics.stdev(deltas) if len(deltas) > 1 else 0
        print(f"  L={L}: per-seed gains = [{', '.join(f'{d:+.1f}%' for d in deltas)}] | mean={m:+.2f}% std={s:.2f}")

# Also reproduce the NTK row
print("\n--- NTK-aware @8K (paper says: Geo 198.1, EVQ2 143.3, EVQ4 331.4) ---")
for method, ids in methods.items():
    ntks = [data[i]["ntk_auto"]["8192"] for i in ids]
    print(f"  {method}: mean PPL@8K (NTK) = {statistics.mean(ntks):.2f}, per-seed = {ntks}")
