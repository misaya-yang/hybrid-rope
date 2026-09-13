"""Winding-Matched CPU audit: phase-novelty profile beyond the WM0 arithmetic receipt.

CPU-only. Uses the canonical native FP32 table via table_for_config.
"""
import json
import math
import sys

import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoConfig
from experiments.olmo_recovery_20260912.recovery_v2_runtime import table_for_config

cfg = AutoConfig.from_pretrained("/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct")
wn32 = np.array(table_for_config(cfg, "Native")["values_float32"], dtype=np.float64)
B, K, L, S = 500000.0, 64, 8192, 8
r = L * wn32 / (2 * math.pi)
nstar = np.floor((S - 1) * r)
nu = wn32 / S + 2 * math.pi * nstar / (S * L)
m = -np.log(nu / wn32) / math.log(S)

print("WM check: nu in [w/S, w]:",
      bool(np.all(nu >= wn32 / S - 1e-15) and np.all(nu <= wn32 + 1e-15)),
      "| strictly decreasing:", bool(np.all(np.diff(nu) < 0)),
      "| sum(m) =", round(float(m.sum()), 4))
print(f"{'slot':>4} {'r_i(turns@8K)':>13} {'n*_i':>5} {'nu/w':>8} {'m_i':>7} {'unseen_arc':>10}")
for i in range(0, 64, 2):
    unseen = max(0.0, 1 - r[i]) if r[i] < 1 else 0.0
    print(f"{i:>4} {r[i]:>13.3f} {int(nstar[i]):>5} {nu[i]/wn32[i]:>8.4f} {m[i]:>7.4f} {unseen:>10.3f}")
mid = [(i, round(1 - r[i], 3)) for i in range(33, 48) if r[i] < 1]
print("slots 33-47 winding novelty (1 - r_i):", mid)
sel = [1 - r[i] for i in range(33, 47) if r[i] < 1]
print("peak-contrast zone 33-46 mean unseen arc:", round(float(np.mean(sel)), 3))
print("\nintermediate-length behavior (phase at X in turns; native reference):")
for i in [10, 14, 20, 24, 28, 32, 36, 40, 44, 48]:
    row = [f"slot {i}: r={r[i]:.2f} n*={int(nstar[i])} slope={nu[i]/wn32[i]:.4f}"]
    for X in (16384, 32768, 65536):
        row.append(f"{X//1024}K:{nu[i]*X/(2*math.pi):7.2f}t")
    print("  ".join(row), f"(native@8K: {r[i]:.2f}t)")

# BM comparison on unseen arc + slope deviation
bm = np.array(table_for_config(cfg, "BM_g8")["values_float32"], dtype=np.float64)
mb = -np.log(bm / wn32) / math.log(S)
print("\nvs BM_g8: sum(m) BM =", round(float(mb.sum()), 2), " WM =", round(float(m.sum()), 2))
print("BM slope deviations |bm/w - 1| >20% on slots:", [int(i) for i in range(K) if abs(bm[i]/wn32[i]-1) > 0.2])
print("WM slope deviations |nu/w - 1| >20% on slots:", [int(i) for i in range(K) if abs(nu[i]/wn32[i]-1) > 0.2])
nov = np.maximum(0.0, 1 - np.minimum(r + nstar, 1.0))
print("WM worst unseen-arc slots:", {int(i): round(float(nov[i]), 3) for i in np.argsort(-nov)[:6]})
out = {"r": r.tolist(), "nstar": nstar.tolist(), "nu_over_w": (nu / wn32).tolist(),
       "m": m.tolist(), "unseen_arc": nov.tolist()}
json.dump(out, open("wm_audit.json", "w"))
print("wrote /tmp/cputheory_r0/wm_audit.json")
