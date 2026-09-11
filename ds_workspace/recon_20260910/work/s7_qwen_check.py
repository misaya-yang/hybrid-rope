"""§7 cross-check: rebuild the Qwen arms by name and compare with the transport.

The r2turns/r2 arms are named turns_a<alpha>_b<beta>, i.e. the turn-window
family.  If that naming is right, rebuilding each table on the Qwen2.5-3B
config must reproduce the measured sum_m exactly.  If it does, then the Qwen
panel already contains a table sitting at the TRANSPORTED budget, and its
measured delta is the empirical answer to §7.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

import s7_s8_math as M

HERE = os.path.dirname(os.path.abspath(__file__))
JL = os.path.join(HERE, "jsonl")

QWEN3B = dict(theta=1e6, window=32768, head_dim=128)


def measured_sum_m(path):
    d = json.load(open(os.path.join(JL, path)))
    acc = d.get("accuracy", d.get("correct"))
    return d["sum_m"], acc, d.get("delta")


def main():
    print("=" * 78)
    print("§7 CROSS-CHECK: are the Qwen `turns_aX_bY` arms the turn-window family?")
    print("=" * 78)
    cfg = QWEN3B
    for arm in ["turns_a1_b64", "turns_a1_b16", "turns_a0p5_b32", "turns_a2_b32"]:
        m = re.match(r"turns_a([0-9p]+)_b(\d+)$", arm)
        alpha = float(m.group(1).replace("p", "."))
        beta = float(m.group(2))
        lo, hi = M.band_from_turns(alpha, beta, cfg["theta"], cfg["window"], cfg["head_dim"])
        tab = M.m_turns(alpha, beta, cfg["theta"], cfg["window"], ramp="beta1")
        got = M.S_of(tab)
        sm, acc, delta = measured_sum_m(f"r2turns/{arm}_summary.json")
        ok = "OK " if abs(got - sm) < 5e-6 else "MISMATCH"
        print(f"  {ok} {arm:16s} alpha={alpha:<5} beta={beta:<4} band=[{lo},{hi}] "
              f"n={hi-lo:2d}  built S={got:9.6f}  measured={sm:9.6f}  "
              f"acc={acc:.4f} delta_vs_mrpro={delta:+.4f}")

    print("\n" + "=" * 78)
    print("§7 THE DECISIVE COMPARISON: transported table vs the Qwen-native ones")
    print("=" * 78)
    src_tab = M.m_incr_beta(1.0, 21, 11)
    m_t, _ = M.transport_linear(src_tab, dict(theta=5e5, window=4096, head_dim=128), cfg)
    mu_t, var_t = M.moments(m_t)
    print(f"  transported OLMo a1_b64 -> Qwen3B:  S={M.S_of(m_t):.6f} mu={mu_t:.6f}")
    print(f"  Qwen-native turns_a1_b64:           S=33.5     mu=30.5")
    print(f"  -> the transported BUDGET (33.47) is 0.03 from the Qwen arm's own "
          f"(33.5)")
    print(f"     and the transported CENTROID (30.529) is 0.029 from it (30.5)")

    tab_q = M.m_turns(1.0, 64.0, cfg["theta"], cfg["window"], ramp="beta1")
    print(f"\n  shape agreement (m-space), transported vs Qwen-native turns_a1_b64:")
    print(f"    max |diff| whole table      = {np.abs(m_t-tab_q).max():.6f}")
    print(f"    n slots 0<m<1  transported  = {int(((m_t>1e-12)&(m_t<1-1e-12)).sum())}")
    print(f"    n slots 0<m<1  qwen-native  = "
          f"{int(((tab_q>1e-12)&(tab_q<1-1e-12)).sum())}")
    print(f"    first/last moved slot  transp = "
          f"{int(np.argmax(m_t>1e-12))}/{int(len(m_t)-1-np.argmax(m_t[::-1]<1-1e-12))}")
    print(f"    first/last moved slot  native = "
          f"{int(np.argmax(tab_q>1e-12))}/{int(len(tab_q)-1-np.argmax(tab_q[::-1]<1-1e-12))}")

    print("\n  per-slot m difference (transported - qwen_native):")
    for j in range(64):
        if abs(m_t[j] - tab_q[j]) > 1e-12:
            print(f"    j={j:3d}  transp={m_t[j]:.6f}  native={tab_q[j]:.6f}  "
                  f"diff={m_t[j]-tab_q[j]:+.6f}")

    print("\n" + "=" * 78)
    print("§7 WHAT THE QWEN PANEL ALREADY SAYS (measured, n=36 RULER)")
    print("=" * 78)
    print("  arm              S       acc      vs MrRoPE   @32768    @131072")
    rows = {}
    for arm in ["turns_a1_b64", "turns_a1_b16", "turns_a0p5_b32", "turns_a2_b32"]:
        rr = [json.loads(l) for l in open(os.path.join(JL, f"r2turns/{arm}.jsonl"))]
        rows[arm] = rr
    for arm, rr in rows.items():
        tab = None
        mm = re.match(r"turns_a([0-9p]+)_b(\d+)$", arm)
        alpha = float(mm.group(1).replace("p", "."))
        beta = float(mm.group(2))
        tab = M.m_turns(alpha, beta, cfg["theta"], cfg["window"], ramp="beta1")
        d = [r["delta"] for r in rr]
        d32 = [r["delta"] for r in rr if r["length_cap"] == 32768]
        d131 = [r["delta"] for r in rr if r["length_cap"] == 131072]
        n = len(d)
        se = float(np.std(d, ddof=1) / math.sqrt(n))
        print(f"  {arm:16s} {M.S_of(tab):6.2f}   "
              f"{np.mean([r['correct'] for r in rr]):.4f}   {np.mean(d):+.4f} "
              f"(se {se:.4f})  {np.mean(d32):+.4f}  {np.mean(d131):+.4f}")

    # mrpro reference
    rr = [json.loads(l) for l in open(os.path.join(JL, "r2turns/turns_a1_b64.jsonl"))]
    print(f"  {'MrRoPE (baseline)':16s} {'--':>6s}   "
          f"{np.mean([r['mrpro_correct'] for r in rr]):.4f}    0.0000")


if __name__ == "__main__":
    main()
