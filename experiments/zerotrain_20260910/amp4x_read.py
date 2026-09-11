"""Read the 4x amplitude prescription (AMP4X_PRESCRIPTION_PREREG_20260911).

RUN ON THE SERVER:
    cd /root/autodl-tmp/phase1_20260910 && python amp4x_read.py

THE PREDICTION UNDER TEST.  The coverage criterion gives the plateau height that
maximises coverage of the target depth band.  For the deployed 4x target it gives
m_p ~ 1.13, whereas every deployed table sits at m_p = 1.  If the prescription is
right, amp4x_1p13 should beat the deployed table on the 350-row panel at 16384,
and amp4x_1p30 brackets the overshoot side.

SIGN CONVENTION, stated because this campaign has lost time to getting it wrong:
`correct` is a score, so HIGHER IS BETTER and a POSITIVE delta favours the arm.
This file never prints a bare signed delta without also naming the winner.

THE FOUR PRE-REGISTERED CRITERIA, in the order the prereg fixes them:
    1.13 > 1.00 with paired t >= 2        -> prescription holds
    |delta(1.13 vs 1.00)| <= 1pp          -> no resolution on the amplitude axis
    1.13 < 1.00 with t <= -2              -> prescription falsified
    1.30 > 1.13                           -> optimum is still higher; keep climbing
The 4096 rows are reported unconditionally: the prereg makes them mandatory,
because the whole family is supposed to pay in-window for out-of-window reach.

REFUSES ON PARTIAL DATA.  An earlier reader in this campaign reported +1.34pp on
112 of 391 rows -- the first two task families in file order, which happen to be
the highest-accuracy ones.  A partial panel is not a small panel.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = "/root/autodl-tmp/phase1_20260910"
BASE = "/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01/MrProBM.jsonl"
ARMS = {"amp4x_1p13": 1.13, "amp4x_1p30": 1.30}      # arm name -> prescribed plateau
REF_PLATEAU = 1.00
PANEL = 350
LONG, SHORT = 16384, 4096


def load(p):
    out = {}
    if not os.path.exists(p):
        return None
    for ln in open(p):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if "row_id" in d:
            out[d["row_id"]] = d
    return out


def paired(arm, base, cap=None):
    ids = [i for i in sorted(set(arm) & set(base))
           if cap is None or arm[i].get("length_cap") == cap]
    if not ids:
        return None
    d = np.array([arm[i]["correct"] - base[i]["correct"] for i in ids], float)
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    w = int((d > 0).sum()); l = int((d < 0).sum()); t_ = int((d == 0).sum())
    return dict(n=len(ids), acc=float(np.mean([arm[i]["correct"] for i in ids])),
                base_acc=float(np.mean([base[i]["correct"] for i in ids])),
                delta=float(d.mean()), se=se,
                t=(float(d.mean() / se) if se and se > 0 else float("nan")),
                w=w, l=l, tie=t_,
                eos=float(np.mean([bool(arm[i].get("ended_eos")) for i in ids])))


def main() -> int:
    base = load(BASE)
    if not base:
        print(f"REFUSING: baseline missing at {BASE}")
        return 2
    if len(base) < PANEL:
        print(f"REFUSING: baseline has {len(base)}/{PANEL} rows")
        return 2

    arms = {}
    for name in ARMS:
        a = load(f"{ROOT}/olmo_amp4x/{name}.jsonl")
        if a:
            arms[name] = a
    if not arms:
        print("REFUSING: no amp4x arms found yet")
        return 2

    print("=" * 78)
    print(f"4x AMPLITUDE PRESCRIPTION   baseline = deployed table (m_p = {REF_PLATEAU})")
    print(f"panel: {len(base)} rows, caps {sorted({r.get('length_cap') for r in base.values()})}")
    print("convention: `correct` higher is better, so delta > 0 favours the ARM")
    print("=" * 78)

    res = {}
    for name, mp in sorted(ARMS.items(), key=lambda kv: kv[1]):
        a = arms.get(name)
        if a is None:
            print(f"\n{name} (m_p={mp}): not started")
            continue
        ids = set(a) & set(base)
        if len(ids) < PANEL:
            print(f"\n{name} (m_p={mp}): REFUSING to report on {len(ids)}/{PANEL} rows "
                  f"-- a partial panel is not a small panel")
            continue
        print(f"\n{name}  (prescribed plateau m_p = {mp})")
        row = {}
        for cap in (LONG, SHORT):
            r = paired(a, base, cap)
            if r is None:
                continue
            row[cap] = r
            who = name if r["delta"] > 0 else ("baseline" if r["delta"] < 0 else "tie")
            print(f"  @{cap:<6} n={r['n']:<4} arm={r['acc']:.4f} base={r['base_acc']:.4f}  "
                  f"delta={r['delta']*100:+.2f}pp  t={r['t']:+.2f}  W/L/T={r['w']}/{r['l']}/{r['tie']}"
                  f"  -> {who}   EOS={r['eos']:.2%}")
        res[name] = row

    # ---- verdict, in the order the prereg fixes -----------------------------
    def d1(name, cap):
        r = res.get(name, {}).get(cap)
        return None if r is None else r
    print("\n" + "-" * 78)
    if "amp4x_1p13" in res:
        r = d1("amp4x_1p13", LONG)
        if r is None:
            print("VERDICT: 1.13 arm has no 16384 rows.")
        elif r["delta"] >= 0.01 and r["t"] >= 2:
            print("CRIT-1 HOLDS: the prescribed 1.13 beats the deployed 1.00 "
                  f"({r['delta']*100:+.2f}pp, t={r['t']:+.2f}).")
        elif abs(r["delta"]) <= 0.01:
            print("CRIT-2: no resolution -- the amplitude axis is flat over [1.00, 1.13].")
        elif r["delta"] <= -0.01 and r["t"] <= -2:
            print("CRIT-3 FALSIFIED: the prescribed 1.13 is significantly WORSE "
                  f"({r['delta']*100:+.2f}pp, t={r['t']:+.2f}).")
        else:
            print(f"UNRESOLVED on the primary: delta={r['delta']*100:+.2f}pp, t={r['t']:+.2f}.")
        if "amp4x_1p30" in res:
            r13, r30 = d1("amp4x_1p13", LONG), d1("amp4x_1p30", LONG)
            if r13 and r30 and r30["delta"] > r13["delta"]:
                print("CRIT-4: 1.30 beats 1.13 -- the optimum is still higher; keep climbing.")
        s = d1("amp4x_1p13", SHORT)
        if s is not None:
            print(f"\nMANDATORY 4096 REPORT: delta={s['delta']*100:+.2f}pp t={s['t']:+.2f}. "
                  "The family should pay in-window for out-of-window reach; a rise here "
                  "would instead undercut the tradeoff reading.")
    else:
        print("VERDICT: 1.13 arm not complete.")
    print("-" * 78)
    print("criteria: AMP4X_PRESCRIPTION_PREREG_20260911.md section 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
