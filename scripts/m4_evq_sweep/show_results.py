#!/usr/bin/env python3
"""Quick results viewer for Phase 12 r-sweep."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/evq_phase12_r_sweep/results_checkpoint.json"
with open(path) as f:
    d = json.load(f)
exps = d["experiments"]
baseline = exps.get("350m_r32_tau1.50_seed42", {}).get("ppl", {})

r42, r137, tau_at_r14 = {}, {}, {}
for k in sorted(exps.keys()):
    if "seed137" in k:
        r137[k] = exps[k]
    elif "_r" in k and "seed42" in k:
        r42[k] = exps[k]
    # tau sweep at r=14
    if "_r14_" in k and "seed42" in k:
        tau_val = float(k.split("tau")[1].split("_")[0])
        tau_at_r14[tau_val] = exps[k]

print("Total: %d runs" % len(exps))
print("  seed42 r-sweep: %d/9" % len(r42))
print("  seed137 r-sweep: %d/5" % len(r137))
print("  tau-sweep at r=14: %d points" % len(tau_at_r14))

sep = "=" * 90
sep2 = "-" * 80

# --- seed=42 r-sweep ---
print("\n" + sep)
print("  SEED=42 R-SWEEP (vs r=32 Geometric baseline)")
print(sep)
hdr = "  %3s  %7s %7s %7s %8s | %6s %7s | %6s"
print(hdr % ("r", "PPL@2K", "PPL@4K", "PPL@8K", "PPL@16K", "d@2K", "d@16K", "PK_ret"))
print("  " + sep2)
for k in sorted(r42.keys(), key=lambda x: int(x.split("_r")[1].split("_")[0])):
    v = r42[k]
    r_val = int(k.split("_r")[1].split("_")[0])
    ppl = v.get("ppl", {})
    pk = v.get("passkey", {}).get("global", {})
    ret = pk.get("retrieval_rate", 0)
    d2k = (ppl.get("2048", 0) / baseline["2048"] - 1) * 100
    d16k = (ppl.get("16384", 0) / baseline["16384"] - 1) * 100 if "16384" in ppl else 0
    tag = " <- r*" if r_val == 14 else (" <- Geo" if r_val == 32 else "")
    print("  %3d  %7.1f %7.1f %7.1f %8.1f | %+5.1f%% %+6.1f%% | %5.0f%%%s" % (
        r_val, ppl.get("2048", 0), ppl.get("4096", 0), ppl.get("8192", 0),
        ppl.get("16384", 0), d2k, d16k, ret * 100, tag))

# --- seed=137 r-sweep ---
if r137:
    b137 = r137.get("350m_r32_tau1.50_seed137", {}).get("ppl", {})
    if not b137:
        b137 = baseline
    print("\n" + sep)
    print("  SEED=137 R-SWEEP (reproducibility check)")
    print(sep)
    print(hdr % ("r", "PPL@2K", "PPL@4K", "PPL@8K", "PPL@16K", "d@2K", "d@16K", "PK_ret"))
    print("  " + sep2)
    for k in sorted(r137.keys(), key=lambda x: int(x.split("_r")[1].split("_")[0])):
        v = r137[k]
        r_val = int(k.split("_r")[1].split("_")[0])
        ppl = v.get("ppl", {})
        pk = v.get("passkey", {}).get("global", {})
        ret = pk.get("retrieval_rate", 0)
        d2k = (ppl.get("2048", 0) / b137["2048"] - 1) * 100 if b137 else 0
        d16k = (ppl.get("16384", 0) / b137["16384"] - 1) * 100 if b137 and "16384" in ppl else 0
        print("  %3d  %7.1f %7.1f %7.1f %8.1f | %+5.1f%% %+6.1f%% | %5.0f%%" % (
            r_val, ppl.get("2048", 0), ppl.get("4096", 0), ppl.get("8192", 0),
            ppl.get("16384", 0), d2k, d16k, ret * 100))

# --- tau sweep at r=14 ---
if len(tau_at_r14) > 1:
    print("\n" + sep)
    print("  TAU-SWEEP at r=14 (theory predicts tau*=1.41)")
    print(sep)
    print("  %5s  %7s %7s %7s %8s | %6s" % ("tau", "PPL@2K", "PPL@4K", "PPL@8K", "PPL@16K", "PK_ret"))
    print("  " + "-" * 60)
    for tau_val in sorted(tau_at_r14.keys()):
        v = tau_at_r14[tau_val]
        ppl = v.get("ppl", {})
        pk = v.get("passkey", {}).get("global", {})
        ret = pk.get("retrieval_rate", 0)
        mark = " <- used" if abs(tau_val - 1.5) < 0.01 else ""
        t_str = "%.1f" % tau_val
        print("  %5s  %7.1f %7.1f %7.1f %8.1f | %5.0f%%%s" % (
            t_str, ppl.get("2048", 0), ppl.get("4096", 0), ppl.get("8192", 0),
            ppl.get("16384", 0), ret * 100, mark))
