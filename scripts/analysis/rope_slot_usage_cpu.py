"""CPU-only theory computations for the fixed-table interval problem.

No CUDA: every function runs on CPU; run with CUDA_VISIBLE_DEVICES="" to enforce.

Subcommands:
  usage   --model <dir> --out <json>     per-slot rotary-pair weight engagement
  census  --model <dir> --frozen <dir> --usage a.json,b.json --out <json>
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

B = 500000.0
K = 64
L_NATIVE = 8192
LENGTHS = [16384, 32768, 49152, 65536]
HALF = 32  # slow-slot threshold in native-window turns

# measured panel scores (task-macro official %), owner: 4080_FIXED_TABLE_RANGE_RESULT_20260913.md
SCORES = {
    "BM_g8":                {8192: 95.83, 16384: 85.42, 32768: 97.92, 49152: 95.83, 65536: 72.92},
    "MrPro_g8":             {8192: 91.67, 16384: 95.83, 32768: 87.50, 49152: 100.00, 65536: 68.75},
    "BetaSym_gamma3_g8":    {8192: 95.83, 16384: 87.50, 32768: 95.83, 49152: 95.83, 65536: 60.42},
    "RangeBridge50_g8":     {8192: 95.83, 16384: 89.58, 32768: 87.50, 49152: 91.67, 65536: 56.25},
    "BM_g8_RangeGain":      {8192: 95.83, 16384: 89.58, 32768: 91.67, 49152: 91.67, 65536: 75.00},
    "SolverProfile":        {8192: 95.83, 16384: 97.92, 32768: 100.00, 49152: 95.83, 65536: 85.42},
    "SolverProfileBandRemap": {8192: 97.92, 16384: 97.92, 32768: 97.92, 49152: 87.50, 65536: 45.83},
}


def native_freqs():
    return (B ** (-np.arange(K) / K)).astype(np.float64)


def m_from_values(values):
    wn = native_freqs()
    v = np.asarray(values, dtype=np.float64)
    return -np.log(v / wn) / math.log(8.0)


def turns_native():
    return L_NATIVE * native_freqs() / (2 * math.pi)


def usage_profile(model_dir, out_json):
    import torch
    torch.set_num_threads(8)
    from safetensors import safe_open
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir)
    n_layers = cfg.num_hidden_layers
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_q
    hd = cfg.hidden_size // n_q
    half = hd // 2
    grp = n_q // n_kv
    assert half == K, (hd, half)

    Rq = np.zeros((n_layers, n_q, K))
    Rk = np.zeros((n_layers, n_kv, K))
    seen = set()
    for fp in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(fp, framework="pt", device="cpu") as f:
            for key in f.keys():
                if ".self_attn.q_proj.weight" not in key and ".self_attn.k_proj.weight" not in key:
                    continue
                l = int(key.split(".")[2])
                W = f.get_tensor(key).float().numpy()
                n_h = W.shape[0] // hd
                Wv = W.reshape(n_h, hd, -1)
                a = np.linalg.norm(Wv[:, :half, :], axis=2)
                b = np.linalg.norm(Wv[:, half:, :], axis=2)
                r = np.sqrt(a * a + b * b)  # [heads, K] pair engagement per head
                if ".q_proj." in key:
                    Rq[l] = r
                else:
                    Rk[l] = r
                seen.add(l)
    assert len(seen) == n_layers, seen
    # GQA-aware mean over q heads of (q-pair norm * its kv head's k-pair norm)
    u = np.zeros((n_layers, K))
    for l in range(n_layers):
        h_idx = np.arange(n_q)
        u[l] = (Rq[l][h_idx] * Rk[l][h_idx // grp]).mean(axis=0)
    U = u.mean(axis=0)
    out = {
        "model": model_dir,
        "pairing": "half",
        "head_dim": hd,
        "n_layers": n_layers,
        "n_q_heads": n_q,
        "n_kv_heads": n_kv,
        "note": "data-free weight engagement proxy for slot pair (i, i+half); "
                "magnitude only, no phase structure; OLMo q/k-norm rescales per head",
        "U_mean_per_layer": u.mean(axis=1).tolist(),
        "U_slot_mean": U.tolist(),
        "U_slot_per_layer": u.tolist(),
    }
    json.dump(out, open(out_json, "w"), indent=1)
    return out


def load_tables(model_dir, frozen_dir):
    from transformers import AutoConfig
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import table_for_config

    cfg = AutoConfig.from_pretrained(model_dir)
    tabs = {}
    for arm in ["BM_g8", "MrPro_g8", "BetaSym_gamma3_g8", "RangeBridge50_g8", "BM_g8_RangeGain"]:
        try:
            tabs[arm] = np.array(table_for_config(cfg, arm)["values_float32"], dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] arm {arm}: {e}")
    for name, fp in [("SolverProfile", "SolverProfile_g8_table.json"),
                     ("SolverProfileBandRemap", "SolverProfileBandRemap_g8_table.json")]:
        tabs[name] = np.array(json.load(open(os.path.join(frozen_dir, fp)))["table"]["values_float32"],
                              dtype=np.float64)
    return tabs


def divergence(ta, tb, wn):
    """Per-slot divergence schedule of table b vs table a."""
    da = ta / wn - 1.0  # alpha_a - 1
    db = tb / wn - 1.0
    dal = db - da
    xs = np.full(K, np.inf)
    nz = np.abs(dal) > 1e-12
    xs[nz] = 1.0 / (wn[nz] * np.abs(dal[nz]))
    return dal, xs


def coherence_by_length(ta, tb, wn):
    dal, xs = divergence(ta, tb, wn)
    moved = np.where(np.abs(dal) > 1e-12)[0]
    out = {}
    for X in [8192] + LENGTHS:
        dp = X * wn * np.abs(dal)
        calm = [int(i) for i in moved if dp[i] < 1.0]
        dab = float(np.sqrt(4 * np.sum(np.sin(X * wn * dal / 2) ** 2)))
        out[X] = {"D_AB": round(dab, 3), "coherent_moved_slots": calm,
                  "max_dphi_rad": round(float(dp[moved].max()), 1) if len(moved) else 0.0}
    return out, xs


def census(model_dir, frozen_dir, usage_paths, out_json):
    wn = native_freqs()
    t_nat = turns_native()
    slow = t_nat < 2.0  # coverage-carrying slots (<2 native turns)
    tabs = load_tables(model_dir, frozen_dir)
    usages = {}
    for p in usage_paths:
        d = json.load(open(p))
        usages[os.path.basename(p).split("_U_slot")[0]] = np.array(d["U_slot_mean"])

    rep = {"slow_slots": int(slow.sum()), "tables": {}}
    for name, v in tabs.items():
        m = m_from_values(v)
        al = v / wn
        d = {"min_m": round(float(m.min()), 4), "max_m": round(float(m.max()), 4),
             "band_low": int(np.argmax(m > 1e-9)), "n_full": int((m > 0.999).sum())}
        for X in LENGTHS:
            E = al * X / L_NATIVE
            d[f"exposure_{X}"] = {
                "n_slow_overexp": int(((E > 1.05) & slow).sum()),
                "slow_overexposure_sum": round(float(np.sum(np.maximum(E - 1, 0)[slow])), 3),
                "slow_overexp_slots": [int(i) for i in np.where((E > 1.05) & slow)[0]],
            }
        rep["tables"][name] = d

    rep["pairwise"] = {}
    for a, b in [("BM_g8", "MrPro_g8"), ("BM_g8", "BetaSym_gamma3_g8"),
                 ("SolverProfile", "SolverProfileBandRemap"), ("BM_g8", "SolverProfile")]:
        if a not in tabs or b not in tabs:
            continue
        coh, xs = coherence_by_length(tabs[a], tabs[b], wn)
        moved = np.where(np.abs(coh[8192]["D_AB"] * 0 + (tabs[b] / wn - tabs[a] / wn)) > 1e-12)[0]
        rep["pairwise"][f"{a} vs {b}"] = {
            "n_moved": int(len(moved)),
            "coherence_by_length": {str(k): v for k, v in coh.items()},
            "X_star_by_slot": {int(i): (round(float(xs[i]), 0) if np.isfinite(xs[i]) else None)
                                for i in moved},
        }

    rep["scores_vs_structure"] = {}
    for name, sc in SCORES.items():
        if name not in tabs:
            continue
        al = tabs[name] / wn
        E64 = al * 65536 / L_NATIVE
        rep["scores_vs_structure"][name] = {
            "score_65536": sc[65536], "score_49152": sc[49152],
            "n_full": int((m_from_values(tabs[name]) > 0.999).sum()),
            "slow_overexp_65536": round(float(np.sum(np.maximum(E64 - 1, 0)[slow])), 3),
        }
    json.dump(rep, open(out_json, "w"), indent=1, default=str)

    print("== exposure census (slow slots = <2 native turns, coverage carriers) ==")
    hdr = f"{'arm':>24} {'n_full(m=1)':>11} {'slow>exp@16K':>12} {'@32K':>6} {'@48K':>6} {'@64K':>6} {'sumE-1@64K':>10} {'64K score':>9}"
    print(hdr)
    for name, d in rep["tables"].items():
        ex = [d[f"exposure_{X}"]["n_slow_overexp"] for X in LENGTHS]
        s64 = rep["scores_vs_structure"].get(name, {}).get("score_65536", "-")
        print(f"{name:>24} {d['n_full']:>11} {ex[0]:>12} {ex[1]:>6} {ex[2]:>6} {ex[3]:>6} "
              f"{d['exposure_65536']['slow_overexposure_sum']:>10} {s64:>9}")

    print("\n== divergence schedules vs BM_g8 (coherent moved slots by length) ==")
    for key, d in rep["pairwise"].items():
        print(f"-- {key}: n_moved={d['n_moved']}")
        for X in [8192] + LENGTHS:
            c = d["coherence_by_length"][str(X)]
            print(f"   X={X:>6}: D_AB={c['D_AB']:>6} coherent={c['coherent_moved_slots']}")

    if usages:
        print("\n== usage (weight engagement) ==")
        for nm, U in usages.items():
            order = np.argsort(-U)
            print(f"-- {nm}: top5 slots={order[:5].tolist()}, bottom5={order[-5:].tolist()}")
        if len(usages) == 2:
            a, b = list(usages.values())
            rho = float(np.corrcoef(np.log(a), np.log(b))[0, 1])
            print(f"-- OLMo vs Llama log-usage pearson r = {rho:.3f}")
        print("\n== usage-weighted load share of table contrasts ==")
        U = usages.get("llama_U", list(usages.values())[0])
        for a, b in [("SolverProfile", "SolverProfileBandRemap"), ("BM_g8", "BetaSym_gamma3_g8"),
                     ("BM_g8", "MrPro_g8")]:
            if a not in tabs or b not in tabs:
                continue
            dal = tabs[b] / wn - tabs[a] / wn
            moved = np.abs(dal) > 1e-12
            for X in [8192, 65536]:
                load = X * wn * np.abs(dal) * U
                tot = load[moved].sum()
                fast = load[moved & (t_nat >= HALF)].sum() / tot
                deep = load[moved & slow].sum() / tot
                mid = 1.0 - fast - deep
                print(f"-- {a} vs {b} @X={X}: load share fast(>=2t)={fast:.2f} mid={mid:.2f} slow(<2t)={deep:.2f}")
    return rep


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pu = sub.add_parser("usage")
    pu.add_argument("--model", required=True)
    pu.add_argument("--out", required=True)
    pc = sub.add_parser("census")
    pc.add_argument("--model", required=True)
    pc.add_argument("--frozen", required=True)
    pc.add_argument("--usage", default="")
    pc.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "usage":
        usage_profile(a.model, a.out)
        print("wrote", a.out)
    else:
        ups = [p for p in a.usage.split(",") if p]
        census(a.model, a.frozen, ups, a.out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
