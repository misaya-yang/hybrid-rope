#!/usr/bin/env python3
"""Phase 8F: Multi-Seed Statistical Verification of 8E headline results.

3 new seeds (137/256/314) × 3 methods (Geo/EVQ τ=1.0/Hybrid τ=1.0)
= 9 new runs. Seed 42 reuses 8C/8E existing results.

Config: 350M, from-scratch 4K, 50M tokens, lr=6e-4, batch=2.
"""

import sys, os, json, math, time, hashlib
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path
from scipy import stats
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evq_sweep import (
    GPT, RotaryEmbedding, evq_cosh_inv_freq,
    TIER_CONFIGS, DEVICE, DTYPE, USE_AUTOCAST,
    eval_model, train_model, set_seed, load_data, load_val,
)
from eval_passkey_scratch import eval_passkey_nll_gap

import torch

BASE = 500000.0
DIM = 64
WORK = Path("/root/autodl-tmp/evq_phase8/multi_seed")
SCRATCH_DIR = Path("/root/autodl-tmp/evq_phase8/from_scratch_4k")
CFG_350M = TIER_CONFIGS["350m"].copy()

SEEDS = [42, 137, 256, 314]
METHODS = ["geo", "evq1.0", "hybrid1.0"]

EVAL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
PK_LENGTHS = [1024, 2048, 4096, 8192]
PK_TRIALS = 100  # per length, total 400 per run


def geometric_inv_freq(dim=DIM, base=BASE):
    return evq_cosh_inv_freq(dim, 0.0, base)


def hybrid_evq_inv_freq(dim=DIM, base=BASE, tau=1.0, n_geometric_high=8):
    """Hybrid EVQ: high-freq channels keep Geometric, low-freq use EVQ-cosh."""
    n = dim // 2  # 32
    geo = geometric_inv_freq(dim, base).double()
    n_evq = n - n_geometric_high  # 24
    theta_max_low = geo[n_geometric_high].item()
    theta_min_low = geo[-1].item()
    u = torch.arange(n_evq, dtype=torch.float64) / (n_evq - 1)
    if abs(tau) < 1e-8:
        phi = 1.0 - u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    evq_part = (theta_min_low ** phi) * (theta_max_low ** (1.0 - phi))
    inv_freq = torch.cat([geo[:n_geometric_high], evq_part])
    return inv_freq.float()


def get_inv_freq(method):
    if method == "geo":
        return geometric_inv_freq()
    elif method == "evq1.0":
        return evq_cosh_inv_freq(DIM, 1.0, BASE)
    elif method == "hybrid1.0":
        return hybrid_evq_inv_freq(DIM, BASE, tau=1.0, n_geometric_high=8)
    else:
        raise ValueError(f"Unknown method: {method}")


def method_desc(method):
    if method == "geo":
        return "Geometric"
    elif method == "evq1.0":
        return "EVQ τ=1.0"
    elif method == "hybrid1.0":
        return "Hybrid τ=1.0"
    return method


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_seed42_results():
    """Load existing seed=42 results from 8C/8E."""
    results = {}

    # Geo from 8C
    geo_file = SCRATCH_DIR / "geo_4k" / "result.json"
    if geo_file.exists():
        with open(geo_file) as f:
            results["geo"] = json.load(f)
        print(f"  [REUSE] seed=42 geo from 8C")

    # EVQ τ=1.0 from 8E
    evq_file = SCRATCH_DIR / "evq1.0_4k" / "result.json"
    if evq_file.exists():
        with open(evq_file) as f:
            results["evq1.0"] = json.load(f)
        print(f"  [REUSE] seed=42 evq1.0 from 8E")

    # Hybrid τ=1.0 from 8E
    hybrid_file = SCRATCH_DIR / "hybrid1.0_4k" / "result.json"
    if hybrid_file.exists():
        with open(hybrid_file) as f:
            results["hybrid1.0"] = json.load(f)
        print(f"  [REUSE] seed=42 hybrid1.0 from 8E")

    return results


def run_single(seed, method, cfg, train_data, val_data, filler, tok):
    """Train and eval a single run."""
    run_name = f"seed{seed}_{method}"
    run_dir = WORK / f"seed{seed}" / f"{method}_4k"
    result_file = run_dir / "result.json"

    if result_file.exists():
        print(f"\n  [SKIP] {run_name} — already done")
        with open(result_file) as f:
            return json.load(f)

    print(f"\n{'─'*60}")
    print(f"  {run_name}: {method_desc(method)} seed={seed}")
    print(f"{'─'*60}")

    inv_freq = get_inv_freq(method)

    set_seed(seed)
    model = GPT(cfg, inv_freq).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params: {n_params:.1f}M")

    t0 = time.time()
    model = train_model(model, train_data, cfg, seed=seed)
    train_time = time.time() - t0
    print(f"  Train time: {train_time/60:.1f} min")

    # PPL eval
    ppl = eval_model(model, val_data, EVAL_LENGTHS, 10)

    # Passkey eval
    print(f"  Passkey eval ({PK_TRIALS} trials/length)...")
    pk = eval_passkey_nll_gap(model, tok, filler,
                              lengths=PK_LENGTHS, depths=[0.5],
                              num_trials=PK_TRIALS)
    g = pk.get("global", {})
    print(f"    retrieval={g.get('retrieval_rate','?')}  gap={g.get('mean_nll_gap','?')}")

    # Save
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")

    res = {
        "method": method_desc(method),
        "seed": seed,
        "ppl": ppl,
        "passkey_global": pk.get("global", {}),
        "passkey_summary": pk.get("summary", {}),
        "train_time_sec": round(train_time, 1),
    }
    _save_json(result_file, res)

    del model
    torch.cuda.empty_cache()
    return res


def compute_statistics(all_results):
    """Compute mean, std, p-values for all methods."""
    stats_out = {}

    for method in METHODS:
        ppl_16k_vals = []
        pk_global_vals = []
        pk_by_length = {str(L): [] for L in PK_LENGTHS}
        ppl_all = {str(L): [] for L in EVAL_LENGTHS}

        for seed in SEEDS:
            res = all_results[seed][method]
            ppl = res.get("ppl", {})
            pk_g = res.get("passkey_global", {})
            pk_s = res.get("passkey_summary", {})

            ppl_16k = ppl.get("16384", None)
            if ppl_16k is not None:
                ppl_16k_vals.append(float(ppl_16k))

            pk_rate = pk_g.get("retrieval_rate", None)
            if pk_rate is not None:
                pk_global_vals.append(float(pk_rate))

            for L in EVAL_LENGTHS:
                v = ppl.get(str(L), None)
                if v is not None:
                    ppl_all[str(L)].append(float(v))

            for L in PK_LENGTHS:
                key = f"L={L}_d=0.5"
                if key in pk_s:
                    pk_by_length[str(L)].append(float(pk_s[key].get("retrieval_rate", 0)))

        stats_out[method] = {
            "ppl_16k": {
                "values": ppl_16k_vals,
                "mean": round(np.mean(ppl_16k_vals), 2) if ppl_16k_vals else None,
                "std": round(np.std(ppl_16k_vals, ddof=1), 2) if len(ppl_16k_vals) > 1 else None,
            },
            "pk_global": {
                "values": pk_global_vals,
                "mean": round(np.mean(pk_global_vals), 4) if pk_global_vals else None,
                "std": round(np.std(pk_global_vals, ddof=1), 4) if len(pk_global_vals) > 1 else None,
            },
            "pk_by_length": {},
            "ppl_all": {},
        }

        for L in PK_LENGTHS:
            vals = pk_by_length[str(L)]
            stats_out[method]["pk_by_length"][str(L)] = {
                "values": vals,
                "mean": round(np.mean(vals), 4) if vals else None,
                "std": round(np.std(vals, ddof=1), 4) if len(vals) > 1 else None,
            }

        for L in EVAL_LENGTHS:
            vals = ppl_all[str(L)]
            stats_out[method]["ppl_all"][str(L)] = {
                "mean": round(np.mean(vals), 2) if vals else None,
                "std": round(np.std(vals, ddof=1), 2) if len(vals) > 1 else None,
            }

    # Statistical tests
    tests = {}

    geo_ppl = stats_out["geo"]["ppl_16k"]["values"]
    geo_pk = stats_out["geo"]["pk_global"]["values"]

    for method in ["evq1.0", "hybrid1.0"]:
        m_ppl = stats_out[method]["ppl_16k"]["values"]
        m_pk = stats_out[method]["pk_global"]["values"]

        # Paired t-test for PPL@16K
        if len(geo_ppl) >= 2 and len(m_ppl) >= 2:
            diff_ppl = [m - g for m, g in zip(m_ppl, geo_ppl)]
            t_stat_ppl, p_ppl = stats.ttest_rel(m_ppl, geo_ppl)
            tests[f"{method}_vs_geo_ppl16k"] = {
                "paired_diffs": [round(d, 2) for d in diff_ppl],
                "mean_diff": round(np.mean(diff_ppl), 2),
                "t_stat": round(t_stat_ppl, 4),
                "p_value": round(p_ppl, 6),
            }

        # Paired t-test for passkey global
        if len(geo_pk) >= 2 and len(m_pk) >= 2:
            diff_pk = [m - g for m, g in zip(m_pk, geo_pk)]
            t_stat_pk, p_pk = stats.ttest_rel(m_pk, geo_pk)
            tests[f"{method}_vs_geo_pk"] = {
                "paired_diffs": [round(d, 4) for d in diff_pk],
                "mean_diff": round(np.mean(diff_pk), 4),
                "t_stat": round(t_stat_pk, 4),
                "p_value": round(p_pk, 6),
            }

        # Pooled passkey: chi-squared test (4 seeds × 400 trials = 1600)
        n_total = len(geo_pk) * 400  # 4 × 400 = 1600
        geo_successes = round(np.mean(geo_pk) * n_total)
        m_successes = round(np.mean(m_pk) * n_total)

        # 2x2 contingency table
        table = [[m_successes, n_total - m_successes],
                 [geo_successes, n_total - geo_successes]]
        chi2, p_chi2, dof, expected = stats.chi2_contingency(table)

        # Cohen's h
        p1 = np.mean(m_pk)
        p2 = np.mean(geo_pk)
        cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

        tests[f"{method}_vs_geo_pk_pooled"] = {
            "method_pooled_rate": round(p1, 4),
            "geo_pooled_rate": round(p2, 4),
            "chi2": round(chi2, 4),
            "p_value_chi2": round(p_chi2, 6),
            "cohens_h": round(cohens_h, 4),
            "n_total": n_total,
        }

    return stats_out, tests


def determine_verdict(stats_out, tests):
    """Determine strong/weak confirm or fail."""
    evq_pk_mean = stats_out["evq1.0"]["pk_global"]["mean"]
    geo_pk_mean = stats_out["geo"]["pk_global"]["mean"]
    hybrid_pk_mean = stats_out["hybrid1.0"]["pk_global"]["mean"]
    hybrid_ppl_mean = stats_out["hybrid1.0"]["ppl_16k"]["mean"]
    geo_ppl_mean = stats_out["geo"]["ppl_16k"]["mean"]

    evq_pk_p = tests.get("evq1.0_vs_geo_pk", {}).get("p_value", 1.0)
    hybrid_pk_p = tests.get("hybrid1.0_vs_geo_pk", {}).get("p_value", 1.0)

    # Check direction consistency
    evq_diffs = tests.get("evq1.0_vs_geo_pk", {}).get("paired_diffs", [])
    all_positive_evq = all(d > 0 for d in evq_diffs) if evq_diffs else False

    hybrid_diffs_ppl = tests.get("hybrid1.0_vs_geo_ppl16k", {}).get("paired_diffs", [])
    hybrid_diffs_pk = tests.get("hybrid1.0_vs_geo_pk", {}).get("paired_diffs", [])

    verdict_lines = []

    # Strong confirm criteria
    if evq_pk_mean and geo_pk_mean and evq_pk_mean > geo_pk_mean and evq_pk_p < 0.05:
        verdict_lines.append(f"STRONG CONFIRM: EVQ τ=1.0 PK ({evq_pk_mean:.1%}) > Geo PK ({geo_pk_mean:.1%}), p={evq_pk_p:.4f}")

    if (hybrid_ppl_mean and geo_ppl_mean and hybrid_ppl_mean < geo_ppl_mean and
        hybrid_pk_mean and geo_pk_mean and hybrid_pk_mean > geo_pk_mean):
        verdict_lines.append(f"STRONG CONFIRM: Hybrid dual-win (PPL {hybrid_ppl_mean:.1f} < {geo_ppl_mean:.1f}, PK {hybrid_pk_mean:.1%} > {geo_pk_mean:.1%})")

    # Weak confirm
    if not verdict_lines:
        if evq_pk_mean and geo_pk_mean and evq_pk_mean > geo_pk_mean:
            if all_positive_evq:
                verdict_lines.append(f"WEAK CONFIRM: EVQ PK > Geo (4/4 seeds consistent), p={evq_pk_p:.4f}")
            else:
                n_pos = sum(1 for d in evq_diffs if d > 0)
                verdict_lines.append(f"WEAK CONFIRM: EVQ PK > Geo ({n_pos}/4 seeds), p={evq_pk_p:.4f}")

    # Fail
    if not verdict_lines:
        if evq_pk_mean and geo_pk_mean and evq_pk_mean <= geo_pk_mean:
            verdict_lines.append(f"FAIL: EVQ PK ({evq_pk_mean:.1%}) ≤ Geo PK ({geo_pk_mean:.1%})")

    if not verdict_lines:
        verdict_lines.append("INCONCLUSIVE")

    return "\n".join(verdict_lines)


def main():
    print(f"\n{'#'*60}")
    print(f"  PHASE 8F: Multi-Seed Statistical Verification")
    print(f"  Seeds: {SEEDS}")
    print(f"  Methods: {METHODS}")
    print(f"  Config: 350M, from-scratch 4K, 50M tokens")
    print(f"{'#'*60}")

    SEQ = 4096
    TOKENS = 50_000_000
    LR = 6e-4
    BATCH = 2

    cfg = CFG_350M.copy()
    cfg["seq_len"] = SEQ
    cfg["max_position_embeddings"] = SEQ
    cfg["train_tokens"] = TOKENS
    cfg["lr"] = LR
    cfg["batch_size"] = BATCH

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Load data (reuse 8C/8E cache)
    data_dir = SCRATCH_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_data = load_data(tok, TOKENS, SEQ, "fineweb-edu", cache_dir=str(data_dir))
    val_data = load_val(tok, 5_000_000, "fineweb-edu", cache_dir=str(data_dir))
    filler = val_data[:50000]

    # Collect all results
    all_results = {}

    # Seed 42: reuse existing results
    print(f"\n{'='*60}")
    print(f"  Seed 42: Reusing 8C/8E results")
    print(f"{'='*60}")
    seed42_results = load_seed42_results()
    all_results[42] = seed42_results

    # Copy seed 42 results to multi_seed dir for consistency
    for method, res in seed42_results.items():
        out_dir = WORK / "seed42" / f"{method}_4k"
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_json(out_dir / "result.json", res)

    # Run order: all Geo first, then EVQ, then Hybrid (per prompt spec)
    for method in METHODS:
        for seed in [137, 256, 314]:
            res = run_single(seed, method, cfg, train_data, val_data, filler, tok)
            if seed not in all_results:
                all_results[seed] = {}
            all_results[seed][method] = res

            # Print running comparison after each seed completes all methods
            if method == "hybrid1.0":
                geo_pk = all_results[seed]["geo"]["passkey_global"].get("retrieval_rate", "?")
                evq_pk = all_results[seed]["evq1.0"]["passkey_global"].get("retrieval_rate", "?")
                hyb_pk = all_results[seed]["hybrid1.0"]["passkey_global"].get("retrieval_rate", "?")
                print(f"\n  >>> Seed {seed} summary: Geo PK={geo_pk}, EVQ PK={evq_pk}, Hybrid PK={hyb_pk}")

    # Statistics
    print(f"\n{'#'*60}")
    print(f"  STATISTICAL ANALYSIS")
    print(f"{'#'*60}")

    stats_out, tests = compute_statistics(all_results)
    verdict = determine_verdict(stats_out, tests)

    # Build JSON output
    per_seed = {}
    for seed in SEEDS:
        per_seed[f"seed_{seed}"] = {}
        for method in METHODS:
            res = all_results[seed][method]
            per_seed[f"seed_{seed}"][method] = {
                "ppl": res.get("ppl", {}),
                "passkey_global": res.get("passkey_global", {}),
                "passkey_summary": res.get("passkey_summary", {}),
            }

    f8_data = {
        "purpose": "Statistical verification of 8E headline results",
        "model": "350M (head_dim=64)",
        "train_config": "from-scratch 4K, 50M tokens, lr=6e-4",
        "seeds": SEEDS,
        "per_seed_results": per_seed,
        "aggregated": {
            method: {
                "ppl_16k_mean": stats_out[method]["ppl_16k"]["mean"],
                "ppl_16k_std": stats_out[method]["ppl_16k"]["std"],
                "pk_global_mean": stats_out[method]["pk_global"]["mean"],
                "pk_global_std": stats_out[method]["pk_global"]["std"],
                "ppl_all": stats_out[method]["ppl_all"],
                "pk_by_length": stats_out[method]["pk_by_length"],
            }
            for method in METHODS
        },
        "statistical_tests": tests,
        "verdict": verdict,
    }

    # Save standalone
    WORK.mkdir(parents=True, exist_ok=True)
    _save_json(WORK / "results_8f.json", f8_data)
    print(f"\n  Saved: {WORK / 'results_8f.json'}")

    # Append to results_phase8.json
    results_file = Path("/root/autodl-tmp/evq_phase8/results_phase8.json")
    if results_file.exists():
        with open(results_file) as f:
            master = json.load(f)
    else:
        master = {"experiments": {}}

    if "experiments" not in master:
        master["experiments"] = {}
    master["experiments"]["8F_multi_seed_verification"] = f8_data
    _save_json(results_file, master)
    print(f"  Updated: {results_file}")

    # Print summary tables
    print(f"\n{'='*60}")
    print(f"  8F RESULTS: Multi-Seed Verification")
    print(f"{'='*60}")

    # PPL table
    print(f"\n  PPL@16K by seed:")
    print(f"  {'Seed':>6s} {'Geo':>10s} {'EVQ τ=1.0':>10s} {'Hybrid':>10s}")
    for seed in SEEDS:
        geo_ppl = all_results[seed]["geo"]["ppl"].get("16384", "?")
        evq_ppl = all_results[seed]["evq1.0"]["ppl"].get("16384", "?")
        hyb_ppl = all_results[seed]["hybrid1.0"]["ppl"].get("16384", "?")
        print(f"  {seed:>6d} {geo_ppl:>10} {evq_ppl:>10} {hyb_ppl:>10}")

    print(f"  {'Mean':>6s} {stats_out['geo']['ppl_16k']['mean']:>10} "
          f"{stats_out['evq1.0']['ppl_16k']['mean']:>10} "
          f"{stats_out['hybrid1.0']['ppl_16k']['mean']:>10}")
    print(f"  {'Std':>6s} {stats_out['geo']['ppl_16k']['std']:>10} "
          f"{stats_out['evq1.0']['ppl_16k']['std']:>10} "
          f"{stats_out['hybrid1.0']['ppl_16k']['std']:>10}")

    # Passkey table
    print(f"\n  Passkey Global by seed:")
    print(f"  {'Seed':>6s} {'Geo':>10s} {'EVQ τ=1.0':>10s} {'Hybrid':>10s}")
    for seed in SEEDS:
        geo_pk = all_results[seed]["geo"]["passkey_global"].get("retrieval_rate", "?")
        evq_pk = all_results[seed]["evq1.0"]["passkey_global"].get("retrieval_rate", "?")
        hyb_pk = all_results[seed]["hybrid1.0"]["passkey_global"].get("retrieval_rate", "?")
        print(f"  {seed:>6d} {geo_pk:>10} {evq_pk:>10} {hyb_pk:>10}")

    print(f"  {'Mean':>6s} {stats_out['geo']['pk_global']['mean']:>10} "
          f"{stats_out['evq1.0']['pk_global']['mean']:>10} "
          f"{stats_out['hybrid1.0']['pk_global']['mean']:>10}")
    print(f"  {'Std':>6s} {stats_out['geo']['pk_global']['std']:>10} "
          f"{stats_out['evq1.0']['pk_global']['std']:>10} "
          f"{stats_out['hybrid1.0']['pk_global']['std']:>10}")

    # Passkey by length
    print(f"\n  Passkey by length (mean ± std):")
    print(f"  {'Length':>6s} {'Geo':>15s} {'EVQ τ=1.0':>15s} {'Hybrid':>15s}")
    for L in PK_LENGTHS:
        for method in METHODS:
            s = stats_out[method]["pk_by_length"][str(L)]
        geo_s = stats_out["geo"]["pk_by_length"][str(L)]
        evq_s = stats_out["evq1.0"]["pk_by_length"][str(L)]
        hyb_s = stats_out["hybrid1.0"]["pk_by_length"][str(L)]
        print(f"  {L:>6d} {geo_s['mean']:.0%}±{geo_s['std']:.1%}  "
              f"{evq_s['mean']:.0%}±{evq_s['std']:.1%}  "
              f"{hyb_s['mean']:.0%}±{hyb_s['std']:.1%}")

    # Statistical tests
    print(f"\n  Statistical Tests:")
    for name, test in tests.items():
        if "p_value" in test:
            print(f"    {name}: p={test['p_value']:.6f}")
        if "p_value_chi2" in test:
            print(f"    {name}: chi2 p={test['p_value_chi2']:.6f}, Cohen's h={test.get('cohens_h', '?')}")

    # Verdict
    print(f"\n  {'='*40}")
    print(f"  VERDICT: {verdict}")
    print(f"  {'='*40}")

    print(f"\n  DONE!")


if __name__ == "__main__":
    main()
