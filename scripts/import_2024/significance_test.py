#!/usr/bin/env python3
"""
Paired bootstrap significance test for 8B LongBench results.

Usage:
    python significance_test.py --data_dir <path_to_longbench_results>

Expected directory structure:
    data_dir/
      baseline/longbench/*.json    (or similar per-task files)
      pi/longbench/*.json
      yarn/longbench/*.json
      sigmoid/longbench/*.json
      anchored_sigmoid/longbench/*.json

Each task JSON should contain per-sample scores.
Adjust RESULT_LOADER below to match your actual file structure.
"""

import json
import os
import glob
import numpy as np
import argparse
from pathlib import Path

# ============================================================
# STEP 0: Configure these paths to match your repo structure
# ============================================================

# The 6 LongBench tasks from your evaluation
TASKS = [
    "multi_news", "narrativeqa", "qasper",
    "2wikimqa", "hotpotqa", "gov_report"
]

METHODS = ["baseline", "pi", "yarn", "sigmoid", "anchored_sigmoid"]


def resolve_data_dir(data_dir):
    """Resolve user-provided data_dir, including common batch_report wrappers."""
    base = Path(data_dir)
    if base.exists():
        return base

    cwd = Path.cwd()
    for batch_root in sorted(cwd.glob("batch_report_*")):
        candidate = batch_root / data_dir
        if candidate.exists():
            return candidate

    return base

def _extract_scores(obj):
    """Extract a 1D numeric score array from common result containers."""
    if obj is None:
        return None

    if isinstance(obj, list):
        if not obj:
            return None
        if all(isinstance(x, (int, float)) for x in obj):
            return np.array(obj, dtype=float)
        if all(isinstance(x, dict) for x in obj):
            for key in ["score", "f1", "rouge", "accuracy", "em"]:
                vals = [x.get(key) for x in obj if isinstance(x.get(key), (int, float))]
                if vals:
                    return np.array(vals, dtype=float)
        return None

    if isinstance(obj, dict):
        for key in ["scores", "results", "per_sample_scores"]:
            vals = obj.get(key)
            if isinstance(vals, list) and vals and all(isinstance(x, (int, float)) for x in vals):
                return np.array(vals, dtype=float)

        examples = obj.get("examples")
        if isinstance(examples, list):
            vals = [x.get("score") for x in examples if isinstance(x, dict) and isinstance(x.get("score"), (int, float))]
            if vals:
                return np.array(vals, dtype=float)

    return None


def load_per_sample_scores(data_dir, method, task):
    """
    Load per-sample scores for a given method and task.
    
    *** YOU MUST ADAPT THIS FUNCTION TO YOUR ACTUAL FILE STRUCTURE ***
    
    Look in your data directory for the actual JSON format.
    Common patterns:
      - data_dir/{method}/longbench/{task}.json  -> list of dicts with "score" key
      - data_dir/{method}/longbench/results.json -> dict with task keys
    """
    # Pattern 1: legacy per-task/per-method files
    candidates = [
        os.path.join(data_dir, method, "longbench", f"{task}.json"),
        os.path.join(data_dir, method, "longbench", f"{task}_results.json"),
        os.path.join(data_dir, method, f"longbench_{task}.json"),
    ]

    # Pattern 2: this repo's consolidated downstream LongBench outputs
    candidates.extend([
        os.path.join(data_dir, "downstream_eval_autorun", "longbench", f"{method}.json"),
        os.path.join(data_dir, "downstream_eval_parallel_seed42_m2", "longbench", f"{method}.json"),
    ])
    candidates.extend(sorted(glob.glob(
        os.path.join(data_dir, "downstream_eval_*", "longbench", f"{method}.json")
    )))

    # Keep order while de-duplicating.
    candidates = list(dict.fromkeys(candidates))
    
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # Direct extraction from root or task-keyed dict.
            scores = _extract_scores(data if not isinstance(data, dict) else data.get(task, data))
            if scores is not None and len(scores) > 0:
                return scores

            # Consolidated format:
            #   { "models": { "hybrid_lora": { "tasks": { task: {...} } } } }
            if isinstance(data, dict) and "models" in data and isinstance(data["models"], dict):
                models = data["models"]
                preferred_model_keys = [k for k in ["hybrid_lora", method] if k in models]
                preferred_model_keys += [k for k in models.keys()
                                         if k not in preferred_model_keys and k != "base_unfinetuned"]
                if "base_unfinetuned" in models:
                    preferred_model_keys.append("base_unfinetuned")

                for model_key in preferred_model_keys:
                    task_blob = models.get(model_key, {}).get("tasks", {}).get(task)
                    scores = _extract_scores(task_blob)
                    if scores is not None and len(scores) > 0:
                        declared_n = task_blob.get("num_scored") if isinstance(task_blob, dict) else None
                        if isinstance(declared_n, int) and declared_n > len(scores):
                            print(
                                f"  [WARN] {method}/{task}: extracted {len(scores)} score(s), "
                                f"but metadata reports num_scored={declared_n}. "
                                f"Current JSON appears to keep only preview examples."
                            )
                        return scores

            print(f"  [WARN] Found {path} but couldn't parse per-sample scores for {method}/{task}")
    
    # Pattern 3: single results file with all tasks
    single_file = os.path.join(data_dir, method, "longbench", "results.json")
    if os.path.exists(single_file):
        with open(single_file) as f:
            data = json.load(f)
        scores = _extract_scores(data.get(task) if isinstance(data, dict) else None)
        if scores is not None and len(scores) > 0:
            return scores
    
    print(f"  [ERROR] Cannot find per-sample scores for {method}/{task}")
    print(f"  Searched: {candidates}")
    print(f"  Please adapt load_per_sample_scores() to your file structure.")
    return None


def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000, seed=42):
    """
    Paired bootstrap test: H0: mean(A) = mean(B)
    Returns: observed_diff, p_value, ci_lower, ci_upper (95% CI of diff)
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    assert len(scores_b) == n, f"Length mismatch: {len(scores_a)} vs {len(scores_b)}"
    
    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    
    # Bootstrap
    diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diffs[i] = np.mean(scores_a[idx]) - np.mean(scores_b[idx])
    
    # Two-sided p-value
    # Under H0, center the bootstrap distribution
    centered_diffs = diffs - np.mean(diffs)
    p_value = np.mean(np.abs(centered_diffs) >= np.abs(observed_diff))
    
    # 95% CI
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    return observed_diff, p_value, ci_lower, ci_upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, 
                        help="Root directory of 8B LongBench results")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Cannot resolve data_dir: {args.data_dir}")
    if str(data_dir) != args.data_dir:
        print(f"  [INFO] Resolved data_dir to: {data_dir}")
    
    # ============================================================
    # STEP 1: Load all per-sample scores
    # ============================================================
    all_scores = {}  # {method: {task: np.array}}
    
    for method in METHODS:
        all_scores[method] = {}
        for task in TASKS:
            scores = load_per_sample_scores(str(data_dir), method, task)
            if scores is not None:
                all_scores[method][task] = scores
                print(f"  Loaded {method}/{task}: {len(scores)} samples, "
                      f"mean={np.mean(scores):.4f}")
    
    # ============================================================
    # STEP 2: Compute per-method averages (sanity check)
    # ============================================================
    print("\n" + "="*60)
    print("SANITY CHECK: Method averages (should match your report)")
    print("="*60)
    
    for method in METHODS:
        if all_scores[method]:
            task_means = [np.mean(all_scores[method][t]) 
                         for t in TASKS if t in all_scores[method]]
            overall = np.mean(task_means) if task_means else float('nan')
            print(f"  {method:20s}: {overall:.4f}")
    
    # ============================================================
    # STEP 3: Pairwise significance tests
    # ============================================================
    # Key comparisons for the paper
    comparisons = [
        ("anchored_sigmoid", "baseline",  "Anc-Sig vs Baseline"),
        ("anchored_sigmoid", "pi",        "Anc-Sig vs PI"),
        ("anchored_sigmoid", "yarn",      "Anc-Sig vs YaRN"),
        ("anchored_sigmoid", "sigmoid",   "Anc-Sig vs Sigmoid (ablation)"),
        ("sigmoid",          "baseline",  "Sigmoid vs Baseline"),
        ("sigmoid",          "pi",        "Sigmoid vs PI"),
    ]
    
    print("\n" + "="*60)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTS")
    print(f"(n_bootstrap={args.n_bootstrap})")
    print("="*60)
    
    results_for_paper = []
    
    for method_a, method_b, label in comparisons:
        print(f"\n--- {label} ---")
        
        # Concatenate per-sample scores across all tasks
        scores_a_all = []
        scores_b_all = []
        
        for task in TASKS:
            if (task in all_scores[method_a] and 
                task in all_scores[method_b]):
                sa = all_scores[method_a][task]
                sb = all_scores[method_b][task]
                n = min(len(sa), len(sb))  # align lengths
                scores_a_all.append(sa[:n])
                scores_b_all.append(sb[:n])
        
        if not scores_a_all:
            print("  [SKIP] No overlapping task data")
            continue
        
        scores_a = np.concatenate(scores_a_all)
        scores_b = np.concatenate(scores_b_all)
        
        diff, p, ci_lo, ci_hi = paired_bootstrap_test(
            scores_a, scores_b, n_bootstrap=args.n_bootstrap
        )
        
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        
        print(f"  N samples:     {len(scores_a)}")
        print(f"  Mean A:        {np.mean(scores_a):.4f}")
        print(f"  Mean B:        {np.mean(scores_b):.4f}")
        print(f"  Diff (A-B):    {diff:+.4f}")
        print(f"  95% CI:        [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  p-value:       {p:.4f}  {sig}")
        
        results_for_paper.append({
            "comparison": label,
            "diff": diff,
            "ci": (ci_lo, ci_hi),
            "p": p,
            "sig": sig
        })
    
    # ============================================================
    # STEP 4: Summary for paper
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY FOR PAPER")
    print("="*60)
    print()
    
    any_significant = False
    for r in results_for_paper:
        status = "SIGNIFICANT" if r["p"] < 0.05 else "NOT significant"
        print(f"  {r['comparison']:30s}  p={r['p']:.4f}  {status}")
        if r["p"] < 0.05:
            any_significant = True
    
    print()
    if any_significant:
        print("  ✅ You CAN write 'statistically significant' for p<0.05 comparisons.")
        print("  Suggested wording: 'statistically significant improvement")
        print("  (paired bootstrap, p < 0.05, 10K resamples)'")
    else:
        print("  ⚠️  No comparison reached p<0.05.")
        print("  Suggested wording: 'numerically higher but not statistically")
        print("  significant at conventional thresholds (paired bootstrap, 10K resamples)'")
    
    # Save results
    output_path = data_dir / "significance_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([{**r, "ci": list(r["ci"])} for r in results_for_paper], f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
