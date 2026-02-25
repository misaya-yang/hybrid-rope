#!/usr/bin/env python3
"""
Pareto Frontier Search for Sparse Attention
===========================================

Grid search over (lambda, gamma) to find the Pareto frontier of:
- X-axis: Sparsity (or avg NNZ)
- Y-axis: Perplexity

Usage:
    # Quick search on M4
    python search_pareto.py --model gpt2 --epochs 1 --max_train_samples 10000
    
    # Full search on A100
    python search_pareto.py --model gpt2 --epochs 3
    
    # Use LoRA for larger models
    python search_pareto.py --model gpt2-large --method lora --epochs 3
"""

import argparse
import json
import itertools
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


def run_experiment(model, method, lam, gamma, epochs, output_dir, **kwargs):
    """Run a single fine-tuning experiment."""
    
    cmd = [
        sys.executable, 'finetune_sparse.py',
        '--model', model,
        '--method', method,
        '--lam', str(lam),
        '--gamma', str(gamma),
        '--epochs', str(epochs),
        '--variant', 'prior_sparse',
        '--prior_mode', 'centered',
        '--output_dir', str(output_dir),
    ]
    
    # Add optional args
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    print(f"\n[RUN] λ={lam}, γ={gamma}")
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    # Find the output directory (latest one matching our pattern)
    output_path = Path(output_dir)
    if output_path.exists():
        dirs = sorted(output_path.glob(f"{model}_{method}_prior_sparse_gamma{gamma}_*"))
        if dirs:
            latest_dir = dirs[-1]
            summary_file = latest_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    return json.load(f)
    
    return None


def search_pareto(args):
    """Main Pareto search loop."""
    
    print("="*70)
    print("PARETO FRONTIER SEARCH")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    print(f"Epochs per config: {args.epochs}")
    
    # First, run dense baseline
    print("\n[PHASE 1] Running dense baseline...")
    baseline = run_experiment(
        args.model, args.method,
        lam=0.0, gamma=1.0,
        epochs=args.epochs,
        output_dir=args.output_dir,
        no_sparse=True,
        max_train_samples=args.max_train_samples,
        batch_size=args.batch_size,
    )
    
    if baseline:
        baseline_ppl = baseline['best_ppl']
        print(f"\n*** Baseline PPL: {baseline_ppl:.2f} ***")
    else:
        print("[WARN] Failed to get baseline, will use PPL=30 as reference")
        baseline_ppl = 30.0
    
    # Grid search over sparse configs
    print("\n[PHASE 2] Grid search over sparse configs...")
    
    results = []
    
    for lam in args.lambdas:
        for gamma in args.gammas:
            result = run_experiment(
                args.model, args.method,
                lam=lam, gamma=gamma,
                epochs=args.epochs,
                output_dir=args.output_dir,
                max_train_samples=args.max_train_samples,
                batch_size=args.batch_size,
                use_amp=args.use_amp,
            )
            
            if result:
                results.append({
                    'lambda': lam,
                    'gamma': gamma,
                    'ppl': result['best_ppl'],
                    'sparsity': result.get('final_sparsity', 0),
                    'nnz': result.get('final_nnz', 0),
                    'ppl_increase': (result['best_ppl'] - baseline_ppl) / baseline_ppl * 100,
                })
                
                print(f"\n  Result: PPL={result['best_ppl']:.2f}, "
                      f"Sparsity={result.get('final_sparsity', 0):.2%}")
            else:
                print(f"\n  [ERROR] Failed to get result for λ={lam}, γ={gamma}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / f"pareto_search_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(results_dir / 'pareto_results.csv', index=False)
    print(f"\n[SAVE] Results saved to {results_dir / 'pareto_results.csv'}")
    
    # Generate Pareto frontier plot
    if len(results) > 0:
        plot_pareto(df, baseline_ppl, results_dir)
        find_and_report_sweet_spot(df, baseline_ppl)
    
    return results


def plot_pareto(df, baseline_ppl, output_dir):
    """Generate Pareto frontier plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: PPL vs Sparsity
    scatter1 = ax1.scatter(
        df['sparsity'] * 100,
        df['ppl'],
        c=df['gamma'],
        cmap='viridis_r',
        s=200,
        edgecolors='black',
        linewidth=2
    )
    
    # Mark baseline
    ax1.scatter([0], [baseline_ppl], marker='*', s=500, color='red',
               edgecolors='black', linewidth=2, label='Baseline (Dense)',
               zorder=5)
    
    # Mark target region
    ax1.axvline(x=70, color='green', linestyle=':', linewidth=2, alpha=0.7,
               label='Target: 70% sparsity')
    ax1.axhline(y=baseline_ppl * 1.05, color='orange', linestyle=':',
               linewidth=2, alpha=0.7, label='Target: +5% PPL')
    
    ax1.set_xlabel('Sparsity (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Perplexity', fontsize=14, fontweight='bold')
    ax1.set_title('Pareto Frontier: Sparsity vs PPL', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('γ (gamma)', fontsize=12, fontweight='bold')
    
    # Plot 2: PPL vs NNZ
    scatter2 = ax2.scatter(
        df['nnz'],
        df['ppl'],
        c=df['lambda'],
        cmap='plasma',
        s=200,
        edgecolors='black',
        linewidth=2
    )
    
    # Mark baseline NNZ (approximate for seq_len=512)
    baseline_nnz = 256  # ~seq_len/2
    ax2.scatter([baseline_nnz], [baseline_ppl], marker='*', s=500, color='red',
               edgecolors='black', linewidth=2, label='Baseline',
               zorder=5)
    
    # Mark target
    ax2.axvline(x=baseline_nnz * 0.3, color='green', linestyle=':',
               linewidth=2, alpha=0.7, label='Target: 70% reduction')
    ax2.axhline(y=baseline_ppl * 1.05, color='orange', linestyle=':',
               linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Avg NNZ per token', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Perplexity', fontsize=14, fontweight='bold')
    ax2.set_title('Pareto Frontier: Compute (NNZ) vs PPL', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('λ (lambda)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier.png', dpi=300, bbox_inches='tight')
    print(f"[SAVE] Plot saved to {output_dir / 'pareto_frontier.png'}")
    plt.close()


def find_and_report_sweet_spot(df, baseline_ppl):
    """Find and report sweet spot configurations."""
    
    print("\n" + "="*70)
    print("SWEET SPOT ANALYSIS")
    print("="*70)
    
    # Criteria 1: Sparsity >= 70%, PPL <= +5%
    candidates = df[
        (df['sparsity'] >= 0.70) &
        (df['ppl'] <= baseline_ppl * 1.05)
    ]
    
    if len(candidates) > 0:
        print("\n✅ FOUND SWEET SPOT(S)!")
        print(f"   Criteria: Sparsity >= 70%, PPL <= +5%")
        print()
        
        # Sort by PPL increase
        candidates = candidates.sort_values('ppl_increase')
        
        for idx, row in candidates.head(3).iterrows():
            print(f"   Config: λ={row['lambda']:.3f}, γ={row['gamma']:.2f}")
            print(f"   PPL: {row['ppl']:.2f} ({row['ppl_increase']:+.1f}%)")
            print(f"   Sparsity: {row['sparsity']:.1%}")
            print(f"   NNZ: {row['nnz']:.1f}")
            print()
        
        return candidates.iloc[0]
    
    # Relax criteria
    print("\n⚠️  No config met strict criteria (>=70% sparsity, <=+5% PPL)")
    print("\n   Relaxing to: Sparsity >= 50%, PPL <= +10%")
    
    relaxed = df[
        (df['sparsity'] >= 0.50) &
        (df['ppl'] <= baseline_ppl * 1.10)
    ]
    
    if len(relaxed) > 0:
        relaxed = relaxed.sort_values('ppl_increase')
        best = relaxed.iloc[0]
        
        print(f"\n   Best relaxed config:")
        print(f"   λ={best['lambda']:.3f}, γ={best['gamma']:.2f}")
        print(f"   PPL: {best['ppl']:.2f} ({best['ppl_increase']:+.1f}%)")
        print(f"   Sparsity: {best['sparsity']:.1%}")
        return best
    
    print("\n❌ No viable configuration found even with relaxed criteria")
    print("   Consider:")
    print("   - More training epochs")
    print("   - Larger model (GPT-2 Medium/Large)")
    print("   - Higher LoRA rank")
    
    return None


def main():
    parser = argparse.ArgumentParser()
    
    # Search space
    parser.add_argument('--lambdas', type=float, nargs='+',
                       default=[0.0, 0.005, 0.01, 0.02, 0.05],
                       help='Lambda values to search')
    parser.add_argument('--gammas', type=float, nargs='+',
                       default=[0.5, 1.0, 2.0, 5.0, 10.0],
                       help='Gamma values to search')
    
    # Model and training
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--method', type=str, default='lora',
                       choices=['full', 'lora'])
    parser.add_argument('--epochs', type=int, default=1,
                       help='Training epochs per config')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/pareto_search')
    
    args = parser.parse_args()
    
    print(f"Search space:")
    print(f"  λ ∈ {args.lambdas}")
    print(f"  γ ∈ {args.gammas}")
    print(f"  Total configs: {len(args.lambdas) * len(args.gammas)}")
    
    results = search_pareto(args)
    
    print("\n" + "="*70)
    print("SEARCH COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
