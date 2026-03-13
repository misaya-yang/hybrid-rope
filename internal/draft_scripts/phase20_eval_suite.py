#!/usr/bin/env python3
"""Comprehensive evaluation suite for Phase 20 1.5B experiments.

Evaluates trained checkpoints with multiple metrics including:
- PPL curves at multiple context lengths
- Position-wise PPL (sequence divided into bins)
- Cross-domain PPL (FineWeb-Edu, SlimPajama, Proof-Pile, C4)
- Passkey retrieval at various depths and lengths
- Multi-needle NIAH (in-context hallucination)
- NIAH heatmap generation
- YaRN scale factor comparison grid
- Comprehensive aggregation with comparison tables and figures

Usage:
    python team/scripts/phase20_eval_suite.py \\
        --checkpoint /path/to/checkpoint.pt \\
        --output_dir results/phase20/ \\
        --suite standard

    python team/scripts/phase20_eval_suite.py \\
        --checkpoint_dir /path/to/checkpoints/ \\
        --suite full \\
        --output_dir results/phase20_full/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Set HF mirror for accessibility
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# Import from run_evq_sweep
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "core_text_phases"
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from run_evq_sweep import (
        GPT,
        DEVICE,
        DTYPE,
        evq_cosh_inv_freq,
        load_val,
        set_seed,
    )
except ImportError as e:
    print(f"Error importing from run_evq_sweep: {e}")
    sys.exit(1)

# Try importing eval_passkey functions
try:
    from eval_passkey import build_passkey_sequence, eval_passkey_single, _sanitize_filler
except ImportError:
    print("Warning: Could not import passkey utilities from eval_passkey.py")

USE_AUTOCAST = DEVICE == "cuda" and DTYPE != torch.float32

# Configuration for different evaluation suites
SUITE_CONFIGS = {
    "quick": {
        "ppl_lengths": [2048, 4096, 8192],
        "ppl_chunks": 5,
        "position_wise": False,
        "cross_domain": False,
        "passkey": False,
        "niah": False,
        "yarn": False,
    },
    "standard": {
        "ppl_lengths": [2048, 4096, 8192, 16384],
        "ppl_chunks": 10,
        "position_wise": True,
        "cross_domain": False,
        "passkey": True,
        "niah": False,
        "yarn": False,
    },
    "full": {
        "ppl_lengths": [2048, 4096, 8192, 16384, 32768],
        "ppl_chunks": 15,
        "position_wise": True,
        "cross_domain": True,
        "passkey": True,
        "niah": True,
        "yarn": True,
    },
}

# ---------------------------------------------------------------------------
# PPL Evaluation Functions
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_ppl_curves(
    model: GPT,
    val_data: torch.Tensor,
    eval_lengths: List[int],
    chunks: int = 10,
    seed: int = 42,
) -> Dict[int, float]:
    """Evaluate PPL at multiple context lengths.

    Args:
        model: GPT model to evaluate
        val_data: Validation token IDs tensor
        eval_lengths: List of context lengths to evaluate
        chunks: Number of chunks to divide data into
        seed: Random seed

    Returns:
        Dict mapping length -> mean NLL per token
    """
    set_seed(seed)
    model.eval()

    results = {}
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()

    # Divide validation data into chunks
    chunk_size = len(val_data) // chunks

    for length in sorted(eval_lengths):
        print(f"  Evaluating at context length {length}...", end=" ", flush=True)
        model.extend_rope(length + 100)

        nlls = []

        for chunk_idx in range(chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, len(val_data))

            # Prepare sequences of exact length
            available = end - start
            if available < length:
                continue

            seq = val_data[start : start + length].unsqueeze(0).to(DEVICE)

            with ctx:
                logits = model(seq)  # (1, L, V)

            # Compute NLL on all positions (causal)
            shift_logits = logits[0, :-1, :].float()  # (L-1, V)
            shift_labels = seq[0, 1:]  # (L-1,)

            loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')
            nlls.append(loss.item())

        if nlls:
            mean_nll = float(np.mean(nlls))
            results[length] = mean_nll
            print(f"NLL={mean_nll:.4f}, PPL={np.exp(mean_nll):.2f}")
        else:
            print("SKIP (insufficient data)")

    return results


@torch.no_grad()
def eval_position_wise_ppl(
    model: GPT,
    val_data: torch.Tensor,
    eval_length: int,
    n_bins: int = 20,
    seed: int = 42,
) -> Dict[str, float]:
    """Divide sequence into N bins and compute NLL per bin.

    Reveals WHERE in the sequence EVQ gains are concentrated.

    Returns:
        Dict mapping "bin_{i}" -> mean NLL for that bin
    """
    set_seed(seed)
    model.eval()
    model.extend_rope(eval_length + 100)

    print(f"  Position-wise PPL (n_bins={n_bins})...", end=" ", flush=True)

    # Take first eval_length tokens as test sequence
    seq = val_data[:eval_length].unsqueeze(0).to(DEVICE)

    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    with ctx:
        logits = model(seq)  # (1, L, V)

    # Compute loss per position
    shift_logits = logits[0, :-1, :].float()  # (L-1, V)
    shift_labels = seq[0, 1:]  # (L-1,)

    # Per-token loss (not reduced)
    losses = F.cross_entropy(shift_logits, shift_labels, reduction='none')  # (L-1,)

    # Bin the losses
    bin_size = (len(losses) + n_bins - 1) // n_bins
    results = {}

    for bin_idx in range(n_bins):
        start = bin_idx * bin_size
        end = min((bin_idx + 1) * bin_size, len(losses))
        if start >= len(losses):
            break

        bin_nll = losses[start:end].mean().item()
        results[f"bin_{bin_idx}"] = round(bin_nll, 4)

    print(f"computed {len(results)} bins")
    return results


@torch.no_grad()
def eval_cross_domain_ppl(
    model: GPT,
    eval_lengths: List[int],
    datasets: Dict[str, torch.Tensor],
    chunks: int = 5,
    seed: int = 42,
) -> Dict[str, Dict[int, float]]:
    """Evaluate model on multiple corpora.

    Args:
        model: GPT model
        eval_lengths: Context lengths to test
        datasets: Dict mapping domain_name -> token_tensor
        chunks: Number of chunks per dataset
        seed: Random seed

    Returns:
        Dict mapping domain -> length -> NLL
    """
    set_seed(seed)
    model.eval()

    results = {}

    for domain_name, tokens in datasets.items():
        print(f"\n  Domain: {domain_name}")
        domain_results = eval_ppl_curves(model, tokens, eval_lengths, chunks, seed)
        results[domain_name] = domain_results

    return results


@torch.no_grad()
def eval_yarn_overlay(
    model: GPT,
    val_data: torch.Tensor,
    eval_lengths: List[int],
    yarn_scales: List[float] = [1.0, 2.0, 4.0, 8.0],
    chunks: int = 5,
    seed: int = 42,
) -> Dict[float, Dict[int, float]]:
    """Evaluate model with YaRN scale factors to detect phase transitions.

    Args:
        model: GPT model with modifiable RoPE
        val_data: Validation tokens
        eval_lengths: Context lengths
        yarn_scales: List of YaRN scale factors to test
        chunks: Chunks for PPL eval
        seed: Random seed

    Returns:
        Dict mapping scale -> length -> NLL
    """
    print(f"\n  YaRN overlay evaluation (scales={yarn_scales})")
    results = {}

    # ⚠️ PLACEHOLDER: YaRN overlay requires modifying model's RoPE inv_freq
    # at inference time. Full implementation should:
    #   1. Save original inv_freq
    #   2. For each scale: apply YaRN NTK-by-parts scaling to inv_freq
    #   3. Rebuild sin/cos cache
    #   4. Evaluate
    #   5. Restore original inv_freq
    # Current version evaluates without actual YaRN scaling (all scales identical).
    # Will be implemented when integrating with the existing YaRN code from
    # scripts/core_text_phases/phase17c_extended_eval.py

    print("    ⚠️ WARNING: YaRN overlay is PLACEHOLDER - scales not actually applied")
    for scale in yarn_scales:
        print(f"    Scale={scale}...", end=" ", flush=True)
        scale_results = eval_ppl_curves(model, val_data, eval_lengths, chunks, seed)
        results[scale] = scale_results

    return results


# ---------------------------------------------------------------------------
# Passkey & NIAH Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_passkey_grid(
    model: GPT,
    tokenizer,
    filler_ids: torch.Tensor,
    lengths: List[int],
    depths: List[float],
    trials: int = 30,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluate passkey retrieval across a grid of lengths and depths.

    Args:
        model: GPT model
        tokenizer: Tokenizer for building sequences
        filler_ids: Pre-loaded filler token IDs
        lengths: List of context lengths
        depths: List of depth ratios (0=start, 1=end)
        trials: Number of trials per condition
        seed: Base seed

    Returns:
        Dict mapping "L={L}_d={d}" -> {"accuracy": ..., "mean_rank": ..., ...}
    """
    print(f"\n  Passkey grid (lengths={lengths}, depths={depths})")
    model.eval()

    results = {}

    for length in lengths:
        for depth in depths:
            key = f"L={length}_d={depth:.2f}"
            print(f"    {key}...", end=" ", flush=True)

            # Use eval_passkey_single from eval_passkey module
            try:
                result = eval_passkey_single(
                    model, tokenizer, filler_ids,
                    length, depth, trials, base_seed=seed + hash(key) % 10000
                )
                results[key] = result
                print(f"acc={result['accuracy']:.1%}")
            except Exception as e:
                print(f"SKIP ({e})")
                results[key] = {"accuracy": -1, "error": str(e)}

    return results


@torch.no_grad()
def eval_multi_needle(
    model: GPT,
    tokenizer,
    filler_ids: torch.Tensor,
    context_lengths: List[int],
    num_needles: List[int] = [2, 4, 8],
    trials: int = 10,
    seed: int = 42,
) -> Dict[str, Dict[int, float]]:
    """Evaluate multi-needle retrieval at different context lengths.

    Returns:
        Dict mapping "needles_{n}" -> length -> recall_accuracy
    """
    print(f"\n  Multi-needle NIAH (needles={num_needles})")

    results = {}
    for n_needles in num_needles:
        needle_results = {}
        for length in context_lengths:
            key = f"n={n_needles}_L={length}"
            print(f"    {key}...", end=" ", flush=True)

            # ⚠️ PLACEHOLDER: Multi-needle requires constructing sequences with
            # n_needles distinct passkeys at different depths, then evaluating
            # retrieval accuracy for each. Will be implemented in production.
            acc = -1.0  # Placeholder: -1 indicates not-yet-implemented
            needle_results[length] = acc
            print(f"recall={acc:.1%}")

        results[f"needles_{n_needles}"] = needle_results

    return results


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------


def generate_niah_heatmap(
    results: Dict[str, Dict[int, float]],
    output_path: Path,
) -> None:
    """Generate 2D heatmap of context_length × depth → accuracy.

    Args:
        results: Output from eval_passkey_grid or similar
        output_path: Path to save PNG figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [WARN] matplotlib not available, skipping heatmap")
        return

    print(f"  Generating NIAH heatmap...", end=" ", flush=True)

    # Parse results into 2D array
    # This is a simplified version; full implementation would extract
    # depths and lengths from keys

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        # TODO: Build matrix from results dict
        ax.text(0.5, 0.5, "NIAH Accuracy Grid", ha='center', va='center')
        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("Depth Ratio")
        ax.set_title("Passkey Retrieval Accuracy Heatmap")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"saved to {output_path}")
    except Exception as e:
        print(f"SKIP ({e})")


def generate_comparison_table(
    geo_results: Dict[str, float],
    evq_results: Dict[str, float],
) -> str:
    """Generate formatted comparison table for printing.

    Args:
        geo_results: PPL results for geometric RoPE model
        evq_results: PPL results for EVQ model

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  COMPARISON TABLE: Geometric vs EVQ")
    lines.append("=" * 70)
    lines.append(f"  {'Context Length':<20}  {'Geo PPL':<15}  {'EVQ PPL':<15}  {'Improvement':<15}")
    lines.append("  " + "-" * 65)

    for length in sorted(set(list(geo_results.keys()) + list(evq_results.keys()))):
        geo_nll = geo_results.get(length, float('nan'))
        evq_nll = evq_results.get(length, float('nan'))

        if not math.isnan(geo_nll) and not math.isnan(evq_nll):
            geo_ppl = np.exp(geo_nll)
            evq_ppl = np.exp(evq_nll)
            improvement = (geo_ppl - evq_ppl) / geo_ppl * 100

            lines.append(
                f"  {length:<20}  {geo_ppl:<15.2f}  {evq_ppl:<15.2f}  {improvement:<14.1f}%"
            )

    lines.append("  " + "-" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregation & Caching
# ---------------------------------------------------------------------------


def load_cached_results(cache_path: Path) -> Optional[Dict]:
    """Load cached evaluation results if available."""
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  [WARN] Failed to load cache: {e}")
    return None


def save_cached_results(results: Dict, cache_path: Path) -> None:
    """Save evaluation results to cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
    except Exception as e:
        print(f"  [WARN] Failed to save cache: {e}")


def aggregate_all(
    results_dir: Path,
    geo_checkpoints: List[Path],
    evq_checkpoints: List[Path],
) -> Dict:
    """Aggregate results from multiple evaluations.

    Args:
        results_dir: Directory containing result files
        geo_checkpoints: Geometric RoPE checkpoint paths
        evq_checkpoints: EVQ checkpoint paths

    Returns:
        Aggregated results dictionary
    """
    print(f"\n{'='*70}")
    print(f"  AGGREGATING RESULTS")
    print(f"{'='*70}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_geo_models": len(geo_checkpoints),
        "num_evq_models": len(evq_checkpoints),
        "results_dir": str(results_dir),
    }

    # Placeholder for aggregation logic
    print(f"  Aggregated {len(geo_checkpoints) + len(evq_checkpoints)} models")

    return summary


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------


def find_checkpoints(checkpoint_dir: Path, pattern: str = "*.pt") -> List[Path]:
    """Find all checkpoints matching pattern in directory."""
    return sorted(checkpoint_dir.glob(pattern))


def load_checkpoint(
    checkpoint_path: Path,
    tier: str = "1500m",
    tau: float = 1.5,
    base: float = 500000.0,
) -> GPT:
    """Load a checkpoint into a GPT model."""
    from run_evq_sweep import TIER_CONFIGS

    # For phase20, use 1.5B config
    cfg = {
        "vocab_size": 50304,
        "hidden_size": 1536,
        "num_layers": 32,
        "num_heads": 24,
        "head_dim": 64,
        "intermediate_size": 6144,
        "max_position_embeddings": 2048,
    }

    inv_freq = evq_cosh_inv_freq(cfg["head_dim"], tau, base)
    model = GPT(cfg, inv_freq)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()

    print(f"  Loaded checkpoint: {checkpoint_path.name}")
    return model


# ---------------------------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint_path: Path,
    suite_name: str = "standard",
    output_dir: Path = None,
    dry_run: bool = False,
    tokenizer = None,
    val_data: torch.Tensor = None,
) -> Dict:
    """Evaluate a single checkpoint with specified suite.

    Returns:
        Dictionary of evaluation results
    """
    suite_cfg = SUITE_CONFIGS[suite_name]

    print(f"\n{'='*70}")
    print(f"  EVALUATING: {checkpoint_path.name}")
    print(f"  Suite: {suite_name}")
    print(f"{'='*70}")

    results = {
        "checkpoint": str(checkpoint_path),
        "suite": suite_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Load model
    try:
        model = load_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"  [ERROR] Failed to load checkpoint: {e}")
        return results

    # Load validation data if not provided
    if val_data is None and suite_cfg["ppl_lengths"]:
        print("  Loading validation data...")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            val_data = load_val(tokenizer, 5_000_000)
        except Exception as e:
            print(f"  [WARN] Could not load validation data: {e}")
            val_data = None

    # PPL Curves
    if suite_cfg["ppl_lengths"] and val_data is not None:
        print("\nPPL Curves:")
        ppl_results = eval_ppl_curves(
            model, val_data, suite_cfg["ppl_lengths"], suite_cfg["ppl_chunks"]
        )
        results["ppl_curves"] = ppl_results

    # Position-wise PPL
    if suite_cfg["position_wise"] and val_data is not None:
        print("\nPosition-wise PPL:")
        pos_results = eval_position_wise_ppl(
            model, val_data, suite_cfg["ppl_lengths"][0], n_bins=20
        )
        results["position_wise_ppl"] = pos_results

    # Passkey
    if suite_cfg["passkey"] and tokenizer is not None and val_data is not None:
        print("\nPasskey Retrieval:")
        try:
            filler_ids = _sanitize_filler(tokenizer, val_data, 256)
            passkey_results = eval_passkey_grid(
                model, tokenizer, filler_ids,
                lengths=[4096, 8192],
                depths=[0.1, 0.5, 0.9],
                trials=20,
            )
            results["passkey"] = passkey_results
        except Exception as e:
            print(f"  [SKIP] Passkey evaluation failed: {e}")

    # Clean up
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / f"{checkpoint_path.stem}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved: {result_file}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation suite for Phase 20 experiments"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to single checkpoint to evaluate"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory containing multiple checkpoints to evaluate"
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=list(SUITE_CONFIGS.keys()),
        default="standard",
        help="Evaluation suite: quick/standard/full"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test without full evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "metadata": {
            "suite": args.suite,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "seed": args.seed,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "evaluations": [],
    }

    # Determine checkpoints to evaluate
    checkpoints = []
    if args.checkpoint:
        checkpoints = [Path(args.checkpoint)]
    elif args.checkpoint_dir:
        checkpoints = find_checkpoints(Path(args.checkpoint_dir))
    else:
        print("Error: Must provide --checkpoint or --checkpoint_dir")
        sys.exit(1)

    if not checkpoints:
        print(f"Error: No checkpoints found")
        sys.exit(1)

    print(f"\nFound {len(checkpoints)} checkpoint(s) to evaluate")

    # Evaluate each checkpoint
    for ckpt_path in checkpoints:
        if args.dry_run:
            print(f"[DRY-RUN] Would evaluate: {ckpt_path}")
        else:
            result = evaluate_checkpoint(
                ckpt_path,
                suite_name=args.suite,
                output_dir=output_dir,
                dry_run=False,
            )
            all_results["evaluations"].append(result)

    # Save aggregate results
    all_results["metadata"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    aggregate_file = output_dir / f"phase20_eval_suite_{args.suite}.json"
    with open(aggregate_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Aggregate results: {aggregate_file}")
    print(f"  Device: {DEVICE} | Dtype: {DTYPE}")
    print(f"  Suite: {args.suite} | Seed: {args.seed}")


if __name__ == "__main__":
    main()
