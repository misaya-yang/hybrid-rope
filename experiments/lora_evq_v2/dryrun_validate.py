#!/usr/bin/env python3
"""
EVQ-Cosh LoRA Dry-Run Validator
================================
Validates the entire experiment pipeline WITHOUT GPU:
  1. EVQ-cosh frequency computation + self-consistency
  2. Theory parameter checks
  3. Dataset accessibility & format
  4. Config serialization
  5. Frequency comparison plots (optional)

Usage:
    python dryrun_validate.py
    python dryrun_validate.py --tau 1.0 --lora_r 32
    python dryrun_validate.py --plot  # generate frequency plot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent to path for importing train script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_evq_lora import (
    compute_evq_cosh_inv_freq,
    compute_geometric_inv_freq,
)


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

class ValidationResult:
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def check(self, name: str, condition: bool, msg: str = "", warn_only: bool = False):
        status = "✅" if condition else ("⚠️" if warn_only else "❌")
        self.checks.append({"name": name, "status": status, "msg": msg})
        if condition:
            self.passed += 1
        elif warn_only:
            self.warnings += 1
        else:
            self.failed += 1
        print(f"  {status} {name}" + (f": {msg}" if msg else ""))

    def summary(self):
        print(f"\n{'=' * 60}")
        print(f"VALIDATION SUMMARY: {self.passed} passed, {self.failed} failed, {self.warnings} warnings")
        print(f"{'=' * 60}")
        return self.failed == 0


def validate_frequencies(args, vr: ValidationResult):
    """Validate EVQ-cosh frequency computation."""
    print("\n--- Frequency Computation ---")

    K = args.head_dim // 2

    # Compute frequencies
    inv_evq = compute_evq_cosh_inv_freq(args.head_dim, args.rope_base, args.tau, midpoint=True)
    inv_geo = compute_geometric_inv_freq(args.head_dim, args.rope_base)

    vr.check("inv_freq shape", inv_evq.shape == (K,),
             f"expected ({K},), got {tuple(inv_evq.shape)}")

    # All positive
    vr.check("all positive", (inv_evq > 0).all().item(),
             f"min={inv_evq.min().item():.2e}")

    # Monotonically decreasing
    diffs = inv_evq[1:] - inv_evq[:-1]
    vr.check("monotonically decreasing", (diffs <= 1e-12).all().item(),
             f"max increase={diffs.max().item():.2e}")

    # Range check: first should be close to 1, last close to 0
    vr.check("first freq ≈ 1", inv_evq[0].item() > 0.5,
             f"inv_freq[0]={inv_evq[0].item():.6f}")
    vr.check("last freq > 0", inv_evq[-1].item() > 0,
             f"inv_freq[-1]={inv_evq[-1].item():.2e}")

    # τ=0 should give geometric
    inv_zero = compute_evq_cosh_inv_freq(args.head_dim, args.rope_base, 0.0, midpoint=True)
    # Note: midpoint u_k vs boundary u_k means τ=0 won't exactly match geometric
    # but should be very close
    max_diff = (inv_zero - inv_geo).abs().max().item()
    # With midpoint quantization, there's a systematic offset at τ=0
    # The key check is that τ=0 gives something very close to geometric
    vr.check("τ=0 ≈ geometric", max_diff < 0.1,
             f"max_diff={max_diff:.6f}")

    # Self-consistency: φ(0) = 0, φ(1) should approach 1
    # φ_k = -log(inv_freq_k) / log(base)
    phi_evq = -torch.log(inv_evq) / math.log(args.rope_base)
    vr.check("φ[0] ≈ 0", abs(phi_evq[0].item()) < 0.05,
             f"φ[0]={phi_evq[0].item():.6f}")
    vr.check("φ[-1] < 1", phi_evq[-1].item() < 1.0 + 1e-6,
             f"φ[-1]={phi_evq[-1].item():.6f}")

    # EVQ should spread low frequencies more than geometric
    # (more channels in φ < 0.5 for EVQ than geometric)
    n_low_evq = (phi_evq < 0.5).sum().item()
    phi_geo = -torch.log(inv_geo) / math.log(args.rope_base)
    n_low_geo = (phi_geo < 0.5).sum().item()
    vr.check("EVQ spreads low freqs", n_low_evq >= n_low_geo,
             f"EVQ: {n_low_evq} channels below φ=0.5, Geo: {n_low_geo}")

    # Channel displacement at this τ
    mid_idx = K // 2
    displacement = abs(phi_evq[mid_idx].item() - phi_geo[mid_idx].item())
    n_displaced = sum(1 for k in range(K)
                      if abs(phi_evq[k].item() - phi_geo[k].item()) > 0.01)
    print(f"\n  [INFO] Channel displacement at τ={args.tau}:")
    print(f"    Mid-channel shift: {displacement:.4f}")
    print(f"    Significantly displaced channels: {n_displaced}/{K}")

    return inv_evq, inv_geo


def validate_theory(args, vr: ValidationResult):
    """Validate theoretical parameters."""
    print("\n--- Theory Parameters ---")

    K = args.head_dim // 2
    tau_theory = args.head_dim / math.sqrt(args.max_seq_len)

    vr.check("τ matches theory", abs(args.tau - tau_theory) < 0.1,
             f"set={args.tau}, theory={tau_theory:.4f}")

    r_ratio = args.lora_r / K
    vr.check("r/K ≥ 1 (phase-safe)", r_ratio >= 1.0,
             f"r={args.lora_r}, K={K}, r/K={r_ratio:.2f}")

    vr.check("r/K ≥ 0.5 (marginal ok)", r_ratio >= 0.5,
             f"r/K={r_ratio:.2f}", warn_only=(r_ratio >= 0.5))

    # Critical rank calculation
    r_c = K  # for well-trained models, r_c ≈ K
    vr.check("r ≥ r_c (critical rank)", args.lora_r >= r_c,
             f"r={args.lora_r}, r_c≈{r_c}")

    # Alpha/r ratio
    alpha_ratio = args.lora_alpha / args.lora_r
    vr.check("alpha/r ∈ [1, 4]", 1.0 <= alpha_ratio <= 4.0,
             f"alpha/r={alpha_ratio:.1f}", warn_only=True)

    # Predicted PPL behavior
    if r_ratio >= 1.0:
        print(f"  [PREDICT] r/K={r_ratio:.2f} → τ*={tau_theory:.3f} should work (full EVQ)")
    elif r_ratio >= 0.5:
        predicted_tau = tau_theory * math.sqrt(r_ratio)
        print(f"  [PREDICT] r/K={r_ratio:.2f} → partial EVQ, effective τ*≈{predicted_tau:.3f}")
    else:
        print(f"  [PREDICT] r/K={r_ratio:.2f} → ❌ EVQ infeasible (phase transition)")


def validate_data(args, vr: ValidationResult):
    """Validate dataset accessibility."""
    print("\n--- Dataset Validation ---")

    try:
        from datasets import load_dataset
        has_datasets = True
        vr.check("datasets library", True)
    except ImportError:
        has_datasets = False
        vr.check("datasets library", False, "pip install datasets")
        return

    if args.local_data_path:
        exists = os.path.exists(args.local_data_path)
        vr.check("local data exists", exists, args.local_data_path)
        if exists:
            # Check format
            with open(args.local_data_path) as f:
                first_line = f.readline().strip()
            try:
                item = json.loads(first_line)
                has_messages = "messages" in item
                has_instruction = "instruction" in item
                vr.check("data format valid", has_messages or has_instruction,
                         f"keys: {list(item.keys())[:5]}")
            except json.JSONDecodeError:
                vr.check("data format valid", False, "not valid JSONL")
    else:
        # Try loading a tiny sample from HuggingFace
        print(f"  [INFO] Testing dataset: {args.dataset_name}")
        try:
            ds = load_dataset(args.dataset_name, split="train[:5]",
                            trust_remote_code=True)
            vr.check("HF dataset accessible", True,
                     f"{args.dataset_name} ({len(ds)} samples loaded)")
            # Check format
            if len(ds) > 0:
                item = ds[0]
                keys = list(item.keys())
                vr.check("data has expected keys", True, f"keys: {keys}")
                # Check for messages or instruction format
                has_format = ("messages" in item or "instruction" in item
                             or "question" in item or "input" in item)
                vr.check("data format compatible", has_format,
                         f"need messages/instruction/question format")
        except Exception as e:
            vr.check("HF dataset accessible", False,
                     f"{args.dataset_name}: {str(e)[:100]}", warn_only=True)
            print(f"  [INFO] Fallback: you can use --local_data_path with a JSONL file")


def validate_dependencies(vr: ValidationResult):
    """Check required packages."""
    print("\n--- Dependencies ---")

    for pkg in ["torch", "transformers", "datasets", "peft", "numpy"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            vr.check(f"{pkg}", True, f"v{ver}")
        except ImportError:
            vr.check(f"{pkg}", False, f"pip install {pkg}")

    # Check bitsandbytes (optional but needed for 4-bit)
    try:
        import bitsandbytes
        vr.check("bitsandbytes (4-bit)", True, f"v{bitsandbytes.__version__}")
    except ImportError:
        vr.check("bitsandbytes (4-bit)", False,
                 "pip install bitsandbytes (needed for 4-bit quant)", warn_only=True)


def validate_serialization(args, vr: ValidationResult):
    """Test frequency serialization round-trip."""
    print("\n--- Serialization ---")

    inv_freq = compute_evq_cosh_inv_freq(args.head_dim, args.rope_base, args.tau)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    try:
        # Save
        torch.save({
            "inv_freq": inv_freq,
            "tau": args.tau,
            "head_dim": args.head_dim,
            "base": args.rope_base,
            "method": "evq_cosh",
        }, tmp_path)

        # Load
        loaded = torch.load(tmp_path, map_location="cpu", weights_only=True)
        loaded_freq = loaded["inv_freq"]

        max_err = (inv_freq - loaded_freq).abs().max().item()
        vr.check("save/load round-trip", max_err < 1e-12,
                 f"max_error={max_err:.2e}")
        vr.check("metadata preserved", loaded["tau"] == args.tau and
                 loaded["method"] == "evq_cosh")
    finally:
        os.unlink(tmp_path)


def generate_freq_plot(args, inv_evq, inv_geo):
    """Generate frequency comparison plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available for plotting")
        return

    K = args.head_dim // 2
    phi_evq = -torch.log(inv_evq).numpy() / math.log(args.rope_base)
    phi_geo = -torch.log(inv_geo).numpy() / math.log(args.rope_base)
    channels = np.arange(K)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: φ vs channel
    axes[0].plot(channels, phi_geo, "b--", label="Geometric (τ=0)", alpha=0.7)
    axes[0].plot(channels, phi_evq, "r-", label=f"EVQ-cosh (τ={args.tau})")
    axes[0].set_xlabel("Channel k")
    axes[0].set_ylabel("φ_k (normalized frequency)")
    axes[0].set_title("Frequency Allocation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: φ displacement
    displacement = phi_evq - phi_geo
    axes[1].bar(channels, displacement, color="coral", alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Channel k")
    axes[1].set_ylabel("Δφ_k (EVQ - Geo)")
    axes[1].set_title("Channel Displacement")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Log inv_freq
    axes[2].semilogy(channels, inv_geo, "b--", label="Geometric", alpha=0.7)
    axes[2].semilogy(channels, inv_evq, "r-", label=f"EVQ-cosh (τ={args.tau})")
    axes[2].set_xlabel("Channel k")
    axes[2].set_ylabel("inv_freq (log scale)")
    axes[2].set_title("Inverse Frequencies")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"freq_comparison_tau{args.tau}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  [PLOT] Saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="EVQ-Cosh Dry-Run Validator")
    p.add_argument("--tau", type=float, default=1.414)
    p.add_argument("--rope_base", type=float, default=500000.0)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--dataset_name", type=str, default="THUDM/LongAlign-10k")
    p.add_argument("--local_data_path", type=str, default=None)
    p.add_argument("--plot", action="store_true", help="Generate frequency comparison plot")
    return p.parse_args()


def main():
    args = parse_args()
    vr = ValidationResult()

    print("=" * 60)
    print("EVQ-COSH LORA DRY-RUN VALIDATION")
    print(f"  τ={args.tau}, r={args.lora_r}, d_head={args.head_dim}")
    print("=" * 60)

    # Run all validations
    validate_dependencies(vr)
    inv_evq, inv_geo = validate_frequencies(args, vr)
    validate_theory(args, vr)
    validate_data(args, vr)
    validate_serialization(args, vr)

    if args.plot:
        generate_freq_plot(args, inv_evq, inv_geo)

    # Summary
    all_pass = vr.summary()

    # Save validation report
    report = {
        "status": "PASS" if all_pass else "FAIL",
        "passed": vr.passed,
        "failed": vr.failed,
        "warnings": vr.warnings,
        "params": {
            "tau": args.tau,
            "lora_r": args.lora_r,
            "head_dim": args.head_dim,
            "rope_base": args.rope_base,
            "max_seq_len": args.max_seq_len,
            "dataset": args.dataset_name,
        },
        "checks": vr.checks,
    }
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "dryrun_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
