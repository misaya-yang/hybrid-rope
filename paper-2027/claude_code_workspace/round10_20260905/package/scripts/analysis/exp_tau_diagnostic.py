#!/usr/bin/env python3
"""
Experiment: τ* Scaling Law Diagnostic — Discriminating B vs C
=============================================================

Purpose:  Determine WHY τ*=d_head/√L fails at L≥4096.
          Two competing hypotheses:
            B: Continuous→discrete quantization loses information (finite-K effect)
            C: Softmax transport's 1/L Jacobian assumption breaks (N_eff ≠ L)

This script runs TWO independent experiments:
  Exp1 (zero-training): Pure numerical discrete collision analysis
  Exp2 (checkpoint):    N_eff estimation from real attention patterns

Design based on GPT/Claude joint analysis, 2026-03-30.

Usage:
  # Exp1 only (no GPU needed, ~2 min on CPU)
  python scripts/analysis/exp_tau_diagnostic.py --exp1

  # Exp2 only (needs checkpoint + GPU, ~10 min)
  python scripts/analysis/exp_tau_diagnostic.py --exp2 \
      --checkpoint results/m4_max_36gb/exp4_progressive_350m/geo_seed42/stage3_L2048/model.pt \
      --model-dim 1024 --n-heads 16 --n-layers 24 --head-dim 64 --vocab-size 50257

  # Both
  python scripts/analysis/exp_tau_diagnostic.py --exp1 --exp2 --checkpoint <path>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Exp 1: Zero-Training Discrete Collision Diagnosis
# ---------------------------------------------------------------------------
# Core idea (from GPT): If the discrete collision matrix's optimal τ already
# shifts rightward at large L (without any training), then B contributes.
# If it doesn't shift, C (softmax transport breakdown) is the sole cause.


def evq_phi(K: int, tau: float) -> np.ndarray:
    """Compute EVQ-Cosh φ_k schedule. Returns array of length K."""
    u = np.arange(K, dtype=np.float64) / K  # u_k = k/K, consistent with schedules.py
    if tau < 1e-8:
        return u.copy()
    sinh_tau = math.sinh(tau)
    phi = 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u) * sinh_tau)
    return phi


def evq_freqs(K: int, tau: float, base: float) -> np.ndarray:
    """Compute ω_k = base^{-φ_k}."""
    phi = evq_phi(K, tau)
    return base ** (-phi)


def discrete_collision_matrix(
    freqs: np.ndarray,
    L: int,
    D_alpha: float = 1.0,
) -> np.ndarray:
    """
    Compute the exact discrete phase-collision Gram matrix.

    G[i,j] = Σ_{Δ=1}^{L} D(Δ) · cos(ω_i·Δ) · cos(ω_j·Δ)

    where D(Δ) = Δ^{-α} (power-law distance prior).

    This is the GROUND TRUTH — no broadband approximation.
    """
    K = len(freqs)
    deltas = np.arange(1, L + 1, dtype=np.float64)
    weights = deltas ** (-D_alpha)  # D(Δ) = Δ^{-α}

    # phases[k, Δ] = ω_k · Δ
    phases = np.outer(freqs, deltas)  # (K, L)
    cos_phases = np.cos(phases)       # (K, L)

    # G = cos_phases @ diag(weights) @ cos_phases.T
    weighted = cos_phases * weights[np.newaxis, :]  # (K, L)
    G = weighted @ cos_phases.T  # (K, K)
    return G


def collision_score(G: np.ndarray) -> Dict[str, float]:
    """
    Extract scalar collision metrics from the Gram matrix.

    Returns:
      - off_diag_energy: Σ_{i≠j} |G[i,j]|  (lower = less collision)
      - mutual_coherence: max_{i≠j} |G[i,j]| / max_i G[i,i]
      - condition_number: λ_max / λ_min
      - total_collision: Frobenius norm of off-diagonal part
    """
    K = G.shape[0]
    diag = np.diag(G).copy()
    off_diag = G.copy()
    np.fill_diagonal(off_diag, 0.0)

    off_energy = float(np.sum(np.abs(off_diag)))
    max_diag = float(np.max(np.abs(diag)))
    max_off = float(np.max(np.abs(off_diag)))
    mu = max_off / max_diag if max_diag > 0 else float("inf")

    eigvals = np.linalg.eigvalsh(G)
    eigvals_pos = eigvals[eigvals > 1e-12]
    if len(eigvals_pos) >= 2:
        cond = float(eigvals_pos[-1] / eigvals_pos[0])
    else:
        cond = float("inf")

    frob_off = float(np.sqrt(np.sum(off_diag ** 2)))

    return {
        "off_diag_energy": off_energy,
        "mutual_coherence": mu,
        "condition_number": cond,
        "frobenius_off_diag": frob_off,
    }


def chi2_stiffness(tau: float, K: int) -> float:
    """Pearson χ² stiffness S_χ²(τ) from softmax transport theory (§4.7.2)."""
    if tau < 1e-8:
        return 0.0
    s = math.sinh(tau)
    return (1.0 / K) * (s * math.atan(s) / (tau ** 2) - 1.0)


def run_exp1(
    bases: List[float] = [10_000.0, 500_000.0],
    Ks: List[int] = [16, 32, 64],
    Ls: List[int] = [256, 512, 1024, 2048, 4096, 8192],
    tau_range: np.ndarray = np.linspace(0.1, 6.0, 60),
    D_alpha: float = 1.0,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Exp1: For each (base, K, L), sweep τ and find the discrete collision optimum.
    Compare with τ*_theory = 2K/√L (since d_head = 2K).

    Key diagnostic:
      If discrete_optimal_τ >> τ*_theory at large L → B confirmed
      If discrete_optimal_τ ≈ τ*_theory at large L → B excluded, C confirmed
    """
    results = {}
    total = len(bases) * len(Ks) * len(Ls)
    done = 0

    for base in bases:
        for K in Ks:
            d_head = 2 * K
            for L in Ls:
                tau_theory = d_head / math.sqrt(L)
                config_key = f"base={base:.0f}_K={K}_L={L}"

                # Limit Δ summation for large L to keep runtime sane
                L_eff_sum = min(L, 8192)

                best_tau = None
                best_score = float("inf")
                sweep_data = []

                for tau in tau_range:
                    freqs = evq_freqs(K, tau, base)
                    G = discrete_collision_matrix(freqs, L_eff_sum, D_alpha)
                    sc = collision_score(G)

                    sweep_data.append({
                        "tau": float(tau),
                        **sc,
                    })

                    # Use off-diagonal energy as primary objective
                    if sc["off_diag_energy"] < best_score:
                        best_score = sc["off_diag_energy"]
                        best_tau = float(tau)

                # Also compute geometric baseline
                geo_freqs = evq_freqs(K, 0.0, base)
                G_geo = discrete_collision_matrix(geo_freqs, L_eff_sum, D_alpha)
                geo_score = collision_score(G_geo)

                # Compute theory τ* score
                freqs_theory = evq_freqs(K, tau_theory, base)
                G_theory = discrete_collision_matrix(freqs_theory, L_eff_sum, D_alpha)
                theory_score = collision_score(G_theory)

                results[config_key] = {
                    "base": base,
                    "K": K,
                    "d_head": d_head,
                    "L": L,
                    "tau_theory": round(tau_theory, 4),
                    "tau_discrete_opt": round(best_tau, 4),
                    "ratio_disc_over_theory": round(best_tau / tau_theory, 3) if tau_theory > 0 else None,
                    "geo_off_diag_energy": round(geo_score["off_diag_energy"], 4),
                    "theory_off_diag_energy": round(theory_score["off_diag_energy"], 4),
                    "optimal_off_diag_energy": round(best_score, 4),
                    "improvement_vs_geo_pct": round(
                        100 * (1 - best_score / geo_score["off_diag_energy"]), 2
                    ) if geo_score["off_diag_energy"] > 0 else None,
                    "geo_mutual_coherence": round(geo_score["mutual_coherence"], 6),
                    "opt_mutual_coherence": round(
                        sweep_data[np.argmin([s["off_diag_energy"] for s in sweep_data])]["mutual_coherence"], 6
                    ),
                    "sweep": sweep_data,
                }

                done += 1
                print(f"  [{done}/{total}] {config_key}: "
                      f"τ*_theory={tau_theory:.3f}, τ*_discrete={best_tau:.3f}, "
                      f"ratio={best_tau/tau_theory:.2f}x")

    # Summary table: focus on the key diagnostic
    print("\n" + "=" * 90)
    print("EXP1 SUMMARY: Does discrete optimal τ shift rightward at large L?")
    print("=" * 90)
    print(f"{'Config':<35} {'τ*_theory':>10} {'τ*_disc':>10} {'Ratio':>8} {'Δ vs Geo':>10}")
    print("-" * 90)
    for key, r in results.items():
        print(f"{key:<35} {r['tau_theory']:>10.3f} {r['tau_discrete_opt']:>10.3f} "
              f"{r['ratio_disc_over_theory']:>8.2f}x {r['improvement_vs_geo_pct']:>9.1f}%")

    print("\n" + "=" * 90)
    print("INTERPRETATION:")
    print("  If ratio ≈ 1.0 across all L  → B excluded, problem is purely C (softmax transport)")
    print("  If ratio grows with L         → B contributes (finite-K effect shifts optimum)")
    print("  If ratio grows with 1/K       → B is K-dependent (MLA more affected)")
    print("=" * 90)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        saveable = {k: {kk: vv for kk, vv in v.items() if kk != "sweep"}
                    for k, v in results.items()}
        with open(output_dir / "exp1a_collision_only.json", "w") as f:
            json.dump(saveable, f, indent=2)
        print(f"\nResults saved to {output_dir}/exp1a_collision_only.json")

    # ---- Exp1b: Full objective F(τ) = S_χ²(τ) - λ·U_discrete(τ,L) ----
    # This is the decisive test: does the STATIC discrete objective predict
    # a τ* that depends on L?
    #
    # Result from dry-run: τ*_static ≈ constant (no L dependence!)
    # This proves: ALL L-dependence in τ* comes from softmax transport.
    print("\n" + "=" * 90)
    print("EXP 1b: Full Static Objective (Stiffness + Discrete Collision)")
    print("  F(τ) = S_χ²(τ) - λ · [CollisionReduction(τ,L)]")
    print("  Calibrate λ at L=512 to match τ*≈d_head/√512, then predict other L")
    print("=" * 90)

    exp1b_results = {}
    for base in bases:
        for K in Ks:
            d_head = 2 * K

            # Calibrate λ: find value that gives τ* ≈ d_head/√512 at L=512
            tau_target_512 = d_head / math.sqrt(512)
            best_lam = None
            best_gap = float("inf")

            for lam_try in np.linspace(0.01, 2.0, 200):
                F_vals = []
                for tau in tau_range:
                    S = chi2_stiffness(tau, K)
                    geo_f = evq_freqs(K, 1e-8, base)
                    evq_f = evq_freqs(K, tau, base)
                    G_geo = discrete_collision_matrix(geo_f, min(512, 8192), D_alpha)
                    G_evq = discrete_collision_matrix(evq_f, min(512, 8192), D_alpha)
                    geo_e = collision_score(G_geo)["off_diag_energy"]
                    evq_e = collision_score(G_evq)["off_diag_energy"]
                    U = (geo_e - evq_e) / geo_e if geo_e > 0 else 0.0
                    F_vals.append(S - lam_try * U)
                opt_tau = tau_range[np.argmin(F_vals)]
                gap = abs(opt_tau - tau_target_512)
                if gap < best_gap:
                    best_gap = gap
                    best_lam = lam_try

            config_key = f"base={base:.0f}_K={K}"
            print(f"\n  {config_key}: Calibrated λ={best_lam:.4f} "
                  f"(target τ*={tau_target_512:.2f} at L=512)")

            # Now predict τ* at all L using calibrated λ
            print(f"  {'L':>8} {'τ*_theory':>10} {'τ*_static':>12} {'Ratio':>8}")
            print(f"  {'-'*45}")
            config_results = {"lambda_calibrated": best_lam, "per_L": {}}

            for L in Ls:
                F_vals = []
                for tau in tau_range:
                    S = chi2_stiffness(tau, K)
                    geo_f = evq_freqs(K, 1e-8, base)
                    evq_f = evq_freqs(K, tau, base)
                    G_geo = discrete_collision_matrix(geo_f, min(L, 8192), D_alpha)
                    G_evq = discrete_collision_matrix(evq_f, min(L, 8192), D_alpha)
                    geo_e = collision_score(G_geo)["off_diag_energy"]
                    evq_e = collision_score(G_evq)["off_diag_energy"]
                    U = (geo_e - evq_e) / geo_e if geo_e > 0 else 0.0
                    F_vals.append(S - best_lam * U)
                opt_tau = tau_range[np.argmin(F_vals)]
                tau_theory = d_head / math.sqrt(L)
                ratio = opt_tau / tau_theory if tau_theory > 0 else float("inf")
                print(f"  {L:>8} {tau_theory:>10.3f} {opt_tau:>12.2f} {ratio:>8.2f}x")
                config_results["per_L"][str(L)] = {
                    "tau_theory": round(tau_theory, 4),
                    "tau_static": round(float(opt_tau), 4),
                    "ratio": round(ratio, 3),
                }

            exp1b_results[config_key] = config_results

    print("\n" + "=" * 90)
    print("EXP 1b CRITICAL FINDING:")
    print("  If τ*_static ≈ constant across L → L-dependence is ENTIRELY from softmax")
    print("  This means the τ* failure at L≥4096 is hypothesis C (softmax transport),")
    print("  NOT hypothesis B (discrete quantization).")
    print("=" * 90)

    if output_dir:
        with open(output_dir / "exp1b_full_objective.json", "w") as f:
            json.dump(exp1b_results, f, indent=2)
        print(f"\nResults saved to {output_dir}/exp1b_full_objective.json")

    results["_exp1b"] = exp1b_results
    return results


# ---------------------------------------------------------------------------
# Exp 2: N_eff Estimation from Checkpoint Attention Patterns
# ---------------------------------------------------------------------------
# Core idea (from GPT): If the real attention distribution has N_eff << L,
# then τ* = d_head/√N_eff (not d_head/√L) should better predict the optimum.
# Measure N_eff from real checkpoints and check.


def compute_n_eff_from_attn(
    attn_weights: np.ndarray,
    method: str = "both",
) -> Dict[str, float]:
    """
    Compute effective competition scale from attention weight matrix.

    attn_weights: shape (n_heads, seq_len, seq_len), softmax-normalized attention.
                  Causal: attn[h, q, k] = 0 for k > q.

    Two methods:
      1. Inverse participation ratio: N_eff = 1/Σ_Δ p(Δ)²
         where p(Δ) = mean attention mass at distance Δ
      2. Entropy: N_eff = exp(H(p)) where H = -Σ p(Δ) log p(Δ)

    Returns per-head and aggregate N_eff values.
    """
    n_heads, seq_len, _ = attn_weights.shape
    max_dist = seq_len

    # Build distance-weighted attention histogram per head
    # p_h(Δ) = (1/Q) Σ_q attn[h, q, q-Δ]  (average attention at distance Δ)
    head_neffs_ipr = []
    head_neffs_entropy = []
    per_head_alpha = []

    for h in range(n_heads):
        # Accumulate p(Δ) for this head
        dist_hist = np.zeros(max_dist, dtype=np.float64)
        count_hist = np.zeros(max_dist, dtype=np.float64)

        for q in range(seq_len):
            for k in range(q + 1):  # causal
                d = q - k
                if d < max_dist:
                    dist_hist[d] += attn_weights[h, q, k]
                    count_hist[d] += 1.0

        # Normalize: average attention mass at each distance
        mask = count_hist > 0
        p = np.zeros(max_dist, dtype=np.float64)
        p[mask] = dist_hist[mask] / count_hist[mask]

        # Re-normalize p to be a proper distribution
        p_sum = p.sum()
        if p_sum > 1e-12:
            p_norm = p / p_sum
        else:
            p_norm = np.ones(max_dist) / max_dist

        # IPR: N_eff = 1 / Σ p²
        ipr = float(np.sum(p_norm ** 2))
        n_eff_ipr = 1.0 / ipr if ipr > 0 else float(max_dist)

        # Entropy: N_eff = exp(H)
        p_pos = p_norm[p_norm > 1e-30]
        H = float(-np.sum(p_pos * np.log(p_pos)))
        n_eff_entropy = float(np.exp(H))

        head_neffs_ipr.append(n_eff_ipr)
        head_neffs_entropy.append(n_eff_entropy)

        # Power-law fit for this head
        from scripts.lib.rope.attn_hist import fit_power_law
        pl_fit = fit_power_law(p, d_min=2, d_max=min(max_dist - 1, seq_len // 2))
        per_head_alpha.append(pl_fit.get("alpha"))

    return {
        "n_eff_ipr_per_head": head_neffs_ipr,
        "n_eff_entropy_per_head": head_neffs_entropy,
        "n_eff_ipr_mean": float(np.mean(head_neffs_ipr)),
        "n_eff_ipr_median": float(np.median(head_neffs_ipr)),
        "n_eff_entropy_mean": float(np.mean(head_neffs_entropy)),
        "n_eff_entropy_median": float(np.median(head_neffs_entropy)),
        "alpha_per_head": per_head_alpha,
        "alpha_mean": float(np.nanmean([a for a in per_head_alpha if a is not None])),
        "seq_len": int(seq_len),
        "n_heads": int(n_heads),
        "ratio_neff_ipr_over_L": float(np.mean(head_neffs_ipr) / seq_len),
        "ratio_neff_entropy_over_L": float(np.mean(head_neffs_entropy) / seq_len),
    }


def compute_n_eff_vectorized(
    attn_weights: np.ndarray,
) -> Dict[str, float]:
    """
    Faster vectorized version for larger seq_len.
    attn_weights: shape (n_heads, seq_len, seq_len).
    """
    n_heads, seq_len, _ = attn_weights.shape
    max_dist = seq_len

    head_neffs_ipr = []
    head_neffs_entropy = []

    for h in range(n_heads):
        attn_h = attn_weights[h]  # (seq_len, seq_len)

        # Build distance histogram efficiently
        dist_hist = np.zeros(max_dist, dtype=np.float64)
        count_hist = np.zeros(max_dist, dtype=np.float64)

        for d in range(max_dist):
            # Diagonal d: attn[q, q-d] for q >= d
            diag_vals = np.diag(attn_h, k=-d)  # elements where col = row - d
            if len(diag_vals) > 0:
                dist_hist[d] = float(np.sum(diag_vals))
                count_hist[d] = float(len(diag_vals))

        mask = count_hist > 0
        p = np.zeros(max_dist, dtype=np.float64)
        p[mask] = dist_hist[mask] / count_hist[mask]
        p_sum = p.sum()
        if p_sum > 1e-12:
            p_norm = p / p_sum
        else:
            p_norm = np.ones(max_dist) / max_dist

        ipr = float(np.sum(p_norm ** 2))
        n_eff_ipr = 1.0 / ipr if ipr > 0 else float(max_dist)

        p_pos = p_norm[p_norm > 1e-30]
        H = float(-np.sum(p_pos * np.log(p_pos)))
        n_eff_entropy = float(np.exp(H))

        head_neffs_ipr.append(n_eff_ipr)
        head_neffs_entropy.append(n_eff_entropy)

    return {
        "n_eff_ipr_per_head": head_neffs_ipr,
        "n_eff_entropy_per_head": head_neffs_entropy,
        "n_eff_ipr_mean": float(np.mean(head_neffs_ipr)),
        "n_eff_entropy_mean": float(np.mean(head_neffs_entropy)),
        "seq_len": int(seq_len),
        "n_heads": int(n_heads),
        "ratio_neff_ipr_over_L": float(np.mean(head_neffs_ipr) / seq_len),
        "ratio_neff_entropy_over_L": float(np.mean(head_neffs_entropy) / seq_len),
    }


def run_exp2_from_checkpoint(
    checkpoint_path: str,
    model_dim: int = 1024,
    n_heads: int = 16,
    n_layers: int = 24,
    head_dim: int = 64,
    vocab_size: int = 50257,
    eval_lengths: List[int] = [512, 1024, 2048, 4096],
    n_samples: int = 4,
    sample_layers: List[int] = [0, 6, 12, 18, 23],
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Exp2: Load a checkpoint, run forward passes at multiple seq_len,
    extract attention patterns, and compute N_eff.

    Key diagnostic:
      If N_eff(L) grows much slower than L → C confirmed
      Specifically: if τ*_neff = d_head/√N_eff matches experiments → smoking gun
    """
    import torch
    import torch.nn as nn

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Infer model architecture from state dict keys
    # Try to detect if it's a raw state_dict or has 'model' key
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # Build a minimal GPT model for attention extraction
    from scripts.analysis._minimal_gpt import MinimalGPT, MinimalGPTConfig

    config = MinimalGPTConfig(
        vocab_size=vocab_size,
        n_layer=n_layers,
        n_head=n_heads,
        n_embd=model_dim,
        max_seq_len=max(eval_lengths),
    )
    model = MinimalGPT(config)

    # Load weights (best effort)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Model loaded on {device}")

    # Clamp sample_layers to valid range
    sample_layers = [l for l in sample_layers if l < n_layers]

    results = {}

    for L in eval_lengths:
        print(f"\n--- Evaluating N_eff at seq_len={L} ---")

        all_neffs_ipr = []
        all_neffs_entropy = []

        for sample_idx in range(n_samples):
            # Random token sequence
            torch.manual_seed(42 + sample_idx)
            input_ids = torch.randint(0, vocab_size, (1, L), device=device)

            with torch.no_grad():
                attns = model.forward_with_attention(input_ids, layers=sample_layers)

            # attns: dict of {layer_idx: (1, n_heads, L, L) numpy}
            for layer_idx, attn_np in attns.items():
                attn_np = attn_np[0]  # remove batch dim → (n_heads, L, L)
                neff = compute_n_eff_vectorized(attn_np)
                all_neffs_ipr.extend(neff["n_eff_ipr_per_head"])
                all_neffs_entropy.extend(neff["n_eff_entropy_per_head"])

        n_eff_ipr = float(np.mean(all_neffs_ipr))
        n_eff_entropy = float(np.mean(all_neffs_entropy))
        tau_neff_ipr = head_dim / math.sqrt(n_eff_ipr) if n_eff_ipr > 0 else float("inf")
        tau_neff_entropy = head_dim / math.sqrt(n_eff_entropy) if n_eff_entropy > 0 else float("inf")
        tau_theory = head_dim / math.sqrt(L)

        results[f"L={L}"] = {
            "L": L,
            "n_eff_ipr": round(n_eff_ipr, 1),
            "n_eff_entropy": round(n_eff_entropy, 1),
            "ratio_ipr_over_L": round(n_eff_ipr / L, 4),
            "ratio_entropy_over_L": round(n_eff_entropy / L, 4),
            "tau_theory": round(tau_theory, 4),
            "tau_from_neff_ipr": round(tau_neff_ipr, 4),
            "tau_from_neff_entropy": round(tau_neff_entropy, 4),
            "n_measurements": len(all_neffs_ipr),
        }

        print(f"  N_eff(IPR)={n_eff_ipr:.0f} ({n_eff_ipr/L:.2%} of L), "
              f"N_eff(H)={n_eff_entropy:.0f} ({n_eff_entropy/L:.2%} of L)")
        print(f"  τ*_theory={tau_theory:.3f}, "
              f"τ*_neff(IPR)={tau_neff_ipr:.3f}, "
              f"τ*_neff(H)={tau_neff_entropy:.3f}")

    # Summary
    print("\n" + "=" * 90)
    print("EXP2 SUMMARY: Does N_eff grow slower than L?")
    print("=" * 90)
    print(f"{'L':>8} {'N_eff(IPR)':>12} {'N_eff/L':>10} {'τ*_theory':>10} {'τ*_neff':>10} {'Ratio':>8}")
    print("-" * 90)
    for key, r in results.items():
        ratio = r["tau_from_neff_ipr"] / r["tau_theory"] if r["tau_theory"] > 0 else float("inf")
        print(f"{r['L']:>8} {r['n_eff_ipr']:>12.0f} {r['ratio_ipr_over_L']:>10.3f} "
              f"{r['tau_theory']:>10.3f} {r['tau_from_neff_ipr']:>10.3f} {ratio:>8.2f}x")

    print("\nINTERPRETATION:")
    print("  If N_eff/L shrinks as L grows → 1/L Jacobian is wrong → C confirmed")
    print("  If τ*_neff ≈ experimental optimal τ (~1.4) at L=4096 → smoking gun for C")
    print("  If N_eff/L ≈ constant → softmax transport is fine → look elsewhere")
    print("=" * 90)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "exp2_neff.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_dir}/exp2_neff.json")

    return results


# ---------------------------------------------------------------------------
# Minimal GPT for Exp2 (standalone, no HuggingFace dependency)
# ---------------------------------------------------------------------------
# This is written to a separate file for clarity.

def _write_minimal_gpt_module(output_path: Path):
    """Write the minimal GPT module needed for Exp2."""
    code = '''#!/usr/bin/env python3
"""Minimal GPT for attention extraction. No HuggingFace dependency."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MinimalGPTConfig:
    vocab_size: int = 50257
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    max_seq_len: int = 8192
    base: float = 500000.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MinimalGPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Precompute RoPE freqs
        inv_freq = 1.0 / (config.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float64) / self.head_dim
        ))
        self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, L, D)"""
        B, H, L, D = x.shape
        positions = torch.arange(L, device=x.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)  # (L, D/2)
        cos_a = angles.cos().float().unsqueeze(0).unsqueeze(0)  # (1,1,L,D/2)
        sin_a = angles.sin().float().unsqueeze(0).unsqueeze(0)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * cos_a - x2 * sin_a
        rx2 = x1 * sin_a + x2 * cos_a
        return torch.stack([rx1, rx2], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, L, C = x.shape
        qkv = self.c_attn(x).reshape(B, L, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, H, D)
        q = q.transpose(1, 2)  # (B, H, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_w = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_w = attn_w.masked_fill(causal_mask, float("-inf"))
        attn_w = F.softmax(attn_w, dim=-1)

        out = (attn_w @ v).transpose(1, 2).reshape(B, L, C)
        out = self.c_proj(out)

        if return_attn:
            return out, attn_w.detach().cpu().numpy()
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: MinimalGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn_w = self.attn(self.ln_1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn_w
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MinimalGPT(nn.Module):
    def __init__(self, config: MinimalGPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_seq_len, config.n_embd)  # may not be used with RoPE
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward_with_attention(
        self,
        input_ids: torch.Tensor,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        """Forward pass returning attention weights for specified layers."""
        B, L = input_ids.shape
        x = self.wte(input_ids)
        # Note: if model uses RoPE, wpe may be zero/unused

        attn_dict = {}
        if layers is None:
            layers = list(range(len(self.blocks)))

        for i, block in enumerate(self.blocks):
            if i in layers:
                x, attn_w = block(x, return_attn=True)
                attn_dict[i] = attn_w  # (B, H, L, L) numpy
            else:
                x = block(x)

        return attn_dict

    def load_state_dict(self, state_dict, strict=True):
        """Flexible loading with key remapping."""
        # Try direct load first
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            pass

        # Try remapping common key patterns
        new_sd = {}
        for k, v in state_dict.items():
            # Strip common prefixes
            nk = k.replace("transformer.", "").replace("model.", "")
            new_sd[nk] = v

        return super().load_state_dict(new_sd, strict=False)
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(code)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="τ* Scaling Law Diagnostic: B vs C discrimination"
    )
    parser.add_argument("--exp1", action="store_true",
                        help="Run Exp1: zero-training discrete collision (CPU only)")
    parser.add_argument("--exp2", action="store_true",
                        help="Run Exp2: N_eff from checkpoint attention")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint for Exp2")
    parser.add_argument("--model-dim", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--output-dir", type=str,
                        default="results/tau_diagnostic")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if not args.exp1 and not args.exp2:
        print("No experiment selected. Use --exp1 and/or --exp2.")
        parser.print_help()
        sys.exit(1)

    if args.exp1:
        print("=" * 90)
        print("EXP 1: Zero-Training Discrete Collision Diagnosis")
        print("   Sweeps τ on the EXACT discrete collision matrix (no broadband approx)")
        print("   If discrete optimal τ >> theory τ* at large L → finite-K effect (B)")
        print("=" * 90)
        t0 = time.time()
        run_exp1(
            bases=[10_000.0, 500_000.0],
            Ks=[16, 32, 64],
            Ls=[256, 512, 1024, 2048, 4096, 8192],
            tau_range=np.linspace(0.1, 5.0, 50),
            D_alpha=1.0,
            output_dir=output_dir,
        )
        print(f"\nExp1 completed in {time.time()-t0:.1f}s")

    if args.exp2:
        if args.checkpoint is None:
            print("ERROR: --checkpoint required for Exp2")
            sys.exit(1)

        # Write the minimal GPT module
        _write_minimal_gpt_module(
            Path("scripts/analysis/_minimal_gpt.py")
        )

        print("\n" + "=" * 90)
        print("EXP 2: N_eff Estimation from Checkpoint Attention")
        print("   Measures effective attention competition scale at multiple seq_len")
        print("   If N_eff << L and grows sublinearly → softmax transport is wrong (C)")
        print("=" * 90)
        t0 = time.time()
        run_exp2_from_checkpoint(
            checkpoint_path=args.checkpoint,
            model_dim=args.model_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            head_dim=args.head_dim,
            vocab_size=args.vocab_size,
            eval_lengths=[512, 1024, 2048, 4096],
            n_samples=4,
            output_dir=output_dir,
        )
        print(f"\nExp2 completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
