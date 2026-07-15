#!/usr/bin/env python3
"""Single-96GB feasibility + tau/coverage study for the 128K EVQ-vs-YaRN LoRA run.

Two questions:
  (1) What fits on ONE RTX Pro 6000 (96 GB) for BF16 LoRA train and 128K eval?
  (2) How to set tau for a model whose base != 500000 and whose train length
      != 8192, using the project's exact EVQ-Cosh schedule?

Pure NumPy, no torch. The EVQ-Cosh schedule mirrors
experiments/lora_evq_v2/train_evq_lora.py:compute_evq_cosh_inv_freq:
    phi_k(tau) = 1 - (1/tau) * arcsinh((1 - u_k) * sinh(tau)),  u_k = (2k-1)/(2K)
    inv_freq_k = base ** (-phi_k)
The dormant-channel metric is validated against the registered 8B/8K anchors
(native=20, midpoint=20, evq=15 at tau=1.414).
"""
from __future__ import annotations

import math
import numpy as np

GB = 1024 ** 3

# ---------------------------------------------------------------------------
# Model configs (verify each against the shipped config.json before launch).
# ---------------------------------------------------------------------------
MODELS = {
    "Llama-3.1-8B": dict(params=8.03e9, layers=32, d_model=4096, n_heads=32,
                         n_kv=8, head_dim=128, d_ff=14336, vocab=128256,
                         base=500000.0, native=8192),
    "Qwen2.5-7B":   dict(params=7.62e9, layers=28, d_model=3584, n_heads=28,
                         n_kv=4, head_dim=128, d_ff=18944, vocab=152064,
                         base=1000000.0, native=32768),
    "Qwen2.5-14B":  dict(params=14.77e9, layers=48, d_model=5120, n_heads=40,
                         n_kv=8, head_dim=128, d_ff=13824, vocab=152064,
                         base=1000000.0, native=32768),
    "Qwen2.5-32B":  dict(params=32.5e9, layers=64, d_model=5120, n_heads=40,
                         n_kv=8, head_dim=128, d_ff=27648, vocab=152064,
                         base=1000000.0, native=32768),
}

# ---------------------------------------------------------------------------
# RoPE inverse-frequency schedules
# ---------------------------------------------------------------------------
def geo_inv_freq(head_dim: int, base: float) -> np.ndarray:
    i = np.arange(0, head_dim, 2, dtype=np.float64)
    return base ** (-(i / head_dim))


def evq_cosh_inv_freq(head_dim: int, base: float, tau: float) -> np.ndarray:
    K = head_dim // 2
    k = np.arange(1, K + 1, dtype=np.float64)
    u = (2 * k - 1) / (2 * K)                 # midpoint quantization
    phi = 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u) * math.sinh(tau))
    return base ** (-phi)


def yarn_inv_freq(head_dim: int, base: float, factor: float, orig_ctx: int,
                  beta_fast: float = 32.0, beta_slow: float = 1.0):
    """HF-standard YaRN (NTK-by-parts). Returns (inv_freq, attention_factor)."""
    dim = head_dim

    def corr_dim(num_rot):
        return (dim * math.log(orig_ctx / (num_rot * 2 * math.pi))) / (2 * math.log(base))

    low = math.floor(corr_dim(beta_fast))
    high = math.ceil(corr_dim(beta_slow))
    low, high = max(low, 0), min(high, dim - 1)

    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0, 1)
    extrap_factor = 1.0 - ramp                       # 1 keep base (high freq), 0 interpolate
    inv_freq = inv_interp * (1 - extrap_factor) + inv_extrap * extrap_factor
    attn_factor = 0.1 * math.log(factor) + 1.0       # mscale (softmax temperature)
    return inv_freq, attn_factor


# ---------------------------------------------------------------------------
# Coverage diagnostics
# ---------------------------------------------------------------------------
def dormant_count(inv_freq: np.ndarray, L: int) -> int:
    """Channels that rotate < 1 rad over the whole train length (omega*L < 1)."""
    return int(np.sum(inv_freq * L < 1.0))


def wrapped_fraction(inv_freq: np.ndarray, T: int) -> float:
    """Fraction of channels completing > 1 full turn over test length T (alias risk)."""
    return float(np.mean(inv_freq * T > 2 * math.pi))


def entropy_effective_rank(inv_freq: np.ndarray, L: int, prior: str = "uniform") -> float:
    """exp(entropy) of the phase-feature Gram eigenspectrum over distances 0..L-1.

    Features per distance d: [cos(omega_k d), sin(omega_k d)] over channels.
    Mirrors the project's entropy_effective_rank definition.
    """
    step = max(1, L // 4096)                          # subsample long L for speed
    d = np.arange(0, L, step, dtype=np.float64)
    if prior == "uniform":
        w = np.ones_like(d)
    else:                                             # causal-triangular weight L-d
        w = (L - d)
    w = w / w.sum()
    ang = np.outer(d, inv_freq)                       # [D, K]
    F = np.concatenate([np.cos(ang), np.sin(ang)], axis=1)   # [D, 2K]
    C = (F * w[:, None]).T @ F                        # [2K, 2K] weighted Gram
    C = 0.5 * (C + C.T)
    ev = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    p = ev[ev > 1e-14]
    p = p / p.sum()
    return float(math.exp(-(p * np.log(p)).sum()))


def recommend_tau(head_dim, base, train_L, test_L, tau_grid,
                  anchor_dormant=15, anchor_erank=36.2):
    """Pick tau reproducing the VALIDATED coverage at a new (base, train_L).

    The 8B/8K operating point tau=1.414 gives dormant=15 active-channel coverage.
    We transfer that coverage (match dormant to the anchor) instead of reusing the
    scalar 1.414, because the tau that yields a given coverage shifts with base and
    train length. Tie-break toward LOWER tau (less high-frequency aliasing at the
    test length). Raising tau monotonically lifts entropy rank but also lifts test
    aliasing, so 'maximize erank' is not a valid objective on its own.
    """
    rows = []
    for tau in tau_grid:
        f = evq_cosh_inv_freq(head_dim, base, tau)
        rows.append(dict(tau=tau,
                         dormant=dormant_count(f, train_L),
                         erank=entropy_effective_rank(f, train_L),
                         wrap_test=wrapped_fraction(f, test_L)))
    best = min(rows, key=lambda r: (abs(r["dormant"] - anchor_dormant),
                                    abs(r["erank"] - anchor_erank)))
    return best, rows


# ---------------------------------------------------------------------------
# VRAM model (BF16 base frozen; LoRA q,k,v,o; full grad checkpointing; flash;
# answer-only tail logits). Documented, auditable, deliberately a bit generous.
# ---------------------------------------------------------------------------
def lora_trainable_params(m, rank):
    dq = m["n_heads"] * m["head_dim"]
    dkv = m["n_kv"] * m["head_dim"]
    per_layer = (rank * (m["d_model"] + dq)      # q
                 + rank * (m["d_model"] + dkv)   # k
                 + rank * (m["d_model"] + dkv)   # v
                 + rank * (dq + m["d_model"]))   # o
    return per_layer * m["layers"]


def vram_train_gb(m, L, microbatch=1, rank=64, answer_tokens=16):
    weights = 2 * m["params"]
    lora_opt = lora_trainable_params(m, rank) * (2 + 4 + 4 + 4)   # bf16 grad + fp32 master+m+v
    ckpt = m["layers"] * L * m["d_model"] * 2 * microbatch        # stored layer inputs (bf16)
    recompute = microbatch * L * m["d_ff"] * 2                    # transient MLP intermediate
    logits = microbatch * answer_tokens * m["vocab"] * 4          # fp32 CE on tail only
    total = (weights + lora_opt + ckpt + recompute + logits)
    return (total * 1.15) / GB + 2.0                             # frag + CUDA context


def vram_eval_nll_gb(m, L, microbatch=1, answer_tokens=16):
    weights = 2 * m["params"]
    act = microbatch * L * m["d_model"] * 2 * 3                  # no backward: few live buffers
    logits = microbatch * answer_tokens * m["vocab"] * 4
    return ((weights + act + logits) * 1.15) / GB + 2.0


def vram_eval_gen_gb(m, T, microbatch=1):
    weights = 2 * m["params"]
    kv = 2 * m["layers"] * m["n_kv"] * m["head_dim"] * T * 2 * microbatch
    work = microbatch * T * m["d_model"] * 2
    return ((weights + kv + work) * 1.10) / GB + 2.0


# ---------------------------------------------------------------------------
def main():
    CAP = 96.0
    line = "=" * 78

    print(line); print("A. DIAGNOSTIC VALIDATION against registered 8B/8K anchors")
    print("   expect dormant native=20, midpoint=20, evq=15 ; erank native~23.6 evq~36.2")
    hd, base8, L8 = 128, 500000.0, 8192
    geo = geo_inv_freq(hd, base8)
    mid = evq_cosh_inv_freq(hd, base8, 1e-6)          # tau->0 == midpoint geo
    evq = evq_cosh_inv_freq(hd, base8, 1.414)
    for name, f in [("native", geo), ("midpoint", mid), ("evq_tau1.414", evq)]:
        print(f"   {name:14s} dormant(8K)={dormant_count(f, L8):2d}  "
              f"erank(8K)={entropy_effective_rank(f, L8):6.2f}  "
              f"wrap(32K)={wrapped_fraction(f, 32768):.3f}")

    print(line); print("B. TAU STUDY: reproduce the validated coverage (dormant~=15), do")
    print("   NOT reuse 1.414 blindly. Higher tau lifts erank AND test aliasing.")
    grid = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.414, 1.6, 1.8, 2.0]
    regimes = [(500000.0, 8192, 32768, "Llama-3.1-8B  base 5e5  train 8K  -> test 32K"),
               (500000.0, 16384, 65536, "Llama-3.1-8B  base 5e5  train 16K -> test 64K"),
               (1000000.0, 16384, 65536, "Qwen2.5-14B   base 1e6  train 16K -> test 64K"),
               (1000000.0, 32768, 131072, "Qwen2.5-14B   base 1e6  train 32K -> test 128K")]
    for base, L, T, tag in regimes:
        best, rows = recommend_tau(hd, base, L, T, grid, anchor_dormant=15)
        print(f"\n  [{tag}]")
        print(f"    -> pick tau={best['tau']:.3f}  (dormant={best['dormant']}, "
              f"erank={best['erank']:.1f}, wrap_test={best['wrap_test']:.2f})")
        near = [r for r in rows if abs(r["tau"] - best["tau"]) <= 0.31]
        print("       dev grid: " + "  ".join(
            f"t{r['tau']:.2f}(d{r['dormant']},e{r['erank']:.0f})" for r in near))

    print(line); print("C. TRAIN VRAM (GB) on 96 GB, BF16, rank64, microbatch=1, grad-ckpt")
    Ls = [8192, 16384, 32768, 65536, 131072]
    print(f"   {'model':14s} " + " ".join(f"{L//1024:>3d}K" for L in Ls))
    for name, m in MODELS.items():
        cells = []
        for L in Ls:
            v = vram_train_gb(m, L, microbatch=1, rank=64)
            cells.append(f"{v:5.0f}" + ("!" if v > CAP else " "))
        print(f"   {name:14s} " + " ".join(cells))
    print("   ('!' = exceeds 96 GB at microbatch=1; grad-accum keeps effective batch)")

    print(line); print("D. EVAL VRAM (GB): teacher-forced NLL (1 fwd) and 128K generation KV")
    print(f"   {'model':14s}  NLL@32K NLL@128K  GEN@32K GEN@128K")
    for name, m in MODELS.items():
        print(f"   {name:14s}  {vram_eval_nll_gb(m,32768):6.0f} {vram_eval_nll_gb(m,131072):7.0f}  "
              f"{vram_eval_gen_gb(m,32768):6.0f} {vram_eval_gen_gb(m,131072):7.0f}")

    print(line); print("E. VERDICT")
    for name, m in MODELS.items():
        fits = [L for L in Ls if vram_train_gb(m, L, 1, 64) <= CAP]
        maxL = max(fits) if fits else 0
        gen128 = vram_eval_gen_gb(m, 131072)
        tag = ("train<=%dK, eval128K %s" % (maxL // 1024, "OK" if gen128 <= CAP else "TIGHT"))
        print(f"   {name:14s}: {tag}")


if __name__ == "__main__":
    main()
