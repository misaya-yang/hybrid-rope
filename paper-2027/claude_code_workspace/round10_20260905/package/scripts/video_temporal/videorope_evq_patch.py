#!/usr/bin/env python3
"""EVQ-Cosh frequency patch for Qwen2-VL RoPE (M-RoPE / VideoRoPE).

VideoRoPE puts temporal at the LAST n_t frequency indices (low-freq slots).
EVQ-Cosh redistributes within those slots for better temporal extrapolation.

4-way experiment:
  1. M-RoPE:           temporal @ indices 0..15 (high-freq), geometric
  2. VideoRoPE:        temporal @ indices 48..63 (low-freq), geometric
  3. M-RoPE + EVQ:     temporal @ indices 0..15, EVQ-Cosh redistribution
  4. VideoRoPE + EVQ:  temporal @ indices 48..63, EVQ-Cosh redistribution

Usage:
    from videorope_evq_patch import apply_evq_temporal
    orig = apply_evq_temporal(model, tau=1.4, layout="videorope")
    # ... eval ...
    restore_inv_freq(model, orig)
"""

import math
import torch


def evq_cosh_phi(K: int, tau: float) -> torch.Tensor:
    """Compute EVQ-Cosh quantile positions phi in [0, 1].

    Args:
        K: number of frequency pairs
        tau: concentration parameter. tau=0 → uniform (geometric).
    Returns:
        phi tensor of shape [K] in float64, values in [0, 1].
    """
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)  # midpoint quantiles
    if abs(tau) < 1e-8:
        return u
    return 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))


def apply_evq_temporal(model, tau: float, layout: str = "mrope"):
    """Apply EVQ-Cosh redistribution to temporal frequencies.

    Args:
        model: Qwen2VLForConditionalGeneration
        tau: EVQ-Cosh parameter. tau=0 → no change (geometric).
        layout: "mrope" (temporal first) or "videorope" (temporal last)

    Returns:
        original inv_freq for restoration.
    """
    # Get config
    rope_cfg = getattr(model.config, "rope_scaling", None) or {}
    mrope_section = rope_cfg.get("mrope_section", [16, 24, 24])
    base = getattr(model.config, "rope_theta", 10000.0)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    n_freq = head_dim // 2  # 64 for head_dim=128

    t_dim = mrope_section[0]  # 16 freq pairs for temporal
    h_dim = mrope_section[1]  # 24 for height
    w_dim = mrope_section[2]  # 24 for width

    # Determine which indices in inv_freq correspond to temporal
    if layout == "videorope":
        # VideoRoPE: temporal uses LAST t_dim indices (low frequency)
        t_start = n_freq - t_dim  # 48
        t_end = n_freq            # 64
    else:
        # M-RoPE: temporal uses FIRST t_dim indices (high frequency)
        t_start = 0
        t_end = t_dim             # 16

    print(f"[EVQ] layout={layout}, tau={tau}, temporal indices [{t_start}:{t_end}]")
    print(f"  mrope_section={mrope_section}, base={base}, n_freq={n_freq}")

    # Access inv_freq
    rope_module = model.model.rotary_emb
    original_inv_freq = rope_module.inv_freq.clone()

    if abs(tau) < 1e-8:
        print(f"  tau≈0, keeping geometric (no change)")
        return original_inv_freq

    # Current geometric inv_freq at temporal indices
    geo_temporal = original_inv_freq[t_start:t_end]
    phi_start = -torch.log(geo_temporal[0].double()) / math.log(base)   # e.g., 0.0 or 0.75
    phi_end = -torch.log(geo_temporal[-1].double()) / math.log(base)    # e.g., 0.234 or 0.984
    print(f"  Geometric phi range: [{phi_start:.4f}, {phi_end:.4f}]")
    print(f"  Geometric freq range: [{geo_temporal[-1]:.6f}, {geo_temporal[0]:.6f}]")

    # EVQ-Cosh redistribution within [phi_start, phi_end]
    phi_evq = evq_cosh_phi(t_dim, tau)  # [0, 1] range
    phi_mapped = phi_start + (phi_end - phi_start) * phi_evq  # map to temporal range
    evq_inv_freq = torch.pow(torch.tensor(base, dtype=torch.float64), -phi_mapped).float()

    print(f"  EVQ phi range: [{phi_mapped[0]:.4f}, {phi_mapped[-1]:.4f}]")
    print(f"  EVQ freq range: [{evq_inv_freq[-1]:.6f}, {evq_inv_freq[0]:.6f}]")

    # Replace temporal frequencies
    new_inv_freq = original_inv_freq.clone()
    new_inv_freq[t_start:t_end] = evq_inv_freq.to(new_inv_freq.device)

    rope_module.inv_freq = new_inv_freq
    if hasattr(rope_module, 'original_inv_freq'):
        rope_module.original_inv_freq = new_inv_freq.clone()

    return original_inv_freq


def restore_inv_freq(model, original_inv_freq):
    """Restore original inv_freq."""
    rope_module = model.model.rotary_emb
    rope_module.inv_freq = original_inv_freq
    if hasattr(rope_module, 'original_inv_freq'):
        rope_module.original_inv_freq = original_inv_freq.clone()


def compute_tau_star(K_t: int, T_train: int, architecture: str = "ar") -> float:
    """Compute optimal tau for given temporal config.

    Args:
        K_t: number of temporal frequency pairs
        T_train: training sequence length (frames)
        architecture: "ar" (autoregressive LLM) or "dit" (bidirectional DiT)
    """
    tau_ar = K_t / math.sqrt(T_train)
    if architecture == "dit":
        return 0.53 * tau_ar
    return tau_ar
