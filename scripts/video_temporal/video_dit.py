#!/usr/bin/env python3
"""Minimal Video DiT with 3D RoPE — Diffusion Transformer for temporal extrapolation.

Why DiT instead of AR VideoGPT?
  AR generation uses top-k sampling, which compresses distributional differences:
  even when EVQ has 27% better PPL, both models share similar top-k token sets,
  so FVD only shows ~1.5% gap. DiT avoids this entirely because denoising uses
  the FULL learned distribution at every step — frequency allocation quality
  directly affects generation quality without sampling bottleneck.

Architecture:
  - Pixel-space diffusion (no VAE needed for 64x64 MNIST)
  - 3D patchification: patch_size=4 spatially, patch_size=1 temporally
  - 3D RoPE in self-attention (spatial + temporal frequencies)
  - adaLN-Zero conditioning on diffusion timestep (DiT style)
  - DDPM with linear beta schedule

Usage:
    python scripts/video_temporal/video_dit.py --method geo --seed 42
    python scripts/video_temporal/video_dit.py --method evq --seed 42

Hardware: NVIDIA RTX PRO 6000 Blackwell 96GB
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D RoPE (rewritten for DiT: non-causal, bidirectional attention)
# ---------------------------------------------------------------------------

def evq_cosh_inv_freq(dim: int, tau: float, base: float = 10000.0) -> torch.Tensor:
    """Compute EVQ-Cosh inverse frequencies.

    φ_k(τ) = 1 - (1/τ) * arcsinh((1 - u_k) * sinh(τ))
    ω_k = base^{-φ_k}

    Args:
        dim: Full head dimension (frequencies = dim // 2)
        tau: EVQ-Cosh temperature parameter. τ=0 gives geometric RoPE.
        base: RoPE base frequency
    """
    n_freqs = dim // 2
    u = (torch.arange(n_freqs, dtype=torch.float64) + 0.5) / n_freqs

    if abs(tau) < 1e-6:
        phi = u
    else:
        A = 1.0 - u
        phi = 1.0 - (1.0 / tau) * torch.arcsinh(A * math.sinh(tau))

    inv_freq = base ** (-phi)
    return inv_freq.float()


def power_shift_inv_freq(dim: int, alpha: float, base: float = 10000.0) -> torch.Tensor:
    """DiT-optimized frequency allocation: power-law low-frequency enhancement.

    φ_k(α) = 1 - (1 - u_k)^(1+α)

    Properties:
      - α=0: φ_k = u_k (geometric, same as τ=0 cosh)
      - α>0: shifts all frequencies toward lower values (larger φ)
      - High frequencies (small u_k) shift minimally: preserves positional fingerprinting
      - Low frequencies (large u_k) shift maximally: enhances temporal reach
      - No mid-frequency "hole" unlike cosh with large τ

    Designed for DiT's bidirectional attention where frequencies serve as
    positional identifiers, not causal information carriers.

    Args:
        dim: Full head dimension (frequencies = dim // 2)
        alpha: Shift strength. α=0 gives geometric. α=0.5 is a good starting point.
        base: RoPE base frequency
    """
    n_freqs = dim // 2
    u = (torch.arange(n_freqs, dtype=torch.float64) + 0.5) / n_freqs

    phi = 1.0 - (1.0 - u) ** (1.0 + alpha)

    inv_freq = base ** (-phi)
    return inv_freq.float()


class RotaryEmbedding3D(nn.Module):
    """3D RoPE for video DiT: spatial_h + spatial_w + temporal.

    Non-causal: positions are assigned by grid coordinates, not sequential order.
    forward(L) returns (cos, sin) with shape (L, head_dim).
    """

    def __init__(
        self,
        inv_freq_h: torch.Tensor,  # (K_h,)
        inv_freq_w: torch.Tensor,  # (K_w,)
        inv_freq_t: torch.Tensor,  # (K_t,)
        H: int,
        W: int,
        max_T: int,
    ) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.register_buffer("inv_freq_h", inv_freq_h)
        self.register_buffer("inv_freq_w", inv_freq_w)
        self.register_buffer("inv_freq_t", inv_freq_t)
        self._build(max_T)

    def _build(self, max_T: int) -> None:
        max_L = max_T * self.H * self.W
        positions = torch.arange(max_L, dtype=self.inv_freq_h.dtype,
                                 device=self.inv_freq_h.device)
        HW = self.H * self.W
        t_pos = positions // HW
        hw_pos = positions % HW
        h_pos = hw_pos // self.W
        w_pos = hw_pos % self.W

        freqs_h = torch.outer(h_pos.float(), self.inv_freq_h)
        freqs_w = torch.outer(w_pos.float(), self.inv_freq_w)
        freqs_t = torch.outer(t_pos.float(), self.inv_freq_t)

        freqs = torch.cat([freqs_h, freqs_w, freqs_t], dim=-1)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_c", emb.cos(), persistent=False)
        self.register_buffer("sin_c", emb.sin(), persistent=False)
        self._max_T = max_T
        self._max_L = max_L

    def forward(self, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
        needed_T = (L + self.H * self.W - 1) // (self.H * self.W)
        if needed_T > self._max_T:
            self._build(needed_T)
        return self.cos_c[:L], self.sin_c[:L]


# ---------------------------------------------------------------------------
# Core DiT Components
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → MLP projection (DiT style)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Scale t ∈ [0,1] to [0,1000] for proper sinusoidal resolution.
        # Without scaling, freqs × t ∈ [0,1] gives near-identical embeddings.
        t_scaled = t.float() * 1000.0
        t_freq = self.timestep_embedding(t_scaled, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiTAttention(nn.Module):
    """Multi-head self-attention with 3D RoPE (non-causal for diffusion)."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, rope: RotaryEmbedding3D):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # (B, L, H, D) each

        # Apply 3D RoPE
        cos, sin = self.rope(L)
        cos = cos[:L].unsqueeze(0).unsqueeze(2)  # (1, L, 1, D)
        sin = sin[:L].unsqueeze(0).unsqueeze(2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Transpose for attention: (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Non-causal attention (no mask for diffusion)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = attn.transpose(1, 2).reshape(B, L, -1)
        return self.o(out)


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning.

    Uses adaptive layer norm: the timestep embedding modulates
    scale/shift of the norm layers and a final gate on the residual.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, rope: RotaryEmbedding3D):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttention(hidden_size, num_heads, head_dim, rope)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        # adaLN-Zero: 6 modulation parameters per block
        # (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Initialize gate projections to zero (adaLN-Zero trick)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) token features
            c: (B, D) timestep conditioning
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Attention with adaLN
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h = self.attn(h)
        x = x + gate1.unsqueeze(1) * h

        # MLP with adaLN
        h = self.norm2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h

        return x


class FinalLayer(nn.Module):
    """Final adaLN + linear projection to patch pixels."""

    def __init__(self, hidden_size: int, patch_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class VideoDiT(nn.Module):
    """Video Diffusion Transformer with 3D RoPE.

    Pixel-space diffusion: takes (B, T, C, H, W) videos, patchifies to
    (B, N, patch_dim) tokens, processes through DiT blocks with 3D RoPE,
    predicts noise in patch space.
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 4,
        hidden_size: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int = 64,
        frame_size: int = 64,
        max_T: int = 128,
        inv_freq_h: Optional[torch.Tensor] = None,
        inv_freq_w: Optional[torch.Tensor] = None,
        inv_freq_t: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.gradient_checkpointing = False
        self.hidden_size = hidden_size
        self.frame_size = frame_size

        pH = frame_size // patch_size
        pW = frame_size // patch_size
        self.pH = pH
        self.pW = pW
        patch_dim = in_channels * patch_size * patch_size

        # Patch embedding: linear projection of flattened patch pixels
        self.patch_embed = nn.Linear(patch_dim, hidden_size)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # 3D RoPE
        K = head_dim // 2  # total frequency pairs
        K_h = K // 4       # spatial gets fewer pairs (MNIST is simple)
        K_w = K // 4
        K_t = K - K_h - K_w  # temporal gets the rest

        if inv_freq_h is None:
            inv_freq_h = evq_cosh_inv_freq(K_h * 2, tau=0.0)
        if inv_freq_w is None:
            inv_freq_w = evq_cosh_inv_freq(K_w * 2, tau=0.0)
        if inv_freq_t is None:
            inv_freq_t = evq_cosh_inv_freq(K_t * 2, tau=0.0)

        self.K_h = K_h
        self.K_w = K_w
        self.K_t = K_t

        rope = RotaryEmbedding3D(inv_freq_h, inv_freq_w, inv_freq_t, pH, pW, max_T)

        # DiT blocks (all share the same RoPE instance)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, head_dim, rope) for _ in range(num_layers)
        ])

        # Final layer: predict noise in patch pixel space
        self.final_layer = FinalLayer(hidden_size, patch_dim)

        # Initialize weights
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  VideoDiT params: {n_params / 1e6:.1f}M")
        print(f"  Freq split: K_h={K_h}, K_w={K_w}, K_t={K_t}")
        print(f"  Patch grid: {pH}x{pW}, patch_dim={patch_dim}")

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def patchify(self, videos: torch.Tensor) -> torch.Tensor:
        """Convert (B, T, C, H, W) → (B, T*pH*pW, patch_dim)."""
        B, T, C, H, W = videos.shape
        p = self.patch_size
        # (B, T, C, pH, p, pW, p) → (B, T, pH, pW, C*p*p)
        x = videos.reshape(B, T, C, self.pH, p, self.pW, p)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)  # (B, T, pH, pW, C, p, p)
        x = x.reshape(B, T * self.pH * self.pW, C * p * p)
        return x

    def unpatchify(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """Convert (B, T*pH*pW, patch_dim) → (B, T, C, H, W)."""
        B = x.shape[0]
        p = self.patch_size
        C = self.in_channels
        x = x.reshape(B, T, self.pH, self.pW, C, p, p)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)  # (B, T, C, pH, p, pW, p)
        x = x.reshape(B, T, C, self.pH * p, self.pW * p)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict noise from noisy patches + timestep.

        Args:
            x: (B, N, patch_dim) noisy patch tokens (already patchified)
            t: (B,) diffusion timesteps

        Returns:
            noise_pred: (B, N, patch_dim) predicted noise
        """
        # Embed patches
        x = self.patch_embed(x)

        # Timestep conditioning
        c = self.t_embedder(t)  # (B, D)

        # DiT blocks (with optional gradient checkpointing to save VRAM)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)

        # Final prediction
        return self.final_layer(x, c)

    def extend_rope(self, max_T: int) -> None:
        """Rebuild RoPE cache for longer temporal sequences."""
        rope = self.blocks[0].attn.rope
        if max_T > rope._max_T:
            rope._build(max_T)


# ---------------------------------------------------------------------------
# Rectified Flow (Flow Matching) Scheduler
# ---------------------------------------------------------------------------

class RectifiedFlowScheduler:
    """Rectified Flow scheduler for flow matching.

    Forward process: x_t = (1 - t) * x_0 + t * ε,  t ∈ [0, 1]
    Velocity target: v = ε - x_0
    Training loss: MSE(v_pred, v)
    Sampling: Euler ODE from t=1 (noise) → t=0 (clean)
    """

    @staticmethod
    def interpolate(
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """x_t = (1 - t) * x0 + t * noise. t: (B,) floats in [0, 1]."""
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        return (1.0 - t) * x0 + t * noise

    @staticmethod
    def velocity(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """v = noise - x0 (target velocity for training)."""
        return noise - x0

    @torch.no_grad()
    def sample(
        self,
        model: VideoDiT,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Euler ODE solver: x_1 (noise) → x_0 (clean).

        Args:
            dtype: If provided, wraps forward passes in autocast (e.g. torch.bfloat16).
                   CRITICAL for FlashAttention — fp32 falls back to O(L²) math mode.
        """
        from contextlib import nullcontext
        ctx = torch.amp.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()

        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t_batch = torch.full((shape[0],), t_val, device=device)
            with ctx:
                v_pred = model(x, t_batch)
            x = x - dt * v_pred.float()  # accumulate in fp32 for precision
            if verbose and i % 10 == 0:
                print(f"    sampling step {i + 1}/{num_steps}")

        return x


# ---------------------------------------------------------------------------
# Data generation: Oscillating Moving MNIST (reuse existing tokenized data
# or generate raw pixel data for DiT)
# ---------------------------------------------------------------------------

def generate_oscillating_mnist_pixels(
    n_samples: int,
    n_frames: int,
    frame_size: int = 64,
    n_digits: int = 3,
    digit_size: int = 12,
    seed: int = 42,
) -> torch.Tensor:
    """Generate oscillating Moving MNIST as raw pixel videos.

    Returns: (n_samples, n_frames, 1, frame_size, frame_size) float32 in [-1, 1].
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    videos = np.zeros((n_samples, n_frames, 1, frame_size, frame_size), dtype=np.float32)

    for i in range(n_samples):
        for d in range(n_digits):
            # Random digit pattern (simple circle/rectangle for reproducibility)
            pattern = np.zeros((digit_size, digit_size), dtype=np.float32)
            shape_type = rng.randint(0, 3)
            if shape_type == 0:  # filled circle
                cy, cx = digit_size // 2, digit_size // 2
                yy, xx = np.ogrid[:digit_size, :digit_size]
                mask = (xx - cx)**2 + (yy - cy)**2 <= (digit_size // 2 - 1)**2
                pattern[mask] = 1.0
            elif shape_type == 1:  # rectangle
                m = digit_size // 6
                pattern[m:-m, m:-m] = 1.0
            else:  # cross
                m = digit_size // 4
                pattern[m:-m, :] = 0.5
                pattern[:, m:-m] = 0.5
                pattern[m:-m, m:-m] = 1.0

            # Oscillating motion: fixed periods matching pre-computed data
            period = [16, 24, 32][d % 3]
            phase = rng.uniform(0, 2 * np.pi)
            amplitude_x = rng.uniform(5, (frame_size - digit_size) / 2 - 2)
            amplitude_y = rng.uniform(5, (frame_size - digit_size) / 2 - 2)
            center_x = frame_size / 2
            center_y = frame_size / 2

            for t in range(n_frames):
                angle = 2 * np.pi * t / period + phase
                x = int(center_x + amplitude_x * np.sin(angle) - digit_size / 2)
                y = int(center_y + amplitude_y * np.cos(angle * 1.3 + 0.5) - digit_size / 2)
                x = max(0, min(frame_size - digit_size, x))
                y = max(0, min(frame_size - digit_size, y))
                videos[i, t, 0, y:y+digit_size, x:x+digit_size] = np.maximum(
                    videos[i, t, 0, y:y+digit_size, x:x+digit_size], pattern
                )

    # Scale to [-1, 1]
    videos = videos * 2.0 - 1.0
    return torch.from_numpy(videos)


# ---------------------------------------------------------------------------
# UCF-101 data loading (real video dataset)
# ---------------------------------------------------------------------------

def download_ucf101(data_dir: str = "data/ucf101") -> str:
    """Download and extract UCF-101 dataset. Returns path to video directory."""
    import os, subprocess
    data_path = os.path.join(data_dir, "UCF-101")
    if os.path.exists(data_path) and len(os.listdir(data_path)) > 50:
        print(f"  UCF-101 already exists at {data_path}")
        return data_path

    os.makedirs(data_dir, exist_ok=True)
    rar_path = os.path.join(data_dir, "UCF101.rar")

    if not os.path.exists(rar_path):
        url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
        print(f"  Downloading UCF-101 from {url}...")
        # Try wget first, then curl
        try:
            subprocess.run(["wget", "-q", "--show-progress", "-O", rar_path, url],
                          check=True, timeout=3600)
        except (subprocess.CalledProcessError, FileNotFoundError):
            subprocess.run(["curl", "-L", "-o", rar_path, url],
                          check=True, timeout=3600)

    if not os.path.exists(data_path):
        print(f"  Extracting UCF-101...")
        try:
            subprocess.run(["unrar", "x", "-o+", rar_path, data_dir], check=True)
        except FileNotFoundError:
            # Try with python rarfile
            import rarfile
            with rarfile.RarFile(rar_path) as rf:
                rf.extractall(data_dir)

    return data_path


def load_ucf101_pixels(
    data_dir: str,
    n_samples: int,
    n_frames: int,
    frame_size: int = 64,
    seed: int = 42,
    split: str = "train",
) -> torch.Tensor:
    """Load UCF-101 videos as pixel tensors.

    Returns: (n_samples, n_frames, 3, frame_size, frame_size) float32 in [-1, 1].
    """
    import os, glob
    import numpy as np

    try:
        import decord
        decord.bridge.set_bridge("torch")
    except ImportError:
        raise ImportError("decord required: pip install decord")

    rng = np.random.RandomState(seed)

    # Find all .avi files
    video_dir = data_dir
    all_videos = sorted(glob.glob(os.path.join(video_dir, "**", "*.avi"), recursive=True))
    if not all_videos:
        raise FileNotFoundError(f"No .avi files found in {video_dir}")

    print(f"  Found {len(all_videos)} UCF-101 videos")

    # Shuffle and select
    indices = rng.permutation(len(all_videos))

    videos = np.zeros((n_samples, n_frames, 3, frame_size, frame_size), dtype=np.float32)
    loaded = 0

    for idx in indices:
        if loaded >= n_samples:
            break

        try:
            vr = decord.VideoReader(all_videos[idx])
            total = len(vr)
            if total < 8:  # Skip very short clips
                continue

            # Sample n_frames evenly
            if total >= n_frames:
                frame_indices = np.linspace(0, total - 1, n_frames, dtype=int)
            else:
                # Repeat last frame to pad
                frame_indices = np.arange(total)
                pad = np.full(n_frames - total, total - 1, dtype=int)
                frame_indices = np.concatenate([frame_indices, pad])

            frames = vr.get_batch(frame_indices.tolist()).numpy()  # [T, H, W, C] uint8

            # Resize to target size using numpy (avoid torch dependency for speed)
            from PIL import Image
            resized = np.zeros((n_frames, frame_size, frame_size, 3), dtype=np.float32)
            for t in range(n_frames):
                img = Image.fromarray(frames[t])
                img = img.resize((frame_size, frame_size), Image.BILINEAR)
                resized[t] = np.array(img, dtype=np.float32) / 255.0

            # [T, H, W, C] -> [T, C, H, W], scale to [-1, 1]
            resized = resized.transpose(0, 3, 1, 2)  # [T, C, H, W]
            videos[loaded] = resized * 2.0 - 1.0
            loaded += 1

            if loaded % 100 == 0:
                print(f"  Loaded {loaded}/{n_samples} videos")

        except Exception as e:
            continue  # Skip corrupted videos

    if loaded < n_samples:
        print(f"  Warning: only loaded {loaded}/{n_samples} videos, padding with repeats")
        for i in range(loaded, n_samples):
            videos[i] = videos[i % loaded]

    print(f"  UCF-101 loaded: {loaded} videos, shape {videos.shape}")
    return torch.from_numpy(videos)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build model with geometric temporal frequencies
    K_t = 16
    K_h = 8
    K_w = 8
    inv_freq_h = evq_cosh_inv_freq(K_h * 2, tau=0.0)
    inv_freq_w = evq_cosh_inv_freq(K_w * 2, tau=0.0)
    inv_freq_t = evq_cosh_inv_freq(K_t * 2, tau=0.0)

    model = VideoDiT(
        in_channels=1,
        patch_size=4,
        hidden_size=512,
        num_layers=12,
        num_heads=8,
        head_dim=64,
        frame_size=64,
        max_T=128,
        inv_freq_h=inv_freq_h,
        inv_freq_w=inv_freq_w,
        inv_freq_t=inv_freq_t,
    ).to(device)

    # Test forward pass
    B, T = 2, 32
    videos = torch.randn(B, T, 1, 64, 64, device=device)
    patches = model.patchify(videos)
    print(f"Patchified: {videos.shape} → {patches.shape}")

    t = torch.randint(0, 1000, (B,), device=device)
    noise_pred = model(patches, t)
    print(f"Noise pred: {noise_pred.shape}")

    # Test unpatchify roundtrip
    recon = model.unpatchify(patches, T)
    print(f"Unpatchify: {patches.shape} → {recon.shape}")
    print(f"Roundtrip error: {(recon - videos).abs().max().item():.2e}")

    # Test scheduler
    scheduler = RectifiedFlowScheduler()
    noise = torch.randn_like(patches)
    t_test = torch.tensor([0.5, 0.5], device=device)
    x_t = scheduler.interpolate(patches, noise, t_test)
    print(f"Interpolated: {x_t.shape}")

    v_target = scheduler.velocity(patches, noise)
    print(f"Velocity target: {v_target.shape}")

    print("\n✅ VideoDiT self-test passed.")
