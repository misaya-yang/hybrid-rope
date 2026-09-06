#!/usr/bin/env python3
"""Prepare tokenized Phase Orbits dataset for video temporal experiments.

Design rationale:
  Moving MNIST fails to differentiate EVQ vs Geo in FVD because the spatial
  content (digit shapes) is too complex — both models make similar spatial
  errors that dominate FVD. The temporal quality difference gets buried.

  Phase Orbits solves this by:
    1. SIMPLE spatial content: circles on a dark background (easy to predict spatially)
    2. COMPLEX temporal structure: multi-frequency orbits + phase-coupled brightness
    3. PHASE ERRORS → SPATIAL ERRORS: if the model gets temporal phase wrong,
       the circle's position AND brightness are visually wrong, which FVD detects.

  Each video contains 3-4 circles:
    - Different orbital frequencies (periods P₁, P₂, P₃)
    - Phase-coupled brightness: brightness = base + amp * sin(2π * t / P_brightness)
    - Elliptical orbits with different eccentricities
    - Optional: gravitational interaction (soft repulsion when close)

  The multi-frequency temporal structure directly tests frequency allocation:
    - Geo allocates temporal channels geometrically (suboptimal for multi-frequency)
    - EVQ allocates via cosh inversion (better coverage of frequency spectrum)

Usage:
    python scripts/data_prep/prepare_phase_orbits.py
    python scripts/data_prep/prepare_phase_orbits.py --train-videos 16000 --difficulty hard
    python scripts/data_prep/prepare_phase_orbits.py --overwrite --difficulty medium

Output: data/video_temporal/generated/phase_orbits_{difficulty}/
    - train_tokens.npy, val_tokens.npy, test_tokens.npy
    - manifest.json (compatible with run_video_temporal_allocation_sweep.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Circle rendering
# ---------------------------------------------------------------------------

def draw_circle(
    canvas: np.ndarray,
    cx: float, cy: float,
    radius: float,
    brightness: float,
    aa: bool = True,
) -> None:
    """Draw a filled circle with anti-aliasing on the canvas (in-place).

    Uses max-blending so overlapping circles don't darken each other.
    """
    H, W = canvas.shape
    r2 = radius * radius

    # Bounding box
    y_min = max(0, int(cy - radius - 1))
    y_max = min(H, int(cy + radius + 2))
    x_min = max(0, int(cx - radius - 1))
    x_max = min(W, int(cx + radius + 2))

    yy = np.arange(y_min, y_max, dtype=np.float32) + 0.5
    xx = np.arange(x_min, x_max, dtype=np.float32) + 0.5

    if len(yy) == 0 or len(xx) == 0:
        return

    dy = yy - cy
    dx = xx - cx
    dist_sq = dy[:, None] ** 2 + dx[None, :] ** 2

    if aa:
        # Anti-aliased edge (smooth over 1 pixel)
        alpha = np.clip(1.0 - (np.sqrt(dist_sq) - radius + 0.5), 0, 1)
    else:
        alpha = (dist_sq <= r2).astype(np.float32)

    patch = alpha * brightness
    canvas[y_min:y_max, x_min:x_max] = np.maximum(
        canvas[y_min:y_max, x_min:x_max], patch
    )


# ---------------------------------------------------------------------------
# Orbit physics
# ---------------------------------------------------------------------------

class OrbitalBody:
    """A circle on an elliptical orbit with phase-coupled brightness."""

    def __init__(
        self,
        center_x: float,
        center_y: float,
        semi_major: float,
        semi_minor: float,
        period: float,           # orbital period (frames)
        phase_offset: float,     # initial orbital phase [0, 2π)
        brightness_base: float,  # base brightness [0, 1]
        brightness_amp: float,   # brightness oscillation amplitude
        brightness_period: float,  # brightness oscillation period
        brightness_phase: float,   # brightness phase offset
        radius: float,           # circle radius (pixels)
        orbit_angle: float = 0,  # rotation of the ellipse
    ):
        self.cx = center_x
        self.cy = center_y
        self.a = semi_major
        self.b = semi_minor
        self.period = period
        self.phase0 = phase_offset
        self.bright_base = brightness_base
        self.bright_amp = brightness_amp
        self.bright_period = brightness_period
        self.bright_phase = brightness_phase
        self.radius = radius
        self.orbit_angle = orbit_angle

    def position_at(self, t: float) -> Tuple[float, float]:
        """Get (x, y) position at time t."""
        theta = 2 * np.pi * t / self.period + self.phase0
        # Position on unrotated ellipse
        ex = self.a * np.cos(theta)
        ey = self.b * np.sin(theta)
        # Rotate
        ca = np.cos(self.orbit_angle)
        sa = np.sin(self.orbit_angle)
        x = self.cx + ex * ca - ey * sa
        y = self.cy + ex * sa + ey * ca
        return float(x), float(y)

    def brightness_at(self, t: float) -> float:
        """Get brightness at time t."""
        b = self.bright_base + self.bright_amp * np.sin(
            2 * np.pi * t / self.bright_period + self.bright_phase
        )
        return float(np.clip(b, 0.05, 1.0))


# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------

PRESETS = {
    "easy": {
        "n_bodies": 3,
        "periods": [(14, 18), (22, 28), (30, 36)],      # (min, max) for orbital period
        "bright_periods": [(14, 18), (22, 28), (30, 36)], # matched to orbital
        "semi_major": (8, 14),
        "eccentricity": (0.0, 0.3),   # low eccentricity (near circular)
        "radius": (3, 5),
        "bright_amp": (0.15, 0.25),
        "interaction": False,
    },
    "medium": {
        "n_bodies": 3,
        "periods": [(10, 16), (18, 26), (28, 40)],
        "bright_periods": [(12, 20), (20, 30), (30, 44)],  # slightly mismatched
        "semi_major": (8, 16),
        "eccentricity": (0.1, 0.5),
        "radius": (2.5, 5),
        "bright_amp": (0.15, 0.30),
        "interaction": True,
        "repulsion_strength": 0.3,
    },
    "hard": {
        "n_bodies": 4,
        "periods": [(8, 14), (14, 22), (22, 32), (32, 48)],
        "bright_periods": [(10, 18), (16, 26), (24, 36), (36, 52)],
        "semi_major": (6, 18),
        "eccentricity": (0.2, 0.6),
        "radius": (2, 5),
        "bright_amp": (0.2, 0.35),
        "interaction": True,
        "repulsion_strength": 0.5,
    },
}


def generate_orbits(
    rng: np.random.RandomState,
    image_size: int,
    preset: dict,
) -> List[OrbitalBody]:
    """Generate orbital bodies for one video according to preset."""
    n = preset["n_bodies"]
    center = image_size / 2.0
    bodies = []

    for i in range(n):
        p_lo, p_hi = preset["periods"][i]
        period = rng.uniform(p_lo, p_hi)

        bp_lo, bp_hi = preset["bright_periods"][i]
        bright_period = rng.uniform(bp_lo, bp_hi)

        sm_lo, sm_hi = preset["semi_major"]
        semi_major = rng.uniform(sm_lo, sm_hi)

        ecc_lo, ecc_hi = preset["eccentricity"]
        ecc = rng.uniform(ecc_lo, ecc_hi)
        semi_minor = semi_major * np.sqrt(1 - ecc ** 2)

        r_lo, r_hi = preset["radius"]
        radius = rng.uniform(r_lo, r_hi)

        amp_lo, amp_hi = preset["bright_amp"]
        bright_amp = rng.uniform(amp_lo, amp_hi)

        # Spread centers around the image center with some offset
        offset_r = rng.uniform(0, image_size * 0.15)
        offset_angle = rng.uniform(0, 2 * np.pi)
        cx = center + offset_r * np.cos(offset_angle)
        cy = center + offset_r * np.sin(offset_angle)

        # Ensure orbits stay within bounds
        max_extent = semi_major + radius + 2
        cx = np.clip(cx, max_extent, image_size - max_extent)
        cy = np.clip(cy, max_extent, image_size - max_extent)

        body = OrbitalBody(
            center_x=cx,
            center_y=cy,
            semi_major=semi_major,
            semi_minor=semi_minor,
            period=period,
            phase_offset=rng.uniform(0, 2 * np.pi),
            brightness_base=rng.uniform(0.4, 0.7),
            brightness_amp=bright_amp,
            brightness_period=bright_period,
            brightness_phase=rng.uniform(0, 2 * np.pi),
            radius=radius,
            orbit_angle=rng.uniform(0, np.pi),
        )
        bodies.append(body)

    return bodies


def apply_soft_repulsion(
    positions: List[Tuple[float, float]],
    radii: List[float],
    bodies: List[OrbitalBody],
    strength: float,
) -> List[Tuple[float, float]]:
    """Apply soft repulsion between bodies to avoid overlap.

    This creates non-trivial temporal interactions: when two circles
    approach each other, they deflect slightly. The model must learn
    these interactions to predict correctly.
    """
    n = len(positions)
    new_pos = list(positions)

    for i in range(n):
        fx, fy = 0.0, 0.0
        xi, yi = positions[i]
        ri = radii[i]

        for j in range(n):
            if i == j:
                continue
            xj, yj = positions[j]
            rj = radii[j]
            dx = xi - xj
            dy = yi - yj
            dist = max(np.sqrt(dx * dx + dy * dy), 0.1)
            overlap = (ri + rj + 2) - dist  # positive when close/overlapping

            if overlap > 0:
                force = strength * overlap / dist
                fx += force * dx
                fy += force * dy

        new_pos[i] = (xi + fx, yi + fy)

    return new_pos


def render_video(
    bodies: List[OrbitalBody],
    n_frames: int,
    image_size: int,
    interaction: bool = False,
    repulsion_strength: float = 0.3,
    background_freq: float = 0.0,
) -> np.ndarray:
    """Render a Phase Orbits video.

    Returns: (n_frames, image_size, image_size) float32 in [0, 1]
    """
    video = np.zeros((n_frames, image_size, image_size), dtype=np.float32)

    for t in range(n_frames):
        canvas = np.zeros((image_size, image_size), dtype=np.float32)

        # Optional: slow background gradient shift
        if background_freq > 0:
            bg_phase = 2 * np.pi * t * background_freq
            bg_level = 0.02 + 0.01 * np.sin(bg_phase)
            canvas[:] = bg_level

        # Get positions and brightness
        positions = [b.position_at(t) for b in bodies]
        brightness = [b.brightness_at(t) for b in bodies]
        radii = [b.radius for b in bodies]

        # Apply interaction forces
        if interaction and len(bodies) > 1:
            positions = apply_soft_repulsion(
                positions, radii, bodies, repulsion_strength
            )

        # Render each body
        for (x, y), b, r in zip(positions, brightness, radii):
            # Clamp to canvas
            x = np.clip(x, r + 1, image_size - r - 1)
            y = np.clip(y, r + 1, image_size - r - 1)
            draw_circle(canvas, x, y, r, b)

        video[t] = np.clip(canvas, 0, 1)

    return video


# ---------------------------------------------------------------------------
# Tokenization (same as Moving MNIST pipeline)
# ---------------------------------------------------------------------------

def patchify_and_quantize(
    video: np.ndarray, patch_size: int, vocab_size: int
) -> np.ndarray:
    """Convert (T, H, W) float32 video to flat token array."""
    T, H, W = video.shape
    pH = H // patch_size
    pW = W // patch_size
    reshaped = video.reshape(T, pH, patch_size, pW, patch_size)
    patch_means = reshaped.mean(axis=(2, 4))
    flat = patch_means.reshape(T * pH * pW)
    tokens = np.clip(np.round(flat * (vocab_size - 1)), 0, vocab_size - 1).astype(np.uint8)
    return tokens


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_split(
    num_videos: int,
    n_frames: int,
    image_size: int,
    patch_size: int,
    vocab_size: int,
    preset: dict,
    seed: int,
) -> np.ndarray:
    """Build tokenized split."""
    rng = np.random.RandomState(seed)
    patches_per_frame = (image_size // patch_size) ** 2
    tokens = np.zeros(
        (num_videos, n_frames * patches_per_frame),
        dtype=np.uint8,
    )

    interaction = preset.get("interaction", False)
    repulsion = preset.get("repulsion_strength", 0.3)

    for i in tqdm(range(num_videos), desc=f"build_split(seed={seed})"):
        bodies = generate_orbits(rng, image_size, preset)
        bg_freq = rng.uniform(0, 0.02) if interaction else 0.0
        video = render_video(
            bodies, n_frames, image_size,
            interaction=interaction,
            repulsion_strength=repulsion,
            background_freq=bg_freq,
        )
        tokens[i] = patchify_and_quantize(video, patch_size, vocab_size)

    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Phase Orbits dataset.")
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=sorted(PRESETS.keys()))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--train-videos", type=int, default=16000)
    parser.add_argument("--val-videos", type=int, default=2000)
    parser.add_argument("--test-videos", type=int, default=2000)
    parser.add_argument("--train-frames", type=int, default=32)
    parser.add_argument("--eval-frames-multiplier", type=int, default=4,
                        help="eval_frames = train_frames * multiplier")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    # 8x extrapolation support
    parser.add_argument("--extra-eval-frames", type=int, default=0,
                        help="If >0, generate val/test with this many frames "
                             "(e.g., 256 for 8x eval on 32f train)")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"data/video_temporal/generated/phase_orbits_{args.difficulty}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists() and not args.overwrite:
        print(f"Manifest exists at {manifest_path}; use --overwrite to rebuild.")
        return

    preset = PRESETS[args.difficulty]
    eval_frames = args.train_frames * args.eval_frames_multiplier
    if args.extra_eval_frames > 0:
        eval_frames = max(eval_frames, args.extra_eval_frames)

    print(f"\n{'#' * 60}")
    print(f"  PHASE ORBITS: difficulty={args.difficulty}")
    print(f"  n_bodies={preset['n_bodies']}  interaction={preset.get('interaction', False)}")
    print(f"  periods={preset['periods']}")
    print(f"  train_frames={args.train_frames}  eval_frames={eval_frames}")
    print(f"  image_size={args.image_size}  patch_size={args.patch_size}")
    print(f"  vocab_size={args.vocab_size}")
    print(f"{'#' * 60}\n")

    # Build splits
    print("Building train split...")
    train_tokens = build_split(
        args.train_videos, args.train_frames,
        args.image_size, args.patch_size, args.vocab_size,
        preset, seed=args.seed,
    )

    print("Building val split...")
    val_tokens = build_split(
        args.val_videos, eval_frames,
        args.image_size, args.patch_size, args.vocab_size,
        preset, seed=args.seed + 1,
    )

    print("Building test split...")
    test_tokens = build_split(
        args.test_videos, eval_frames,
        args.image_size, args.patch_size, args.vocab_size,
        preset, seed=args.seed + 2,
    )

    # Save
    np.save(out_dir / "train_tokens.npy", train_tokens)
    np.save(out_dir / "val_tokens.npy", val_tokens)
    np.save(out_dir / "test_tokens.npy", test_tokens)

    # Dummy labels (compatibility with sweep script)
    np.save(out_dir / "train_labels.npy", np.zeros((args.train_videos, 1), dtype=np.uint8))
    np.save(out_dir / "val_labels.npy", np.zeros((args.val_videos, 1), dtype=np.uint8))
    np.save(out_dir / "test_labels.npy", np.zeros((args.test_videos, 1), dtype=np.uint8))

    manifest = {
        "dataset": f"phase_orbits_{args.difficulty}",
        "train_videos": args.train_videos,
        "val_videos": args.val_videos,
        "test_videos": args.test_videos,
        "train_frames": args.train_frames,
        "eval_frames": eval_frames,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "vocab_size": args.vocab_size,
        "patches_per_frame": (args.image_size // args.patch_size) ** 2,
        "train_tokens_per_video": args.train_frames * (args.image_size // args.patch_size) ** 2,
        "eval_tokens_per_video": eval_frames * (args.image_size // args.patch_size) ** 2,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "n_bodies": preset["n_bodies"],
        "interaction": preset.get("interaction", False),
        "periods": preset["periods"],
        "files": {
            "train_tokens": "train_tokens.npy",
            "train_labels": "train_labels.npy",
            "val_tokens": "val_tokens.npy",
            "val_labels": "val_labels.npy",
            "test_tokens": "test_tokens.npy",
            "test_labels": "test_labels.npy",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nSaved to {out_dir}/")
    print(f"  train: {train_tokens.shape}")
    print(f"  val:   {val_tokens.shape}")
    print(f"  test:  {test_tokens.shape}")
    print(f"  manifest: {manifest_path}")

    # Print sample info
    print(f"\n  Sample orbit periods: {preset['periods']}")
    print(f"  Brightness periods: {preset['bright_periods']}")
    print(f"  Eccentricity range: {preset['eccentricity']}")
    print(f"  With interactions: {preset.get('interaction', False)}")


if __name__ == "__main__":
    main()
