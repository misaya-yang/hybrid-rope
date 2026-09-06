#!/usr/bin/env python3
"""Prepare tokenized Multi-Frequency Phase Orbits for temporal extrapolation stress tests.

Design rationale:
  - MNIST digits are spatially complex → model spends capacity on spatial patterns
  - Simple circles are spatially trivial → model capacity focuses on temporal modeling
  - Multiple frequency components in BOTH motion and brightness
  - This maximally stresses temporal frequency allocation quality

Temporal structure per circle:
  x(t) = cx + rx * sin(2π·t / P_motion + φ_x)
  y(t) = cy + ry * cos(2π·t / P_motion + φ_y)
  brightness(t) = base + amp * sin(2π·t / P_brightness + φ_b)

Key properties:
  - 4 circles with distinct motion periods (12, 20, 28, 36 frames)
  - 4 circles with distinct brightness periods (10, 16, 24, 32 frames)
  - Elastic collisions between circles (non-periodic transient events)
  - Spatially trivial (solid circles) → temporal modeling dominates
  - At 4x extrapolation (32→128 frames), model must maintain 8+ frequency components

Compatible with existing training/eval pipeline (same manifest format).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm


def patchify_and_quantize(video: np.ndarray, patch_size: int, vocab_size: int) -> np.ndarray:
    """Convert (frames, H, W) float32 video to flat token sequence."""
    frames, height, width = video.shape
    p_h = height // patch_size
    p_w = width // patch_size
    reshaped = video.reshape(frames, p_h, patch_size, p_w, patch_size)
    patch_means = reshaped.mean(axis=(2, 4))
    flat = patch_means.reshape(frames * p_h * p_w)
    tokens = np.clip(np.round(flat * (vocab_size - 1)), 0, vocab_size - 1).astype(np.uint8)
    return tokens


def draw_circle(canvas: np.ndarray, cx: float, cy: float, radius: float, brightness: float):
    """Draw a filled circle on canvas with anti-aliasing."""
    H, W = canvas.shape
    y_coords, x_coords = np.ogrid[:H, :W]
    dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
    r_sq = radius ** 2

    # Hard circle with 1-pixel soft edge for anti-aliasing
    mask = np.clip(1.0 - (np.sqrt(dist_sq) - radius + 0.5), 0.0, 1.0)
    canvas[:] = np.maximum(canvas, mask * brightness)


def elastic_collision(s1: dict, s2: dict):
    """Apply elastic collision between two circles (equal mass).

    Modifies velocities in-place if circles overlap.
    """
    dx = s1["cx"] - s2["cx"]
    dy = s1["cy"] - s2["cy"]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = s1["radius"] + s2["radius"]

    if dist < min_dist and dist > 0.01:
        # Normal vector
        nx = dx / dist
        ny = dy / dist

        # Relative velocity
        dvx = s1["vx"] - s2["vx"]
        dvy = s1["vy"] - s2["vy"]

        # Relative velocity along normal
        dvn = dvx * nx + dvy * ny

        # Only collide if approaching
        if dvn < 0:
            # Equal mass elastic collision: swap velocity components along normal
            s1["vx"] -= dvn * nx
            s1["vy"] -= dvn * ny
            s2["vx"] += dvn * nx
            s2["vy"] += dvn * ny

        # Separate overlapping circles
        overlap = min_dist - dist
        s1["cx"] += overlap * 0.5 * nx
        s1["cy"] += overlap * 0.5 * ny
        s2["cx"] -= overlap * 0.5 * nx
        s2["cy"] -= overlap * 0.5 * ny


def render_phase_orbits(
    image_size: int,
    frames: int,
    rng: np.random.RandomState,
    num_circles: int = 4,
    motion_periods: tuple = (12, 20, 28, 36),
    brightness_periods: tuple = (10, 16, 24, 32),
) -> np.ndarray:
    """Render circles moving in orbital paths with periodic brightness changes.

    Each circle has:
      - Orbital motion: elliptical path with a unique period
      - Brightness modulation: sinusoidal with a different unique period
      - Wall bouncing and elastic inter-circle collisions

    This creates a rich multi-frequency temporal signal with minimal spatial complexity.
    """
    video = np.zeros((frames, image_size, image_size), dtype=np.float32)
    max_radius = 6.0
    min_radius = 3.0

    states = []
    for i in range(num_circles):
        radius = rng.uniform(min_radius, max_radius)
        margin = radius + 2

        # Orbital motion parameters
        mp = motion_periods[i % len(motion_periods)]
        # Orbit radii (how far the circle moves from center)
        orbit_rx = rng.uniform(5.0, (image_size - 2 * margin) / 3)
        orbit_ry = rng.uniform(5.0, (image_size - 2 * margin) / 3)
        # Orbit center (stays within bounds)
        orbit_cx = rng.uniform(margin + orbit_rx, image_size - margin - orbit_rx)
        orbit_cy = rng.uniform(margin + orbit_ry, image_size - margin - orbit_ry)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)

        # Also add a slow drift velocity for extra complexity
        vx = rng.uniform(-0.3, 0.3)
        vy = rng.uniform(-0.3, 0.3)

        # Brightness modulation parameters
        bp = brightness_periods[i % len(brightness_periods)]
        brightness_base = rng.uniform(0.4, 0.7)
        brightness_amp = rng.uniform(0.15, 0.3)
        brightness_phase = rng.uniform(0, 2 * np.pi)

        states.append({
            "radius": radius,
            "cx": orbit_cx,  # current position (updated with drift)
            "cy": orbit_cy,
            "orbit_cx": orbit_cx,  # orbit center (drifts slowly)
            "orbit_cy": orbit_cy,
            "orbit_rx": orbit_rx,
            "orbit_ry": orbit_ry,
            "motion_period": mp,
            "phase_x": phase_x,
            "phase_y": phase_y,
            "vx": vx,
            "vy": vy,
            "brightness_period": bp,
            "brightness_base": brightness_base,
            "brightness_amp": brightness_amp,
            "brightness_phase": brightness_phase,
        })

    for t in range(frames):
        canvas = np.zeros((image_size, image_size), dtype=np.float32)

        # Update positions
        for s in states:
            # Orbital position
            angle_x = 2 * np.pi * t / s["motion_period"] + s["phase_x"]
            angle_y = 2 * np.pi * t / s["motion_period"] + s["phase_y"]
            s["cx"] = s["orbit_cx"] + s["orbit_rx"] * np.sin(angle_x)
            s["cy"] = s["orbit_cy"] + s["orbit_ry"] * np.cos(angle_y)

            # Drift the orbit center slowly
            s["orbit_cx"] += s["vx"]
            s["orbit_cy"] += s["vy"]

            # Wall bounce for orbit center
            margin = s["radius"] + s["orbit_rx"] + 2
            if s["orbit_cx"] < margin or s["orbit_cx"] > image_size - margin:
                s["vx"] = -s["vx"]
                s["orbit_cx"] = np.clip(s["orbit_cx"], margin, image_size - margin)
            margin_y = s["radius"] + s["orbit_ry"] + 2
            if s["orbit_cy"] < margin_y or s["orbit_cy"] > image_size - margin_y:
                s["vy"] = -s["vy"]
                s["orbit_cy"] = np.clip(s["orbit_cy"], margin_y, image_size - margin_y)

            # Clamp actual position
            s["cx"] = np.clip(s["cx"], s["radius"], image_size - s["radius"])
            s["cy"] = np.clip(s["cy"], s["radius"], image_size - s["radius"])

        # Elastic collisions between all pairs
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                elastic_collision(states[i], states[j])

        # Draw circles with brightness modulation
        for s in states:
            brightness = s["brightness_base"] + s["brightness_amp"] * np.sin(
                2 * np.pi * t / s["brightness_period"] + s["brightness_phase"]
            )
            brightness = np.clip(brightness, 0.05, 1.0)
            draw_circle(canvas, s["cx"], s["cy"], s["radius"], brightness)

        video[t] = canvas

    return video


def build_split_tokens(
    num_videos: int,
    frames: int,
    image_size: int,
    num_circles: int,
    patch_size: int,
    vocab_size: int,
    seed: int,
    motion_periods: tuple = (12, 20, 28, 36),
    brightness_periods: tuple = (10, 16, 24, 32),
) -> np.ndarray:
    """Build tokenized video dataset."""
    rng = np.random.RandomState(seed)
    patches_per_frame = (image_size // patch_size) ** 2
    tokens = np.zeros(
        (num_videos, frames * patches_per_frame),
        dtype=np.uint8,
    )

    for i in tqdm(range(num_videos), desc=f"build_split(seed={seed})"):
        video = render_phase_orbits(
            image_size=image_size,
            frames=frames,
            rng=rng,
            num_circles=num_circles,
            motion_periods=motion_periods,
            brightness_periods=brightness_periods,
        )
        tokens[i] = patchify_and_quantize(video, patch_size=patch_size, vocab_size=vocab_size)

    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare tokenized Multi-Frequency Phase Orbits."
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/video_temporal/generated/phase_orbits"),
    )
    parser.add_argument("--train-videos", type=int, default=16000)
    parser.add_argument("--val-videos", type=int, default=2000)
    parser.add_argument("--test-videos", type=int, default=2000)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-circles", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--motion-periods", type=str, default="12,20,28,36",
        help="Comma-separated motion periods per circle (frames)",
    )
    parser.add_argument(
        "--brightness-periods", type=str, default="10,16,24,32",
        help="Comma-separated brightness modulation periods per circle (frames)",
    )
    args = parser.parse_args()

    motion_periods = tuple(int(p) for p in args.motion_periods.split(","))
    brightness_periods = tuple(int(p) for p in args.brightness_periods.split(","))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(f"Manifest exists at {manifest_path}; use --overwrite to rebuild.")
        return

    # Train: 32 frames (model trains on these)
    print(f"Building train split ({args.train_videos} videos, {args.frames}f)...")
    train_tokens = build_split_tokens(
        num_videos=args.train_videos, frames=args.frames,
        image_size=args.image_size, num_circles=args.num_circles,
        patch_size=args.patch_size, vocab_size=args.vocab_size,
        seed=args.seed,
        motion_periods=motion_periods, brightness_periods=brightness_periods,
    )

    # Val/Test: 4x frames for extrapolation evaluation
    eval_frames = args.frames * 4
    print(f"Building val split ({args.val_videos} videos, {eval_frames}f)...")
    val_tokens = build_split_tokens(
        num_videos=args.val_videos, frames=eval_frames,
        image_size=args.image_size, num_circles=args.num_circles,
        patch_size=args.patch_size, vocab_size=args.vocab_size,
        seed=args.seed + 1,
        motion_periods=motion_periods, brightness_periods=brightness_periods,
    )

    print(f"Building test split ({args.test_videos} videos, {eval_frames}f)...")
    test_tokens = build_split_tokens(
        num_videos=args.test_videos, frames=eval_frames,
        image_size=args.image_size, num_circles=args.num_circles,
        patch_size=args.patch_size, vocab_size=args.vocab_size,
        seed=args.seed + 2,
        motion_periods=motion_periods, brightness_periods=brightness_periods,
    )

    # Save
    np.save(out_dir / "train_tokens.npy", train_tokens)
    np.save(out_dir / "val_tokens.npy", val_tokens)
    np.save(out_dir / "test_tokens.npy", test_tokens)

    manifest = {
        "dataset": "phase_orbits",
        "train_videos": args.train_videos,
        "val_videos": args.val_videos,
        "test_videos": args.test_videos,
        "train_frames": args.frames,
        "eval_frames": eval_frames,
        "image_size": args.image_size,
        "num_circles": args.num_circles,
        "patch_size": args.patch_size,
        "vocab_size": args.vocab_size,
        "patches_per_frame": (args.image_size // args.patch_size) ** 2,
        "train_tokens_per_video": args.frames * (args.image_size // args.patch_size) ** 2,
        "eval_tokens_per_video": eval_frames * (args.image_size // args.patch_size) ** 2,
        "seed": args.seed,
        "motion_periods": list(motion_periods),
        "brightness_periods": list(brightness_periods),
        "train_file": "train_tokens.npy",
        "val_file": "val_tokens.npy",
        "test_file": "test_tokens.npy",
        "files": {
            "train_tokens": "train_tokens.npy",
            "val_tokens": "val_tokens.npy",
            "test_tokens": "test_tokens.npy",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote manifest to {manifest_path}")
    print(f"  train: {train_tokens.shape}")
    print(f"  val:   {val_tokens.shape}")
    print(f"  test:  {test_tokens.shape}")
    print(f"  motion periods: {motion_periods}")
    print(f"  brightness periods: {brightness_periods}")
    print(f"  total frequencies to track: {len(motion_periods) + len(brightness_periods)} "
          f"({len(motion_periods)} motion + {len(brightness_periods)} brightness)")


if __name__ == "__main__":
    main()
