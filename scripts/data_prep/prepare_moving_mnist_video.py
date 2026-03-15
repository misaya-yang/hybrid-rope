#!/usr/bin/env python3
"""Prepare tokenized Moving MNIST caches for video temporal experiments.

This script avoids heavyweight vision dependencies by downloading raw MNIST
gzip files directly and generating moving-digit videos locally.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_if_missing(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image magic {magic} for {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label magic {magic} for {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if len(data) != n:
        raise ValueError(f"label count mismatch in {path}: {len(data)} != {n}")
    return data


def load_mnist(raw_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    for filename in MNIST_FILES.values():
        download_if_missing(f"{MNIST_BASE_URL}/{filename}", raw_dir / filename)

    train_images = read_idx_images(raw_dir / MNIST_FILES["train_images"])
    train_labels = read_idx_labels(raw_dir / MNIST_FILES["train_labels"])
    test_images = read_idx_images(raw_dir / MNIST_FILES["test_images"])
    test_labels = read_idx_labels(raw_dir / MNIST_FILES["test_labels"])
    return {
        "train": (train_images, train_labels),
        "test": (test_images, test_labels),
    }


def patchify_and_quantize(video: np.ndarray, patch_size: int, vocab_size: int) -> np.ndarray:
    frames, height, width = video.shape
    p_h = height // patch_size
    p_w = width // patch_size
    reshaped = video.reshape(frames, p_h, patch_size, p_w, patch_size)
    patch_means = reshaped.mean(axis=(2, 4))
    flat = patch_means.reshape(frames * p_h * p_w)
    tokens = np.clip(np.round(flat * (vocab_size - 1)), 0, vocab_size - 1).astype(np.uint8)
    return tokens


def render_sequence(
    digits: np.ndarray,
    image_size: int,
    frames: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    video = np.zeros((frames, image_size, image_size), dtype=np.float32)
    digit_size = digits.shape[-1]

    states = []
    for digit in digits:
        x = rng.uniform(0, image_size - digit_size)
        y = rng.uniform(0, image_size - digit_size)
        speed = rng.uniform(1.5, 4.0)
        angle = rng.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        states.append([digit.astype(np.float32) / 255.0, x, y, vx, vy])

    for t in range(frames):
        canvas = np.zeros((image_size, image_size), dtype=np.float32)
        for idx, (digit, x, y, vx, vy) in enumerate(states):
            x0 = int(round(x))
            y0 = int(round(y))
            x1 = min(x0 + digit_size, image_size)
            y1 = min(y0 + digit_size, image_size)
            dx = x1 - x0
            dy = y1 - y0
            if dx > 0 and dy > 0:
                canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], digit[:dy, :dx])

            x += vx
            y += vy
            if x <= 0 or x >= image_size - digit_size:
                vx = -vx
                x = np.clip(x, 0, image_size - digit_size)
            if y <= 0 or y >= image_size - digit_size:
                vy = -vy
                y = np.clip(y, 0, image_size - digit_size)
            states[idx] = [digit, x, y, vx, vy]
        video[t] = canvas
    return video


def build_split_tokens(
    images: np.ndarray,
    labels: np.ndarray,
    num_videos: int,
    frames: int,
    image_size: int,
    num_digits: int,
    patch_size: int,
    vocab_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    tokens = np.zeros(
        (num_videos, frames * (image_size // patch_size) ** 2),
        dtype=np.uint8,
    )
    sampled_labels = np.zeros((num_videos, num_digits), dtype=np.uint8)

    for i in tqdm(range(num_videos), desc=f"build_split(seed={seed})"):
        indices = rng.randint(0, len(images), size=num_digits)
        digits = images[indices]
        sampled_labels[i] = labels[indices]
        video = render_sequence(digits, image_size=image_size, frames=frames, rng=rng)
        tokens[i] = patchify_and_quantize(video, patch_size=patch_size, vocab_size=vocab_size)
    return tokens, sampled_labels


def save_split(out_dir: Path, name: str, tokens: np.ndarray, labels: np.ndarray) -> None:
    np.save(out_dir / f"{name}_tokens.npy", tokens)
    np.save(out_dir / f"{name}_labels.npy", labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tokenized Moving MNIST caches.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/video_temporal/generated/moving_mnist_medium"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/video_temporal/external/mnist_raw"))
    parser.add_argument("--train-videos", type=int, default=16000)
    parser.add_argument("--val-videos", type=int, default=2000)
    parser.add_argument("--test-videos", type=int, default=2000)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-digits", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(f"Manifest exists at {manifest_path}; use --overwrite to rebuild.")
        return

    mnist = load_mnist(args.raw_dir)
    train_images, train_labels = mnist["train"]
    test_images, test_labels = mnist["test"]

    train_tokens, train_digit_labels = build_split_tokens(
        train_images,
        train_labels,
        num_videos=args.train_videos,
        frames=args.frames,
        image_size=args.image_size,
        num_digits=args.num_digits,
        patch_size=args.patch_size,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    val_tokens, val_digit_labels = build_split_tokens(
        test_images,
        test_labels,
        num_videos=args.val_videos,
        frames=args.frames * 4,
        image_size=args.image_size,
        num_digits=args.num_digits,
        patch_size=args.patch_size,
        vocab_size=args.vocab_size,
        seed=args.seed + 1,
    )
    test_tokens, test_digit_labels = build_split_tokens(
        test_images,
        test_labels,
        num_videos=args.test_videos,
        frames=args.frames * 4,
        image_size=args.image_size,
        num_digits=args.num_digits,
        patch_size=args.patch_size,
        vocab_size=args.vocab_size,
        seed=args.seed + 2,
    )

    save_split(out_dir, "train", train_tokens, train_digit_labels)
    save_split(out_dir, "val", val_tokens, val_digit_labels)
    save_split(out_dir, "test", test_tokens, test_digit_labels)

    manifest = {
        "dataset": "moving_mnist",
        "train_videos": args.train_videos,
        "val_videos": args.val_videos,
        "test_videos": args.test_videos,
        "train_frames": args.frames,
        "eval_frames": args.frames * 4,
        "image_size": args.image_size,
        "num_digits": args.num_digits,
        "patch_size": args.patch_size,
        "vocab_size": args.vocab_size,
        "patches_per_frame": (args.image_size // args.patch_size) ** 2,
        "train_tokens_per_video": args.frames * (args.image_size // args.patch_size) ** 2,
        "eval_tokens_per_video": args.frames * 4 * (args.image_size // args.patch_size) ** 2,
        "seed": args.seed,
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
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
