#!/usr/bin/env python3
"""
Build a reproducible long/short instruction corpus from local text data.

Purpose:
- Replace previous synthetic-only fallback with an explicit dataset artifact.
- Keep everything local/offline for server stability.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List


def read_blocks(path: Path, min_chars: int) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in text.split("\n\n") if len(b.strip()) >= min_chars]
    return blocks


def _compose_long_context(
    blocks: List[str],
    rng: random.Random,
    target_chars: int,
    max_chars: int,
) -> str:
    parts: List[str] = []
    cur = 0
    attempts = 0
    while cur < target_chars and attempts < 128:
        attempts += 1
        part = blocks[rng.randrange(len(blocks))]
        if len(part) < 200:
            continue
        parts.append(part)
        cur += len(part) + 2
    if not parts:
        return ""
    ctx = "\n\n".join(parts)
    if len(ctx) > max_chars:
        start = rng.randint(0, max(0, len(ctx) - max_chars))
        ctx = ctx[start : start + max_chars]
    return ctx


def make_long_samples(
    blocks: List[str],
    n_long: int,
    seed: int,
    target_long_chars_min: int,
    target_long_chars_max: int,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    out: List[Dict[str, str]] = []
    if not blocks:
        return out

    for _ in range(n_long):
        target_chars = rng.randint(target_long_chars_min, target_long_chars_max)
        ctx = _compose_long_context(
            blocks=blocks,
            rng=rng,
            target_chars=target_chars,
            max_chars=max(target_long_chars_max + 2000, target_long_chars_min + 4000),
        )
        if not ctx:
            continue
        cut = int(len(ctx) * rng.uniform(0.62, 0.82))
        cut = max(1400, min(cut, len(ctx) - 500))
        if cut <= 0:
            continue
        prefix = ctx[:cut]
        answer = ctx[cut : min(len(ctx), cut + 2200)].strip()
        prompt = (
            "Continue the passage faithfully from the given long context.\n\n"
            f"{prefix}\n\n"
            "Write the next part of the passage."
        )
        if not answer:
            continue
        out.append({"instruction": "Long-context continuation.", "input": prompt, "output": answer})
    return out


def make_short_samples(blocks: List[str], n_short: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed + 1)
    out: List[Dict[str, str]] = []
    if not blocks:
        return out

    for _ in range(n_short):
        ctx = blocks[rng.randrange(len(blocks))]
        if len(ctx) < 260:
            continue
        end = min(len(ctx), rng.randint(180, 700))
        chunk = ctx[:end].strip()
        if not chunk:
            continue

        mode = rng.randrange(3)
        if mode == 0:
            instruction = "Summarize the text in one concise sentence."
            output = chunk[: min(220, len(chunk))]
        elif mode == 1:
            instruction = "Extract five key words from the text."
            words = [w.strip(".,;:!?()[]{}\"'").lower() for w in chunk.split()]
            uniq = [w for w in words if w]
            uniq = list(dict.fromkeys(uniq))[:5]
            output = ", ".join(uniq) if uniq else "text, context, summary"
        else:
            instruction = "Write one factual question and answer from the text."
            sent = chunk.split(".")[0].strip()
            output = f"Q: What is the passage about?\nA: {sent if sent else chunk[:120]}"

        out.append({"instruction": instruction, "input": chunk, "output": output})
    return out


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _percentile(sorted_values: List[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = int(max(0, min(len(sorted_values) - 1, round((len(sorted_values) - 1) * q))))
    return int(sorted_values[idx])


def main() -> None:
    ap = argparse.ArgumentParser(description="Build long/short instruction corpus from local text.")
    ap.add_argument("--wikitext_train", type=Path, required=True)
    ap.add_argument("--out_long", type=Path, required=True)
    ap.add_argument("--out_short", type=Path, required=True)
    ap.add_argument("--n_long", type=int, default=6000)
    ap.add_argument("--n_short", type=int, default=2600)
    ap.add_argument("--min_chars", type=int, default=280)
    ap.add_argument("--target_long_chars_min", type=int, default=12000)
    ap.add_argument("--target_long_chars_max", type=int, default=22000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    blocks = read_blocks(args.wikitext_train, min_chars=int(args.min_chars))
    if not blocks:
        raise RuntimeError(f"No valid blocks from {args.wikitext_train}")

    long_rows = make_long_samples(
        blocks,
        n_long=int(args.n_long),
        seed=int(args.seed),
        target_long_chars_min=int(args.target_long_chars_min),
        target_long_chars_max=int(args.target_long_chars_max),
    )
    short_rows = make_short_samples(blocks, n_short=int(args.n_short), seed=int(args.seed))
    if len(long_rows) < 800 or len(short_rows) < 400:
        raise RuntimeError(
            f"Insufficient dataset generated: long={len(long_rows)} short={len(short_rows)}"
        )

    write_jsonl(args.out_long, long_rows)
    write_jsonl(args.out_short, short_rows)

    long_char_lens = [len(x.get("input", "")) for x in long_rows]
    long_char_lens_sorted = sorted(long_char_lens)
    print(
        json.dumps(
            {
                "wikitext_train": str(args.wikitext_train),
                "out_long": str(args.out_long),
                "out_short": str(args.out_short),
                "long_rows": len(long_rows),
                "short_rows": len(short_rows),
                "long_input_chars_p50": int(statistics.median(long_char_lens)) if long_char_lens else 0,
                "long_input_chars_p90": _percentile(long_char_lens_sorted, 0.9),
                "long_input_chars_max": int(max(long_char_lens)) if long_char_lens else 0,
                "seed": int(args.seed),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
