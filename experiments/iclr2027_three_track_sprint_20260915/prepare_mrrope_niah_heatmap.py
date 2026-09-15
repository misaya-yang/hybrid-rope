#!/usr/bin/env python3
"""Prepare a frozen Llama TailSpline/MrPro NIAH length-by-depth grid."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import random
import re


LENGTHS = (8192, 16384, 24576, 32768)
DEPTHS = (10, 20, 30, 40, 50, 60, 70, 80, 90)
REPEATS = 3
MAX_NEW_TOKENS = 32
SEED = 20260915


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_digest(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def grid_cells(repeats: int = REPEATS):
    return [
        (length, depth, repeat)
        for length in LENGTHS
        for depth in DEPTHS
        for repeat in range(repeats)
    ]


def cyclic_slice(values: list[int], *, start: int, count: int) -> list[int]:
    if not values or count < 0:
        raise ValueError("cyclic token source must be nonempty and count nonnegative")
    return [values[(start + offset) % len(values)] for offset in range(count)]


def chat_shell(tokenizer) -> tuple[list[int], list[int]]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=True,
        add_generation_prompt=True,
    )
    # Recent tokenizers return a BatchEncoding even without return_tensors;
    # older versions return the input-id list directly.
    empty = list(rendered["input_ids"] if isinstance(rendered, Mapping) else rendered)
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if not isinstance(eot, int) or eot < 0 or eot not in empty:
        raise ValueError("Llama chat template has no user-turn EOT boundary")
    split = empty.index(eot)
    prefix, suffix = empty[:split], empty[split:]
    if not prefix or not suffix:
        raise ValueError("invalid Llama chat shell")
    return prefix, suffix


def build_row(
    *, tokenizer, shell_prefix: list[int], shell_suffix: list[int],
    filler: list[int], length: int, depth: int, repeat: int, seed: int = SEED,
) -> dict:
    rng = random.Random(seed + length * 1009 + depth * 101 + repeat)
    passkey = str(rng.randrange(10_000_000, 100_000_000))
    intro = tokenizer.encode(
        "You will read a long context containing one special magic number. "
        "Remember it exactly.\n\n[CONTEXT_START]\n",
        add_special_tokens=False,
    )
    needle = tokenizer.encode(
        f"\nThe special magic number is {passkey}.\n",
        add_special_tokens=False,
    )
    outro = tokenizer.encode(
        "\n[CONTEXT_END]\nWhat is the special magic number? "
        "Return only the number.",
        add_special_tokens=False,
    )
    target_input_tokens = length - MAX_NEW_TOKENS
    fixed = len(shell_prefix) + len(shell_suffix) + len(intro) + len(needle) + len(outro)
    filler_budget = target_input_tokens - fixed
    if filler_budget <= 0:
        raise ValueError(f"length {length} cannot fit the NIAH prompt shell")
    start = rng.randrange(len(filler))
    haystack = cyclic_slice(filler, start=start, count=filler_budget)
    insert_at = round(filler_budget * depth / 100.0)
    prompt = (
        shell_prefix + intro + haystack[:insert_at] + needle
        + haystack[insert_at:] + outro + shell_suffix
    )
    if len(prompt) != target_input_tokens:
        raise RuntimeError("NIAH prompt did not preserve the exact token budget")
    needle_start = len(shell_prefix) + len(intro) + insert_at
    row_id = f"mrrope_niah_l{length}_d{depth:02d}_r{repeat}"
    return {
        "row_id": row_id,
        "example_id": row_id,
        "task": "niah_single_heatmap",
        "family": "retrieval",
        "length_cap": length,
        "prompt_ids": prompt,
        "prompt_sha256": token_digest(prompt),
        "input_tokens": len(prompt),
        "actual_length": len(prompt),
        "references": [passkey],
        "max_new_tokens": MAX_NEW_TOKENS,
        "depth_target": [depth / 100.0],
        "depth_percent": depth,
        "repeat": repeat,
        "evidence_positions": [needle_start],
        "source_document_id": row_id,
        "document_cluster_id": row_id,
        "semantic_group_id": row_id,
        "source_seed": seed,
        "filler_start_token": start,
        "selection_mode": "predeclared_length_depth_grid",
        "selection_uses_model_outputs": False,
        "scorer_revision": "single-number-rouge1-recall-v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--paul-graham-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats-per-cell", type=int, default=REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.repeats_per_cell < 1:
        raise ValueError("repeats-per-cell must be positive")
    expected_rows = len(LENGTHS) * len(DEPTHS) * args.repeats_per_cell
    output = args.out.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text())
        if (
            value.get("status") == "COMPLETE"
            and value.get("rows") == expected_rows
            and value.get("repeats_per_cell") == args.repeats_per_cell
            and value.get("seed") == args.seed
        ):
            print(json.dumps({"status": "SKIP_COMPLETE", "rows": value["rows"]}))
            return
        raise FileExistsError("NIAH output contains a different or incomplete manifest")

    from transformers import AutoTokenizer

    config = json.loads((args.model / "config.json").read_text())
    identity = (
        config.get("model_type"), config.get("hidden_size"),
        config.get("num_hidden_layers"), config.get("num_attention_heads"),
        config.get("num_key_value_heads"), config.get("rope_theta"),
        config.get("max_position_embeddings"), config.get("rope_scaling"),
    )
    expected = ("llama", 4096, 32, 32, 8, 500000.0, 8192, None)
    if identity != expected:
        raise ValueError(f"checkpoint identity {identity!r} != {expected!r}")
    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    shell_prefix, shell_suffix = chat_shell(tokenizer)
    source = json.loads(args.paul_graham_json.read_text())
    text = re.sub(r"\s+", " ", str(source["text"])).strip()
    filler = tokenizer.encode(text, add_special_tokens=False)
    if len(filler) < 4096:
        raise ValueError("Paul Graham filler corpus is unexpectedly short")
    rows = [
        build_row(
            tokenizer=tokenizer, shell_prefix=shell_prefix, shell_suffix=shell_suffix,
            filler=filler, length=length, depth=depth, repeat=repeat, seed=args.seed,
        )
        for length, depth, repeat in grid_cells(args.repeats_per_cell)
    ]
    if (
        len(rows) != expected_rows
        or len({row["row_id"] for row in rows}) != expected_rows
        or len({row["prompt_sha256"] for row in rows}) != expected_rows
    ):
        raise ValueError("NIAH grid coverage or prompt uniqueness drift")
    for length in LENGTHS:
        for depth in DEPTHS:
            cell = [row for row in rows if row["length_cap"] == length and row["depth_percent"] == depth]
            if len(cell) != args.repeats_per_cell:
                raise ValueError(f"NIAH cell coverage drift: {length}/{depth}")

    output.mkdir(parents=True)
    rows_path = output / "inputs.jsonl"
    with rows_path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "status": "COMPLETE",
        "contract": "TAILSPLINE_MRPRO_LLAMA_S4_NIAH_HEATMAP_V2",
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256(args.model / "config.json"),
        "tokenizer_sha256": sha256(args.model / "tokenizer.json"),
        "filler_source": str(args.paul_graham_json.resolve()),
        "filler_source_sha256": sha256(args.paul_graham_json),
        "lengths": list(LENGTHS),
        "depths_percent": list(DEPTHS),
        "repeats_per_cell": args.repeats_per_cell,
        "seed": args.seed,
        "rows": len(rows),
        "max_new_tokens": MAX_NEW_TOKENS,
        "input_tokens_by_length": {
            str(length): length - MAX_NEW_TOKENS for length in LENGTHS
        },
        "inputs_sha256": sha256(rows_path),
        "selection_uses_model_outputs": False,
        "metric": "ROUGE-1 recall on a single numeric reference; official substring recall also retained",
        "scope": (
            "MrRoPE-style NIAH length-by-depth diagnostic over the claimed 1x-4x window; "
            "paired TailSpline/MrPro prompts; not an independent replacement for Full-13 RULER"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "rows": len(rows), "out": str(output)}))


if __name__ == "__main__":
    main()
