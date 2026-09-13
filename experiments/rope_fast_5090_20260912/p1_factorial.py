#!/usr/bin/env python3
"""Prepare, run, and score the fixed 384-generation P1 factorial.

Preparation (CPU/tokenizer; no model weights)::
  python p1_factorial.py prepare --model MODEL --e2-prepared E2 --sources SOURCES --out PREPARED

GPU execution is metadata-only unless ``--execute`` is present::
  python p1_factorial.py run --model MODEL --e2-prepared E2 --sources SOURCES --out RUNS --prepared PREPARED
  python p1_factorial.py run --model MODEL --e2-prepared E2 --sources SOURCES --out RUNS --prepared PREPARED --execute

Scoring is CPU-only and consumes the complete raw generations::
  python p1_factorial.py score --prepared PREPARED --runs RUNS --out SCORE.json

The 128 frozen inputs are 16 content seeds x distance(.2L/.8L) x local
key/value gap(4/64 tokens) x effective distractor records(8/64).  Each of the
three same-gain E2 tables runs every row, yielding exactly 384 generations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path


ARMS = ("bm_g4", "mrpro_g4", "official_yarn_g4")
DISTANCES = ("near", "far")
GAPS = (4, 64)
DISTRACTORS = (8, 64)
CONTENT_SEEDS = tuple(range(16))
PROMPT_TOKENS = 16_256
GENERATION_BUDGET = 128


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def random_surface(rng: random.Random, prefix: str) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return prefix + "-" + "".join(rng.choice(alphabet) for _ in range(10))


def build_row(tokenizer, background: list[int], seed: int, distance_name: str,
              requested_gap: int, distractors: int) -> dict:
    rng = random.Random(7_312_019 + seed)
    target_key, target_value = random_surface(rng, "KEY"), random_surface(rng, "VALUE")
    all_distractors = [(random_surface(rng, "KEY"), random_surface(rng, "VALUE")) for _ in range(64)]
    marker = "__P1_NATIVE_CHAT_CONTENT_MARKER_7312019__"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], tokenize=False,
        add_generation_prompt=True,
    )
    if rendered.count(marker) != 1:
        raise ValueError("native chat template did not preserve the unique content marker")
    native_prefix_text, native_suffix_text = rendered.split(marker)
    instruction = encode(tokenizer, native_prefix_text) + encode(
        tokenizer, "Use the records below. Return only the complete value for the requested key.\n\n"
    )
    query = encode(tokenizer, f"\nQuestion: What is the value for {target_key}?") + encode(
        tokenizer, native_suffix_text
    )
    key_tokens = encode(tokenizer, f"Record {target_key}:")
    value_tokens = encode(tokenizer, f" {target_value}\n")
    distractor_tokens = []
    for key, value in all_distractors[:distractors]:
        distractor_tokens.extend(encode(tokenizer, f"Record {key}: {value}\n"))
    desired_distance = round((0.2 if distance_name == "near" else 0.8) * PROMPT_TOKENS)
    desired_key_start = (PROMPT_TOKENS - 1) - desired_distance
    prefix_needed = desired_key_start - len(instruction)
    if prefix_needed < 0:
        raise ValueError("instruction leaves no room for requested distance")
    # The local gap is measured from the end of the encoded key field to the
    # beginning of the encoded value field.  A real PG19 token slice supplies
    # exactly the requested number of intervening tokens.
    gap_tokens = background[prefix_needed:prefix_needed + requested_gap]
    target_record = key_tokens + gap_tokens + value_tokens
    fixed = len(instruction) + prefix_needed + len(target_record) + len(distractor_tokens) + len(query)
    suffix_needed = PROMPT_TOKENS - fixed
    if suffix_needed < 0:
        raise ValueError("factorial record set exceeds the fixed physical prompt")
    prefix = background[:prefix_needed]
    suffix_start = prefix_needed + requested_gap
    suffix = background[suffix_start:suffix_start + suffix_needed]
    if len(prefix) != prefix_needed or len(gap_tokens) != requested_gap or len(suffix) != suffix_needed:
        raise ValueError("continuous PG19 background slice is too short")
    prompt_ids = instruction + prefix + target_record + distractor_tokens + suffix + query
    if len(prompt_ids) != PROMPT_TOKENS or len(prompt_ids) + GENERATION_BUDGET > 16_384:
        raise AssertionError("physical 16K budget changed")
    key_start = len(instruction) + len(prefix)
    key_end = key_start + len(key_tokens)
    value_start = key_end + len(gap_tokens)
    query_position = len(prompt_ids) - 1
    actual_distance = query_position - key_start
    row_id = f"p1_s{seed:02d}_d{distance_name}_b{requested_gap}_n{distractors}"
    return {"row_id": row_id, "content_seed": seed, "distance_level": distance_name,
            "requested_distance_fraction": 0.2 if distance_name == "near" else 0.8,
            "actual_key_to_final_query_token_distance": actual_distance,
            "actual_distance_fraction": actual_distance / len(prompt_ids),
            "requested_local_gap_tokens": requested_gap,
            "actual_local_gap_tokens": value_start - key_end,
            "requested_distractor_records": distractors,
            "actual_distractor_records": distractors,
            "target_key": target_key, "gold_answer": target_value,
            "prompt_ids": prompt_ids, "input_tokens": len(prompt_ids),
            "generation_budget": GENERATION_BUDGET,
            "token_positions": {"target_key_start": key_start, "target_key_end": key_end,
                                "target_value_start": value_start, "final_query_token": query_position}}


def prepare(args) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    books = json.loads((args.sources.resolve() / "pg19_books.json").read_text())
    train_books = [row for row in books if row["split"] == "train"]
    if not train_books:
        raise ValueError("PG19 manifest has no train book")
    stride = PROMPT_TOKENS + 4096
    segments = []
    for book in train_books:
        candidate = args.sources.resolve() / "pg19" / book["key"]
        candidate_tokens = encode(tokenizer, candidate.read_text())
        for offset in range(0, len(candidate_tokens) - stride + 1, stride):
            segments.append({"tokens": candidate_tokens[offset:offset + stride],
                             "source": candidate, "source_key": book["key"],
                             "span": [offset, offset + stride]})
            if len(segments) == len(CONTENT_SEEDS):
                break
        if len(segments) == len(CONTENT_SEEDS):
            break
    if len(segments) != len(CONTENT_SEEDS):
        raise ValueError(f"PG19 train books provide only {len(segments)} disjoint usable content slices; need 16")
    args.out.mkdir(parents=True, exist_ok=False)
    count = 0
    with (args.out / "inputs.jsonl").open("x") as stream:
        for seed in CONTENT_SEEDS:
            segment_record = segments[seed]
            segment = segment_record["tokens"]
            for distance in DISTANCES:
                for gap in GAPS:
                    for distractors in DISTRACTORS:
                        row = build_row(tokenizer, segment, seed, distance, gap, distractors)
                        row["background_source"] = segment_record["source_key"]
                        row["background_token_span"] = segment_record["span"]
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        count += 1
    if count != 128:
        raise AssertionError(count)
    manifest = {"status": "PREPARED_GPU_NOT_RUN", "asset_identity_policy": "user_attested_clone/no_sha_validation",
        "rows": count, "content_seeds": len(CONTENT_SEEDS), "variants_per_seed": 8,
        "arms": list(ARMS), "expected_generations": 384, "prompt_tokens": PROMPT_TOKENS,
        "generation_budget": GENERATION_BUDGET, "total_physical_cap": 16_384,
        "background": {"kind": "continuous PG19 train text", "sources": sorted({str(record["source"]) for record in segments}),
                       "role": "controlled synthetic evaluation background; not training data"},
        "selection": "all 16 content seeds and all eight predeclared variants; no model output used",
        "scoring": "complete answer after stripping only surrounding whitespace; substring matches forbidden; EOS separate"}
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


def load_e2(e2: Path):
    manifest = json.loads((e2 / "manifest.json").read_text())
    tables = json.loads((e2 / "tables.json").read_text())
    model = manifest.get("model", {})
    return manifest, tables, model


def validate_table(table: dict) -> tuple[list[float], float]:
    values = table.get("values_float32", table.get("values"))
    if not isinstance(values, list) or len(values) != 64 or not all(math.isfinite(float(x)) and float(x) > 0 for x in values):
        raise ValueError("invalid 64-pair frequency table")
    if not all(values[index] > values[index + 1] for index in range(63)):
        raise ValueError("frequency table must strictly decrease")
    gain = float(table["gain"])
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("invalid gain")
    return values, gain


def run(args) -> None:
    metadata = {"status": "METADATA_ONLY" if not args.execute else "STARTING",
        "prepared": str(args.prepared.resolve()), "arms": list(ARMS), "rows": 128,
        "expected_generations": 384, "execution_requires": "--execute and RTX 5090 sm120",
        "asset_identity_policy": "user_attested_clone/no_sha_validation"}
    print(json.dumps(metadata, sort_keys=True))
    if not args.execute:
        return
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.experiments.olmo_fast_screen.runtime import install
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    properties = torch.cuda.get_device_properties(0)
    if torch.cuda.get_device_capability(0) != (12, 0) or "5090" not in properties.name.upper():
        raise RuntimeError("P1 requires RTX 5090 sm120")
    torch.backends.cuda.enable_flash_sdp(True); torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_cudnn_sdp(False)
    prepared_manifest = json.loads((args.prepared / "manifest.json").read_text())
    inputs = list(read_jsonl(args.prepared / "inputs.jsonl"))
    if prepared_manifest["rows"] != 128 or len(inputs) != 128 or len({row["row_id"] for row in inputs}) != 128:
        raise ValueError("prepared factorial is incomplete")
    _, tables, model_record = load_e2(args.e2_prepared.resolve())
    gains = []
    for arm in ARMS:
        _, gain = validate_table(tables[arm]); gains.append(gain)
    if len(set(gains)) != 1:
        raise ValueError("P1 phase comparison requires one exactly shared gain")
    model_path = args.model.resolve()
    config = json.loads((model_path / "config.json").read_text())
    if config["model_type"] != "olmo2" or config["num_hidden_layers"] != 16 or config["hidden_size"] != 2048:
        raise ValueError("unexpected OLMo instrument")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda"}, attn_implementation="sdpa").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for arm in ARMS:
        raw_path = args.out / f"{arm}.jsonl"
        saved = list(read_jsonl(raw_path)) if raw_path.exists() else []
        if len(saved) > 128:
            raise ValueError(f"saved arm exceeds panel: {arm}")
        for index, record in enumerate(saved):
            if record["row_id"] != inputs[index]["row_id"] or record["arm"] != arm:
                raise ValueError(f"saved row prefix mismatch: {arm}/{index}")
        install(model, tables[arm])
        values, gain = validate_table(tables[arm])
        rotary = model.model.rotary_emb
        if rotary.inv_freq.shape != (64,) or not torch.isfinite(rotary.inv_freq).all() or float(rotary.attention_scaling) != gain:
            raise RuntimeError("installed runtime table is invalid")
        with raw_path.open("a") as stream, torch.inference_mode():
            for row in inputs[len(saved):]:
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device="cuda")
                tick = time.monotonic()
                generated = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, num_beams=1,
                    max_new_tokens=row["generation_budget"], eos_token_id=list(eos_ids),
                    pad_token_id=tokenizer.pad_token_id, use_cache=True)[0, ids.shape[1]:].tolist()
                ended = bool(generated and generated[-1] in eos_ids)
                text = tokenizer.decode(generated[:-1] if ended else generated, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                record = {key: row[key] for key in ("row_id", "content_seed", "distance_level",
                    "actual_key_to_final_query_token_distance", "actual_distance_fraction",
                    "requested_local_gap_tokens", "actual_local_gap_tokens",
                    "requested_distractor_records", "actual_distractor_records", "gold_answer",
                    "input_tokens", "generation_budget", "token_positions")}
                record.update(arm=arm, observed_gain=gain, generated_ids=generated, output_text=text,
                              whole_answer_exact=float(text.strip() == row["gold_answer"].strip()),
                              ended_eos=ended, hit_cap=bool(len(generated) == row["generation_budget"] and not ended),
                              elapsed_seconds=time.monotonic() - tick)
                stream.write(json.dumps(record, sort_keys=True) + "\n"); stream.flush()
                write_json(args.out / "live.json", {"arm": arm, "completed": len(saved) + 1,
                    "total_per_arm": 128, "row_id": row["row_id"]})
                saved.append(record)
    write_json(args.out / "status.json", {"status": "COMPLETE", "rows_per_arm": 128,
        "generations": 384, "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()), "gpu": properties.name,
        "model": model_record, "asset_identity_policy": "user_attested_clone/no_sha_validation"})


def signed_factor(row: dict, factor: str) -> int:
    if factor == "distance": return 1 if row["distance_level"] == "far" else -1
    if factor == "gap": return 1 if row["actual_local_gap_tokens"] == 64 else -1
    if factor == "distractors": return 1 if row["actual_distractor_records"] == 64 else -1
    raise KeyError(factor)


def score(args) -> None:
    data = {arm: list(read_jsonl(args.runs / f"{arm}.jsonl")) for arm in ARMS}
    if any(len(values) != 128 for values in data.values()):
        raise ValueError({arm: len(values) for arm, values in data.items()})
    by_arm = {arm: {row["row_id"]: row for row in values} for arm, values in data.items()}
    ids = [row["row_id"] for row in read_jsonl(args.prepared / "inputs.jsonl")]
    if any(list(by_arm[arm]) != ids for arm in ARMS):
        raise ValueError("three arms are not the same ordered 128-row panel")
    factors = ("distance", "gap", "distractors")
    seed_effects = defaultdict(dict)
    for arm in ARMS:
        for seed in CONTENT_SEEDS:
            subset = [row for row in data[arm] if row["content_seed"] == seed]
            if len(subset) != 8: raise ValueError("incomplete seed cluster")
            effects = {}
            for factor in factors:
                effects[factor] = sum(signed_factor(row, factor) * row["whole_answer_exact"] for row in subset) / 4
            for left, right in (("distance", "gap"), ("distance", "distractors"), ("gap", "distractors")):
                effects[f"{left}_x_{right}"] = sum(signed_factor(row, left) * signed_factor(row, right) * row["whole_answer_exact"] for row in subset) / 2
            seed_effects[arm][str(seed)] = effects
    arm_summary = {arm: {"rows": 128,
        "whole_answer_exact": sum(row["whole_answer_exact"] for row in data[arm]) / 128,
        "eos_rate": sum(row["ended_eos"] for row in data[arm]) / 128,
        "cap_hits": sum(row["hit_cap"] for row in data[arm]),
        "mean_cluster_effects": {name: sum(seed_effects[arm][str(seed)][name] for seed in CONTENT_SEEDS) / 16
            for name in next(iter(seed_effects[arm].values()))}} for arm in ARMS}
    contrasts = {}
    for candidate, baseline in (("bm_g4", "mrpro_g4"), ("bm_g4", "official_yarn_g4"), ("mrpro_g4", "official_yarn_g4")):
        per_seed = {str(seed): sum(by_arm[candidate][row_id]["whole_answer_exact"] - by_arm[baseline][row_id]["whole_answer_exact"]
            for row_id in ids if by_arm[candidate][row_id]["content_seed"] == seed) / 8 for seed in CONTENT_SEEDS}
        contrasts[f"{candidate}_minus_{baseline}"] = {"mean": sum(per_seed.values()) / 16, "per_content_seed": per_seed}
    write_json(args.out, {"status": "COMPLETE", "rows": 128, "generations": 384,
        "cluster_unit": "content_seed (n=16); eight factorial variants are paired within seed",
        "arms": arm_summary, "per_seed_factor_effects": seed_effects, "method_contrasts": contrasts,
        "scoring": "whole output equals gold after stripping surrounding whitespace only; EOS reported separately"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=Path, required=True); common.add_argument("--e2-prepared", type=Path, required=True)
    common.add_argument("--sources", type=Path, required=True); common.add_argument("--out", type=Path, required=True)
    prepare_parser = sub.add_parser("prepare", parents=[common]); prepare_parser.set_defaults(function=prepare)
    run_parser = sub.add_parser("run", parents=[common]); run_parser.add_argument("--prepared", type=Path, required=True); run_parser.add_argument("--execute", action="store_true"); run_parser.set_defaults(function=run)
    score_parser = sub.add_parser("score"); score_parser.add_argument("--prepared", type=Path, required=True); score_parser.add_argument("--runs", type=Path, required=True); score_parser.add_argument("--out", type=Path, required=True); score_parser.set_defaults(function=score)
    args = parser.parse_args(); args.function(args)


if __name__ == "__main__": main()
