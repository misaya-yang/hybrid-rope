"""CPU-only RULER accounting and future batching plans; never executes a model.

Costs are attention-pair/KV-token proxies, not predicted GPU wall times. Plans
depend only on input metadata, never on an arm's generated answers or scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def row_key(row):
    key = row.get("row_id") or row.get("eval_id") or row.get("id")
    if key is None:
        raise ValueError("row requires a stable identifier")
    return str(key)


def panel_metadata(rows):
    """Discard prompt token arrays after checking their recorded lengths."""
    result = {}
    for row in rows:
        key = row_key(row)
        if key in result:
            raise ValueError("duplicate panel row: " + key)
        length = len(row["prompt_ids"]) if "prompt_ids" in row else int(row["input_tokens"])
        if row.get("input_tokens") is not None and int(row["input_tokens"]) != length:
            raise ValueError("recorded prompt length differs: " + key)
        budget, cap = int(row["max_new_tokens"]), int(row["length_cap"])
        if length <= 0 or budget <= 0 or length + budget > cap:
            raise ValueError("invalid physical length/budget: " + key)
        result[key] = {"row_id": key, "task": row["task"], "length_cap": cap,
                       "prompt_tokens": length, "max_new_tokens": budget}
    if not result:
        raise ValueError("empty panel")
    return result


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return None
    def value(q):
        position = (len(ordered) - 1) * q
        lower, upper = math.floor(position), math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"min": ordered[0], "p50": value(.5), "p90": value(.9),
            "p95": value(.95), "p99": value(.99), "max": ordered[-1],
            "mean": sum(ordered) / len(ordered)}


def generation_metadata(rows, panel, scorer=None):
    """Join completed outputs by identity; optionally time the existing scorer."""
    result = {}
    scoring_seconds = 0.0
    mismatch = 0
    for row in rows:
        key = row_key(row)
        if key in result or key not in panel:
            raise ValueError("duplicate or unknown generation row: " + key)
        source = panel[key]
        for field in ("task", "length_cap"):
            if row[field] != source[field]:
                raise ValueError("generation identity differs: " + key)
        if row.get("input_tokens") is not None and int(row["input_tokens"]) != source["prompt_tokens"]:
            raise ValueError("generation prompt length differs: " + key)
        count = len(row["generated_ids"])
        ended, hit_cap = bool(row["ended_eos"]), bool(row["hit_cap"])
        if count > source["max_new_tokens"] or count <= 0:
            raise ValueError("invalid generated token count: " + key)
        if hit_cap != (count == source["max_new_tokens"] and not ended):
            raise ValueError("inconsistent completion flag: " + key)
        result[key] = {"generated_tokens": count, "ended_eos": ended, "hit_cap": hit_cap}
        if scorer is not None:
            started = time.perf_counter()
            value = scorer(row, row["output_text"])
            scoring_seconds += time.perf_counter() - started
            mismatch += int(value != row["ruler_official_score"])
    if result.keys() != panel.keys():
        raise ValueError("generation does not cover the complete panel")
    return result, {"measured": scorer is not None, "scoring_seconds": scoring_seconds,
                    "score_mismatches": mismatch,
                    "scope": "official RULER score only; excludes JSON IO and other metrics"}


def profile(panel, generated):
    def summarize(keys):
        keys = list(keys)
        budgets = [panel[k]["max_new_tokens"] for k in keys]
        lengths = [panel[k]["prompt_tokens"] for k in keys]
        counts = [generated[k]["generated_tokens"] for k in keys]
        return {"rows": len(keys), "prompt_tokens": quantiles(lengths),
                "generated_tokens": quantiles(counts),
                "budget_counts": dict(sorted(Counter(budgets).items())),
                "eos_rows": sum(generated[k]["ended_eos"] for k in keys),
                "hit_cap_rows": sum(generated[k]["hit_cap"] for k in keys),
                "total_prompt_tokens": sum(lengths), "total_generated_tokens": sum(counts),
                "total_reserved_output_tokens": sum(budgets)}
    tasks = defaultdict(list)
    for key, row in panel.items():
        tasks[row["task"]].append(key)
    return {"overall": summarize(panel),
            "tasks": {task: summarize(keys) for task, keys in sorted(tasks.items())}}


def batch_plan(panel, *, batch_size, max_kv_tokens, left_pad=False, max_padding_fraction=.01,
               group_by_task=False):
    """Cost-aware deterministic grouping; token ceilings are candidate limits.

    max_kv_tokens bounds B*(longest prompt + unchanged decode budget). This does
    not certify GPU memory: weights, activations and allocator use are separate.
    """
    if batch_size < 1 or max_kv_tokens < 1 or not 0 <= max_padding_fraction < 1:
        raise ValueError("invalid batch planning limits")
    ordered = sorted(panel.values(), key=lambda r: (
        r["length_cap"], r["max_new_tokens"], r["task"] if group_by_task else "",
        r["prompt_tokens"], r["task"], r["row_id"]))
    batches, current = [], []
    for row in ordered:
        if row["prompt_tokens"] + row["max_new_tokens"] > max_kv_tokens:
            raise ValueError("single row exceeds the candidate KV-token ceiling")
        trial = current + [row]
        longest = row["prompt_tokens"]
        compatible = not current or (
            row["length_cap"] == current[0]["length_cap"]
            and row["max_new_tokens"] == current[0]["max_new_tokens"]
            and (not group_by_task or row["task"] == current[0]["task"])
            and (left_pad or longest == current[0]["prompt_tokens"]))
        padding_fraction = 1 - sum(r["prompt_tokens"] for r in trial) / (len(trial) * longest)
        fits = (len(trial) <= batch_size
                and len(trial) * (longest + row["max_new_tokens"]) <= max_kv_tokens
                and padding_fraction <= max_padding_fraction)
        if current and not (compatible and fits):
            batches.append(current)
            current = []
        current.append(row)
    if current:
        batches.append(current)
    return [[r["row_id"] for r in batch] for batch in batches]


def plan_costs(batches, panel, generated):
    flat = [key for batch in batches for key in batch]
    if len(flat) != len(set(flat)) or set(flat) != set(panel) or any(not b for b in batches):
        raise ValueError("batches must cover every row exactly once")
    baseline_pairs = sum(r["prompt_tokens"] ** 2 for r in panel.values())
    padded_pairs = sum(len(b) * max(panel[k]["prompt_tokens"] for k in b) ** 2 for b in batches)
    serial_steps = sum(generated[k]["generated_tokens"] for k in panel)
    padded_steps = sum(len(b) * max(generated[k]["generated_tokens"] for k in b) for b in batches)
    return {"batches": len(batches), "batch_size_counts": dict(sorted(Counter(map(len, batches)).items())),
            "prefill_square_cost_ratio": padded_pairs / baseline_pairs,
            "observed_synchronous_decode_token_ratio": padded_steps / serial_steps,
            "max_reserved_kv_tokens": max(len(b) * (max(panel[k]["prompt_tokens"] for k in b)
                                                   + max(panel[k]["max_new_tokens"] for k in b)) for b in batches),
            "scope": "work proxies only; lower call count is not a measured speedup; decode uses this completed arm only"}


def kv_bytes_per_token(config, dtype_bytes=2):
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    return 2 * config["num_hidden_layers"] * config.get("num_key_value_heads", config["num_attention_heads"]) * head_dim * dtype_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-kv-tokens", type=int, default=65536)
    parser.add_argument("--max-padding-fraction", type=float, default=.01)
    parser.add_argument("--elapsed-seconds", type=float)
    parser.add_argument("--rescore-official", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.elapsed_seconds is not None and args.elapsed_seconds <= 0:
        raise ValueError("elapsed time must be positive")
    panel = panel_metadata(jsonl(args.panel))
    scorer = None
    if args.rescore_official:
        from scripts.experiments.olmo_fast_screen.ruler_bench import score
        scorer = score
    generated, scoring = generation_metadata(jsonl(args.generations), panel, scorer)
    report = {"status": "CPU_ACCOUNTING_ONLY", "panel": str(args.panel),
              "generations": str(args.generations), "profile": profile(panel, generated),
              "official_scoring": scoring, "plans": {}}
    for name, left_pad, by_task in (("equal_length", False, False),
                                     ("masked_left_pad", True, False),
                                     ("masked_left_pad_by_task", True, True)):
        batches = batch_plan(panel, batch_size=args.batch_size, max_kv_tokens=args.max_kv_tokens,
                             left_pad=left_pad, max_padding_fraction=args.max_padding_fraction,
                             group_by_task=by_task)
        report["plans"][name] = {"costs": plan_costs(batches, panel, generated), "batches": batches,
                                 "gpu_validation": "NOT_RUN"}
    if args.elapsed_seconds is not None:
        report["observed_run_seconds"] = args.elapsed_seconds
        report["observed_seconds_per_row"] = args.elapsed_seconds / len(panel)
    if args.model_config:
        per_token = kv_bytes_per_token(json.loads(args.model_config.read_text()))
        report["kv_cache"] = {"bytes_per_token_bfloat16": per_token,
                              "candidate_ceiling_bytes": per_token * args.max_kv_tokens,
                              "scope": "KV tensors only; excludes weights, temporaries, allocator and nonstandard caches"}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
