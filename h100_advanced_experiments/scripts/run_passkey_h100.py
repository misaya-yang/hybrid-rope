#!/usr/bin/env python3
"""
Passkey retrieval benchmark for long-context evaluation.

Designed for quick H100 bring-up:
- length x depth matrix
- logprob scoring (recommended)
- optional MCQ and generation scoring
"""

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DIGIT_RE = re.compile(r"\d{6}")


@dataclass
class TrialResult:
    length: int
    depth: float
    trial: int
    key: str
    logprob: float
    mcq_correct: int
    gen_correct: int
    generated: str


def parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def random_key(rng: random.Random) -> str:
    return f"{rng.randint(0, 999999):06d}"


def build_filler_tokens(tokenizer, min_tokens: int) -> List[int]:
    chunk = (
        "In a quiet library, stories flow across pages with rhythm and detail. "
        "Readers move from one shelf to another, collecting facts, dates, and clues. "
    )
    ids = []
    while len(ids) < min_tokens:
        ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
    return ids


def make_prefix_ids(
    tokenizer,
    filler_tokens: List[int],
    length: int,
    depth: float,
    key: str,
    offset: int,
) -> List[int]:
    instr = "Read the context and remember the 6-digit passkey.\n"
    needle = f"The passkey is {key}. Keep this number exactly.\n"
    query = "Question: What is the passkey? Answer with exactly 6 digits:\nAnswer:"

    instr_ids = tokenizer.encode(instr, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    query_ids = tokenizer.encode(query, add_special_tokens=False)

    min_core = len(instr_ids) + len(needle_ids) + len(query_ids)
    if length <= min_core + 8:
        raise ValueError(f"length={length} too small for prompt core={min_core}")

    filler_total = length - min_core
    before = int(round(filler_total * depth))
    before = max(0, min(before, filler_total))
    after = filler_total - before

    stream = filler_tokens[offset:] + filler_tokens[:offset]
    before_ids = stream[:before]
    after_ids = stream[before : before + after]

    return instr_ids + before_ids + needle_ids + after_ids + query_ids


def seq_logprob(model, token_ids: List[int], answer_ids: List[int], device: str, dtype) -> float:
    full = torch.tensor([token_ids + answer_ids], device=device, dtype=torch.long)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        logits = model(full[:, :-1]).logits
        lp = F.log_softmax(logits, dim=-1)

    start = len(token_ids) - 1
    total = 0.0
    for i, tok in enumerate(answer_ids):
        total += lp[0, start + i, tok].item()
    return total


def mcq_options(rng: random.Random, correct: str) -> List[str]:
    opts = {correct}
    while len(opts) < 4:
        opts.add(random_key(rng))
    arr = list(opts)
    rng.shuffle(arr)
    return arr


def evaluate_one(
    model,
    tokenizer,
    filler_tokens: List[int],
    length: int,
    depth: float,
    trial: int,
    rng: random.Random,
    device: str,
    dtype,
    run_mcq: bool,
    run_generate: bool,
    max_new_tokens: int,
) -> TrialResult:
    key = random_key(rng)
    offset = rng.randint(0, max(0, len(filler_tokens) - 1))
    prefix_ids = make_prefix_ids(tokenizer, filler_tokens, length, depth, key, offset)

    answer_ids = tokenizer.encode(" " + key, add_special_tokens=False)
    lp = seq_logprob(model, prefix_ids, answer_ids, device=device, dtype=dtype)

    mcq_correct = -1
    if run_mcq:
        opts = mcq_options(rng, key)
        letters = ["A", "B", "C", "D"]
        mcq_prompt = (
            tokenizer.decode(prefix_ids, skip_special_tokens=True)
            + "\nChoose one option:\n"
            + "\n".join([f"{letters[i]}) {opts[i]}" for i in range(4)])
            + "\nAnswer:"
        )
        mcq_ids = tokenizer.encode(mcq_prompt, add_special_tokens=False)
        scores = []
        for i, _ in enumerate(opts):
            cand = tokenizer.encode(" " + letters[i], add_special_tokens=False)
            scores.append(seq_logprob(model, mcq_ids, cand, device=device, dtype=dtype))
        pred = max(range(4), key=lambda i: scores[i])
        mcq_correct = 1 if opts[pred] == key else 0

    gen_correct = -1
    generated = ""
    if run_generate:
        x = torch.tensor([prefix_ids], device=device, dtype=torch.long)
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        tail = out[0, x.shape[1] :]
        generated = tokenizer.decode(tail, skip_special_tokens=True)
        m = DIGIT_RE.search(generated)
        gen_correct = 1 if (m and m.group(0) == key) else 0

    return TrialResult(
        length=length,
        depth=depth,
        trial=trial,
        key=key,
        logprob=lp,
        mcq_correct=mcq_correct,
        gen_correct=gen_correct,
        generated=generated,
    )


def summarize(records: List[TrialResult]) -> Dict:
    grouped: Dict[Tuple[int, float], List[TrialResult]] = {}
    for r in records:
        grouped.setdefault((r.length, r.depth), []).append(r)

    out = {}
    for (length, depth), rows in sorted(grouped.items()):
        lps = [x.logprob for x in rows]
        mcq = [x.mcq_correct for x in rows if x.mcq_correct >= 0]
        gen = [x.gen_correct for x in rows if x.gen_correct >= 0]
        out.setdefault(str(length), {})[f"depth_{depth:.2f}"] = {
            "logprob_mean": round(mean(lps), 6),
            "logprob_std": round(pstdev(lps) if len(lps) > 1 else 0.0, 6),
            "mcq_acc": round(mean(mcq), 6) if mcq else None,
            "gen_acc": round(mean(gen), 6) if gen else None,
            "n": len(rows),
        }
    return out


def run_model(args, model_name: str, model_path: str) -> Dict:
    print(f"\n=== Model: {model_name} ({model_path}) ===")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    rng = random.Random(args.seed)
    max_len = max(args.lengths)
    filler_tokens = build_filler_tokens(tokenizer, min_tokens=max_len * 3)

    all_records: List[TrialResult] = []
    t0 = time.time()
    total = len(args.lengths) * len(args.depths) * args.trials
    done = 0

    for L in args.lengths:
        for d in args.depths:
            for t in range(args.trials):
                rec = evaluate_one(
                    model=model,
                    tokenizer=tokenizer,
                    filler_tokens=filler_tokens,
                    length=L,
                    depth=d,
                    trial=t,
                    rng=rng,
                    device="cuda",
                    dtype=dtype,
                    run_mcq=args.mcq,
                    run_generate=args.generate,
                    max_new_tokens=args.max_new_tokens,
                )
                all_records.append(rec)
                done += 1
                if done % 5 == 0 or done == total:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done)
                    print(f"[{model_name}] {done}/{total} done, ETA {eta/60:.1f} min")

    payload = {
        "model_name": model_name,
        "model_path": model_path,
        "seed": args.seed,
        "lengths": args.lengths,
        "depths": args.depths,
        "trials": args.trials,
        "metrics": summarize(all_records),
        "records": [
            {
                "length": r.length,
                "depth": r.depth,
                "trial": r.trial,
                "key": r.key,
                "logprob": r.logprob,
                "mcq_correct": r.mcq_correct,
                "gen_correct": r.gen_correct,
                "generated": r.generated,
            }
            for r in all_records
        ],
    }

    del model
    torch.cuda.empty_cache()
    return payload


def parse_models(values: List[str]) -> List[Tuple[str, str]]:
    parsed = []
    for item in values:
        if ":" not in item:
            raise ValueError(f"Invalid --model item: {item}, expected name:path")
        name, path = item.split(":", 1)
        parsed.append((name.strip(), path.strip()))
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name:path, repeatable")
    ap.add_argument("--output_dir", default="/opt/dfrope/results/passkey_h100")
    ap.add_argument("--lengths", default="2048,4096,8192,12288,16384")
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--mcq", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    args.lengths = parse_csv_ints(args.lengths)
    args.depths = parse_csv_floats(args.depths)
    if any(d < 0.0 or d > 1.0 for d in args.depths):
        raise ValueError("--depths must be in [0, 1]")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    models = parse_models(args.model)

    all_out = {"timestamp": time.strftime("%Y-%m-%d_%H%M%S"), "runs": {}}
    for name, path in models:
        all_out["runs"][name] = run_model(args, name, path)
        out_file = Path(args.output_dir) / "passkey_results.json"
        out_file.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
