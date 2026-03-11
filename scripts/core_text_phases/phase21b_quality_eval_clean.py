#!/usr/bin/env python3
"""
Phase 21B: QuALITY clean extrapolation eval with article-only distractor padding.

Protocol:
1. Preserve the finetuning prompt template.
2. Insert distractor text only inside the article region.
3. Score the 4 candidate option texts by length-normalized conditional NLL.

This keeps the task logic aligned with finetuning while still testing
longer contexts via padded article content.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import DTYPE, DEVICE, GPT, USE_AUTOCAST, evq_cosh_inv_freq, set_seed


TIER_CONFIGS = {
    "454m": dict(
        vocab_size=50304,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        head_dim=64,
        intermediate_size=4096,
    ),
    "750m": dict(
        vocab_size=50304,
        hidden_size=1536,
        num_layers=18,
        num_heads=24,
        head_dim=64,
        intermediate_size=6144,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 21B: QuALITY clean extrapolation eval"
    )
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--tier", default="750m", choices=["454m", "750m"])
    parser.add_argument("--rope", required=True, choices=["geo", "evq"])
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--base", type=float, default=500000.0)
    parser.add_argument("--yarn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--yarn_scale", type=float, default=2.0)
    parser.add_argument("--target_len", type=int, default=8192)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocol",
        type=str,
        default="article_pad_extrap",
        choices=["article_pad_extrap", "in_dist_nopad"],
        help="Evaluation protocol: padded extrapolation or true in-distribution no-padding baseline.",
    )
    parser.add_argument(
        "--scoring_mode",
        type=str,
        default="gold_answer_nll",
        choices=["gold_answer_nll", "options_nll"],
        help="How to score model outputs: teacher-forced gold answer NLL or 4-option NLL.",
    )
    parser.add_argument("--article_pos_min", type=float, default=0.2)
    parser.add_argument("--article_pos_max", type=float, default=0.8)
    return parser.parse_args()


def geometric_inv_freq(dim=64, base=500000.0):
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


def apply_yarn_scaling(inv_freq, scale):
    """Apply real YaRN progressive frequency scaling with smoothstep ramp.

    WARNING: Previous version was NTK-aware (all channels / same factor),
    which DESTROYS EVQ frequency structure. Real YaRN preserves it.
    """
    if scale <= 1.0:
        return inv_freq.clone()
    K = len(inv_freq)
    idx = torch.arange(K, dtype=torch.float64)
    start = int(0.20 * K)
    end = int(0.90 * K)
    if end <= start:
        end = min(K - 1, start + 1)
    ramp = torch.clamp((idx - start) / float(max(1, end - start)), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
    temperature = 1.0 + 0.07 * math.log2(scale)
    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    return (inv_freq.double() / yarn_scale).float()


def _load_state(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
    rope_keys = [k for k in state if ".rope." in k]
    for key in rope_keys:
        del state[key]
    return state


def parse_quality_sample(sample: dict) -> Tuple[str, str, str]:
    """Split a SCROLLS QuALITY sample into question/options, article, gold answer."""
    inp = sample["input"]
    answer = sample["output"].strip()
    lines = inp.split("\n")

    d_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("(D)") or stripped.startswith("(D "):
            d_idx = i
            break

    if d_idx == -1:
        parts = inp.split("\n\n", 2)
        if len(parts) >= 3:
            question_opts = parts[0] + "\n\n" + parts[1]
            article = parts[2]
        else:
            question_opts = inp[:200]
            article = inp[200:]
    else:
        question_opts = "\n".join(lines[: d_idx + 1]).strip()
        article = "\n".join(lines[d_idx + 1 :]).strip()

    return question_opts, article, answer


def extract_options(question_opts: str) -> List[Tuple[str, str]]:
    pattern = re.compile(
        r"\(\s*([ABCD])\s*\)\s*(.*?)(?=(?:\n\s*\(\s*[ABCD]\s*\))|\Z)",
        re.DOTALL,
    )
    options = []
    for letter, text in pattern.findall(question_opts):
        cleaned = " ".join(text.strip().split())
        options.append((letter, cleaned))
    return options


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_gold_option(gold_answer: str, options: Sequence[Tuple[str, str]]) -> Optional[int]:
    gold_norm = normalize_text(gold_answer)
    if not gold_norm:
        return None

    best_idx = None
    best_overlap = 0.0
    gold_tokens = set(gold_norm.split())

    for idx, (_, option_text) in enumerate(options):
        opt_norm = normalize_text(option_text)
        if not opt_norm:
            continue
        if gold_norm == opt_norm or gold_norm in opt_norm or opt_norm in gold_norm:
            return idx
        opt_tokens = set(opt_norm.split())
        if not opt_tokens:
            continue
        overlap = len(gold_tokens & opt_tokens) / max(len(gold_tokens | opt_tokens), 1)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    if best_overlap >= 0.5:
        return best_idx
    return None


def build_distractor_pool(data_dir: str, split: str = "train") -> List[str]:
    path = os.path.join(data_dir, "quality", f"{split}.jsonl")
    print(f"  Loading distractor pool: {path}")
    articles = []
    seen_articles = set()
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            _, article, _ = parse_quality_sample(sample)
            key = article[:200]
            if key not in seen_articles:
                seen_articles.add(key)
                articles.append(article)
    print(f"  Distractor pool: {len(articles)} unique articles")
    return articles


def truncate_middle(ids: List[int], budget: int) -> List[int]:
    if len(ids) <= budget:
        return ids
    if budget <= 0:
        return []
    left = budget // 2
    return ids[:left] + ids[-(budget - left) :]


def sample_distractor_ids(tokenizer, distractor_pool: Sequence[str], n_tokens: int, rng) -> List[int]:
    if n_tokens <= 0:
        return []

    pieces: List[int] = []
    indices = list(range(len(distractor_pool)))
    rng.shuffle(indices)
    sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)

    for idx in indices:
        if len(pieces) >= n_tokens:
            break
        article_ids = tokenizer.encode(distractor_pool[idx].strip(), add_special_tokens=False)
        candidate = sep_ids + article_ids
        remain = n_tokens - len(pieces)
        pieces.extend(candidate[:remain])

    return pieces[:n_tokens]


def construct_clean_prompt_ids(
    question_opts: str,
    article: str,
    target_len: int,
    tokenizer,
    distractor_pool: Sequence[str],
    rng,
    pos_min: float,
    pos_max: float,
) -> List[int]:
    prefix = f"Read the following and answer the question.\n\n{question_opts}\n\n"
    suffix = "\n\nAnswer:"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    article_ids = tokenizer.encode(article.strip(), add_special_tokens=False)

    article_budget = target_len - len(prefix_ids) - len(suffix_ids)
    if article_budget <= 32:
        raise ValueError(
            f"target_len={target_len} leaves no room for article area "
            f"(prefix={len(prefix_ids)}, suffix={len(suffix_ids)})"
        )

    article_ids = truncate_middle(article_ids, article_budget)
    distractor_budget = max(article_budget - len(article_ids), 0)

    front_ratio = rng.uniform(pos_min, pos_max)
    front_budget = int(distractor_budget * front_ratio)
    back_budget = distractor_budget - front_budget

    front_ids = sample_distractor_ids(tokenizer, distractor_pool, front_budget, rng)
    back_ids = sample_distractor_ids(tokenizer, distractor_pool, back_budget, rng)
    sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)

    article_area: List[int] = []
    if front_ids:
        article_area.extend(front_ids)
        article_area.extend(sep_ids)
    article_area.extend(article_ids)
    if back_ids:
        article_area.extend(sep_ids)
        article_area.extend(back_ids)

    if len(article_area) < article_budget:
        article_area.extend(
            sample_distractor_ids(tokenizer, distractor_pool, article_budget - len(article_area), rng)
        )
    article_area = article_area[:article_budget]

    return prefix_ids + article_area + suffix_ids


def construct_in_dist_prompt_ids(
    raw_input: str,
    target_len: int,
    tokenizer,
    max_option_tokens: int,
) -> List[int]:
    prompt = f"Read the following and answer the question.\n\n{raw_input}\n\nAnswer:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_budget = max(target_len - max_option_tokens, 64)
    return truncate_middle(prompt_ids, prompt_budget)


@torch.no_grad()
def score_completion_nll(model, prompt_ids: List[int], completion_text: str, tokenizer) -> Tuple[float, int]:
    completion_ids = tokenizer.encode(f" {completion_text.strip()}", add_special_tokens=False)
    if not completion_ids:
        return float("inf"), 0

    seq_ids = prompt_ids + completion_ids
    ids = torch.tensor(seq_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    inputs = ids[:, :-1]
    targets = ids[:, 1:]
    start = len(prompt_ids) - 1

    ctx = (
        torch.amp.autocast("cuda", dtype=DTYPE)
        if USE_AUTOCAST and DEVICE == "cuda"
        else nullcontext()
    )
    with ctx:
        logits = model(inputs)
        option_logits = logits[:, start : start + len(completion_ids), :]
        option_targets = targets[:, start : start + len(completion_ids)]
        nll = F.cross_entropy(
            option_logits.reshape(-1, option_logits.size(-1)),
            option_targets.reshape(-1),
            reduction="mean",
        )

    return float(nll.item()), len(completion_ids)


def evaluate_quality_clean(model, tokenizer, val_samples, distractor_pool, args) -> Dict:
    model.eval()
    rng = random.Random(args.seed)

    correct = 0
    total = 0
    skipped = 0
    gold_nlls: List[float] = []
    margins: List[float] = []
    answer_token_counts: List[int] = []

    print(f"\n  Evaluating clean QuALITY on {args.eval_samples} samples...")
    print(f"  Target context length: {args.target_len} tokens")
    if args.scoring_mode == "options_nll":
        print("  Scoring: 4-option length-normalized conditional NLL")
    else:
        print("  Scoring: teacher-forced gold answer conditional NLL")
    print(f"  Protocol: {args.protocol}")
    t0 = time.time()

    for i, sample in enumerate(val_samples[: args.eval_samples]):
        question_opts, article, gold_answer = parse_quality_sample(sample)
        options = extract_options(question_opts)
        gold_idx = find_gold_option(gold_answer, options)

        if not article or len(options) != 4 or gold_idx is None:
            skipped += 1
            continue

        max_option_tokens = max(
            len(tokenizer.encode(f" {text}", add_special_tokens=False)) for _, text in options
        )
        gold_answer_tokens = len(tokenizer.encode(f" {gold_answer.strip()}", add_special_tokens=False))
        if args.protocol == "in_dist_nopad":
            prompt_ids = construct_in_dist_prompt_ids(
                raw_input=sample["input"],
                target_len=args.target_len,
                tokenizer=tokenizer,
                max_option_tokens=max_option_tokens,
            )
        else:
            prompt_ids = construct_clean_prompt_ids(
                question_opts=question_opts,
                article=article,
                target_len=args.target_len,
                tokenizer=tokenizer,
                distractor_pool=distractor_pool,
                rng=rng,
                pos_min=args.article_pos_min,
                pos_max=args.article_pos_max,
            )
        model.extend_rope(len(prompt_ids) + max(max_option_tokens, gold_answer_tokens) + 8)
        gold_nll, gold_tokens = score_completion_nll(model, prompt_ids, gold_answer, tokenizer)
        gold_nlls.append(gold_nll)
        answer_token_counts.append(gold_tokens)

        if args.scoring_mode == "options_nll":
            scores = []
            for letter, option_text in options:
                nll, n_tokens = score_completion_nll(model, prompt_ids, option_text, tokenizer)
                scores.append(
                    {
                        "letter": letter,
                        "text": option_text,
                        "nll": nll,
                        "n_tokens": n_tokens,
                    }
                )

            pred_idx = min(range(len(scores)), key=lambda idx: scores[idx]["nll"])
            best_nll = scores[pred_idx]["nll"]
            sorted_scores = sorted(scores, key=lambda item: item["nll"])
            second_best = sorted_scores[1]["nll"]
            margins.append(second_best - best_nll)

            if pred_idx == gold_idx:
                correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (args.eval_samples - i - 1) / rate if rate > 0 else 0.0
            if args.scoring_mode == "options_nll":
                acc = correct / max(total, 1) * 100
                print(
                    f"    [{i+1}/{args.eval_samples}] acc={acc:.1f}% ({correct}/{total})  "
                    f"(skip={skipped}, {rate:.2f} samples/s, ETA={eta:.0f}s)"
                )
            else:
                mean_nll = sum(gold_nlls) / max(len(gold_nlls), 1)
                print(
                    f"    [{i+1}/{args.eval_samples}] mean_nll={mean_nll:.4f}  "
                    f"(skip={skipped}, {rate:.2f} samples/s, ETA={eta:.0f}s)"
                )

        if DEVICE == "cuda" and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    mean_gold_nll = sum(gold_nlls) / max(len(gold_nlls), 1)
    result = {
        "total": total,
        "skipped": skipped,
        "mean_gold_nll": round(mean_gold_nll, 4),
        "mean_answer_tokens": round(sum(answer_token_counts) / max(len(answer_token_counts), 1), 2),
        "eval_time_sec": round(elapsed, 1),
        "target_len": args.target_len,
        "protocol": args.protocol,
        "scoring": args.scoring_mode,
        "template": (
            "training_prompt_exact_raw_input_no_padding"
            if args.protocol == "in_dist_nopad"
            else "training_prompt_preserved_article_only_padding"
        ),
    }

    print(f"\n  Clean QuALITY Results ({total} scored, {skipped} skipped, {elapsed:.0f}s):")
    print(f"    Mean gold-answer NLL: {mean_gold_nll:.4f}")

    if args.scoring_mode == "options_nll":
        accuracy = correct / max(total, 1) * 100
        random_baseline = 25.0
        print(f"    Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"    Random baseline: {random_baseline}%")
        print(f"    Above random: {accuracy - random_baseline:+.2f}%")
        result.update(
            {
                "accuracy": round(accuracy, 2),
                "correct": correct,
                "random_baseline": random_baseline,
                "above_random": round(accuracy - random_baseline, 2),
                "mean_margin": round(sum(margins) / max(len(margins), 1), 4),
                "scoring": "length_normalized_option_nll",
            }
        )

    return result


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 64)
    print("  Phase 21B: QuALITY Clean Extrapolation Eval")
    print(f"  Model: {args.model_pt}")
    print(f"  RoPE: {args.rope}  |  Target len: {args.target_len}")
    print("=" * 64)

    tier_cfg = TIER_CONFIGS[args.tier]
    dim = tier_cfg["head_dim"]
    if args.rope == "evq":
        inv_freq = evq_cosh_inv_freq(head_dim=dim, tau=args.tau, base=args.base)
        print(f"  RoPE: EVQ (tau={args.tau}, base={args.base})")
    else:
        inv_freq = geometric_inv_freq(dim, args.base)
        print(f"  RoPE: Geometric (base={args.base})")

    if args.yarn:
        inv_freq = apply_yarn_scaling(inv_freq, args.yarn_scale)
        print(f"  YaRN scaling: scale={args.yarn_scale}")

    cfg = tier_cfg.copy()
    cfg["max_position_embeddings"] = args.target_len + 128
    cfg["seq_len"] = args.target_len + 128

    print(f"  Loading model from {args.model_pt}...")
    model = GPT(cfg, inv_freq).to(DEVICE)
    state = _load_state(args.model_pt)
    missing, _ = model.load_state_dict(state, strict=False)
    other_missing = [key for key in missing if ".rope." not in key]
    if other_missing:
        print(f"  WARNING missing non-rope keys: {other_missing}")

    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(inv_freq.to(block.attn.rope.inv_freq.device))
    model.extend_rope(args.target_len + 128)
    del state

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_path = os.path.join(args.data_dir, "quality", "validation.jsonl")
    print(f"  Loading validation: {val_path}")
    val_samples = []
    with open(val_path) as f:
        for line in f:
            val_samples.append(json.loads(line))
    print(f"  Val: {len(val_samples)} samples")

    distractor_pool = None
    if args.protocol == "article_pad_extrap":
        distractor_pool = build_distractor_pool(args.data_dir)
    eval_result = evaluate_quality_clean(model, tokenizer, val_samples, distractor_pool, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "phase": "21b",
        "task": "quality",
        "protocol": args.protocol,
        "rope": args.rope,
        "tau": args.tau if args.rope == "evq" else None,
        "base": args.base,
        "yarn": args.yarn,
        "target_len": args.target_len,
        "eval": eval_result,
        "model_pt": args.model_pt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result_file = output_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved: {result_file}")
    print(json.dumps(eval_result, indent=2))


if __name__ == "__main__":
    main()
