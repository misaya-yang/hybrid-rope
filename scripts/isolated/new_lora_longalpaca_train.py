#!/usr/bin/env python3
"""
Isolated new dataset LoRA trainer for NeurIPS remediation.

Hard constraints:
- Base model: Meta-Llama-3-8B-Instruct (8K native).
- RoPE injection path: inv_freq.copy() via train_cross_model_lora_fast_tuned.py.
- Low-compute budget: 800 steps, LoRA rank 32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer


MODEL_LOCK_NAME = "Meta-Llama-3-8B-Instruct"
MODEL_LOCK_MAX_POS = 8192
MODEL_LOCK_DEFAULT = Path("/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct")
ROPE_THETA_LOCK = 500000.0


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_seed_csv(text: str) -> List[int]:
    return [int(x) for x in parse_csv(text)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def assert_model_lock(base_model_path: Path) -> Dict[str, object]:
    model_str = base_model_path.as_posix()
    low = model_str.lower()
    if MODEL_LOCK_NAME.lower() not in low:
        raise RuntimeError(
            "Model lock violation: expected Meta-Llama-3-8B-Instruct path, got "
            f"{model_str}"
        )
    if "3.1" in low or "128k" in low:
        raise RuntimeError(
            "Model lock violation: Llama-3.1 / 128K-native model names are forbidden."
        )

    cfg_path = base_model_path / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"Missing config.json under {base_model_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    max_pos = int(cfg.get("max_position_embeddings", -1))
    model_type = str(cfg.get("model_type", ""))
    if max_pos != MODEL_LOCK_MAX_POS:
        raise RuntimeError(
            f"Model lock violation: expected max_position_embeddings={MODEL_LOCK_MAX_POS}, got {max_pos}."
        )
    if model_type and model_type.lower() != "llama":
        raise RuntimeError(f"Unexpected model_type for lock model: {model_type}")

    return {
        "base_model_path": model_str,
        "config_path": cfg_path.as_posix(),
        "config_sha256": sha256_file(cfg_path),
        "model_type": model_type,
        "max_position_embeddings": max_pos,
    }


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def iter_json_or_jsonl(path: Path) -> Iterable[Dict]:
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
        return
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item
        return
    raise RuntimeError(f"Unsupported dataset format for {path}; expected .jsonl or JSON list.")


def normalize_messages(obj: Dict) -> List[Dict[str, str]]:
    messages = obj.get("messages")
    if isinstance(messages, list) and messages:
        parsed: List[Dict[str, str]] = []
        for item in messages:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role and content:
                parsed.append({"role": role, "content": content})
        if parsed:
            return parsed

    conversations = obj.get("conversations")
    if isinstance(conversations, list) and conversations:
        parsed = []
        for item in conversations:
            role_raw = str(item.get("from", item.get("role", ""))).strip().lower()
            if role_raw in {"human", "user"}:
                role = "user"
            elif role_raw in {"assistant", "gpt"}:
                role = "assistant"
            elif role_raw == "system":
                role = "system"
            else:
                role = ""
            content = str(item.get("value", item.get("content", ""))).strip()
            if role and content:
                parsed.append({"role": role, "content": content})
        if parsed:
            return parsed

    instruction = str(obj.get("instruction", "")).strip()
    extra_input = str(obj.get("input", "")).strip()
    output = str(obj.get("output", obj.get("response", obj.get("answer", "")))).strip()
    if instruction and output:
        merged = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
        return [{"role": "user", "content": merged}, {"role": "assistant", "content": output}]

    question = str(obj.get("question", "")).strip()
    context = str(obj.get("context", obj.get("document", ""))).strip()
    answer = str(obj.get("answer", "")).strip()
    if question and answer:
        prompt = question if not context else f"{context}\n\nQuestion: {question}"
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    return []


def render_text(messages: List[Dict[str, str]], tokenizer: AutoTokenizer) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return str(text).strip()


def build_wikitext_instruction_samples(wiki_train_txt: Path, max_samples: int, seed: int) -> List[Dict]:
    text = wiki_train_txt.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in text.split("\n\n") if len(b.strip()) >= 600]
    if not blocks:
        return []
    rng = random.Random(seed)
    rng.shuffle(blocks)
    blocks = blocks[:max_samples]

    out: List[Dict] = []
    for block in blocks:
        cut = max(220, int(len(block) * 0.6))
        cut = min(cut, len(block) - 120)
        if cut <= 0:
            continue
        left = block[:cut].strip()
        right = block[cut:].strip()
        if len(left) < 120 or len(right) < 80:
            continue
        out.append(
            {
                "instruction": "Continue the passage faithfully while preserving topic and style.",
                "input": left,
                "output": right,
            }
        )
    return out


def load_long_instruction_pool(
    longalpaca_path: Optional[Path],
    longqa_path: Optional[Path],
    max_records: int,
) -> List[Dict]:
    pool: List[Dict] = []
    for p in [longalpaca_path, longqa_path]:
        if p is None:
            continue
        if not p.exists():
            raise RuntimeError(f"Dataset path not found: {p}")
        for obj in iter_json_or_jsonl(p):
            pool.append(obj)
            if max_records > 0 and len(pool) >= max_records:
                return pool
    return pool


def prepare_mixed_text_data(
    tokenizer: AutoTokenizer,
    long_pool: List[Dict],
    wiki_pool: List[Dict],
    out_dir: Path,
    mix_long_ratio: float,
    mix_wiki_ratio: float,
    seed: int,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    total_candidates = len(long_pool) + len(wiki_pool)
    if total_candidates <= 0:
        raise RuntimeError("No candidates available to build mixed dataset.")

    mix_sum = float(mix_long_ratio + mix_wiki_ratio)
    if mix_sum <= 0:
        raise RuntimeError("Invalid mix ratio: long and wiki ratios sum to <= 0.")
    long_r = mix_long_ratio / mix_sum
    wiki_r = mix_wiki_ratio / mix_sum

    target_n = total_candidates
    n_long = int(round(target_n * long_r))
    n_wiki = target_n - n_long
    n_long = min(n_long, len(long_pool))
    n_wiki = min(n_wiki, len(wiki_pool))
    if n_long + n_wiki < min(target_n, 1000):
        # backfill if one source is short
        remaining = target_n - (n_long + n_wiki)
        if remaining > 0 and len(long_pool) > n_long:
            add = min(remaining, len(long_pool) - n_long)
            n_long += add
            remaining -= add
        if remaining > 0 and len(wiki_pool) > n_wiki:
            add = min(remaining, len(wiki_pool) - n_wiki)
            n_wiki += add

    long_sel = rng.sample(long_pool, n_long) if n_long > 0 else []
    wiki_sel = rng.sample(wiki_pool, n_wiki) if n_wiki > 0 else []
    merged = long_sel + wiki_sel
    rng.shuffle(merged)

    rendered: List[str] = []
    bad = 0
    for obj in merged:
        msgs = normalize_messages(obj)
        if not msgs:
            bad += 1
            continue
        txt = render_text(msgs, tokenizer)
        if not txt:
            bad += 1
            continue
        rendered.append(txt)

    if len(rendered) < 200:
        raise RuntimeError(
            f"Too few valid rendered samples ({len(rendered)}). "
            "Check dataset format or increase source data."
        )

    n = len(rendered)
    n_valid = max(8, int(n * 0.01))
    n_test = max(8, int(n * 0.01))
    n_train = max(1, n - n_valid - n_test)
    train = rendered[:n_train]
    valid = rendered[n_train : n_train + n_valid]
    test = rendered[n_train + n_valid :]

    train_txt = out_dir / "train.txt"
    valid_txt = out_dir / "valid.txt"
    test_txt = out_dir / "test.txt"
    train_txt.write_text("\n\n".join(train) + "\n", encoding="utf-8")
    valid_txt.write_text("\n\n".join(valid) + "\n", encoding="utf-8")
    test_txt.write_text("\n\n".join(test) + "\n", encoding="utf-8")

    return {
        "target_total_candidates": total_candidates,
        "selected_long_count": n_long,
        "selected_wiki_count": n_wiki,
        "rendered_total": n,
        "rendered_dropped": bad,
        "train_count": len(train),
        "valid_count": len(valid),
        "test_count": len(test),
        "train_txt": train_txt.as_posix(),
        "valid_txt": valid_txt.as_posix(),
        "test_txt": test_txt.as_posix(),
        "train_sha256": sha256_file(train_txt),
        "valid_sha256": sha256_file(valid_txt),
        "test_sha256": sha256_file(test_txt),
    }


def build_train_cmd(args: argparse.Namespace, method: str, seed: int, output_dir: Path) -> List[str]:
    return [
        sys.executable,
        "scripts/train_cross_model_lora_fast_tuned.py",
        "--method",
        method,
        "--base_model_path",
        str(args.base_model_path),
        "--output_dir",
        str(output_dir),
        "--run_name",
        output_dir.name,
        "--data_dir",
        str(args.prepared_data_dir),
        "--seed",
        str(seed),
        "--max_seq_len",
        str(args.max_seq_len),
        "--max_steps",
        str(args.max_steps),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--warmup_steps",
        str(args.warmup_steps),
        "--logging_steps",
        str(args.logging_steps),
        "--save_steps",
        str(args.save_steps),
        "--lr_scheduler_type",
        str(args.lr_scheduler_type),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_target_modules",
        str(args.lora_target_modules),
        "--rope_base",
        str(args.rope_base),
        "--anchor_factor",
        str(args.anchor_factor),
        "--slope_raw",
        str(args.slope_raw),
        "--center_ratio",
        str(args.center_ratio),
        "--attn_implementation",
        str(args.attn_implementation),
        "--model_cache_dir",
        str(args.model_cache_dir),
        "--optim",
        str(args.optim),
        "--bf16",
        "--gradient_checkpointing",
        "--trust_remote_code",
        "--local_files_only",
        "--load_in_4bit",
    ]


def get_git_head(repo_root: Path) -> str:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root.as_posix(),
        capture_output=True,
        text=True,
    )
    return p.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="New LongAlpaca/LongQA isolated LoRA trainer.")
    parser.add_argument("--base_model_path", type=Path, default=MODEL_LOCK_DEFAULT)
    parser.add_argument("--longalpaca_path", type=Path, default=Path("/root/autodl-tmp/dfrope/datasets/LongAlpaca-12k.json"))
    parser.add_argument("--longqa_path", type=Path, default=None)
    parser.add_argument("--wikitext_train_path", type=Path, default=Path("/root/autodl-tmp/wikitext_data/train.txt"))
    parser.add_argument("--mix_longalpaca_ratio", type=float, default=0.7)
    parser.add_argument("--mix_wikitext_ratio", type=float, default=0.3)
    parser.add_argument("--max_long_records", type=int, default=12000)
    parser.add_argument("--max_wiki_samples", type=int, default=6000)
    parser.add_argument("--prepared_data_dir", type=Path, default=Path("artifacts/new_dataset_v1/prepared_data"))
    parser.add_argument("--output_root", type=Path, default=Path("artifacts/new_dataset_v1/runs"))
    parser.add_argument("--manifest_json", type=Path, default=Path("artifacts/new_dataset_v1/train_manifest.json"))
    parser.add_argument("--methods", type=str, default="baseline,anchored_sigmoid")
    parser.add_argument("--seeds", type=str, default="42,1337")
    parser.add_argument("--max_seq_len", type=int, default=16384)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--attn_implementation", type=str, default="auto")
    parser.add_argument("--model_cache_dir", type=Path, default=Path("/root/autodl-tmp/dfrope/ms_models"))
    parser.add_argument("--rope_base", type=float, default=ROPE_THETA_LOCK)
    parser.add_argument("--anchor_factor", type=float, default=4.0)
    parser.add_argument("--slope_raw", type=float, default=20.0)
    parser.add_argument("--center_ratio", type=float, default=0.70)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    model_info = assert_model_lock(args.base_model_path)
    if abs(float(args.rope_base) - ROPE_THETA_LOCK) > 1e-6:
        raise RuntimeError(f"rope_base must stay locked at {ROPE_THETA_LOCK}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    long_pool = load_long_instruction_pool(
        longalpaca_path=args.longalpaca_path if args.longalpaca_path.exists() else None,
        longqa_path=args.longqa_path if args.longqa_path else None,
        max_records=args.max_long_records,
    )
    if not long_pool:
        raise RuntimeError(
            "No long-instruction data available. Provide --longalpaca_path or --longqa_path."
        )

    if not args.wikitext_train_path.exists():
        raise RuntimeError(f"Missing WikiText train file: {args.wikitext_train_path}")
    wiki_pool = build_wikitext_instruction_samples(
        wiki_train_txt=args.wikitext_train_path,
        max_samples=args.max_wiki_samples,
        seed=42,
    )
    if not wiki_pool:
        raise RuntimeError("Failed to build instruction-style WikiText samples.")

    dataset_info = prepare_mixed_text_data(
        tokenizer=tokenizer,
        long_pool=long_pool,
        wiki_pool=wiki_pool,
        out_dir=args.prepared_data_dir,
        mix_long_ratio=float(args.mix_longalpaca_ratio),
        mix_wiki_ratio=float(args.mix_wikitext_ratio),
        seed=42,
    )

    methods = parse_csv(args.methods)
    seeds = parse_seed_csv(args.seeds)
    if not methods or not seeds:
        raise RuntimeError("methods/seeds cannot be empty.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    run_records: List[Dict[str, object]] = []
    for method in methods:
        if method not in {"baseline", "anchored_sigmoid"}:
            raise RuntimeError(f"Unsupported method: {method}")
        for seed in seeds:
            run_name = f"newds_{method}_seed{seed}_r{args.lora_rank}_s{args.max_steps}"
            out_dir = args.output_root / run_name
            if args.skip_existing and out_dir.exists() and (out_dir / "adapter_config.json").exists():
                run_records.append(
                    {
                        "run_name": run_name,
                        "method": method,
                        "seed": seed,
                        "status": "skipped_existing",
                        "output_dir": out_dir.as_posix(),
                    }
                )
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_train_cmd(args=args, method=method, seed=seed, output_dir=out_dir)
            rec: Dict[str, object] = {
                "run_name": run_name,
                "method": method,
                "seed": seed,
                "status": "dry_run" if args.dry_run else "started",
                "output_dir": out_dir.as_posix(),
                "cmd": cmd,
            }
            run_records.append(rec)
            (out_dir / "new_dataset_launch.json").write_text(
                json.dumps(rec, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if args.dry_run:
                print("DRY RUN:", " ".join(cmd), flush=True)
                continue
            print(f"[{now()}] Launch {run_name}", flush=True)
            subprocess.run(cmd, cwd=repo_root.as_posix(), check=True)
            rec["status"] = "finished"

    manifest = {
        "timestamp": now(),
        "script": Path(__file__).name,
        "git_head": get_git_head(repo_root),
        "model_lock": model_info,
        "dataset_policy": {
            "longalpaca_path": args.longalpaca_path.as_posix(),
            "longqa_path": args.longqa_path.as_posix() if args.longqa_path else "",
            "wikitext_train_path": args.wikitext_train_path.as_posix(),
            "mix_longalpaca_ratio": float(args.mix_longalpaca_ratio),
            "mix_wikitext_ratio": float(args.mix_wikitext_ratio),
            "max_long_records": int(args.max_long_records),
            "max_wiki_samples": int(args.max_wiki_samples),
        },
        "dataset_build": dataset_info,
        "training_lock": {
            "max_steps": int(args.max_steps),
            "lora_rank": int(args.lora_rank),
            "lora_alpha": int(args.lora_alpha),
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.per_device_train_batch_size),
            "grad_accum": int(args.gradient_accumulation_steps),
            "optimizer": args.optim,
            "rope_base": float(args.rope_base),
            "injection_path": "inv_freq.copy() (train_cross_model_lora_fast_tuned.py)",
            "anchor_factor": float(args.anchor_factor),
            "slope_raw": float(args.slope_raw),
            "center_ratio": float(args.center_ratio),
        },
        "code_hashes": {
            "new_lora_longalpaca_train.py": sha256_file(Path(__file__)),
            "scripts/train_cross_model_lora_fast_tuned.py": sha256_file(repo_root / "scripts" / "train_cross_model_lora_fast_tuned.py"),
            "scripts/prepare_long_instruction_mix.py": sha256_file(repo_root / "scripts" / "prepare_long_instruction_mix.py"),
        },
        "runs": run_records,
    }

    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest: {args.manifest_json}", flush=True)


if __name__ == "__main__":
    main()
