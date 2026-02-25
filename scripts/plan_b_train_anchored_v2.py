#!/usr/bin/env python3
"""
Plan B training pipeline for strict NeurIPS audit recovery.

Design goals:
- Hard-lock model integrity to Meta-Llama-3-8B-Instruct (8K-native).
- Use long-context instruction data (not plain WikiText).
- Keep the existing low-compute LoRA footprint and inv_freq.copy() training path.
- Emit reproducibility manifests for every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from transformers import AutoTokenizer


MODEL_LOCK_NAME = "Meta-Llama-3-8B-Instruct"
MODEL_LOCK_POS_EMB = 8192
FORBIDDEN_MODEL_TOKENS = ("llama-3.1", "128k")
FORBIDDEN_DATASET_TOKENS = ("wikitext", "wiki103", "wiki-103")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_seed_csv(raw: str) -> List[int]:
    return [int(x) for x in parse_csv(raw)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_model_integrity(base_model_path: Path) -> Dict[str, object]:
    p = base_model_path.as_posix()
    p_low = p.lower()
    if MODEL_LOCK_NAME.lower() not in p_low:
        raise RuntimeError(
            "Model integrity violation: expected base model path containing "
            f"'{MODEL_LOCK_NAME}', got '{p}'."
        )
    for token in FORBIDDEN_MODEL_TOKENS:
        if token in p_low:
            raise RuntimeError(
                "Model integrity violation: forbidden model token detected in path "
                f"('{token}') for '{p}'."
            )

    cfg_path = base_model_path / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"Missing config.json under base model path: {cfg_path}")
    cfg = read_json(cfg_path)
    max_pos = int(cfg.get("max_position_embeddings", -1))
    model_type = str(cfg.get("model_type", ""))
    if max_pos != MODEL_LOCK_POS_EMB:
        raise RuntimeError(
            "Model integrity violation: expected max_position_embeddings=8192 "
            f"for 8K-native Llama-3-8B, got {max_pos}."
        )
    if model_type and model_type.lower() != "llama":
        raise RuntimeError(
            f"Model integrity violation: expected model_type='llama', got '{model_type}'."
        )
    return {
        "base_model_path": p,
        "config_json": cfg_path.as_posix(),
        "model_type": model_type,
        "max_position_embeddings": max_pos,
        "config_sha256": sha256_file(cfg_path),
    }


def assert_dataset_integrity(long_instruction_jsonl: Path) -> None:
    p = long_instruction_jsonl.as_posix().lower()
    for token in FORBIDDEN_DATASET_TOKENS:
        if token in p:
            raise RuntimeError(
                "Dataset integrity violation: Plan B expects instruction-style long-context data, "
                f"but dataset path contains forbidden token '{token}': {long_instruction_jsonl}"
            )
    if not long_instruction_jsonl.exists():
        raise RuntimeError(f"Missing dataset jsonl: {long_instruction_jsonl}")


def convert_messages(obj: Dict) -> List[Dict[str, str]]:
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
            elif role_raw in {"gpt", "assistant"}:
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
    output = str(obj.get("output", obj.get("response", ""))).strip()
    prompt = instruction
    if extra_input:
        prompt = f"{instruction}\n\n{extra_input}" if instruction else extra_input
    if prompt and output:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output},
        ]
    return []


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def build_training_text(
    tokenizer: AutoTokenizer,
    dataset_jsonl: Path,
    output_data_dir: Path,
    max_records: int,
) -> Dict[str, object]:
    output_data_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_data_dir / "train.txt"
    valid_path = output_data_dir / "valid.txt"
    test_path = output_data_dir / "test.txt"

    all_texts: List[str] = []
    invalid_count = 0
    for idx, obj in enumerate(iter_jsonl(dataset_jsonl)):
        if max_records > 0 and idx >= max_records:
            break
        messages = convert_messages(obj)
        if not messages:
            invalid_count += 1
            continue
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        text = str(text).strip()
        if text:
            all_texts.append(text)
        else:
            invalid_count += 1

    if not all_texts:
        raise RuntimeError("No valid training samples were built from long-instruction jsonl.")

    n = len(all_texts)
    n_valid = max(1, int(n * 0.01))
    n_test = max(1, int(n * 0.01))
    n_train = max(1, n - n_valid - n_test)
    train_texts = all_texts[:n_train]
    valid_texts = all_texts[n_train : n_train + n_valid]
    test_texts = all_texts[n_train + n_valid :]
    if not valid_texts:
        valid_texts = train_texts[:1]
    if not test_texts:
        test_texts = train_texts[:1]

    train_path.write_text("\n\n".join(train_texts) + "\n", encoding="utf-8")
    valid_path.write_text("\n\n".join(valid_texts) + "\n", encoding="utf-8")
    test_path.write_text("\n\n".join(test_texts) + "\n", encoding="utf-8")

    return {
        "dataset_jsonl": dataset_jsonl.as_posix(),
        "dataset_jsonl_sha256": sha256_file(dataset_jsonl),
        "num_valid_samples": len(all_texts),
        "num_invalid_or_skipped": int(invalid_count),
        "train_txt": train_path.as_posix(),
        "valid_txt": valid_path.as_posix(),
        "test_txt": test_path.as_posix(),
        "train_txt_sha256": sha256_file(train_path),
    }


def build_train_cmd(args: argparse.Namespace, method: str, seed: int, output_dir: Path, data_dir: Path) -> List[str]:
    cmd = [
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
        str(data_dir),
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
        "--attn_implementation",
        str(args.attn_implementation),
        "--optim",
        str(args.optim),
        "--bf16",
        "--gradient_checkpointing",
        "--trust_remote_code",
        "--local_files_only",
        "--load_in_4bit",
        "--anchor_factor",
        str(args.anchor_factor),
        "--slope_raw",
        str(args.slope_raw),
        "--center_ratio",
        str(args.center_ratio),
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan B strict training launcher (isolated file).")
    parser.add_argument(
        "--base_model_path",
        type=Path,
        default=Path("/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"),
    )
    parser.add_argument(
        "--long_instruction_jsonl",
        type=Path,
        required=True,
        help="High-quality long-context instruction data (e.g., LongAlpaca-style jsonl).",
    )
    parser.add_argument(
        "--prepared_data_dir",
        type=Path,
        default=Path("artifacts/plan_b_data/long_instruction_llama3_8k"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("artifacts/plan_b_runs"),
    )
    parser.add_argument("--seeds", type=str, default="42,1337")
    parser.add_argument("--methods", type=str, default="baseline,anchored_sigmoid")
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=16384)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--attn_implementation", type=str, default="auto")
    parser.add_argument("--anchor_factor", type=float, default=4.0)
    parser.add_argument("--slope_raw", type=float, default=20.0)
    parser.add_argument("--center_ratio", type=float, default=0.70)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_info = assert_model_integrity(args.base_model_path)
    assert_dataset_integrity(args.long_instruction_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    data_info = build_training_text(
        tokenizer=tokenizer,
        dataset_jsonl=args.long_instruction_jsonl,
        output_data_dir=args.prepared_data_dir,
        max_records=args.max_records,
    )

    methods = parse_csv(args.methods)
    seeds = parse_seed_csv(args.seeds)
    if not methods:
        raise RuntimeError("No methods selected. Expected baseline and/or anchored_sigmoid.")
    if not seeds:
        raise RuntimeError("No seeds selected.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": now(),
        "script": Path(__file__).name,
        "base_model": base_info,
        "dataset": data_info,
        "protocol": {
            "methods": methods,
            "seeds": seeds,
            "max_steps": int(args.max_steps),
            "max_seq_len": int(args.max_seq_len),
            "batch_size": int(args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "learning_rate": float(args.learning_rate),
            "lora_rank": int(args.lora_rank),
            "anchor_factor": float(args.anchor_factor),
            "slope_raw": float(args.slope_raw),
            "center_ratio": float(args.center_ratio),
        },
        "runs": [],
    }

    for method in methods:
        if method not in {"baseline", "anchored_sigmoid"}:
            raise RuntimeError(f"Unsupported method for Plan B: {method}")
        for seed in seeds:
            run_name = f"planb_llama3_8b_{method}_{seed}"
            output_dir = args.output_root / run_name
            checkpoints = sorted(output_dir.glob("checkpoint-*")) if output_dir.exists() else []
            if args.skip_existing and checkpoints:
                manifest["runs"].append(
                    {
                        "run_name": run_name,
                        "status": "skipped_existing",
                        "output_dir": output_dir.as_posix(),
                        "checkpoints": [p.name for p in checkpoints],
                    }
                )
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_train_cmd(args=args, method=method, seed=seed, output_dir=output_dir, data_dir=args.prepared_data_dir)
            run_rec = {
                "run_name": run_name,
                "status": "dry_run" if args.dry_run else "scheduled",
                "output_dir": output_dir.as_posix(),
                "command": cmd,
            }
            (output_dir / "plan_b_launch.json").write_text(json.dumps(run_rec, indent=2), encoding="utf-8")
            manifest["runs"].append(run_rec)
            if args.dry_run:
                print("DRY RUN:", " ".join(cmd), flush=True)
                continue

            print(f"[{now()}] Launch {run_name}", flush=True)
            subprocess.run(cmd, cwd=repo_root.as_posix(), check=True)
            run_rec["status"] = "finished"

    manifest_path = args.output_root / "plan_b_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
