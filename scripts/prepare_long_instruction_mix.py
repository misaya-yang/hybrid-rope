#!/usr/bin/env python3
"""
Prepare a mixed long-instruction text corpus for LoRA training.

Features:
- Accepts multiple JSONL sources with heterogeneous schemas:
  - {"messages": [...]}
  - {"conversations": [...]}
  - {"instruction", "input", "output"/"response"}
- Enforces target mix for train split:
  - language ratio (default en/zh = 70/30)
  - task-group ratio (default qa_retrieval/summary/dialogue_code_structured = 40/30/30)
  - length-bucket weighting (2k/4k/8k/16k default 1:2:3:4)
- Emits train/valid/test text files and an auditable mix_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional dependency for offline utility
    AutoTokenizer = None  # type: ignore


LANG_KEYS = ("en", "zh")
TASK_KEYS = ("qa_retrieval", "summary", "dialogue_code_structured")
BUCKET_KEYS = ("2k", "4k", "8k", "16k")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_ratio_spec(text: str, keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid ratio token '{part}', expected key:value")
        k, v = part.split(":", 1)
        k = k.strip()
        if k not in keys:
            raise ValueError(f"Unsupported ratio key '{k}', expected one of {keys}")
        out[k] = float(v.strip())
    if not out:
        raise ValueError("Ratio spec is empty.")
    for k in keys:
        out.setdefault(k, 0.0)
    s = sum(out.values())
    if s <= 0:
        raise ValueError(f"Invalid ratio sum {s}, must be > 0.")
    return {k: float(v / s) for k, v in out.items()}


def parse_source_spec(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid source token '{part}', expected k=v")
        k, v = part.split("=", 1)
        fields[k.strip()] = v.strip()
    required = ("name", "path")
    miss = [x for x in required if not fields.get(x)]
    if miss:
        raise ValueError(f"Missing source fields {miss} in '{text}'")
    fields.setdefault("lang", "auto")
    fields.setdefault("task_type", "auto")
    fields.setdefault("filter_rule", "none")
    return fields


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def infer_lang(text: str) -> str:
    if not text:
        return "en"
    cjk = sum(1 for ch in text if is_cjk(ch))
    ratio = cjk / max(1, len(text))
    return "zh" if ratio >= 0.12 else "en"


def normalize_task_type(raw: str) -> str:
    v = (raw or "").strip().lower()
    if v in {"qa", "retrieval", "qa_retrieval"}:
        return "qa_retrieval"
    if v in {"summary", "summarization"}:
        return "summary"
    if v in {"dialogue", "code", "structured", "dialogue_code_structured"}:
        return "dialogue_code_structured"
    return "dialogue_code_structured"


def infer_task_type(source_name: str, text: str, obj: Dict) -> str:
    hint = f"{source_name} {json.dumps(obj, ensure_ascii=False)[:500]}".lower()
    if any(k in hint for k in ("qasper", "hotpot", "retrieval", "qa", "question")):
        return "qa_retrieval"
    if any(k in hint for k in ("summary", "summar", "tl;dr", "multi_news", "gov_report", "qmsum")):
        return "summary"
    if "```" in text or re.search(r"\b(def|class|function|import)\b", text):
        return "dialogue_code_structured"
    return "dialogue_code_structured"


def token_length(text: str, tokenizer) -> int:
    if tokenizer is not None:
        return int(len(tokenizer.encode(text, add_special_tokens=False)))
    # Fallback approximation when transformers/tokenizer is unavailable.
    return max(1, len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)))


def length_bucket(n_tokens: int) -> str:
    if n_tokens <= 2048:
        return "2k"
    if n_tokens <= 4096:
        return "4k"
    if n_tokens <= 8192:
        return "8k"
    return "16k"


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


def render_messages(messages: List[Dict[str, str]], tokenizer) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return str(txt).strip()
        except Exception:
            pass
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages).strip()


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


@dataclass
class Sample:
    text: str
    source_name: str
    lang: str
    task_group: str
    token_len: int
    bucket: str
    text_sha1: str


def sample_with_fallback(
    rng: random.Random,
    pool_exact: List[int],
    pool_lang_task: List[int],
    pool_lang: List[int],
    pool_any: List[int],
    n: int,
) -> List[int]:
    picked: List[int] = []
    sources = [pool_exact, pool_lang_task, pool_lang, pool_any]
    for p in sources:
        if len(picked) >= n:
            break
        need = n - len(picked)
        if not p:
            continue
        if len(p) >= need:
            picked.extend(rng.sample(p, need))
        else:
            picked.extend(rng.sample(p, len(p)))
            for _ in range(need - len(p)):
                picked.append(rng.choice(p))
    return picked


def normalize_targets(targets: Dict[Tuple[str, str, str], int], total: int) -> Dict[Tuple[str, str, str], int]:
    cur = sum(targets.values())
    if cur == total:
        return targets
    keys = sorted(targets.keys())
    if cur < total:
        i = 0
        while cur < total:
            targets[keys[i % len(keys)]] += 1
            cur += 1
            i += 1
    else:
        i = 0
        while cur > total and keys:
            k = keys[i % len(keys)]
            if targets[k] > 0:
                targets[k] -= 1
                cur -= 1
            i += 1
    return targets


def build_targets(
    total: int,
    lang_ratio: Dict[str, float],
    task_ratio: Dict[str, float],
    bucket_ratio: Dict[str, float],
) -> Dict[Tuple[str, str, str], int]:
    targets: Dict[Tuple[str, str, str], int] = {}
    for lang in LANG_KEYS:
        for task in TASK_KEYS:
            for bucket in BUCKET_KEYS:
                frac = lang_ratio[lang] * task_ratio[task] * bucket_ratio[bucket]
                targets[(lang, task, bucket)] = int(round(total * frac))
    return normalize_targets(targets, total)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare mixed long-instruction corpus text files.")
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Source spec, repeatable. Format: "
            "name=<id>;path=<jsonl>;lang=<en|zh|auto>;task_type=<qa_retrieval|summary|dialogue_code_structured|auto>;"
            "filter_rule=<text>"
        ),
    )
    ap.add_argument("--tokenizer_path", type=str, default="")
    ap.add_argument("--output_dir", type=str, default="artifacts/plan_b_data/long_instruction_mix")
    ap.add_argument("--target_train_samples", type=int, default=0)
    ap.add_argument("--max_records_per_source", type=int, default=0)
    ap.add_argument("--valid_ratio", type=float, default=0.01)
    ap.add_argument("--test_ratio", type=float, default=0.01)
    ap.add_argument("--language_ratio", type=str, default="en:0.7,zh:0.3")
    ap.add_argument(
        "--task_ratio",
        type=str,
        default="qa_retrieval:0.4,summary:0.3,dialogue_code_structured:0.3",
    )
    ap.add_argument("--bucket_ratio", type=str, default="2k:1,4k:2,8k:3,16k:4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable_dedupe", action="store_true")
    args = ap.parse_args()

    if not args.source:
        raise RuntimeError("At least one --source spec is required.")

    if args.valid_ratio < 0 or args.test_ratio < 0 or args.valid_ratio + args.test_ratio >= 0.5:
        raise ValueError("valid_ratio/test_ratio must be non-negative and sum < 0.5")

    lang_ratio = parse_ratio_spec(args.language_ratio, LANG_KEYS)
    task_ratio = parse_ratio_spec(args.task_ratio, TASK_KEYS)
    bucket_ratio = parse_ratio_spec(args.bucket_ratio, BUCKET_KEYS)

    tokenizer = None
    if args.tokenizer_path:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is unavailable but tokenizer_path was provided.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, local_files_only=True)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    samples: List[Sample] = []
    seen: set[str] = set()
    source_stats: List[Dict[str, object]] = []

    for spec_text in args.source:
        spec = parse_source_spec(spec_text)
        source_name = spec["name"]
        source_path = Path(spec["path"]).expanduser().resolve()
        source_lang = spec.get("lang", "auto").strip().lower()
        source_task = spec.get("task_type", "auto").strip().lower()
        filter_rule = spec.get("filter_rule", "none")
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source file: {source_path}")
        source_sha = sha256_file(source_path)

        raw_count = 0
        kept_count = 0
        token_total = 0
        for obj in iter_jsonl(source_path):
            raw_count += 1
            if args.max_records_per_source > 0 and raw_count > int(args.max_records_per_source):
                break
            messages = convert_messages(obj)
            if not messages:
                continue
            text = render_messages(messages, tokenizer=tokenizer)
            if not text:
                continue
            text_hash = sha1_text(text)
            if not args.disable_dedupe and text_hash in seen:
                continue
            seen.add(text_hash)
            lang = source_lang if source_lang in LANG_KEYS else infer_lang(text)
            if lang not in LANG_KEYS:
                lang = "en"
            if source_task == "auto":
                task = infer_task_type(source_name=source_name, text=text, obj=obj)
            else:
                task = normalize_task_type(source_task)
            task = normalize_task_type(task)
            tlen = token_length(text, tokenizer=tokenizer)
            bucket = length_bucket(tlen)
            samples.append(
                Sample(
                    text=text,
                    source_name=source_name,
                    lang=lang,
                    task_group=task,
                    token_len=tlen,
                    bucket=bucket,
                    text_sha1=text_hash,
                )
            )
            kept_count += 1
            token_total += tlen

        source_stats.append(
            {
                "source_name": source_name,
                "lang": source_lang,
                "task_type": normalize_task_type(source_task) if source_task != "auto" else "auto",
                "count": kept_count,
                "sha256": source_sha,
                "filter_rule": filter_rule,
                "source_path": source_path.as_posix(),
                "source_sha256": source_sha,
                "lang_declared": source_lang,
                "task_type_declared": source_task,
                "raw_records": raw_count,
                "kept_records": kept_count,
                "kept_tokens": token_total,
            }
        )

    if not samples:
        raise RuntimeError("No valid samples collected from provided sources.")

    total_tokens = sum(s["kept_tokens"] for s in source_stats)
    for s in source_stats:
        tok = int(s["kept_tokens"])
        s["token_ratio"] = float(tok / total_tokens) if total_tokens > 0 else 0.0

    target_train = int(args.target_train_samples) if int(args.target_train_samples) > 0 else len(samples)
    targets = build_targets(target_train, lang_ratio=lang_ratio, task_ratio=task_ratio, bucket_ratio=bucket_ratio)

    idx_by_cell: Dict[Tuple[str, str, str], List[int]] = {}
    idx_by_lang_task: Dict[Tuple[str, str], List[int]] = {}
    idx_by_lang: Dict[str, List[int]] = {}
    for idx, s in enumerate(samples):
        idx_by_cell.setdefault((s.lang, s.task_group, s.bucket), []).append(idx)
        idx_by_lang_task.setdefault((s.lang, s.task_group), []).append(idx)
        idx_by_lang.setdefault(s.lang, []).append(idx)
    all_idx = list(range(len(samples)))

    selected_idx: List[int] = []
    for cell, n in sorted(targets.items()):
        lang, task, bucket = cell
        if n <= 0:
            continue
        exact = idx_by_cell.get((lang, task, bucket), [])
        lang_task = idx_by_lang_task.get((lang, task), [])
        lang_pool = idx_by_lang.get(lang, [])
        picked = sample_with_fallback(
            rng=rng,
            pool_exact=exact,
            pool_lang_task=lang_task,
            pool_lang=lang_pool,
            pool_any=all_idx,
            n=n,
        )
        selected_idx.extend(picked)

    if len(selected_idx) < target_train:
        for _ in range(target_train - len(selected_idx)):
            selected_idx.append(rng.choice(all_idx))
    elif len(selected_idx) > target_train:
        selected_idx = selected_idx[:target_train]

    rng.shuffle(selected_idx)
    train_samples = [samples[i] for i in selected_idx]

    valid_n = max(1, int(round(target_train * float(args.valid_ratio))))
    test_n = max(1, int(round(target_train * float(args.test_ratio))))
    candidate_rest = [i for i in all_idx if i not in set(selected_idx)]
    rng.shuffle(candidate_rest)
    if len(candidate_rest) < valid_n + test_n:
        # Fallback to train pool with replacement when corpus is small.
        candidate_rest.extend(rng.choices(selected_idx, k=valid_n + test_n - len(candidate_rest)))
    valid_samples = [samples[i] for i in candidate_rest[:valid_n]]
    test_samples = [samples[i] for i in candidate_rest[valid_n : valid_n + test_n]]

    train_path = out_dir / "train.txt"
    valid_path = out_dir / "valid.txt"
    test_path = out_dir / "test.txt"
    train_path.write_text("\n\n".join(s.text for s in train_samples) + "\n", encoding="utf-8")
    valid_path.write_text("\n\n".join(s.text for s in valid_samples) + "\n", encoding="utf-8")
    test_path.write_text("\n\n".join(s.text for s in test_samples) + "\n", encoding="utf-8")

    def dist(items: Sequence[Sample], key_name: str) -> Dict[str, Dict[str, float]]:
        counts: Dict[str, int] = {}
        for it in items:
            k = getattr(it, key_name)
            counts[k] = counts.get(k, 0) + 1
        n = max(1, len(items))
        return {k: {"count": int(v), "ratio": float(v / n)} for k, v in sorted(counts.items())}

    mix_manifest = {
        "timestamp": now(),
        "seed": int(args.seed),
        "tokenizer_path": args.tokenizer_path,
        "target_train_samples": int(target_train),
        "valid_ratio": float(args.valid_ratio),
        "test_ratio": float(args.test_ratio),
        "language_ratio_target": lang_ratio,
        "task_ratio_target": task_ratio,
        "bucket_ratio_target": bucket_ratio,
        "dedupe_enabled": not bool(args.disable_dedupe),
        "sources": source_stats,
        "output_files": {
            "train_txt": train_path.as_posix(),
            "valid_txt": valid_path.as_posix(),
            "test_txt": test_path.as_posix(),
            "train_sha256": sha256_file(train_path),
            "valid_sha256": sha256_file(valid_path),
            "test_sha256": sha256_file(test_path),
        },
        "train_distribution": {
            "language": dist(train_samples, "lang"),
            "task_group": dist(train_samples, "task_group"),
            "length_bucket": dist(train_samples, "bucket"),
            "avg_token_len": float(sum(s.token_len for s in train_samples) / max(1, len(train_samples))),
        },
        "valid_distribution": {
            "language": dist(valid_samples, "lang"),
            "task_group": dist(valid_samples, "task_group"),
            "length_bucket": dist(valid_samples, "bucket"),
        },
        "test_distribution": {
            "language": dist(test_samples, "lang"),
            "task_group": dist(test_samples, "task_group"),
            "length_bucket": dist(test_samples, "bucket"),
        },
    }

    manifest_path = out_dir / "mix_manifest.json"
    manifest_path.write_text(json.dumps(mix_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(mix_manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
