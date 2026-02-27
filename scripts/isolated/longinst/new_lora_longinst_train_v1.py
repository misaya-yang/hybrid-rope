#!/usr/bin/env python3
"""Ultra-fast long-instruction LoRA v1 (response-only) pipeline.

Task coverage:
- Task 0: freeze known-good protocol
- Task 1: build long-instruction mix dataset + stats
- Task 2: response-only LoRA training with anchored RoPE injection
- Task 3: gate eval on qasper/musique
- Task 4: optional full lb21 eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


DEFAULT_BASE_MODEL = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
DEFAULT_KNOWN_GOOD_RUN_CONFIG = (
    "artifacts/new_attnbias_v1/train/llama3_jointopt_v5_sdpa_higher_util_s42/run_config.json"
)
DEFAULT_LONGALPACA = "/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl"
DEFAULT_LONGQA = "/root/autodl-tmp/dfrope/datasets/LongQA.jsonl"
DEFAULT_WIKITEXT = "/root/autodl-tmp/wikitext_data/train.txt"
DEFAULT_MORNING_REF = (
    "artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json"
)
TRAIN_TRUNCATE_MODE = "head_tail_keep_drop_middle"
TRAIN_TRUNCATE_HEAD_CAP = 500


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def canonicalize_source_name(raw_source: str) -> str:
    s = str(raw_source or "").strip().lower()
    if not s:
        return "long"
    wiki_aliases = {
        "wiki",
        "wikitext",
        "power_law",
        "powerlaw",
        "power-law",
        "redpajama",
        "slimpajama",
        "book",
        "arxiv",
    }
    synthetic_aliases = {
        "synthetic",
        "multihop",
        "multi_hop",
        "multihop_qa",
        "qa",
        "reasoning",
        "hotpotqa",
        "musique",
        "2wikimqa",
        "mid_band",
        "midband",
    }
    long_aliases = {
        "long",
        "longinst",
        "long_instruction",
        "instruction",
        "format_scaffold",
        "scaffold",
        "sharegpt",
        "alpaca",
    }
    if s in wiki_aliases:
        return "wiki"
    if s in synthetic_aliases:
        return "synthetic"
    if s in long_aliases:
        return "long"
    return "long"


def parse_ratio_like(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    if v > 1.0:
        v = v / 100.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def lookup_manifest_ratio(manifest: Dict, keys: List[str]) -> Optional[float]:
    containers: List[Dict] = []
    if isinstance(manifest, dict):
        containers.append(manifest)
        targets = manifest.get("targets")
        if isinstance(targets, dict):
            containers.append(targets)
    for container in containers:
        for key in keys:
            if key in container:
                parsed = parse_ratio_like(container.get(key))
                if parsed is not None:
                    return parsed
    return None


def iter_json_records(path: Path) -> Iterable[Dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
        return
    raw = load_json(path)
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item


@dataclass
class RenderedSample:
    source: str
    messages: List[Dict[str, str]]
    text: str
    prefix_text: str


class CausalLMCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []
        for f in features:
            ids = list(f["input_ids"])
            attn = list(f["attention_mask"])
            lbl = list(f["labels"])
            pad_n = max_len - len(ids)
            if pad_n > 0:
                ids = ids + [self.pad_token_id] * pad_n
                attn = attn + [0] * pad_n
                lbl = lbl + [-100] * pad_n
            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lbl)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class JsonlLogCallback(TrainerCallback):
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        row = {"step": int(state.global_step), **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class ResponseOnlyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_effective_label_tokens = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        labels = inputs.get("labels")
        if labels is not None:
            self.last_effective_label_tokens = float((labels != -100).sum().item())
        out = model(**inputs)
        loss = out.loss
        return (loss, out) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:  # type: ignore[override]
        logs = dict(logs)
        logs["effective_label_tokens"] = float(self.last_effective_label_tokens)
        super().log(logs, *args, **kwargs)


# Remaining implementation blocks are appended below.


def normalize_messages(obj: Dict) -> List[Dict[str, str]]:
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs:
        out: List[Dict[str, str]] = []
        for m in msgs:
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", "")).strip()
            if role in {"system", "user", "assistant"} and content:
                out.append({"role": role, "content": content})
        if out:
            return out

    inst = str(obj.get("instruction", "")).strip()
    inp = str(obj.get("input", "")).strip()
    out_text = str(obj.get("output", obj.get("response", obj.get("answer", "")))).strip()
    if inst and out_text:
        user = inst if not inp else f"{inst}\n\n{inp}"
        return [{"role": "user", "content": user}, {"role": "assistant", "content": out_text}]

    conv = obj.get("conversations")
    if isinstance(conv, list) and conv:
        out: List[Dict[str, str]] = []
        for c in conv:
            raw_role = str(c.get("from", c.get("role", ""))).strip().lower()
            if raw_role in {"human", "user"}:
                role = "user"
            elif raw_role in {"assistant", "gpt"}:
                role = "assistant"
            elif raw_role == "system":
                role = "system"
            else:
                role = ""
            content = str(c.get("value", c.get("content", ""))).strip()
            if role and content:
                out.append({"role": role, "content": content})
        if out:
            return out
    return []


def split_sentences(text: str) -> List[str]:
    parts: List[str] = []
    cur = []
    for ch in text:
        cur.append(ch)
        if ch in ".!?":
            s = "".join(cur).strip()
            if s:
                parts.append(s)
            cur = []
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def build_wikitext_rows(path: Path, max_samples: int, seed: int) -> List[Dict]:
    if not path.exists() or max_samples <= 0:
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    # WikiText paragraphs are often short; use a lower threshold and a fallback chunker.
    blocks = [b.strip() for b in text.split("\n\n") if len(b.strip()) >= 300]
    if len(blocks) < max_samples:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("=")]
        cur: List[str] = []
        cur_chars = 0
        stitched: List[str] = []
        for ln in lines:
            cur.append(ln)
            cur_chars += len(ln) + 1
            if cur_chars >= 450:
                stitched.append(" ".join(cur))
                cur = []
                cur_chars = 0
        if cur_chars >= 300:
            stitched.append(" ".join(cur))
        blocks.extend(stitched)
    rng = random.Random(seed)
    rng.shuffle(blocks)
    rows: List[Dict] = []
    seen: set[str] = set()
    for block in blocks:
        if len(rows) >= max_samples:
            break
        block = " ".join(block.split())
        if len(block) < 400:
            continue
        if block in seen:
            continue
        seen.add(block)
        sents = split_sentences(block)
        if len(sents) >= 2:
            summary = " ".join(sents[:2]).strip()
        else:
            summary = (block[:200].strip() + "...") if len(block) > 220 else block.strip()
        if not summary:
            continue
        rows.append(
            {
                "instruction": "Summarize the passage in two concise sentences.",
                "input": block[:2200],
                "output": summary,
            }
        )
    return rows


def build_synthetic_long_qa_rows(n_samples: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    rows: List[Dict] = []
    for i in range(max(0, int(n_samples))):
        key = f"K{seed % 1000:03d}-{i:05d}"
        project = f"Project-{key}"
        ref_num = str(700000 + i)
        owner = f"Owner-{rng.randint(10, 99)}"
        shard = f"Shard-{rng.randint(1, 8)}"
        ticket = f"TKT-{rng.randint(100000, 999999)}"
        checksum = f"C{rng.randint(1000000, 9999999)}"
        filler_left = " ".join([f"tok{rng.randint(1000,9999)}" for _ in range(220)])
        filler_right = " ".join([f"tok{rng.randint(1000,9999)}" for _ in range(220)])
        answer = (
            f"The complete verification information is as follows. "
            f"The verification key is {key}. The project codename is {project}. "
            f"The reference number is {ref_num}. The owner is {owner} and the serving shard is {shard}. "
            f"The confirmation ticket is {ticket} with checksum {checksum}. "
            f"This answer is obtained by combining the ledger entry, project registry, and audit trail statements in the context."
        )
        context = (
            f"{filler_left} "
            f"Ledger entry: key [{key}] maps to project [{project}] and reference number [{ref_num}]. "
            f"Registry entry: project [{project}] belongs to owner [{owner}] on shard [{shard}]. "
            f"Audit entry: owner [{owner}] approved ticket [{ticket}] with checksum [{checksum}] for key [{key}]. "
            f"{filler_right}"
        )
        rows.append(
            {
                "instruction": "Read the long context carefully and answer with a complete sentence.",
                "input": (
                    f"Context:\n{context}\n\n"
                    "Question:\nWhat is the complete verification information for this record? "
                    "Include key, project codename, reference number, owner, shard, ticket, and checksum."
                ),
                "output": answer,
            }
        )
    return rows


def looks_like_continuation_prompt(messages: List[Dict[str, str]]) -> bool:
    if not messages:
        return False
    user_text = ""
    for m in messages:
        if m.get("role") == "user":
            user_text = str(m.get("content", "")).strip().lower()
            break
    if not user_text:
        return False
    markers = [
        "long-context continuation",
        "continue the passage",
        "continue the text",
        "faithfully from the given long context",
    ]
    return any(m in user_text for m in markers)


def render_sample(tokenizer: AutoTokenizer, messages: List[Dict[str, str]], source: str) -> Optional[RenderedSample]:
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        return None
    if not str(messages[-1].get("content", "")).strip():
        return None
    prefix_msgs = list(messages[:-1]) + [{"role": "assistant", "content": ""}]
    text = str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    prefix_text = str(tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=False))
    # Do not strip boundary whitespace; response-only masking relies on exact prefix boundary.
    if not text.strip() or not prefix_text.strip():
        return None
    return RenderedSample(source=source, messages=messages, text=text, prefix_text=prefix_text)


def quantiles(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    vals = sorted(values)

    def q(p: float) -> float:
        if len(vals) == 1:
            return float(vals[0])
        idx = int(round((len(vals) - 1) * p))
        idx = max(0, min(idx, len(vals) - 1))
        return float(vals[idx])

    return {
        "min": float(vals[0]),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": float(vals[-1]),
        "mean": float(statistics.mean(vals)),
    }


def build_longinst_mix(
    tokenizer: AutoTokenizer,
    long_paths: List[Path],
    wikitext_path: Path,
    max_long_records: int,
    max_wiki_samples: int,
    mix_long_ratio: float,
    mix_wiki_ratio: float,
    synthetic_ratio: float,
    allow_continuation_dominant_corpus: bool,
    seed: int,
    out_jsonl: Path,
    preview_txt: Path,
) -> Tuple[List[RenderedSample], Dict[str, object]]:
    rng = random.Random(seed)
    long_rows: List[Dict] = []
    for p in long_paths:
        if not p.exists():
            continue
        for obj in iter_json_records(p):
            long_rows.append(obj)
            if len(long_rows) >= max_long_records:
                break
        if len(long_rows) >= max_long_records:
            break
    if not long_rows:
        raise RuntimeError("No long-instruction rows found (LongAlpaca/LongQA missing).")

    wiki_cap_requested = max(0, int(max_wiki_samples))
    wiki_cap_effective = wiki_cap_requested
    wiki_rows = build_wikitext_rows(wikitext_path, max_samples=wiki_cap_effective, seed=seed + 19)

    total_target_rows = len(long_rows)
    r_long = max(0.0, float(mix_long_ratio))
    r_wiki = max(0.0, float(mix_wiki_ratio))
    r_syn = max(0.0, float(synthetic_ratio))
    r_sum = r_long + r_wiki + r_syn
    if r_sum <= 0.0:
        raise RuntimeError("Invalid mix ratios: mix_long_ratio + mix_wiki_ratio + synthetic_ratio must be > 0.")

    avg_token_defaults = {"long": 2000.0, "wiki": 400.0, "synthetic": 3000.0}

    def estimate_avg_tokens(rows: List[Dict], source: str, probe_n: int, seed_offset: int) -> float:
        if not rows:
            return float(avg_token_defaults[source])
        idxs = list(range(len(rows)))
        probe_rng = random.Random(seed + seed_offset)
        probe_rng.shuffle(idxs)
        lens: List[int] = []
        for idx in idxs[: max(1, min(probe_n, len(idxs)))]:
            msgs = normalize_messages(rows[idx])
            if not msgs:
                continue
            sample = render_sample(tokenizer, msgs, source=source)
            if sample is None:
                continue
            tok_n = len(tokenizer(sample.text, add_special_tokens=False, truncation=False)["input_ids"])
            if tok_n > 0:
                lens.append(int(tok_n))
        if lens:
            return float(statistics.mean(lens))
        return float(avg_token_defaults[source])

    synthetic_probe_rows = build_synthetic_long_qa_rows(64, seed=seed + 37)
    avg_long_tokens = estimate_avg_tokens(long_rows, "long", probe_n=64, seed_offset=11)
    avg_wiki_tokens = estimate_avg_tokens(wiki_rows, "wiki", probe_n=64, seed_offset=19)
    avg_synthetic_tokens = estimate_avg_tokens(synthetic_probe_rows, "synthetic", probe_n=64, seed_offset=37)

    total_target_tokens = max(1, int(round(float(total_target_rows) * float(avg_long_tokens))))
    target_long_tokens = float(total_target_tokens) * float(r_long / r_sum)
    target_wiki_tokens = float(total_target_tokens) * float(r_wiki / r_sum)
    target_syn_tokens = float(total_target_tokens) * float(r_syn / r_sum)

    desired_long = int(round(target_long_tokens / max(1.0, float(avg_long_tokens))))
    desired_wiki = int(round(target_wiki_tokens / max(1.0, float(avg_wiki_tokens))))
    desired_syn = int(round(target_syn_tokens / max(1.0, float(avg_synthetic_tokens))))
    if r_long > 0 and desired_long <= 0:
        desired_long = 1
    if r_wiki > 0 and desired_wiki <= 0 and wiki_rows:
        desired_wiki = 1
    if r_syn > 0 and desired_syn <= 0:
        desired_syn = 1

    if desired_wiki > wiki_cap_effective:
        wiki_cap_effective = max(int(desired_wiki * 2), wiki_cap_effective)
        wiki_rows = build_wikitext_rows(wikitext_path, max_samples=wiki_cap_effective, seed=seed + 19)
        avg_wiki_tokens = estimate_avg_tokens(wiki_rows, "wiki", probe_n=64, seed_offset=19)
        desired_wiki = int(round(target_wiki_tokens / max(1.0, float(avg_wiki_tokens))))
        if r_wiki > 0 and desired_wiki <= 0 and wiki_rows:
            desired_wiki = 1

    wiki_target = min(len(wiki_rows), max(0, desired_wiki))
    synthetic_target = max(0, desired_syn)
    long_target = min(len(long_rows), max(0, desired_long))
    planned_tokens = (
        float(long_target) * float(avg_long_tokens)
        + float(wiki_target) * float(avg_wiki_tokens)
        + float(synthetic_target) * float(avg_synthetic_tokens)
    )
    if planned_tokens < float(total_target_tokens) and long_target < len(long_rows):
        deficit_tokens = float(total_target_tokens) - planned_tokens
        extra_long = int(math.ceil(deficit_tokens / max(1.0, float(avg_long_tokens))))
        long_target = min(len(long_rows), long_target + max(0, extra_long))

    rng.shuffle(long_rows)
    rng.shuffle(wiki_rows)

    synthetic_rows = build_synthetic_long_qa_rows(synthetic_target, seed=seed + 37)
    selected = (
        [("long", row) for row in long_rows[:long_target]]
        + [("synthetic", row) for row in synthetic_rows]
        + [("wiki", row) for row in wiki_rows[:wiki_target]]
    )
    rng.shuffle(selected)

    rendered: List[RenderedSample] = []
    continuation_like = 0
    source_token_counts = {"long": 0, "synthetic": 0, "wiki": 0}
    for source, row in selected:
        msgs = normalize_messages(row)
        if not msgs:
            continue
        sample = render_sample(tokenizer, msgs, source=source)
        if sample is not None:
            rendered.append(sample)
            tok_n = len(tokenizer(sample.text, add_special_tokens=False, truncation=False)["input_ids"])
            source_token_counts[source] += int(max(1, tok_n))
            if looks_like_continuation_prompt(sample.messages):
                continuation_like += 1

    if len(rendered) < 500:
        raise RuntimeError(f"Too few usable rendered samples: {len(rendered)}")

    continuation_ratio = float(continuation_like) / float(max(1, len(rendered)))
    if (not bool(allow_continuation_dominant_corpus)) and continuation_ratio > 0.70:
        raise RuntimeError(
            "Detected continuation-dominant corpus "
            f"(continuation_ratio={continuation_ratio:.3f}). "
            "This setting strongly hurts multi-hop QA (e.g. musique). "
            "Use a real long-instruction/QA mix or pass --allow_continuation_dominant_corpus."
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, s in enumerate(rendered):
            rec = {
                "id": i,
                "source": s.source,
                "messages": s.messages,
                "text": s.text,
                "prefix_text": s.prefix_text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    preview_txt.parent.mkdir(parents=True, exist_ok=True)
    with preview_txt.open("w", encoding="utf-8") as f:
        for i, s in enumerate(rendered[:20]):
            user_text = ""
            for m in s.messages:
                if m.get("role") == "user":
                    user_text = str(m.get("content", ""))
            ans = str(s.messages[-1].get("content", ""))
            f.write(f"[#{i}] source={s.source}\n")
            f.write("USER:\n" + user_text[:1200] + "\n")
            f.write("ASSISTANT:\n" + ans[:600] + "\n")
            f.write("-" * 80 + "\n")

    selected_counts = {
        "long": int(sum(1 for s in rendered if s.source == "long")),
        "synthetic": int(sum(1 for s in rendered if s.source == "synthetic")),
        "wiki": int(sum(1 for s in rendered if s.source == "wiki")),
    }
    total_selected_tokens = int(sum(int(v) for v in source_token_counts.values()))
    target_long_ratio = float(r_long / r_sum)
    target_wiki_ratio = float(r_wiki / r_sum)
    target_synthetic_ratio = float(r_syn / r_sum)
    actual_wiki = float(selected_counts["wiki"]) / float(max(1, len(rendered)))
    actual_synthetic = float(selected_counts["synthetic"]) / float(max(1, len(rendered)))
    actual_wiki_token = float(source_token_counts["wiki"]) / float(max(1, total_selected_tokens))
    actual_synthetic_token = float(source_token_counts["synthetic"]) / float(max(1, total_selected_tokens))
    actual_long_token = float(source_token_counts["long"]) / float(max(1, total_selected_tokens))
    print(
        "[DATA CHECK] "
        f"actual_wiki_ratio={actual_wiki:.4f}, actual_synthetic_ratio={actual_synthetic:.4f}, "
        f"actual_wiki_token_ratio={actual_wiki_token:.4f}, actual_synthetic_token_ratio={actual_synthetic_token:.4f}, "
        f"actual_long_token_ratio={actual_long_token:.4f}",
        flush=True,
    )
    if target_wiki_ratio > 0 and actual_wiki_token < max(0.02, target_wiki_ratio * 0.60):
        raise RuntimeError(
            f"WikiText token ratio too low: {actual_wiki_token:.4f} "
            f"(target={target_wiki_ratio:.4f}, min={max(0.02, target_wiki_ratio * 0.60):.4f})."
        )
    if target_synthetic_ratio > 0 and actual_synthetic_token < max(0.05, target_synthetic_ratio * 0.60):
        raise RuntimeError(
            f"Synthetic token ratio too low: {actual_synthetic_token:.4f} "
            f"(target={target_synthetic_ratio:.4f}, min={max(0.05, target_synthetic_ratio * 0.60):.4f})."
        )
    ratio_tolerance = 0.08
    if target_long_ratio > 0 and abs(actual_long_token - target_long_ratio) > ratio_tolerance:
        raise RuntimeError(
            f"Long token ratio drift too large: actual={actual_long_token:.4f}, "
            f"target={target_long_ratio:.4f}, tol={ratio_tolerance:.4f}."
        )
    if target_wiki_ratio > 0 and abs(actual_wiki_token - target_wiki_ratio) > ratio_tolerance:
        raise RuntimeError(
            f"Wiki token ratio drift too large: actual={actual_wiki_token:.4f}, "
            f"target={target_wiki_ratio:.4f}, tol={ratio_tolerance:.4f}."
        )
    if target_synthetic_ratio > 0 and abs(actual_synthetic_token - target_synthetic_ratio) > ratio_tolerance:
        raise RuntimeError(
            f"Synthetic token ratio drift too large: actual={actual_synthetic_token:.4f}, "
            f"target={target_synthetic_ratio:.4f}, tol={ratio_tolerance:.4f}."
        )

    stats = {
        "long_rows_raw": len(long_rows),
        "wiki_rows_raw": len(wiki_rows),
        "max_wiki_samples_requested": int(wiki_cap_requested),
        "max_wiki_samples_effective": int(wiki_cap_effective),
        "avg_long_tokens_probe": float(avg_long_tokens),
        "avg_wiki_tokens_probe": float(avg_wiki_tokens),
        "avg_synthetic_tokens_probe": float(avg_synthetic_tokens),
        "target_total_tokens": int(total_target_tokens),
        "target_long_tokens": float(target_long_tokens),
        "target_wiki_tokens": float(target_wiki_tokens),
        "target_synthetic_tokens": float(target_syn_tokens),
        "target_long_ratio": target_long_ratio,
        "target_wiki_ratio": target_wiki_ratio,
        "target_synthetic_ratio": target_synthetic_ratio,
        "target_long_count": int(desired_long),
        "target_wiki_count": int(desired_wiki),
        "target_synthetic_count": int(desired_syn),
        "selected_long": selected_counts["long"],
        "selected_synthetic": selected_counts["synthetic"],
        "selected_wiki": selected_counts["wiki"],
        "selected_long_tokens": int(source_token_counts["long"]),
        "selected_wiki_tokens": int(source_token_counts["wiki"]),
        "selected_synthetic_tokens": int(source_token_counts["synthetic"]),
        "rendered_total": len(rendered),
        "actual_long_ratio": float(selected_counts["long"]) / float(max(1, len(rendered))),
        "actual_wiki_ratio": actual_wiki,
        "actual_synthetic_ratio": actual_synthetic,
        "actual_wiki_token_ratio": actual_wiki_token,
        "actual_synthetic_token_ratio": actual_synthetic_token,
        "actual_long_token_ratio": actual_long_token,
        "actual_wiki_ratio_final": actual_wiki,
        "actual_synthetic_ratio_final": actual_synthetic,
        "continuation_like_ratio": continuation_ratio,
        "longinst_mix_jsonl": out_jsonl.as_posix(),
        "preview_20": preview_txt.as_posix(),
    }
    return rendered, stats


def write_rendered_preview(samples: List[RenderedSample], preview_txt: Path, limit: int = 20) -> None:
    preview_txt.parent.mkdir(parents=True, exist_ok=True)
    with preview_txt.open("w", encoding="utf-8") as f:
        for i, s in enumerate(samples[: max(1, int(limit))]):
            user_text = ""
            for m in s.messages:
                if m.get("role") == "user":
                    user_text = str(m.get("content", ""))
                    break
            ans = str(s.messages[-1].get("content", "")) if s.messages else ""
            f.write(f"[#{i}] source={s.source}\n")
            f.write("USER:\n" + user_text[:1200] + "\n")
            f.write("ASSISTANT:\n" + ans[:600] + "\n")
            f.write("-" * 80 + "\n")


def load_prebuilt_mixed_split(
    tokenizer: AutoTokenizer,
    mixed_dataset_dir: Path,
    split: str,
    preview_txt: Path,
) -> Tuple[List[RenderedSample], Dict[str, object], Path]:
    split = str(split).strip().lower() or "train"
    candidates = [mixed_dataset_dir / f"{split}.jsonl"]
    if split == "all":
        candidates.insert(0, mixed_dataset_dir / "mixed_prior_finetune.jsonl")
    candidates.extend(
        [
            mixed_dataset_dir / "mixed_prior_finetune.jsonl",
            mixed_dataset_dir / "longinst_mix.jsonl",
        ]
    )
    source_jsonl = next((p for p in candidates if p.exists()), None)
    if source_jsonl is None:
        raise RuntimeError(
            f"mixed_dataset_dir provided but no dataset split found under {mixed_dataset_dir.as_posix()}. "
            f"Tried: {[p.name for p in candidates]}"
        )

    rendered: List[RenderedSample] = []
    source_counts: Dict[str, int] = {"long": 0, "synthetic": 0, "wiki": 0}
    source_token_counts: Dict[str, int] = {"long": 0, "synthetic": 0, "wiki": 0}
    with source_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            msgs = row.get("messages")
            if not isinstance(msgs, list):
                msgs = normalize_messages(row)
            if not msgs:
                continue
            source_raw = str(
                row.get("source_prior", row.get("source", row.get("source_name", "mixed_dataset")))
            ).strip()
            source = canonicalize_source_name(source_raw)
            sample = render_sample(tokenizer, msgs, source=source)
            if sample is None:
                continue
            rendered.append(sample)
            source_counts[source] = source_counts.get(source, 0) + 1
            tok_n = len(tokenizer(sample.text, add_special_tokens=False, truncation=False)["input_ids"])
            source_token_counts[source] = source_token_counts.get(source, 0) + int(max(1, tok_n))

    if len(rendered) < 500:
        raise RuntimeError(f"Too few usable samples in prebuilt mixed dataset: {len(rendered)}")

    write_rendered_preview(rendered, preview_txt=preview_txt, limit=20)

    manifest_path = mixed_dataset_dir / "mix_manifest.json"
    prebuilt_manifest = load_json(manifest_path) if manifest_path.exists() else {}

    total_selected_tokens = int(sum(int(v) for v in source_token_counts.values()))
    actual_long_token_ratio = float(source_token_counts["long"]) / float(max(1, total_selected_tokens))
    actual_wiki_token_ratio = float(source_token_counts["wiki"]) / float(max(1, total_selected_tokens))
    actual_synthetic_token_ratio = float(source_token_counts["synthetic"]) / float(max(1, total_selected_tokens))
    target_long_ratio = lookup_manifest_ratio(
        prebuilt_manifest,
        ["target_long_ratio", "long_token_ratio", "long_ratio", "mix_long_ratio"],
    )
    target_wiki_ratio = lookup_manifest_ratio(
        prebuilt_manifest,
        ["target_wiki_ratio", "wiki_token_ratio", "wiki_ratio", "mix_wiki_ratio"],
    )
    target_synthetic_ratio = lookup_manifest_ratio(
        prebuilt_manifest,
        ["target_synthetic_ratio", "synthetic_token_ratio", "synthetic_ratio", "mix_synthetic_ratio"],
    )
    if actual_wiki_token_ratio < 0.05:
        raise RuntimeError(
            f"Prebuilt mixed dataset wiki token ratio too low: {actual_wiki_token_ratio:.4f} (required >= 0.05)."
        )
    if actual_synthetic_token_ratio < 0.10:
        raise RuntimeError(
            f"Prebuilt mixed dataset synthetic token ratio too low: {actual_synthetic_token_ratio:.4f} (required >= 0.10)."
        )
    ratio_tolerance = 0.08
    if target_long_ratio is not None and abs(actual_long_token_ratio - float(target_long_ratio)) > ratio_tolerance:
        raise RuntimeError(
            f"Prebuilt mixed dataset long token ratio drift too large: actual={actual_long_token_ratio:.4f}, "
            f"target={float(target_long_ratio):.4f}, tol={ratio_tolerance:.4f}."
        )
    if target_wiki_ratio is not None and abs(actual_wiki_token_ratio - float(target_wiki_ratio)) > ratio_tolerance:
        raise RuntimeError(
            f"Prebuilt mixed dataset wiki token ratio drift too large: actual={actual_wiki_token_ratio:.4f}, "
            f"target={float(target_wiki_ratio):.4f}, tol={ratio_tolerance:.4f}."
        )
    if target_synthetic_ratio is not None and abs(actual_synthetic_token_ratio - float(target_synthetic_ratio)) > ratio_tolerance:
        raise RuntimeError(
            f"Prebuilt mixed dataset synthetic token ratio drift too large: actual={actual_synthetic_token_ratio:.4f}, "
            f"target={float(target_synthetic_ratio):.4f}, tol={ratio_tolerance:.4f}."
        )

    stats = {
        "using_prebuilt_mixed_dataset": True,
        "mixed_dataset_dir": mixed_dataset_dir.as_posix(),
        "mixed_dataset_split": split,
        "mixed_source_jsonl_path": source_jsonl.as_posix(),
        "rendered_total": len(rendered),
        "selected_by_source": source_counts,
        "selected_token_by_source": source_token_counts,
        "actual_long_token_ratio": actual_long_token_ratio,
        "actual_wiki_token_ratio": actual_wiki_token_ratio,
        "actual_synthetic_token_ratio": actual_synthetic_token_ratio,
        "target_long_ratio": target_long_ratio,
        "target_wiki_ratio": target_wiki_ratio,
        "target_synthetic_ratio": target_synthetic_ratio,
        "prebuilt_manifest_json": manifest_path.as_posix() if manifest_path.exists() else "",
        "prebuilt_manifest_targets": prebuilt_manifest.get("targets", {}),
    }
    return rendered, stats, source_jsonl


def tokenize_response_only(
    tokenizer: AutoTokenizer,
    samples: List[RenderedSample],
    max_seq_len: int,
    min_supervised_tokens: int,
    require_offset_boundary: bool,
) -> Tuple[Dataset, Dict[str, object], List[Dict[str, object]]]:
    input_ids_all: List[List[int]] = []
    attention_mask_all: List[List[int]] = []
    labels_all: List[List[int]] = []
    total_tokens_list: List[int] = []
    assistant_tokens_list: List[int] = []
    seg_preview: List[Dict[str, object]] = []
    dropped_no_assistant = 0
    dropped_low_supervised = 0
    head_tail_truncations = 0
    boundary_dropped_middle = 0
    boundary_offset_count = 0
    boundary_prefix_count = 0
    mask_check_printed = 0

    for i, s in enumerate(samples):
        full_offsets = None
        try:
            full_enc = tokenizer(
                s.text,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            full_ids = list(full_enc["input_ids"])
            raw_offsets = full_enc.get("offset_mapping")
            if (
                isinstance(raw_offsets, list)
                and len(raw_offsets) == len(full_ids)
                and len(raw_offsets) > 0
                and isinstance(raw_offsets[0], (list, tuple))
                and len(raw_offsets[0]) == 2
            ):
                full_offsets = raw_offsets
        except Exception:
            full_ids = tokenizer(s.text, add_special_tokens=False, truncation=False)["input_ids"]

        pref_ids = tokenizer(s.prefix_text, add_special_tokens=False, truncation=False)["input_ids"]
        if not full_ids or not pref_ids:
            dropped_no_assistant += 1
            continue

        assistant_start_full = int(len(pref_ids))
        boundary_mode = "prefix_len"
        if full_offsets is not None:
            prefix_char_len = len(s.prefix_text)
            assistant_start_full = int(
                next((idx for idx, pair in enumerate(full_offsets) if int(pair[0]) >= prefix_char_len), len(full_ids))
            )
            boundary_mode = "offset_mapping"
        elif bool(require_offset_boundary):
            raise RuntimeError(
                "Tokenizer did not provide offset_mapping; cannot guarantee strict response-only masking safety."
            )

        if len(full_ids) > max_seq_len:
            head_len = min(int(TRAIN_TRUNCATE_HEAD_CAP), max(0, int(max_seq_len) - 1))
            tail_len = max(0, int(max_seq_len) - head_len)
            middle_start = int(head_len)
            middle_end = int(len(full_ids) - tail_len)
            if tail_len > 0:
                truncated = list(full_ids[:head_len] + full_ids[-tail_len:])
            else:
                truncated = list(full_ids[:head_len])
            if assistant_start_full < middle_start:
                assistant_start = int(assistant_start_full)
            elif assistant_start_full >= middle_end:
                assistant_start = int(head_len + (assistant_start_full - middle_end))
            else:
                # Assistant boundary is lost in dropped middle chunk; reject this sample later.
                assistant_start = int(len(truncated))
                boundary_dropped_middle += 1
            head_tail_truncations += 1
        else:
            truncated = list(full_ids)
            assistant_start = int(assistant_start_full)

        assistant_start = max(0, int(assistant_start))
        if boundary_mode == "offset_mapping":
            boundary_offset_count += 1
        else:
            boundary_prefix_count += 1

        if assistant_start >= len(truncated):
            # Assistant part is out of window or boundary parse failed; drop this sample.
            dropped_no_assistant += 1
            continue

        labels = list(truncated)
        if len(labels) != len(truncated):
            raise RuntimeError("Label length mismatch: labels and input_ids must have identical length.")
        for j in range(max(0, min(assistant_start, len(labels)))):
            labels[j] = -100
        supervised = sum(1 for x in labels if int(x) != -100)
        if supervised < int(min_supervised_tokens):
            dropped_low_supervised += 1
            continue

        input_ids_all.append(truncated)
        attention_mask_all.append([1] * len(truncated))
        labels_all.append(labels)
        total_tokens_list.append(len(truncated))
        assistant_tokens_list.append(supervised)

        if mask_check_printed < 2:
            supervised_ids = [int(tok) for tok, lbl in zip(truncated, labels) if int(lbl) != -100]
            decoded = tokenizer.decode(supervised_ids, skip_special_tokens=False) if supervised_ids else ""
            decoded = decoded.replace("\n", "\\n")
            print(
                f"[MASK CHECK] sample={i} source={s.source} boundary={boundary_mode} "
                f"assistant_start={assistant_start} supervised_tokens={len(supervised_ids)}",
                flush=True,
            )
            print(f"[MASK CHECK] decoded={decoded[:200]}", flush=True)
            mask_check_printed += 1

        if len(seg_preview) < 32:
            seg_preview.append(
                {
                    "index": i,
                    "source": s.source,
                    "total_tokens": len(truncated),
                    "assistant_tokens": supervised,
                    "assistant_start": assistant_start,
                    "boundary_mode": boundary_mode,
                    "user_preview": next((m["content"] for m in s.messages if m["role"] == "user"), "")[:300],
                    "assistant_preview": str(s.messages[-1].get("content", ""))[:200],
                }
            )

    if not input_ids_all:
        raise RuntimeError("No valid training samples after response-only tokenization.")

    ds = Dataset.from_dict(
        {
            "input_ids": input_ids_all,
            "attention_mask": attention_mask_all,
            "labels": labels_all,
        }
    )

    stats = {
        "num_samples_after_tokenize": len(input_ids_all),
        "num_dropped_no_assistant": int(dropped_no_assistant),
        "num_dropped_low_supervised": int(dropped_low_supervised),
        "num_head_tail_truncations": int(head_tail_truncations),
        "num_boundary_in_dropped_middle": int(boundary_dropped_middle),
        "num_boundary_offset_mapping": int(boundary_offset_count),
        "num_boundary_prefix_len": int(boundary_prefix_count),
        "require_offset_boundary": bool(require_offset_boundary),
        "total_tokens": quantiles(total_tokens_list),
        "assistant_tokens": quantiles(assistant_tokens_list),
        "assistant_tokens_lt64_ratio": float(sum(1 for x in assistant_tokens_list if x < 64)) / float(max(1, len(assistant_tokens_list))),
    }
    return ds, stats, seg_preview


def split_dataset(ds: Dataset, seed: int) -> Tuple[Dataset, Dataset]:
    ds = ds.shuffle(seed=seed)
    n = len(ds)
    n_val = max(32, int(n * 0.01))
    n_train = max(1, n - n_val)
    return ds.select(range(0, n_train)), ds.select(range(n_train, n))


def find_rotary_modules_with_inv_freq(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    out: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq") and isinstance(getattr(module, "inv_freq"), torch.Tensor):
            out.append((name, module))
    return out


def build_anchored_inv_freq(head_dim: int, base: float, anchor_factor: float, slope_raw: float, center_ratio: float) -> torch.Tensor:
    n = head_dim // 2
    idx = torch.arange(n, dtype=torch.float32)
    base_inv = 1.0 / (float(base) ** (idx / float(n)))
    slope = float(slope_raw) / float(head_dim)
    center = float(center_ratio) * float(n)
    sig = torch.sigmoid(slope * (idx - center))
    return base_inv / (1.0 + (float(anchor_factor) - 1.0) * sig)


def build_evq_cosh_inv_freq(head_dim: int, base: float, tau: float) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if float(base) <= 1.0:
        raise ValueError(f"rope base must be > 1.0, got {base}")
    tau = float(tau)
    if tau < 0:
        raise ValueError(f"evq_tau must be non-negative, got {tau}")
    n = head_dim // 2
    idx = torch.arange(n, dtype=torch.float64)
    u = idx / float(n)
    if tau <= 1e-8:
        # Exact geometric equivalence branch.
        phi = u
    else:
        tau_t = torch.tensor(tau, dtype=torch.float64)
        phi = 1.0 - (1.0 / tau_t) * torch.asinh((1.0 - u) * torch.sinh(tau_t))
    inv = torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi)
    if torch.any(inv <= 0):
        raise RuntimeError("EVQ-Cosh produced non-positive inv_freq entries.")
    if n > 1 and not bool(torch.all(inv[:-1] > inv[1:]).item()):
        raise RuntimeError("EVQ-Cosh inv_freq must be strictly decreasing by frequency index.")
    return inv.to(dtype=torch.float32)


def build_evq_exp_inv_freq(head_dim: int, base: float, beta: float) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    beta = float(beta)
    if beta <= 0:
        raise ValueError(f"evq_beta must be positive, got {beta}")
    n = head_dim // 2
    idx = torch.arange(n, dtype=torch.float32)
    u = idx / float(n)
    if beta <= 1e-6:
        phi = u
    else:
        phi = (torch.pow(torch.tensor(1.0 + beta, dtype=torch.float32), u) - 1.0) / beta
    return torch.pow(torch.tensor(float(base), dtype=torch.float32), -phi)


def invalidate_rotary_cache(module: torch.nn.Module) -> None:
    for attr in ("_cos_cached", "_sin_cached", "cos_cached", "sin_cached"):
        if hasattr(module, attr):
            try:
                delattr(module, attr)
            except Exception:
                try:
                    setattr(module, attr, None)
                except Exception:
                    pass
    for attr in ("max_seq_len_cached", "seq_len_cached"):
        if hasattr(module, attr):
            try:
                setattr(module, attr, 0)
            except Exception:
                pass


def inject_inv_freq_copy(
    model: torch.nn.Module,
    rope_base: float,
    rope_schedule: str,
    anchor_factor: float,
    slope_raw: float,
    center_ratio: float,
    evq_tau: float,
    evq_beta: float,
) -> Dict[str, object]:
    modules = find_rotary_modules_with_inv_freq(model)
    if not modules:
        raise RuntimeError("No rotary modules with inv_freq found for inv_freq.copy_().")
    ref = modules[0][1].inv_freq
    ref_numel = int(ref.numel())
    for name, module in modules:
        cur_numel = int(module.inv_freq.numel())
        if cur_numel != ref_numel:
            raise RuntimeError(
                f"Rotary head_dim mismatch: module={name} inv_freq_numel={cur_numel}, "
                f"expected={ref_numel}."
            )
    head_dim = int(ref.numel()) * 2
    schedule = str(rope_schedule).strip().lower()
    if schedule == "anchored_sigmoid":
        inv = build_anchored_inv_freq(
            head_dim=head_dim,
            base=float(rope_base),
            anchor_factor=float(anchor_factor),
            slope_raw=float(slope_raw),
            center_ratio=float(center_ratio),
        )
    elif schedule == "evq_cosh":
        inv = build_evq_cosh_inv_freq(
            head_dim=head_dim,
            base=float(rope_base),
            tau=float(evq_tau),
        )
    elif schedule == "evq_exp":
        inv = build_evq_exp_inv_freq(
            head_dim=head_dim,
            base=float(rope_base),
            beta=float(evq_beta),
        )
    else:
        raise ValueError(f"Unsupported rope_schedule: {rope_schedule}")

    inv = inv.to(dtype=ref.dtype)
    with torch.no_grad():
        for _, module in modules:
            module.inv_freq.copy_(inv.to(device=module.inv_freq.device, dtype=module.inv_freq.dtype))
            if hasattr(module, "original_inv_freq"):
                original_inv = getattr(module, "original_inv_freq")
                if isinstance(original_inv, torch.Tensor):
                    if int(original_inv.numel()) != int(module.inv_freq.numel()):
                        raise RuntimeError(
                            "original_inv_freq shape mismatch; cannot safely keep rotary cache coherent."
                        )
                    original_inv.copy_(inv.to(device=original_inv.device, dtype=original_inv.dtype))
        for _, module in modules:
            invalidate_rotary_cache(module)
    inv_sha = hashlib.sha256(inv.detach().cpu().numpy().tobytes()).hexdigest()
    return {
        "rope_schedule": schedule,
        "rotary_module_count": len(modules),
        "inv_freq_numel": int(inv.numel()),
        "inv_sha256": inv_sha,
        "head_dim": int(head_dim),
        "inv_freq_tensor": inv.detach().cpu(),
    }


def infer_rope_base_from_config(cfg: AutoConfig, fallback: float = 500000.0) -> float:
    theta = getattr(cfg, "rope_theta", None)
    if theta is not None:
        try:
            theta_f = float(theta)
            if theta_f > 1.0:
                return theta_f
        except Exception:
            pass

    rope_params = getattr(cfg, "rope_parameters", None)
    candidates: List[object] = []
    if isinstance(rope_params, dict):
        if "rope_theta" in rope_params:
            candidates.append(rope_params.get("rope_theta"))
        for v in rope_params.values():
            if isinstance(v, dict) and "rope_theta" in v:
                candidates.append(v.get("rope_theta"))
    for cand in candidates:
        try:
            theta_f = float(cand)
            if theta_f > 1.0:
                return theta_f
        except Exception:
            continue

    return float(fallback)


def strict_single_96gb_preflight(args: argparse.Namespace, cfg: AutoConfig) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("strict_single_96gb requires CUDA, but no CUDA device is visible.")
    dev_count = int(torch.cuda.device_count())
    if dev_count != 1:
        raise RuntimeError(
            f"strict_single_96gb requires exactly 1 visible GPU, got {dev_count}. "
            "Please set CUDA_VISIBLE_DEVICES to a single 96GB-class GPU."
        )

    props = torch.cuda.get_device_properties(0)
    gpu_name = str(props.name)
    mem_gib = float(props.total_memory) / float(1024 ** 3)
    min_mem = float(args.min_gpu_memory_gib)
    if mem_gib < min_mem:
        raise RuntimeError(
            f"strict_single_96gb expected >= {min_mem:.2f} GiB VRAM, but detected {mem_gib:.2f} GiB."
        )
    require_name_substr = str(args.require_gpu_name_substring).strip().lower()
    if require_name_substr and require_name_substr not in gpu_name.lower():
        raise RuntimeError(
            "strict_single_96gb GPU name guard failed: "
            f"required substring `{require_name_substr}`, got `{gpu_name}`."
        )

    required_version = str(args.require_transformers_version).strip()
    cur_version = str(transformers.__version__).strip()

    def _norm_ver(v: str) -> str:
        v = str(v).strip()
        if "+" in v:
            v = v.split("+", 1)[0]
        if ".post" in v:
            v = v.split(".post", 1)[0]
        if ".dev" in v:
            v = v.split(".dev", 1)[0]
        return v

    if required_version and _norm_ver(cur_version) != _norm_ver(required_version):
        raise RuntimeError(
            f"strict_single_96gb requires transformers=={required_version}, got {cur_version}."
        )

    if not bool(args.load_in_4bit):
        raise RuntimeError("strict_single_96gb requires --load_in_4bit for launch safety.")

    max_pos_cfg = int(getattr(cfg, "max_position_embeddings", 8192) or 8192)
    if int(args.max_seq_len) > max_pos_cfg:
        raise RuntimeError(
            f"max_seq_len={int(args.max_seq_len)} exceeds model max_position_embeddings={max_pos_cfg}."
        )

    train_tokens_per_microbatch = int(args.per_device_train_batch_size) * int(args.max_seq_len)
    # 96GB-class GPU with 4-bit quant can handle batch=4 * seq=8192 = 32768 tokens.
    # Original limit of 16384 was too conservative and blocked the default config.
    max_train_tokens = 49152
    if train_tokens_per_microbatch > max_train_tokens:
        raise RuntimeError(
            "strict_single_96gb token budget exceeded: "
            f"per_device_train_batch_size*max_seq_len={train_tokens_per_microbatch} > {max_train_tokens}."
        )

    eval_bs = max(1, int(args.eval_batch_size))
    max_batch_tokens = max(1, int(args.max_batch_input_tokens))
    max_input_eval = max(1, int(args.max_input_tokens_eval))
    eval_cap_by_tokens = max(1, max_batch_tokens // max_input_eval)
    effective_eval_bs = min(eval_bs, eval_cap_by_tokens)
    if effective_eval_bs > 8:
        raise RuntimeError(
            "strict_single_96gb eval batch risk too high: "
            f"effective_eval_bs={effective_eval_bs} (>8). "
            "Lower --eval_batch_size or --max_batch_input_tokens."
        )

    return {
        "gpu_name": gpu_name,
        "gpu_memory_gib": float(mem_gib),
        "transformers_version": cur_version,
        "train_tokens_per_microbatch": int(train_tokens_per_microbatch),
        "effective_eval_batch": int(effective_eval_bs),
        "eval_cap_by_tokens": int(eval_cap_by_tokens),
    }


def missing_reusable_train_artifacts(train_dir: Path) -> List[str]:
    adapter_dir = train_dir / "adapter"
    inv_path = train_dir / "artifacts" / "custom_inv_freq.pt"
    run_cfg_path = train_dir / "run_config.json"

    missing: List[str] = []
    if not adapter_dir.exists():
        missing.append(adapter_dir.as_posix())
    else:
        adapter_cfg = adapter_dir / "adapter_config.json"
        adapter_weight_safe = adapter_dir / "adapter_model.safetensors"
        adapter_weight_bin = adapter_dir / "adapter_model.bin"
        if not adapter_cfg.exists():
            missing.append(adapter_cfg.as_posix())
        if not (adapter_weight_safe.exists() or adapter_weight_bin.exists()):
            missing.append(f"{adapter_weight_safe.as_posix()} | {adapter_weight_bin.as_posix()}")

    for p in (inv_path, run_cfg_path):
        if not p.exists():
            missing.append(p.as_posix())
    return missing


def write_frozen_protocol(
    path: Path,
    base_model_path: str,
    known_good_path: Path,
    known_good_cfg: Optional[Dict],
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> None:
    cfg_path = Path(base_model_path) / "config.json"
    cfg_sha = sha256_file(cfg_path) if cfg_path.exists() else ""
    chat_tpl = str(getattr(tokenizer, "chat_template", "") or "")
    lines = [
        "# frozen_protocol (longinst_v1)",
        "",
        f"- timestamp: {now()}",
        f"- base_model_path: `{base_model_path}`",
        f"- base_model_config_sha256: `{cfg_sha}`",
        f"- tokenizer_chat_template_sha256: `{sha256_text(chat_tpl)}`",
        f"- known_good_run_config: `{known_good_path.as_posix()}`",
        f"- known_good_available: `{bool(known_good_cfg is not None)}`",
        "",
        "## Training lock",
        f"- max_steps: `{args.max_steps}`",
        f"- max_seq_len: `{args.max_seq_len}`",
        f"- per_device_train_batch_size: `{args.per_device_train_batch_size}`",
        f"- gradient_accumulation_steps: `{args.gradient_accumulation_steps}`",
        f"- effective_batch: `{args.per_device_train_batch_size * args.gradient_accumulation_steps}`",
        f"- learning_rate: `{args.learning_rate}`",
        f"- warmup_steps: `{args.warmup_steps}`",
        f"- lr_scheduler_type: `{args.lr_scheduler_type}`",
        f"- optimizer: `{args.optim}`",
        f"- attn_implementation: `{args.attn_implementation}`",
        f"- lora_rank: `{args.lora_rank}`",
        f"- lora_alpha: `{args.lora_alpha}`",
        f"- lora_target_modules: `{args.lora_target_modules}`",
        f"- response_only_loss: `true`",
        f"- tokenizer_truncation_side: `{str(tokenizer.truncation_side)}`",
        f"- training_truncate_mode: `{TRAIN_TRUNCATE_MODE}`",
        f"- training_window_policy: `if len>max_seq_len keep first min({TRAIN_TRUNCATE_HEAD_CAP}, max_seq_len-1) tokens and last remainder; drop middle`",
        f"- packing_strategy: `no-packing (sample-level)`",
        "",
        "## RoPE injection lock",
        "- injection_path: `inv_freq.copy_()`",
        "- no_hf_rope_scaling: `true`",
        f"- rope_schedule: `{args.rope_schedule}`",
        f"- rope_base: `{args.rope_base}` (0 means infer from model config)",
        f"- anchor_factor: `{args.anchor_factor}`",
        f"- slope_raw: `{args.slope_raw}`",
        f"- center_ratio: `{args.center_ratio}`",
        f"- evq_tau: `{args.evq_tau}`",
        f"- evq_beta: `{args.evq_beta}`",
        "",
        "## Eval decode lock",
        "- prompt_source: `official`",
        "- chat_template: `auto`",
        "- truncate_mode: `middle`",
        "- max_new_tokens_policy: `official`",
        "- do_sample: `false`",
        "- score_scale: `pct`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd.as_posix(), check=True)


def load_morning_task_score(path: Path, task: str, model_key: str = "anchored_sigmoid") -> Optional[float]:
    if not path.exists():
        return None
    raw = load_json(path)
    model = raw.get("models", {}).get(model_key, {})
    task_obj = model.get("tasks", {}).get(task, {})
    score = task_obj.get("score_pct", None)
    if score is None:
        return None
    try:
        return float(score)
    except Exception:
        return None


def parse_compare_scores(compare_json: Path) -> Dict[str, Dict[str, float]]:
    raw = load_json(compare_json)
    out: Dict[str, Dict[str, float]] = {}
    comp = raw.get("comparison", {})
    if isinstance(comp, dict) and comp:
        for task, info in comp.items():
            if not isinstance(info, dict):
                continue
            base = float(info.get("base_unfinetuned", 0.0))
            lora = float(info.get("hybrid_lora", 0.0))
            delta = float(info.get("delta_alt_minus_base", lora - base))
            out[str(task)] = {"base": base, "lora": lora, "delta": delta}
        return out
    gate_scores = raw.get("gate_scores", {})
    if isinstance(gate_scores, dict):
        for task, info in gate_scores.items():
            if not isinstance(info, dict):
                continue
            base = float(info.get("base", 0.0))
            lora = float(info.get("lora", 0.0))
            delta = float(info.get("delta", lora - base))
            out[str(task)] = {"base": base, "lora": lora, "delta": delta}
    return out


def make_length_buckets(compare_json: Path) -> Dict[str, Dict[str, float]]:
    raw = load_json(compare_json)
    model_tasks = raw.get("models", {}).get("hybrid_lora", {}).get("tasks", {})
    buckets: Dict[str, List[float]] = {"<=2k": [], "2-4k": [], "4-6k": [], "6-8k": []}
    for _, task_obj in model_tasks.items():
        traces = task_obj.get("per_sample_traces", [])
        for tr in traces:
            toks = int(tr.get("input_tokens_after_trunc", tr.get("input_tokens", 0)) or 0)
            score = float(tr.get("score_pct", float(tr.get("score_raw", 0.0)) * 100.0))
            if toks <= 2000:
                buckets["<=2k"].append(score)
            elif toks <= 4000:
                buckets["2-4k"].append(score)
            elif toks <= 6000:
                buckets["4-6k"].append(score)
            elif toks <= 8000:
                buckets["6-8k"].append(score)
    out: Dict[str, Dict[str, float]] = {}
    for k, vals in buckets.items():
        out[k] = {
            "count": int(len(vals)),
            "mean_pct": float(statistics.mean(vals)) if vals else 0.0,
        }
    return out


def load_jsonl_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def build_triparty_summary(
    gate_json: Path,
    qwen_seed42_json: Path,
    qwen_seed1337_json: Path,
    output_json: Path,
    output_md: Path,
) -> Dict[str, object]:
    gate_obj = load_json(gate_json)
    gate_scores = gate_obj.get("gate_scores", {}) if isinstance(gate_obj, dict) else {}
    q42 = load_json(qwen_seed42_json) if qwen_seed42_json.exists() else {}
    q1337 = load_json(qwen_seed1337_json) if qwen_seed1337_json.exists() else {}

    def _qwen_score(raw: Dict, task: str) -> Optional[float]:
        try:
            val = raw.get("models", {}).get("anchored_sigmoid", {}).get("tasks", {}).get(task, {}).get("score")
            return None if val is None else float(val)
        except Exception:
            return None

    out: Dict[str, object] = {
        "meta": {
            "timestamp": now(),
            "gate_json": gate_json.as_posix(),
            "qwen_seed42_json": qwen_seed42_json.as_posix(),
            "qwen_seed1337_json": qwen_seed1337_json.as_posix(),
            "score_unit": "pct_0_100",
        },
        "tasks": {},
    }
    rows = []
    for task in ["qasper", "musique"]:
        g = gate_scores.get(task, {}) if isinstance(gate_scores, dict) else {}
        base = float(g.get("base", 0.0)) if isinstance(g, dict) else 0.0
        lora = float(g.get("lora", 0.0)) if isinstance(g, dict) else 0.0
        q42s = _qwen_score(q42, task)
        q1337s = _qwen_score(q1337, task)
        qmean = ((q42s + q1337s) / 2.0) if (q42s is not None and q1337s is not None) else None
        obj = {
            "llama_base": base,
            "llama_lora": lora,
            "delta_lora_minus_base": lora - base,
            "qwen_seed42": q42s,
            "qwen_seed1337": q1337s,
            "qwen_mean": qmean,
            "delta_lora_minus_qwen_mean": (lora - qmean) if qmean is not None else None,
        }
        out["tasks"][task] = obj
        rows.append((task, obj))

    save_json(output_json, out)
    md_lines = [
        "# Triparty Summary (Gate)",
        "",
        f"- gate_json: `{gate_json.as_posix()}`",
        f"- qwen_seed42_json: `{qwen_seed42_json.as_posix()}`",
        f"- qwen_seed1337_json: `{qwen_seed1337_json.as_posix()}`",
        "",
        "| task | llama_base | llama_lora | delta_lora-base | qwen_s42 | qwen_s1337 | qwen_mean | delta_lora-qwen_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, obj in rows:
        def fmt(v: Optional[float]) -> str:
            return "NA" if v is None else f"{float(v):.4f}"
        md_lines.append(
            f"| {task} | {fmt(obj['llama_base'])} | {fmt(obj['llama_lora'])} | {fmt(obj['delta_lora_minus_base'])} | "
            f"{fmt(obj['qwen_seed42'])} | {fmt(obj['qwen_seed1337'])} | {fmt(obj['qwen_mean'])} | {fmt(obj['delta_lora_minus_qwen_mean'])} |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return out


def build_loss_slope_report(train_log_jsonl: Path, out_json: Path) -> Dict[str, object]:
    rows = load_jsonl_rows(train_log_jsonl)
    loss_by_step: Dict[int, float] = {}
    for r in rows:
        if "loss" in r and "step" in r:
            try:
                loss_by_step[int(r["step"])] = float(r["loss"])
            except Exception:
                continue

    def nearest_step(target: int) -> Optional[int]:
        if not loss_by_step:
            return None
        keys = sorted(loss_by_step.keys())
        best = min(keys, key=lambda k: abs(k - target))
        return int(best)

    s600 = nearest_step(600)
    s800 = nearest_step(800)
    l600 = loss_by_step.get(s600) if s600 is not None else None
    l800 = loss_by_step.get(s800) if s800 is not None else None
    drop_pct = None
    recommendation = "insufficient_data"
    if l600 is not None and l800 is not None and l600 > 0:
        drop_pct = float((l600 - l800) / l600 * 100.0)
        if drop_pct < 1.0:
            recommendation = "600_steps_can_replace_800"
        elif drop_pct >= 3.0:
            recommendation = "keep_800_steps"
        else:
            recommendation = "both_600_800_need_gate_validation"
    out = {
        "timestamp": now(),
        "train_log_jsonl": train_log_jsonl.as_posix(),
        "step_600_observed": s600,
        "step_800_observed": s800,
        "loss_600": l600,
        "loss_800": l800,
        "drop_pct_600_to_800": drop_pct,
        "recommendation": recommendation,
    }
    save_json(out_json, out)
    return out


def run_gate_and_optional_full(args: argparse.Namespace, repo_root: Path, run_dir: Path, data_stats: Dict[str, object], seg_preview: List[Dict[str, object]]) -> None:
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    gate_raw_json = eval_dir / "qasper_musique_compare_raw.json"
    gate_json = eval_dir / "qasper_musique_compare.json"

    adapter_path = run_dir / "adapter"
    inv_path = run_dir / "artifacts" / "custom_inv_freq.pt"
    missing = missing_reusable_train_artifacts(run_dir)
    if missing:
        raise RuntimeError(
            "Cannot run evaluation: missing required trained artifacts:\n- " + "\n- ".join(missing)
        )

    gate_cmd = [
        sys.executable,
        "scripts/eval_longbench.py",
        "--base_model_path",
        args.base_model_path,
        "--hybrid_adapter_path",
        adapter_path.as_posix(),
        "--custom_inv_freq_path",
        inv_path.as_posix(),
        "--task_set",
        "lb6",
        "--tasks",
        "qasper,musique",
        "--max_samples_per_task",
        "0",
        "--max_input_tokens",
        str(args.max_input_tokens_eval),
        "--batch_size",
        str(args.eval_batch_size),
        "--max_batch_input_tokens",
        str(args.max_batch_input_tokens),
        "--prompt_source",
        "official",
        "--chat_template",
        "auto",
        "--truncate_mode",
        "middle",
        "--max_new_tokens_policy",
        "official",
        "--score_scale",
        "pct",
        "--seed",
        str(args.seed),
        "--longbench_local_data_dir",
        args.longbench_local_data_dir,
        "--save_per_sample_traces",
        "1",
        "--trace_output_max_chars",
        "1024",
        "--strict_parity_check",
        "--do_sample",
        "false",
        "--output_json",
        gate_raw_json.as_posix(),
    ]
    run_cmd(gate_cmd, cwd=repo_root)

    gate = parse_compare_scores(gate_raw_json)
    morning_ref = load_morning_task_score(Path(args.morning_reference_json), task=args.morning_compare_task)
    compact = {
        "timestamp": now(),
        "run_name": args.run_name,
        "source_json": gate_raw_json.as_posix(),
        "gate_scores": gate,
        "morning_reference": {
            "path": args.morning_reference_json,
            "task": args.morning_compare_task,
            "score_pct": morning_ref,
            "delta_vs_morning": (gate.get(args.morning_compare_task, {}).get("lora", 0.0) - morning_ref)
            if morning_ref is not None
            else None,
        },
    }
    save_json(gate_json, compact)
    build_triparty_summary(
        gate_json=gate_json,
        qwen_seed42_json=Path(args.qwen_seed42_json),
        qwen_seed1337_json=Path(args.qwen_seed1337_json),
        output_json=eval_dir / "triparty_summary.json",
        output_md=eval_dir / "triparty_summary.md",
    )

    qasper = gate.get("qasper")
    musique = gate.get("musique")
    gate_ready = isinstance(qasper, dict) and isinstance(musique, dict)
    if gate_ready:
        q_base = qasper.get("base")
        q_lora = qasper.get("lora")
        m_base = musique.get("base")
        m_lora = musique.get("lora")
        gate_ready = all(isinstance(x, (int, float)) for x in [q_base, q_lora, m_base, m_lora])
    else:
        q_base = q_lora = m_base = m_lora = None

    qasper_pass = bool(gate_ready and float(q_lora) >= float(q_base))
    musique_pass = bool(gate_ready and float(m_lora) >= float(m_base) - 1.0)
    if not gate_ready or not (qasper_pass and musique_pass):
        diag = {
            "timestamp": now(),
            "stop_reason": "gate_failed",
            "gate": compact,
            "gate_rule": {
                "qasper_rule": "lora >= base",
                "musique_rule": "lora >= base - 1.0",
                "gate_ready": bool(gate_ready),
                "qasper_pass": bool(qasper_pass),
                "musique_pass": bool(musique_pass),
            },
            "assistant_token_stats": data_stats,
            "segmentation_preview_10": seg_preview[:10],
        }
        save_json(eval_dir / "gate_diagnostics.json", diag)
        print("[STOP] Gate failed; wrote gate_diagnostics.json", flush=True)
        return

    if not bool(args.run_full_eval):
        print("[DONE] Gate passed; run_full_eval disabled.", flush=True)
        return

    full_raw_json = eval_dir / "longbench_full_compare_raw.json"
    full_json = eval_dir / "longbench_full_compare.json"
    full_cmd = [
        sys.executable,
        "scripts/eval_longbench.py",
        "--base_model_path",
        args.base_model_path,
        "--hybrid_adapter_path",
        adapter_path.as_posix(),
        "--custom_inv_freq_path",
        inv_path.as_posix(),
        "--task_set",
        "lb21",
        "--max_samples_per_task",
        "0",
        "--max_input_tokens",
        str(args.max_input_tokens_eval),
        "--batch_size",
        str(args.eval_batch_size),
        "--max_batch_input_tokens",
        str(args.max_batch_input_tokens),
        "--prompt_source",
        "official",
        "--chat_template",
        "auto",
        "--truncate_mode",
        "middle",
        "--max_new_tokens_policy",
        "official",
        "--score_scale",
        "pct",
        "--seed",
        str(args.seed),
        "--longbench_local_data_dir",
        args.longbench_local_data_dir,
        "--save_per_sample_traces",
        "1",
        "--trace_output_max_chars",
        "1024",
        "--strict_parity_check",
        "--do_sample",
        "false",
        "--output_json",
        full_raw_json.as_posix(),
    ]
    run_cmd(full_cmd, cwd=repo_root)

    raw_full = load_json(full_raw_json)
    comp = parse_compare_scores(full_raw_json)
    macro_base = statistics.mean([v["base"] for v in comp.values()]) if comp else 0.0
    macro_lora = statistics.mean([v["lora"] for v in comp.values()]) if comp else 0.0
    summary = {
        "timestamp": now(),
        "macro_base_pct": float(macro_base),
        "macro_lora_pct": float(macro_lora),
        "macro_delta_pct": float(macro_lora - macro_base),
        "per_task": comp,
        "length_buckets_lora": make_length_buckets(full_raw_json),
        "source_json": full_raw_json.as_posix(),
        "source_json_full_object": raw_full,
    }
    save_json(full_json, summary)
    print("[DONE] Full lb21 evaluation completed.", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Long-instruction response-only LoRA v1 pipeline")
    ap.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL)
    ap.add_argument("--known_good_run_config", type=str, default=DEFAULT_KNOWN_GOOD_RUN_CONFIG)
    ap.add_argument("--output_root", type=str, default="artifacts/llama8k_theory_v1")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--longalpaca_path", type=str, default=DEFAULT_LONGALPACA)
    ap.add_argument("--longqa_path", type=str, default=DEFAULT_LONGQA)
    ap.add_argument("--wikitext_train_path", type=str, default=DEFAULT_WIKITEXT)
    ap.add_argument("--mixed_dataset_dir", type=str, default="")
    ap.add_argument("--mixed_dataset_split", type=str, default="train")
    ap.add_argument("--max_long_records", type=int, default=12000)
    ap.add_argument("--max_wiki_samples", type=int, default=12000)
    ap.add_argument("--mix_long_ratio", type=float, default=0.7)
    ap.add_argument("--mix_wiki_ratio", type=float, default=0.1)
    ap.add_argument("--synthetic_ratio", type=float, default=0.20)
    ap.add_argument("--allow_continuation_dominant_corpus", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--min_supervised_tokens", type=int, default=32)
    ap.add_argument("--require_offset_boundary", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--debug_mix", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--max_seq_len", type=int, default=8192)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--optim", type=str, default="paged_adamw_8bit")
    ap.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    ap.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

    ap.add_argument("--rope_base", type=float, default=0.0)
    ap.add_argument("--rope_schedule", type=str, default="evq_cosh", choices=["anchored_sigmoid", "evq_cosh", "evq_exp"])
    ap.add_argument("--anchor_factor", type=float, default=4.0)
    ap.add_argument("--slope_raw", type=float, default=20.0)
    ap.add_argument("--center_ratio", type=float, default=0.70)
    ap.add_argument("--evq_tau", type=float, default=1.5)
    ap.add_argument("--evq_beta", type=float, default=3.0)

    ap.add_argument("--morning_reference_json", type=str, default=DEFAULT_MORNING_REF)
    ap.add_argument("--morning_compare_task", type=str, default="qasper")

    ap.add_argument("--longbench_local_data_dir", type=str, default="/root/autodl-tmp/dfrope/ms_datasets/LongBench/data")
    ap.add_argument("--qwen_seed42_json", type=str, default="artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json")
    ap.add_argument("--qwen_seed1337_json", type=str, default="artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/anchored_sigmoid.json")
    ap.add_argument("--max_input_tokens_eval", type=int, default=8192)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--max_batch_input_tokens", type=int, default=98304)
    ap.add_argument("--strict_single_96gb", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--strict_single_h800_96gb", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--min_gpu_memory_gib", type=float, default=90.0)
    ap.add_argument("--require_gpu_name_substring", type=str, default="")
    ap.add_argument("--require_transformers_version", type=str, default="")
    ap.add_argument("--run_full_eval", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--skip_training", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--data_only", action=argparse.BooleanOptionalAction, default=False)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[CONFIG] rope_schedule={args.rope_schedule}, evq_tau={args.evq_tau}", flush=True)
    if str(args.rope_schedule).strip().lower() == "anchored_sigmoid":
        print(
            "[WARNING] Using anchored_sigmoid schedule. "
            "For EVQ runs use --rope_schedule evq_cosh with explicit --evq_tau.",
            flush=True,
        )
    if str(args.rope_schedule).strip().lower() == "evq_cosh" and float(args.evq_tau) == 0.0:
        print("[INFO] tau=0.0 detected -> running as pure geometric RoPE", flush=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    repo_root = Path(__file__).resolve().parents[3]
    output_root = Path(args.output_root)
    run_name = args.run_name.strip() or f"lora_longinst_s{args.seed}"
    args.run_name = run_name
    data_dir = output_root / "data" / run_name
    train_dir = output_root / "train" / run_name
    frozen_protocol = output_root / "frozen_protocol.md"

    model_key = str(args.base_model_path).lower()
    if "llama-3.1" in model_key:
        raise RuntimeError("base_model_path must be Meta-Llama-3-8B-Instruct (8K native), not Llama-3.1.")
    if "meta-llama-3-8b-instruct" not in model_key:
        raise RuntimeError("base_model_path must be Meta-Llama-3-8B-Instruct.")

    cfg_guard = AutoConfig.from_pretrained(args.base_model_path, trust_remote_code=True, local_files_only=True)
    max_pos = int(getattr(cfg_guard, "max_position_embeddings", 0) or 0)
    if max_pos not in {8192, 0}:
        raise RuntimeError(
            f"Unexpected max_position_embeddings={max_pos}. Expected 8192 for LLaMA-3-8B-Instruct protocol."
        )
    rope_scaling_cfg = getattr(cfg_guard, "rope_scaling", None)
    if rope_scaling_cfg is not None and rope_scaling_cfg != {}:
        raise RuntimeError(
            f"Unsupported rope_scaling={rope_scaling_cfg}. "
            "This pipeline requires base LLaMA-3-8B-Instruct without dynamic rope_scaling."
        )
    strict_guard_enabled = bool(args.strict_single_96gb) or bool(args.strict_single_h800_96gb)
    if bool(args.strict_single_h800_96gb):
        print("[DEPRECATED] --strict_single_h800_96gb is deprecated. Use --strict_single_96gb.", flush=True)
    if strict_guard_enabled:
        hw_info = strict_single_96gb_preflight(args=args, cfg=cfg_guard)
        print(
            "[HARDWARE CHECK] strict_single_96gb passed: "
            f"gpu={hw_info['gpu_name']}, vram={float(hw_info['gpu_memory_gib']):.2f}GiB, "
            f"transformers={hw_info['transformers_version']}, "
            f"train_tokens/microbatch={int(hw_info['train_tokens_per_microbatch'])}, "
            f"effective_eval_batch={int(hw_info['effective_eval_batch'])}",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, local_files_only=True)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    known_path = Path(args.known_good_run_config)
    known_cfg = load_json(known_path) if known_path.exists() else None
    write_frozen_protocol(
        path=frozen_protocol,
        base_model_path=args.base_model_path,
        known_good_path=known_path,
        known_good_cfg=known_cfg,
        tokenizer=tokenizer,
        args=args,
    )

    long_paths = [Path(args.longalpaca_path)]
    if str(args.longqa_path).strip():
        long_paths.append(Path(args.longqa_path))

    preview_txt = data_dir / "preview_20.txt"
    mixed_dataset_dir = Path(str(args.mixed_dataset_dir)).expanduser() if str(args.mixed_dataset_dir).strip() else None
    if mixed_dataset_dir is not None:
        rendered, mix_stats, mix_jsonl = load_prebuilt_mixed_split(
            tokenizer=tokenizer,
            mixed_dataset_dir=mixed_dataset_dir,
            split=str(args.mixed_dataset_split),
            preview_txt=preview_txt,
        )
    else:
        mix_jsonl = data_dir / "longinst_mix.jsonl"
        rendered, mix_stats = build_longinst_mix(
            tokenizer=tokenizer,
            long_paths=long_paths,
            wikitext_path=Path(args.wikitext_train_path),
            max_long_records=int(args.max_long_records),
            max_wiki_samples=int(args.max_wiki_samples),
            mix_long_ratio=float(args.mix_long_ratio),
            mix_wiki_ratio=float(args.mix_wiki_ratio),
            synthetic_ratio=float(args.synthetic_ratio),
            allow_continuation_dominant_corpus=bool(args.allow_continuation_dominant_corpus),
            seed=int(args.seed),
            out_jsonl=mix_jsonl,
            preview_txt=preview_txt,
        )
    if bool(args.debug_mix):
        print("[DEBUG MIX] First 5 rendered samples:", flush=True)
        for idx, sample in enumerate(rendered[:5]):
            user_preview = next((m["content"] for m in sample.messages if m.get("role") == "user"), "")
            assistant_preview = str(sample.messages[-1].get("content", ""))
            print(
                f"[DEBUG MIX][{idx}] source={sample.source} "
                f"user={user_preview[:180]!r} assistant={assistant_preview[:180]!r}",
                flush=True,
            )

    ds, token_stats, seg_preview = tokenize_response_only(
        tokenizer=tokenizer,
        samples=rendered,
        max_seq_len=int(args.max_seq_len),
        min_supervised_tokens=int(args.min_supervised_tokens),
        require_offset_boundary=bool(args.require_offset_boundary),
    )
    stats = {**mix_stats, **token_stats}
    if float(stats.get("assistant_tokens_lt64_ratio", 0.0)) > 0.10:
        raise RuntimeError(
            f"assistant_tokens_lt64_ratio too high: {stats.get('assistant_tokens_lt64_ratio')}. "
            "Stop to prevent weak supervision collapse."
        )
    save_json(data_dir / "stats.json", stats)
    seg_preview_path = data_dir / "segmentation_preview_10.json"
    save_json(seg_preview_path, {"rows": seg_preview[:10]})
    data_manifest = {
        "timestamp": now(),
        "run_name": run_name,
        "seed": int(args.seed),
        "base_model_path": args.base_model_path,
        "source_paths": {
            "longalpaca_path": args.longalpaca_path,
            "longqa_path": args.longqa_path,
            "wikitext_train_path": args.wikitext_train_path,
            "mixed_dataset_dir": args.mixed_dataset_dir,
            "mixed_dataset_split": args.mixed_dataset_split,
        },
        "mix": {
            "mix_long_ratio": float(args.mix_long_ratio),
            "mix_wiki_ratio": float(args.mix_wiki_ratio),
            "synthetic_ratio": float(args.synthetic_ratio),
            "using_prebuilt_mixed_dataset": bool(str(args.mixed_dataset_dir).strip()),
        },
        "stats": stats,
        "mix_jsonl": mix_jsonl.as_posix(),
        "segmentation_preview_10": seg_preview_path.as_posix(),
    }
    save_json(data_dir / "data_manifest.json", data_manifest)
    data_hash = sha256_file(mix_jsonl)
    (data_dir / "data_hash.sha256").write_text(data_hash + "\n", encoding="utf-8")

    if bool(args.data_only):
        print("[DATA ONLY] Dataset artifacts prepared. Training and evaluation skipped.", flush=True)
        return

    train_ds, val_ds = split_dataset(ds, seed=args.seed)

    if not bool(args.skip_training):
        train_dir.mkdir(parents=True, exist_ok=True)
        model_cfg = AutoConfig.from_pretrained(args.base_model_path, trust_remote_code=True, local_files_only=True)
        rope_base = float(args.rope_base) if float(args.rope_base) > 0 else infer_rope_base_from_config(model_cfg)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        bnb_cfg = None
        if bool(args.load_in_4bit):
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": True,
            "attn_implementation": args.attn_implementation,
        }
        if bnb_cfg is not None:
            model_kwargs["quantization_config"] = bnb_cfg
        model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **model_kwargs)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        rope_info = inject_inv_freq_copy(
            model=model,
            rope_base=rope_base,
            rope_schedule=str(args.rope_schedule),
            anchor_factor=float(args.anchor_factor),
            slope_raw=float(args.slope_raw),
            center_ratio=float(args.center_ratio),
            evq_tau=float(args.evq_tau),
            evq_beta=float(args.evq_beta),
        )
        inv_head = rope_info["inv_freq_tensor"][:8].detach().cpu().tolist()
        inv_head_str = ", ".join(f"{float(v):.6e}" for v in inv_head)
        print(f"[ROPE] inv_freq[0:8]=[{inv_head_str}]", flush=True)

        artifacts_dir = train_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        custom_inv_path = artifacts_dir / "custom_inv_freq.pt"
        torch.save(rope_info["inv_freq_tensor"], custom_inv_path)

        if bool(args.load_in_4bit):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(args.gradient_checkpointing))
        elif bool(args.gradient_checkpointing) and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(args.lora_rank),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=0.0,
            bias="none",
            target_modules=parse_csv(args.lora_target_modules),
        )
        model = get_peft_model(model, lora_cfg)

        targs = TrainingArguments(
            output_dir=str(train_dir),
            run_name=train_dir.name,
            max_steps=int(args.max_steps),
            per_device_train_batch_size=int(args.per_device_train_batch_size),
            per_device_eval_batch_size=max(1, int(args.per_device_train_batch_size)),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            learning_rate=float(args.learning_rate),
            warmup_steps=int(args.warmup_steps),
            logging_steps=int(args.logging_steps),
            save_steps=int(args.save_steps),
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            fp16=False,
            optim=args.optim,
            lr_scheduler_type=args.lr_scheduler_type,
            gradient_checkpointing=bool(args.gradient_checkpointing),
            report_to=[],
            eval_strategy="steps",
            eval_steps=int(args.save_steps),
            save_strategy="steps",
            dataloader_num_workers=2,
        )

        trainer = ResponseOnlyTrainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=CausalLMCollator(pad_token_id=int(tokenizer.pad_token_id or 0)),
            callbacks=[JsonlLogCallback(train_dir / "train_log.jsonl")],
        )
        result = trainer.train()
        trainer.save_model(str(train_dir / "adapter"))
        tokenizer.save_pretrained(str(train_dir / "adapter"))

        run_cfg = {
            "timestamp": now(),
            "script": Path(__file__).name,
            "base_model_path": args.base_model_path,
            "seed": int(args.seed),
            "training": {
                "max_steps": int(args.max_steps),
                "max_seq_len": int(args.max_seq_len),
                "per_device_train_batch_size": int(args.per_device_train_batch_size),
                "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
                "effective_batch": int(args.per_device_train_batch_size * args.gradient_accumulation_steps),
                "learning_rate": float(args.learning_rate),
                "warmup_steps": int(args.warmup_steps),
                "lr_scheduler_type": args.lr_scheduler_type,
                "optim": args.optim,
                "attn_implementation": args.attn_implementation,
            },
            "lora": {
                "rank": int(args.lora_rank),
                "alpha": int(args.lora_alpha),
                "target_modules": parse_csv(args.lora_target_modules),
            },
            "rope": {
                "rope_schedule": str(rope_info.get("rope_schedule", args.rope_schedule)),
                "rope_base": float(rope_base),
                "anchor_factor": float(args.anchor_factor),
                "slope_raw": float(args.slope_raw),
                "center_ratio": float(args.center_ratio),
                "evq_tau": float(args.evq_tau),
                "evq_beta": float(args.evq_beta),
                "inv_sha256": str(rope_info["inv_sha256"]),
                "custom_inv_freq_path": custom_inv_path.as_posix(),
                "custom_inv_freq_sha256": sha256_file(custom_inv_path),
            },
            "dataset": {
                "mix_jsonl": mix_jsonl.as_posix(),
                "data_manifest_json": (data_dir / "data_manifest.json").as_posix(),
                "data_hash_sha256": data_hash,
                "preview": preview_txt.as_posix(),
                "segmentation_preview_10": seg_preview_path.as_posix(),
                "stats": stats,
                "tokenizer_chat_template_sha256": sha256_text(str(getattr(tokenizer, "chat_template", "") or "")),
                "tokenizer_truncation_side": str(tokenizer.truncation_side),
                "training_truncate_mode": TRAIN_TRUNCATE_MODE,
                "training_truncate_head_cap_tokens": int(TRAIN_TRUNCATE_HEAD_CAP),
                "training_truncate_head_tokens": int(min(int(TRAIN_TRUNCATE_HEAD_CAP), max(0, int(args.max_seq_len) - 1))),
                "training_truncate_tail_tokens": int(
                    max(0, int(args.max_seq_len) - min(int(TRAIN_TRUNCATE_HEAD_CAP), max(0, int(args.max_seq_len) - 1)))
                ),
                # Legacy key retained for compatibility with older downstream readers.
                "truncation_side": str(tokenizer.truncation_side),
                "response_only": True,
            },
            "train_result": dict(result.metrics),
            "code_hash": sha256_file(Path(__file__)),
        }
        save_json(train_dir / "run_config.json", run_cfg)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        save_json(artifacts_dir / "data_manifest.json", data_manifest)
        (artifacts_dir / "data_hash.sha256").write_text(data_hash + "\n", encoding="utf-8")
        build_loss_slope_report(
            train_log_jsonl=train_dir / "train_log.jsonl",
            out_json=train_dir / "loss_slope_report.json",
        )
    else:
        missing = missing_reusable_train_artifacts(train_dir)
        if missing:
            raise RuntimeError(
                "--skip_training was requested but required artifacts are missing:\n- "
                + "\n- ".join(missing)
            )
        print(
            f"[SKIP TRAINING] Reusing existing artifacts under {train_dir.as_posix()}",
            flush=True,
        )

    run_gate_and_optional_full(
        args=args,
        repo_root=repo_root,
        run_dir=train_dir,
        data_stats=stats,
        seg_preview=seg_preview,
    )

    print("[DONE] longinst_v1 pipeline finished.", flush=True)


if __name__ == "__main__":
    main()
