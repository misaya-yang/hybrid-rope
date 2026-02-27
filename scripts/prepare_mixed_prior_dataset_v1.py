#!/usr/bin/env python3
"""Prepare a token-faithful mixed-prior dataset for long-context LoRA.

Theoretical mapping (distance prior D(Delta)):
- power_law_base: long-range smooth semantic prior
- bimodal_reasoning: local/mid-band multihop reasoning prior
- uniform_scaffold: short generic instruction alignment prior

Default token ratio is enforced by TOKEN budget (not row count):
- power_law_base: 50%
- bimodal_reasoning: 40%
- uniform_scaffold: 10%

Outputs (under artifacts/datasets/mixed_prior_v1_<timestamp>):
- mixed_prior_finetune.jsonl (unified shuffled corpus)
- train.jsonl / valid.jsonl / test.jsonl
- mix_manifest.json
- quality_report.md
- preview_50.txt
- label_mask_preview.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

from transformers import AutoTokenizer


DEFAULT_LONGALPACA = "/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl"
DEFAULT_LONGQA = "/root/autodl-tmp/dfrope/datasets/LongQA.jsonl"
DEFAULT_WIKITEXT = "/root/autodl-tmp/wikitext_data/train.txt"
DEFAULT_TOKENIZER = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/dfrope/hybrid-rope/artifacts/datasets"

NOISE_VOCAB = [
    "ledger",
    "delta",
    "vector",
    "audit",
    "protocol",
    "checkpoint",
    "segment",
    "context",
    "trace",
    "summary",
    "marker",
    "window",
    "anchor",
    "signal",
    "bridge",
    "kernel",
    "matrix",
    "evidence",
    "record",
    "notebook",
    "timeline",
    "registry",
    "module",
    "gateway",
    "catalog",
    "router",
    "buffer",
]
TRAIN_TRUNCATE_HEAD_CAP = 500


@dataclass
class PreparedSample:
    prior: str
    source_name: str
    lang: str
    task_type: str
    messages: List[Dict[str, str]]
    meta: Dict[str, object]
    full_tokens: int
    truncated_tokens: int
    assistant_tokens: int
    assistant_start: int
    dedup_sha256: str


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_csv_paths(text: str) -> List[Path]:
    out: List[Path] = []
    for part in str(text).split(","):
        x = part.strip()
        if x:
            out.append(Path(x))
    return out


def quantiles(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    vals = sorted(int(v) for v in values)

    def q(p: float) -> float:
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


def iter_json_records(path: Path) -> Iterator[Tuple[int, Dict]]:
    if not path.exists():
        return
    sfx = path.suffix.lower()
    if sfx == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield i, obj
        return

    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return

    if isinstance(obj, list):
        for i, row in enumerate(obj):
            if isinstance(row, dict):
                yield i, row
    elif isinstance(obj, dict):
        yield 0, obj


def normalize_messages(obj: Dict) -> List[Dict[str, str]]:
    if not isinstance(obj, dict):
        return []

    if isinstance(obj.get("messages"), list):
        out: List[Dict[str, str]] = []
        for m in obj["messages"]:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", "")).strip()
            if role in {"system", "user", "assistant"} and content:
                out.append({"role": role, "content": content})
        if out:
            return out

    ins = str(obj.get("instruction", "")).strip()
    inp_raw = obj.get("input", "")
    inp = "" if inp_raw is None else str(inp_raw).strip()
    out_text = str(obj.get("output", obj.get("response", obj.get("answer", "")))).strip()
    if (ins or inp) and out_text:
        user = ins if not inp else f"{ins}\n\n{inp}"
        return [{"role": "user", "content": user}, {"role": "assistant", "content": out_text}]

    if isinstance(obj.get("conversations"), list):
        out = []
        for c in obj["conversations"]:
            if not isinstance(c, dict):
                continue
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
    if not text:
        return []
    normalized = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?。！？])\s+", normalized)
    return [p.strip() for p in parts if p and p.strip()]


def is_top_level_heading(line: str) -> bool:
    s = line.strip()
    if not s.startswith("=") or not s.endswith("="):
        return False
    left = len(s) - len(s.lstrip("="))
    right = len(s) - len(s.rstrip("="))
    return left == 1 and right == 1 and len(s) > 4


def load_wikitext_articles(path: Path, min_chars: int = 1400) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    heads = [i for i, line in enumerate(lines) if is_top_level_heading(line)]
    if not heads:
        body = "\n".join([x for x in lines if x.strip() and not x.strip().startswith("=")])
        if len(body) >= min_chars:
            return [body]
        return []

    articles: List[str] = []
    for i, st in enumerate(heads):
        ed = heads[i + 1] if i + 1 < len(heads) else len(lines)
        chunk = "\n".join([x for x in lines[st:ed] if x.strip() and not x.strip().startswith("=")])
        if len(chunk) >= min_chars:
            articles.append(chunk)
    return articles


def guess_lang_from_messages(messages: List[Dict[str, str]]) -> str:
    text = " ".join(str(m.get("content", "")) for m in messages)
    if not text:
        return "unknown"
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_alpha = len(re.findall(r"[A-Za-z]", text))
    if zh > 30 and zh > ascii_alpha * 0.25:
        return "zh"
    return "en"


def build_noise_paragraph(rng: random.Random, words: int) -> str:
    words = max(160, int(words))
    toks: List[str] = []
    for i in range(words):
        tok = NOISE_VOCAB[rng.randint(0, len(NOISE_VOCAB) - 1)]
        if i % 17 == 0:
            tok = f"{tok}{rng.randint(10, 99)}"
        toks.append(tok)
    sents: List[str] = []
    cur: List[str] = []
    for i, tok in enumerate(toks, 1):
        cur.append(tok)
        if i % 24 == 0:
            sents.append(" ".join(cur).capitalize() + ".")
            cur = []
    if cur:
        sents.append(" ".join(cur).capitalize() + ".")
    return " ".join(sents)


def build_wikitext_continuation_candidate(
    *,
    articles: List[str],
    idx: int,
    rng: random.Random,
) -> Optional[Dict[str, object]]:
    if not articles:
        return None

    target_chars_cycle = [6500, 9500, 14000, 19000, 24000]
    target_chars = target_chars_cycle[idx % len(target_chars_cycle)]
    start = rng.randint(0, len(articles) - 1)

    chunks: List[str] = []
    total = 0
    ptr = start
    while total < target_chars and len(chunks) < 5:
        block = articles[ptr % len(articles)]
        chunks.append(block)
        total += len(block)
        ptr += 1
    context = "\n\n".join(chunks)[:target_chars]
    sents = split_sentences(context)
    if len(sents) < 18:
        return None

    cut = max(8, int(len(sents) * 0.72))
    cut = min(cut, len(sents) - 6)
    prompt_chunk = " ".join(sents[:cut])
    target_chunk = " ".join(sents[cut: cut + 8])
    if len(target_chunk) < 280:
        target_chunk = " ".join(sents[cut: cut + 12])
    if len(target_chunk) < 220:
        return None

    user = (
        "Continue the passage with the next coherent segment. Keep the same factual style and avoid adding new entities.\n"
        "Output 6-10 complete sentences.\n\n"
        f"Passage:\n{prompt_chunk}"
    )

    assistant = (
        "Continuation:\n"
        f"{target_chunk}\n\n"
        "Consistency note: wording follows the same entity timeline and terminology from the passage."
    )

    return {
        "prior": "power_law_base",
        "source_name": "wikitext",
        "lang": "en",
        "task_type": "long_continuation",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "generator": "wikitext_long_continuation_v2",
            "target_chars": target_chars,
            "article_start_index": start,
        },
    }


def build_synthetic_multihop_candidate(idx: int, rng: random.Random) -> Dict[str, object]:
    n_entities = 12
    projects = [f"PRJ-{idx:05d}-{i:02d}" for i in range(n_entities)]
    analysts = [f"Analyst-{idx:05d}-{i:02d}" for i in range(n_entities)]
    mentors = [f"Mentor-{idx:05d}-{i:02d}" for i in range(n_entities)]
    cities = [f"City-{idx:05d}-{i:02d}" for i in range(n_entities)]
    archives = [f"ARC-{idx % 97:02d}-{i:02d}-{7000 + (idx * 17 + i) % 900}" for i in range(n_entities)]

    rng.shuffle(analysts)
    rng.shuffle(mentors)
    rng.shuffle(cities)

    project_to_analyst = {p: analysts[i] for i, p in enumerate(projects)}
    analyst_to_mentor = {analysts[i]: mentors[(i * 3 + 1) % n_entities] for i in range(n_entities)}
    mentor_to_city = {mentors[i]: cities[(i * 5 + 2) % n_entities] for i in range(n_entities)}
    city_to_archive = {cities[i]: archives[i] for i in range(n_entities)}

    target_project = projects[rng.randint(0, n_entities - 1)]
    target_analyst = project_to_analyst[target_project]
    target_mentor = analyst_to_mentor[target_analyst]
    target_city = mentor_to_city[target_mentor]
    answer = city_to_archive[target_city]

    filler_words_cycle = [900, 1300, 1900, 2600]
    filler_words = filler_words_cycle[idx % len(filler_words_cycle)]
    filler_pre = build_noise_paragraph(rng, filler_words // 2)
    filler_post = build_noise_paragraph(rng, filler_words - filler_words // 2)

    table_1 = "\n".join([f"- {p} -> {project_to_analyst[p]}" for p in projects])
    table_2 = "\n".join([f"- {a} -> {analyst_to_mentor[a]}" for a in analysts])
    table_3 = "\n".join([f"- {m} -> {mentor_to_city[m]}" for m in mentors])
    table_4 = "\n".join([f"- {c} -> {city_to_archive[c]}" for c in cities])

    distractor = "\n".join([f"- DR-{idx:05d}-{i:02d} -> Z{rng.randint(100, 999)}" for i in range(28)])

    context = (
        "Operational memo with noisy logs:\n"
        f"{filler_pre}\n\n"
        "Table A: project-to-lead analyst\n"
        f"{table_1}\n\n"
        "Table B: analyst-to-mentor\n"
        f"{table_2}\n\n"
        "Table C: mentor-to-city\n"
        f"{table_3}\n\n"
        "Table D: city-to-archive-code\n"
        f"{table_4}\n\n"
        "Distractor index (not part of the chain):\n"
        f"{distractor}\n\n"
        "Noisy appendix:\n"
        f"{filler_post}"
    )

    question = (
        f"For project {target_project}, find the archive code of the city where the mentor of its lead analyst works. "
        "You must provide an explicit reasoning trace and end with one line `Final answer: <CODE>`."
    )

    user = (
        "Solve the multihop dependency query from the dossier.\n"
        "Requirements:\n"
        "1) At least five reasoning steps.\n"
        "2) Quote exact table entries at each step.\n"
        "3) Output final answer in the required format.\n\n"
        f"Dossier:\n{context}\n\nQuestion:\n{question}"
    )

    assistant = (
        "Reasoning trace:\n"
        f"Step 1: Table A maps {target_project} to lead analyst {target_analyst}.\n"
        f"Step 2: Table B maps analyst {target_analyst} to mentor {target_mentor}.\n"
        f"Step 3: Table C maps mentor {target_mentor} to city {target_city}.\n"
        f"Step 4: Table D maps city {target_city} to archive code {answer}.\n"
        "Step 5: The mapping chain is deterministic for this key because each table row has one value.\n"
        "Step 6: Distractor index entries are not referenced by the A->B->C->D chain, so they are ignored.\n"
        f"Final answer: {answer}"
    )

    return {
        "prior": "bimodal_reasoning",
        "source_name": "synthetic_multihop_qa",
        "lang": "en",
        "task_type": "multihop_qa_trace",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "generator": "synthetic_multihop_trace_v2",
            "hop_count": 4,
            "target_project": target_project,
            "target_analyst": target_analyst,
            "target_mentor": target_mentor,
            "target_city": target_city,
            "answer": answer,
            "filler_words": filler_words,
        },
    }


def build_synthetic_scaffold_candidate(idx: int, rng: random.Random) -> Dict[str, object]:
    intents = [
        "rewrite",
        "summarize",
        "classify",
        "extract",
        "translate",
    ]
    intent = intents[idx % len(intents)]

    user = (
        f"You are helping with a quick assistant-format check. Intent: {intent}.\n"
        "Please follow the exact output schema:\n"
        "1) One-line answer\n"
        "2) One bullet with rationale\n"
        "3) One bullet with caveat"
    )

    answer_core = {
        "rewrite": "Rewritten text keeps the original meaning with clearer structure.",
        "summarize": "Summary captures the main point and excludes side details.",
        "classify": "Label is selected by dominant signal in the provided text.",
        "extract": "Requested field is extracted directly from the statement.",
        "translate": "Translation keeps meaning and neutral register.",
    }[intent]

    assistant = (
        f"{answer_core}\n"
        "- Rationale: Output stays concise and follows instruction boundaries.\n"
        "- Caveat: If source context is ambiguous, confidence should be reduced."
    )

    lang = "zh" if rng.random() < 0.25 else "en"
    if lang == "zh":
        user = (
            f"这是一个助手格式对齐样本，任务类型：{intent}。\n"
            "请严格按以下格式输出：\n"
            "1) 一行结论\n2) 一条理由\n3) 一条限制"
        )
        assistant = (
            "结论：输出已按要求生成并保持信息完整。\n"
            "- 理由：格式和指令一致，表达简洁。\n"
            "- 限制：若原文歧义较大，需要降低置信度。"
        )

    return {
        "prior": "uniform_scaffold",
        "source_name": "synthetic_scaffold",
        "lang": lang,
        "task_type": "format_alignment",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "generator": "synthetic_scaffold_v1",
            "intent": intent,
        },
    }


def iter_jsonl_prior_candidates(
    *,
    paths: List[Path],
    prior: str,
    source_tag_prefix: str,
    default_task_type: str,
) -> Iterator[Dict[str, object]]:
    for path in paths:
        if not path.exists():
            continue
        for line_idx, obj in iter_json_records(path):
            msgs = normalize_messages(obj)
            if len(msgs) < 2:
                continue
            source_name = f"{source_tag_prefix}:{path.stem}"
            task_type = default_task_type
            meta = {
                "origin_path": path.as_posix(),
                "origin_line": int(line_idx),
            }
            lang = guess_lang_from_messages(msgs)
            yield {
                "prior": prior,
                "source_name": source_name,
                "lang": lang,
                "task_type": task_type,
                "messages": msgs,
                "meta": meta,
            }


def iter_powerlaw_candidates(
    *,
    wikitext_articles: List[str],
    powerlaw_json_paths: List[Path],
    seed: int,
) -> Iterator[Dict[str, object]]:
    rng = random.Random(seed + 11)

    # External long natural corpora (if provided) are consumed first.
    for cand in iter_jsonl_prior_candidates(
        paths=powerlaw_json_paths,
        prior="power_law_base",
        source_tag_prefix="powerlaw",
        default_task_type="long_semantic",
    ):
        yield cand

    idx = 0
    while True:
        cand = build_wikitext_continuation_candidate(articles=wikitext_articles, idx=idx, rng=rng)
        idx += 1
        if cand is not None:
            yield cand
        if not wikitext_articles and idx > 4:
            break


def iter_bimodal_candidates(
    *,
    bimodal_json_paths: List[Path],
    seed: int,
) -> Iterator[Dict[str, object]]:
    rng = random.Random(seed + 23)

    for cand in iter_jsonl_prior_candidates(
        paths=bimodal_json_paths,
        prior="bimodal_reasoning",
        source_tag_prefix="bimodal",
        default_task_type="long_instruction_qa",
    ):
        yield cand

    idx = 0
    while True:
        yield build_synthetic_multihop_candidate(idx=idx, rng=rng)
        idx += 1


def iter_scaffold_candidates(
    *,
    scaffold_json_paths: List[Path],
    seed: int,
) -> Iterator[Dict[str, object]]:
    rng = random.Random(seed + 31)

    for cand in iter_jsonl_prior_candidates(
        paths=scaffold_json_paths,
        prior="uniform_scaffold",
        source_tag_prefix="scaffold",
        default_task_type="assistant_alignment",
    ):
        yield cand

    idx = 0
    while True:
        yield build_synthetic_scaffold_candidate(idx=idx, rng=rng)
        idx += 1


def build_response_only_labels(
    *,
    full_ids: List[int],
    assistant_start_full: int,
    max_seq_len: int,
    truncate_head_cap: int,
) -> Tuple[List[int], List[int], int, int]:
    if len(full_ids) > int(max_seq_len):
        head_len = min(int(truncate_head_cap), max(0, int(max_seq_len) - 1))
        tail_len = max(0, int(max_seq_len) - head_len)
        middle_start = int(head_len)
        middle_end = int(len(full_ids) - tail_len)
        if tail_len > 0:
            truncated = list(full_ids[:head_len] + full_ids[-tail_len:])
        else:
            truncated = list(full_ids[:head_len])
        if int(assistant_start_full) < middle_start:
            assistant_start = int(assistant_start_full)
        elif int(assistant_start_full) >= middle_end:
            assistant_start = int(head_len + (int(assistant_start_full) - middle_end))
        else:
            # Assistant boundary falls into dropped middle; sample must be rejected by caller.
            assistant_start = int(len(truncated))
    else:
        truncated = list(full_ids)
        assistant_start = int(assistant_start_full)

    labels = list(truncated)
    for j in range(max(0, min(assistant_start, len(labels)))):
        labels[j] = -100

    supervised = sum(1 for x in labels if int(x) != -100)
    return truncated, labels, assistant_start, supervised


def render_and_prepare(
    *,
    tokenizer,
    candidate: Dict[str, object],
    max_seq_len: int,
    min_supervised_tokens: int,
    truncate_head_cap: int,
    require_offset_boundary: bool,
) -> Tuple[Optional[PreparedSample], str]:
    messages = candidate.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        return None, "bad_messages"
    if str(messages[-1].get("role", "")).strip().lower() != "assistant":
        return None, "bad_messages"
    if not str(messages[-1].get("content", "")).strip():
        return None, "empty_assistant"

    prefix_msgs = list(messages[:-1]) + [{"role": "assistant", "content": ""}]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_text = tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=False)
    except Exception:
        return None, "template_error"

    # Keep exact rendered boundary (no strip) to avoid assistant offset drift.
    text = str(text)
    prefix_text = str(prefix_text)
    if not text.strip() or not prefix_text.strip():
        return None, "render_error"

    full_offsets = None
    try:
        full_enc = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
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
        pref_ids = tokenizer(prefix_text, add_special_tokens=False, truncation=False)["input_ids"]
    except Exception:
        return None, "tokenize_error"

    if not full_ids or not pref_ids:
        return None, "tokenize_error"
    assistant_start_full = int(len(pref_ids))
    if full_offsets is not None:
        prefix_char_len = len(prefix_text)
        assistant_start_full = int(
            next((idx for idx, pair in enumerate(full_offsets) if int(pair[0]) >= prefix_char_len), len(full_ids))
        )
    elif bool(require_offset_boundary):
        return None, "missing_offset_mapping"

    truncated, labels, assistant_start, supervised = build_response_only_labels(
        full_ids=full_ids,
        assistant_start_full=assistant_start_full,
        max_seq_len=max_seq_len,
        truncate_head_cap=int(truncate_head_cap),
    )
    if assistant_start >= len(truncated):
        return None, "assistant_out_of_window"

    if supervised < int(min_supervised_tokens):
        return None, "low_supervised"

    sample = PreparedSample(
        prior=str(candidate.get("prior", "unknown")),
        source_name=str(candidate.get("source_name", "unknown")),
        lang=str(candidate.get("lang", "unknown")),
        task_type=str(candidate.get("task_type", "unknown")),
        messages=messages,
        meta=dict(candidate.get("meta", {}) or {}),
        full_tokens=len(full_ids),
        truncated_tokens=len(truncated),
        assistant_tokens=supervised,
        assistant_start=assistant_start,
        dedup_sha256=sha256_text(text),
    )
    _ = labels  # labels are re-generated in previews and train script, not stored per-sample.
    return sample, "ok"


def collect_prior_pool(
    *,
    prior_name: str,
    provider: Iterator[Dict[str, object]],
    target_tokens: int,
    pool_factor: float,
    max_attempts: int,
    tokenizer,
    max_seq_len: int,
    min_supervised_tokens: int,
    truncate_head_cap: int,
    require_offset_boundary: bool,
    seen_hashes: Set[str],
    strict: bool,
) -> Tuple[List[PreparedSample], Dict[str, object]]:
    need_pool_tokens = int(max(1, round(float(target_tokens) * float(pool_factor))))
    attempts = 0
    kept: List[PreparedSample] = []
    kept_tokens = 0

    stats: Dict[str, object] = {
        "prior": prior_name,
        "target_tokens": int(target_tokens),
        "pool_token_goal": int(need_pool_tokens),
        "attempted": 0,
        "kept": 0,
        "kept_tokens": 0,
        "dropped_duplicate": 0,
        "dropped_bad_messages": 0,
        "dropped_empty_assistant": 0,
        "dropped_template_error": 0,
        "dropped_render_error": 0,
        "dropped_tokenize_error": 0,
        "dropped_low_supervised": 0,
        "retention_after_response_filter": 0.0,
        "retention_after_dedup": 0.0,
        "provider_exhausted": False,
    }

    passed_response_filter = 0

    while attempts < int(max_attempts) and kept_tokens < need_pool_tokens:
        attempts += 1
        try:
            cand = next(provider)
        except StopIteration:
            stats["provider_exhausted"] = True
            break

        sample, reason = render_and_prepare(
            tokenizer=tokenizer,
            candidate=cand,
            max_seq_len=max_seq_len,
            min_supervised_tokens=min_supervised_tokens,
            truncate_head_cap=int(truncate_head_cap),
            require_offset_boundary=bool(require_offset_boundary),
        )
        if sample is None:
            key = f"dropped_{reason}"
            stats[key] = int(stats.get(key, 0)) + 1
            continue

        passed_response_filter += 1
        if sample.dedup_sha256 in seen_hashes:
            stats["dropped_duplicate"] = int(stats.get("dropped_duplicate", 0)) + 1
            continue

        seen_hashes.add(sample.dedup_sha256)
        kept.append(sample)
        kept_tokens += int(sample.truncated_tokens)

    stats["attempted"] = int(attempts)
    stats["kept"] = int(len(kept))
    stats["kept_tokens"] = int(kept_tokens)
    stats["retention_after_response_filter"] = float(passed_response_filter) / float(max(1, attempts))
    stats["retention_after_dedup"] = float(len(kept)) / float(max(1, attempts))

    if kept_tokens < int(target_tokens) and bool(strict):
        raise RuntimeError(
            f"Insufficient {prior_name} token pool: kept_tokens={kept_tokens}, target_tokens={target_tokens}, "
            f"attempted={attempts}, stats={stats}"
        )

    return kept, stats


def select_to_target_tokens(
    *,
    pool: List[PreparedSample],
    target_tokens: int,
    rng: random.Random,
) -> Tuple[List[PreparedSample], int]:
    if not pool:
        return [], 0

    rows = list(pool)
    rng.shuffle(rows)
    selected: List[PreparedSample] = []
    total = 0

    for s in rows:
        if total >= int(target_tokens) and selected:
            break
        selected.append(s)
        total += int(s.truncated_tokens)

    if not selected:
        selected.append(rows[0])
        total = int(rows[0].truncated_tokens)

    return selected, int(total)


def stratified_split(
    *,
    by_prior: Dict[str, List[PreparedSample]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Dict[str, List[PreparedSample]]:
    out = {"train": [], "valid": [], "test": []}
    for _, rows in by_prior.items():
        items = list(rows)
        rng.shuffle(items)
        n = len(items)
        if n == 0:
            continue

        n_valid = max(1, int(round(n * valid_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        if n_valid + n_test >= n:
            n_valid = max(1, min(n - 2, n_valid))
            n_test = max(1, n - n_valid - 1)
        n_train = n - n_valid - n_test

        out["train"].extend(items[:n_train])
        out["valid"].extend(items[n_train: n_train + n_valid])
        out["test"].extend(items[n_train + n_valid:])

    for split in out:
        rng.shuffle(out[split])
    return out


def summarize_samples(samples: List[PreparedSample]) -> Dict[str, object]:
    by_prior_counts: Dict[str, int] = {}
    by_prior_tokens: Dict[str, int] = {}
    by_source_counts: Dict[str, int] = {}
    by_source_tokens: Dict[str, int] = {}
    by_lang_tokens: Dict[str, int] = {}
    by_task_tokens: Dict[str, int] = {}

    full_tokens: List[int] = []
    trunc_tokens: List[int] = []
    assistant_tokens: List[int] = []
    for s in samples:
        by_prior_counts[s.prior] = by_prior_counts.get(s.prior, 0) + 1
        by_prior_tokens[s.prior] = by_prior_tokens.get(s.prior, 0) + int(s.truncated_tokens)

        by_source_counts[s.source_name] = by_source_counts.get(s.source_name, 0) + 1
        by_source_tokens[s.source_name] = by_source_tokens.get(s.source_name, 0) + int(s.truncated_tokens)

        by_lang_tokens[s.lang] = by_lang_tokens.get(s.lang, 0) + int(s.truncated_tokens)
        by_task_tokens[s.task_type] = by_task_tokens.get(s.task_type, 0) + int(s.truncated_tokens)

        full_tokens.append(int(s.full_tokens))
        trunc_tokens.append(int(s.truncated_tokens))
        assistant_tokens.append(int(s.assistant_tokens))

    total_tokens = sum(trunc_tokens)
    prior_share = {k: float(v) / float(max(1, total_tokens)) for k, v in sorted(by_prior_tokens.items())}
    source_share = {k: float(v) / float(max(1, total_tokens)) for k, v in sorted(by_source_tokens.items())}

    return {
        "num_samples": int(len(samples)),
        "total_truncated_tokens": int(total_tokens),
        "prior_counts": by_prior_counts,
        "prior_tokens": by_prior_tokens,
        "prior_token_share": prior_share,
        "source_counts": by_source_counts,
        "source_tokens": by_source_tokens,
        "source_token_share": source_share,
        "lang_tokens": by_lang_tokens,
        "task_tokens": by_task_tokens,
        "full_tokens": quantiles(full_tokens),
        "truncated_tokens": quantiles(trunc_tokens),
        "assistant_tokens": quantiles(assistant_tokens),
        "assistant_tokens_lt64_ratio": float(sum(1 for x in assistant_tokens if x < 64)) / float(max(1, len(assistant_tokens))),
    }


def write_jsonl(path: Path, rows: List[PreparedSample], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(rows):
            rec = {
                "id": f"{split}_{i:07d}",
                "source_prior": s.prior,
                "source_name": s.source_name,
                "lang": s.lang,
                "task_type": s.task_type,
                "messages": s.messages,
                "meta": {
                    **s.meta,
                    "full_tokens": int(s.full_tokens),
                    "truncated_tokens": int(s.truncated_tokens),
                    "assistant_tokens": int(s.assistant_tokens),
                    "assistant_start": int(s.assistant_start),
                    "dedup_sha256": s.dedup_sha256,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_preview(path: Path, rows: List[PreparedSample], n: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(rows[:n]):
            user_text = ""
            for m in s.messages:
                if m.get("role") == "user":
                    user_text = str(m.get("content", ""))
                    break
            assistant_text = str(s.messages[-1].get("content", "")) if s.messages else ""
            f.write(
                f"[#{i}] prior={s.prior} source={s.source_name} "
                f"tokens(full/trunc/asst)={s.full_tokens}/{s.truncated_tokens}/{s.assistant_tokens}\n"
            )
            f.write("USER:\n" + user_text[:1800] + "\n")
            f.write("ASSISTANT:\n" + assistant_text[:1200] + "\n")
            f.write("-" * 100 + "\n")


def build_label_mask_preview(
    *,
    tokenizer,
    rows: List[PreparedSample],
    max_seq_len: int,
    min_supervised_tokens: int,
    truncate_head_cap: int,
    require_offset_boundary: bool,
    sample_n: int,
) -> Tuple[Dict[str, object], bool]:
    previews: List[Dict[str, object]] = []
    ok = True

    for i, s in enumerate(rows[: max(1, int(sample_n))]):
        prefix_msgs = list(s.messages[:-1]) + [{"role": "assistant", "content": ""}]
        full_text = tokenizer.apply_chat_template(s.messages, tokenize=False, add_generation_prompt=False)
        prefix_text = tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=False)
        full_offsets = None
        full_enc = tokenizer(str(full_text), add_special_tokens=False, truncation=False, return_offsets_mapping=True)
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
        pref_ids = tokenizer(str(prefix_text), add_special_tokens=False, truncation=False)["input_ids"]
        assistant_start_full = int(len(pref_ids))
        if full_offsets is not None:
            prefix_char_len = len(str(prefix_text))
            assistant_start_full = int(
                next((idx for idx, pair in enumerate(full_offsets) if int(pair[0]) >= prefix_char_len), len(full_ids))
            )
        elif bool(require_offset_boundary):
            previews.append(
                {
                    "index": int(i),
                    "prior": s.prior,
                    "source_name": s.source_name,
                    "check_pass": False,
                    "error": "missing_offset_mapping",
                }
            )
            ok = False
            continue
        trunc_ids, labels, assistant_start, supervised = build_response_only_labels(
            full_ids=full_ids,
            assistant_start_full=assistant_start_full,
            max_seq_len=max_seq_len,
            truncate_head_cap=int(truncate_head_cap),
        )

        has_masked_prefix = all(x == -100 for x in labels[:assistant_start]) if assistant_start > 0 else True
        has_supervised_suffix = any(x != -100 for x in labels[assistant_start:]) if assistant_start < len(labels) else False
        local_ok = bool(has_masked_prefix and has_supervised_suffix and supervised >= int(min_supervised_tokens))
        ok = bool(ok and local_ok)

        previews.append(
            {
                "index": int(i),
                "prior": s.prior,
                "source_name": s.source_name,
                "truncated_len": int(len(trunc_ids)),
                "assistant_start": int(assistant_start),
                "supervised_tokens": int(supervised),
                "first_64_input_ids": [int(x) for x in trunc_ids[:64]],
                "first_64_labels": [int(x) for x in labels[:64]],
                "last_64_input_ids": [int(x) for x in trunc_ids[-64:]],
                "last_64_labels": [int(x) for x in labels[-64:]],
                "check_has_masked_prefix": bool(has_masked_prefix),
                "check_has_supervised_suffix": bool(has_supervised_suffix),
                "check_pass": bool(local_ok),
            }
        )

    return {
        "sampled": int(min(len(rows), max(1, int(sample_n)))),
        "previews": previews,
    }, bool(ok)


def build_source_audit(
    *,
    rows: List[PreparedSample],
    input_files: Dict[str, Dict[str, object]],
    filter_rule: str,
) -> List[Dict[str, object]]:
    counts: Dict[str, int] = {}
    token_sum: Dict[str, int] = {}
    lang_vote: Dict[str, Dict[str, int]] = {}
    task_vote: Dict[str, Dict[str, int]] = {}

    for s in rows:
        counts[s.source_name] = counts.get(s.source_name, 0) + 1
        token_sum[s.source_name] = token_sum.get(s.source_name, 0) + int(s.truncated_tokens)
        if s.source_name not in lang_vote:
            lang_vote[s.source_name] = {}
        if s.source_name not in task_vote:
            task_vote[s.source_name] = {}
        lang_vote[s.source_name][s.lang] = lang_vote[s.source_name].get(s.lang, 0) + 1
        task_vote[s.source_name][s.task_type] = task_vote[s.source_name].get(s.task_type, 0) + 1

    total_tokens = sum(token_sum.values())

    source_sha: Dict[str, Optional[str]] = {}
    for k, info in input_files.items():
        p = str(info.get("path", ""))
        stem = Path(p).stem
        if stem:
            source_sha[f"powerlaw:{stem}"] = info.get("sha256")
            source_sha[f"bimodal:{stem}"] = info.get("sha256")
            source_sha[f"scaffold:{stem}"] = info.get("sha256")
    source_sha["wikitext"] = input_files.get("wikitext", {}).get("sha256")
    source_sha["synthetic_multihop_qa"] = sha256_text("synthetic_multihop_trace_v2")
    source_sha["synthetic_scaffold"] = sha256_text("synthetic_scaffold_v1")

    out: List[Dict[str, object]] = []
    for src in sorted(counts.keys()):
        lang_d = lang_vote.get(src, {})
        task_d = task_vote.get(src, {})
        lang = max(lang_d.items(), key=lambda x: x[1])[0] if lang_d else "unknown"
        task = max(task_d.items(), key=lambda x: x[1])[0] if task_d else "unknown"
        out.append(
            {
                "source_name": src,
                "lang": lang,
                "task_type": task,
                "count": int(counts[src]),
                "token_ratio": float(token_sum[src]) / float(max(1, total_tokens)),
                "sha256": source_sha.get(src),
                "filter_rule": filter_rule,
            }
        )
    return out


def build_quality_report(path: Path, manifest: Dict[str, object]) -> None:
    q = manifest["quality_checks"]
    summary = manifest["summary"]
    target_tokens = manifest["targets"]["tokens"]
    actual_tokens = summary["prior_tokens"]

    lines: List[str] = []
    lines.append("# Mixed Prior Dataset v1 Quality Report")
    lines.append("")
    lines.append(f"- generated_at: `{manifest['generated_at']}`")
    lines.append(f"- output_dir: `{manifest['output_dir']}`")
    lines.append(f"- dry_run: `{manifest['dry_run']}`")
    lines.append("")

    lines.append("## Prior Recipe (Token Budget)")
    lines.append("")
    lines.append("| Prior | Target Tokens | Actual Tokens | Actual Share |")
    lines.append("|---|---:|---:|---:|")
    for prior in ["power_law_base", "bimodal_reasoning", "uniform_scaffold"]:
        act = int(actual_tokens.get(prior, 0))
        share = float(summary["prior_token_share"].get(prior, 0.0))
        lines.append(f"| {prior} | {int(target_tokens.get(prior, 0))} | {act} | {share:.4f} |")
    lines.append("")

    lines.append("## Key Checks")
    lines.append("")
    lines.append("| Check | Value | Target | Status |")
    lines.append("|---|---:|---:|:---:|")
    lines.append(
        f"| max_prior_ratio_abs_error | {float(q['max_prior_ratio_abs_error']):.4f} | <= {float(q['ratio_tolerance']):.4f} | "
        f"{'PASS' if q['mix_ratio_ok'] else 'FAIL'} |"
    )
    lines.append(
        f"| assistant_tokens_lt64_ratio | {float(summary['assistant_tokens_lt64_ratio']):.4f} | < 0.05 | "
        f"{'PASS' if q['assistant_lt64_ok'] else 'FAIL'} |"
    )
    lines.append(
        f"| label_mask_preview_pass | {bool(manifest['label_mask_preview'].get('pass', False))} | true | "
        f"{'PASS' if q['label_mask_preview_ok'] else 'FAIL'} |"
    )
    lines.append("")

    lines.append("## Filter / Dedup Audit")
    lines.append("")
    lines.append("| Prior | Attempted | Kept | Kept Tokens | Retention | dropped_low_supervised | dropped_duplicate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for prior in ["power_law_base", "bimodal_reasoning", "uniform_scaffold"]:
        st = manifest["filter_stats"][prior]
        lines.append(
            f"| {prior} | {int(st['attempted'])} | {int(st['kept'])} | {int(st['kept_tokens'])} | "
            f"{float(st['retention_after_dedup']):.4f} | {int(st.get('dropped_low_supervised', 0))} | "
            f"{int(st.get('dropped_duplicate', 0))} |"
        )

    lines.append("")
    lines.append("## Required Source Audit Fields")
    lines.append("")
    lines.append("| source_name | lang | task_type | count | token_ratio | sha256 | filter_rule |")
    lines.append("|---|---|---|---:|---:|---|---|")
    for row in manifest.get("source_audit", []):
        lines.append(
            f"| {row.get('source_name','')} | {row.get('lang','')} | {row.get('task_type','')} | "
            f"{int(row.get('count',0))} | {float(row.get('token_ratio',0.0)):.4f} | "
            f"{row.get('sha256') or ''} | {row.get('filter_rule','')} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare strict mixed-prior finetune dataset by token ratio")

    ap.add_argument("--wikitext_path", type=str, default=DEFAULT_WIKITEXT)
    ap.add_argument("--powerlaw_jsonl_paths", type=str, default="")
    ap.add_argument("--bimodal_jsonl_paths", type=str, default=f"{DEFAULT_LONGALPACA},{DEFAULT_LONGQA}")
    ap.add_argument("--scaffold_jsonl_paths", type=str, default="")
    ap.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER)

    ap.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--output_dir", type=str, default="")
    ap.add_argument("--dataset_prefix", type=str, default="mixed_prior_v1")

    ap.add_argument("--target_total_tokens", type=int, default=200_000_000)
    ap.add_argument("--powerlaw_ratio", type=float, default=0.50)
    ap.add_argument("--bimodal_ratio", type=float, default=0.40)
    ap.add_argument("--scaffold_ratio", type=float, default=0.10)

    ap.add_argument("--train_ratio", type=float, default=0.96)
    ap.add_argument("--valid_ratio", type=float, default=0.02)
    ap.add_argument("--test_ratio", type=float, default=0.02)

    ap.add_argument("--max_seq_len", type=int, default=16384)
    ap.add_argument("--training_truncate_head_cap_tokens", type=int, default=TRAIN_TRUNCATE_HEAD_CAP)
    ap.add_argument("--min_supervised_tokens", type=int, default=64)
    ap.add_argument("--require_offset_boundary", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pool_factor", type=float, default=1.12)

    ap.add_argument("--max_attempts_powerlaw", type=int, default=180000)
    ap.add_argument("--max_attempts_bimodal", type=int, default=180000)
    ap.add_argument("--max_attempts_scaffold", type=int, default=80000)

    ap.add_argument("--ratio_tolerance", type=float, default=0.02)
    ap.add_argument("--preview_count", type=int, default=50)
    ap.add_argument("--label_preview_samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--dry_run_tokens", type=int, default=2_500_000)
    ap.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))

    output_dir = Path(args.output_dir) if str(args.output_dir).strip() else Path(args.output_root) / f"{args.dataset_prefix}_{now_ts()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_total_tokens = int(args.target_total_tokens)
    if bool(args.dry_run):
        target_total_tokens = min(target_total_tokens, int(args.dry_run_tokens))

    r_power = max(0.0, float(args.powerlaw_ratio))
    r_bi = max(0.0, float(args.bimodal_ratio))
    r_scf = max(0.0, float(args.scaffold_ratio))
    r_sum = r_power + r_bi + r_scf
    if r_sum <= 0.0:
        raise RuntimeError("Invalid prior ratios.")

    target_tokens = {
        "power_law_base": int(round(target_total_tokens * (r_power / r_sum))),
        "bimodal_reasoning": int(round(target_total_tokens * (r_bi / r_sum))),
        "uniform_scaffold": 0,
    }
    target_tokens["uniform_scaffold"] = max(0, int(target_total_tokens) - target_tokens["power_law_base"] - target_tokens["bimodal_reasoning"])

    powerlaw_paths = parse_csv_paths(args.powerlaw_jsonl_paths)
    bimodal_paths = parse_csv_paths(args.bimodal_jsonl_paths)
    scaffold_paths = parse_csv_paths(args.scaffold_jsonl_paths)

    input_files: Dict[str, Dict[str, object]] = {}
    for tag, path in [("wikitext", Path(args.wikitext_path))]:
        input_files[tag] = {
            "path": path.as_posix(),
            "exists": bool(path.exists()),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    for p in powerlaw_paths:
        input_files[f"powerlaw::{p.stem}"] = {
            "path": p.as_posix(),
            "exists": bool(p.exists()),
            "sha256": sha256_file(p) if p.exists() else None,
        }
    for p in bimodal_paths:
        input_files[f"bimodal::{p.stem}"] = {
            "path": p.as_posix(),
            "exists": bool(p.exists()),
            "sha256": sha256_file(p) if p.exists() else None,
        }
    for p in scaffold_paths:
        input_files[f"scaffold::{p.stem}"] = {
            "path": p.as_posix(),
            "exists": bool(p.exists()),
            "sha256": sha256_file(p) if p.exists() else None,
        }

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    wikitext_articles = load_wikitext_articles(Path(args.wikitext_path), min_chars=1400)

    provider_power = iter_powerlaw_candidates(
        wikitext_articles=wikitext_articles,
        powerlaw_json_paths=powerlaw_paths,
        seed=int(args.seed),
    )
    provider_bi = iter_bimodal_candidates(
        bimodal_json_paths=bimodal_paths,
        seed=int(args.seed),
    )
    provider_scf = iter_scaffold_candidates(
        scaffold_json_paths=scaffold_paths,
        seed=int(args.seed),
    )

    seen_hashes: Set[str] = set()

    pool_power, st_power = collect_prior_pool(
        prior_name="power_law_base",
        provider=provider_power,
        target_tokens=int(target_tokens["power_law_base"]),
        pool_factor=float(args.pool_factor),
        max_attempts=int(args.max_attempts_powerlaw),
        tokenizer=tokenizer,
        max_seq_len=int(args.max_seq_len),
        min_supervised_tokens=int(args.min_supervised_tokens),
        truncate_head_cap=int(args.training_truncate_head_cap_tokens),
        require_offset_boundary=bool(args.require_offset_boundary),
        seen_hashes=seen_hashes,
        strict=bool(args.strict),
    )
    pool_bi, st_bi = collect_prior_pool(
        prior_name="bimodal_reasoning",
        provider=provider_bi,
        target_tokens=int(target_tokens["bimodal_reasoning"]),
        pool_factor=float(args.pool_factor),
        max_attempts=int(args.max_attempts_bimodal),
        tokenizer=tokenizer,
        max_seq_len=int(args.max_seq_len),
        min_supervised_tokens=int(args.min_supervised_tokens),
        truncate_head_cap=int(args.training_truncate_head_cap_tokens),
        require_offset_boundary=bool(args.require_offset_boundary),
        seen_hashes=seen_hashes,
        strict=bool(args.strict),
    )
    pool_scf, st_scf = collect_prior_pool(
        prior_name="uniform_scaffold",
        provider=provider_scf,
        target_tokens=int(target_tokens["uniform_scaffold"]),
        pool_factor=float(args.pool_factor),
        max_attempts=int(args.max_attempts_scaffold),
        tokenizer=tokenizer,
        max_seq_len=int(args.max_seq_len),
        min_supervised_tokens=int(args.min_supervised_tokens),
        truncate_head_cap=int(args.training_truncate_head_cap_tokens),
        require_offset_boundary=bool(args.require_offset_boundary),
        seen_hashes=seen_hashes,
        strict=bool(args.strict),
    )

    selected_power, sel_tok_power = select_to_target_tokens(
        pool=pool_power,
        target_tokens=int(target_tokens["power_law_base"]),
        rng=rng,
    )
    selected_bi, sel_tok_bi = select_to_target_tokens(
        pool=pool_bi,
        target_tokens=int(target_tokens["bimodal_reasoning"]),
        rng=rng,
    )
    selected_scf, sel_tok_scf = select_to_target_tokens(
        pool=pool_scf,
        target_tokens=int(target_tokens["uniform_scaffold"]),
        rng=rng,
    )

    all_rows = list(selected_power) + list(selected_bi) + list(selected_scf)
    rng.shuffle(all_rows)

    if not all_rows:
        raise RuntimeError("No selected rows. Dataset construction failed.")

    by_prior = {
        "power_law_base": selected_power,
        "bimodal_reasoning": selected_bi,
        "uniform_scaffold": selected_scf,
    }
    splits = stratified_split(
        by_prior=by_prior,
        train_ratio=float(args.train_ratio),
        valid_ratio=float(args.valid_ratio),
        test_ratio=float(args.test_ratio),
        rng=rng,
    )

    summary = summarize_samples(all_rows)

    label_preview_obj, label_preview_ok = build_label_mask_preview(
        tokenizer=tokenizer,
        rows=splits["train"],
        max_seq_len=int(args.max_seq_len),
        min_supervised_tokens=int(args.min_supervised_tokens),
        truncate_head_cap=int(args.training_truncate_head_cap_tokens),
        require_offset_boundary=bool(args.require_offset_boundary),
        sample_n=int(args.label_preview_samples),
    )
    label_preview_obj["pass"] = bool(label_preview_ok)
    for p in label_preview_obj.get("previews", []):
        idx = int(p.get("index", 0))
        print(
            f"[MASK CHECK #{idx}] source={p.get('source_name')} "
            f"assistant_start={p.get('assistant_start')} supervised={p.get('supervised_tokens')} "
            f"pass={p.get('check_pass')}",
            flush=True,
        )
        print(f"  tail_input_ids={p.get('last_64_input_ids', [])[:16]}", flush=True)
        print(f"  tail_labels={p.get('last_64_labels', [])[:16]}", flush=True)

    # Required pre-flight visibility.
    actual_share = summary["prior_token_share"]
    print(
        "[DATA CHECK] token_ratio "
        f"power_law_base={float(actual_share.get('power_law_base', 0.0)):.4f}, "
        f"bimodal_reasoning={float(actual_share.get('bimodal_reasoning', 0.0)):.4f}, "
        f"uniform_scaffold={float(actual_share.get('uniform_scaffold', 0.0)):.4f}",
        flush=True,
    )

    target_share = {
        "power_law_base": float(r_power / r_sum),
        "bimodal_reasoning": float(r_bi / r_sum),
        "uniform_scaffold": float(r_scf / r_sum),
    }

    prior_abs_err = {
        k: abs(float(actual_share.get(k, 0.0)) - float(target_share.get(k, 0.0)))
        for k in ["power_law_base", "bimodal_reasoning", "uniform_scaffold"]
    }
    max_abs_err = max(prior_abs_err.values()) if prior_abs_err else 1.0

    quality_checks = {
        "mix_ratio_ok": bool(max_abs_err <= float(args.ratio_tolerance)),
        "assistant_lt64_ok": bool(float(summary["assistant_tokens_lt64_ratio"]) < 0.05),
        "label_mask_preview_ok": bool(label_preview_ok),
        "ratio_tolerance": float(args.ratio_tolerance),
        "max_prior_ratio_abs_error": float(max_abs_err),
    }

    filter_rule = (
        "response_only("
        f"truncate_mode=head_tail_keep_drop_middle; head_cap={int(args.training_truncate_head_cap_tokens)}; "
        f"max_seq_len={int(args.max_seq_len)}; "
        f"require_offset_boundary={bool(args.require_offset_boundary)}; "
        f"min_supervised_tokens={int(args.min_supervised_tokens)}; "
        "labels[:assistant_start]=-100; assistant_out_of_window=drop)"
    )
    source_audit = build_source_audit(rows=all_rows, input_files=input_files, filter_rule=filter_rule)

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    test_path = output_dir / "test.jsonl"
    unified_path = output_dir / "mixed_prior_finetune.jsonl"
    preview_path = output_dir / "preview_50.txt"
    label_preview_path = output_dir / "label_mask_preview.json"

    write_jsonl(train_path, splits["train"], split="train")
    write_jsonl(valid_path, splits["valid"], split="valid")
    write_jsonl(test_path, splits["test"], split="test")
    write_jsonl(unified_path, all_rows, split="all")
    write_preview(preview_path, splits["train"], n=int(args.preview_count))
    label_preview_path.write_text(json.dumps(label_preview_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    split_counts = {
        split: {
            "num_samples": int(len(rows)),
            "tokens": int(sum(int(x.truncated_tokens) for x in rows)),
            "prior_counts": {
                p: int(sum(1 for x in rows if x.prior == p))
                for p in ["power_law_base", "bimodal_reasoning", "uniform_scaffold"]
            },
        }
        for split, rows in splits.items()
    }

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output_dir.as_posix(),
        "dry_run": bool(args.dry_run),
        "seed": int(args.seed),
        "tokenizer_path": str(args.tokenizer_path),
        "max_seq_len": int(args.max_seq_len),
        "training_truncate_head_cap_tokens": int(args.training_truncate_head_cap_tokens),
        "training_truncate_mode": "head_tail_keep_drop_middle",
        "min_supervised_tokens": int(args.min_supervised_tokens),
        "require_offset_boundary": bool(args.require_offset_boundary),
        "targets": {
            "ratios": target_share,
            "tokens": {
                "power_law_base": int(target_tokens["power_law_base"]),
                "bimodal_reasoning": int(target_tokens["bimodal_reasoning"]),
                "uniform_scaffold": int(target_tokens["uniform_scaffold"]),
                "total": int(target_total_tokens),
            },
        },
        "actual": {
            "prior_token_share": summary["prior_token_share"],
            "prior_tokens": summary["prior_tokens"],
            "total_tokens": int(summary["total_truncated_tokens"]),
            "selected_tokens": {
                "power_law_base": int(sel_tok_power),
                "bimodal_reasoning": int(sel_tok_bi),
                "uniform_scaffold": int(sel_tok_scf),
            },
        },
        "summary": summary,
        "split_counts": split_counts,
        "filter_stats": {
            "power_law_base": st_power,
            "bimodal_reasoning": st_bi,
            "uniform_scaffold": st_scf,
        },
        "quality_checks": quality_checks,
        "label_mask_preview": label_preview_obj,
        "source_audit": source_audit,
        "input_files": input_files,
        "filter_rule": filter_rule,
        "raw_source_counts": {
            "wikitext_articles": int(len(wikitext_articles)),
            "powerlaw_pool_rows": int(len(pool_power)),
            "bimodal_pool_rows": int(len(pool_bi)),
            "scaffold_pool_rows": int(len(pool_scf)),
        },
        "output_files": {
            "unified": unified_path.as_posix(),
            "train": train_path.as_posix(),
            "valid": valid_path.as_posix(),
            "test": test_path.as_posix(),
            "preview": preview_path.as_posix(),
            "label_mask_preview": label_preview_path.as_posix(),
        },
    }

    manifest_path = output_dir / "mix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = output_dir / "quality_report.md"
    build_quality_report(report_path, manifest)

    output_hashes = {
        "unified_sha256": sha256_file(unified_path),
        "train_sha256": sha256_file(train_path),
        "valid_sha256": sha256_file(valid_path),
        "test_sha256": sha256_file(test_path),
        "preview_sha256": sha256_file(preview_path),
        "label_mask_preview_sha256": sha256_file(label_preview_path),
        "manifest_sha256": sha256_file(manifest_path),
        "quality_report_sha256": sha256_file(report_path),
    }
    manifest["output_hashes"] = output_hashes
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if bool(args.strict):
        failed = [k for k, v in quality_checks.items() if k.endswith("_ok") and not bool(v)]
        if failed:
            raise RuntimeError(f"Quality gate failed: {failed}")

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": output_dir.as_posix(),
                "target_total_tokens": int(target_total_tokens),
                "actual_total_tokens": int(summary["total_truncated_tokens"]),
                "prior_token_share": summary["prior_token_share"],
                "quality_checks": quality_checks,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
