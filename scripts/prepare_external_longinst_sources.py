#!/usr/bin/env python3
"""Prepare external long-instruction sources for mixed-prior training.

This script downloads public datasets from Hugging Face (optionally via mirror)
and converts them into the JSONL schema consumed by
`scripts/prepare_mixed_prior_dataset_v1.py`.

Outputs:
- hotpotqa_multihop.jsonl
- alpaca_scaffold.jsonl
- source_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build external JSONL sources for longinst mixed-prior dataset.")
    ap.add_argument("--output_dir", type=str, default="artifacts/datasets/external_sources")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    ap.add_argument("--hf_cache_dir", type=str, default="/root/autodl-tmp/.cache/huggingface")

    ap.add_argument("--hotpot_dataset", type=str, default="hotpotqa/hotpot_qa")
    ap.add_argument("--hotpot_config", type=str, default="fullwiki")
    ap.add_argument("--hotpot_split", type=str, default="train")
    ap.add_argument("--hotpot_max_samples", type=int, default=30000)
    ap.add_argument("--hotpot_max_passages", type=int, default=10)
    ap.add_argument("--hotpot_max_sent_per_passage", type=int, default=6)

    ap.add_argument("--alpaca_dataset", type=str, default="yahma/alpaca-cleaned")
    ap.add_argument("--alpaca_split", type=str, default="train")
    ap.add_argument("--alpaca_max_samples", type=int, default=15000)

    ap.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    return ap.parse_args()


def set_hf_env(hf_endpoint: str, hf_cache_dir: str) -> None:
    endpoint = str(hf_endpoint).strip()
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint
    cache_dir = str(hf_cache_dir).strip()
    if cache_dir:
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir


def choose_indices(total: int, k: int, seed: int) -> List[int]:
    if total <= 0 or k <= 0:
        return []
    if k >= total:
        return list(range(total))
    rng = random.Random(seed)
    idx = list(range(total))
    rng.shuffle(idx)
    return idx[:k]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_hotpot_user_prompt(question: str, dossier: Sequence[str]) -> str:
    dossier_text = "\n\n".join(dossier)
    return (
        "You are given a long multi-hop QA dossier. "
        "Use the evidence to answer the question.\n\n"
        f"Question:\n{question}\n\n"
        f"Dossier:\n{dossier_text}\n\n"
        "First provide concise reasoning from evidence, then output `Final answer:`."
    )


def build_hotpot_assistant_answer(answer: str, evidence_lines: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append("Reasoning:")
    if evidence_lines:
        for i, line in enumerate(evidence_lines[:6], start=1):
            lines.append(f"{i}. {line}")
    else:
        lines.append("1. The dossier contains multiple linked facts supporting the answer.")
        lines.append("2. Combining those facts yields a single consistent entity or phrase.")
    lines.append(f"Final answer: {answer}")
    return "\n".join(lines)


def compact_sentences(sents: Sequence[str], max_sent: int) -> str:
    clean = [str(s).strip() for s in sents if str(s).strip()]
    return " ".join(clean[: max(1, int(max_sent))]).strip()


def build_hotpot_rows(ds, indices: Iterable[int], max_passages: int, max_sent_per_passage: int) -> List[Dict]:
    rows: List[Dict] = []
    for idx in indices:
        x = ds[int(idx)]
        q = str(x.get("question", "")).strip()
        a = str(x.get("answer", "")).strip()
        if not q or not a:
            continue

        context = x.get("context", {}) or {}
        titles = list(context.get("title", []) or [])
        sentences_per_title = list(context.get("sentences", []) or [])

        title_to_sents: Dict[str, List[str]] = {}
        for t, sents in zip(titles, sentences_per_title):
            title_to_sents[str(t)] = [str(s).strip() for s in (sents or []) if str(s).strip()]

        dossier: List[str] = []
        for t, sents in zip(titles[: max(1, int(max_passages))], sentences_per_title[: max(1, int(max_passages))]):
            txt = compact_sentences(sents or [], max_sent=max_sent_per_passage)
            if txt:
                dossier.append(f"[{t}] {txt}")

        if not dossier:
            continue

        supporting = x.get("supporting_facts", {}) or {}
        sf_titles = list(supporting.get("title", []) or [])
        sf_ids = list(supporting.get("sent_id", []) or [])

        evidence_lines: List[str] = []
        for t, sid in zip(sf_titles, sf_ids):
            title = str(t)
            sent_idx = int(sid) if isinstance(sid, int) else -1
            sents = title_to_sents.get(title, [])
            if 0 <= sent_idx < len(sents):
                evidence_lines.append(f"From [{title}]: {sents[sent_idx]}")

        user_text = build_hotpot_user_prompt(question=q, dossier=dossier)
        assistant_text = build_hotpot_assistant_answer(answer=a, evidence_lines=evidence_lines)

        rows.append(
            {
                "source_name": "hotpotqa_fullwiki",
                "task_type": "multihop_qa_trace",
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                "meta": {"dataset": "hotpotqa/hotpot_qa", "index": int(idx)},
            }
        )
    return rows


def build_alpaca_rows(ds, indices: Iterable[int]) -> List[Dict]:
    rows: List[Dict] = []
    for idx in indices:
        x = ds[int(idx)]
        instruction = str(x.get("instruction", "")).strip()
        input_text = str(x.get("input", "")).strip()
        output_text = str(x.get("output", "")).strip()
        if not instruction or not output_text:
            continue

        if input_text:
            user_text = f"{instruction}\n\nInput:\n{input_text}"
        else:
            user_text = instruction

        rows.append(
            {
                "source_name": "alpaca_cleaned",
                "task_type": "assistant_alignment",
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": output_text},
                ],
                "meta": {"dataset": "yahma/alpaca-cleaned", "index": int(idx)},
            }
        )
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_hf_env(args.hf_endpoint, args.hf_cache_dir)

    from datasets import load_dataset  # delayed import after env setup

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hotpot_path = out_dir / "hotpotqa_multihop.jsonl"
    alpaca_path = out_dir / "alpaca_scaffold.jsonl"
    manifest_path = out_dir / "source_manifest.json"

    if not args.force:
        for p in [hotpot_path, alpaca_path, manifest_path]:
            if p.exists():
                raise FileExistsError(f"{p.as_posix()} exists. Pass --force to overwrite.")

    ds_hotpot = load_dataset(args.hotpot_dataset, args.hotpot_config, split=args.hotpot_split)
    idx_hotpot = choose_indices(total=len(ds_hotpot), k=int(args.hotpot_max_samples), seed=int(args.seed))
    hotpot_rows = build_hotpot_rows(
        ds=ds_hotpot,
        indices=idx_hotpot,
        max_passages=int(args.hotpot_max_passages),
        max_sent_per_passage=int(args.hotpot_max_sent_per_passage),
    )

    ds_alpaca = load_dataset(args.alpaca_dataset, split=args.alpaca_split)
    idx_alpaca = choose_indices(total=len(ds_alpaca), k=int(args.alpaca_max_samples), seed=int(args.seed) + 17)
    alpaca_rows = build_alpaca_rows(ds=ds_alpaca, indices=idx_alpaca)

    write_jsonl(hotpot_path, hotpot_rows)
    write_jsonl(alpaca_path, alpaca_rows)

    manifest = {
        "generated_by": "scripts/prepare_external_longinst_sources.py",
        "seed": int(args.seed),
        "hf_endpoint": str(args.hf_endpoint),
        "hotpot": {
            "dataset": args.hotpot_dataset,
            "config": args.hotpot_config,
            "split": args.hotpot_split,
            "requested": int(args.hotpot_max_samples),
            "written": len(hotpot_rows),
            "path": hotpot_path.as_posix(),
            "sha256": sha256_file(hotpot_path),
        },
        "alpaca": {
            "dataset": args.alpaca_dataset,
            "split": args.alpaca_split,
            "requested": int(args.alpaca_max_samples),
            "written": len(alpaca_rows),
            "path": alpaca_path.as_posix(),
            "sha256": sha256_file(alpaca_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "hotpot_rows": len(hotpot_rows),
        "alpaca_rows": len(alpaca_rows),
        "hotpot_path": hotpot_path.as_posix(),
        "alpaca_path": alpaca_path.as_posix(),
        "manifest": manifest_path.as_posix(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
