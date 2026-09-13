#!/usr/bin/env python3
"""Build a source-separated v2 public SFT pool without touching current data.

Selected families are explicit and may be prepared independently. Missing
selected sources are recorded as pending; the manifest never labels a partial
pool complete. LongAlign and LongAlpaca contribute intact 8K/16K prompt rows.
UltraChat contributes intact <=4K multi-turn replay whose final assistant turn,
including the native terminal token, is supervised.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.evq_recovery.data import chat_ids, write_json
from experiments.evq_recovery.prepare import qasper_context, shingles, tar_json


FAMILIES = ("longalign", "longalpaca", "ultrachat")
SPLITS = ("train", "dev", "test")
LONG_BUCKETS = (8192, 16384)
SHORT_CAP = 4096
_WORKER_TOKENIZER = None
_WORKER_SHORT_CAP = SHORT_CAP


def stable_id(value: str) -> str:
    # Matches the existing LongAlign source-group rule used by v1 preparation.
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_split(group: str) -> str:
    value = int(stable_id(group)[:8], 16) % 10
    return "train" if value < 8 else "dev" if value == 8 else "test"


def legacy_longalign_split(group_material: str) -> str:
    """Preserve v1 exactly: hash%10 0=dev, 1=test, everything else=train."""
    value = int(stable_id(group_material)[:8], 16) % 10
    return "dev" if value == 0 else "test" if value == 1 else "train"


def english_target(text: str) -> bool:
    return bool(text) and len(re.findall("[\u4e00-\u9fff]", text)) <= 0.01 * len(text)


def source_path(primary: Path, legacy: Path | None, name: str) -> Path | None:
    for root in (primary, legacy):
        if root is not None and (root / name).is_file():
            return root / name
    return None


def qasper_denied(primary: Path, legacy: Path | None) -> tuple[set, list[str]]:
    denied = set()
    missing = []
    for name in ("qasper-train-dev.tgz", "qasper-test.tgz"):
        path = source_path(primary, legacy, name)
        if path is None:
            missing.append(name)
            continue
        for member, papers in tar_json(path):
            if "train" in Path(member).name:
                continue
            for paper in papers.values():
                if isinstance(paper, dict) and "full_text" in paper:
                    denied.update(shingles(qasper_context(paper)))
    return denied, missing


def complete_chat(tokenizer, messages: list[dict]) -> dict:
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("conversation must end in an assistant turn")
    answer = str(messages[-1].get("content", "")).strip()
    if not answer:
        raise ValueError("empty final assistant answer")
    prompt = chat_ids(tokenizer, messages[:-1], generation=True)
    full = chat_ids(tokenizer, messages, generation=False)
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("native chat prompt is not an exact prefix of the completed conversation")
    if full[-1] != tokenizer.eos_token_id:
        raise ValueError("completed assistant turn lacks configured native EOS")
    return {"input_ids": full, "target_start": len(prompt), "prompt_ids": prompt,
            "prompt_tokens": len(prompt), "answer_tokens": len(full) - len(prompt),
            "references": [answer], "assistant_turns": sum(m.get("role") == "assistant" for m in messages)}


def init_ultrachat_worker(model_path: str, short_cap: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_SHORT_CAP
    from transformers import AutoTokenizer
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    _WORKER_SHORT_CAP = int(short_cap)


def tokenize_ultrachat(raw: dict):
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("UltraChat worker tokenizer was not initialized")
    prompt_id = str(raw.get("prompt_id", "")).strip()
    if not prompt_id:
        return "missing_prompt_id", None
    try:
        row = complete_chat(_WORKER_TOKENIZER, raw.get("messages") or [])
    except ValueError:
        return "invalid_chat", None
    if len(row["input_ids"]) > _WORKER_SHORT_CAP:
        return f"over_{_WORKER_SHORT_CAP}_intact", None
    return "selected", {"prompt_id": prompt_id, "row": row}


def long_bucket(row: dict) -> int | None:
    prompt, total = row["prompt_tokens"], len(row["input_ids"])
    for bucket in LONG_BUCKETS:
        if prompt >= math.ceil(0.75 * bucket) and total <= bucket:
            return bucket
    return None


class Outputs:
    def __init__(self, root: Path, long_buckets=LONG_BUCKETS):
        self.long_buckets = tuple(long_buckets)
        self.handles = {}
        for split in SPLITS:
            self.handles["native", split] = (root / f"native_short_sft_{split}.jsonl").open("x")
            for bucket in self.long_buckets:
                self.handles[bucket, split] = (root / f"long_sft_{bucket}_{split}.jsonl").open("x")
        self.counts = Counter(); self.tokens = Counter(); self.targets = Counter()
        self.lengths = defaultdict(lambda: [None, 0]); self.seen_rows = set(); self.group_splits = {}

    def add(self, family: str, split: str, bucket, group: str, row: dict) -> bool:
        if self.group_splits.setdefault(group, split) != split:
            raise ValueError("one source group crossed splits")
        row_id = row["id"]
        if row_id in self.seen_rows:
            self.counts[f"{family}_duplicate_skipped"] += 1
            return False
        self.seen_rows.add(row_id)
        row.update(family=family, split=split, source_id=group,
                   input_tokens=len(row["input_ids"]), supervised_tokens=len(row["input_ids"]) - row["target_start"])
        self.handles[bucket, split].write(json.dumps(row) + "\n")
        key = f"{family}/{bucket}/{split}"
        self.counts[key] += 1; self.tokens[key] += row["input_tokens"]; self.targets[key] += row["supervised_tokens"]
        bounds = self.lengths[key]; bounds[0] = row["input_tokens"] if bounds[0] is None else min(bounds[0], row["input_tokens"]); bounds[1] = max(bounds[1], row["input_tokens"])
        return True

    def close(self):
        for handle in self.handles.values(): handle.close()

    def receipt(self):
        total = sum(self.tokens.values())
        training = {key: value for key, value in self.tokens.items() if key.endswith("/train")}
        family_training = defaultdict(int)
        for key, value in training.items(): family_training[key.split("/", 1)[0]] += value
        train_total = sum(training.values())
        return {"rows": dict(self.counts), "unique_input_tokens": dict(self.tokens),
                "supervised_answer_tokens": dict(self.targets), "input_token_ranges": dict(self.lengths),
                "total_unique_input_tokens_all_splits": total,
                "train_only_input_tokens": train_total, "two_pass_train_input_tokens": 2 * train_total,
                "train_input_tokens_by_family": dict(family_training),
                "token_counting_semantics": "Sum over deduplicated selected examples. This is not a claim that shared text spans contain fully unique information.",
                "unique_rows": len(self.seen_rows)}


def prepare_longalign(path: Path, tokenizer, denied: set, output: Outputs):
    counts = Counter()
    with path.open() as stream:
        for index, line in enumerate(stream):
            if (index + 1) % 500 == 0:
                print(json.dumps({"family": "longalign", "rows_read": index + 1,
                                  "selected": sum(value for key, value in output.counts.items() if key.startswith("longalign/"))}), flush=True)
            raw = json.loads(line); messages = raw.get("messages", [])
            if len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
                counts["unsupported_conversation"] += 1; continue
            question = str(messages[0].get("content", ""))
            if not english_target(question): counts["non_english_over_1pct_cjk"] += 1; continue
            if len(shingles(question, stride=1) & denied) >= 3:
                counts["qasper_overlap"] += 1; continue
            group_material = " ".join(question.split())[:8192]
            group = "longalign:" + stable_id(group_material); split = legacy_longalign_split(group_material)
            try: row = complete_chat(tokenizer, messages)
            except ValueError: counts["invalid_chat"] += 1; continue
            bucket = long_bucket(row)
            if bucket is None: counts["outside_true_long_buckets"] += 1; continue
            row.update(id="longalign:" + str(raw.get("id", stable_id(line))), source_row=index,
                       length_bucket=bucket, provenance="LongAlign-10k synthetic assistant label; intact row, no padding/truncation")
            output.add("longalign", split, bucket, group, row)
    return dict(counts)


def prepare_longalpaca(path: Path, tokenizer, denied: set, output: Outputs):
    counts = Counter(); payload = json.loads(path.read_text())
    if not isinstance(payload, list): raise ValueError("LongAlpaca source must be one JSON array")
    for index, raw in enumerate(payload):
        if (index + 1) % 500 == 0:
            print(json.dumps({"family": "longalpaca", "rows_read": index + 1,
                              "selected": sum(value for key, value in output.counts.items() if key.startswith("longalpaca/"))}), flush=True)
        instruction, context, answer = (str(raw.get(name, "")).strip() for name in ("instruction", "input", "output"))
        if not instruction or not answer: counts["missing_instruction_or_output"] += 1; continue
        question = instruction if not context else instruction + "\n\nInput:\n" + context
        if not english_target(question): counts["non_english_over_1pct_cjk"] += 1; continue
        if len(shingles(question, stride=1) & denied) >= 3: counts["qasper_overlap"] += 1; continue
        group_material = " ".join((context if context else question).split())[:8192]
        group = "longalpaca:" + stable_id(group_material); split = stable_split(group_material)
        try: row = complete_chat(tokenizer, [{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        except ValueError: counts["invalid_chat"] += 1; continue
        bucket = long_bucket(row)
        if bucket is None: counts["outside_true_long_buckets"] += 1; continue
        row.update(id="longalpaca:" + stable_id(json.dumps(raw, sort_keys=True)), source_row=index,
                   length_bucket=bucket, provenance="Yukang LongAlpaca original instruction/input/output; intact row, no padding/truncation")
        output.add("longalpaca", split, bucket, group, row)
    return dict(counts)


def prepare_ultrachat(paths: list[Path], model: Path, output: Outputs, workers: int, short_cap: int):
    import pyarrow.parquet as pq
    counts = Counter()
    read = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=init_ultrachat_worker,
                             initargs=(str(model.resolve()), short_cap)) as pool:
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(columns=["messages", "prompt_id"], batch_size=128):
                # At most one 128-row batch is in flight. Reading results in
                # submission order preserves the original deterministic order.
                futures = [pool.submit(tokenize_ultrachat, raw) for raw in batch.to_pylist()]
                for future in futures:
                    reason, payload = future.result(); read += 1
                    if reason != "selected":
                        counts[reason] += 1
                    else:
                        prompt_id, row = payload["prompt_id"], payload["row"]
                        group = "ultrachat:" + prompt_id; split = stable_split(prompt_id)
                        row.update(id=group, length_bucket=short_cap,
                                   provenance="HuggingFaceH4 ultrachat_200k train_sft full conversation; final assistant turn supervised")
                        output.add("ultrachat", split, "native", group, row)
                    if read % 500 == 0:
                        print(json.dumps({"family": "ultrachat", "rows_read": read,
                                          "selected": sum(value for key, value in output.counts.items() if key.startswith("ultrachat/"))}), flush=True)
    return dict(counts)


def main():
    global LONG_BUCKETS, SHORT_CAP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True); parser.add_argument("--legacy-sources", type=Path)
    parser.add_argument("--model", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--long-buckets", type=int, nargs=2, default=list(LONG_BUCKETS))
    parser.add_argument("--short-cap", type=int, default=SHORT_CAP)
    parser.add_argument("--workers", type=int, default=4,
                        help="UltraChat tokenizer processes (1-20); other families stay sequential")
    args = parser.parse_args(); sources = args.sources.resolve(); legacy = args.legacy_sources.resolve() if args.legacy_sources else None
    LONG_BUCKETS = tuple(args.long_buckets); SHORT_CAP = int(args.short_cap)
    if tuple(sorted(set(LONG_BUCKETS))) != LONG_BUCKETS or SHORT_CAP <= 0 or SHORT_CAP >= LONG_BUCKETS[0]:
        raise ValueError("require positive short cap below two strictly increasing long buckets")
    if len(set(args.families)) != len(args.families): raise ValueError("duplicate family selection")
    if not 1 <= args.workers <= 20: raise ValueError("--workers must be between 1 and 20")
    if args.output.exists(): raise FileExistsError(args.output)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    if tokenizer.eos_token_id is None or not tokenizer.chat_template: raise ValueError("target tokenizer lacks native chat/EOS contract")
    denied, missing_qasper = qasper_denied(sources, legacy)
    pending = []
    if missing_qasper and any(name in args.families for name in ("longalign", "longalpaca")):
        pending.extend("QASPER exclusion source: " + name for name in missing_qasper)
    args.output.mkdir(parents=True); output = Outputs(args.output, LONG_BUCKETS); diagnostics = {}
    try:
        if "longalign" in args.families:
            path = source_path(sources, legacy, "longalign.jsonl")
            if path is None or missing_qasper: pending.append("longalign.jsonl or complete QASPER exclusion archives")
            else: diagnostics["longalign"] = prepare_longalign(path, tokenizer, denied, output)
        if "longalpaca" in args.families:
            path = source_path(sources, legacy, "longalpaca.json")
            if path is None or missing_qasper: pending.append("longalpaca.json or complete QASPER exclusion archives")
            else: diagnostics["longalpaca"] = prepare_longalpaca(path, tokenizer, denied, output)
        if "ultrachat" in args.families:
            paths = sorted(sources.glob("ultrachat_train_sft*.parquet"))
            if len(paths) != 3: pending.append(f"all 3 ultrachat_train_sft parquet shards (found {len(paths)})")
            else: diagnostics["ultrachat"] = prepare_ultrachat(paths, args.model, output, args.workers, SHORT_CAP)
    finally: output.close()
    receipt = output.receipt(); complete = not pending and set(diagnostics) == set(args.families)
    manifest = {"status": "COMPLETE" if complete else "PARTIAL_PENDING_SOURCES",
        "selected_families": args.families, "completed_families": sorted(diagnostics), "pending": sorted(set(pending)),
        "asset_identity_policy": "user_attested_clone/no_sha_validation", "model": str(args.model.resolve()),
        "split_policy": "stable source-level 80/10/10; all variants from one source group remain in one split",
        "long_contract": f"prompt>=75% of {LONG_BUCKETS} and intact prompt+answer+EOS<=bucket; no padding/truncation",
        "short_contract": f"complete UltraChat conversation<={SHORT_CAP}; all previous turns prompt the final supervised assistant+EOS",
        "ultrachat_workers": args.workers,
        "language_policy": "LongAlign and LongAlpaca retain the existing English target: CJK characters <=1% of question characters; exclusions are counted.",
        "qasper_exclusion": "32-word shingles from QASPER dev/test documents; training member excluded",
        "diagnostics": diagnostics, **receipt,
        "limitations": ["LongAlign/LongAlpaca synthetic answers retain their published provenance.",
                        "Prompt length does not prove decisive-evidence distance.",
                        "No unrelated short examples are concatenated; no NIAH evaluation row is training data."]}
    write_json(args.output / "manifest.json", manifest); print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__": main()
