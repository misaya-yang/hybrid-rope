#!/usr/bin/env python3
"""Build source-separated Llama-3-8B recovery data without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.evq_recovery.acquire import file_hash
from experiments.evq_recovery.data import chat_ids, sha_text, supervised_chat, write_json
from experiments.evq_recovery.prepare import (
    prepare_pg19, prepare_qasper, prepare_sft, screen_sft_files,
)


REQUIRED_SOURCES = (
    "pg19_books.json", "longalign.jsonl", "qasper-train-dev.tgz",
    "qasper-test.tgz", "native_rows.jsonl",
)


def add_kl_positions(path: Path) -> dict:
    """Freeze up to 128 teacher-logit positions without touching answer labels."""
    temporary = path.with_suffix(".kl.incomplete")
    rows = positions = 0
    with path.open() as source, temporary.open("x") as target:
        for line in source:
            row = json.loads(line)
            length = len(row["input_ids"])
            if row.get("task") == "text":
                count = min(128, length - 1)
                selected = np.linspace(0, length - 2, num=count, dtype=np.int64).tolist()
            else:
                upper = int(row["target_start"]) - 2
                if upper < 0:
                    raise ValueError("replay prompt has no non-answer prediction position")
                count = min(128, upper + 1)
                selected = list(range(upper - count + 1, upper + 1))
            if len(selected) != len(set(selected)) or not all(0 <= p < length - 1 for p in selected):
                raise AssertionError("invalid KL position construction")
            if row.get("task") != "text" and not all(p + 1 < row["target_start"] for p in selected):
                raise AssertionError("KL position leaked into gold answer targets")
            row["kl_positions"] = selected
            target.write(json.dumps(row) + "\n")
            rows += 1
            positions += len(selected)
    temporary.replace(path)
    return {"rows": rows, "positions": positions,
            "semantics": "0-based hidden/logit p predicts input_ids[p+1]; task rows use prompt-only positions"}


def prepare_native_converted(sources: Path, output: Path, source_tokenizer, tokenizer) -> dict:
    """Decode historical OLMo ids with their tokenizer, then retokenize the text."""
    from collections import Counter, defaultdict
    counts, groups = Counter(), defaultdict(set)
    handles = {split: (output / f"native_{split}.jsonl").open("x") for split in ("train", "dev", "test")}
    try:
        for line in (sources / "native_rows.jsonl").open():
            old = json.loads(line)
            split = {"train": "train", "validation": "dev", "test": "test"}.get(old["split"])
            if split is None:
                continue
            groups[split].add(old["source_id"])
            if old["group"] == "text":
                text = source_tokenizer.decode(old["input_ids"], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) < 2:
                    raise ValueError("retokenized replay text is empty")
                item = {"input_ids": ids[:4096], "target_start": 1, "prompt_tokens": 0,
                        "answer_tokens": min(len(ids), 4096) - 1, "prompt_ids": [], "references": []}
            else:
                rendered = source_tokenizer.decode(old["prompt_ids"], skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)
                prefix, suffix = "<|user|>\n", "\n<|assistant|>\n"
                if rendered.startswith(source_tokenizer.bos_token or "\0"):
                    rendered = rendered[len(source_tokenizer.bos_token):]
                if not rendered.startswith(prefix) or not rendered.endswith(suffix):
                    raise ValueError("unknown historical OLMo replay template")
                question = rendered[len(prefix):-len(suffix)]
                references = old.get("accepted_full_answers", [])
                if not references:
                    raise ValueError("verified replay answer missing")
                item = supervised_chat(tokenizer, question, references[0])
                item["prompt_ids"] = item["input_ids"][:item["target_start"]]
                item["references"] = references
            if len(item["input_ids"]) > 4096:
                counts["over_4096_after_target_retokenization"] += 1
                continue
            item.update(id=old["id"], source_id=old["source_id"], split=split, task=old["group"],
                        generation_budget=old.get("generation_budget", 0),
                        provenance="historical source-separated replay decoded with source tokenizer and retokenized with target tokenizer")
            handles[split].write(json.dumps(item) + "\n")
            counts[f"{split}_{old['group']}"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    if any(groups[a] & groups[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Native replay source split overlap")
    return dict(counts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True,
                   help="target OLMo or Llama checkpoint directory; tokenizer/config only are read")
    p.add_argument("--source-tokenizer", type=Path, required=True,
                   help="original OLMo tokenizer used by native_rows.jsonl")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sources, model, output = args.sources.resolve(), args.model.resolve(), args.output.resolve()
    missing = [str(sources / name) for name in REQUIRED_SOURCES if not (sources / name).exists()]
    if missing:
        raise FileNotFoundError("missing real source assets; no substitute pool will be generated: " + ", ".join(missing))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    if config.model_type not in {"llama", "olmo2"}:
        raise ValueError("target must be the selected OLMo or Llama recovery checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer.resolve(), local_files_only=True)
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError("target tokenizer lacks native EOS/chat template")
    terminal_probe = chat_ids(tokenizer, [{"role": "user", "content": "x"},
                                           {"role": "assistant", "content": "y"}])[-1]
    if terminal_probe != tokenizer.eos_token_id:
        raise ValueError(f"assistant terminal token {terminal_probe} differs from configured eos {tokenizer.eos_token_id}")

    denied, qasper = prepare_qasper(sources, output, tokenizer)
    pg19 = prepare_pg19(sources, output, tokenizer)
    sft = prepare_sft(sources, output, tokenizer, denied)
    sft.update(screen_sft_files(output, tokenizer, denied))
    native = prepare_native_converted(sources, output, source_tokenizer, tokenizer)
    kl = add_kl_positions(output / "native_train.jsonl")
    manifest = {
        "schema_version": 1, "status": "CPU_DATA_READY_GPU_NOT_RUN",
        "model_type": config.model_type, "model_path": str(model),
        "tokenizer_sha256": file_hash(model / "tokenizer.json"),
        "source_tokenizer_sha256": file_hash(args.source_tokenizer.resolve() / "tokenizer.json"),
        "tokenizer_chat_template_sha256": sha_text(tokenizer.chat_template),
        "eos_token_id": tokenizer.eos_token_id,
        "assistant_terminal_id": terminal_probe,
        "sources": {name: file_hash(sources / name) for name in REQUIRED_SOURCES},
        "qasper": qasper, "pg19": pg19, "sft": sft, "native": native,
        "native_teacher_kl_positions": kl,
        "native_teacher_kl_policy": "Text uses the frozen positions above. Instruction KL regenerates frozen Native greedy answer/termination prefixes at runtime; stored prompt positions are not its supervision positions.",
        "loss_roles": {
            "cpt": "dense causal CE on every real contiguous PG19 target token",
            "sft": "assistant answer plus native EOS CE on intact LongAlign examples",
            "replay": "short verified reference CE; separately mean-normalized and logged",
        },
        "split_policy": "PG19 official splits; QASPER official document splits; LongAlign and replay source-group disjoint splits",
        "evaluation": "QASPER full-response F1/exact/EOS plus PG19 whole and tail NLL; no answer substring scoring",
        "limitations": [
            "LongAlign assistant labels are synthetic and do not establish professional-domain truth.",
            "Historical Native replay is source-separated but not newly blind.",
            "This preparation does not establish that FFN LoRA is necessary.",
        ],
    }
    manifest["files"] = {path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
                         for path in output.iterdir() if path.is_file()}
    def dataset_entry(name: str, rows: int) -> dict:
        path = output / name
        return {"path": str(path), "sha256": file_hash(path), "rows": int(rows)}
    manifest["cpt_train"] = dataset_entry("cpt_train.npy", pg19["train_windows"])
    manifest["sft_train"] = dataset_entry(
        "sft_train.jsonl", sum(value for key, value in sft.items() if key.startswith("train_") and "removed" not in key))
    manifest["replay_train"] = dataset_entry(
        "native_train.jsonl", sum(value for key, value in native.items() if key.startswith("train_")))
    manifest["replay_train"]["identity"] = "alias of source-separated native_train.jsonl; replay CE and optional teacher KL are distinct losses"
    write_json(output / "data_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "qasper": qasper, "pg19": pg19,
                      "sft": sft, "native": native}, sort_keys=True))


if __name__ == "__main__":
    main()
