"""CPU-only PM-Keep data: complete context, compress, then unseen question.

Run as ``python -m experiments.pm_keep.prepare``. Natural rows are explicitly
context-first LongBench variants, scored with the source English QA metric;
they are not original LongBench protocol scores. No document is truncated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
TASKS = ("single_kv", "multi_kv_order", "hotpotqa", "2wikimqa")
SEEDS = {"dev": 202609091, "test": 202609097}
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
EXACT = "literal_full_string_plus_terminal_eos_v1"
QA = "longbench_qa_f1_context_first_v1"


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def render_split(tokenizer, context, question, answer_instruction):
    before = "Read and remember the complete context below.\n\nCONTEXT\n" + context + "\nEND CONTEXT"
    after = "\n\nQuestion: " + question + "\n" + answer_instruction
    messages = [{"role": "user", "content": before + after}]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    content_start = full.index(before)
    boundary_char = content_start + len(before)
    question_char = boundary_char + len("\n\nQuestion: ")
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    # A BPE token may straddle the intended boundary. In that case it belongs
    # entirely to the future suffix; never let a question token enter the prefix.
    boundary = next((i for i, (_, end) in enumerate(offsets) if end > boundary_char), len(ids))
    prefix_ids, suffix_ids = ids[:boundary], ids[boundary:]
    prefix = tokenizer.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    suffix = tokenizer.decode(suffix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not prefix_ids or not suffix_ids or prefix + suffix != full:
        raise ValueError("token boundary is not a lossless complete-text split")
    if any(end > boundary_char for _, end in offsets[:boundary]):
        raise ValueError("prefix contains future suffix characters")
    if question in prefix:
        raise ValueError("future question leaked into prefix")
    if context not in prefix:
        raise ValueError("prefix does not preserve the complete context")
    return dict(prompt_ids=ids, prefix_ids=prefix_ids, suffix_ids=suffix_ids,
                prefix_length=boundary, prefix_tokens=boundary, input_tokens=len(ids),
                suffix_tokens=len(suffix_ids), full_prompt=full, prompt=full,
                prefix_text=prefix, suffix_text=suffix, raw_messages=messages,
                raw_context=context, raw_question=question,
                boundary_char=boundary_char, first_question_char=question_char,
                last_prefix_offset_end=max(end for _, end in offsets[:boundary]),
                prompt_sha256=sha_text(full), context_sha256=sha_text(context),
                prompt_ids_sha256=sha_text(json.dumps(ids, separators=(",", ":"))),
                prefix_ids_sha256=sha_text(json.dumps(prefix_ids, separators=(",", ":"))),
                future_question_visible_to_scorer=False,
                protocol="single_user_context_first_unseen_query_v1")


def synthetic_material(task, split, index):
    seed = SEEDS[split] * 1000 + index + (100000 if task == "multi_kv_order" else 0)
    rng = random.Random(seed)
    namespace = "d" if split == "dev" else "t"
    used = set()

    def word(n):
        while True:
            value = namespace + "".join(rng.choices("abcdefghjkmnpqrstuvwxyz", k=n))
            if value not in used:
                used.add(value)
                return value

    if task == "single_kv":
        key, value = word(7), word(9)
        records = [(key, value)]
        depth = (0.10, 0.35, 0.60, 0.85)[index % 4]
        positions = [depth]
        question = f"What value is stored for key {key}?"
        expected = value
        query = {"keys": [key], "ordinals": [1]}
    else:
        keys = [word(7), word(7)]
        # Two keys each have two chronological occurrences. Eight other keys
        # provide the same record syntax, without leaking which keys are queried.
        targets = [(keys[0], word(9)), (keys[1], word(9)),
                   (keys[0], word(9)), (keys[1], word(9))]
        slots = sorted(rng.sample(range(12), 4))
        records, cursor = [], 0
        for slot in range(12):
            if slot in slots:
                records.append(targets[cursor])
                cursor += 1
            else:
                records.append((word(7), word(9)))
        positions = [(i + 1) / 13 for i in range(12)]
        query = {"keys": keys, "ordinals": [-1, 1]}
        question = (f"In written chronological order, what is the latest value for key {keys[0]}, "
                    f"and what is the first value for key {keys[1]}? Return the answers in that order.")
        expected = targets[2][1] + "," + targets[1][1]
    return dict(seed=seed, records=records, positions=positions, question=question,
                expected=expected, query=query)


def synthetic_context(material, noise_count):
    blocks, consumed = [], 0
    for position, (key, value) in zip(material["positions"], material["records"]):
        next_count = int(position * noise_count)
        blocks.extend([NOISE] * (next_count - consumed))
        blocks.append(f"Record key={key}; value={value}.")
        consumed = next_count
    blocks.extend([NOISE] * (noise_count - consumed))
    return "\n".join(blocks)


def make_synthetic(tokenizer, task, split, index, target):
    material = synthetic_material(task, split, index)
    instruction = ("Output exactly the value string and then stop. No explanation, punctuation, or extra spaces."
                   if task == "single_kv" else
                   "Output exactly the two value strings separated by one comma, with no spaces, explanation, or other text.")
    low, high, best = 0, target // 15, None
    while low <= high:
        mid = (low + high) // 2
        result = render_split(tokenizer, synthetic_context(material, mid), material["question"], instruction)
        if result["prefix_tokens"] <= target:
            best, low = (mid, result), mid + 1
        else:
            high = mid - 1
    if best is None or best[1]["prefix_tokens"] < target - 100:
        raise ValueError("synthetic prefix did not reach the declared length interval")
    noise_count, result = best
    row_id = f"{task}_{split}_{index:03d}"
    result.update(row_id=row_id, task=task, split=split, doc_id=row_id,
                  material_cluster_id=row_id, seed=material["seed"],
                  expected=material["expected"], references=[material["expected"]],
                  score_contract=EXACT, max_new_tokens=64, prefix_target=target,
                  records=[{"key": k, "value": v} for k, v in material["records"]],
                  query=material["query"], noise_sentence_count=noise_count,
                  answer_depth_fraction=material["positions"], constituent_doc_ids=[row_id],
                  source={"kind": "local_fixed_synthetic_generator", "generator": "experiments/pm_keep/prepare.py",
                          "related_existing_generator": "experiments/nosa_position/prepare.py",
                          "official_benchmark": False})
    return result


def constituent_ids(context):
    # LongBench retrieval contexts contain Passage N headers and document titles.
    titles = re.findall(r"(?:^|\n)Passage \d+:\s*\n([^\n]+)", context)
    return sorted({"title:" + sha_text(" ".join(title.casefold().split())) for title in titles}) or ["context:" + sha_text(context)]


def natural_candidates(tokenizer, source_dir, max_prefix, min_prefix=1040):
    candidates, source_receipts = {}, {}
    acquisition_path = source_dir / "source_receipt.json"
    acquisition = json.loads(acquisition_path.read_text()) if acquisition_path.exists() else {}
    for task in ("hotpotqa", "2wikimqa"):
        path = source_dir / f"{task}.jsonl"
        source_receipts[task] = {"path": str(path.resolve()), "sha256": sha_file(path),
                                 "dataset_revision": acquisition.get("revision")}
        eligible, seen, excluded_length = [], set(), 0
        for ordinal, source in enumerate(read_jsonl(path)):
            context, question = source["context"], source["input"]
            context_hash = sha_text(context)
            if context_hash in seen or question in context:
                continue
            seen.add(context_hash)
            result = render_split(tokenizer, context, question,
                                  "Answer concisely using only the context. Output only the answer, then stop.")
            if not min_prefix <= result["prefix_tokens"] <= max_prefix:
                excluded_length += 1
                continue
            source_id = str(source.get("_id", context_hash))
            result.update(task=task, doc_id=source_id, material_cluster_id=context_hash,
                          expected=None, references=source["answers"], score_contract=QA,
                          max_new_tokens=128, constituent_doc_ids=constituent_ids(context),
                          source={"kind": "LongBench", "dataset": "THUDM/LongBench", "task": task,
                                  "dataset_revision": acquisition.get("revision"),
                                  "original_id": source_id, "original_row_index": ordinal,
                                  "raw_row_sha256": sha_text(json.dumps(source, sort_keys=True, ensure_ascii=False)),
                                  "raw_source_sha256": source_receipts[task]["sha256"],
                                  "original_task_protocol": False,
                                  "protocol_change": "Context is complete and precedes the previously unseen question; original question-first prompt is not used."})
            eligible.append(result)
        # Deterministic source-only selection. Prefer the predeclared 4K..12K
        # neighborhood, but allow shorter/longer complete contexts if necessary.
        eligible.sort(key=lambda r: (not 4096 <= r["prefix_tokens"] <= 12288,
                                     sha_text(r["context_sha256"] + ":20260909")))
        candidates[task] = eligible
        source_receipts[task]["eligible_complete_contexts"] = len(eligible)
        source_receipts[task]["excluded_by_prefix_length"] = excluded_length
    return candidates, source_receipts


def select_natural(candidates, dev_count, test_count):
    rows, dev_docs, chosen_contexts = [], set(), set()
    # Select both development tasks before selecting either test task. An entire
    # source context and every detected document title is held out across splits.
    for split, count in (("dev", dev_count), ("test", test_count)):
        for task in ("hotpotqa", "2wikimqa"):
            selected = 0
            for candidate in candidates[task]:
                docs = set(candidate["constituent_doc_ids"])
                if candidate["context_sha256"] in chosen_contexts or (split == "test" and docs & dev_docs):
                    continue
                row = dict(candidate)
                row.update(row_id=f"{task}_{split}_{selected:03d}", split=split)
                rows.append(row)
                chosen_contexts.add(row["context_sha256"])
                if split == "dev":
                    dev_docs.update(docs)
                selected += 1
                if selected == count:
                    break
            if selected != count:
                raise ValueError(f"only {selected}/{count} independent complete {task} {split} contexts; do not truncate or duplicate")
    return rows


def validate(rows, tokenizer, native_limit=32768):
    docs, contexts = {"dev": set(), "test": set()}, {"dev": set(), "test": set()}
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate row IDs")
    for row in rows:
        if row["prefix_ids"] + row["suffix_ids"] != row["prompt_ids"]:
            raise ValueError("prefix/suffix IDs do not reconstruct full input")
        if tokenizer.encode(row["full_prompt"], add_special_tokens=False) != row["prompt_ids"]:
            raise ValueError("complete input tokenization changed")
        if row["prefix_length"] != len(row["prefix_ids"]) or row["prefix_tokens"] != row["prefix_length"]:
            raise ValueError("prefix length mismatch")
        if row["raw_question"] in row["prefix_text"] or row["last_prefix_offset_end"] > row["boundary_char"]:
            raise ValueError("future question leakage")
        if row["raw_context"] not in row["prefix_text"] or row["prefix_text"] + row["suffix_text"] != row["full_prompt"]:
            raise ValueError("context or chat text was cut")
        if len(row["prompt_ids"]) + row["max_new_tokens"] > native_limit:
            raise ValueError("native context budget exceeded")
        if row["prompt_sha256"] != sha_text(row["full_prompt"]):
            raise ValueError("prompt hash changed")
        if row["score_contract"] == EXACT:
            parsed = re.findall(r"Record key=([a-z]+); value=([a-z]+)\.", row["raw_context"])
            answers = []
            for key, ordinal in zip(row["query"]["keys"], row["query"]["ordinals"]):
                found = [v for k, v in parsed if k == key]
                answers.append(found[-1] if ordinal == -1 else found[ordinal - 1])
            if row["expected"] != ",".join(answers) or row["references"] != [row["expected"]]:
                raise ValueError("synthetic answer mapping mismatch")
        elif row["score_contract"] != QA or row["expected"] is not None:
            raise ValueError("invalid natural score contract")
        contexts[row["split"]].add(row["context_sha256"])
        docs[row["split"]].update(row["constituent_doc_ids"])
    if docs["dev"] & docs["test"] or contexts["dev"] & contexts["test"]:
        raise ValueError("document leakage between development and confirmation")
    return {"rows": len(rows), "full_tokenization_verified": True,
            "complete_context_preserved": True, "dev_test_shared_document_ids": 0,
            "future_question_visible_to_scorer": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=Path("results/pm_keep_20260909/source/tokenizer"))
    parser.add_argument("--natural-source", type=Path, default=Path("results/pm_keep_20260909/source/longbench"))
    parser.add_argument("--out", type=Path, default=Path("results/pm_keep_20260909/data"))
    parser.add_argument("--dev-count", type=int, default=16)
    parser.add_argument("--test-count", type=int, default=32)
    parser.add_argument("--prefix-target", type=int, default=8192)
    parser.add_argument("--natural-max-prefix", type=int, default=16384)
    parser.add_argument("--natural-min-prefix", type=int, default=1040,
                        help="1040 ensures floor(.25*T) includes 4 sinks plus 256 recent tokens")
    parser.add_argument("--synthetic-smoke-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    config = json.loads((args.tokenizer / "config.json").read_text())
    generation_path = args.tokenizer / "generation_config.json"
    native_generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    eos_ids = native_generation.get("eos_token_id", [tokenizer.eos_token_id])
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    if config.get("hidden_size") != 2048 or config.get("num_hidden_layers") != 36:
        raise ValueError("requires the copied Qwen2.5-3B model configuration")
    if args.verify_only:
        receipt = json.loads((args.out / "manifest.json").read_text())
        if sha_file(args.out / "rows.jsonl") != receipt["rows_sha256"]:
            raise ValueError("frozen data changed")
        print(json.dumps(validate(read_jsonl(args.out / "rows.jsonl"), tokenizer)))
        return
    if args.out.exists():
        raise FileExistsError(f"refuse to overwrite frozen data {args.out}")
    if not args.synthetic_smoke_only:
        missing = [str(args.natural_source / f"{task}.jsonl") for task in TASKS[2:]
                   if not (args.natural_source / f"{task}.jsonl").is_file()]
        if missing:
            raise FileNotFoundError("natural raw source required before preparation: " + ", ".join(missing))
    if min(args.dev_count, args.test_count, args.prefix_target) <= 0:
        raise ValueError("positive counts and prefix target required")
    work = args.out.with_name(args.out.name + ".building")
    work.mkdir(parents=True, exist_ok=False)
    rows = []
    for task in TASKS[:2]:
        split_counts = [("dev", 2)] if args.synthetic_smoke_only else [("dev", args.dev_count), ("test", args.test_count)]
        for split, count in split_counts:
            rows.extend(make_synthetic(tokenizer, task, split, i, args.prefix_target) for i in range(count))
            print(f"prepared {task} {split}: {count}", flush=True)
    sources = {}
    if not args.synthetic_smoke_only:
        candidates, sources = natural_candidates(tokenizer, args.natural_source, args.natural_max_prefix,
                                                 args.natural_min_prefix)
        rows.extend(select_natural(candidates, args.dev_count, args.test_count))
    checks = validate(rows, tokenizer)
    with (work / "rows.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    smoke = [r for r in rows if r["split"] == "dev" and r["row_id"].endswith(("000", "001"))]
    (work / "smoke.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in smoke))
    shards = {f"{task}/{split}": [r["prefix_tokens"] for r in rows if r["task"] == task and r["split"] == split]
              for task in TASKS for split in ("dev", "test")}
    receipt = dict(status="CPU_FROZEN_SMOKE" if args.synthetic_smoke_only else "CPU_FROZEN_DEV64_TEST128",
                   model_id=MODEL_ID, model_revision=MODEL_REVISION,
                   model_remote_path="/root/autodl-tmp/rope_qwen_baseline_20260907/model",
                   tokenizer_files_sha256={p.name: sha_file(p) for p in args.tokenizer.iterdir() if p.is_file()},
                   chat_template=tokenizer.chat_template, eos_token_ids=eos_ids,
                   native_context_limit=config["max_position_embeddings"], prepare_code_sha256=sha_file(__file__),
                   rows_sha256=sha_file(work / "rows.jsonl"), smoke_sha256=sha_file(work / "smoke.jsonl"),
                   natural_sources=sources, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   checks=checks, split_counts=dict(Counter(r["split"] for r in rows)),
                   shards={k: {"count": len(v), "min_prefix_tokens": min(v), "max_prefix_tokens": max(v)} for k, v in shards.items() if v},
                   protocol="context-first/unseen-query; compaction occurs at prefix_length before every future question token",
                   scores={EXACT: "Entire decoded answer equals expected literally and is followed by one terminal EOS; no substring or whitespace normalization.",
                           QA: "LongBench English QA token F1, max over source references; report as context-first protocol, not original benchmark scores."},
                   caveats=["Synthetic context is length-fitted with whole noise sentences; all factual records remain complete.",
                            "Natural contexts are complete, never padded or truncated; selected by source hashes and document disjointness without model outputs.",
                            "Prepared test inputs do not authorize using test outcomes for tuning."])
    (work / "manifest.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    work.rename(args.out)
    print(json.dumps({"out": str(args.out), "rows_sha256": receipt["rows_sha256"], **checks}), flush=True)


if __name__ == "__main__":
    main()
