"""Freeze new domains and controlled background families without model inference.

Natural source documents are never truncated. Controlled retrieval uses explicitly
labelled prose excerpts. Existing DEV and TEST documents are excluded before
choosing the new splits. The ongoing experiment's data is never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from experiments.pm_keep.prepare import constituent_ids, NOISE, EXACT, QA
from experiments.refcarry_audit.prepare_mrcr_pairs import family as mrcr_family

MRCR = "mrcr_marker_sequence_plus_exact_v1"
NATURAL_TASKS = ("qasper", "multifieldqa_en", "narrativeqa", "hotpotqa", "2wikimqa")
SEED = 2026090927


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def render_split(tokenizer, context, question, instruction):
    """Use offsets from one native tokenization, including normalizing tokenizers.

    The source text stays intact. decode(encode(text)) need not reproduce Unicode
    whitespace for every native tokenizer, so offsets and token IDs own the
    query-blind boundary rather than a lossy decoded string.
    """
    before = "Read and remember the complete context below.\n\nCONTEXT\n" + context + "\nEND CONTEXT"
    after = "\n\nQuestion: " + question + "\n" + instruction
    messages = [{"role": "user", "content": before + after}]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    boundary_char = full.index(before) + len(before)
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    boundary = next((i for i, (_, end) in enumerate(offsets) if end > boundary_char), len(ids))
    if not 0 < boundary < len(ids):
        raise ValueError("empty prefix or question suffix")
    text_boundary = offsets[boundary][0]
    prefix = full[:text_boundary]
    if question in prefix or context not in prefix:
        raise ValueError("future question leaked or context was cut at the boundary")
    return dict(full_prompt=full, prompt=full, prompt_ids=ids, prefix_ids=ids[:boundary],
                suffix_ids=ids[boundary:], prefix_length=boundary, prefix_tokens=boundary,
                input_tokens=len(ids), suffix_tokens=len(ids)-boundary,
                prefix_text=prefix, suffix_text=full[text_boundary:], raw_context=context,
                raw_question=question, raw_messages=messages, boundary_char=boundary_char,
                last_prefix_offset_end=max(end for _, end in offsets[:boundary]),
                context_sha256=sha(context), prompt_sha256=sha(full),
                future_question_visible_to_scorer=False)


def attach(row, rendered, limit):
    out = {**row, **rendered, "length_cap": limit, "suite": "broader_position_panel_v1"}
    out.setdefault("family_id", out["material_cluster_id"])
    if out["input_tokens"] + out["max_new_tokens"] > limit:
        raise ValueError("complete prompt and generation exceed evaluation limit")
    if out["prefix_length"] + 128 > limit:
        raise ValueError("prefix plus frozen PM horizon exceeds evaluation limit")
    return out


def excluded_sources(old_rows):
    contexts, documents = set(), set()
    for row in old_rows:
        if row.get("context_sha256"):
            contexts.add(row["context_sha256"])
        documents.update(row.get("constituent_doc_ids", []))
    return contexts, documents


def prepare_natural(tokenizers, limits, args, output, provenance):
    excluded_contexts, excluded_docs = excluded_sources(read_rows(args.old_pm_data))
    used_contexts, used_docs = set(excluded_contexts), set(excluded_docs)
    candidates = {}
    for task in NATURAL_TASKS:
        folder = args.old_longbench if task in ("hotpotqa", "2wikimqa") else args.new_longbench
        path = folder / (task + ".jsonl")
        provenance[str(path)] = sha(path.read_bytes())
        eligible = []
        for source in read_rows(path):
            context, question = source["context"], source["input"]
            context_id = sha(context)
            docs = set(constituent_ids(context)) | {"context:" + context_id}
            if context_id in excluded_contexts or docs & excluded_docs or question in context:
                continue
            renders = {}
            for model, tok in tokenizers.items():
                view = render_split(tok, context, question,
                                    "Answer concisely using only the context. Output only the answer, then stop.")
                if view["prefix_length"] < 2304 or view["input_tokens"] + 256 > limits[model]:
                    continue
                renders[model] = view
            if "pm" in renders:
                eligible.append((context_id, docs, source, renders))
        candidates[task] = sorted(eligible, key=lambda item: sha(task + item[0]))
        print(f"eligible new {task} source contexts: {len({x[0] for x in eligible})}", flush=True)

    # A source-only allocation rule uses smaller complete pools honestly. In
    # particular, many NarrativeQA questions share a long original story; they
    # cannot be counted as independent new documents or shortened to fill quotas.
    counts = {}
    for task, pool in candidates.items():
        available = len({item[0] for item in pool})
        if available < 2:
            raise ValueError(f"{task}: fewer than two intact new source contexts")
        dev = min(args.dev_count, max(1, available // 3))
        counts[task] = {"dev": dev, "test": min(args.test_count, available - dev)}

    # All development documents are selected before any confirmation documents.
    # Shared passage titles are also excluded across tasks and splits.
    for split in ("dev", "test"):
        for task in NATURAL_TASKS:
            count = counts[task][split]
            selected = 0
            for context_id, docs, source, renders in candidates[task]:
                if context_id in used_contexts or docs & used_docs:
                    continue
                row_id = f"broad_{task}_{split}_{selected:03d}"
                row = dict(row_id=row_id, task="natural_" + task, split=split,
                           material_cluster_id="longbench:" + context_id,
                           doc_id=source.get("_id", context_id), constituent_doc_ids=sorted(docs),
                           expected=None, references=source["answers"], score_contract=QA,
                           max_new_tokens=256, background="unaltered_natural_document",
                           source={"dataset": "THUDM/LongBench", "task": task,
                                   "revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
                                   "row_sha256": sha(json.dumps(source, sort_keys=True)),
                                   "original_benchmark_protocol": False,
                                   "adaptation": "Complete context before unseen question, native model chat template"})
                for model, view in renders.items():
                    output[model].append(attach(row, view, limits[model]))
                used_contexts.add(context_id)
                used_docs.update(docs)
                selected += 1
                if selected == count:
                    break
            if selected == 0:
                raise ValueError(f"{task}/{split}: no unused source document after cross-task exclusion")
            print(f"prepared natural {task} {split}: {selected}/{count}; per-model length coverage recorded separately", flush=True)


def record_bank(split, index, count=256):
    rng = random.Random(SEED + index + (100000 if split == "test" else 0))
    namespace = "bt" if split == "test" else "bd"
    used, bank = set(), []
    while len(bank) < count:
        key = namespace + "".join(rng.choices("abcdefghjkmnpqrstuvwxyz", k=7))
        value = namespace + "".join(rng.choices("abcdefghjkmnpqrstuvwxyz", k=9))
        if key not in used and value not in used:
            used.update((key, value))
            bank.append((key, value))
    return bank


def interleave(background, records, chars):
    text = background[:chars]
    chunks = [text[len(text) * i // (len(records) + 1):len(text) * (i + 1) // (len(records) + 1)]
              for i in range(len(records) + 1)]
    return "\n".join(part for i, chunk in enumerate(chunks)
                     for part in ([chunk, f"Record key={records[i][0]}; value={records[i][1]}."]
                                  if i < len(records) else [chunk]))


def fit_retrieval(background, records, question, tokenizers, target):
    low, high, best = 0, len(background), None
    while low <= high:
        chars = (low + high) // 2
        context = interleave(background, records, chars)
        views = {name: render_split(tok, context, question,
                 "Output exactly the value string and then stop. No explanation, punctuation, or extra spaces.")
                 for name, tok in tokenizers.items()}
        longest = max(view["prefix_length"] for view in views.values())
        if longest <= target:
            best, low = (views, chars), chars + 1
        else:
            high = chars - 1
    if best is None or max(v["prefix_length"] for v in best[0].values()) < target - 128:
        raise ValueError("background cannot fill the specified retrieval length without repetition")
    return best


def prepare_retrieval(tokenizers, limits, args, output, provenance):
    raw = args.prose.read_bytes()
    provenance[str(args.prose)] = sha(raw)
    paragraphs = [p.strip() for p in json.loads(raw)["text"].split("\n\n") if p.strip()]
    # Nonoverlapping source segments across all families and both splits.
    corpus = "\n\n".join(paragraphs)
    total = args.dev_count + args.test_count
    span = len(corpus) // total
    for split, count in (("dev", args.dev_count), ("test", args.test_count)):
        for index in range(count):
            global_index = index + (args.dev_count if split == "test" else 0)
            prose = corpus[global_index * span:(global_index + 1) * span]
            family_id = f"broad_retrieval_{split}_{index:03d}"
            bank = record_bank(split, index)
            key, value = bank[0]
            question = f"What value is stored for key {key}?"
            depth = (0.10, 0.35, 0.60, 0.85)[index % 4]
            for condition, n_records, background in (
                ("repeat_16", 16, (NOISE + "\n") * 2000),
                ("prose_16", 16, prose),
                ("prose_256", 256, prose),
            ):
                records = list(bank[1:n_records])
                records.insert(round(depth * len(records)), bank[0])
                views, chars = fit_retrieval(background, records, question, tokenizers, args.prefix_target)
                row = dict(row_id=family_id + "_" + condition, task="retrieval_" + condition,
                           split=split, family_id=family_id, material_cluster_id=family_id,
                           doc_id=family_id, constituent_doc_ids=[f"prose-segment:{sha(prose)}"],
                           expected=value, references=[value], score_contract=EXACT,
                           max_new_tokens=64, background=condition, record_count=n_records,
                           records=[{"key": k, "value": v} for k, v in records],
                           query={"keys": [key], "ordinals": [1]}, answer_depth_fraction=depth,
                           source={"kind": "controlled_background_transfer",
                                   "background_corpus": "RULER PaulGrahamEssays.json",
                                   "corpus_sha256": sha(raw), "segment_index": global_index,
                                   "excerpt_characters": chars, "original_benchmark_protocol": False,
                                   "scope": "Explicitly excerpted prose and synthetic records; not natural QA"})
                for model in tokenizers:
                    output[model].append(attach(row, views[model], limits[model]))
        print(f"prepared background families {split}: {count} x 3 conditions", flush=True)


def render_messages(tok, record):
    full = tok.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=True)
    prefix = tok.apply_chat_template(record["messages"][:-1], tokenize=False, add_generation_prompt=False)
    ids = tok.encode(full, add_special_tokens=False)
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    if not full.startswith(prefix) or ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("native chat template is not prefix-stable for this conversation")
    if record["messages"][-1]["content"] in prefix:
        raise ValueError("future question already appears in MRCR prefix")
    return dict(full_prompt=full, prompt=full, prompt_ids=ids, prefix_ids=prefix_ids,
                suffix_ids=ids[len(prefix_ids):], prefix_length=len(prefix_ids),
                prefix_tokens=len(prefix_ids), suffix_tokens=len(ids)-len(prefix_ids),
                input_tokens=len(ids), prefix_text=prefix, suffix_text=full[len(prefix):],
                prompt_sha256=sha(full), context_sha256=sha(prefix),
                future_question_visible_to_scorer=False,
                raw_question=record["messages"][-1]["content"])


def prepare_mrcr(tokenizers, limits, args, output, provenance):
    import pyarrow.parquet as pq

    source_hash = sha(args.mrcr_source.read_bytes())
    receipt = json.loads(args.mrcr_receipt.read_text())
    if source_hash != receipt["expected_hub_metadata"]["lfs"]["oid"]:
        raise ValueError("MRCR source LFS hash mismatch")
    provenance[str(args.mrcr_source)] = source_hash
    excluded = {r["source_row_sha256"] for r in read_rows(args.old_mrcr_data)}
    eligible = []
    for index, batch in enumerate(pq.ParquetFile(args.mrcr_source).iter_batches(batch_size=1)):
        source = batch.to_pylist()[0]
        if source["n_chars"] > 120000 or len(source["answer"]) > 1800:
            continue
        group = mrcr_family(source, index)
        if group[0]["source_row_sha256"] in excluded:
            continue
        group = [r for r in group if r["world"] == 0]  # two long queries and their compact controls
        if any(len(tokenizers["pm"].encode(r["references"][0], add_special_tokens=False)) > 448 for r in group):
            continue
        views = {r["row_id"]: render_messages(tokenizers["pm"], r) for r in group}
        if any(v["input_tokens"] + 512 > limits["pm"] for v in views.values()):
            continue
        eligible.append((group[0]["family_id"], group, views))
    eligible.sort(key=lambda item: item[0])
    needed = args.mrcr_dev_families + args.mrcr_test_families
    if len(eligible) < needed:
        raise ValueError(f"only {len(eligible)}/{needed} new intact MRCR source families")
    exclusions = []
    for index, (family_id, group, pm_views) in enumerate(eligible[:needed]):
        split = "dev" if index < args.mrcr_dev_families else "test"
        for record in group:
            variant = "compact" if record["variant"] == "compact_control" else "long"
            row = {**record, "row_id": "broad_mrcr_" + split + "_" + record["row_id"],
                   "task": "mrcr_" + variant, "split": split, "doc_id": family_id,
                   "material_cluster_id": "mrcr:" + family_id,
                   "constituent_doc_ids": ["mrcr:" + family_id],
                   "max_new_tokens": 512, "score_contract": MRCR,
                   "expected": record["references"][0], "background": "public_synthetic_conversation",
                   "source": {"dataset": "openai/mrcr", "parquet_sha256": source_hash,
                              "source_row_sha256": record["source_row_sha256"],
                              "original_benchmark_protocol": False,
                              "scope": "Original/alternate ordinal query pairs and all-needle compact controls; new source conversations"}}
            for model, tok in tokenizers.items():
                view = pm_views[record["row_id"]] if model == "pm" else render_messages(tok, record)
                try:
                    output[model].append(attach(row, view, limits[model]))
                except ValueError as error:
                    exclusions.append({"model": model, "row_id": row["row_id"], "reason": str(error)})
    print(f"prepared MRCR new families: {needed}; model-length exclusions: {len(exclusions)}", flush=True)
    return exclusions


def validate(rows, tokenizer, old_rows=()):
    old_contexts, old_docs = excluded_sources(old_rows)
    seen, split_docs = set(), {"dev": set(), "test": set()}
    for row in rows:
        if row["row_id"] in seen:
            raise ValueError("duplicate row ID")
        seen.add(row["row_id"])
        if row["prefix_ids"] + row["suffix_ids"] != row["prompt_ids"]:
            raise ValueError("query-blind boundary does not reconstruct the full tokenization")
        encoded = tokenizer(row["full_prompt"], add_special_tokens=False, return_offsets_mapping=True)
        if encoded["input_ids"] != row["prompt_ids"]:
            raise ValueError("frozen token IDs disagree with native tokenizer")
        if row["prefix_length"] != len(row["prefix_ids"]) or row["raw_question"] in row["prefix_text"]:
            raise ValueError("prefix length mismatch or future-question leakage")
        if row.get("last_prefix_offset_end", 0) > row.get("boundary_char", 0):
            raise ValueError("a scored prefix token straddles the future-question boundary")
        if "boundary_char" in row and max(end for _, end in encoded["offset_mapping"][:row["prefix_length"]]) > row["boundary_char"]:
            raise ValueError("actual prefix token offsets include future suffix content")
        if row["score_contract"] == MRCR and render_messages(tokenizer, row)["prefix_ids"] != row["prefix_ids"]:
            raise ValueError("MRCR prefix differs from the intact conversation history")
        if row["context_sha256"] in old_contexts or set(row["constituent_doc_ids"]) & old_docs:
            raise ValueError("new panel reuses an old DEV/TEST document")
        if row["input_tokens"] + row["max_new_tokens"] > row["length_cap"]:
            raise ValueError("length budget exceeded")
        if row["score_contract"] == EXACT:
            matches = re.findall(r"Record key=([a-z]+); value=([a-z]+)\.", row["raw_context"])
            values = [value for key, value in matches if key == row["query"]["keys"][0]]
            if values != [row["expected"]] or len(matches) != row["record_count"]:
                raise ValueError("retrieval record/answer mapping changed")
        split_docs[row["split"]].update(row["constituent_doc_ids"])
    if split_docs["dev"] & split_docs["test"]:
        raise ValueError("new DEV/TEST share source material")
    return {"rows": len(rows), "unique_ids": len(seen), "source_split_overlap": 0,
            "old_panel_document_overlap": 0, "complete_tokenization_verified": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pm-tokenizer", type=Path, default=Path("results/pm_keep_20260909/source/tokenizer"))
    p.add_argument("--pc2-tokenizer", type=Path, default=Path("results/nosa_position_20260909/source"))
    p.add_argument("--old-pm-data", type=Path, default=Path("results/pm_keep_20260909/data/rows.jsonl"))
    p.add_argument("--old-mrcr-data", type=Path, default=Path("results/reference_position_20260909/mrcr_pairs_dev/inputs.jsonl"))
    p.add_argument("--old-longbench", type=Path, default=Path("results/pm_keep_20260909/source/longbench"))
    p.add_argument("--new-longbench", type=Path, default=Path("results/broad_position_eval_20260909/source"))
    p.add_argument("--prose", type=Path, default=Path("results/nosa_position_20260909/data/upstream_generator/data/synthetic/json/PaulGrahamEssays.json"))
    p.add_argument("--mrcr-source", type=Path, default=Path("results/reference_position_20260909/mrcr_source/8needle_0.parquet"))
    p.add_argument("--mrcr-receipt", type=Path, default=Path("results/reference_position_20260909/mrcr_source/receipt.json"))
    p.add_argument("--dev-count", type=int, default=8)
    p.add_argument("--test-count", type=int, default=16)
    p.add_argument("--mrcr-dev-families", type=int, default=4)
    p.add_argument("--mrcr-test-families", type=int, default=8)
    p.add_argument("--prefix-target", type=int, default=8192)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to overwrite a frozen or partially prepared panel")
    from transformers import AutoTokenizer
    tokenizers = {name: AutoTokenizer.from_pretrained(path, local_files_only=True)
                  for name, path in (("pm", args.pm_tokenizer), ("pc2", args.pc2_tokenizer))}
    limits = {"pm": 32768, "pc2": 16384}
    rows, sources = {"pm": [], "pc2": []}, {}
    prepare_natural(tokenizers, limits, args, rows, sources)
    prepare_retrieval(tokenizers, limits, args, rows, sources)
    exclusions = prepare_mrcr(tokenizers, limits, args, rows, sources)
    checks = {name: validate(items, tokenizers[name], read_rows(args.old_pm_data)) for name, items in rows.items()}
    args.output.mkdir(parents=True)
    receipts = {}
    for name, items in rows.items():
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items)
        (args.output / (name + ".jsonl")).write_text(payload)
        receipts[name] = {"sha256": sha(payload), "rows": len(items),
                          "counts": {f"{task}/{split}": count for (task, split), count in sorted(Counter((r['task'], r['split']) for r in items).items())},
                          "tokenizer_sha256": sha((getattr(args, name + "_tokenizer") / "tokenizer.json").read_bytes()),
                          "evaluation_limit": limits[name],
                          "limit_scope": "PM native 32K; PC2 existing authorized 16K evaluation setting, not a new native-length claim"}
    write_json(args.output / "manifest.json", {"status": "PREPARED_NO_MODEL_INFERENCE", "seed": SEED,
               "models": receipts, "source_files": sources, "checks": checks, "exclusions": exclusions,
               "code_sha256": sha(Path(__file__).read_bytes()),
               "unit": "source document/conversation or matched background family; never query head",
               "selection": "source-only, deterministic; original panel DEV/TEST sources excluded; no outcomes read",
               "test_rule": "New confirmation split stays frozen; do not tune on either old or new TEST"})
    print(json.dumps(receipts, indent=2), flush=True)


if __name__ == "__main__":
    main()
