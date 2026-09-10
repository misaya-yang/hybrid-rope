"""Freeze NOSA-tokenized public RULER and order counterfactual data on CPU.

Run from the repository root. Public rows retain the upstream task, prompt,
answer prefix, output reserve and recall metric. Counterfactual rows are a
separate diagnostic with literal full-string plus terminal-EOS scoring.
Neither dataset is the complete public RULER benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys


PUBLIC_TASKS = ("niah_single_1", "niah_multikey_1", "niah_multiquery", "vt")
SPLIT_SEEDS = {"dev": 910901, "test": 920903}
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
PUBLIC_CONTRACT = "ruler_official_string_match_all_v1"
EXACT_CONTRACT = "literal_full_string_plus_terminal_eos_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def render_chat(tokenizer, text):
    # Never decode/slice/re-encode an already rendered chat to fit the bucket.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def add_tokens(tokenizer, row):
    row["prompt_ids"] = tokenizer.encode(row["prompt"], add_special_tokens=False)
    row["input_tokens"] = len(row["prompt_ids"])
    if not 0 < row["input_tokens"] + row["max_new_tokens"] <= row["length_cap"]:
        raise ValueError(f"untruncated rendered prompt exceeds bucket: {row['row_id']}")
    row["prompt_sha256"] = text_sha(row["prompt"])
    row["prompt_ids_sha256"] = text_sha(json.dumps(row["prompt_ids"], separators=(",", ":")))
    return row


def official_constants(path):
    spec = importlib.util.spec_from_file_location("nosa_ruler_constants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TASKS


def copy_generator(upstream, target, essay_path):
    """Copy only unmodified required upstream files and one local public asset."""
    paths = ["synthetic.yaml", "data/tokenizer.py", "data/manifest_utils.py",
             "data/synthetic/constants.py", "data/synthetic/niah.py",
             "data/synthetic/variable_tracking.py", "eval/synthetic/constants.py"]
    receipts = {}
    for relative in paths:
        original = upstream / "benchmarks/RULER/scripts" / relative
        copied = target / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copied)
        receipts[relative] = sha(original)
    copied_essay = target / "data/synthetic/json/PaulGrahamEssays.json"
    copied_essay.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(essay_path, copied_essay)
    receipts["data/synthetic/json/PaulGrahamEssays.json"] = sha(essay_path)
    return receipts


def prepare_public(args, tokenizer, work, generator_root, rows):
    import yaml

    if args.reuse_public_from:
        previous = args.reuse_public_from.resolve()
        receipt = json.loads((previous / "manifest.json").read_text())
        if sha(previous / "rows.jsonl") != receipt["rows_sha256"]:
            raise ValueError("cannot reuse changed frozen data")
        for filename, expected_hash in receipt["tokenizer"]["files_sha256"].items():
            if sha(args.tokenizer / filename) != expected_hash:
                raise ValueError("public reuse tokenizer changed")
        for filename, expected_hash in receipt["upstream_files_sha256"].items():
            if sha(generator_root / filename) != expected_hash:
                raise ValueError("public reuse generator or public corpus changed")
        reused = [r for r in load_jsonl(previous / "rows.jsonl")
                  if r["score_contract"] == PUBLIC_CONTRACT]
        expected_counts = {(task, split, length): count for task in PUBLIC_TASKS
                           for split, count in (("dev", args.dev_count), ("test", args.test_count))
                           for length in args.lengths}
        if Counter((r["task"], r["split"], r["length_cap"]) for r in reused) != expected_counts:
            raise ValueError("public reuse counts/tasks/lengths disagree with requested configuration")
        for call in receipt["generation_calls"]:
            if sha(previous / call["source_jsonl"]) != call["source_sha256"]:
                raise ValueError("public reuse raw source changed")
        shutil.copytree(previous / "public_source", work / "public_source")
        shutil.copytree(previous / "logs", work / "logs")
        rows.extend(reused)
        print(f"reused public rows byte-preserving: {len(reused)}", flush=True)
        return receipt["generation_calls"]

    configs = yaml.safe_load((generator_root / "synthetic.yaml").read_text())
    constants = official_constants(generator_root / "data/synthetic/constants.py")
    calls = []
    for task_index, task in enumerate(PUBLIC_TASKS):
        task_config = configs[task]
        base = constants[task_config["task"]]
        # This upstream script writes input and answer_prefix separately. Rendering
        # before generation preserves the complete chat, then reconstructing the
        # two upstream fields restores the exact bytes of the generated prompt.
        template = render_chat(tokenizer, base["template"]) + base["answer_prefix"]
        for split, count in (("dev", args.dev_count), ("test", args.test_count)):
            for length in args.lengths:
                seed = SPLIT_SEEDS[split] + task_index * 1000 + length
                shard = work / "public_source" / split / str(length)
                generator = generator_root / f"data/synthetic/{task_config['task']}.py"
                argv = [sys.executable, str(generator), "--save_dir", str(shard),
                        "--save_name", task, "--subset", split,
                        "--tokenizer_path", str(args.tokenizer), "--tokenizer_type", "hf",
                        "--max_seq_length", str(length), "--tokens_to_generate",
                        str(base["tokens_to_generate"]), "--num_samples", str(count),
                        "--random_seed", str(seed), "--template", template]
                for key, value in task_config["args"].items():
                    argv += ["--" + key, str(value)]
                log_path = work / "logs" / f"{split}_{length}_{task}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w") as log:
                    subprocess.run(argv, check=True, timeout=600, stdout=log,
                                   stderr=subprocess.STDOUT,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                                        "TOKENIZERS_PARALLELISM": "false",
                                        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                        "NLTK_DATA": str(args.nltk_data),
                                        "HF_HUB_OFFLINE": "1"})
                source_path = shard / task / f"{split}.jsonl"
                source_rows = load_jsonl(source_path)
                if len(source_rows) != count:
                    raise ValueError(f"upstream generated wrong row count: {source_path}")
                calls.append({"task": task, "split": split, "length_cap": length,
                              "seed": seed, "argv": argv,
                              "source_jsonl": str(source_path.relative_to(work)),
                              "source_sha256": sha(source_path)})
                for index, source in enumerate(source_rows):
                    row_id = f"ruler_{split}_{task}_{length}_{index:03d}"
                    row = dict(
                        row_id=row_id, task=task, suite="public_ruler_subset",
                        family_id=row_id, split=split, seed=seed,
                        length_cap=length, expected=None, references=source["outputs"],
                        score_contract=PUBLIC_CONTRACT,
                        max_new_tokens=base["tokens_to_generate"],
                        prompt=source["input"] + source["answer_prefix"],
                        source={"kind": "unmodified_upstream_generator", "task": task,
                                "path": str(source_path.relative_to(work)), "line": index + 1,
                                "upstream_index": source["index"]},
                        answer_format="upstream answer prefix and references; no exact-format rewrite",
                    )
                    rows.append(add_tokens(tokenizer, row))
                print(f"prepared public {split} {length} {task}: {count}", flush=True)
    return calls


def family_material(split, index):
    seed = SPLIT_SEEDS[split] * 100000 + index
    rng = random.Random(seed)
    # Prefixes make dev/test *content strings*, not only seeds, disjoint.
    namespace = "d" if split == "dev" else "t"
    used = set()

    def word(size=12):
        while True:
            value = namespace + "".join(rng.choices("abcdefghjkmnpqrstuvwxyz", k=size))
            if value not in used:
                used.add(value)
                return value

    key = word(8)
    values = [word() for _ in range(4)]
    target_slots = sorted(rng.sample(range(12), 4))
    records, next_target = [], 0
    for slot in range(12):
        if slot in target_slots:
            records.append({"key": key, "value": values[next_target], "target": True})
            next_target += 1
        else:
            records.append({"key": word(8), "value": word(), "target": False})
    return {"seed": seed, "key": key, "values": values,
            "records": records, "target_slots": target_slots}


def counterfactual_user(material, query_index, content_swap, noise_count):
    values = material["values"][::-1] if content_swap else material["values"]
    records = [dict(record) for record in material["records"]]
    for occurrence, slot in enumerate(material["target_slots"]):
        records[slot]["value"] = values[occurrence]
    blocks = []
    # Use complete sentences and complete records at every size. History order is
    # the only source of temporal information; no timestamps or ordinal labels.
    for i in range(13):
        n = noise_count // 13 + int(i < noise_count % 13)
        if n:
            blocks.append("\n".join([NOISE] * n))
        if i < 12:
            record = records[i]
            blocks.append(f"Record key={record['key']}; value={record['value']}.")
    label = ("first", "second", "third", "latest")[query_index]
    text = (
        "Read the complete history below in its written order, from earliest to latest. "
        "Each Record gives one key and one value. A key may appear repeatedly. "
        "Ignore records with other keys.\n\nHISTORY\n" + "\n".join(blocks) +
        "\nEND HISTORY\n\n" +
        f"For key {material['key']}, return the value in its {label} Record. "
        "Output exactly the value string and then stop. Do not add any spaces, "
        "punctuation, explanation, or other text."
    )
    return text, values[query_index], records


def prepare_counterfactual(args, tokenizer, rows):
    for split, count in (("dev", args.dev_families), ("test", args.test_families)):
        for family_index in range(count):
            material = family_material(split, family_index)
            for task, query_pair in (("repeat_key_first_latest", (0, 3)),
                                     ("repeat_key_nth", (1, 2))):
                for length in args.lengths:
                    family_id = f"{task}_{split}_{family_index:03d}"

                    def candidate(noise_count, content_swap, query_swap):
                        text, expected, records = counterfactual_user(
                            material, query_pair[query_swap], content_swap, noise_count)
                        prompt = render_chat(tokenizer, text)
                        return prompt, expected, records

                    # Fit against all four members. The same history filler count
                    # is fixed across cells, preserving the factorial contrast.
                    noise_tokens = len(tokenizer.encode(NOISE + "\n", add_special_tokens=False))
                    low, high, best = 0, length // max(1, noise_tokens) + 2, 0
                    while low <= high:
                        mid = (low + high) // 2
                        largest = max(len(tokenizer.encode(candidate(mid, c, q)[0],
                                                           add_special_tokens=False))
                                      for c in (0, 1) for q in (0, 1))
                        if largest + 32 <= length:
                            best, low = mid, mid + 1
                        else:
                            high = mid - 1
                    for content_swap in (0, 1):
                        for query_swap in (0, 1):
                            prompt, expected, records = candidate(best, content_swap, query_swap)
                            row = dict(
                                row_id=f"{family_id}_{length}_c{content_swap}q{query_swap}",
                                task=task, suite="counterfactual_order_diagnostic",
                                family_id=family_id, split=split, seed=material["seed"],
                                material_cluster_id=f"{split}:{material['seed']}",
                                length_cap=length, expected=expected, references=[expected],
                                score_contract=EXACT_CONTRACT, max_new_tokens=32,
                                prompt=prompt, content_swap=content_swap,
                                query_swap=query_swap, query_ordinal=query_pair[query_swap] + 1,
                                query_key=material["key"], records=records,
                                target_slots=material["target_slots"],
                                distractor_count=8, noise_sentence_count=best,
                                source={"kind": "local_counterfactual_generator",
                                        "code": "experiments/nosa_position/prepare.py"},
                                answer_format="literal value string, no normalization; terminal EOS required",
                                expected_ids=tokenizer.encode(expected, add_special_tokens=False),
                            )
                            rows.append(add_tokens(tokenizer, row))
            print(f"prepared counterfactual {split} family {family_index + 1}/{count}", flush=True)


def validate_rows(rows, tokenizer=None):
    if not rows or len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("empty data or duplicate row IDs")
    groups, contents = {}, {"dev": set(), "test": set()}
    for row in rows:
        if not 0 < len(row["prompt_ids"]) + row["max_new_tokens"] <= row["length_cap"]:
            raise ValueError("prompt budget mismatch")
        if len(row["prompt_ids"]) != row["input_tokens"]:
            raise ValueError("input token count mismatch")
        if tokenizer and tokenizer.encode(row["prompt"], add_special_tokens=False) != row["prompt_ids"]:
            raise ValueError("frozen prompt IDs disagree with tokenizer")
        if text_sha(row["prompt"]) != row["prompt_sha256"]:
            raise ValueError("prompt hash mismatch")
        if not row["references"] or any(not ref for ref in row["references"]):
            raise ValueError("empty references")
        if row["score_contract"] == PUBLIC_CONTRACT:
            if row["expected"] is not None or row["task"] not in PUBLIC_TASKS:
                raise ValueError("public recall task mislabeled as exact")
            continue
        if row["score_contract"] != EXACT_CONTRACT:
            raise ValueError("unknown score contract")
        if row["material_cluster_id"] != f"{row['split']}:{row['seed']}":
            raise ValueError("material clustering identity mismatch")
        records = row["records"]
        observed = re.findall(r"Record key=([a-z]+); value=([a-z]+)\.", row["prompt"])
        if observed != [(r["key"], r["value"]) for r in records]:
            raise ValueError("history was modified or truncated")
        targets = [r["value"] for r in records if r["key"] == row["query_key"]]
        if len(targets) != 4 or sum(r["key"] != row["query_key"] for r in records) != 8:
            raise ValueError("incorrect number of target/distractor records")
        if row["expected"] != targets[row["query_ordinal"] - 1]:
            raise ValueError("answer mapping mismatch")
        if row["references"] != [row["expected"]]:
            raise ValueError("diagnostic references mismatch")
        for record in records:
            contents[row["split"]].update((record["key"], record["value"]))
        groups.setdefault((row["family_id"], row["length_cap"]), []).append(row)
    if contents["dev"] & contents["test"]:
        raise ValueError("dev/test content leakage")
    for family in groups.values():
        if {(r["content_swap"], r["query_swap"]) for r in family} != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError("incomplete counterfactual family")
        if len({r["noise_sentence_count"] for r in family}) != 1:
            raise ValueError("counterfactual filler mismatch")
        by_cell = {(r["content_swap"], r["query_swap"]): r for r in family}
        if (by_cell[0, 0]["expected"] != by_cell[1, 1]["expected"] or
            by_cell[0, 1]["expected"] != by_cell[1, 0]["expected"] or
            by_cell[0, 0]["expected"] == by_cell[0, 1]["expected"]):
            raise ValueError("counterfactual swap algebra failed")
    return {"rows": len(rows), "factorial_families_per_length": len(groups),
            "dev_test_content_intersection": 0, "complete_history_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("/tmp/hybrid-NOSA-20260909"))
    parser.add_argument("--tokenizer", type=Path, default=Path("results/nosa_position_20260909/source"))
    parser.add_argument("--model-id", default="openbmb/NOSA-1B")
    parser.add_argument("--out", type=Path, default=Path("results/nosa_position_20260909/data"))
    parser.add_argument("--essay", type=Path)
    parser.add_argument("--reuse-public-from", type=Path,
                        help="Reuse verified public rows/raw byte-for-byte while preparing more diagnostic families")
    parser.add_argument("--nltk-data", type=Path, default=Path("/tmp/nosa-nltk-data"))
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 16384])
    parser.add_argument("--dev-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=32)
    parser.add_argument("--dev-families", type=int, default=2)
    parser.add_argument("--test-families", type=int, default=32)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    args.tokenizer, args.upstream, args.out = args.tokenizer.resolve(), args.upstream.resolve(), args.out.resolve()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    if args.verify_only:
        rows = load_jsonl(args.out / "rows.jsonl")
        manifest = json.loads((args.out / "manifest.json").read_text())
        if sha(args.out / "rows.jsonl") != manifest["rows_sha256"]:
            raise ValueError("frozen dataset SHA mismatch")
        print(json.dumps(validate_rows(rows, tokenizer)))
        return
    if args.out.exists():
        raise FileExistsError(f"frozen output already exists: {args.out}; use a new --out")
    if (len(set(args.lengths)) != len(args.lengths) or min(args.lengths) < 1024 or
        min(args.dev_count, args.test_count, args.dev_families, args.test_families) <= 0):
        raise ValueError("unique lengths >=1024 and positive sample counts required")
    work = args.out.with_name(args.out.name + ".building")
    work.mkdir(parents=True, exist_ok=False)
    essay_path = args.essay or args.upstream / "dependencies/ShadowKV/data/ruler/synthetic/json/PaulGrahamEssays.json"
    generator_root = work / "upstream_generator"
    source_hashes = copy_generator(args.upstream, generator_root, essay_path)
    rows = []
    calls = prepare_public(args, tokenizer, work, generator_root, rows)
    prepare_counterfactual(args, tokenizer, rows)
    validation = validate_rows(rows, tokenizer)
    with (work / "rows.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    from importlib.metadata import version
    shards = {}
    for row in rows:
        key = f"{row['split']}/{row['length_cap']}/{row['task']}"
        shards.setdefault(key, []).append(row["input_tokens"])
    tokenizer_files = {p.name: sha(p) for p in args.tokenizer.iterdir()
                       if p.is_file() and (p.name.startswith("tokenizer") or p.name in
                                          ("special_tokens_map.json", "config.json"))}
    manifest = dict(
        status="FROZEN_CPU_PREPARED", model_id=args.model_id,
        upstream_repo="https://github.com/thunlp/NOSA",
        upstream_commit=subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
        upstream_files_sha256=source_hashes, essay_original_path=str(essay_path),
        tokenizer={"path_at_preparation": str(args.tokenizer), "class": type(tokenizer).__name__,
                   "vocab_size": len(tokenizer), "files_sha256": tokenizer_files,
                   "eos_token_id": tokenizer.eos_token_id, "allowed_terminal_ids": [2, tokenizer.eos_token_id],
                   "chat_template": tokenizer.chat_template, "enable_thinking": False,
                   "add_special_tokens_after_render": False},
        package_versions={name: version(name) for name in ("transformers", "tokenizers", "nltk", "wonderwords", "numpy", "tenacity", "PyYAML")},
        nltk_english_files_sha256={p.name: sha(p) for p in (args.nltk_data / "tokenizers/punkt_tab/english").glob("*") if p.is_file()},
        prepare_code_sha256=sha(__file__), split_seeds=SPLIT_SEEDS,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        generation_calls=calls, validation=validation, rows_sha256=sha(work / "rows.jsonl"),
        shard_summary={key: {"rows": len(values), "min_input_tokens": min(values),
                            "max_input_tokens": max(values)} for key, values in shards.items()},
        score_contracts={PUBLIC_CONTRACT: "Unmodified upstream string_match_all: case-insensitive reference substring recall, macro-averaged over examples; report separately from exact and EOS.",
                         EXACT_CONTRACT: "Decode the entire generated answer before its final EOS without normalization; it must equal expected exactly, and the only EOS must be terminal. Truncation, leading/trailing text or whitespace, and missing EOS fail."},
        limitations=["Four public RULER tasks and fixed local sample counts; not full RULER or a reproduction of published scores.",
                     "Counterfactual tasks are synthetic diagnostics, not public benchmark tasks.",
                     "Public task rows are independent across lengths; counterfactual content families are paired across lengths.",
                     "Counterfactual material_cluster_id groups shared content across both query tasks and all lengths; use it for cluster bootstrap. Each task/family_id/length_cap has all four factorial cells.",
                     "No model inference, task success, or sparse path activation is established by CPU preparation.",
                     "LongBench not added: this build uses public RULER plus the order diagnostic."],
    )
    dump_json(work / "manifest.json", manifest)
    work.rename(args.out)
    print(json.dumps({"out": str(args.out), "status": manifest["status"],
                      "rows_sha256": manifest["rows_sha256"], **validation}), flush=True)


if __name__ == "__main__":
    main()
