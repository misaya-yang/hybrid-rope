#!/usr/bin/env python3
"""One fixed single-code measurement repair; reuse natural inputs, never V1 scores.

No parquet reread, model execution, RoPE intervention, rescoring, or template
search. The original two-code builder and artifacts remain unchanged.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import string
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data import prepare_native_reference_calibration as common  # noqa: E402

GRID = (1024, 2048, 4096, 8192)
SEEDS = {"calibration": 202609021, "confirmation": 202609022}
NATURAL_COUNTS = {"calibration": 32, "confirmation": 64}
CAPABILITY_COUNTS = {"calibration": 64, "confirmation": 128}
GENERATION_BUDGET = 48
ASSISTANT_PREFIX = "The code is "
PROTOCOL = "single_code_format_repair_v1"
READY = "NATIVE_REFERENCE_CALIBRATION_DATA_READY_V1"
NATURAL_KEYS = {
    "family", "split", "sample_id", "variant", "length", "input_ids", "target_start",
    "target_tokens", "source_row", "source_text_sha256", "prompt_ids_sha256", "target_ids_sha256",
}


def load_parent_inputs(root: Path) -> tuple[dict, dict[str, list[dict]], dict]:
    manifest_path = root / "manifest.json"
    parent = json.loads(manifest_path.read_text())
    if parent.get("status") != READY or parent.get("grid") != list(GRID):
        raise ValueError("parent must be complete reference-calibration input data")
    if parent.get("measurement_protocol") not in (None, "two_code"):
        raise ValueError("only one repair of the original two-code probe is permitted")
    natural, receipts, source_sets = {}, {}, {}
    for split, count in NATURAL_COUNTS.items():
        entry = parent["files"][split]
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("parent row file must be inside parent data root")
        file_hash = common.sha256_file(path)
        if file_hash != entry["sha256"]:
            raise ValueError("parent input row-file hash drift")
        selected, groups, total = [], defaultdict(list), 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                total += 1
                if row.get("family") != "natural":
                    continue
                # A model-output JSONL is never an admissible input source.
                if set(row) != NATURAL_KEYS:
                    raise ValueError("parent natural row is not the pure V1 input schema")
                ids, length = row["input_ids"], row["length"]
                if (row["split"] != split or row["variant"] != "natural" or length not in GRID
                        or len(ids) != length or ids[0] != parent["bos_token_id"]
                        or row["target_start"] != length - 256 or row["target_tokens"] != 256
                        or common.ids_hash(ids) != row["prompt_ids_sha256"]
                        or common.ids_hash(ids[-256:]) != row["target_ids_sha256"]):
                    raise ValueError("parent natural input contract drift")
                selected.append(row)
                groups[row["sample_id"]].append(row)
        if total != entry["rows"] or len(selected) != count * len(GRID) or len(groups) != count:
            raise ValueError("parent input row/sample count drift")
        for rows in groups.values():
            if (sorted(r["length"] for r in rows) != list(GRID)
                    or len({r["target_ids_sha256"] for r in rows}) != 1
                    or len({r["source_text_sha256"] for r in rows}) != 1
                    or len({r["source_row"] for r in rows}) != 1):
                raise ValueError("parent natural pairing drift")
        sources = {row["source_text_sha256"] for row in selected}
        if len(sources) != count:
            raise ValueError("parent natural documents are not unique")
        source_sets[split] = sources
        natural[split] = selected
        receipts[split] = {
            "path": entry["path"], "sha256": file_hash, "parent_rows": total,
            "reused_natural_rows": len(selected),
            "natural_rows_canonical_sha256": common.canonical_hash(selected),
            "source_document_set_sha256": common.canonical_hash(sorted(sources)),
        }
    if source_sets["calibration"] & source_sets["confirmation"]:
        raise ValueError("parent natural calibration/confirmation source overlap")
    return parent, natural, {"manifest_sha256": common.sha256_file(manifest_path), "files": receipts}


def make_blueprints(split: str) -> list[dict]:
    rng = random.Random(SEEDS[split])
    result = []
    for index in range(CAPABILITY_COUNTS[split]):
        names: list[str] = []
        while len(names) < 8:
            name = "".join(rng.choices(string.ascii_lowercase, k=8))
            if name not in names:
                names.append(name)
        values = rng.sample(range(100000, 1000000), 8)
        facts = [{"index": i, "key": name, "code": str(value)}
                 for i, (name, value) in enumerate(zip(names, values))]
        query_index = index % 8
        identity = {"facts": facts, "query_index": query_index}
        result.append({**identity, "sample_id": f"single-code-{split}-{index:03d}",
                       "depth_stratum": query_index // 2,
                       "blueprint_sha256": common.canonical_hash(identity)})
    return result


def expected_text(blueprint: dict) -> str:
    return blueprint["facts"][blueprint["query_index"]]["code"] + "."


def render_user(blueprint: dict, filler_words: int, filler_tail: str = "") -> str:
    base, remainder = divmod(filler_words, 9)
    gaps = [" ".join(common.FILLER[j % len(common.FILLER)] for j in range(base + (i < remainder)))
            for i in range(9)]
    gaps[-1] += filler_tail
    parts = ["Read the eight records below. Only the explicitly labeled records contain codes.", gaps[0]]
    for i, fact in enumerate(blueprint["facts"]):
        parts.extend([f"Record {i + 1}: {fact['key']} has code {fact['code']}.", gaps[i + 1]])
    key = blueprint["facts"][blueprint["query_index"]]["key"]
    parts.append(f"Return the six-digit code for {key}, followed by one period. "
                 "Continue the assistant prefix with only the code and period. "
                 "Do not output any other words or numbers.")
    return "\n\n".join(part for part in parts if part)


def chat_ids(tokenizer: Any, blueprint: dict, filler_words: int, filler_tail: str = "") -> list[int]:
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": render_user(blueprint, filler_words, filler_tail)}],
        tokenize=False, add_generation_prompt=True,
    )
    return [int(v) for v in tokenizer.encode(chat + ASSISTANT_PREFIX, add_special_tokens=False)]


def fit_prompt(tokenizer: Any, blueprint: dict, budget: int) -> tuple[list[int], int, str]:
    """The V1 filler-only fitting algorithm, explicitly bound to the single-code renderer."""
    best_ids, best_words = chat_ids(tokenizer, blueprint, 0), 0
    if len(best_ids) > budget:
        raise ValueError("single-code facts/query/template exceed prompt budget")
    low, high = 0, budget * 2
    if len(chat_ids(tokenizer, blueprint, high)) <= budget:
        raise ValueError("filler tokenization cannot bracket prompt length")
    while low <= high:
        middle = (low + high) // 2
        candidate = chat_ids(tokenizer, blueprint, middle)
        if len(candidate) <= budget:
            if len(candidate) >= len(best_ids):
                best_ids, best_words = candidate, middle
            low = middle + 1
        else:
            high = middle - 1
    if budget - len(best_ids) > common.MAX_PROMPT_UNDERAGE:
        raise ValueError("excessive prompt underage after filler-only fitting")
    best_tail, tail = "", " The routine review continues." * 2
    for chars in range(1, len(tail) + 1):
        if len(best_ids) == budget:
            break
        candidate = chat_ids(tokenizer, blueprint, best_words, tail[:chars])
        if len(best_ids) < len(candidate) <= budget:
            best_ids, best_tail = candidate, tail[:chars]
    return best_ids, best_words, best_tail


def capability_rows(blueprint: dict, split: str, tokenizer: Any) -> Iterable[dict]:
    expected = expected_text(blueprint)
    expected_ids = tokenizer.encode(expected, add_special_tokens=False)
    if (tokenizer.decode(expected_ids, skip_special_tokens=False,
                         clean_up_tokenization_spaces=False) != expected
            or len(expected_ids) + 1 > GENERATION_BUDGET):
        raise ValueError("single-code expected output round-trip/budget failed")
    for length in (0, *GRID):
        if length == 0:
            ids, words, tail = chat_ids(tokenizer, blueprint, 0), 0, ""
        else:
            ids, words, tail = fit_prompt(tokenizer, blueprint, length - GENERATION_BUDGET)
        yield {
            "family": "capability", "split": split, "sample_id": blueprint["sample_id"],
            "variant": "compact" if length == 0 else "distributed", "length": length,
            "input_ids": ids, "prompt_tokens": len(ids),
            "prompt_token_budget": len(ids) if length == 0 else length - GENERATION_BUDGET,
            "prompt_underage": 0 if length == 0 else length - GENERATION_BUDGET - len(ids),
            "prompt_ids_sha256": common.ids_hash(ids), "filler_words": words, "filler_tail": tail,
            "assistant_prefix": ASSISTANT_PREFIX, "expected_text": expected,
            "terminal_ids": [int(tokenizer.eos_token_id)], "generation_budget": GENERATION_BUDGET,
            "facts": blueprint["facts"], "query_index": blueprint["query_index"],
            "target_key": blueprint["facts"][blueprint["query_index"]]["key"],
            "depth_stratum": blueprint["depth_stratum"],
            "blueprint_sha256": blueprint["blueprint_sha256"], "measurement_protocol": PROTOCOL,
        }


def scorer_fixtures(tokenizer: Any) -> dict:
    expected, eos = "123456.", int(tokenizer.eos_token_id)
    encode = lambda text: [int(v) for v in tokenizer.encode(text, add_special_tokens=False)]
    correct = encode(expected)
    cases = [
        ("exact_with_terminal_eos", correct + [eos], True),
        ("outer_whitespace_only", encode(" \n" + expected + "\n ") + [eos], True),
        ("missing_terminal_eos", correct, False),
        ("missing_period", encode("123456") + [eos], False),
        ("wrong_code", encode("654321.") + [eos], False),
        ("two_codes_not_accepted", encode("123456, 654321.") + [eos], False),
        ("extra_prose", encode("The code is 123456.") + [eos], False),
        ("internal_spacing_changed", encode("123456 .") + [eos], False),
        ("wrong_punctuation", encode("123456!") + [eos], False),
        ("tokens_after_eos", correct + [eos] + encode("x"), False),
        ("early_eos", encode("123") + [eos], False),
    ]
    return {
        "expected_text": expected, "terminal_ids": [eos],
        "contract": "Generated continuation only: first EOS must be final raw token; decode preceding tokens with skip_special_tokens=False and clean_up_tokenization_spaces=False; strip outer whitespace only, then exact equality. No substring, punctuation, internal spacing, or format-variant acceptance.",
        "cases": [{"name": name, "generated_ids": ids, "expected_pass": passed}
                  for name, ids, passed in cases],
    }


def prepare(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError("output directory must be fresh")
    parent, natural, parent_receipt = load_parent_inputs(args.parent_data_root)
    config_path = args.checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    token_files = common.tokenizer_file_receipts(args.checkpoint)
    if (common.sha256_file(config_path) != parent["config_sha256"]
            or token_files != parent["tokenizer_files"]):
        raise ValueError("checkpoint config/tokenizer differs from reused natural inputs")
    tokenizer = common.load_tokenizer(args.checkpoint)
    if (tokenizer.bos_token_id != parent["bos_token_id"]
            or tokenizer.eos_token_id != parent["eos_token_id"]
            or common.canonical_hash(tokenizer.chat_template) != parent["chat_template_sha256"]):
        raise ValueError("tokenizer BOS/EOS/chat template differs from parent")
    blueprints = {split: make_blueprints(split) for split in SEEDS}
    blueprint_sets = {split: {b["blueprint_sha256"] for b in rows} for split, rows in blueprints.items()}
    if blueprint_sets["calibration"] & blueprint_sets["confirmation"]:
        raise ValueError("single-code blueprint splits overlap")
    fixtures = scorer_fixtures(tokenizer)
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split in SEEDS:
        path = args.output / f"{split}.jsonl"
        temporary = path.with_suffix(".jsonl.incomplete")
        count, max_underage = 0, 0
        with temporary.open("x", encoding="utf-8") as handle:
            for row in natural[split]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                count += 1
            for blueprint in blueprints[split]:
                for row in capability_rows(blueprint, split, tokenizer):
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    count += 1
                    max_underage = max(max_underage, row["prompt_underage"])
        temporary.replace(path)
        files[split] = {
            "path": path.name, "sha256": common.sha256_file(path), "rows": count,
            "natural_documents": NATURAL_COUNTS[split], "natural_rows": len(natural[split]),
            "capability_blueprints": CAPABILITY_COUNTS[split],
            "capability_rows": CAPABILITY_COUNTS[split] * (len(GRID) + 1),
            "source_document_set_sha256": parent_receipt["files"][split]["source_document_set_sha256"],
            "blueprint_set_sha256": common.canonical_hash(sorted(blueprint_sets[split])),
            "max_prompt_underage": max_underage,
        }
    manifest = {
        "status": READY, "schema_version": 1, "measurement_protocol": PROTOCOL,
        "superseded_probe": "two_code", "measurement_repair_count": 1,
        "model_evaluation_status": "NOT_RUN", "L_ref_selected": False,
        "model_type": config["model_type"], "config_sha256": parent["config_sha256"],
        "tokenizer_files": token_files, "tokenizer_files_sha256": common.canonical_hash(token_files),
        "tokenizer_class": type(tokenizer).__name__, "bos_token_id": int(tokenizer.bos_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "chat_template_sha256": common.canonical_hash(tokenizer.chat_template),
        "script_sha256": common.sha256_file(Path(__file__)),
        "common_utility_script_sha256": common.sha256_file(Path(common.__file__)),
        "grid": list(GRID), "seeds": SEEDS, "files": files,
        "sample_ids": {
            split: {
                "natural": sorted({row["sample_id"] for row in natural[split]}),
                "capability": [blueprint["sample_id"] for blueprint in blueprints[split]],
            } for split in SEEDS
        },
        "generation_budget": GENERATION_BUDGET, "source": parent["source"],
        "natural_input_reuse": {
            "parent_data": parent_receipt, "exact_input_rows_reused": True,
            "source_parquet_reread": False, "model_outputs_read": False, "V1_rescored": False,
            "exposure_declared_by_parent": {"calibration": "previously evaluated",
                                            "confirmation": "not previously evaluated"},
        },
        "natural_contract": parent["natural_contract"],
        "capability_contract": {
            "records": 8, "queried_facts": 1, "distractors": 7,
            "query_index": "blueprint index modulo 8; each record position equally represented",
            "depth_stratum": "query_index // 2", "generation_budget": GENERATION_BUDGET,
            "assistant_prefix": ASSISTANT_PREFIX, "expected_text": "six digits followed by one period",
            "prompt_budget": "L-48", "max_underage": common.MAX_PROMPT_UNDERAGE,
            "filler": "fixed neutral procedural sentences, nine gaps; edit filler only",
            "compact_length": 0, "near_query": "OMITTED", "template_search": False,
        },
        "disjointness": {"cal_confirmation_source_documents": True,
                         "cal_confirmation_blueprints": True,
                         "capability_uses_source_documents": False,
                         "source_exclusion_provenance": "inherited unchanged from hash-bound parent input manifest"},
        "full_output_scorer_fixtures": fixtures,
        "prompt_ids_sha256_encoding": "contiguous little-endian signed int64 raw token bytes",
        "ready_contract": "Both split files completed and hash-bound before manifest; no model evaluation performed.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
