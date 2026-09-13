#!/usr/bin/env python3
"""Build the complete independent E3 RULER panel with the stock generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import contextlib
import importlib.metadata
import importlib.util
import os
import runpy
import shutil
import sys
from pathlib import Path

from experiments.rope_fast_5090_20260912.e3_tables import tables
from experiments.rope_fast_5090_20260912.e3_validate import TASKS, digest, validate


SEED = 2_026_091_203
QA_OFFSET = 3_000


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-prepared", type=Path, required=True, help="old 350-row prepared OLMo panel")
    parser.add_argument("--upstream", type=Path, required=True, help="pinned NVIDIA RULER checkout")
    parser.add_argument("--development-screen", type=Path, required=True)
    parser.add_argument("--prior-registry", type=Path, help="optional additional prior panel identities; the original 350-row development panel is always excluded")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--qa-offset", type=int, default=QA_OFFSET, help="4K QA offset; 16K uses offset+1000")
    args = parser.parse_args()
    if args.seed == 20260921:
        raise ValueError("E3 confirmation seed must differ from the development seed")
    old = args.reuse_prepared.resolve()
    upstream = args.upstream.resolve()
    prepared = args.out.resolve()
    old_manifest = json.loads((old / "manifest.json").read_text())
    model_path = Path(old_manifest["model_path"])
    from scripts.experiments.olmo_fast_screen.ruler_bench import score
    config = json.loads((model_path / "config.json").read_text())
    if config.get("model_type") != "olmo2" or config.get("hidden_size") != 2048 or config.get("num_hidden_layers") != 16:
        raise ValueError("unexpected OLMo model configuration")
    prepared.mkdir(parents=True, exist_ok=False)
    os.environ.update(USE_TORCH="0", USE_TF="0", TOKENIZERS_PARALLELISM="false")
    import yaml
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    definitions = yaml.safe_load((upstream / "scripts/synthetic.yaml").read_text())
    constants = load_module("e3_ruler_data_constants", upstream / "scripts/data/synthetic/constants.py")
    metrics = load_module("e3_ruler_metrics", upstream / "scripts/eval/synthetic/constants.py")
    families = old_manifest["families"]
    cells = ((4096, 40), (16384, 100))
    row_order = []
    token_total = 0
    token_min = None
    token_max = 0
    collection = hashlib.sha256()
    with (prepared / "screen.jsonl").open("x") as screen, (prepared / "prompts.jsonl").open("x") as prompts:
        for cap, count in cells:
            for task in TASKS:
                config = definitions[task]
                base = constants.TASKS[config["task"]]
                budget = base["tokens_to_generate"]
                template = tokenizer.apply_chat_template(
                    [{"role": "user", "content": base["template"]}], tokenize=False,
                    add_generation_prompt=True,
                ) + base.get("answer_prefix", "")
                command = [sys.executable, str(upstream / f"scripts/data/synthetic/{config['task']}.py"),
                    "--save_dir", str(prepared / "source" / str(cap)), "--save_name", task,
                    "--subset", "validation", "--tokenizer_path", str(model_path), "--tokenizer_type", "hf",
                    "--max_seq_length", str(cap), "--tokens_to_generate", str(budget),
                    "--num_samples", str(count), "--random_seed", str(args.seed), "--template", template]
                for name, value in config["args"].items():
                    command.extend(["--" + name, str(value)])
                if task.startswith("qa_"):
                    cell_qa_offset = args.qa_offset + (1000 if cap == 16384 else 0)
                    command.extend(["--pre_samples", str(cell_qa_offset)])
                previous_argv, previous_path, previous_cwd = sys.argv, sys.path[:], os.getcwd()
                with (prepared / f"{cap}_{task}.log").open("x") as log:
                    try:
                        sys.argv = command[1:]
                        sys.path.insert(0, str(upstream / "scripts/data/synthetic"))
                        os.chdir(upstream)
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            runpy.run_path(command[1], run_name="__main__")
                    finally:
                        sys.argv, sys.path = previous_argv, previous_path
                        os.chdir(previous_cwd)
                source_path = prepared / "source" / str(cap) / task / "validation.jsonl"
                generated_count = 0
                with source_path.open() as source:
                    for index, line in enumerate(source):
                        if not line.strip():
                            continue
                        raw = json.loads(line); generated_count += 1
                        text = raw["input"] + raw.get("answer_prefix", "")
                        ids = tokenizer.encode(text, add_special_tokens=False)
                        if not ids or len(ids) + budget > cap:
                            raise ValueError(f"{task}/{cap} input+output budget invalid: {len(ids)}+{budget}")
                        refs = raw["outputs"]
                        if not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
                            raise ValueError("invalid RULER references")
                        row_id = f"e3c_s{args.seed}_{task}_{cap}_{index}"
                        row = {"row_id": row_id, "task": task, "family": families[task],
                            "upstream_index": raw["index"], "source_seed": args.seed,
                            "confirmation_role": "independent E3 confirmation", "length_cap": cap,
                            "prompt_ids": ids, "prompt_sha256": digest(ids), "input_tokens": len(ids),
                            "references": refs, "max_new_tokens": budget}
                        if task.startswith("qa_"):
                            row["qa_source_index"] = cell_qa_offset + int(raw["index"])
                            row["qa_dataset"] = config["args"]["dataset"]
                        metric = metrics.string_match_part if task.startswith("qa_") else metrics.string_match_all
                        for prediction in ("", refs[0], " ".join(refs), "irrelevant answer"):
                            if round(score(row, prediction) * 100, 2) != metric([prediction], [refs]):
                                raise AssertionError("local scorer differs from pinned upstream")
                        screen.write(json.dumps(row, sort_keys=True) + "\n")
                        prompts.write(json.dumps({"row_id": row_id, "prompt_text": text,
                            "references": refs, "upstream_index": raw["index"], "source_seed": args.seed}) + "\n")
                        row_order.append(row_id); token_total += len(ids)
                        token_min = len(ids) if token_min is None else min(token_min, len(ids)); token_max = max(token_max, len(ids))
                        collection.update(bytes.fromhex(row["prompt_sha256"]))
                        del raw, text, ids, row
                if generated_count != count:
                    raise ValueError(f"generator count mismatch for {task}/{cap}: {generated_count}")

    (prepared / "qualification.jsonl").write_text("")
    shutil.copyfile(old / "generation_config.json", prepared / "generation_config.json")
    write(prepared / "tables.json", tables())
    write(prepared / "queue.json", {
        "max_candidates": 1,
        "ordered_candidates": [{
            "id": "C42V24", "eligible": True, "review_status": "REVIEWED_FOR_GPU",
            "definition": "P1 section G.7 C42V24 exact polynomial; same endpoints, band, sum_m=42 and increment centroid as C42",
            "hypothesis": "At 16K, the task-equal official RULER score differs because internal exponent structure matters beyond total displacement.",
            "failure_rule": "Complete all 980 paired inputs. A null or reversal narrows this specific development ordering; do not search a third profile.",
        }],
    })
    write(prepared / "e3_methods.json", {
        "methods": {"C42": ["C42"] * 16, "C42V24": ["C42V24"] * 16},
        "role": "uniform-table E3 pair; layer_screen is used only for rowwise recovery",
    })
    manifest_path = prepared / "manifest.json"
    manifest = old_manifest.copy()
    root = Path(__file__).resolve().parents[2]
    manifest.update(
        status="E3_PREPARED_GPU_NOT_RUN", reference_arm="C42", complete_candidate_queue=True,
        physical_caps=[4096, 16384], tasks=list(TASKS), seed=args.seed,
        qa_offset_by_cap={"4096": args.qa_offset, "16384": args.qa_offset + 1000},
        samples_per_task_by_cap={"4096": 40, "16384": 100}, screen_rows=980,
        screen_input_tokens=token_total, screen_min_max_tokens=[token_min, token_max],
        row_order=row_order, prompt_collection_sha256=collection.hexdigest(),
        confirmation_role="New fixed independent confirmation panel; development outcomes remain excluded.",
        primary_endpoint="16K equal-task official RULER C42V24-C42; paired task bootstrap downstream",
        secondary_endpoints="4K cost; strict complete-string exact; terminal EOS; generation-cap exhaustion",
        gpu_execution="NOT_RUN; execute exactly C42 then C42V24 on shared 980-row inputs",
        development_seed=20260921,
        source_identity_policy="QA source dataset and question index; synthetic prompt hash. Equal reference answers do not identify equal cases.",
        software={name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")},
        upstream_path=str(upstream), upstream_revision_label="c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
        memory_policy="stream one task shard at a time; compatible with 2 GiB preparation cgroup",
    )
    manifest["code_files"] = {}
    manifest["prepared_files"] = {}
    write(manifest_path, manifest)
    report = validate(prepared, args.development_screen.resolve(), None if args.prior_registry is None else args.prior_registry.resolve())
    write(prepared / "e3_cpu_validation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
