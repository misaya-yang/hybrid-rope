#!/usr/bin/env python3
"""Build one matched RULER panel for a fixed 8K-to-64K RoPE table."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


CAPS = (8192, 16384, 32768, 65536)
TASKS = ("niah_single_1", "niah_multikey_1", "niah_multivalue")
SEED = 2_026_091_301


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows-per-cell", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--length-cap", type=int, action="append", default=[])
    args = parser.parse_args()
    if args.rows_per_cell <= 0 or args.out.exists():
        raise ValueError("positive row count and a new output directory are required")

    caps = tuple(args.length_cap or CAPS)
    if len(set(caps)) != len(caps) or any(cap < 8192 or cap > 65536 for cap in caps):
        raise ValueError("length caps must be unique and inside the declared 8K-to-64K interval")
    model = args.model.resolve()
    upstream = args.upstream.resolve()
    config = json.loads((model / "config.json").read_text())
    if not (
        config.get("model_type") == "llama"
        and config.get("max_position_embeddings") == 8192
        and config.get("num_hidden_layers") == 32
        and config.get("rope_theta") == 500000.0
    ):
        raise ValueError("expected the retained Native-8K Llama-3-8B checkpoint")

    os.environ.update(USE_TORCH="0", USE_TF="0", TOKENIZERS_PARALLELISM="false")
    import yaml
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    definitions = yaml.safe_load((upstream / "scripts/synthetic.yaml").read_text())
    constants = load_module(
        "fixed_interval_ruler_constants",
        upstream / "scripts/data/synthetic/constants.py",
    )
    args.out.mkdir(parents=True)
    rows = []
    with (args.out / "screen.jsonl").open("x") as screen:
        for cap in caps:
            for task in TASKS:
                definition = definitions[task]
                base = constants.TASKS[definition["task"]]
                budget = int(base["tokens_to_generate"])
                template = tokenizer.apply_chat_template(
                    [{"role": "user", "content": base["template"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                ) + base.get("answer_prefix", "")
                command = [
                    sys.executable,
                    str(upstream / f"scripts/data/synthetic/{definition['task']}.py"),
                    "--save_dir", str(args.out / "source" / str(cap)),
                    "--save_name", task,
                    "--subset", "validation",
                    "--tokenizer_path", str(model),
                    "--tokenizer_type", "hf",
                    "--max_seq_length", str(cap),
                    "--tokens_to_generate", str(budget),
                    "--num_samples", str(args.rows_per_cell),
                    "--random_seed", str(args.seed),
                    "--template", template,
                ]
                for name, value in definition["args"].items():
                    command.extend(["--" + name, str(value)])
                previous_argv, previous_path, previous_cwd = sys.argv, sys.path[:], os.getcwd()
                try:
                    sys.argv = command[1:]
                    sys.path.insert(0, str(upstream / "scripts/data/synthetic"))
                    os.chdir(upstream)
                    with (args.out / f"{cap}_{task}.log").open("x") as log:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            runpy.run_path(command[1], run_name="__main__")
                finally:
                    sys.argv, sys.path = previous_argv, previous_path
                    os.chdir(previous_cwd)

                source_path = args.out / "source" / str(cap) / task / "validation.jsonl"
                generated = 0
                with source_path.open() as source:
                    for index, line in enumerate(source):
                        if not line.strip():
                            continue
                        raw = json.loads(line)
                        text = raw["input"] + raw.get("answer_prefix", "")
                        prompt = tokenizer.encode(text, add_special_tokens=False)
                        references = raw["outputs"]
                        if not prompt or len(prompt) + budget > cap or not references:
                            raise ValueError(f"invalid generated row for {task}/{cap}")
                        row = {
                            "row_id": f"fixed_g8_s{args.seed}_{task}_{cap}_{index}",
                            "task": task,
                            "family": "fixed_table_interval_retrieval",
                            "source_seed": args.seed,
                            "upstream_index": raw["index"],
                            "length_cap": cap,
                            "prompt_ids": prompt,
                            "prompt_sha256": digest(prompt),
                            "input_tokens": len(prompt),
                            "references": references,
                            "max_new_tokens": budget,
                        }
                        screen.write(json.dumps(row, sort_keys=True) + "\n")
                        rows.append(row)
                        generated += 1
                if generated != args.rows_per_cell:
                    raise ValueError(f"generator produced {generated} rows for {task}/{cap}")

    manifest = {
        "status": "READY_GPU_NOT_RUN",
        "scope": "development panel; three RULER tasks, not full RULER",
        "model": str(model),
        "native_length": 8192,
        "deployment_horizon": 65536,
        "design_scale": 8,
        "runtime_lengths": list(caps),
        "tasks": list(TASKS),
        "rows_per_cell": args.rows_per_cell,
        "rows": len(rows),
        "fixed_table_contract": {
            "one_table_per_arm_for_entire_session": True,
            "same_table_at_every_runtime_length": True,
            "same_table_in_every_layer": True,
            "runtime_table_switching": False,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.out / "data_manifest.json").write_text("{}\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
