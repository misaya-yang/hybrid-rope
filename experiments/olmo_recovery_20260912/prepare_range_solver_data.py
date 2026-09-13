#!/usr/bin/env python3
"""Prepare a fixed OLMo multi-length fit/select/confirm panel on CPU."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import sys


CAPS = (4096, 6144, 8192, 10240, 12288, 14336, 16384)
TASKS = ("niah_single_1", "niah_multikey_1", "qa_2")
ROWS_PER_CELL = 16
SEED = 2_026_091_303
QA_OFFSET = 5_000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def split_for_index(index: int) -> str:
    if 0 <= index < 8:
        return "fit"
    if index < 12:
        return "select"
    if index < 16:
        return "internal_confirm"
    raise ValueError("row index exceeds the frozen 8/4/4 split")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    model = args.model.resolve()
    upstream = args.upstream.resolve()
    config = json.loads((model / "config.json").read_text())
    if not (
        config.get("model_type") == "olmo2"
        and config.get("max_position_embeddings") == 4096
        and config.get("num_hidden_layers") == 16
        and config.get("rope_theta") == 500000
    ):
        raise ValueError("expected the retained Native-4K OLMo-2-1B checkpoint")

    os.environ.update(USE_TORCH="0", USE_TF="0", TOKENIZERS_PARALLELISM="false")
    import yaml
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    definitions = yaml.safe_load((upstream / "scripts/synthetic.yaml").read_text())
    constants = load_module("range_solver_ruler_constants", upstream / "scripts/data/synthetic/constants.py")
    args.out.mkdir(parents=True)
    rows = []
    with (args.out / "rows.jsonl").open("x") as output:
        for cap in CAPS:
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
                    "--num_samples", str(ROWS_PER_CELL),
                    "--random_seed", str(args.seed),
                    "--template", template,
                ]
                for name, value in definition["args"].items():
                    command.extend(["--" + name, str(value)])
                if task.startswith("qa_"):
                    command.extend(["--pre_samples", str(QA_OFFSET)])
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
                for sample_index, line in enumerate(source_path.read_text().splitlines()):
                    if not line:
                        continue
                    raw = json.loads(line)
                    if not 0 <= sample_index < ROWS_PER_CELL or len(raw["outputs"]) != 1:
                        raise ValueError(f"{task}/{cap} must have one canonical answer per row")
                    text = raw["input"] + raw.get("answer_prefix", "")
                    prompt = tokenizer.encode(text, add_special_tokens=False)
                    target = tokenizer.encode(" " + raw["outputs"][0], add_special_tokens=False)
                    target.append(int(tokenizer.eos_token_id))
                    if not prompt or len(target) < 2 or len(prompt) + budget > cap or len(target) > budget:
                        raise ValueError(f"invalid physical or answer contract for {task}/{cap}/{sample_index}")
                    row = {
                        "row_id": f"range_s{args.seed}_{task}_{cap}_{sample_index}",
                        "task": task,
                        "length_cap": cap,
                        "split": split_for_index(sample_index),
                        "source_seed": args.seed,
                        "upstream_index": int(raw.get("index", sample_index)),
                        "prompt_ids": prompt,
                        "prompt_sha256": digest(prompt),
                        "input_tokens": len(prompt),
                        "references": raw["outputs"],
                        "target_ids": target,
                        "target_includes_eos": True,
                        "max_new_tokens": budget,
                    }
                    output.write(json.dumps(row, sort_keys=True) + "\n")
                    rows.append(row)
                    generated += 1
                if generated != ROWS_PER_CELL:
                    raise ValueError(f"expected {ROWS_PER_CELL} rows for {task}/{cap}, got {generated}")

    counts = {
        f"{split}/{cap}/{task}": sum(
            row["split"] == split and row["length_cap"] == cap and row["task"] == task
            for row in rows
        )
        for split in ("fit", "select", "internal_confirm")
        for cap in CAPS
        for task in TASKS
    }
    expected = {"fit": 8, "select": 4, "internal_confirm": 4}
    if any(count != expected[key.split("/", 1)[0]] for key, count in counts.items()):
        raise RuntimeError("prepared panel is not a complete 8/4/4 task-length grid")
    manifest = {
        "status": "RANGE_SOLVER_DATA_READY",
        "model": str(model),
        "native_length": 4096,
        "deployment_horizon": 16384,
        "design_scale": 4,
        "transition_band": [14, 32],
        "length_caps": list(CAPS),
        "tasks": list(TASKS),
        "rows": len(rows),
        "rows_per_cell": ROWS_PER_CELL,
        "split_counts_per_cell": expected,
        "counts": counts,
        "target": "one canonical answer plus terminal EOS; all selected tasks have one reference",
        "fixed_table_contract": "one shared table and gain for every layer, task, and length",
        "scope": "development fit/select/internal-confirm; a new seed namespace is still required for final confirmation",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(model / "generation_config.json", args.out / "generation_config.json")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
