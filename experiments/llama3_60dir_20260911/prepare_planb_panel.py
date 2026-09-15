"""Prepare a checkpoint-tokenized Plan B P/S/V/H RULER-derived panel.

The upstream generators are pinned inputs, but their output is tokenized with
the actual Meta-Llama-3-8B-Instruct checkpoint.  Every stage uses a distinct
seed and every task/length cell receives independent scenarios.  This script
does CPU preparation only and never reads model outcomes.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import sys


UPSTREAM_REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
TASKS = (
    "niah_single_2", "niah_multikey_2", "niah_multivalue", "niah_multiquery",
    "vt", "fwe", "qa_1", "qa_2",
)
REGISTERED_TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
FAMILIES = {
    "niah_single_1": "retrieval", "niah_single_2": "retrieval",
    "niah_single_3": "retrieval", "niah_multikey_1": "retrieval",
    "niah_multikey_2": "retrieval", "niah_multikey_3": "retrieval",
    "niah_multivalue": "retrieval", "niah_multiquery": "retrieval",
    "vt": "tracking", "cwe": "aggregation", "fwe": "aggregation",
    "qa_1": "qa", "qa_2": "qa",
}
STAGE_COUNTS = {"P": (4, 4), "S": (8, 4), "V": (16, 8), "H": (64, 32)}
CAPS = (8192, 16384, 32768)
RESERVED_OUTPUT_TOKENS = 128
STAGE_SEED_OFFSET = {"P": 0, "S": 100_000, "V": 200_000, "H": 300_000}
QA_STAGE_ROW_OFFSET = {"P": 0, "S": 1_000, "V": 2_500, "H": 5_000}
SINGLE_EVIDENCE_TASKS = {
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
}
MULTI_EVIDENCE_TASKS = {"niah_multivalue", "niah_multiquery"}
DEPTH_TARGETS = (0.10, 0.35, 0.65, 0.90)
MULTI_DEPTH_PROFILES = (
    ("dispersed", (0.10, 0.35, 0.65, 0.90)),
    ("middle_cluster", (0.35, 0.45, 0.55, 0.65)),
)


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def substring_starts(haystack, needle):
    """Return all literal character starts, including overlapping matches."""
    if not needle:
        return []
    starts = []
    offset = 0
    while True:
        found = haystack.find(needle, offset)
        if found < 0:
            return starts
        starts.append(found)
        offset = found + 1


def token_index_for_char(offsets, char_index):
    """Map a literal evidence character to the token that contains it."""
    for token_index, (start, end) in enumerate(offsets):
        if start <= char_index < end:
            return token_index
    return None


def qa_dataset_row_count(path, dataset):
    """Count valid QA rows before invoking the upstream generator."""
    value = json.loads(Path(path).read_text())
    if dataset == "squad":
        return sum(
            not qa["is_impossible"]
            for item in value["data"]
            for paragraph in item["paragraphs"]
            for qa in paragraph["qas"]
        )
    if dataset == "hotpotqa":
        return len(value)
    raise ValueError(f"unsupported QA dataset for range validation: {dataset}")


def validate_qa_index_range(*, task, start, count, available):
    """Fail instead of entering upstream QA's exception-swallowing retry loop."""
    if start < 0 or count < 1 or start + count > available:
        raise ValueError(
            f"{task} QA source range [{start}, {start + count}) exceeds "
            f"the {available}-row dataset"
        )


def select_depth_balanced(
    task, candidates, count, cap, pilot, *, depth_targets=DEPTH_TARGETS,
):
    """Select a deterministic frozen subset with preregistered depth coverage."""
    if pilot or task not in SINGLE_EVIDENCE_TASKS | MULTI_EVIDENCE_TASKS:
        return candidates[:count]
    chosen = []
    remaining = list(candidates)
    if task in SINGLE_EVIDENCE_TASKS:
        targets = [("point", (depth_targets[i % len(depth_targets)],))
                   for i in range(count)]
    else:
        targets = [MULTI_DEPTH_PROFILES[i % len(MULTI_DEPTH_PROFILES)]
                   for i in range(count)]
    for label, target in targets:
        def loss(row):
            observed = sorted(p / cap for p in row["evidence_positions"])
            if len(observed) != len(target):
                return float("inf")
            return sum(abs(a - b) for a, b in zip(observed, target)) / len(target)
        best = min(remaining, key=lambda row: (loss(row), row["row_id"]))
        error = loss(best)
        tolerance = 0.10 if label == "point" else 0.18
        if not math.isfinite(error) or error > tolerance:
            raise SystemExit(
                f"REFUSING: {task}/{cap} cannot fill depth profile {label}; "
                f"best mean absolute depth error is {error:.3f}")
        best["depth_profile"] = label
        best["depth_target"] = list(target)
        best["depth_error_mean_abs"] = error
        chosen.append(best)
        remaining.remove(best)
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stage", choices=sorted(STAGE_COUNTS), required=True)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--caps", default=",".join(str(x) for x in CAPS),
                    help="comma-separated context caps; defaults to the Plan B 8K/16K/32K grid")
    ap.add_argument("--per-cell", type=int, default=None,
                    help="override long rows per task/length (engineering pilot only)")
    ap.add_argument("--guard-per-task", type=int, default=None,
                    help="override 8K rows per task (engineering pilot only)")
    ap.add_argument("--model-contract", choices=("llama3-8b", "generic"),
                    default="llama3-8b",
                    help="keep the historical exact Llama identity gate or use an explicitly recorded checkpoint")
    ap.add_argument("--tasks", default=",".join(TASKS),
                    help="comma-separated registered RULER task subset")
    ap.add_argument(
        "--contract", choices=("planb", "tailspline-classic"), default="planb",
        help="tailspline-classic freezes Full-13, 10 rows/cell and five depth targets",
    )
    ap.add_argument(
        "--counts-by-cap", default="",
        help="comma-separated CAP:ROWS counts; required by tailspline-classic",
    )
    ap.add_argument(
        "--depth-targets", default=",".join(str(value) for value in DEPTH_TARGETS),
        help="comma-separated fractional targets for single-answer NIAH tasks",
    )
    ap.add_argument(
        "--qa-base-offset", type=int, default=None,
        help="override the stage QA row offset to freeze a disjoint confirmation block",
    )
    ap.add_argument(
        "--selection-mode", choices=("balanced", "source-order"), default="balanced",
        help=(
            "balanced oversamples NIAH and selects registered depths; source-order keeps the "
            "requested official generator sample count without model-based selection"
        ),
    )
    args = ap.parse_args(argv)

    caps = tuple(int(x) for x in args.caps.split(",") if x.strip())
    if not caps or len(set(caps)) != len(caps) or any(
            cap <= RESERVED_OUTPUT_TOKENS for cap in caps):
        raise SystemExit("REFUSING: --caps must contain unique positive context caps")

    selected_tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
    if not selected_tasks or len(set(selected_tasks)) != len(selected_tasks) or any(
            task not in REGISTERED_TASKS for task in selected_tasks):
        raise SystemExit("REFUSING: --tasks must be a unique nonempty subset of registered tasks")
    depth_targets = tuple(float(value) for value in args.depth_targets.split(",") if value.strip())
    if (
        not depth_targets
        or len(set(depth_targets)) != len(depth_targets)
        or any(not 0.0 < value < 1.0 for value in depth_targets)
    ):
        raise SystemExit("REFUSING: --depth-targets must be unique fractions in (0,1)")
    counts_by_cap = {}
    for item in (value.strip() for value in args.counts_by_cap.split(",") if value.strip()):
        try:
            cap_text, count_text = item.split(":", 1)
            cap, count = int(cap_text), int(count_text)
        except ValueError as error:
            raise SystemExit("REFUSING: --counts-by-cap must use CAP:ROWS entries") from error
        if cap in counts_by_cap or count <= 0:
            raise SystemExit("REFUSING: --counts-by-cap has duplicate caps or nonpositive rows")
        counts_by_cap[cap] = count
    if counts_by_cap and set(counts_by_cap) != set(caps):
        raise SystemExit("REFUSING: --counts-by-cap must cover every requested cap exactly")
    if args.contract == "tailspline-classic":
        if (
            selected_tasks != REGISTERED_TASKS
            or caps != CAPS
            or counts_by_cap != {8192: 10, 16384: 10, 32768: 10}
            or depth_targets != (0.10, 0.30, 0.50, 0.70, 0.90)
            or args.per_cell is not None
            or args.guard_per_task is not None
        ):
            raise SystemExit(
                "REFUSING: tailspline-classic requires Full-13, 10 rows per 8/16/32K cell, "
                "and depths 0.10/0.30/0.50/0.70/0.90"
            )
    if args.qa_base_offset is not None and args.qa_base_offset < 0:
        raise SystemExit("REFUSING: --qa-base-offset must be nonnegative")
    qa_base_offset = (
        QA_STAGE_ROW_OFFSET[args.stage]
        if args.qa_base_offset is None else args.qa_base_offset
    )

    import yaml
    from transformers import AutoTokenizer

    model = args.model.resolve()
    upstream = args.upstream.resolve()
    out = args.out.resolve()
    config = json.loads((model / "config.json").read_text())
    identity = (
        config.get("model_type"), config.get("hidden_size"),
        config.get("num_hidden_layers"), config.get("num_attention_heads"),
        config.get("num_key_value_heads"), config.get("rope_theta"),
        config.get("max_position_embeddings"), config.get("rope_scaling"),
    )
    expected = ("llama", 4096, 32, 32, 8, 500000.0, 8192, None)
    if args.model_contract == "llama3-8b" and identity != expected:
        raise SystemExit(f"REFUSING: checkpoint identity {identity!r} != {expected!r}")
    if args.model_contract == "generic" and (
            not config.get("model_type") or int(config.get("max_position_embeddings", 0)) <= 0):
        raise SystemExit("REFUSING: generic checkpoint lacks model_type/native length")

    long_count, guard_count = STAGE_COUNTS[args.stage]
    pilot = args.per_cell is not None or args.guard_per_task is not None
    long_count = args.per_cell if args.per_cell is not None else long_count
    guard_count = args.guard_per_task if args.guard_per_task is not None else guard_count
    if long_count < 1 or guard_count < 1:
        raise SystemExit("REFUSING: every cell needs at least one row")

    def rows_for_cap(cap):
        if counts_by_cap:
            return counts_by_cap[cap]
        return guard_count if cap == 8192 else long_count

    out.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "PREPARING", "stage": args.stage,
        "role": (
            "TAILSPLINE_UNIFIED_CLASSIC" if args.contract == "tailspline-classic"
            else "ENGINEERING_PILOT" if pilot else "PLAN_B_STAGE"
        ),
        "model": str(model), "upstream_revision": UPSTREAM_REVISION,
        "long_rows_per_task_length": (
            10 if args.contract == "tailspline-classic" else long_count
        ),
        "guard_rows_per_task": (
            10 if args.contract == "tailspline-classic" else guard_count
        ),
        "reserved_output_tokens": RESERVED_OUTPUT_TOKENS,
        "model_contract": args.model_contract,
        "model_identity": identity,
        "tasks": list(selected_tasks),
        "qa_base_offset": qa_base_offset,
    }
    atomic_json(out / "manifest.json", status)

    os.environ["USE_TORCH"] = "0"
    os.environ["USE_TF"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    padding_unit = list(tok.encode(
        "Irrelevant archive padding record; it does not answer the question.\n",
        add_special_tokens=False))
    if not padding_unit:
        raise SystemExit("REFUSING: tokenizer produced no padding tokens")
    definitions = yaml.safe_load((upstream / "scripts/synthetic.yaml").read_text())
    constants = load_module(
        "planb_ruler_data_constants", upstream / "scripts/data/synthetic/constants.py")
    rows = []
    partial_rows_path = out / "rows.partial.jsonl"

    def checkpoint_rows():
        tmp = partial_rows_path.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        tmp.replace(partial_rows_path)

    for cap_index, cap in enumerate(caps):
        count = rows_for_cap(cap)
        for task_index, task in enumerate(selected_tasks):
            source_count = count
            if (
                not pilot
                and args.selection_mode == "balanced"
                and task in SINGLE_EVIDENCE_TASKS | MULTI_EVIDENCE_TASKS
            ):
                # A deterministic oversample pool makes all four registered
                # depths/profile types fillable without using model outcomes.
                # H grows only 2x (64 -> 128), while small P/S/V cells get the
                # larger absolute pool needed for repeated depth targets.
                source_count = max(
                    64,
                    (4 if args.contract == "tailspline-classic" else 2) * count,
                )
            task_config = definitions[task]
            base = constants.TASKS[task_config["task"]]
            budget = int(base["tokens_to_generate"])
            template = tok.apply_chat_template(
                [{"role": "user", "content": base["template"]}],
                tokenize=False, add_generation_prompt=True) + base.get("answer_prefix", "")
            content_char_start = template.find(base["template"])
            if content_char_start < 0:
                raise SystemExit(f"REFUSING: cannot locate user content start for {task}")
            cell_seed = (args.seed + STAGE_SEED_OFFSET[args.stage]
                         + cap_index * 10_000 + task_index * 100)
            source_path = out / "source" / str(cap) / task / "validation.jsonl"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(upstream / f"scripts/data/synthetic/{task_config['task']}.py"),
                "--save_dir", str(out / "source" / str(cap)),
                "--save_name", task, "--subset", "validation",
                "--tokenizer_path", str(model), "--tokenizer_type", "hf",
                # Generate a haystack that leaves the protocol-wide 128-token
                # reserve.  VT internally ignores ``tokens_to_generate`` when
                # few-shot examples are disabled, so the reduced sequence
                # length (rather than that argument) is the reliable contract.
                "--max_seq_length", str(cap - RESERVED_OUTPUT_TOKENS),
                "--tokens_to_generate", str(budget),
                "--num_samples", str(source_count), "--random_seed", str(cell_seed),
                "--template", template,
            ]
            for key, value in task_config["args"].items():
                command.extend(["--" + key, str(value)])
            if task.startswith("qa_"):
                # QA generation chooses the question by row index; changing the
                # distractor seed alone does not make P/S/V/H independent.
                # Reserve disjoint question ranges per stage and per cap.
                prior_cap_rows = sum(
                    rows_for_cap(prior_cap)
                    for prior_cap in caps[:cap_index])
                qa_start = qa_base_offset + prior_cap_rows
                dataset = task_config["args"]["dataset"]
                available = qa_dataset_row_count(
                    upstream / "scripts" / "data" / "synthetic" / "json" / f"{dataset}.json",
                    dataset,
                )
                validate_qa_index_range(
                    task=task, start=qa_start, count=source_count, available=available
                )
                command.extend(["--pre_samples", str(qa_start)])

            if not source_path.exists():
                old_argv, old_path, old_cwd = sys.argv, sys.path[:], os.getcwd()
                with (out / f"{cap}_{task}.log").open("w") as log:
                    try:
                        sys.argv = command[1:]
                        sys.path.insert(0, str(upstream / "scripts/data/synthetic"))
                        os.chdir(upstream)
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            runpy.run_path(command[1], run_name="__main__")
                    finally:
                        sys.argv, sys.path = old_argv, old_path
                        os.chdir(old_cwd)

            generated = [json.loads(line) for line in source_path.read_text().splitlines() if line]
            if len(generated) != source_count:
                raise SystemExit(
                    f"REFUSING: {task}/{cap} generated {len(generated)}, "
                    f"expected {source_count}")
            cell_rows = []
            for index, raw in enumerate(generated):
                text = raw["input"] + raw.get("answer_prefix", "")
                refs = raw.get("outputs") or []
                if not refs or any(not isinstance(x, str) or not x.strip() for x in refs):
                    raise SystemExit(f"REFUSING: {task}/{cap}/{index} has invalid references")
                encoded = tok(
                    text, add_special_tokens=False, return_offsets_mapping=True)
                prompt_ids = list(encoded["input_ids"])
                offsets = list(encoded["offset_mapping"])
                if not prompt_ids or len(prompt_ids) + RESERVED_OUTPUT_TOKENS > cap:
                    raise SystemExit(
                        f"REFUSING: {task}/{cap}/{index} has invalid reserved length "
                        f"{len(prompt_ids)}+{RESERVED_OUTPUT_TOKENS}")
                # Upstream RULER reduces the haystack in coarse units.  Plan B
                # permits padding only when it adds irrelevant filler and never
                # truncates evidence, question or instructions.  Insert neutral
                # tokens immediately after the user header and fill exactly to
                # the reserved generation boundary.
                occurrence_map = []
                missing_refs = []
                for ref in refs:
                    positions = substring_starts(text, ref)
                    mapped = [token_index_for_char(offsets, p) for p in positions]
                    mapped = [p for p in mapped if p is not None]
                    if not mapped:
                        missing_refs.append(ref)
                    occurrence_map.extend(mapped)
                if not task.startswith("qa_") and missing_refs:
                    raise SystemExit(
                        f"REFUSING: synthetic evidence not found in {task}/{cap}/{index}: "
                        f"{missing_refs!r}")

                padding_tokens = cap - RESERVED_OUTPUT_TOKENS - len(prompt_ids)
                if padding_tokens > 0:
                    pad = (padding_unit * ((padding_tokens + len(padding_unit) - 1)
                                           // len(padding_unit)))[:padding_tokens]
                    if args.model_contract == "llama3-8b":
                        try:
                            insert_at = prompt_ids.index(128007) + 1  # Llama end_header token
                        except ValueError:
                            insert_at = 1
                    else:
                        insert_at = token_index_for_char(offsets, content_char_start)
                        if insert_at is None:
                            raise SystemExit(
                                f"REFUSING: cannot map content start for {task}/{cap}/{index}")
                    prompt_ids = prompt_ids[:insert_at] + pad + prompt_ids[insert_at:]
                    occurrence_map = [
                        p + padding_tokens if p >= insert_at else p
                        for p in occurrence_map
                    ]
                if len(prompt_ids) + RESERVED_OUTPUT_TOKENS != cap:
                    raise SystemExit(
                        f"REFUSING: {task}/{cap}/{index} did not reach frozen cap after padding")
                # Upstream NIAH overwrites its loop index with the character
                # position of the first answer.  That value is not a scenario
                # identity and can collide across independently generated
                # prompts.  Bind the cluster to our enumerated sample and its
                # frozen token hash instead.
                semantic = digest([
                    "planb-semantic-v2", args.stage, task, cap, cell_seed,
                    index, digest(prompt_ids)])
                cell_rows.append({
                    "row_id": f"{args.stage}_{task}_{cap}_{index:04d}",
                    "example_id": f"{args.stage}_{task}_{cap}_{index:04d}",
                    "semantic_group_id": semantic,
                    "source_document_id": semantic,
                    "source_id": semantic,
                    "group_id": semantic,
                    "task": task, "family": FAMILIES[task], "length_cap": cap,
                    "prompt_ids": prompt_ids, "prompt_sha256": digest(prompt_ids),
                    "prompt_input_ids_sha256": digest(prompt_ids),
                    "input_tokens": len(prompt_ids), "actual_length": len(prompt_ids),
                    "references": refs, "gold": refs,
                    "evidence_positions": sorted(set(occurrence_map)),
                    "distractor_positions": [], "max_new_tokens": budget,
                    "split": args.stage, "scorer_revision": "ruler-upstream-string-match-v1",
                    "upstream_index": raw.get("index"), "generator_seed": cell_seed,
                    "qa_source_index": (
                        qa_base_offset + prior_cap_rows + index
                        if task.startswith("qa_") else None),
                    "irrelevant_padding_tokens": padding_tokens,
                })
            rows.extend(
                cell_rows[:count]
                if args.selection_mode == "source-order"
                else select_depth_balanced(
                    task, cell_rows, count=count, cap=cap, pilot=pilot,
                    depth_targets=depth_targets,
                )
            )
            checkpoint_rows()
            print(json.dumps({
                "task": task, "cap": cap, "rows": count,
                "generated_pool": source_count}), flush=True)

    expected_rows = len(selected_tasks) * sum(
        rows_for_cap(cap) for cap in caps)
    if len(rows) != expected_rows or len({r["row_id"] for r in rows}) != expected_rows:
        raise SystemExit(f"REFUSING: row identity/count mismatch {len(rows)} != {expected_rows}")
    if len({r["prompt_sha256"] for r in rows}) != expected_rows:
        raise SystemExit("REFUSING: duplicate prompt token sequences")

    rows_path = out / "rows.jsonl"
    tmp = rows_path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tmp.replace(rows_path)
    partial_rows_path.unlink(missing_ok=True)
    status.update({
        "status": "COMPLETE", "rows": len(rows),
        "tasks": list(selected_tasks), "caps": list(caps),
        "cell_counts": {f"{task}|{cap}": sum(
            r["task"] == task and r["length_cap"] == cap for r in rows)
            for task in selected_tasks for cap in caps},
        "rows_sha256": sha_file(rows_path),
        "model_config_sha256": sha_file(model / "config.json"),
        "tokenizer_sha256": sha_file(model / "tokenizer.json"),
        "upstream_source_sha256": {
            "synthetic_yaml": sha_file(upstream / "scripts/synthetic.yaml"),
            "constants": sha_file(upstream / "scripts/data/synthetic/constants.py"),
        },
        "independence": "distinct seed per task-length cell; unique semantic/source id per row",
        "depth_selection": ({
            "single_evidence_targets": list(depth_targets),
            "multi_evidence_profiles": {name: list(values)
                                        for name, values in MULTI_DEPTH_PROFILES},
            "selection_uses_model_outputs": False,
        } if not pilot and args.selection_mode == "balanced" else {
            "mode": "source-order",
            "selection_uses_model_outputs": False,
            "generated_rows_retained": "all requested rows in generator order",
        } if not pilot else "not applied to engineering pilot"),
        "selection_mode": args.selection_mode,
        "claim_scope": (
            "TailSpline unified Full-13 classic benchmark; task evidence only after paired controls"
            if args.contract == "tailspline-classic"
            else "engineering pilot only" if pilot
            else f"Plan B {args.stage} split; task evidence only after paired controls"
        ),
        "contract": args.contract,
        "counts_by_cap": {str(cap): rows_for_cap(cap) for cap in caps},
    })
    atomic_json(out / "manifest.json", status)
    print(json.dumps({"status": "COMPLETE", "rows": len(rows), "out": str(rows_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
