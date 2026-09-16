#!/usr/bin/env python3
"""Run frozen X3/X7 natural long-context panels with official task scorers.

The command is plan-only unless ``--execute`` is supplied.  Model generation is
delegated to the existing recovery-v2 evaluator; this wrapper owns portable
identity, strict prefix resume, the shared single-GPU lock, and natural-task
scoring.  In particular, no RULER substring score is accepted as an X3/X7
result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import string
import subprocess
from typing import Iterable


ARMS = ("tailspline", "mrpro")
SCORE_CONTRACTS = {
    "longbench_v2_mc_direct_answer_v1": "accuracy",
    "infinitebench_en_dia_accuracy_v1": "accuracy",
    "infinitebench_en_qa_rouge_f1_v1": "qa_f1",
}
SCORER_SOURCES = {
    "longbench_v2_mc_direct_answer_v1": (
        "THUDM/LongBench pred.py:extract_answer and result.py:judge"
    ),
    "infinitebench_en_dia_accuracy_v1": (
        "OpenBMB/InfiniteBench src/compute_scores.py:"
        "get_score_one_longdialogue_qa_eng"
    ),
    "infinitebench_en_qa_rouge_f1_v1": (
        "OpenBMB/InfiniteBench src/compute_scores.py:"
        "get_score_one_longbook_qa_eng"
    ),
}
CONTRACT_TASKS = {
    "longbench_v2_mc_direct_answer_v1": ("longbench_v2", None),
    "infinitebench_en_dia_accuracy_v1": (
        "infinitebench", "longdialogue_qa_eng",
    ),
    "infinitebench_en_qa_rouge_f1_v1": (
        "infinitebench", "longbook_qa_eng",
    ),
}
GPU_LOCK = Path("/tmp/hybrid-rope-gpu0.lock")
RUNNER_CONTRACT = "strong-natural-long-run-v1"
REPORT_CONTRACT = "strong-natural-long-paired-report-v1"
BOOTSTRAP_SEED = 20261105
BOOTSTRAP_DRAWS = 2000


@contextmanager
def acquire_gpu_lock(path: Path = GPU_LOCK, *, inherited_fd: int = 9):
    """Own the GPU lock, reusing an outer shell's inherited fd when present.

    Queue wrappers use ``exec 9>...; flock -n 9`` before invoking this module.
    Opening the same path again would create a distinct open-file description
    and can conflict with that already-held lock.  Duplicating fd 9 preserves
    the same open-file description; a fresh invocation still opens and locks
    the path normally.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = None
    try:
        inherited = os.fstat(inherited_fd)
        target = os.stat(path)
        if (inherited.st_dev, inherited.st_ino) == (target.st_dev, target.st_ino):
            descriptor = os.dup(inherited_fd)
    except (OSError, ValueError):
        pass
    if descriptor is None:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o666)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Hybrid-RoPE evaluation owns GPU 0") from error
        yield
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _manifest_inputs_receipt(manifest: dict) -> tuple[str | None, str | None]:
    """Return the portable input path hint and required SHA receipt."""
    candidates = []
    if isinstance(manifest.get("inputs"), dict):
        candidates.append(manifest["inputs"])
    if isinstance(manifest.get("input"), dict):
        candidates.append(manifest["input"])
    if isinstance(manifest.get("files"), dict):
        for key in ("inputs", "input_rows", "selected_inputs"):
            if isinstance(manifest["files"].get(key), dict):
                candidates.append(manifest["files"][key])
    for receipt in candidates:
        digest = receipt.get("sha256") or receipt.get("inputs_sha256")
        hint = receipt.get("path") or receipt.get("path_hint")
        if digest:
            return hint, digest
    return manifest.get("inputs_path"), manifest.get("inputs_sha256")


def resolve_inputs(data_root: Path, manifest: dict) -> tuple[Path, str]:
    hint, expected_sha = _manifest_inputs_receipt(manifest)
    relative = Path(hint or "inputs.jsonl")
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("data manifest input path must be portable and relative")
    path = Path(data_root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("data manifest lacks a valid inputs SHA256 receipt")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError("prepared inputs differ from the data manifest SHA256")
    return path, actual_sha


def parse_lengths(values: Iterable[str | int]) -> tuple[int, ...]:
    parsed = []
    for value in values:
        parsed.extend(int(item) for item in str(value).split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)) or any(item <= 0 for item in parsed):
        raise ValueError("lengths must be unique positive integers")
    return tuple(parsed)


def _prompt_digest(ids: list[int]) -> str:
    encoded = json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_prepared_data(
    manifest: dict,
    rows: list[dict],
    *,
    model_id: str,
    scale: float,
    lengths: tuple[int, ...],
    rows_per_task: int,
    benchmark: str | None,
) -> None:
    if manifest.get("status") != "COMPLETE":
        raise ValueError("natural data manifest is not COMPLETE")
    if manifest.get("model_id") != model_id:
        raise ValueError("prepared model_id differs from the requested model")
    if not math.isclose(float(manifest.get("scale", float("nan"))), scale):
        raise ValueError("prepared scale differs from the requested scale")
    prepared_lengths = parse_lengths(manifest.get("lengths", []))
    if prepared_lengths != lengths:
        raise ValueError("prepared lengths differ from the requested lengths")
    prepared_limit = manifest.get("rows_per_task")
    if prepared_limit is None:
        if manifest.get("benchmark") == "longbench_v2" and rows_per_task != 0:
            raise ValueError("uncapped LongBench-v2 requires --rows-per-task 0")
    elif int(prepared_limit) != rows_per_task:
        raise ValueError("prepared rows-per-task differs from the requested cap")
    if not rows:
        raise ValueError("prepared natural panel is empty")

    requested_lengths = set(lengths)
    seen = set()
    counts = Counter()
    seen_benchmarks = set()
    for row in rows:
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            raise ValueError(f"missing or duplicate row_id: {row_id!r}")
        seen.add(row_id)
        row_benchmark = row.get("benchmark")
        task = row.get("task")
        contract = row.get("score_contract")
        if contract not in SCORE_CONTRACTS:
            raise ValueError(f"unsupported natural score contract: {contract!r}")
        expected_benchmark, expected_task = CONTRACT_TASKS[contract]
        if row_benchmark != expected_benchmark or (
            expected_task is not None and task != expected_task
        ):
            raise ValueError(f"row {row_id} does not match its score contract")
        seen_benchmarks.add(row_benchmark)
        prompt = row.get("prompt_ids")
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(type(token) is not int or token < 0 for token in prompt)
        ):
            raise ValueError(f"row {row_id} has invalid prompt_ids")
        if int(row.get("input_tokens", -1)) != len(prompt):
            raise ValueError(f"row {row_id} input token count differs")
        prompt_sha = row.get("prompt_sha256")
        if prompt_sha != _prompt_digest(prompt):
            raise ValueError(f"row {row_id} prompt SHA256 differs")
        cap = int(row.get("length_cap", 0))
        budget = int(row.get("max_new_tokens", 0))
        if cap not in requested_lengths or budget <= 0 or len(prompt) + budget > cap:
            raise ValueError(f"row {row_id} exceeds or leaves the requested length grid")
        references = row.get("references")
        if not isinstance(references, list) or not references or any(
            not isinstance(reference, str) or not reference.strip()
            for reference in references
        ):
            raise ValueError(f"row {row_id} has invalid references")
        if contract == "longbench_v2_mc_direct_answer_v1" and any(
            reference not in "ABCD" or len(reference) != 1 for reference in references
        ):
            raise ValueError(f"row {row_id} has a non-MC LongBench-v2 answer")
        counts[(row_benchmark, str(task))] += 1

    if benchmark and seen_benchmarks != {benchmark}:
        raise ValueError("prepared rows do not match --benchmark")
    manifest_benchmark = manifest.get("benchmark")
    if manifest_benchmark and seen_benchmarks != {manifest_benchmark}:
        raise ValueError("prepared rows do not match the manifest benchmark")
    if rows_per_task < 0:
        raise ValueError("rows-per-task cannot be negative")
    if prepared_limit is not None and rows_per_task and any(
        count > rows_per_task for count in counts.values()
    ):
        raise ValueError("prepared panel exceeds rows-per-task")
    if "rows" in manifest and int(manifest["rows"]) != len(rows):
        raise ValueError("prepared row count differs from the manifest")
    selected_rows = (manifest.get("summary") or {}).get("selected_rows")
    if selected_rows is not None and int(selected_rows) != len(rows):
        raise ValueError("prepared row count differs from the manifest summary")


def extract_longbench_v2_answer(text: str) -> str | None:
    """Exact local adapter of LongBench-v2's direct-answer parser."""
    response = str(text).replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    return match.group(1) if match else None


def normalize_infinitebench_answer(text: str) -> str:
    """Official InfiniteBench English QA normalization."""
    lowered = str(text).lower()
    unpunctuated = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    without_articles = re.sub(r"\b(a|an|the)\b", " ", unpunctuated)
    return " ".join(without_articles.split())


def infinitebench_qa_f1(text: str, references: list[str]) -> float:
    prediction = normalize_infinitebench_answer(text).split()
    best = 0.0
    for reference in references:
        target = normalize_infinitebench_answer(reference).split()
        overlap = sum((Counter(prediction) & Counter(target)).values())
        if overlap:
            precision = overlap / len(prediction)
            recall = overlap / len(target)
            best = max(best, 2.0 * precision * recall / (precision + recall))
    return best


def score_natural_output(row: dict, text: str) -> dict:
    """Score one output only with the task's declared official adapter."""
    contract = row.get("score_contract")
    references = list(row.get("references") or [])
    if contract not in SCORE_CONTRACTS:
        raise ValueError(f"unsupported natural score contract: {contract!r}")
    if not references:
        raise ValueError("natural row has no references")
    text = str(text)
    extracted = None
    if contract == "longbench_v2_mc_direct_answer_v1":
        extracted = extract_longbench_v2_answer(text)
        score = float(extracted in references) if extracted is not None else 0.0
    elif contract == "infinitebench_en_dia_accuracy_v1":
        normalized = text.strip().upper()
        score = float(bool(normalized) and any(
            reference.strip().upper() in normalized
            for reference in references if reference.strip()
        ))
    elif contract == "infinitebench_en_qa_rouge_f1_v1":
        # InfiniteBench calls this English QA token F1 despite the historical
        # README's "ROUGE F1" wording; this mirrors compute_scores.py.
        score = float(infinitebench_qa_f1(text, references))
    else:  # pragma: no cover - guarded above and kept explicit for reviewers.
        raise AssertionError(contract)
    return {
        "official_score": score,
        "official_metric": SCORE_CONTRACTS[contract],
        "official_score_contract": contract,
        "parsed_answer": extracted,
    }


def classify_answer(output: str, *, parsed: str | None, hit_cap: bool) -> str:
    if not str(output).strip():
        return "EMPTY"
    if parsed is None and hit_cap:
        return "TRUNCATED_OR_UNPARSEABLE"
    if parsed is None:
        return "UNKNOWN_OR_UNPARSEABLE"
    return "PARSED"


def _table_command(args, method: str, table_path: Path) -> list[str]:
    return [
        str(args.python), "-m",
        "experiments.fixed_rope_three_interfaces_20260913.tables", "analytic",
        "--config", str(Path(args.model) / "config.json"),
        "--method", method,
        "--scale", str(args.scale),
        "--candidate-id", f"strong_{args.model_id}_s{args.scale:g}_{method}",
        "--model-id", args.model_id,
        "--role", "candidate" if method == "tailspline" else "baseline",
        "--changed-variable", "internal_frequency_allocation",
        "--out", str(table_path),
    ]


def _run_command(
    args, *, arm: str, inputs_path: Path, table_path: Path, run_path: Path,
) -> list[str]:
    command = [
        str(args.python), "-m",
        "experiments.olmo_recovery_20260912.recovery_v2_eval",
        "--data", str(args.data_manifest),
        "--model", str(args.model),
        "--arm", "Native",
        "--extra-panel", str(inputs_path),
        "--only-extra-panels", "--skip-lm",
        "--batch-size", "1",
        "--prefill-chunk-size", str(args.prefill_chunk_size),
        "--static-table-json", str(table_path),
        "--table-label", f"strong_{args.model_id}_s{args.scale:g}_natural_{arm}",
        "--out", str(run_path), "--execute",
    ]
    for length in args.lengths:
        command.extend(("--length-cap", str(length)))
    return command


def _portable_run_contract(
    args, *, arm: str, rows: list[dict], data_sha: str, inputs_sha: str,
    table_sha: str,
) -> dict:
    return {
        "status": RUNNER_CONTRACT,
        "arm": arm,
        "model_id": args.model_id,
        "model_config_sha256": sha256_file(Path(args.model) / "config.json"),
        "scale": float(args.scale),
        "lengths": list(args.lengths),
        "rows_per_task": int(args.rows_per_task),
        "benchmark": args.benchmark,
        "data_manifest_sha256": data_sha,
        "inputs_sha256": inputs_sha,
        "table_receipt_sha256": table_sha,
        "row_ids": [row["row_id"] for row in rows],
        "batch_size": 1,
        "prefill_chunk_size": int(args.prefill_chunk_size),
        "generation_backend": "recovery_v2_eval",
        "natural_score_source": "posthoc_declared_official_adapter",
        "ruler_contains_is_natural_score": False,
    }


def _ensure_contract(path: Path, expected: dict) -> None:
    if path.exists():
        if read_json(path) != expected:
            raise ValueError(f"existing run has a different wrapper contract: {path}")
    else:
        owned_outputs = (
            "contract.json", "generations.jsonl", "status.json", "live.json",
            "scored_generations.jsonl",
        )
        if any((path.parent / name).exists() for name in owned_outputs):
            raise ValueError(
                f"existing run outputs lack the frozen wrapper contract: {path.parent}"
            )
        atomic_json(path, expected)


def _validate_table_receipt(args, arm: str, path: Path) -> None:
    receipt = read_json(path)
    table = receipt.get("table", receipt)
    construction = table.get("construction", {})
    expected_method = (
        "tailspline_exact_finite_grid" if arm == "tailspline"
        else "mrpro"
    )
    if (
        receipt.get("model_id") != args.model_id
        or not math.isclose(float(receipt.get("scale", float("nan"))), args.scale)
        or receipt.get("role") != ("candidate" if arm == "tailspline" else "baseline")
        or construction.get("method") != expected_method
        or not isinstance(table.get("values_float32"), list)
        or not table["values_float32"]
        or not math.isfinite(float(table.get("gain", float("nan"))))
    ):
        raise ValueError(f"existing {arm} table differs from the frozen contract")


def _ensure_table(args, arm: str, path: Path, *, repo_root: Path, env: dict) -> None:
    if not path.exists():
        subprocess.run(_table_command(args, arm, path), cwd=repo_root, env=env, check=True)
    _validate_table_receipt(args, arm, path)


def _validate_complete_run(run_path: Path, rows: list[dict]) -> list[dict]:
    status_path = run_path / "status.json"
    generation_path = run_path / "generations.jsonl"
    if not status_path.is_file() or not generation_path.is_file():
        raise ValueError(f"completed arm lacks status or generations: {run_path}")
    expected_status = {"status": "COMPLETE", "rows": len(rows), "lm_rows": 0}
    if read_json(status_path) != expected_status:
        raise ValueError(f"arm status differs from the frozen natural panel: {run_path}")
    generations = read_jsonl(generation_path)
    if len(generations) != len(rows):
        raise ValueError(f"arm generations are incomplete: {run_path}")
    for expected, actual in zip(rows, generations):
        if actual.get("row_id") != expected["row_id"]:
            raise ValueError(f"arm generations are not the expected exact prefix: {run_path}")
        if "output_text" not in actual or "generated_ids" not in actual:
            raise ValueError(f"arm generation does not preserve full output: {run_path}")
    return generations


def _run_is_complete(run_path: Path, rows: list[dict]) -> bool:
    if not (run_path / "status.json").exists():
        return False
    _validate_complete_run(run_path, rows)
    return True


def score_arm(run_path: Path, rows: list[dict], arm: str) -> list[dict]:
    generations = _validate_complete_run(run_path, rows)
    scored = []
    for prepared, generated in zip(rows, generations):
        output = str(generated["output_text"])
        official = score_natural_output(prepared, output)
        record = dict(generated)  # preserve the complete raw generation record
        record.update({key: prepared.get(key) for key in (
            "benchmark", "task", "domain", "sub_domain", "source_id",
            "source_index", "source_cluster_id", "source_row_sha256",
            "input_tokens", "length_cap", "length_bucket", "references",
            "score_contract", "prompt_sha256",
        )})
        record.update(official)
        record["arm"] = arm
        record["natural_answer_status"] = classify_answer(
            output,
            parsed=official["parsed_answer"] if official["official_metric"] == "accuracy"
            and prepared["score_contract"] == "longbench_v2_mc_direct_answer_v1"
            else ("OFFICIAL_NON_MC" if output.strip() else None),
            hit_cap=bool(generated.get("hit_cap")),
        )
        # A pre-existing synthetic scorer field is retained as raw provenance
        # but explicitly cannot influence official_score.
        record["ruler_contains_used_for_natural_score"] = False
        scored.append(record)
    atomic_jsonl(run_path / "scored_generations.jsonl", scored)
    return scored


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("cannot take a percentile of an empty sample")
    position = (len(values) - 1) * q
    left = int(position)
    right = min(left + 1, len(values) - 1)
    weight = position - left
    return values[left] * (1.0 - weight) + values[right] * weight


def _cluster_ci(rows: list[dict], *, draws: int, seed: int) -> list[float]:
    clusters = defaultdict(list)
    for row in rows:
        cluster = row.get("source_cluster_id") or row.get("source_id") or row["row_id"]
        clusters[str(cluster)].append(float(row["delta_tailspline_minus_mrpro"]))
    keys = sorted(clusters)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [clusters[rng.choice(keys)] for _ in keys]
        flattened = [value for cluster in selected for value in cluster]
        samples.append(sum(flattened) / len(flattened))
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def _summarize_group(rows: list[dict], *, seed: int) -> dict:
    metrics = sorted({row["official_metric"] for row in rows})
    result = {"rows": len(rows), "metrics": metrics}
    if len(metrics) != 1:
        result.update({
            "pooled_score": None,
            "reason": "mixed metric contracts are not pooled",
        })
        return result
    tail = sum(row["tailspline_score"] for row in rows) / len(rows)
    pro = sum(row["mrpro_score"] for row in rows) / len(rows)
    result.update({
        "metric": metrics[0],
        "tailspline": tail,
        "mrpro": pro,
        "delta_tailspline_minus_mrpro": tail - pro,
        "cluster_bootstrap_ci95": _cluster_ci(rows, draws=BOOTSTRAP_DRAWS, seed=seed),
        "paired_outcomes": {
            "tailspline_higher": sum(row["delta_tailspline_minus_mrpro"] > 0 for row in rows),
            "tie": sum(row["delta_tailspline_minus_mrpro"] == 0 for row in rows),
            "mrpro_higher": sum(row["delta_tailspline_minus_mrpro"] < 0 for row in rows),
        },
    })
    return result


def build_report(args, rows: list[dict], scored: dict[str, list[dict]], *, data_sha: str, inputs_sha: str) -> dict:
    arm_maps = {
        arm: {row["row_id"]: row for row in values}
        for arm, values in scored.items()
    }
    expected = [row["row_id"] for row in rows]
    if any(list(mapping) != expected for mapping in arm_maps.values()):
        raise ValueError("natural arms do not contain the exact same ordered prompts")
    paired = []
    for prepared in rows:
        row_id = prepared["row_id"]
        tail = arm_maps["tailspline"][row_id]
        pro = arm_maps["mrpro"][row_id]
        if tail["official_score_contract"] != pro["official_score_contract"]:
            raise ValueError("paired arms used different score contracts")
        paired.append({
            **{key: prepared.get(key) for key in (
                "row_id", "benchmark", "task", "domain", "sub_domain",
                "source_id", "source_cluster_id", "length_cap", "length_bucket",
            )},
            "official_metric": tail["official_metric"],
            "tailspline_score": float(tail["official_score"]),
            "mrpro_score": float(pro["official_score"]),
            "delta_tailspline_minus_mrpro": float(
                tail["official_score"] - pro["official_score"]
            ),
        })

    def grouped(key):
        buckets = defaultdict(list)
        for row in paired:
            buckets[str(row.get(key) or "UNSPECIFIED")].append(row)
        return {
            name: _summarize_group(values, seed=BOOTSTRAP_SEED + index + 1)
            for index, (name, values) in enumerate(sorted(buckets.items()))
        }

    return {
        "status": "COMPLETE",
        "contract": REPORT_CONTRACT,
        "identity": {
            "model_id": args.model_id,
            "scale": float(args.scale),
            "lengths": list(args.lengths),
            "rows_per_task": int(args.rows_per_task),
            "rows": len(rows),
            "data_manifest_sha256": data_sha,
            "inputs_sha256": inputs_sha,
            "arms": list(ARMS),
            "paired_same_prompts": True,
        },
        "scoring": {
            "contracts": sorted({row["score_contract"] for row in rows}),
            "official_adapter_sources": {
                contract: SCORER_SOURCES[contract]
                for contract in sorted({row["score_contract"] for row in rows})
            },
            "ruler_contains_used": False,
            "full_outputs_retained": True,
            "empty_unknown_or_unparseable_answers_score_zero": True,
            "cap_hit_is_reported_and_does_not_replace_official_scoring": True,
        },
        "overall": _summarize_group(paired, seed=BOOTSTRAP_SEED),
        "by_benchmark": grouped("benchmark"),
        "by_task": grouped("task"),
        "by_length_bucket": grouped("length_bucket"),
        "paired_rows": paired,
    }


def _repo_environment(repo_root: Path) -> dict:
    environment = dict(os.environ)
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repo_root) + (os.pathsep + current if current else "")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def execute(args) -> dict:
    config = Path(args.model) / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    if not Path(args.python).is_file():
        raise FileNotFoundError(args.python)
    manifest = read_json(args.data_manifest)
    inputs_path, inputs_sha = resolve_inputs(args.data_root, manifest)
    rows = read_jsonl(inputs_path)
    validate_prepared_data(
        manifest, rows, model_id=args.model_id, scale=args.scale,
        lengths=args.lengths, rows_per_task=args.rows_per_task,
        benchmark=args.benchmark,
    )
    data_sha = sha256_file(args.data_manifest)
    out = Path(args.out)
    repo_root = Path(__file__).resolve().parents[2]
    env = _repo_environment(repo_root)

    with acquire_gpu_lock():
        out.mkdir(parents=True, exist_ok=True)
        (out / "tables").mkdir(exist_ok=True)
        (out / "runs").mkdir(exist_ok=True)
        (out / "logs").mkdir(exist_ok=True)
        completed = {}
        for arm in ARMS:
            table_path = out / "tables" / f"{arm}.json"
            _ensure_table(args, arm, table_path, repo_root=repo_root, env=env)
            table_sha = sha256_file(table_path)
            run_path = out / "runs" / arm
            expected_contract = _portable_run_contract(
                args, arm=arm, rows=rows, data_sha=data_sha, inputs_sha=inputs_sha,
                table_sha=table_sha,
            )
            run_path.mkdir(parents=True, exist_ok=True)
            _ensure_contract(run_path / "wrapper_contract.json", expected_contract)
            if not _run_is_complete(run_path, rows):
                log_path = out / "logs" / f"{arm}.log"
                with log_path.open("a") as log:
                    subprocess.run(
                        _run_command(
                            args, arm=arm, inputs_path=inputs_path,
                            table_path=table_path, run_path=run_path,
                        ),
                        cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT,
                        check=True,
                    )
            completed[arm] = score_arm(run_path, rows, arm)
        report = build_report(
            args, rows, completed, data_sha=data_sha, inputs_sha=inputs_sha,
        )
        atomic_json(out / "report.json", report)
        atomic_json(out / "status.json", {
            "status": "COMPLETE", "rows_per_arm": len(rows), "arms": list(ARMS),
        })
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--lengths", nargs="+", required=True)
    parser.add_argument("--rows-per-task", type=int, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--benchmark", choices=("longbench-v2", "longbench_v2", "infinitebench"),
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)
    parser.add_argument("--execute", action="store_true")
    return parser


def plan(args) -> dict:
    return {
        "status": "PLAN_ONLY",
        "execute": False,
        "forward_passes_started": 0,
        "model_id": args.model_id,
        "scale": float(args.scale),
        "lengths": list(args.lengths),
        "rows_per_task": int(args.rows_per_task),
        "benchmark": args.benchmark,
        "arm_order": list(ARMS),
        "single_gpu_lock": str(GPU_LOCK),
        "strict_resume": True,
        "full_outputs": True,
        "natural_score_contracts": sorted(SCORE_CONTRACTS),
        "ruler_contains_used": False,
        "next_action": "rerun with --execute after reviewing this frozen plan",
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.lengths = parse_lengths(args.lengths)
    if args.benchmark:
        args.benchmark = args.benchmark.replace("-", "_")
    if args.scale <= 1 or not math.isfinite(args.scale):
        raise ValueError("scale must be finite and exceed one")
    if args.rows_per_task < 0 or args.prefill_chunk_size < 0:
        raise ValueError("row and prefill limits must be nonnegative")
    if not args.execute:
        print(json.dumps(plan(args), indent=2, sort_keys=True))
        return
    report = execute(args)
    print(json.dumps({
        "status": report["status"],
        "rows": report["identity"]["rows"],
        "report": str(Path(args.out) / "report.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
