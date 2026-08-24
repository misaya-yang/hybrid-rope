#!/usr/bin/env python3
"""Held-out LongBench QA evaluation for frozen zero-training RoPE operators."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import string
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from scripts.analysis.export_uniqueness_budgeted_tables import (
    build_default_tables,
    float32_sha256,
)
from scripts.eval.target_free_formal_eval import set_target_aware_factor
from scripts.lib.rope.length_conditioned_budgeted import (
    install_length_conditioned_rope,
    matched_attention_scaling,
    select_observed_session_factor,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    configure_cuda,
    configure_ruler_flash_attention,
    greedy_generate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    endpoint_geo_inv_freq,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import load_model
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer import (
    apply_frequency,
    official_yarn_config,
    verify_official_yarn,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_4k_stage_a import (
    ready_checkpoint_digest,
)
STATUS = "FROZEN_ZERO_TRAINING_2WIKI_COMPLETE"
GENERIC_STATUS = "FROZEN_ZERO_TRAINING_LONGBENCH_QA_COMPLETE"
OFFICIAL_2WIKI_MEMBER = "2wikimqa.jsonl"
OFFICIAL_2WIKI_PROMPT = (
    "Answer the question based on the given passages. Only give me the "
    "answer and do not output any other words.\n\n"
    "The following are given passages.\n{context}\n\n"
    "Answer the question based on the given passages. Only give me the "
    "answer and do not output any other words.\n\n"
    "Question: {question}\nAnswer:"
)
TASK_PROMPTS = {
    "2wikimqa": OFFICIAL_2WIKI_PROMPT,
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the "
        "article, write \"unanswerable\". If the question is a yes/no question, "
        "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nArticle: {context}\n\n Answer the question based on the "
        "above article as concisely as you can, using a single phrase or sentence "
        "if possible. If the question cannot be answered based on the information "
        "in the article, write \"unanswerable\". If the question is a yes/no "
        "question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide "
        "any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
}
TASK_GENERATION_TOKENS = {"2wikimqa": 32, "qasper": 128}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_bytes() if path.exists() else b""
    line = (
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        mode="wb",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(previous)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def normalize_2wiki(text: str) -> str:
    value = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    value = "".join(
        character for character in value
        if character not in string.punctuation
    )
    return " ".join(value.split())


def token_f1(prediction: str, references: Sequence[str]) -> float:
    pred = normalize_2wiki(prediction).split()
    best = 0.0
    for reference in references:
        gold = normalize_2wiki(reference).split()
        if not pred or not gold:
            continue
        overlap = sum(
            min(pred.count(token), gold.count(token))
            for token in set(pred)
        )
        if overlap:
            precision = overlap / len(pred)
            recall = overlap / len(gold)
            best = max(best, 2 * precision * recall / (precision + recall))
    return float(best)


def normalized_exact(prediction: str, references: Sequence[str]) -> float:
    value = normalize_2wiki(prediction)
    return float(any(value == normalize_2wiki(ref) for ref in references))


def load_official_2wiki_rows(path: Path, *, task: str = "2wikimqa") -> dict[str, Any]:
    member_name = f"{task}.jsonl"
    with zipfile.ZipFile(path.resolve()) as archive:
        candidates = [
            name for name in archive.namelist()
            if name == member_name or name.endswith("/" + member_name)
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one official {task} member, got {candidates}"
            )
        member = candidates[0]
        rows = [
            json.loads(line)
            for line in archive.read(member).decode("utf-8").splitlines()
            if line.strip()
        ]
    if len(rows) != 200:
        raise RuntimeError(
            f"LongBench {task} must contain 200 rows, got {len(rows)}"
        )
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("input"), str)
            or not isinstance(row.get("answers"), list)
        ):
            raise RuntimeError(f"malformed official 2Wiki row {index}")
    return {
        "dataset": f"THUDM/LongBench:{task}",
        "zip_sha256": sha256_file(path),
        "member": member,
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
    }


def _chat_ids(tokenizer: Any, prompt: str) -> list[int]:
    value = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(value, Mapping):
        value = value.get("input_ids")
        if value is None:
            raise RuntimeError("chat template output lacks input_ids")
    if getattr(value, "ndim", None) == 2:
        return [int(item) for item in value[0].tolist()]
    return [int(item) for item in value]


def _fit_chat_prompt(
    tokenizer: Any,
    prompt: str,
    length: int,
    max_new_tokens: int,
) -> tuple[list[int], bool]:
    raw = list(tokenizer(prompt, add_special_tokens=False).input_ids)
    ids = _chat_ids(tokenizer, prompt)
    if len(ids) + max_new_tokens <= length:
        return ids, False
    candidate = raw
    while True:
        content = tokenizer.decode(
            candidate,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        ids = _chat_ids(tokenizer, content)
        if len(ids) + max_new_tokens <= length:
            return ids, True
        allowed = len(candidate) - (len(ids) + max_new_tokens - length)
        if allowed <= 64:
            raise RuntimeError(f"2Wiki query cannot fit physical L{length}")
        head = allowed // 2
        candidate = raw[:head] + raw[-(allowed - head):]


def build_2wiki_jobs(
    tokenizer: Any,
    package: Mapping[str, Any],
    *,
    lengths: Sequence[int],
    task: str = "2wikimqa",
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for index, source in enumerate(package["rows"]):
        prompt = TASK_PROMPTS[task].format(
            context=str(source["context"]),
            input=str(source["input"]),
            question=str(source["input"]),
        )
        source_hash = canonical_sha256(source)
        for length in lengths:
            ids, truncated = _fit_chat_prompt(
                tokenizer,
                prompt,
                int(length),
                max_new_tokens=TASK_GENERATION_TOKENS[task],
            )
            row_id = f"{task}:{source_hash}:L{int(length)}"
            jobs.append({
                "row_id": row_id,
                "nominal_length": int(length),
                "input_ids": ids,
                "references": [str(value) for value in source["answers"]],
                "max_new_tokens": TASK_GENERATION_TOKENS[task],
                "truncated": truncated,
                "row_sha256": canonical_sha256({
                    "row_id": row_id,
                    "source": source_hash,
                    "length": int(length),
                    "input_ids": ids,
                }),
                "source_row_sha256": source_hash,
                "source_index": index,
            })
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--longbench-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASK_PROMPTS), default="2wikimqa")
    parser.add_argument("--length", type=int, choices=(4096, 8192, 16384), required=True)
    parser.add_argument(
        "--frequency",
        choices=(
            "native",
            "official_yarn",
            "budgeted",
            "session_adaptive",
            "session_binary_s4",
            "external_static",
            "external_session",
        ),
        required=True,
    )
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--table-name")
    parser.add_argument("--expected-table-sha256")
    parser.add_argument("--long-attention-scaling", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def _load_completed(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = str(row["row_id"])
        if row_id in rows:
            raise RuntimeError(f"duplicate completed 2Wiki row: {row_id}")
        rows[row_id] = row
    return rows


def main() -> int:
    args = parse_args()
    if args.frequency == "native" and (args.length != 4096 or args.factor != 1.0):
        raise RuntimeError("Native owner is registered only at 4K/factor one")
    if args.frequency == "official_yarn" and args.factor <= 1.0:
        raise RuntimeError("official YaRN requires factor > 1")
    if args.frequency in {"budgeted", "external_static", "external_session"}:
        if args.table is None or not args.table_name or args.factor <= 1.0:
            raise RuntimeError("external table method requires table, name, and factor > 1")
        if int(round(4096 * float(args.factor))) != int(args.length):
            raise RuntimeError("external table factor/length mismatch")
        if args.frequency in {"external_static", "external_session"}:
            if not args.expected_table_sha256:
                raise RuntimeError("external method requires --expected-table-sha256")
            if (
                not math.isfinite(float(args.long_attention_scaling))
                or float(args.long_attention_scaling) <= 0.0
            ):
                raise RuntimeError("external method requires positive attention scaling")
    elif args.table is not None or args.table_name is not None:
        raise RuntimeError("only external-table methods accept a frozen table")
    if (
        args.frequency in {"session_adaptive", "session_binary_s4"}
        and float(args.factor) != 1.0
    ):
        raise RuntimeError("session selection does not accept a target factor")
    if not 1 <= int(args.limit) <= 200:
        raise RuntimeError("2Wiki limit must be in [1,200]")

    external_values = None
    if args.frequency in {"external_static", "external_session"}:
        external_values = np.load(args.table.resolve(), allow_pickle=False)
        expected_native = endpoint_geo_inv_freq().numpy()
        if (
            external_values.dtype != np.float32
            or external_values.shape != expected_native.shape
            or not np.isfinite(external_values).all()
            or not np.all(external_values[:-1] > external_values[1:])
            or external_values[0] != np.float32(expected_native[0])
            or external_values[-1] != np.float32(expected_native[-1])
            or float32_sha256(external_values)
            != str(args.expected_table_sha256)
        ):
            raise RuntimeError("external table identity drift")
        external_values = np.ascontiguousarray(
            external_values,
            dtype="<f4",
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha = ready_checkpoint_digest(checkpoint, args.ready_receipt.resolve())
    package = load_official_2wiki_rows(
        args.longbench_zip.resolve(),
        task=str(args.task),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    jobs = build_2wiki_jobs(
        tokenizer,
        package,
        lengths=(int(args.length),),
        task=str(args.task),
    )[
        : int(args.limit)
    ]
    jobs_sha = canonical_sha256(
        [{key: value for key, value in job.items() if key != "input_ids"} for job in jobs]
    )

    configure_cuda()
    model_config = (
        official_yarn_config(
            checkpoint,
            factor=float(args.factor),
            original_max_position_embeddings=4096,
        )
        if args.frequency == "official_yarn"
        else None
    )
    model = load_model(checkpoint, config=model_config)
    native_inv = model.model.rotary_emb.inv_freq.detach().cpu().float().clone()
    session_tables = {}
    state = None
    if args.frequency == "official_yarn":
        method = verify_official_yarn(model, model_config)
    elif args.frequency == "native":
        method = apply_frequency(model, "native")
    elif args.frequency in {"session_adaptive", "session_binary_s4", "external_session"}:
        session_tables = build_default_tables()
        binary = args.frequency in {"session_binary_s4", "external_session"}
        if args.frequency == "external_session":
            assert external_values is not None
            if not np.array_equal(
                external_values[[0, -1]], native_inv.numpy()[[0, -1]]
            ):
                raise RuntimeError("runtime Native support differs from external receipt")
            session_tables = {4.0: external_values}
        method = {
            "method": (
                "native_or_external_table_observed_request_rope"
                if args.frequency == "external_session"
                else "native_or_frozen_s4_observed_request_rope"
                if args.frequency == "session_binary_s4"
                else "observed_request_session_adaptive_budgeted_rope"
            ),
            "selection": (
                "Native within L_native; frozen external table beyond L_native"
                if args.frequency == "external_session"
                else "Native within L_native; frozen s4 beyond L_native"
                if args.frequency == "session_binary_s4"
                else "smallest frozen profile covering prefill_tokens plus max_new_tokens"
            ),
            "requires_L_target": False,
            "runtime_requires_request_target": False,
            "construction_horizon_bound_by_external_receipt": (
                args.frequency == "external_session"
            ),
            "construction_table_factor": (
                float(args.factor)
                if args.frequency == "external_session"
                else None
            ),
            "native_context_length": 4096,
            "supported_factors": [1, 4] if binary else [1, 2, 4],
            "cache_policy": "profile fixed before prefill for the KV-cache lifetime",
            "native_inv_freq_sha256_float32": float32_sha256(native_inv.numpy()),
            "long_table_sha256_float32": {
                str(int(factor)): float32_sha256(values)
                for factor, values in sorted(session_tables.items())
            },
            "table_name": str(args.table_name) if args.frequency == "external_session" else None,
            "expected_table_sha256_float32": (
                str(args.expected_table_sha256)
                if args.frequency == "external_session"
                else None
            ),
            "table_file_sha256": (
                sha256_file(args.table.resolve())
                if args.frequency == "external_session"
                else None
            ),
            "long_attention_scaling": (
                float(args.long_attention_scaling)
                if args.frequency == "external_session"
                else None
            ),
        }
    elif args.frequency == "external_static":
        assert external_values is not None
        if not np.array_equal(
            external_values[[0, -1]], native_inv.numpy()[[0, -1]]
        ):
            raise RuntimeError("runtime Native support differs from external receipt")
        rotary = model.model.rotary_emb
        with torch.no_grad():
            rotary.inv_freq.copy_(torch.from_numpy(external_values).to(rotary.inv_freq))
        if hasattr(rotary, "original_inv_freq"):
            rotary.original_inv_freq = rotary.inv_freq.detach().clone()
        rotary.attention_scaling = float(args.long_attention_scaling)
        method = {
            "method": "external_static_fixed_support_z",
            "selection": "one frozen external table at every evaluated length",
            "requires_L_target": False,
            "runtime_requires_request_target": False,
            "construction_horizon_bound_by_external_receipt": True,
            "construction_table_factor": float(args.factor),
            "native_context_length": 4096,
            "native_inv_freq_sha256_float32": float32_sha256(native_inv.numpy()),
            "active_sha256_float32": float32_sha256(external_values),
            "expected_table_sha256_float32": str(args.expected_table_sha256),
            "table_name": str(args.table_name),
            "table_file_sha256": sha256_file(args.table.resolve()),
            "fixed_native_sampled_support": True,
            "attention_scaling": float(args.long_attention_scaling),
            "model_weight_updates": 0,
            "evaluation_parameter_updates": 0,
        }
    else:
        values = np.load(args.table.resolve(), allow_pickle=False)
        if values.dtype != np.float32 or values.shape != (64,):
            raise RuntimeError("budgeted table must be float32 [64]")
        state, method = install_length_conditioned_rope(
            model,
            long_inv_freq=torch.from_numpy(np.ascontiguousarray(values)),
            long_attention_scaling=matched_attention_scaling(float(args.factor)),
            long_name=str(args.table_name),
            reference_length=4096,
            long_context_budget=int(args.length),
        )
        state.force_for_budget(int(args.length))
        method["table_path"] = str(args.table.resolve())
        method["table_file_sha256"] = sha256_file(args.table.resolve())
    method_table_sha256 = None
    if args.frequency == "budgeted":
        method_table_sha256 = str(method["long_branch"]["inv_freq_sha256_float32"])
    elif args.frequency == "external_static":
        method_table_sha256 = str(method["active_sha256_float32"])
    elif args.frequency == "external_session":
        method_table_sha256 = str(method["long_table_sha256_float32"]["4"])
    model.config.max_position_embeddings = int(args.length)
    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    model.eval().to("cuda")
    torch.cuda.reset_peak_memory_stats()

    rows_path = output / "examples.jsonl"
    completed = _load_completed(rows_path)
    expected = {str(job["row_id"]): job for job in jobs}
    for row_id, row in completed.items():
        if (
            row_id not in expected
            or row.get("row_sha256") != expected[row_id]["row_sha256"]
            or (
                args.frequency in {"budgeted", "external_static", "external_session"}
                and row.get("method_table_sha256_float32") != method_table_sha256
            )
        ):
            raise RuntimeError(f"completed 2Wiki row identity drift: {row_id}")
    eos_token_id = tokenizer.eos_token_id
    for ordinal, job in enumerate(jobs, start=1):
        row_id = str(job["row_id"])
        if row_id in completed:
            continue
        if state is not None:
            state.force_for_budget(int(args.length))
        active_profile = None
        if args.frequency in {"session_adaptive", "session_binary_s4", "external_session"}:
            selected_factor = select_observed_session_factor(
                prefill_tokens=len(job["input_ids"]),
                max_new_tokens=int(job["max_new_tokens"]),
                native_context_length=4096,
                supported_factors=(
                    (1, 4)
                    if args.frequency in {"session_binary_s4", "external_session"}
                    else (1, 2, 4)
                ),
            )
            active_profile = set_target_aware_factor(
                model,
                selected_factor,
                session_tables,
                native_inv,
            )
        input_ids = torch.tensor(
            [list(job["input_ids"])], dtype=torch.long, device="cuda"
        )
        started = time.perf_counter()
        generated = greedy_generate(
            model,
            input_ids,
            max_new_tokens=int(job["max_new_tokens"]),
            eos_token_id=eos_token_id,
        )[0].detach().cpu().tolist()
        prediction = tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        row = {
            "ordinal": ordinal,
            "task": str(args.task),
            "row_id": row_id,
            "row_sha256": str(job["row_sha256"]),
            "source_index": int(job["source_index"]),
            "source_row_sha256": str(job["source_row_sha256"]),
            "nominal_length": int(args.length),
            "input_tokens": len(job["input_ids"]),
            "input_sha256": canonical_sha256(job["input_ids"]),
            "truncated": bool(job["truncated"]),
            "references": list(job["references"]),
            "prediction": prediction,
            "generated_token_ids": [int(value) for value in generated],
            "generated_tokens": len(generated),
            "token_f1": token_f1(prediction, job["references"]),
            "normalized_exact": normalized_exact(prediction, job["references"]),
            "elapsed_seconds": time.perf_counter() - started,
            "active_profile": active_profile,
            "method_table_sha256_float32": method_table_sha256,
        }
        atomic_append_jsonl(rows_path, row)
        completed[row_id] = row
        if ordinal == 1 or ordinal % 20 == 0 or ordinal == len(jobs):
            print(
                f"{ordinal}/{len(jobs)} F1={row['token_f1']:.3f} "
                f"exact={row['normalized_exact']:.0f} input={row['input_tokens']}",
                flush=True,
            )

    rows = [completed[str(job["row_id"])] for job in jobs]
    token_f1_macro = float(np.mean([row["token_f1"] for row in rows]))
    exact_macro = float(np.mean([row["normalized_exact"] for row in rows]))
    receipt = {
        "status": STATUS if args.task == "2wikimqa" else GENERIC_STATUS,
        "metric_boundary": "Official LongBench-style normalized token F1; normalized exact is auxiliary.",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "ready_receipt_sha256": sha256_file(args.ready_receipt.resolve()),
        "longbench": {
            key: value for key, value in package.items() if key != "rows"
        },
        "jobs_sha256": jobs_sha,
        "method": method,
        "protocol": {
            "task": str(args.task),
            "frequency": str(args.frequency),
            "factor": float(args.factor),
            "nominal_length": int(args.length),
            "rows": len(rows),
            "max_new_tokens": TASK_GENERATION_TOKENS[str(args.task)],
            "method_selection": False,
            "greedy": True,
        },
        "results": {
            "token_f1_macro": token_f1_macro,
            "normalized_exact_macro": exact_macro,
            "input_tokens_min": min(int(row["input_tokens"]) for row in rows),
            "input_tokens_max": max(int(row["input_tokens"]) for row in rows),
            "truncated_rows": sum(bool(row["truncated"]) for row in rows),
            "examples_sha256": sha256_file(rows_path),
            "selected_factor_counts": {
                str(factor): sum(
                    row.get("active_profile", {}).get("multiplier") == factor
                    for row in rows
                )
                for factor in (1, 2, 4)
            } if args.frequency in {"session_adaptive", "session_binary_s4", "external_session"} else None,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "code_sha256": {
            "evaluator": sha256_file(Path(__file__).resolve()),
            "length_conditioned_runtime": sha256_file(
                Path(__file__).resolve().parents[4]
                / "scripts/lib/rope/length_conditioned_budgeted.py"
            ),
        },
    }
    atomic_json(output / "results.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
