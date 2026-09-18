#!/usr/bin/env python3
"""Run the frozen T/C position-gap intervention or its offset-parity check."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PARITY_LOGIT_MAX_ABS_TOLERANCE = 0.02


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value) -> str:
    return sha256_bytes(json.dumps(value, separators=(",", ":")).encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage1(path: Path) -> dict:
    report = json.loads(path.read_text())
    if (
        report.get("status") != "TC_EXISTING_MULTIKEY_BEHAVIOR_AUDIT_COMPLETE_V2"
        or report.get("rows_expected") != 750
        or report.get("rows_qualified") != 750
        or report.get("mapping_coverage") != 1.0
        or not report.get("gpu_stage_qualified")
        or not all(report.get("gpu_distance_competition_gate", {}).values())
    ):
        raise ValueError("complete qualifying Stage 1 V2 report is required")
    return report


def validate_reconciliation(path: Path) -> dict:
    report = json.loads(path.read_text())
    if (
        report.get("status") != "TC_STAGE1_OFFICIAL_RECONCILIATION_COMPLETE_V1"
        or report.get("source_rows") != 750
        or report.get("arm_rows") != {"dose_control_c": 750, "tailspline": 750}
    ):
        raise ValueError("complete Stage 1 official-score reconciliation is required")
    return report


def load_table(path: Path) -> dict:
    import numpy as np

    payload = json.loads(path.read_text())
    table = payload.get("table", payload)
    values = np.asarray(table.get("values_float32"), dtype=np.float32)
    gain = float(table.get("gain"))
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or not np.all(values[:-1] > values[1:])
        or not np.isfinite(gain)
        or gain <= 0
    ):
        raise ValueError("invalid static table")
    return {
        "values_float32": values.tolist(),
        "gain": gain,
        "construction": table.get("construction", {}),
    }


def validate_panel(panel: Path, manifest_path: Path, panel_kind: str) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "TC_DISTANCE_COMPETITION_CONFIRM_PANEL_FROZEN_V2":
        raise ValueError("frozen V2 panel manifest is required")
    expected = manifest[panel_kind]
    if sha256_file(panel) != expected["input_file_sha256"]:
        raise ValueError("panel file differs from frozen manifest")
    rows = read_jsonl(panel)
    if len(rows) != expected["rows"] or len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("panel row count or IDs differ")
    identity = sorted(
        (row["row_id"], row["prompt_sha256"], row["position_ids_sha256"])
        for row in rows
    )
    if sha256_json(identity) != expected["row_identity_sha256"]:
        raise ValueError("row identity receipt differs")
    for row in rows:
        prompt = row["prompt_ids"]
        positions = row["position_ids"]
        if len(prompt) != len(positions) or len(prompt) != row["input_tokens"]:
            raise ValueError(f"prompt/position length mismatch: {row['row_id']}")
        if sha256_json(prompt) != row["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch: {row['row_id']}")
        if sha256_json(positions) != row["position_ids_sha256"]:
            raise ValueError(f"position hash mismatch: {row['row_id']}")
        if any(right <= left for left, right in zip(positions, positions[1:])):
            raise ValueError(f"position IDs are not strictly increasing: {row['row_id']}")
        actual_distance = (
            positions[row["query_token_index"]]
            - positions[row["target_value_token_index"]]
        )
        if actual_distance != row["dependency_distance"]:
            raise ValueError(f"dependency-distance mismatch: {row['row_id']}")
    return rows, manifest


def load_static_model(model_path: Path, table: dict):
    import numpy as np
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model
    from scripts.experiments.cross_audit.tables import install_static

    model, _, _ = load_model(model_path, "Native", checkpoint=None, training=False)
    install_static(
        model,
        np.asarray(table["values_float32"], dtype=np.float32),
        table["gain"],
    )
    actual = model.model.rotary_emb.inv_freq.detach().cpu().float().numpy()
    if not np.array_equal(actual, np.asarray(table["values_float32"], dtype=np.float32)):
        raise RuntimeError("installed table differs from the frozen receipt")
    model.eval()
    return model


def greedy_generate(model, row: dict, eos_ids: set[int], *, common_offset: int = 0):
    import torch

    prompt_ids = torch.tensor([row["prompt_ids"]], device="cuda", dtype=torch.long)
    positions = torch.tensor(
        [[value + common_offset for value in row["position_ids"]]],
        device="cuda",
        dtype=torch.long,
    )
    physical_length = prompt_ids.shape[1]
    attention_mask = torch.ones_like(prompt_ids)
    output = model(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        position_ids=positions,
        cache_position=torch.arange(physical_length, device="cuda"),
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    cache = output.past_key_values
    logits = output.logits[:, -1, :].float()
    first_logits = logits[0].detach().cpu()
    generated = []
    last_position = int(positions[0, -1])
    for step in range(int(row["max_new_tokens"])):
        token = logits.argmax(dim=-1)
        value = int(token.item())
        generated.append(value)
        if value in eos_ids or step + 1 == int(row["max_new_tokens"]):
            break
        cache_position = physical_length + step
        next_position = last_position + step + 1
        output = model(
            input_ids=token[:, None],
            attention_mask=torch.ones(
                (1, cache_position + 1), device="cuda", dtype=torch.long,
            ),
            position_ids=torch.tensor([[next_position]], device="cuda", dtype=torch.long),
            cache_position=torch.tensor([cache_position], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        cache = output.past_key_values
        logits = output.logits[:, -1, :].float()
    del output, cache, logits, prompt_ids, positions, attention_mask
    return generated, first_logits


def select_parity_rows(rows: list[dict]) -> list[dict]:
    base_ids = sorted({row["base_sample_id"] for row in rows})[:2]
    selected = [row for row in rows if row["base_sample_id"] in base_ids]
    expected_cells = {
        (distance, competition)
        for distance in ("near", "far")
        for competition in ("neutral", "structured_kv")
    }
    for base_id in base_ids:
        cells = {
            (row["distance_condition"], row["competition_condition"])
            for row in selected
            if row["base_sample_id"] == base_id
        }
        if cells != expected_cells:
            raise ValueError("parity selection lacks a complete four-cell base sample")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("parity", "execute"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-kind", choices=("confirmation", "parity"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage1-report", type=Path, required=True)
    parser.add_argument("--reconciliation-report", type=Path, required=True)
    parser.add_argument("--static-table-json", type=Path, required=True)
    parser.add_argument("--table-label", required=True)
    parser.add_argument("--parity-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "parity" and args.panel_kind != "parity":
        raise ValueError("parity mode requires the separate parity panel")
    if args.mode == "execute" and args.panel_kind != "confirmation":
        raise ValueError("execute mode requires the frozen confirmation panel")
    if args.mode == "execute" and not args.parity_receipt:
        raise ValueError("execute mode requires a passing parity receipt")

    stage1 = validate_stage1(args.stage1_report)
    reconciliation = validate_reconciliation(args.reconciliation_report)
    rows, manifest = validate_panel(args.panel, args.manifest, args.panel_kind)
    table = load_table(args.static_table_json)
    table_sha256 = sha256_file(args.static_table_json)
    manifest_sha256 = sha256_file(args.manifest)
    stage1_sha256 = sha256_file(args.stage1_report)
    reconciliation_sha256 = sha256_file(args.reconciliation_report)
    common_offset = int(manifest["parity_common_offset"])
    if args.mode == "execute":
        parity = json.loads(args.parity_receipt.read_text())
        if (
            parity.get("status") != "TC_POSITION_OFFSET_PARITY_PASS_V2"
            or parity.get("table_label") != args.table_label
            or parity.get("table_file_sha256") != table_sha256
            or parity.get("manifest_file_sha256") != manifest_sha256
        ):
            raise ValueError("passing parity receipt does not match this table/manifest")

    import torch
    import transformers
    from transformers import AutoTokenizer
    from experiments.olmo_recovery_20260912.runtime import validate_cuda
    from scripts.eval.longbench_metrics import qa_f1_score
    from scripts.experiments.olmo_fast_screen.ruler_bench import score as ruler_score

    validate_cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = load_static_model(args.model, table)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    contract = {
        "status": "TC_POSITION_GAP_RUNTIME_CONTRACT_V2",
        "mode": args.mode,
        "table_label": args.table_label,
        "table_path": str(args.static_table_json.resolve()),
        "table_file_sha256": table_sha256,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_file_sha256": manifest_sha256,
        "panel_path": str(args.panel.resolve()),
        "panel_file_sha256": sha256_file(args.panel),
        "stage1_status": stage1["status"],
        "stage1_file_sha256": stage1_sha256,
        "reconciliation_status": reconciliation["status"],
        "reconciliation_file_sha256": reconciliation_sha256,
        "checkpoint": str(args.model.resolve()),
        "model_weight_updates": 0,
        "runtime_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "model_dtype": "bfloat16",
        },
        "row_ids": [row["row_id"] for row in rows],
        "primary_inference_unit": "base_sample",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("output directory contains a different runtime contract")
    write_json(contract_path, contract)

    if args.mode == "parity":
        checks = []
        passed = True
        with torch.inference_mode():
            for row in select_parity_rows(rows):
                tokens, logits = greedy_generate(model, row, eos_ids, common_offset=0)
                shifted_tokens, shifted_logits = greedy_generate(
                    model, row, eos_ids, common_offset=common_offset,
                )
                max_abs = float((logits - shifted_logits).abs().max())
                token_exact = tokens == shifted_tokens
                row_pass = token_exact and max_abs <= PARITY_LOGIT_MAX_ABS_TOLERANCE
                passed = passed and row_pass
                checks.append({
                    "row_id": row["row_id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "position_ids_sha256": row["position_ids_sha256"],
                    "common_offset": common_offset,
                    "first_step_logit_max_abs": max_abs,
                    "generated_tokens_exact": token_exact,
                    "generated_token_count": len(tokens),
                    "pass": row_pass,
                })
        receipt = {
            "status": (
                "TC_POSITION_OFFSET_PARITY_PASS_V2"
                if passed
                else "TC_POSITION_OFFSET_PARITY_FAIL_V2"
            ),
            "table_label": args.table_label,
            "table_file_sha256": table_sha256,
            "manifest_file_sha256": manifest_sha256,
            "common_offset": common_offset,
            "first_step_logit_max_abs_tolerance": PARITY_LOGIT_MAX_ABS_TOLERANCE,
            "rows_checked": len(checks),
            "checks": checks,
        }
        write_json(args.out / "parity_receipt.json", receipt)
        print(json.dumps({
            "status": receipt["status"],
            "rows_checked": len(checks),
            "maximum_first_step_logit_abs": max(
                check["first_step_logit_max_abs"] for check in checks
            ),
            "all_generated_tokens_exact": all(
                check["generated_tokens_exact"] for check in checks
            ),
        }, indent=2, sort_keys=True))
        if not passed:
            raise SystemExit(2)
        return

    generations_path = args.out / "generations.jsonl"
    saved = read_jsonl(generations_path) if generations_path.exists() else []
    if len(saved) > len(rows) or any(
        saved[index]["row_id"] != rows[index]["row_id"]
        for index in range(len(saved))
    ):
        raise ValueError("saved generation prefix differs from the frozen panel")
    with generations_path.open("a") as stream, torch.inference_mode():
        for row in rows[len(saved):]:
            tokens, _ = greedy_generate(model, row, eos_ids)
            ended = bool(tokens and tokens[-1] in eos_ids)
            decoded_tokens = tokens[:-1] if ended else tokens
            text = tokenizer.decode(
                decoded_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            record = {
                "row_id": row["row_id"],
                "base_sample_id": row["base_sample_id"],
                "task": row["task"],
                "distance_condition": row["distance_condition"],
                "competition_condition": row["competition_condition"],
                "competition_label": row["competition_label"],
                "dependency_distance": row["dependency_distance"],
                "input_tokens": row["input_tokens"],
                "prompt_sha256": row["prompt_sha256"],
                "position_ids_sha256": row["position_ids_sha256"],
                "table_file_sha256": table_sha256,
                "arm": args.table_label,
                "references": row["references"],
                "generated_ids": tokens,
                "output_text": text,
                "ended_eos": ended,
                "hit_cap": len(tokens) == int(row["max_new_tokens"]) and not ended,
                "whole_response_f1": qa_f1_score(text, row["references"]),
                "ruler_official_score": float(ruler_score(row, text)),
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            saved.append(record)
            write_json(args.out / "live.json", {
                "phase": "generation",
                "completed": len(saved),
                "total": len(rows),
            })
    write_json(args.out / "status.json", {
        "status": "COMPLETE",
        "rows": len(saved),
    })
    print(json.dumps({
        "status": "COMPLETE",
        "arm": args.table_label,
        "rows": len(saved),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
