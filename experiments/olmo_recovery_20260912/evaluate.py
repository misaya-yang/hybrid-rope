#!/usr/bin/env python3
"""Resumable full-generation, Native-retention, and held-out LM evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.evq_recovery.data import JsonlIndex, qa_scores
from .runtime import load_model, validate_cuda
from .train import semantic_contract, verify_bound


LENGTHS = (4_096, 8_192, 16_384, 32_768)


def atomic(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def contract(root: Path, arm: str, checkpoint: Path | None, base_native: bool,
             unadapted: bool = False) -> tuple[dict, dict, dict]:
    plan_path, tables_path = root / "plan.json", root / "tables.json"
    plan, tables = json.loads(plan_path.read_text()), json.loads(tables_path.read_text())
    manifest_path = Path(plan["data_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "CPU_DATA_READY_GPU_NOT_RUN":
        raise ValueError("real data manifest is not ready")
    verify_bound(plan, manifest_path, tables_path)
    for name, receipt in manifest["files"].items():
        path = Path(receipt.get("path", manifest_path.parent / name))
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"data artifact missing or empty: {name}")
    if arm not in tables or arm not in plan["arms"]:
        raise ValueError("unknown table arm")
    table = tables[arm]
    values = np.asarray(table["values_float32"], dtype=np.float32)
    if values.shape != (64,) or not np.isfinite(values).all() or not np.all(values[:-1] > values[1:]):
        raise ValueError("table shape/order is invalid")
    if base_native or unadapted:
        if checkpoint is not None or (base_native and arm != "Native"):
            raise ValueError("original baseline is exactly Native with no adapter checkpoint")
        checkpoint_identity = None
    else:
        if checkpoint is None:
            raise ValueError("adapter evaluation requires --checkpoint")
        checkpoint = checkpoint.resolve()
        state = json.loads((checkpoint / "state.json").read_text())
        expected = semantic_contract(plan, arm, plan["gradient_accumulation"])
        if state.get("semantic_contract") is not None and state["semantic_contract"] != expected:
            raise ValueError("checkpoint semantic training contract mismatch")
        if state.get("arm", arm) != arm:
            raise ValueError("checkpoint arm mismatch")
        adapter_files = sorted(checkpoint.glob("adapter_model*.safetensors"))
        if not adapter_files:
            raise FileNotFoundError("adapter weights missing")
        checkpoint_identity = {"adapter_files": [p.name for p in adapter_files],
                               "arm": state.get("arm", arm), "cpt_tokens": state.get("cpt_tokens"),
                               "semantic_contract": state.get("semantic_contract")}
    identity = {"schema_version": 1, "asset_identity_policy": "user_attested_clone/no_sha_validation",
                "arm": arm, "table": table, "base_native_no_adapter": base_native,
                "unadapted_original_weights": base_native or unadapted,
                "checkpoint": checkpoint_identity, "lengths": list(LENGTHS), "tail_targets": 128,
                "generation": "greedy native chat prompt; raw token ids retained; terminal EOS/EOT removed only from decoded scoring text",
                "metrics": "whole-response F1/exact/literal exact+terminal; LM per-document loss_sum/count for whole and tail128"}
    return identity, plan, manifest


def select_generation_rows(data_root: Path, split: str) -> list[dict]:
    selected = []
    for suite in ("qa", "native"):
        path = data_root / f"{suite}_{split}.jsonl"
        index = JsonlIndex(path)
        for i in range(len(index)):
            row = index[i]
            if suite == "native" and (row.get("task") == "text" or not row.get("generation_budget")):
                continue
            if not row.get("references") or not row.get("prompt_ids"):
                raise ValueError(f"generation row lacks prompt/reference: {suite}/{row.get('id')}")
            selected.append({**row, "eval_id": f"{suite}:{row['id']}", "suite": suite})
    if len({row["eval_id"] for row in selected}) != len(selected):
        raise ValueError("duplicate generation identities")
    return selected


def lm_loss_rows(model, token_ids: torch.Tensor, *, tail: int = 128, chunk_size: int = 128) -> dict:
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] < 2:
        raise ValueError("LM window must have shape [1,L+1]")
    hidden = model.model(input_ids=token_ids[:, :-1], use_cache=False).last_hidden_state[0]
    targets = token_ids[0, 1:]
    count = int(targets.numel())
    tail_count = min(int(tail), count)
    whole_sum = tail_sum = 0.0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        logits = F.linear(hidden[start:stop], model.lm_head.weight).float()
        losses = F.cross_entropy(logits, targets[start:stop], reduction="none")
        whole_sum += float(losses.sum().detach())
        overlap = max(start, count - tail_count)
        if overlap < stop:
            tail_sum += float(losses[overlap - start:].sum().detach())
    return {"whole_loss_sum": whole_sum, "whole_target_count": count,
            "tail128_loss_sum": tail_sum, "tail128_target_count": tail_count}


def validate_saved(saved: list[dict], expected: list[dict], contract_id: str) -> None:
    if len(saved) > len(expected):
        raise ValueError("saved generation count exceeds panel")
    for index, row in enumerate(saved):
        if row.get("evaluation_contract") != contract_id or row.get("eval_id") != expected[index]["eval_id"]:
            raise ValueError(f"saved generation prefix drift at {index}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--arm", choices=("Native", "Cosh", "OfficialYaRN"), required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--base-native", action="store_true", help="evaluate untouched Native with adapter disabled")
    p.add_argument("--unadapted", action="store_true", help="0-step original weights under the specified static table; no adapter")
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--output", type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    identity, plan, manifest = contract(root, args.arm, args.checkpoint, args.base_native, args.unadapted)
    identity["split"] = args.split
    data_root = Path(plan["data_manifest"]).resolve().parent
    generation_rows = select_generation_rows(data_root, args.split)
    lm_path = data_root / f"lm_{'validation' if args.split == 'dev' else 'test'}.npy"
    lm = np.load(lm_path, mmap_mode="r", allow_pickle=False)
    if lm.ndim != 2 or lm.shape[1] != 32_769:
        raise ValueError("held-out LM payload must contain 32769-token document windows")
    identity["generation_rows"] = len(generation_rows)
    identity["lm_documents"] = len(lm)
    contract_id = f"{args.arm}:{args.split}:{'base' if args.base_native else 'unadapted' if args.unadapted else 'adapted'}"
    if args.dry_run:
        model_path = Path(plan["model_path"])
        status = "DRY_RUN_PASS" if model_path.exists() and any(model_path.glob("*.safetensors")) else "MODEL_MISSING_DATA_READY"
        print(json.dumps({"status": status, "generation_rows": len(generation_rows), "lm_documents": len(lm),
                          "evaluation_contract": contract_id, "asset_identity_policy": "user_attested_clone/no_sha_validation"}, sort_keys=True))
        return
    if args.output is None:
        p.error("--output is required with --execute")
    validate_cuda()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != identity:
        raise ValueError("cannot resume under a changed evaluation contract")
    atomic(contract_path, identity)

    without_adapter = args.base_native or args.unadapted
    checkpoint = None if without_adapter else args.checkpoint.resolve()
    table = json.loads((root / "tables.json").read_text())[args.arm]
    model, wrapper = load_model(plan, table, training=False, checkpoint=checkpoint)
    model.eval()
    tokenizer_path = Path(plan["model_path"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    eos_ids = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
                  else [model.generation_config.eos_token_id])
    eos_ids.add(int(manifest["assistant_terminal_id"]))
    def adapter_context():
        return wrapper.disable_adapter() if without_adapter else contextlib.nullcontext()
    raw_path = output / "generations.jsonl"
    saved = jsonl(raw_path)
    validate_saved(saved, generation_rows, contract_id)
    with raw_path.open("a") as stream, torch.inference_mode(), adapter_context():
        for index, row in enumerate(generation_rows[len(saved):], start=len(saved)):
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device="cuda")
            started = time.monotonic()
            generated = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, num_beams=1,
                                       max_new_tokens=row["generation_budget"], eos_token_id=list(eos_ids),
                                       pad_token_id=tokenizer.pad_token_id, use_cache=True)[0, ids.shape[1]:].tolist()
            ended = bool(generated and generated[-1] in eos_ids)
            score_ids = generated[:-1] if ended else generated
            text = tokenizer.decode(score_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            scores = qa_scores(text, row["references"])
            record = {"evaluation_contract": contract_id, "eval_id": row["eval_id"], "source_id": row["source_id"],
                      "suite": row["suite"], "task": row["task"], "length_bucket": row.get("length_bucket", 4096),
                      "prompt_sha256": row.get("prompt_sha256"), "generated_ids": generated, "prediction": text,
                      "eos_or_eot_terminated": ended, "hit_cap": len(generated) == row["generation_budget"] and not ended,
                      "empty": not text.strip(), "whole_response_f1": scores["f1"], "normalized_exact": scores["exact"],
                      "literal_complete_exact_plus_terminal": float(ended and any(text == ref for ref in row["references"])),
                      "elapsed_seconds": time.monotonic() - started}
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            atomic(output / "live.json", {"stage": "generation", "completed": index + 1, "total": len(generation_rows)})

    lm_path_out = output / "lm_rows.jsonl"
    completed = jsonl(lm_path_out)
    expected_cells = [(doc, length) for doc in range(len(lm)) for length in LENGTHS]
    if len(completed) > len(expected_cells):
        raise ValueError("saved LM stream exceeds the complete panel")
    for index, row in enumerate(completed):
        if row.get("evaluation_contract") != contract_id or (row["document_index"], row["context_length"]) != expected_cells[index]:
            raise ValueError(f"saved LM prefix drift at {index}")
    with lm_path_out.open("a") as stream, torch.inference_mode(), adapter_context():
        for index, (document, length) in enumerate(expected_cells[len(completed):], start=len(completed)):
            window = np.asarray(lm[document, -(length + 1):], dtype=np.int64)
            row = lm_loss_rows(model, torch.from_numpy(window).unsqueeze(0).cuda(), tail=128)
            row.update(evaluation_contract=contract_id, document_index=document, context_length=length)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            atomic(output / "live.json", {"stage": "lm", "completed": index + 1, "total": len(expected_cells)})
    generation_complete = jsonl(raw_path)
    grouped = defaultdict(list)
    for row in generation_complete:
        grouped[row["suite"], row["task"], row["length_bucket"]].append(row)
    generation_summary = {}
    for key, values in grouped.items():
        generation_summary["/".join(map(str, key))] = {
            "rows": len(values),
            "whole_response_f1": float(np.mean([row["whole_response_f1"] for row in values])),
            "normalized_exact": float(np.mean([row["normalized_exact"] for row in values])),
            "literal_complete_exact_plus_terminal": float(np.mean([row["literal_complete_exact_plus_terminal"] for row in values])),
            "terminal_rate": float(np.mean([row["eos_or_eot_terminated"] for row in values])),
            "cap_hits": sum(row["hit_cap"] for row in values),
            "empty_outputs": sum(row["empty"] for row in values),
            "mean_generated_tokens_including_terminal": float(np.mean([len(row["generated_ids"]) for row in values])),
        }
    lm_complete = jsonl(lm_path_out)
    lm_summary = {}
    for length in LENGTHS:
        values = [row for row in lm_complete if row["context_length"] == length]
        whole_sum = sum(row["whole_loss_sum"] for row in values)
        whole_count = sum(row["whole_target_count"] for row in values)
        tail_sum = sum(row["tail128_loss_sum"] for row in values)
        tail_count = sum(row["tail128_target_count"] for row in values)
        lm_summary[str(length)] = {"documents": len(values), "whole_loss_sum": whole_sum,
                                   "whole_target_count": whole_count, "whole_nll": whole_sum / whole_count,
                                   "tail128_loss_sum": tail_sum, "tail128_target_count": tail_count,
                                   "tail128_nll": tail_sum / tail_count}
    atomic(output / "summary.json", {"status": "COMPLETE", "evaluation_contract": contract_id,
           "asset_identity_policy": "user_attested_clone/no_sha_validation",
           "generation": generation_summary, "lm": lm_summary,
           "boundary": "Native retention, full generated responses, and teacher-forced LM NLL are separate outcomes."})
    atomic(output / "status.json", {"status": "COMPLETE", "evaluation_contract": contract_id,
           "asset_identity_policy": "user_attested_clone/no_sha_validation",
           "generation_rows": len(generation_rows), "lm_rows": len(expected_cells),
           "peak_cuda_bytes": int(torch.cuda.max_memory_allocated())})


if __name__ == "__main__":
    main()
