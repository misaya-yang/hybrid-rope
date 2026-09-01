#!/usr/bin/env python3
"""Native-only paired reference-length calibration; no candidate tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expected-data-manifest-sha256", required=True)
    parser.add_argument("--split", choices=("calibration", "confirmation"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--batch-tokens", type=int, default=16384)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
        configure_cuda, configure_ruler_flash_attention,
    )
    from scripts.lib.checkpoint_identity import safetensors_weight_set_sha256

    data_path = args.data_root / "manifest.json"
    manifest_hash = file_hash(data_path)
    if manifest_hash != args.expected_data_manifest_sha256:
        raise RuntimeError("data manifest changed")
    data = json.loads(data_path.read_text())
    if data.get("status") != "NATIVE_REFERENCE_CALIBRATION_DATA_READY_V1":
        raise RuntimeError("data preparation is incomplete")
    if file_hash(args.checkpoint / "config.json") != data["config_sha256"]:
        raise RuntimeError("data belongs to a different checkpoint config")
    for token_file in data["tokenizer_files"]:
        if file_hash(args.checkpoint / token_file["name"]) != token_file["sha256"]:
            raise RuntimeError("tokenizer or chat-template identity changed")
    grid = [1024, 2048, 4096, 8192]
    if data.get("grid") != grid:
        raise RuntimeError("P0 grid drift")
    selection = None
    if args.split == "confirmation":
        if args.selection is None:
            raise RuntimeError("confirmation requires a frozen calibration decision")
        selection = json.loads(args.selection.read_text())
        if selection.get("selected_length") not in grid or not selection.get("confirmation_lengths"):
            raise RuntimeError("calibration abstained; confirmation must not run")
        if selection.get("data_manifest_sha256") != manifest_hash:
            raise RuntimeError("selection belongs to different calibration data")
        lengths = set(selection["confirmation_lengths"])
    else:
        if args.selection is not None:
            raise RuntimeError("calibration cannot read a selection result")
        lengths = set(grid)
    entry = data["files"][args.split]
    rows_path = args.data_root / entry["path"]
    if file_hash(rows_path) != entry["sha256"]:
        raise RuntimeError("data rows changed")
    all_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if len(all_rows) != entry["rows"]:
        raise RuntimeError("data row count changed")
    rows = [row for row in all_rows if row["length"] in lengths or row["variant"] == "compact"]
    if any(row["split"] != args.split for row in rows):
        raise RuntimeError("cross-split row detected")
    weight_hash = safetensors_weight_set_sha256(args.checkpoint)
    if weight_hash != args.expected_weight_sha256:
        raise RuntimeError("checkpoint weights changed")
    args.output.mkdir(parents=True, exist_ok=True)
    examples_path = args.output / "examples.jsonl"
    if examples_path.exists():
        raise RuntimeError("fresh output required; never append a different P0 run")
    configure_cuda()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval().to("cuda")
    if int(model.config.max_position_embeddings) != 8192:
        raise RuntimeError("this P0 protocol requires original config length 8192")
    configure_ruler_flash_attention(model)
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    fixtures = data["full_output_scorer_fixtures"]
    for fixture in fixtures["cases"]:
        tokens = fixture["generated_ids"]
        terminal = fixtures["terminal_ids"]
        exact = bool(tokens) and tokens[-1] in terminal and not any(t in terminal for t in tokens[:-1])
        if exact:
            text = tokenizer.decode(tokens[:-1], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            exact = text.strip() == fixtures["expected_text"]
        if exact != fixture["expected_pass"]:
            raise RuntimeError("complete-output scoring contract failed")
    rotary = model.model.rotary_emb
    initial_inv = rotary.inv_freq.detach().clone()
    native_hash = hashlib.sha256(initial_inv.cpu().float().numpy().tobytes()).hexdigest()
    if native_hash != args.expected_native_sha256 or float(rotary.attention_scaling) != 1.0:
        raise RuntimeError("P0 is Native only; no table or gain intervention is allowed")
    run_manifest = {
        "status": "P0_NATIVE_REFERENCE_FROZEN", "split": args.split,
        "data_manifest_sha256": manifest_hash, "rows_sha256": entry["sha256"],
        "checkpoint_sha256": weight_hash, "native_sha256_float32": native_hash,
        "config_sha256": file_hash(args.checkpoint / "config.json"),
        "tokenizer_files": {p.name: file_hash(p) for p in args.checkpoint.iterdir()
                            if p.is_file() and (p.name.startswith("tokenizer") or p.name in
                                                {"special_tokens_map.json", "generation_config.json"})},
        "model_source_sha256": file_hash(Path(inspect.getsourcefile(type(model)))),
        "script_sha256": file_hash(Path(__file__)),
        "attention_source_sha256": file_hash(Path(inspect.getsourcefile(configure_ruler_flash_attention))),
        "selection_sha256": None if args.selection is None else file_hash(args.selection),
        "selection": selection, "lengths": sorted(lengths), "rows": len(rows),
        "transformers": transformers.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "batch_tokens": args.batch_tokens,
        "max_batch_size": args.max_batch_size, "attention_scaling": 1.0,
        "compile": False, "model_updates": 0, "candidate_tables": 0,
        "flash_only": True, "shutdown": False,
        "measurement_protocol": data.get("measurement_protocol", "two_code_exact_v1"),
        "scoring": data["full_output_scorer_fixtures"]["contract"],
    }
    atomic_json(args.output / "run_manifest.json", run_manifest)
    torch.cuda.reset_peak_memory_stats()
    groups = defaultdict(list)
    for row in rows:
        count = len(row["input_ids"])
        if row["family"] == "natural":
            if count != row["length"] or row["target_start"] != count - 256 or row["target_tokens"] != 256:
                raise RuntimeError("natural suffix identity is invalid")
        else:
            if row["variant"] != "compact" and count + row["generation_budget"] > row["length"]:
                raise RuntimeError("capability request exceeds its length budget")
            if row["terminal_ids"] != [tokenizer.eos_token_id]:
                raise RuntimeError("terminal contract drift")
        groups[(row["family"], row["length"], row["variant"], count)].append(row)

    @torch.inference_mode()
    def generate(ids, budget, terminal_ids):
        chunks = []
        finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            for step in range(budget):
                token = out.logits[:, -1, :].argmax(-1)
                chunks.append(token)
                finished |= torch.isin(token, torch.tensor(terminal_ids, device=ids.device))
                if bool(finished.all()) or step == budget - 1:
                    break
                past = out.past_key_values
                out = model(input_ids=token[:, None], past_key_values=past,
                            use_cache=True, logits_to_keep=1)
        return torch.stack(chunks, dim=1).cpu().tolist()

    total_started = time.perf_counter()
    written = 0
    generated_tokens = 0
    input_tokens = 0
    first_nll = None
    instrument_counts = {"compact": [0, 0], "baseline": [0, 0]}
    with examples_path.open("w") as handle:
        for key in sorted(groups):
            family, length, variant, count = key
            batch_size = max(1, min(args.max_batch_size, args.batch_tokens // count))
            selected = groups[key]
            for offset in range(0, len(selected), batch_size):
                batch = selected[offset:offset + batch_size]
                ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device="cuda")
                started = time.perf_counter()
                if family == "natural":
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(input_ids=ids, use_cache=False, logits_to_keep=257).logits[:, :-1]
                    target = ids[:, -256:]
                    losses = []
                    for left in range(0, 256, 64):
                        losses.append(F.cross_entropy(
                            logits[:, left:left+64].float().transpose(1, 2),
                            target[:, left:left+64], reduction="none",
                        ).detach())
                    per_row_nll = torch.cat(losses, dim=1).mean(1)
                    if not bool(torch.isfinite(per_row_nll).all()):
                        raise RuntimeError("nonfinite Native loss")
                    metrics = [{"nll": float(value), "target_tokens": 256} for value in per_row_nll.cpu()]
                    if first_nll is None:
                        first_nll = metrics[0]["nll"]
                    del logits, losses, per_row_nll
                else:
                    sequences = generate(ids, batch[0]["generation_budget"], batch[0]["terminal_ids"])
                    metrics = []
                    for row, tokens in zip(batch, sequences):
                        stops = [j for j, token in enumerate(tokens) if token in row["terminal_ids"]]
                        terminated = bool(stops)
                        if terminated:
                            tokens = tokens[:stops[0] + 1]
                        content = tokens[:-1] if terminated else tokens
                        prediction = tokenizer.decode(content, skip_special_tokens=False,
                                                      clean_up_tokenization_spaces=False)
                        exact_text = prediction.strip() == row["expected_text"]
                        metrics.append({
                            "generated_ids": tokens, "prediction": prediction,
                            "expected_text": row["expected_text"], "terminated": terminated,
                            "stop_reason": "eos" if terminated else "token_budget",
                            "exact_text": exact_text, "exact_match": exact_text and terminated,
                        })
                        generated_tokens += len(tokens)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if not torch.equal(initial_inv, rotary.inv_freq):
                    raise RuntimeError("Native table mutated inside calibration")
                if torch.cuda.max_memory_reserved() > 30 * 1024**3:
                    raise RuntimeError("P0 exceeded the 30 GiB memory budget")
                for row, result in zip(batch, metrics):
                    record = {k: value for k, value in row.items() if k != "input_ids"}
                    record.update(result)
                    record.update({"batch_size": len(batch), "batch_elapsed_seconds": elapsed,
                                   "prompt_tokens": count, "data_manifest_sha256": manifest_hash})
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    written += 1
                    if family == "capability" and (variant == "compact" or length == 1024):
                        control = "compact" if variant == "compact" else "baseline"
                        instrument_counts[control][0] += int(result["exact_match"])
                        instrument_counts[control][1] += 1
                handle.flush()
                input_tokens += count * len(batch)
                if (args.split == "calibration"
                        and data.get("measurement_protocol") == "single_code_format_repair_v1"):
                    expected_controls = data["files"]["calibration"]["capability_blueprints"]
                    failed = [name for name, (success, total) in instrument_counts.items()
                              if total == expected_controls and success / total < 0.75]
                    if failed:
                        atomic_json(args.output / "results.json", {
                            "status": "P0_INSTRUMENT_FAILED_ABSTAIN", "split": args.split,
                            "data_manifest_sha256": manifest_hash, "rows": written,
                            "failed_controls": failed, "instrument_counts": instrument_counts,
                            "selected_length": None, "further_prompt_repair_allowed": False,
                        })
                        print(f"P0 instrument gate failed: {failed}; no longer-length rows opened", flush=True)
                        return 0
                if written == len(batch) or offset + batch_size >= len(selected):
                    print(json.dumps({"completed": written, "total": len(rows), "family": family,
                          "length": length, "variant": variant, "batch_size": len(batch),
                          "batch_seconds": elapsed, "example_metric": metrics[0].get("nll", metrics[0].get("exact_match")),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved()}), flush=True)
        os.fsync(handle.fileno())
    result = {
        "status": "P0_NATIVE_REFERENCE_EVAL_COMPLETE", "split": args.split,
        "data_manifest_sha256": manifest_hash, "rows": written,
        "examples_sha256": file_hash(examples_path), "first_real_nll": first_nll,
        "input_tokens": input_tokens, "generated_tokens": generated_tokens,
        "elapsed_seconds": time.perf_counter() - total_started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "selected_length": None, "selection_performed": False,
    }
    atomic_json(args.output / "results.json", result)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
