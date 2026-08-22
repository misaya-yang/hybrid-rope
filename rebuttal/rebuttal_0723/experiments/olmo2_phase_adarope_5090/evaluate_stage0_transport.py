"""Evaluate a fixed Stage-0 adapter under Native or official YaRN transport."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .receipts import atomic_json, sha256_file
from .train_phase_adarope import (
    EXPECTED_DATA_ROOT_SHA256,
    PairView,
    _base_model,
    _checkpoint_identity,
    _configure_cuda,
    _dependency_versions,
    _greedy_generate,
    _install_lora,
    _load_base,
    _manifest_hash,
    _selected_logits,
    _validate_model_shape,
)


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().float().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _load_candidate(checkpoint: Path, parent: Path, *, factor: float, torch: Any) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoModelForCausalLM
    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
        configure_ruler_flash_attention,
    )
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer import (
        official_yarn_config,
        verify_official_yarn,
    )

    if float(factor) == 1.0:
        base = _load_base(checkpoint, torch)
        frequency = {
            "active_frequency": "native",
            "factor": 1.0,
            "active_sha256_float32": _tensor_sha256(base.model.rotary_emb.inv_freq),
            "attention_scaling": float(base.model.rotary_emb.attention_scaling),
        }
    else:
        config = official_yarn_config(
            checkpoint,
            factor=float(factor),
            original_max_position_embeddings=4096,
        )
        base = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        frequency = verify_official_yarn(base, config)
        frequency["factor"] = float(factor)
    model = _install_lora(base, parent=parent, trainable=False)
    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    return model, frequency


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--factor", type=float, choices=(1.0, 2.0, 4.0), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.authorize:
        raise PermissionError("explicit --authorize is required")
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from transformers import AutoTokenizer

    runtime = _configure_cuda(torch)
    dependencies = _dependency_versions(torch)
    checkpoint = _checkpoint_identity(args.checkpoint, args.checkpoint_ready)
    parent_receipt = json.loads((args.parent / "receipt.json").read_text(encoding="utf-8"))
    if parent_receipt.get("status") != "COMPLETE" or parent_receipt.get("stage") != "stage0":
        raise ValueError("transport evaluation requires the original completed Stage-0 parent")
    data_root_sha = sha256_file(args.data.parent / "manifest.json")
    if data_root_sha != EXPECTED_DATA_ROOT_SHA256:
        raise ValueError("V3 data root identity drift")
    view = PairView.load(args.data)
    length = int(view.correct.shape[1])
    if (length, float(args.factor)) not in {(8192, 1.0), (8192, 2.0), (16384, 1.0), (16384, 4.0)}:
        raise ValueError("registered transport cells are Native/YaRN2 at 8K and Native/YaRN4 at 16K")

    model, frequency = _load_candidate(args.checkpoint, args.parent, factor=float(args.factor), torch=torch)
    _validate_model_shape(model, length)
    model.eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    eos_token_id = int(tokenizer.eos_token_id)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(view.correct)):
            prompt_stop = int(view.generation_prompt_stops[index])
            expected = np.asarray(view.target_tokens[index, 0], dtype=np.int64)
            prompt = torch.as_tensor(
                np.array(view.correct[index : index + 1, :prompt_stop], copy=True),
                device="cuda",
                dtype=torch.long,
            )
            generated = _greedy_generate(
                model,
                prompt,
                max_new_tokens=len(expected),
                eos_token_id=eos_token_id,
                phase_context_budget=None,
                torch=torch,
            )[0].detach().cpu().numpy().astype(np.int64)

            positions = torch.as_tensor(
                np.asarray(view.positions_by_row[index : index + 1]),
                device="cuda",
                dtype=torch.long,
            )
            correct = torch.as_tensor(
                np.array(view.correct[index : index + 1], copy=True),
                device="cuda",
                dtype=torch.long,
            )
            swapped = torch.as_tensor(
                np.array(view.swapped[index : index + 1], copy=True),
                device="cuda",
                dtype=torch.long,
            )
            tokens = torch.as_tensor(
                np.asarray(view.target_tokens[index : index + 1, 0]),
                device="cuda",
                dtype=torch.long,
            )
            correct_logits = _selected_logits(model, correct, positions, torch=torch)
            swapped_logits = _selected_logits(model, swapped, positions, torch=torch)
            correct_lp = correct_logits.float().log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            swapped_lp = swapped_logits.float().log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            prefix = 0
            for actual, gold in zip(generated, expected[:-1]):
                if int(actual) != int(gold):
                    break
                prefix += 1
            first_rank = int(
                1
                + (
                    correct_logits[:, 0]
                    > correct_logits[:, 0, tokens[:, 0].item()].view(-1, 1)
                ).sum().item()
            )
            rows.append(
                {
                    "index": index,
                    "generated_ids": generated.tolist(),
                    "expected_ids": expected.tolist(),
                    "first_token_correct": bool(len(generated) and generated[0] == expected[0]),
                    "answer8_exact": bool(len(generated) >= 8 and np.array_equal(generated[:8], expected[:8])),
                    "full_exact_terminal_eos": bool(np.array_equal(generated, expected)),
                    "terminal_eos_after_answer": bool(len(generated) == len(expected) and generated[-1] == eos_token_id),
                    "correct_prefix_tokens": prefix,
                    "teacher_forced_nll": float((-correct_lp).mean()),
                    "source_effect": float((correct_lp - swapped_lp).mean()),
                    "first_token_gold_rank": first_rank,
                }
            )

    count = len(rows)
    result = {
        "status": "COMPLETE",
        "method": "stage0_fixed_adapter_transport_eval_v1",
        "runtime": runtime,
        "dependencies": dependencies,
        "checkpoint": checkpoint,
        "checkpoint_ready_sha256": sha256_file(args.checkpoint_ready),
        "parent_receipt_sha256": sha256_file(args.parent / "receipt.json"),
        "adapter_sha256": parent_receipt["bundle"]["adapter_files"]["adapter_model.safetensors"],
        "data_root_manifest_sha256": data_root_sha,
        "data_manifest_sha256": _manifest_hash(args.data),
        "length": length,
        "frequency": frequency,
        "examples": count,
        "first_token_exact": sum(row["first_token_correct"] for row in rows) / count,
        "answer8_exact": sum(row["answer8_exact"] for row in rows) / count,
        "full_exact_terminal_eos": sum(row["full_exact_terminal_eos"] for row in rows) / count,
        "terminal_eos_after_answer": sum(row["terminal_eos_after_answer"] for row in rows) / count,
        "mean_correct_prefix_tokens": float(np.mean([row["correct_prefix_tokens"] for row in rows])),
        "mean_teacher_forced_nll": float(np.mean([row["teacher_forced_nll"] for row in rows])),
        "mean_source_effect": float(np.mean([row["source_effect"] for row in rows])),
        "positive_source_effect_fraction": sum(row["source_effect"] > 0 for row in rows) / count,
        "median_first_token_gold_rank": float(np.median([row["first_token_gold_rank"] for row in rows])),
        "rows": rows,
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
