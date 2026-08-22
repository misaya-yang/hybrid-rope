"""Continue the completed Stage-0 adapter on the matched single-query EOS view."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from .receipts import (
    atomic_json,
    canonical_sha256,
    require_dual_authorization,
    sha256_file,
)
from .train_phase_adarope import (
    EXPECTED_DATA_ROOT_SHA256,
    PairView,
    RawReplayView,
    _atomic_jsonl,
    _base_model,
    _checkpoint_identity,
    _code_hashes,
    _configure_cuda,
    _dependency_versions,
    _fresh_reload_check,
    _install_lora,
    _load_base,
    _manifest_hash,
    _save_bundle,
    _set_trainable,
    _train,
    _validate_model_shape,
    public_interfaces,
)


STEPS = 96
MICRO_BATCH = 2
ACCUMULATION = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    args = parser.parse_args()

    require_dual_authorization(cli_authorize=bool(args.authorize))
    import torch

    final_output = args.output.resolve()
    work_output = final_output.with_name(final_output.name + ".incomplete")
    if final_output.exists() or work_output.exists():
        raise FileExistsError(final_output if final_output.exists() else work_output)
    if shutil.disk_usage(final_output.parent).free < 5 * (1 << 30):
        raise RuntimeError("STOP: less than 5 GiB free disk space")

    parent_receipt_path = args.parent.resolve() / "receipt.json"
    parent_receipt = json.loads(parent_receipt_path.read_text(encoding="utf-8"))
    if parent_receipt.get("status") != "COMPLETE" or parent_receipt.get("stage") != "stage0":
        raise ValueError("EOS continuation requires the completed Stage-0 parent")

    dependencies = _dependency_versions(torch)
    checkpoint = _checkpoint_identity(args.checkpoint, args.checkpoint_ready)
    data_root_sha = sha256_file(args.data.resolve().parent / "manifest.json")
    if data_root_sha != EXPECTED_DATA_ROOT_SHA256:
        raise ValueError("V3 data root identity drift")
    data_manifest = json.loads((args.data.resolve() / "manifest.json").read_text(encoding="utf-8"))
    if data_manifest.get("split") != "train_eos" or int(data_manifest.get("length", 0)) != 4096:
        raise ValueError("EOS continuation requires the registered 4K train_eos view")

    runtime = _configure_cuda(torch)
    base = _load_base(args.checkpoint, torch)
    model = _install_lora(base, parent=args.parent.resolve(), trainable=True)
    _set_trainable(model, None, lora=True)
    shape = _validate_model_shape(model, 4096)
    compiled_backbone = torch.compile(
        _base_model(model).model,
        mode=str(args.compile_mode),
        dynamic=False,
    )
    model = model.to("cuda")
    view = PairView.load(args.data)
    replay = RawReplayView.load(args.raw_replay)

    started = time.time()
    result = _train(
        model,
        view,
        steps=STEPS,
        micro_batch=MICRO_BATCH,
        accumulation=ACCUMULATION,
        device=torch.device("cuda"),
        torch=torch,
        smoke=False,
        raw_replay=replay,
        backbone=compiled_backbone,
        stage_name="eos_repair",
        global_start=32,
        global_total=32 + STEPS,
    )

    work_output.mkdir(parents=True, exist_ok=False)
    steps_sha = _atomic_jsonl(work_output / "train_steps.jsonl", result["step_rows"])
    attention = public_interfaces()["attention"]
    bundle = _save_bundle(
        model,
        None,
        work_output,
        attention=attention,
        metadata={
            "stage": "stage0_eos_continue",
            "arm": "native",
            "parent_sha256": parent_receipt.get("content_sha256"),
        },
    )
    roundtrip = _fresh_reload_check(
        checkpoint=args.checkpoint,
        output=work_output,
        model=model,
        bank=None,
        attention=attention,
        torch=torch,
    )
    receipt = {
        "status": "COMPLETE",
        "stage": "stage0_eos_continue",
        "arm": "native",
        "parent_receipt_sha256": sha256_file(parent_receipt_path),
        "parent_content_sha256": parent_receipt.get("content_sha256"),
        "contract": {
            "steps": STEPS,
            "global_start": 32,
            "global_total": 32 + STEPS,
            "training_length": 4096,
            "micro_batch": MICRO_BATCH,
            "gradient_accumulation": ACCUMULATION,
            "objective": "single_query_answer_eos_margin_plus_raw_4k_replay",
        },
        "runtime": runtime,
        "dependencies": dependencies,
        "checkpoint": checkpoint,
        "model_shape": shape,
        "checkpoint_ready_sha256": sha256_file(args.checkpoint_ready),
        "data_root_manifest_sha256": data_root_sha,
        "data_manifest_sha256": _manifest_hash(args.data),
        "raw_replay_manifest_sha256": _manifest_hash(args.raw_replay),
        "code_sha256": {
            **_code_hashes(),
            "continue_stage0_eos.py": sha256_file(Path(__file__).resolve()),
        },
        "train_steps_jsonl_sha256": steps_sha,
        "training": {
            key: result[key]
            for key in (
                "steps",
                "metrics",
                "trainable_parameters",
                "first_step_gradients",
                "memory",
                "tokens_per_second",
            )
        },
        "bundle": bundle,
        "roundtrip": roundtrip,
        "elapsed_seconds": time.time() - started,
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    atomic_json(work_output / "receipt.json", receipt)
    work_output.replace(final_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
