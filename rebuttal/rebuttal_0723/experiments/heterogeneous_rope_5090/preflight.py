"""CPU-only R3-prime heterogeneous RoPE preflight.

Example (no training, no GPU):

    PYTHONPATH=. python -m \
      rebuttal.rebuttal_0723.experiments.heterogeneous_rope_5090.preflight \
      --r0-json /path/to/r0.json --tier 350m --attn-type mla \
      --d-rope 32 --seq-len 8192 --output /tmp/r3prime_ready.json

The receipt contains per-layer realized frequency hashes, the unchanged
parameter/shape contract, shared-table forward parity when applicable, and a
per-head feasibility gate.  It intentionally does not contain a launch command
or a training status other than the explicit dry-run stop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .dry_run_matrix import build_dry_run_matrix
from .protocol import (
    HeterogeneousRopePlan,
    build_layer_inv_freqs,
    build_model_config,
    install_layerwise_rope,
    load_r0_json,
    model_parameter_contract,
    per_head_feasibility_gate,
    realized_frequency_receipt,
)


def _raw_candidate_profiles(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    for key in ("r0", "R0", "allocation", "heterogeneous_rope"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            payload = value
            break
    value = payload.get("candidate_profiles", payload.get("profiles", {}))
    return value if isinstance(value, Mapping) else {}


def _import_gpt():
    from scripts.core_text_phases.run_gqa_evq_experiment import GPT

    return GPT


def _parameter_values_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_items = list(left.named_parameters())
    right_items = list(right.named_parameters())
    if [name for name, _ in left_items] != [name for name, _ in right_items]:
        return False
    return all(torch.equal(first, second) for (_, first), (_, second) in zip(left_items, right_items))


def _forward_parity(
    shared: torch.nn.Module,
    layerwise: torch.nn.Module,
    *,
    vocab_size: int,
    seq_len: int = 8,
) -> dict[str, Any]:
    if seq_len <= 0:
        raise ValueError("parity seq_len must be positive")
    tokens = (torch.arange(seq_len, dtype=torch.long).view(1, -1) % int(vocab_size)).contiguous()
    shared.eval()
    layerwise.eval()
    with torch.no_grad():
        left = shared(tokens)
        right = layerwise(tokens)
    difference = (left - right).abs()
    max_abs = float(difference.max())
    return {
        "status": "PASS" if torch.equal(left, right) else "CHECK_NUMERIC",
        "input_shape": list(tokens.shape),
        "logits_shape": list(left.shape),
        "max_abs_diff": max_abs,
        "exact_equal": bool(torch.equal(left, right)),
        "finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
    }


def _assert_parameter_contract(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    same_count = before["parameter_count"] == after["parameter_count"]
    same_tensors = before["parameter_tensor_count"] == after["parameter_tensor_count"]
    same_schema = before["parameter_schema_sha256"] == after["parameter_schema_sha256"]
    result = {
        "status": "PASS" if same_count and same_tensors and same_schema else "FAIL",
        "parameter_count_before": before["parameter_count"],
        "parameter_count_after": after["parameter_count"],
        "parameter_tensor_count_before": before["parameter_tensor_count"],
        "parameter_tensor_count_after": after["parameter_tensor_count"],
        "parameter_schema_same": same_schema,
    }
    if result["status"] != "PASS":
        raise AssertionError(f"parameter contract changed: {result}")
    return result


def run_preflight(
    plan: HeterogeneousRopePlan,
    config: Mapping[str, Any],
    *,
    seed: int = 42,
    run_forward: bool = True,
    parity_seq_len: int = 8,
    candidate_profiles: Mapping[str, Any] | None = None,
    shared_tau_values: list[float] | None = None,
) -> dict[str, Any]:
    """Build shared and layerwise models on CPU and emit a receipt."""

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if torch.cuda.is_available() or mps_available:
        # The preflight is intentionally CPU-only even on an accelerator host.
        # This is a guard against accidentally turning preparation into paid
        # compute.  GPT is never moved off CPU below.
        device = "cpu_forced"
    else:
        device = "cpu"
    if str(config.get("attn_type", plan.attention_type)).lower() != plan.attention_type:
        raise ValueError("config attn_type and R0 attention_type disagree")
    if int(config.get("num_layers", plan.num_layers)) != plan.num_layers:
        raise ValueError("config num_layers and R0 num_layers disagree")
    expected_rope_dim = int(config.get("d_rope", plan.rope_dim)) if plan.attention_type == "mla" else int(config.get("head_dim", plan.rope_dim))
    if expected_rope_dim != plan.rope_dim:
        raise ValueError(
            f"config RoPE dimension {expected_rope_dim} != R0 rope_dim {plan.rope_dim}"
        )

    layer_inv_freqs = build_layer_inv_freqs(plan, dtype=torch.float32)
    GPT = _import_gpt()

    torch.manual_seed(int(seed))
    shared_model = GPT(dict(config), layer_inv_freqs[0]).cpu()
    shared_contract = model_parameter_contract(shared_model)
    per_head_gate = per_head_feasibility_gate(shared_model, config)

    torch.manual_seed(int(seed))
    layerwise_model = GPT(dict(config), layer_inv_freqs[0]).cpu()
    if not _parameter_values_equal(shared_model, layerwise_model):
        raise AssertionError("shared and layerwise pre-install parameters differ under same seed")
    before_layerwise_contract = model_parameter_contract(layerwise_model)
    install_receipt = install_layerwise_rope(layerwise_model, layer_inv_freqs)
    after_layerwise_contract = model_parameter_contract(layerwise_model)
    parameter_contract = _assert_parameter_contract(
        before_layerwise_contract, after_layerwise_contract
    )

    all_same = all(torch.equal(layer_inv_freqs[0], value) for value in layer_inv_freqs[1:])
    if run_forward and all_same:
        parity = _forward_parity(
            shared_model,
            layerwise_model,
            vocab_size=int(config["vocab_size"]),
            seq_len=int(parity_seq_len),
        )
    elif all_same:
        parity = {"status": "SKIPPED_BY_FLAG", "exact_equal": None}
    else:
        parity = {
            "status": "NOT_APPLICABLE_HETEROGENEOUS",
            "reason": "forward parity is a shared-table gate; heterogeneous rows are not expected to match",
            "exact_equal": None,
        }

    matrix = build_dry_run_matrix(
        plan,
        candidate_profiles=candidate_profiles,
        shared_tau_values=shared_tau_values,
    )
    return {
        "schema_version": 1,
        "status": "READY_FOR_AUTHORIZED_GATE" if parameter_contract["status"] == "PASS" else "FAIL",
        "mode": "CPU_ONLY_PREFLIGHT",
        "device": device,
        "training_started": False,
        "training_authorized": False,
        "gpu_training_allowed": False,
        "seed_for_initialization_parity": int(seed),
        "plan": plan.as_receipt(),
        "model_config": {
            key: value
            for key, value in dict(config).items()
            if key not in {"train_data", "validation_data", "tokenizer"}
        },
        "shared_model_parameter_contract": shared_contract,
        "layerwise_installation": install_receipt,
        "layerwise_parameter_contract": parameter_contract,
        "realized_frequency": realized_frequency_receipt(layerwise_model),
        "forward_parity": parity,
        "per_head_feasibility_gate": per_head_gate,
        "dry_run_matrix": matrix,
        "limitations": [
            "No optimizer, data loader, backward pass, CUDA context, or GPU inference was executed.",
            "Per-head tables remain gated until a head-axis cos/sin contract is separately implemented and tested.",
            "A READY receipt proves code/config/model-shape preflight only; it is not training readiness or a result.",
        ],
    }


def _plan_and_config(args: argparse.Namespace) -> tuple[HeterogeneousRopePlan, dict[str, Any], Mapping[str, Any]]:
    r0_path = Path(args.r0_json).resolve()
    plan = load_r0_json(r0_path)
    if args.attention_type is not None and args.attention_type != plan.attention_type:
        raise ValueError(
            f"--attention-type={args.attention_type} disagrees with R0 "
            f"attention_type={plan.attention_type}"
        )
    if args.d_rope is not None and args.d_rope != plan.rope_dim:
        raise ValueError(
            f"--d-rope={args.d_rope} disagrees with R0 rope_dim={plan.rope_dim}"
        )
    config = build_model_config(
        tier=args.tier,
        attention_type=plan.attention_type,
        seq_len=args.seq_len if args.seq_len is not None else plan.train_length,
        d_rope=plan.rope_dim if plan.attention_type == "mla" else None,
        d_nope=plan.d_nope,
        n_kv_heads=plan.n_kv_heads,
    )
    if config["num_layers"] != plan.num_layers:
        raise ValueError(
            f"tier {args.tier} has {config['num_layers']} layers but R0 has {plan.num_layers}; "
            "select the matching tier or use run_preflight with an explicit tiny config"
        )
    candidate_profiles = _raw_candidate_profiles(r0_path)
    return plan, config, candidate_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-json", required=True, type=Path)
    parser.add_argument("--tier", default="50m")
    parser.add_argument(
        "--attention-type",
        "--attn-type",
        dest="attention_type",
        choices=("mha", "gqa", "mla"),
    )
    parser.add_argument("--d-rope", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parity-seq-len", type=int, default=8)
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--shared-tau-grid", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan, config, candidate_profiles = _plan_and_config(args)
    shared_grid = [float(value) for value in args.shared_tau_grid.split(",") if value.strip()]
    receipt = run_preflight(
        plan,
        config,
        seed=args.seed,
        run_forward=not args.skip_forward,
        parity_seq_len=args.parity_seq_len,
        candidate_profiles=candidate_profiles,
        shared_tau_values=shared_grid,
    )
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
