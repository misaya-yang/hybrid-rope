"""Build a training matrix receipt without launching any run.

The matrix is deliberately declarative.  It contains layer-wise realized
``tau`` values, model-shape metadata, and an explicit ``training_authorized``
false gate; it contains no metrics and makes no scientific decision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import (
    HeterogeneousRopePlan,
    build_layer_inv_freqs,
    hash_tensor_raw,
    load_r0_json,
)


def _profile_values(profile: Any, *, plan: HeterogeneousRopePlan, name: str) -> tuple[float, ...]:
    """Resolve a candidate profile to per-layer tau values."""

    if isinstance(profile, Mapping):
        kind = str(profile.get("kind", "tau")).lower()
        values = profile.get("values", profile.get("profile"))
    else:
        kind = "tau"
        values = profile
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"candidate profile {name!r} must contain an array")
    if len(values) != plan.num_layers:
        raise ValueError(
            f"candidate profile {name!r} length {len(values)} != num_layers {plan.num_layers}"
        )
    if kind == "tau":
        return tuple(float(value) for value in values)
    if kind == "m":
        factor = plan.effective_dim / math.sqrt(float(plan.train_length))
        return tuple(float(value) * factor for value in values)
    raise ValueError(f"candidate profile {name!r} kind must be tau or m, got {kind!r}")


def _row(
    *,
    arm_id: str,
    plan: HeterogeneousRopePlan,
    layer_tau: Sequence[float] | None,
    source: str,
) -> dict[str, Any]:
    if layer_tau is None:
        inv = build_layer_inv_freqs(plan)
        tau_receipt = None
    else:
        if len(layer_tau) != plan.num_layers:
            raise ValueError(f"{arm_id}: layer_tau length mismatch")
        candidate = plan_from_tau(plan, layer_tau)
        inv = build_layer_inv_freqs(candidate)
        tau_receipt = [float(value) for value in layer_tau]
    return {
        "arm_id": arm_id,
        "source": source,
        "attention_type": plan.attention_type,
        "num_layers": plan.num_layers,
        "rope_dim": plan.rope_dim,
        "base": plan.base,
        "train_length": plan.train_length,
        "layer_tau": tau_receipt,
        "layer_inv_freq_sha256_raw": [hash_tensor_raw(value) for value in inv],
        "parameter_count_expected_unchanged": True,
        "training_authorized": False,
        "status": "DRY_RUN_ONLY",
        "metrics": None,
    }


def plan_from_tau(plan: HeterogeneousRopePlan, layer_tau: Sequence[float]) -> HeterogeneousRopePlan:
    from .protocol import plan_from_values

    return plan_from_values(
        num_layers=plan.num_layers,
        rope_dim=plan.rope_dim,
        base=plan.base,
        train_length=plan.train_length,
        effective_dim=plan.effective_dim,
        tau=list(layer_tau),
        attention_type=plan.attention_type,
        num_heads=plan.num_heads,
        head_dim=plan.head_dim,
        d_rope=plan.d_rope,
        d_nope=plan.d_nope,
        n_kv_heads=plan.n_kv_heads,
    )


def build_dry_run_matrix(
    plan: HeterogeneousRopePlan,
    *,
    candidate_profiles: Mapping[str, Any] | None = None,
    shared_tau_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return a declarative matrix suitable for later authorized training.

    If no candidate profiles are supplied, the output includes only the R0
    configuration and, when heterogeneous, a shared-mean reference.  This
    avoids inventing an experimental result or silently selecting a profile.
    """

    configured_tau = plan.taus if all(item.tau is not None for item in plan.layers) else None
    rows: list[dict[str, Any]] = []
    if configured_tau is not None:
        rows.append(
            _row(
                arm_id="configured_r0",
                plan=plan,
                layer_tau=configured_tau,
                source="R0",
            )
        )
    else:
        rows.append(
            _row(
                arm_id="configured_r0",
                plan=plan,
                layer_tau=None,
                source="R0_direct_inv_freq",
            )
        )
    if configured_tau is not None and not plan.all_layers_share_tau:
        mean_tau = sum(configured_tau) / len(configured_tau)
        rows.append(
            _row(
                arm_id="shared_mean_tau_reference",
                plan=plan,
                layer_tau=[mean_tau] * plan.num_layers,
                source="derived_reference_not_result",
            )
        )
    if shared_tau_values:
        for value in shared_tau_values:
            tau = float(value)
            rows.append(
                _row(
                    arm_id=f"shared_tau_{tau:g}",
                    plan=plan,
                    layer_tau=[tau] * plan.num_layers,
                    source="CLI_candidate",
                )
            )
    for name, profile in (candidate_profiles or {}).items():
        values = _profile_values(profile, plan=plan, name=str(name))
        rows.append(
            _row(
                arm_id=str(name),
                plan=plan,
                layer_tau=values,
                source="R0_candidate_profile",
            )
        )

    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique[row["arm_id"]] = row
    return {
        "schema_version": 1,
        "status": "DRY_RUN_ONLY",
        "training_started": False,
        "training_authorized": False,
        "plan": plan.as_receipt(),
        "arms": list(unique.values()),
        "decision_scope": "planning receipt only; no metrics, ranking, or Pareto claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-json", required=True, type=Path)
    parser.add_argument("--shared-tau-grid", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.r0_json.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("R0 JSON root must be an object")
    root = payload
    for key in ("r0", "R0", "allocation", "heterogeneous_rope"):
        value = root.get(key)
        if isinstance(value, Mapping):
            root = value
            break
    profiles = root.get("candidate_profiles", root.get("profiles", {}))
    if not isinstance(profiles, Mapping):
        profiles = {}
    shared = [float(value) for value in args.shared_tau_grid.split(",") if value.strip()]
    result = build_dry_run_matrix(
        load_r0_json(args.r0_json),
        candidate_profiles=profiles,
        shared_tau_values=shared,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
