"""Read-only R4' protocol contracts.

The module intentionally contains no model imports and no execution path.  It
can build manifests, validate the proposed seed matrix, and describe the
registered gates before any checkpoint or dataset is bound.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

METHOD_ID = "olmo2_demand_retrofit_r4prime_v1"
PROTOCOL_VERSION = 1
PHYSICAL_TRAIN_LENGTH = 4_096
ROTARY_PAIRS = 64
HEAD_DIM = 128
LAYERS = 16
HIDDEN_SIZE = 2_048
MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"

DEFAULT_SEEDS = (20_260_821, 20_260_822, 20_260_823)
DEFAULT_PHASE_EXPOSURE = {
    "continuous_4k_batches": 1,
    "target_8k_phase_batches": 1,
    "target_16k_phase_batches": 2,
}


class ContractError(ValueError):
    """Raised when a registered contract is malformed or unsafe."""


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ContractError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class NegativeMorphReproduction:
    """Historical A1-D1 contrast; never a new training arm by default."""

    enabled: bool = False
    mode: str = "reproduce-only-historical-contrast"
    arm_ids: tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1", "C2", "D1")
    source_owner: str = (
        "rebuttal/rebuttal_0723/theory_results/"
        "olmo2_1b_non_ruler_adaptation_search_20260731.json"
    )
    training_entrypoint: str | None = None
    claim_role: str = "historical-negative-only"

    def validate(self) -> None:
        if self.enabled:
            raise ContractError(
                "historical morph reproduction is disabled in R4'; "
                "do not turn the seven-arm negative into a new experiment"
            )
        if self.mode != "reproduce-only-historical-contrast":
            raise ContractError("negative morph mode drift")
        if self.training_entrypoint is not None:
            raise ContractError("negative morph must not carry a training entrypoint")
        if self.claim_role != "historical-negative-only":
            raise ContractError("negative morph claim role drift")
        if tuple(self.arm_ids) != ("A1", "A2", "B1", "B2", "C1", "C2", "D1"):
            raise ContractError("seven-arm historical identity drift")


@dataclass(frozen=True)
class LossWeights:
    """Teacher-targeted attention-function objective weights."""

    attention_kl: float = 1.0
    context_mse: float = 1.0
    relation_kl: float = 0.25
    lambda_target: str = (
        "native_teacher_post_rope_attention_and_context"
    )

    def validate(self) -> None:
        for name in ("attention_kl", "context_mse", "relation_kl"):
            _finite_nonnegative(getattr(self, name), name)
        if float(self.attention_kl) <= 0.0 or float(self.context_mse) <= 0.0:
            raise ContractError("attention and context targets must be active")
        if self.lambda_target != (
            "native_teacher_post_rope_attention_and_context"
        ):
            raise ContractError("lambda target identity drift")


@dataclass(frozen=True)
class PhaseCurriculum:
    """Explicit target-phase exposure with a physical 4K token cap."""

    continuous_4k_batches: int = 1
    target_8k_phase_batches: int = 1
    target_16k_phase_batches: int = 2
    physical_sequence_length: int = PHYSICAL_TRAIN_LENGTH
    explicit_position_ids: bool = True
    virtual_position_ids: bool = False

    def validate(self) -> None:
        if self.physical_sequence_length != PHYSICAL_TRAIN_LENGTH:
            raise ContractError("physical sequence length must remain 4096")
        if any(
            int(getattr(self, name)) <= 0
            for name in (
                "continuous_4k_batches",
                "target_8k_phase_batches",
                "target_16k_phase_batches",
            )
        ):
            raise ContractError("phase curriculum must contain every phase")
        if not self.explicit_position_ids or self.virtual_position_ids:
            raise ContractError(
                "target phase exposure requires explicit position IDs and "
                "forbids a virtual-position alias"
            )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class GateContract:
    """Registered capability gates; failure stops the route."""

    two_wiki_token_f1_drop_max_points: float = 5.0
    ruler_macro_drop_max_points: float = 10.0
    natural_nll_delta_max: float = 0.10
    require_no_native_positive_family_collapse: bool = True
    require_independent_retention_slice: bool = True
    eight_k_requires_four_k_pass: bool = True
    eight_k_requires_strict_autoregressive_metric: bool = True

    def validate(self) -> None:
        _finite_nonnegative(
            self.two_wiki_token_f1_drop_max_points,
            "two_wiki_token_f1_drop_max_points",
        )
        _finite_nonnegative(
            self.ruler_macro_drop_max_points,
            "ruler_macro_drop_max_points",
        )
        _finite_nonnegative(
            self.natural_nll_delta_max,
            "natural_nll_delta_max",
        )
        if not self.require_no_native_positive_family_collapse:
            raise ContractError("family-collapse protection cannot be disabled")
        if not self.require_independent_retention_slice:
            raise ContractError("independent retention gate cannot be disabled")
        if not self.eight_k_requires_four_k_pass:
            raise ContractError("8K gate cannot bypass the 4K gate")
        if not self.eight_k_requires_strict_autoregressive_metric:
            raise ContractError("8K gate cannot use NLL/PPL as its endpoint")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class QKOnlyProtocol:
    """Matched Native/EVQ Q/K-only continuation matrix."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    arms: tuple[str, ...] = ("native", "evq")
    model_id: str = MODEL_ID
    steps: int = 300
    rank: int = 64
    alpha: float = 128.0
    trainable_qk_parameters: int = 8_388_608
    frozen_inherited_vo_parameters: int = 8_388_608
    learning_rate: float = 5e-5
    warmup_steps: int = 20
    physical_sequence_length: int = PHYSICAL_TRAIN_LENGTH
    lambda_target: str = (
        "native_teacher_post_rope_attention_and_context"
    )
    loss_weights: LossWeights = field(default_factory=LossWeights)
    phase_curriculum: PhaseCurriculum = field(default_factory=PhaseCurriculum)
    gates: GateContract = field(default_factory=GateContract)
    status: str = "PROPOSED_NOT_RUN"

    def validate(self) -> None:
        if self.model_id != MODEL_ID:
            raise ContractError("model identity drift")
        if self.status != "PROPOSED_NOT_RUN":
            raise ContractError("Q/K matrix must remain unrun")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ContractError("matched matrix requires distinct seeds")
        if tuple(self.arms) != ("native", "evq"):
            raise ContractError("matched matrix must contain Native and EVQ arms")
        if self.steps != 300 or self.rank != 64 or self.alpha != 128.0:
            raise ContractError("Q/K-only continuation contract drift")
        if self.trainable_qk_parameters != 8_388_608:
            raise ContractError("Q/K trainable-parameter count drift")
        if self.frozen_inherited_vo_parameters != 8_388_608:
            raise ContractError("frozen V/O parameter count drift")
        if self.physical_sequence_length != PHYSICAL_TRAIN_LENGTH:
            raise ContractError("Q/K physical sequence length drift")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ContractError("learning rate must be finite and positive")
        if self.warmup_steps <= 0:
            raise ContractError("warmup steps must be positive")
        if self.lambda_target != self.loss_weights.lambda_target:
            raise ContractError("lambda target and loss target disagree")
        self.loss_weights.validate()
        self.phase_curriculum.validate()
        self.gates.validate()

    def seed_matrix(self) -> list[dict[str, Any]]:
        self.validate()
        rows: list[dict[str, Any]] = []
        for seed in self.seeds:
            for arm in self.arms:
                rows.append(
                    {
                        "seed": int(seed),
                        "arm": arm,
                        "frequency_table": (
                            "native_original_endpoint"
                            if arm == "native"
                            else "endpoint_evq_cosh_tau2"
                        ),
                        "status": "PLANNED_NOT_RUN",
                        "physical_sequence_length": self.physical_sequence_length,
                        "phase_curriculum": self.phase_curriculum.as_dict(),
                        "lambda_target": self.lambda_target,
                    }
                )
        return rows

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["seed_matrix"] = self.seed_matrix()
        return payload


@dataclass(frozen=True)
class ProtectedTableConfig:
    """Conditional table interface; raw index splicing is never accepted."""

    enabled: bool = False
    adapter_mode: str = "conditional-candidate-required"
    max_protected_pairs: int = 16
    collision_gap_fraction_of_native_spacing: float = 0.20
    protected_set_source: str = "offline_native_importance_receipt"
    never_claim_theorem_noop: bool = True

    def validate(self) -> None:
        if self.adapter_mode != "conditional-candidate-required":
            raise ContractError("protected table adapter mode drift")
        if not 0 <= self.max_protected_pairs <= ROTARY_PAIRS:
            raise ContractError("protected pair budget is invalid")
        if not (
            math.isfinite(self.collision_gap_fraction_of_native_spacing)
            and self.collision_gap_fraction_of_native_spacing > 0.0
        ):
            raise ContractError("collision gap fraction must be positive")
        if not self.never_claim_theorem_noop:
            raise ContractError("theorem scope cannot be disabled")


@dataclass(frozen=True)
class SlowResidualConfig:
    """Low-dimensional Native + slow EVQ residual route."""

    enabled: bool = False
    route_threshold: int = PHYSICAL_TRAIN_LENGTH
    minimum_wavelength_tokens: float = 500_000.0
    max_pairs: int = 16
    residual_head_dim: int = 32
    global_native_path_frozen: bool = True
    short_route_bitwise_gate_required: bool = True

    def validate(self) -> None:
        if self.route_threshold != PHYSICAL_TRAIN_LENGTH:
            raise ContractError("residual route threshold drift")
        if (
            not math.isfinite(self.minimum_wavelength_tokens)
            or self.minimum_wavelength_tokens <= 0.0
        ):
            raise ContractError("minimum residual wavelength must be positive")
        if not 1 <= self.max_pairs <= 16:
            raise ContractError("slow residual pair budget must be in [1, 16]")
        if self.residual_head_dim != 2 * self.max_pairs:
            raise ContractError(
                "residual head dimension must equal two coordinates per pair"
            )
        if not self.global_native_path_frozen:
            raise ContractError("Native global path must remain frozen")
        if not self.short_route_bitwise_gate_required:
            raise ContractError("4K bitwise gate cannot be disabled")


@dataclass(frozen=True)
class ExecutionConfig:
    """Hard no-training/no-download boundary for this preparation bundle."""

    default_action: str = "dry-run"
    training_authorized: bool = False
    gpu_enabled: bool = False
    downloads_allowed: bool = False
    network_allowed: bool = False
    flash_only: bool = True
    allow_attention_fallback: bool = False

    def validate(self) -> None:
        if self.default_action != "dry-run":
            raise ContractError("default action must remain dry-run")
        if self.training_authorized or self.gpu_enabled:
            raise ContractError("R4' preparation cannot authorize GPU training")
        if self.downloads_allowed or self.network_allowed:
            raise ContractError("R4' preparation cannot download or use network")
        if not self.flash_only or self.allow_attention_fallback:
            raise ContractError("Flash-only/no-fallback contract drift")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON manifest must be an object: {path}")
    return value


def build_protocol_manifest(
    *,
    source_manifest: Mapping[str, Any],
    execution: ExecutionConfig | None = None,
    negative_morph: NegativeMorphReproduction | None = None,
    protected_table: ProtectedTableConfig | None = None,
    slow_residual: SlowResidualConfig | None = None,
    qk_matrix: QKOnlyProtocol | None = None,
) -> dict[str, Any]:
    execution = execution or ExecutionConfig()
    negative_morph = negative_morph or NegativeMorphReproduction()
    protected_table = protected_table or ProtectedTableConfig()
    slow_residual = slow_residual or SlowResidualConfig()
    qk_matrix = qk_matrix or QKOnlyProtocol()

    execution.validate()
    negative_morph.validate()
    protected_table.validate()
    slow_residual.validate()
    qk_matrix.validate()

    manifest = {
        "schema_version": PROTOCOL_VERSION,
        "method_id": METHOD_ID,
        "status": "PREPARED_NO_GPU_DEFAULT_DISABLED",
        "model_contract": {
            "model_id": MODEL_ID,
            "actual_parameters": 1_484_916_736,
            "hidden_size": HIDDEN_SIZE,
            "layers": LAYERS,
            "attention_heads": 16,
            "head_dim": HEAD_DIM,
            "rotary_pairs": ROTARY_PAIRS,
            "rope_base": 500_000.0,
            "physical_train_length": PHYSICAL_TRAIN_LENGTH,
        },
        "execution": asdict(execution),
        "negative_morph_reproduction": asdict(negative_morph),
        "protected_table": asdict(protected_table),
        "slow_residual": asdict(slow_residual),
        "qk_only_matrix": qk_matrix.as_dict(),
        "source_manifest": dict(source_manifest),
        "claim_boundary": [
            "historical seven-arm morph is reproduce-only and disabled",
            "protected table requires an offline selected set and a conditional candidate",
            "Theorem 3 remains an exact obstruction for changed frequencies",
            "slow residual is a new Native-plus-residual attention operator",
            "all capability gates are planned, not completed evidence",
        ],
    }
    manifest["protocol_sha256"] = canonical_json_hash(manifest)
    return manifest


def ensure_no_secret_like_values(value: Any, *, path: str = "$") -> None:
    """Reject obvious credential/path fields from durable receipts."""

    secret_tokens = ("password", "token", "api_key", "secret", "private_key")
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in secret_tokens):
                raise ContractError(f"secret-like field is forbidden in receipt: {path}.{key}")
            ensure_no_secret_like_values(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            ensure_no_secret_like_values(child, path=f"{path}[{index}]")
