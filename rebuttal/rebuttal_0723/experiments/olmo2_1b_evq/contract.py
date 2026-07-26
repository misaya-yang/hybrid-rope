#!/usr/bin/env python3
"""Frozen scientific and artifact contract for the OLMo-2 EVQ run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from scripts.lib.rope.schedules import evq_cosh_inv_freq


EXPERIMENT_ID = "olmo2_1b_evq_tau2_2p097b"
REVIEW_CONCERNS = ("R27bE.2", "R27bE.5", "AC.2", "AC.4")

MODEL_ID = "allenai/OLMo-2-0425-1B-early-training"
STEP0_REVISION = "9f46fe81fa53e429771f051b36aa51c60f7e6c0f"
GEO1000_REVISION = "aeef9be8719cbdd31e3f89258e722360642afd2d"
GEO2000_REVISION = "07a0254be1449bcc7f78d05602d835464abf43c6"
GEO5000_REVISION = "c667e21af89dd476540c52a2e9912f75580c8271"
STEP0_BRANCH = "stage1-step0-tokens0B"
GEO1000_BRANCH = "stage1-step1000-tokens3B"
GEO2000_BRANCH = "stage1-step2000-tokens5B"
GEO5000_BRANCH = "stage1-step5000-tokens11B"

OLMO_REPOSITORY = "https://github.com/allenai/OLMo.git"
OLMO_COMMIT = "090253dac6688f2532509daa7aa2eb5fae50e956"
OFFICIAL_CONFIG_RELATIVE_PATH = "configs/official-0425/OLMo2-1B-stage1.yaml"
OFFICIAL_CONFIG_URL = (
    "https://raw.githubusercontent.com/allenai/OLMo/"
    f"{OLMO_COMMIT}/{OFFICIAL_CONFIG_RELATIVE_PATH}"
)
OFFICIAL_CONFIG_SHA256 = (
    "bd75e78bf7a818168d7f6e57b561888b380938b6a3abab4f6222ee9b340cd7c7"
)

STEP0_WEIGHT_FILES = {
    "model-00001-of-00002.safetensors": {
        "size": 4_983_360_992,
        "sha256": "f19d1ad425d7988f24451a166fe8b76930f40185d0e093e87216c0db0b011917",
    },
    "model-00002-of-00002.safetensors": {
        "size": 956_326_560,
        "sha256": "7fcde27cace6b639eb7f7adb7aa9eae155c8dbbf74b06f14ca2642e9895f996b",
    },
}
GEO1000_WEIGHT_FILES = {
    "model-00001-of-00002.safetensors": {
        "size": 4_983_360_992,
        "sha256": "e392eaf30a1034a7b89c98610587d43687c1891b609996ebfad033fef1c55fee",
    },
    "model-00002-of-00002.safetensors": {
        "size": 956_326_560,
        "sha256": "dbe3dfb66d4015c1dd6e1172b5c941142e1a6889f802195fc2f25f7416c6d1b2",
    },
}
GEO2000_WEIGHT_FILES = {
    "model-00001-of-00002.safetensors": {
        "size": 4_983_360_992,
        "sha256": "5ddbe9490c5f9046c3d05d4231f85e54c717629d18109b84f0f4c89224661730",
    },
    "model-00002-of-00002.safetensors": {
        "size": 956_326_560,
        "sha256": "40e105307a33f42124dcff1d244c0797f2ea4f3900abc6b0c605ed49e284af01",
    },
}
GEO5000_WEIGHT_FILES = {
    "model-00001-of-00002.safetensors": {
        "size": 4_983_360_992,
        "sha256": "6c01f08fc7d9a7cc9629ce12190c3c038d2f7f950aeab752f67c61da0c784352",
    },
    "model-00002-of-00002.safetensors": {
        "size": 956_326_560,
        "sha256": "8e85b3aed54394174509b8c2062b96d1d2aae07292285b77743c17d8a9262842",
    },
}

MODEL_CONTRACT = {
    "architectures": ["Olmo2ForCausalLM"],
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 128,
    "max_position_embeddings": 4096,
    "rope_theta": 500_000.0,
    "vocab_size": 100_352,
    "tie_word_embeddings": False,
}
ACTUAL_PARAMETER_COUNT = 1_484_916_736

SEED = 6198
SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SEQUENCES = 512
GLOBAL_BATCH_TOKENS = GLOBAL_BATCH_SEQUENCES * SEQUENCE_LENGTH
FIRST_GATE_STEPS = 1000
FIRST_GATE_TOKENS = GLOBAL_BATCH_TOKENS * FIRST_GATE_STEPS
TAU = 2.0

OPTIMIZER_CONTRACT = {
    "name": "adamw",
    "learning_rate": 4.0e-4,
    "weight_decay": 0.1,
    "eps": 1.0e-8,
    "betas": [0.9, 0.95],
    "decay_norm_and_bias": True,
    "decay_embeddings": False,
    "max_grad_norm": 1.0,
    "warmup_tokens": 8_388_608_000,
    "scheduler": "cosine_with_warmup",
    "scheduler_max_tokens": 5_000_000_000_000,
    "scheduler_alpha_f": 0.1,
    "z_loss_multiplier": 1.0e-5,
    "precision": "amp_bf16",
}

TOKENIZER_MARKERS = {
    "eos_token_id": 100_257,
    "pad_token_id": 100_277,
    "embedding_size": 100_352,
    "official_identifier": "tokenizers/allenai_dolma2.json",
}


def sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def endpoint_geo_inv_freq(
    *,
    head_dim: int = MODEL_CONTRACT["head_dim"],
    base: float = MODEL_CONTRACT["rope_theta"],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if int(head_dim) % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    # Match Transformers' default RoPE initialization arithmetic exactly:
    # integer indices are cast to FP32 before exponentiation.  Computing this
    # in FP64 and casting afterwards differs from the native buffer by one ULP
    # at some coordinates even though the real-valued formula is identical.
    indices = torch.arange(0, int(head_dim), 2, dtype=torch.int64).to(
        dtype=torch.float32
    )
    native = 1.0 / (
        torch.tensor(float(base), dtype=torch.float32)
        ** (indices / float(head_dim))
    )
    return native.to(dtype=dtype)


def endpoint_evq_inv_freq(
    *,
    head_dim: int = MODEL_CONTRACT["head_dim"],
    base: float = MODEL_CONTRACT["rope_theta"],
    tau: float = TAU,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """EVQ-Cosh on OLMo's native endpoint grid u_k=k/K."""
    if abs(float(tau)) < 1e-8:
        return endpoint_geo_inv_freq(
            head_dim=int(head_dim),
            base=float(base),
            dtype=dtype,
        )
    return evq_cosh_inv_freq(
        head_dim=int(head_dim),
        tau=float(tau),
        base=float(base),
        midpoint=False,
        dtype=torch.float64,
    ).to(dtype=dtype)


def assert_frequency_contract() -> dict[str, Any]:
    geo = endpoint_geo_inv_freq()
    tau_zero = endpoint_evq_inv_freq(tau=0.0)
    evq = endpoint_evq_inv_freq(tau=TAU)
    if not torch.equal(geo, tau_zero):
        raise RuntimeError("endpoint EVQ tau=0 does not exactly reproduce native Geo")
    if not torch.isfinite(evq).all():
        raise RuntimeError("EVQ inverse frequencies contain NaN or Inf")
    if not bool(torch.all(evq[1:] < evq[:-1])):
        raise RuntimeError("EVQ inverse frequencies are not strictly decreasing")
    if not math.isclose(float(evq[0]), 1.0, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("endpoint EVQ must preserve omega_0=1")
    return {
        "grid": "endpoint",
        "u_formula": "u_k=2k/d_head=k/K",
        "phi_formula": (
            "1-asinh((1-u_k)*sinh(tau))/tau; phi_0=0 and tau=0 limit=u_k"
        ),
        "head_dim": MODEL_CONTRACT["head_dim"],
        "base": MODEL_CONTRACT["rope_theta"],
        "tau": TAU,
        "geo_sha256_float32": tensor_sha256(geo),
        "evq_sha256_float32": tensor_sha256(evq),
        "geo": [float(value) for value in geo],
        "evq": [float(value) for value in evq],
    }


def assert_model_config(config: Any) -> None:
    actual = {
        "architectures": list(config.architectures),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(
            getattr(
                config,
                "head_dim",
                config.hidden_size // config.num_attention_heads,
            )
        ),
        "max_position_embeddings": int(config.max_position_embeddings),
        "rope_theta": float(config.rope_theta),
        "vocab_size": int(config.vocab_size),
        "tie_word_embeddings": bool(config.tie_word_embeddings),
    }
    if actual != MODEL_CONTRACT:
        raise RuntimeError(
            "OLMo-2 model config drift:\n"
            f"expected={json.dumps(MODEL_CONTRACT, sort_keys=True)}\n"
            f"actual={json.dumps(actual, sort_keys=True)}"
        )


def patch_endpoint_evq(model: Any, *, tau: float = TAU) -> dict[str, Any]:
    """Replace only the non-persistent rotary inverse-frequency buffer."""
    rotary = model.model.rotary_emb
    native = rotary.inv_freq.detach().cpu().to(torch.float32)
    expected_native = endpoint_geo_inv_freq()
    if not torch.equal(native, expected_native):
        raise RuntimeError(
            "loaded model's native inv_freq does not match the endpoint Geo contract"
        )
    replacement = endpoint_evq_inv_freq(tau=tau).to(
        device=rotary.inv_freq.device, dtype=rotary.inv_freq.dtype
    )
    rotary.inv_freq.copy_(replacement)
    rotary.original_inv_freq = rotary.inv_freq
    receipt = assert_frequency_contract()
    if tensor_sha256(rotary.inv_freq) != receipt["evq_sha256_float32"]:
        raise RuntimeError("model rotary buffer does not match EVQ receipt")
    return receipt


def named_parameter_metadata(module: torch.nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in module.named_parameters()
    ]


def trainable_parameter_count(module: torch.nn.Module) -> int:
    return sum(
        int(parameter.numel())
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def parameter_identity_digest(module: torch.nn.Module) -> str:
    """Hash parameter bytes; the non-persistent inv_freq buffer is excluded."""
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_weight_files(
    snapshot_dir: Path, expected: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, contract in expected.items():
        path = snapshot_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != contract["size"]:
            raise RuntimeError(f"{path}: size {size} != {contract['size']}")
        digest = sha256_file(path)
        if digest != contract["sha256"]:
            raise RuntimeError(
                f"{path}: sha256 {digest} != {contract['sha256']}"
            )
        rows.append({"path": name, "size": size, "sha256": digest})
    return rows


def relative_file_manifest(root: Path, files: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = root.resolve()
    for path in sorted(files):
        resolved = path.resolve()
        rows.append(
            {
                "path": str(resolved.relative_to(root)),
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return rows
