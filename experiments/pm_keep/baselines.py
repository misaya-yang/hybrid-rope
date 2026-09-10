"""Pinned author implementations for PM-Keep's external baselines.

EA's scoring formula is executed from its original source, not reimplemented.
Only package initialization is skipped: KVPress's __init__ imports unrelated GPU
packages and globally patches attention, neither of which a score callback needs.
The caller owns fixed-budget selection and original-position cache gathering.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Any

import torch

KVPRESS_COMMIT = "71640b4f9061054a7630c5049bb9ee659a01523c"
KVPRESS_REPOSITORY = "https://github.com/NVIDIA/kvpress"
PINNED_SHA256 = {
    "kvpress/utils.py": "e8a19f23bfec0e06c058a9563b86f823935849ac7b93287657d602e4f694fce2",
    "kvpress/presses/base_press.py": "b37e5d08bce40ef90cd3dde1e9853ed7fa3fdc9996d42942a2dc6da0fd16fe1a",
    "kvpress/presses/scorer_press.py": "39191c9fb161238a2c44b32df8852e7406c3026423cd198470a0be026b8dfc50",
    "kvpress/presses/expected_attention_press.py": "6fbae9ab7b976e295959a53b80d145fac3687138cd91fa8fcf0e39d48f6f6c00",
    "kvpress/presses/keydiff_press.py": "4c3d43816e760887f3d112461dedd23ef68de0c3cab22b5382423a1688616738",
    "kvpress/presses/kvzip_press.py": "f604639314622d1b2abc32bdb0a6dcbc0ed2bd3f716d4bdbf61f4945bc6e5f98",
}
EA_DEFAULTS = {
    "compression_ratio": 0.0,
    "n_future_positions": 512,
    "n_sink": 4,
    "use_covariance": True,
    "use_vnorm": True,
    "epsilon": 0.0,
}
KEYDIFF_DEFAULTS = {"compression_ratio": 0.0}


class UnsupportedBaseline(RuntimeError):
    """A requested baseline cannot be faithfully executed by this entry point."""


def _root(source_root: str | Path | None) -> Path:
    return Path(source_root or os.environ.get("PM_KEEP_KVPRESS_ROOT", "/tmp/hybrid-kvpress-20260909")).resolve()


def source_receipt(source_root: str | Path | None = None) -> dict[str, Any]:
    """Verify exact source blobs; a copied, git-free snapshot is supported."""
    root = _root(source_root)
    hashes = {}
    for relative, expected in PINNED_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise UnsupportedBaseline(f"Missing pinned KVPress source: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise UnsupportedBaseline(f"Pinned source SHA256 mismatch: {path}; expected {expected}, got {actual}")
        hashes[relative] = actual
    actual_commit = None
    if (root / ".git").exists():
        actual_commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if actual_commit != KVPRESS_COMMIT:
            raise UnsupportedBaseline(f"KVPress HEAD is {actual_commit}, expected {KVPRESS_COMMIT}")
    import transformers
    return {
        "repository": KVPRESS_REPOSITORY,
        "pinned_commit": KVPRESS_COMMIT,
        "checkout_commit": actual_commit,
        "source_root": str(root),
        "source_sha256": hashes,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "ea_actual_defaults": dict(EA_DEFAULTS),
        "loader": "original source modules; package __init__ and global attention patches not executed",
        "ea_score_scope": "all prefix layer-input hidden states excluding first four sinks; no future question",
        "official_ea_allocation": "author ScorerPress top-k; EA protects four sinks, no protected recent tail",
        "unified_allocation": "adaptation: fixed sink=4 and recent=256 are included in total per-head B",
        "keydiff_scope": "KVPress original KeyDiff score, single-pass allocation adaptation; not paper BlockPress protocol",
    }


def _namespace(name: str, path: Path) -> None:
    existing = sys.modules.get(name)
    if existing is not None:
        locations = {str(Path(p).resolve()) for p in getattr(existing, "__path__", [])}
        if str(path) not in locations:
            raise UnsupportedBaseline(f"A different {name} package is already loaded; use a clean process for pinned sources")
        if getattr(existing, "__file__", None):
            raise UnsupportedBaseline(f"{name}.__init__ was already executed; use a clean process without global KVPress patches")
        return
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = [str(path)]
    sys.modules[name] = module


def load_author_class(name: str, source_root: str | Path | None = None):
    """Load an unchanged, hash-verified author class without importing all presses."""
    module_names = {
        "ExpectedAttentionPress": "expected_attention_press",
        "KeyDiffPress": "keydiff_press",
        "KVzipPress": "kvzip_press",
    }
    if name not in module_names:
        raise UnsupportedBaseline(f"No verified author entry point for {name}")
    root = _root(source_root)
    source_receipt(root)
    _namespace("kvpress", root / "kvpress")
    _namespace("kvpress.presses", root / "kvpress/presses")
    full_name = f"kvpress.presses.{module_names[name]}"
    module = importlib.import_module(full_name)
    expected_path = root / "kvpress/presses" / (module_names[name] + ".py")
    if Path(module.__file__).resolve() != expected_path:
        raise UnsupportedBaseline(f"Loaded {full_name} from an unpinned path")
    return getattr(module, name)


def _prefix_args(data: Any):
    module, hidden, keys, values = data.attention_module, data.hidden_states, data.keys, data.values
    length = int(data.prefix_length)
    if hidden.ndim != 3 or keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("Expected hidden [1,T,Dmodel] and K/V [1,Hkv,T,D]")
    if hidden.shape[0] != 1 or keys.shape[0] != 1:
        raise ValueError("PM-Keep baseline callback supports a single unpadded prefix")
    if hidden.shape[1] != length or keys.shape[2] != length:
        raise ValueError("Official EA requires the complete prefix layer input and matching uncompressed K/V")
    if length <= 4:
        raise ValueError("Prefix must contain more than the four EA sink tokens")
    if keys.shape[-1] != module.head_dim or module.config.num_attention_heads % keys.shape[1]:
        raise ValueError("Head layout disagrees with the model's original GQA configuration")
    if hidden.device != keys.device or values.device != keys.device:
        raise ValueError("Prefix hidden states and K/V must be on the same device")
    return module, hidden, keys, values


@torch.inference_mode()
def ea_prefix_scores(data: Any, *, source_root: str | Path | None = None) -> torch.Tensor:
    """Return original author EA scores [Hkv,T]; do not prune in the hook.

    data requires attention_module, hidden_states, keys, values, prefix_length.
    The adapter temporarily binds module.rotary_emb to the model's real rotary
    module. It must call this before any future question enters the model.
    """
    module, hidden, keys, values = _prefix_args(data)
    if not callable(getattr(module, "rotary_emb", None)):
        raise ValueError("Bind attention_module.rotary_emb to the model's native rotary module")
    cls = load_author_class("ExpectedAttentionPress", source_root)
    press = cls(**EA_DEFAULTS)
    for name, value in EA_DEFAULTS.items():
        if getattr(press, name) != value:
            raise UnsupportedBaseline(f"Unexpected EA setting {name}")
    scores = press.score(module, hidden, keys, values, attentions=None, kwargs={})
    if scores.shape != keys.shape[:3] or not torch.isfinite(scores).all():
        raise RuntimeError("Author EA produced nonfinite scores or an incompatible score shape")
    return scores[0]


@torch.inference_mode()
def keydiff_prefix_scores(data: Any, *, source_root: str | Path | None = None) -> torch.Tensor:
    """Original KVPress KeyDiff scores, not its iterative BlockPress protocol.

    This is an additional real query-agnostic control. Its single-pass unified
    protection policy must not be advertised as full paper reproduction or a
    substitute for the deferred reconstruction-based KVzip comparison.
    """
    module, hidden, keys, values = _prefix_args(data)
    press = load_author_class("KeyDiffPress", source_root)(**KEYDIFF_DEFAULTS)
    scores = press.score(module, hidden, keys, values, attentions=None, kwargs={})
    if scores.shape != keys.shape[:3] or not torch.isfinite(scores).all():
        raise RuntimeError("Author KeyDiff produced nonfinite scores or an incompatible score shape")
    return scores[0]


def ea_official_keep_indices(scores: torch.Tensor, total_budget: int) -> torch.Tensor:
    """Author EA/ScorerPress top-k policy, returned in original token order.

    Unlike the study's protection adaptation this adds no forced recent tail.
    Original EA score already gives its four sinks higher scores. Sorting only
    orders the chosen slots for the original-position adapter; it does not
    change the keep set or rescale its scores.
    """
    if scores.ndim != 2 or not 1 <= total_budget <= scores.shape[-1]:
        raise ValueError("Expected [Hkv,T] scores and 1 <= total_budget <= T")
    if not torch.isfinite(scores).all():
        raise ValueError("Selection scores must be finite")
    return scores.topk(total_budget, dim=-1).indices.sort(dim=-1).values


def strong_baseline_status() -> dict[str, Any]:
    return {
        "KeyDiff_score": {
            "status": "implemented_original_author_score",
            "label": "KVPress KeyDiff single-pass score with unified protection adaptation",
            "paper_reproduction": False,
            "missing_paper_policy": "iterative BlockPress; no claim of full KeyDiff benchmark reproduction",
        },
        "KVzip": {
            "status": "deferred_not_runtime_validated",
            "source_entry_point": "load_author_class('KVzipPress')",
            "defaults": {"layerwise": False, "n_sink": 4, "kvzip_plus_normalization": False},
            "requires": [
                "full model and matching tokenizer/chat-prefix boundary",
                "author context reconstruction forward passes using prefix only",
                "capture score_val before author __call__ finally resets it",
                "count reconstruction tokens, scoring time and peak cache memory",
                "explicit allocation adapter: author policy is across heads/layers and uses fake masked compression",
                "do not call unpatched author context and assume masked_key_indices prunes cache",
            ],
        },
        "FastKVzip": {"status": "deferred", "reason": "released trained gate weights and exact model match not verified"},
        "TriAttention": {"status": "unsupported", "reason": "no verified source entry point; no surrogate implementation"},
    }
