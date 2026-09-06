#!/usr/bin/env python3
"""Build a sanitized manifest for external EVQ-Cosh artifacts.

Run this on the machine that still has checkpoints/data, then copy the JSON
manifest back into the reviewer repo. By default the output avoids absolute
paths and records only labels, path hints, hashes, tensor metadata, and optional
RoPE inv_freq audit summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core_text_phases import audit_rope_checkpoint  # noqa: E402

DEFAULT_GLOBS = [
    "model*.pt",
    "inv_freq.npy",
    "custom_inv_freq.pt",
    "results.json",
    "config.json",
    "metadata.json",
    "train*.pt",
    "val*.pt",
    "validation*.pt",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_entry(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
    else:
        path = raw
        label = Path(raw).name
    if not label:
        raise ValueError(f"empty label in entry: {raw!r}")
    return label, Path(path).expanduser()


def path_hint(path: Path, root: Path, include_paths: bool) -> str:
    if include_paths:
        return str(path)
    try:
        if root.is_dir():
            return str(path.relative_to(root))
    except ValueError:
        pass
    return path.name


def audit_error_summary(exc: Exception, include_paths: bool) -> str:
    if include_paths:
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return "RoPE audit did not find an inv_freq/model artifact in this entry."
    return "RoPE audit failed; rerun locally with --include-paths for full details."


def discover_files(root: Path, recursive: bool, globs: list[str]) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    if recursive:
        return sorted(p for p in root.rglob("*") if p.is_file())
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in globs:
        for p in root.glob(pattern):
            if p.is_file() and p not in seen:
                files.append(p)
                seen.add(p)
    return sorted(files)


def tensor_summary(path: Path, inspect_tensors: bool) -> dict[str, Any] | None:
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        return {"kind": "npy", "shape": list(arr.shape), "dtype": str(arr.dtype)}

    if not inspect_tensors or path.suffix not in {".pt", ".pth"}:
        return None

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return {"kind": "torch_load_error", "error": type(exc).__name__}

    if torch.is_tensor(obj):
        return {
            "kind": "tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "numel": int(obj.numel()),
        }
    if isinstance(obj, dict):
        tensor_keys: dict[str, Any] = {}
        for key, value in obj.items():
            if torch.is_tensor(value):
                tensor_keys[str(key)] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "numel": int(value.numel()),
                }
        return {
            "kind": "dict",
            "num_keys": len(obj),
            "tensor_keys": tensor_keys,
        }
    return {"kind": type(obj).__name__}


def summarize_rope_audit(report: dict[str, Any]) -> dict[str, Any]:
    source = report.get("source", {})
    inv = report.get("actual_inv_freq", {})
    return {
        "source": {
            "kind": source.get("kind"),
            "first_buffer": source.get("first_buffer"),
            "num_buffers": source.get("num_buffers"),
            "mismatched_buffers": source.get("mismatched_buffers", []),
            "max_buffer_mismatch": source.get("max_buffer_mismatch"),
        },
        "rope_base": report.get("rope_base"),
        "tau_requested_or_inferred": report.get("tau_requested_or_inferred"),
        "d_eff": report.get("d_eff"),
        "d_rope_or_drot": report.get("d_rope_or_drot"),
        "d_head": report.get("d_head"),
        "n_freqs": report.get("n_freqs"),
        "actual_inv_freq": {
            "sha256": inv.get("sha256"),
            "min": inv.get("min"),
            "max": inv.get("max"),
            "first_8": inv.get("first_8"),
            "last_8": inv.get("last_8"),
        },
        "closest_reference": report.get("closest_reference"),
        "estimated_tau_grid": report.get("estimated_tau_grid"),
        "classification": report.get("classification"),
    }


def build_entry_manifest(
    label: str,
    root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": label,
        "exists": root.exists(),
        "kind": "directory" if root.is_dir() else "file" if root.is_file() else "missing",
        "path_hint": str(root) if args.include_paths else root.name,
    }
    if not root.exists():
        return entry

    globs = args.glob or DEFAULT_GLOBS
    files = discover_files(root, recursive=args.recursive, globs=globs)
    entry["files"] = []
    for p in files:
        item: dict[str, Any] = {
            "path_hint": path_hint(p, root, args.include_paths),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        ts = tensor_summary(p, inspect_tensors=args.inspect_tensors)
        if ts is not None:
            item["tensor"] = ts
        entry["files"].append(item)

    if args.rope_audit:
        audit_args = argparse.Namespace(
            checkpoint=str(root),
            base=args.base,
            tau=args.tau,
            rope_dim=args.rope_dim,
            d_rope=args.d_rope,
            d_head=args.d_head,
            d_eff=args.d_eff,
            tolerance=args.tolerance,
            estimate_tau_max=args.estimate_tau_max,
            estimate_tau_steps=args.estimate_tau_steps,
        )
        try:
            entry["rope_audit"] = summarize_rope_audit(
                audit_rope_checkpoint.build_report(audit_args)
            )
        except Exception as exc:
            entry["rope_audit"] = {
                "available": False,
                "error_type": type(exc).__name__,
                "error": audit_error_summary(exc, args.include_paths),
            }

    return entry


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    entries = [parse_entry(raw) for raw in args.entry]
    return {
        "schema": "evq_cosh_artifact_manifest.v1",
        "path_policy": (
            "absolute_paths_included" if args.include_paths else "sanitized_path_hints_only"
        ),
        "notes": args.note,
        "entries": [
            build_entry_manifest(label, path, args)
            for label, path in entries
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a sanitized hash/audit manifest for EVQ-Cosh artifacts."
    )
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Artifact entry as LABEL=PATH. May be repeated.",
    )
    parser.add_argument("--output", type=str, default="-", help="Output JSON path, or '-'")
    parser.add_argument("--note", action="append", default=[], help="Freeform manifest note")
    parser.add_argument("--include-paths", action="store_true", help="Include full input paths")
    parser.add_argument("--recursive", action="store_true", help="Hash files recursively")
    parser.add_argument("--glob", action="append", help="Directory glob to include; repeatable")
    parser.add_argument("--inspect-tensors", action="store_true", help="Load .pt tensors for shape/dtype")
    parser.add_argument("--no-rope-audit", dest="rope_audit", action="store_false")
    parser.set_defaults(rope_audit=True)
    parser.add_argument("--base", type=float, default=500000.0)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--rope-dim", type=int, default=None)
    parser.add_argument("--d-rope", type=int, default=None)
    parser.add_argument("--d-head", type=int, default=None)
    parser.add_argument("--d-eff", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--estimate-tau-max", type=float, default=8.0)
    parser.add_argument("--estimate-tau-steps", type=int, default=800)
    args = parser.parse_args()

    manifest = build_manifest(args)
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(text)
    else:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
