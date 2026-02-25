#!/usr/bin/env python3
"""Build a reproducible registry for base models and LoRA artifacts.

This script is intentionally dependency-free so it can run on both local and
server environments with the project Python interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def to_abs(path_text: str, repo_root: Path) -> Path:
    p = Path(path_text).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_get(d: Dict, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def read_json(path: Path) -> Optional[Dict]:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            vals: List[str] = []
            for c in columns:
                v = row.get(c, "")
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.8g}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class RunRecord:
    run_id: str
    run_dir: Path
    output_root: Path
    status: str
    model_name: str
    method: str
    seed: str
    timestamp: str
    base_model_path: str
    base_model_exists: bool
    adapter_layout: str
    adapter_resolved_path: Path
    adapter_path: Path
    adapter_exists: bool
    root_adapter_path: Path
    root_adapter_exists: bool
    final_lora_path: Path
    final_lora_exists: bool
    summary_path: Path
    summary_exists: bool
    inv_path: Path
    inv_exists: bool
    inv_sha256: str
    max_steps: Optional[int]
    per_device_batch: Optional[int]
    grad_accum: Optional[int]
    gradient_checkpointing: Optional[bool]
    learning_rate: Optional[float]
    max_seq_len: Optional[int]
    anchor_factor: Optional[float]
    slope_raw: Optional[float]
    center_ratio: Optional[float]
    train_loss: Optional[float]
    train_hours: Optional[float]
    train_runtime: Optional[float]
    train_steps_per_second: Optional[float]
    peak_cuda_allocated_gb: Optional[float]

    def to_row(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "run_dir": str(self.run_dir),
            "output_root": str(self.output_root),
            "model_name": self.model_name,
            "method": self.method,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "base_model_path": self.base_model_path,
            "base_model_exists": int(self.base_model_exists),
            "adapter_layout": self.adapter_layout,
            "adapter_resolved_path": str(self.adapter_resolved_path),
            "adapter_path": str(self.adapter_path),
            "adapter_exists": int(self.adapter_exists),
            "root_adapter_path": str(self.root_adapter_path),
            "root_adapter_exists": int(self.root_adapter_exists),
            "final_lora_path": str(self.final_lora_path),
            "final_lora_exists": int(self.final_lora_exists),
            "summary_path": str(self.summary_path),
            "summary_exists": int(self.summary_exists),
            "inv_path": str(self.inv_path),
            "inv_exists": int(self.inv_exists),
            "inv_sha256": self.inv_sha256,
            "max_steps": self.max_steps,
            "per_device_batch": self.per_device_batch,
            "grad_accum": self.grad_accum,
            "gradient_checkpointing": "1" if self.gradient_checkpointing else ("0" if self.gradient_checkpointing is not None else ""),
            "learning_rate": self.learning_rate,
            "max_seq_len": self.max_seq_len,
            "anchor_factor": self.anchor_factor,
            "slope_raw": self.slope_raw,
            "center_ratio": self.center_ratio,
            "train_loss": self.train_loss,
            "train_hours": self.train_hours,
            "train_runtime": self.train_runtime,
            "train_steps_per_second": self.train_steps_per_second,
            "peak_cuda_allocated_gb": self.peak_cuda_allocated_gb,
        }


def infer_name_fields(run_name: str, summary: Optional[Dict]) -> Tuple[str, str, str]:
    method = str(safe_get(summary or {}, "method", default="")).strip().lower()
    seed = str(safe_get(summary or {}, "seed", default="")).strip()

    if not seed:
        parts = [x for x in run_name.split("_") if x]
        if parts and parts[-1].isdigit():
            seed = parts[-1]

    suffix_seed = f"_{seed}" if seed else ""
    if not method:
        if run_name.endswith(f"_anchored_sigmoid{suffix_seed}"):
            method = "anchored_sigmoid"
        elif run_name.endswith(f"_baseline{suffix_seed}"):
            method = "baseline"

    model_name = run_name
    if seed and run_name.endswith("_" + seed):
        model_name = run_name[: -(len(seed) + 1)]
    if method and model_name.endswith("_" + method):
        model_name = model_name[: -(len(method) + 1)]
    return model_name, method, seed


def gather_base_models(cache_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not cache_dir.exists():
        return rows

    seen: set[str] = set()
    for config in cache_dir.rglob("config.json"):
        model_dir = config.parent
        if str(model_dir) in seen:
            continue
        seen.add(str(model_dir))
        rel = model_dir.relative_to(cache_dir)
        repo_hint = str(rel).replace(os.sep, "/")
        rows.append(
            {
                "model_dir": str(model_dir),
                "repo_hint": repo_hint,
                "config_json": str(config),
                "exists": 1,
            }
        )

    rows.sort(key=lambda x: str(x["model_dir"]))
    return rows


def is_adapter_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "adapter_config.json").exists():
        return False
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()


def gather_run_records(run_roots: Sequence[Path]) -> List[RunRecord]:
    records: List[RunRecord] = []

    for root in run_roots:
        if not root.exists():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue

            summary_path = run_dir / "artifacts" / "summary.json"
            summary = read_json(summary_path)
            final_lora_path = run_dir / "final_lora"
            root_adapter_path = run_dir
            inv_path = run_dir / "artifacts" / "custom_inv_freq.pt"

            final_lora_exists = is_adapter_dir(final_lora_path)
            root_adapter_exists = is_adapter_dir(root_adapter_path)
            if final_lora_exists and root_adapter_exists:
                adapter_layout = "dual_root_and_final_lora"
                adapter_resolved_path = final_lora_path
            elif final_lora_exists:
                adapter_layout = "final_lora_only"
                adapter_resolved_path = final_lora_path
            elif root_adapter_exists:
                adapter_layout = "root_adapter_only"
                adapter_resolved_path = root_adapter_path
            else:
                adapter_layout = "none"
                adapter_resolved_path = root_adapter_path
            adapter_exists = final_lora_exists or root_adapter_exists
            summary_exists = summary is not None
            inv_exists = inv_path.exists()

            if not (adapter_exists or summary_exists or inv_exists):
                continue

            if adapter_exists and summary_exists:
                status = "ready"
            elif summary_exists:
                status = "summary_only"
            elif adapter_exists:
                status = "adapter_only"
            else:
                status = "partial"

            model_name, method, seed = infer_name_fields(run_dir.name, summary)
            hyper = safe_get(summary or {}, "hyperparams", default={}) or {}
            rope = safe_get(summary or {}, "rope", "schedule_params", default={}) or {}
            train = safe_get(summary or {}, "train", default={}) or {}

            inv_sha = str(safe_get(summary or {}, "rope", "inv_sha256", default="") or "")
            if not inv_sha and inv_exists:
                inv_sha = sha256_file(inv_path)
            base_model_path = str(safe_get(summary or {}, "base_model_path_resolved", default="") or "")
            base_model_exists = bool(base_model_path) and Path(base_model_path).exists()

            record = RunRecord(
                run_id=run_dir.name,
                run_dir=run_dir.resolve(),
                output_root=root.resolve(),
                status=status,
                model_name=model_name,
                method=method,
                seed=seed,
                timestamp=str(safe_get(summary or {}, "timestamp", default="") or ""),
                base_model_path=base_model_path,
                base_model_exists=base_model_exists,
                adapter_layout=adapter_layout,
                adapter_resolved_path=adapter_resolved_path.resolve(),
                adapter_path=adapter_resolved_path.resolve(),
                adapter_exists=adapter_exists,
                root_adapter_path=root_adapter_path.resolve(),
                root_adapter_exists=root_adapter_exists,
                final_lora_path=final_lora_path.resolve(),
                final_lora_exists=final_lora_exists,
                summary_path=summary_path.resolve(),
                summary_exists=summary_exists,
                inv_path=inv_path.resolve(),
                inv_exists=inv_exists,
                inv_sha256=inv_sha,
                max_steps=safe_get(hyper, "max_steps", default=None),
                per_device_batch=safe_get(hyper, "per_device_train_batch_size", default=None),
                grad_accum=safe_get(hyper, "gradient_accumulation_steps", default=None),
                gradient_checkpointing=safe_get(hyper, "gradient_checkpointing", default=None),
                learning_rate=safe_get(hyper, "learning_rate", default=None),
                max_seq_len=safe_get(hyper, "max_seq_len", default=None),
                anchor_factor=safe_get(rope, "anchor_factor_effective", default=safe_get(rope, "anchor_factor_requested", default=None)),
                slope_raw=safe_get(rope, "slope_raw", default=None),
                center_ratio=safe_get(rope, "center_ratio", default=None),
                train_loss=safe_get(train, "train_loss", default=None),
                train_hours=safe_get(train, "train_hours", default=None),
                train_runtime=safe_get(train, "train_runtime", default=None),
                train_steps_per_second=safe_get(train, "train_steps_per_second", default=None),
                peak_cuda_allocated_gb=safe_get(train, "peak_cuda_allocated_gb", default=None),
            )
            records.append(record)

    records.sort(key=lambda x: (x.status != "ready", x.model_name, x.method, x.seed, x.run_id))
    return records


def pick_latest(records: Sequence[RunRecord]) -> List[RunRecord]:
    buckets: Dict[Tuple[str, str], RunRecord] = {}
    for rec in records:
        if rec.status != "ready":
            continue
        key = (rec.model_name, rec.method)
        old = buckets.get(key)
        if old is None:
            buckets[key] = rec
            continue
        old_key = (old.timestamp, old.run_dir.stat().st_mtime)
        new_key = (rec.timestamp, rec.run_dir.stat().st_mtime)
        if new_key > old_key:
            buckets[key] = rec
    out = list(buckets.values())
    out.sort(key=lambda x: (x.model_name, x.method, x.seed, x.run_id))
    return out


def write_readme(out_dir: Path, cache_dir: Path, run_roots: Sequence[Path], total: int, ready: int, partial: int) -> None:
    run_roots_text = "\n".join(f"- `{p}`" for p in run_roots)
    text = f"""# Model Registry

This folder is auto-generated by `scripts/build_model_registry.py`.

## Scope

- Base model cache: `{cache_dir}`
- Scanned run roots:
{run_roots_text}

## Files

- `base_models.tsv`: discovered base model directories (`config.json` present).
- `lora_runs.tsv`: all discovered LoRA runs (ready + partial).
- `lora_weights.tsv`: ready-to-eval runs (`summary.json` + adapter present).
  - Adapter can be either `<run_dir>/final_lora` or root-level PEFT files.
- `lora_partial.tsv`: incomplete runs (keep for debugging/recovery).
- `latest_by_method.tsv`: latest ready run per `(model_name, method)`.
- `registry_summary.json`: counts and generation metadata.

## Fast Load Recipe (base + LoRA)

1. Pick a row from `latest_by_method.tsv` (or `lora_weights.tsv`).
2. Run evaluation with explicit adapter/inv path.

Example:

```bash
python scripts/run_eval.py \\
  --exp cross_model_eval \\
  --model <BASE_MODEL_PATH> \\
  --method anchored_sigmoid \\
  --ctx 16384 \\
  --seed 42 \\
  --adapter_override <ADAPTER_RESOLVED_PATH> \\
  --custom_inv_freq_path <RUN_DIR>/artifacts/custom_inv_freq.pt \\
  --suite ppl,longbench_full,needle \\
  --longbench_score_scale pct
```

## Current Snapshot

- Total runs: {total}
- Ready runs: {ready}
- Partial runs: {partial}
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build base/LoRA registry tables.")
    ap.add_argument("--repo_root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--model_cache_dir", type=str, default="/root/autodl-tmp/dfrope/ms_models")
    ap.add_argument(
        "--run_roots",
        type=str,
        default="artifacts/cross_model,artifacts/cross_model_fast_tuned,artifacts/cross_model_fast_tuned_b1_gc",
        help="Comma-separated run root directories to scan.",
    )
    ap.add_argument("--output_dir", type=str, default="artifacts/model_registry")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = to_abs(args.repo_root, Path.cwd())
    cache_dir = to_abs(args.model_cache_dir, repo_root)
    run_roots = [to_abs(p, repo_root) for p in parse_csv(args.run_roots)]
    out_dir = to_abs(args.output_dir, repo_root)

    base_rows = gather_base_models(cache_dir)
    records = gather_run_records(run_roots)
    ready_rows = [r.to_row() for r in records if r.status == "ready"]
    partial_rows = [r.to_row() for r in records if r.status != "ready"]
    latest_rows = [r.to_row() for r in pick_latest(records)]
    all_rows = [r.to_row() for r in records]

    base_cols = ["model_dir", "repo_hint", "config_json", "exists"]
    run_cols = [
        "run_id",
        "status",
        "run_dir",
        "output_root",
        "model_name",
        "method",
        "seed",
        "timestamp",
        "base_model_path",
        "base_model_exists",
        "adapter_layout",
        "adapter_resolved_path",
        "adapter_path",
        "adapter_exists",
        "root_adapter_path",
        "root_adapter_exists",
        "final_lora_path",
        "final_lora_exists",
        "summary_path",
        "summary_exists",
        "inv_path",
        "inv_exists",
        "inv_sha256",
        "max_steps",
        "per_device_batch",
        "grad_accum",
        "gradient_checkpointing",
        "learning_rate",
        "max_seq_len",
        "anchor_factor",
        "slope_raw",
        "center_ratio",
        "train_loss",
        "train_hours",
        "train_runtime",
        "train_steps_per_second",
        "peak_cuda_allocated_gb",
    ]

    write_tsv(out_dir / "base_models.tsv", base_rows, base_cols)
    write_tsv(out_dir / "lora_runs.tsv", all_rows, run_cols)
    write_tsv(out_dir / "lora_weights.tsv", ready_rows, run_cols)
    write_tsv(out_dir / "lora_partial.tsv", partial_rows, run_cols)
    write_tsv(out_dir / "latest_by_method.tsv", latest_rows, run_cols)

    summary = {
        "repo_root": str(repo_root),
        "model_cache_dir": str(cache_dir),
        "run_roots": [str(p) for p in run_roots],
        "counts": {
            "base_models": len(base_rows),
            "runs_total": len(all_rows),
            "runs_ready": len(ready_rows),
            "runs_partial": len(partial_rows),
            "latest_rows": len(latest_rows),
        },
    }
    write_json(out_dir / "registry_summary.json", summary)
    write_readme(
        out_dir=out_dir,
        cache_dir=cache_dir,
        run_roots=run_roots,
        total=len(all_rows),
        ready=len(ready_rows),
        partial=len(partial_rows),
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
