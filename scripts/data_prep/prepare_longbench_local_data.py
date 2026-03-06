#!/usr/bin/env python3
"""
Download and freeze LongBench subsets into local jsonl files.

This makes downstream eval deterministic and offline-friendly.
Supports fallback data source selection for environments without HuggingFace access:
- `hf`: HuggingFace datasets
- `modelscope`/`dashscope`: Alibaba ecosystem mirror (via ModelScope SDK)
- `auto`: try HF first, then fallback to ModelScope
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def longbench_cfg_candidates(task: str) -> List[str]:
    base = task.strip()
    candidates: List[str] = []
    for cfg in (base, base.lower(), f"{base}_e", f"{base.lower()}_e"):
        if cfg not in candidates:
            candidates.append(cfg)
    return candidates


def _load_hf(task: str, split: str) -> Tuple[object, str, str]:
    errors: List[str] = []
    for cfg in longbench_cfg_candidates(task):
        try:
            ds = load_dataset("THUDM/LongBench", cfg, split=split, trust_remote_code=True)
            return ds, cfg, "hf:THUDM/LongBench"
        except Exception as e:
            errors.append(f"cfg={cfg}: {type(e).__name__}: {e}")
    raise RuntimeError(f"[hf] Cannot load LongBench task={task}\n" + "\n".join(errors))


def _extract_split_obj(raw_obj: object, split: str) -> Optional[object]:
    if raw_obj is None:
        return None
    if isinstance(raw_obj, dict):
        if split in raw_obj:
            return raw_obj[split]
        if "test" in raw_obj:
            return raw_obj["test"]
        if "validation" in raw_obj:
            return raw_obj["validation"]
    if hasattr(raw_obj, split):
        return getattr(raw_obj, split)
    return raw_obj


def _as_indexable_dataset(raw_obj: object, split: str) -> object:
    obj = _extract_split_obj(raw_obj, split)
    if obj is None:
        raise RuntimeError("Dataset object is empty after split extraction.")

    # HF Dataset and most wrappers satisfy this.
    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        return obj

    # ModelScope dataset wrappers often provide conversion helpers.
    for method in ("to_hf_dataset", "to_dataset"):
        if hasattr(obj, method):
            converted = getattr(obj, method)()
            converted = _extract_split_obj(converted, split)
            if converted is not None and hasattr(converted, "__len__") and hasattr(converted, "__getitem__"):
                return converted
    raise RuntimeError(f"Unsupported dataset object type: {type(obj)}")


def _load_modelscope(task: str, split: str) -> Tuple[object, str, str]:
    try:
        from modelscope.msdatasets import MsDataset
    except Exception as e:
        raise RuntimeError(
            "[modelscope] Missing dependency. Install with: pip install modelscope\n"
            f"Import error: {type(e).__name__}: {e}"
        ) from e

    errors: List[str] = []
    dataset_name_candidates = [
        "THUDM/LongBench",
        "AI-ModelScope/LongBench",
        "LongBench",
    ]
    for cfg in longbench_cfg_candidates(task):
        for dname in dataset_name_candidates:
            for kw in (
                {"dataset_name": dname, "subset_name": cfg, "split": split},
                {"dataset_name": dname, "subset_name": cfg},
            ):
                try:
                    raw = MsDataset.load(**kw)
                    ds = _as_indexable_dataset(raw, split=split)
                    return ds, cfg, f"modelscope:{dname}"
                except Exception as e:
                    errors.append(f"dataset={dname}, cfg={cfg}, kw={kw}: {type(e).__name__}: {e}")
    raise RuntimeError(f"[modelscope] Cannot load LongBench task={task}\n" + "\n".join(errors))


def load_longbench_task(task: str, split: str, source: str) -> Tuple[object, str, str]:
    source = source.strip().lower()
    if source not in {"auto", "hf", "modelscope", "dashscope"}:
        raise ValueError(f"Unsupported source: {source}")

    errs: List[str] = []
    if source in {"auto", "hf"}:
        try:
            return _load_hf(task=task, split=split)
        except Exception as e:
            errs.append(str(e))

    # DashScope is treated as ModelScope mirror channel for dataset acquisition.
    if source in {"auto", "modelscope", "dashscope"}:
        try:
            return _load_modelscope(task=task, split=split)
        except Exception as e:
            errs.append(str(e))

    raise RuntimeError("\n\n".join(errs))


def choose_context(sample: Dict) -> str:
    for k in ("context", "input", "article", "document", "passage"):
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare local LongBench jsonl data.")
    ap.add_argument(
        "--tasks",
        type=str,
        default="qasper,hotpotqa,2wikimqa,multi_news,gov_report,narrativeqa",
    )
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_samples_per_task", type=int, default=0, help="<=0 means all")
    ap.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_datasets/LongBench/data",
    )
    ap.add_argument(
        "--build_eval_corpus_path",
        type=str,
        default="",
        help="Optional path to write a concatenated text corpus from task contexts.",
    )
    ap.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "hf", "modelscope", "dashscope"],
        help="Download source backend when local jsonl is absent.",
    )
    ap.add_argument(
        "--force_refresh",
        action="store_true",
        help="Ignore existing local jsonl and rebuild from source.",
    )
    args = ap.parse_args()

    tasks = parse_csv(args.tasks)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "meta": {
            "timestamp": now(),
            "tasks": tasks,
            "split": args.split,
            "max_samples_per_task": int(args.max_samples_per_task),
            "output_dir": str(out_dir),
            "source": args.source,
            "force_refresh": bool(args.force_refresh),
        },
        "tasks": {},
    }

    corpus_chunks: List[str] = []

    for task in tasks:
        out_path = out_dir / f"{task}.jsonl"
        # Offline-first path: reuse existing local jsonl when available.
        if out_path.exists() and not args.force_refresh:
            n_all = 0
            n_keep = 0
            with out_path.open("r", encoding="utf-8") as f:
                for line in f:
                    n_all += 1
                    if args.max_samples_per_task > 0 and n_keep >= int(args.max_samples_per_task):
                        continue
                    n_keep += 1
                    try:
                        row = json.loads(line)
                    except Exception:
                        row = {}
                    ctx = choose_context(row)
                    if ctx:
                        corpus_chunks.append(ctx)

            manifest["tasks"][task] = {
                "config_used": "local_jsonl",
                "n_all": int(n_all),
                "n_saved": int(n_keep),
                "jsonl_path": str(out_path),
            }
            print(f"[ok] {task}: local_jsonl, saved={n_keep}/{n_all} -> {out_path}", flush=True)
            continue

        ds, cfg, backend = load_longbench_task(task=task, split=args.split, source=args.source)
        n_all = len(ds)
        n_keep = n_all if args.max_samples_per_task <= 0 else min(n_all, int(args.max_samples_per_task))

        with out_path.open("w", encoding="utf-8") as f:
            for i in range(n_keep):
                row = dict(ds[i])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                ctx = choose_context(row)
                if ctx:
                    corpus_chunks.append(ctx)

        manifest["tasks"][task] = {
            "config_used": cfg,
            "backend": backend,
            "n_all": int(n_all),
            "n_saved": int(n_keep),
            "jsonl_path": str(out_path),
        }
        print(f"[ok] {task}: backend={backend}, cfg={cfg}, saved={n_keep}/{n_all} -> {out_path}", flush=True)

    if args.build_eval_corpus_path:
        corpus_path = Path(args.build_eval_corpus_path)
        if not corpus_path.is_absolute():
            corpus_path = (out_dir / corpus_path).resolve()
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n\n".join(corpus_chunks)
        corpus_path.write_text(text, encoding="utf-8")
        manifest["meta"]["eval_corpus_path"] = str(corpus_path)
        manifest["meta"]["eval_corpus_chars"] = int(len(text))
        print(f"[ok] eval corpus: {corpus_path} chars={len(text)}", flush=True)

    manifest_path = out_dir / "manifest_longbench_local.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
