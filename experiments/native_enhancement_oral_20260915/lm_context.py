#!/usr/bin/env python3
"""CPU preparation and paired analysis for native full-versus-recent NLL.

Let x contain L+1 document tokens. The full model input is x[:-1] (L
tokens), and the last T tokens x[L+1-T:L+1] are predicted by logits
[L-T:L]. The recent sequence contains H preceding tokens and exactly the
same T targets; its input has H+T-1 tokens and target logits [H-1:H+T-1].
Both position_ids restart at zero. All relative positions among shared
tokens are unchanged. Earlier target tokens are teacher-forced in both.

No model runtime, tokenizer, downloads or GPU code is imported here. Existing
prefix-average NLL output cannot recover this experiment: target-only losses
and a separate recent-context forward pass are required.

CLI: ``python -m experiments.native_enhancement_oral_20260915.lm_context canary``.
Use ``prepare --help`` to describe an existing .npy matrix, and ``analyze
--help`` for four-condition nll_sum/target_count JSONL records. Score rows must
carry pair_id, arm (native/ncp), context (full/recent), target_sha256,
nll_sum and target_count. A future GPU caller must retain the same checkpoint,
tokenizer, precision, causal mask and frozen frequency table across contexts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


CONTRACT = "NATIVE_SAME_TARGET_CONTEXT_NLL_V1"
CONDITIONS = (("native", "full"), ("native", "recent"),
              ("ncp", "full"), ("ncp", "recent"))


def build_context_pair(tokens, *, native_length: int, recent_history: int = 512,
                       target_tokens: int = 256, window_start: int = 0) -> dict:
    """Build two explicit causal inputs without retokenization or padding."""
    for name, value in (("native_length", native_length), ("recent_history", recent_history),
                        ("target_tokens", target_tokens), ("window_start", window_start)):
        if type(value) is not int or value < (0 if name == "window_start" else 1):
            raise ValueError(f"invalid {name}")
    if native_length < recent_history + target_tokens:
        raise ValueError("full input must provide more history than recent input")
    values = np.asarray(tokens)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("tokens must be a one-dimensional integer array")
    stop = window_start + native_length + 1
    if len(values) < stop:
        raise ValueError("document row must contain window_start + native_length + 1 tokens")
    full = np.asarray(values[window_start:stop], dtype=np.int64)
    if np.any(full < 0):
        raise ValueError("negative token ID")
    recent_start = len(full) - target_tokens - recent_history
    output = {}
    for name, span, offset in (("full", full, 0), ("recent", full[recent_start:], recent_start)):
        input_ids = span[:-1].copy()
        # Logit j predicts span[j+1], hence the -1 at both slice boundaries.
        target_start = len(span) - target_tokens
        output[name] = {
            "input_ids": input_ids,
            "position_ids": np.arange(len(input_ids), dtype=np.int64),
            "target_ids": span[-target_tokens:].copy(),
            "loss_positions": np.arange(target_start - 1, len(span) - 1, dtype=np.int64),
            "source_input_start": window_start + offset,
            "source_target_start": stop - target_tokens,
        }
    return output


def target_digest(target_ids) -> str:
    """Small target identity only; does not rehash the entire source asset."""
    return hashlib.sha256(np.asarray(target_ids, dtype="<i8").tobytes()).hexdigest()


def target_nll_from_logits(logits, context: dict) -> dict:
    """CPU reference scoring: gather prediction positions, with no second shift."""
    values = np.asarray(logits, dtype=np.float64)
    targets = context["target_ids"]
    if values.ndim != 2 or values.shape[0] != len(context["input_ids"]):
        raise ValueError("expected [input_tokens, vocabulary] logits")
    if values.shape[1] == 0 or np.any(targets >= values.shape[1]):
        raise ValueError("target ID outside logits vocabulary")
    selected = values[context["loss_positions"]]
    if not np.all(np.isfinite(selected)):
        raise ValueError("nonfinite target logits")
    maximum = selected.max(axis=1)
    log_partition = maximum + np.log(np.exp(selected - maximum[:, None]).sum(axis=1))
    losses = log_partition - selected[np.arange(len(targets)), targets]
    return {"nll_sum": float(losses.sum()), "target_count": len(targets),
            "nll": float(losses.mean())}


def build_manifest(array_path: Path, *, native_length: int, split: str,
                   document_ids: list[str] | None = None,
                   excluded_document_ids: list[str] | None = None,
                   exclusions_verified: bool = False, recent_history: int = 512,
                   target_tokens: int = 256, window_start: int = 0) -> dict:
    """Describe slices; row numbers or fresh slices never establish fresh documents.

    Confirmation requires source document IDs and explicit acknowledgement that
    exclusions cover documents previously used in method design/selection. This
    validates the supplied list, not the completeness of external provenance.
    Repeated IDs are allowed and will be averaged within document in analysis.
    """
    if split not in ("development", "confirmation"):
        raise ValueError("split must be explicitly development or confirmation")
    matrix = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("expected a nonempty two-dimensional .npy token matrix")
    for name, ids in (("document_ids", document_ids), ("excluded_document_ids", excluded_document_ids)):
        if ids is not None and (not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids)):
            raise ValueError(f"{name} must be a JSON list of nonempty source IDs")
    if document_ids is not None and len(document_ids) != len(matrix):
        raise ValueError("one document ID is required per matrix row")
    if split == "confirmation" and (document_ids is None or excluded_document_ids is None or exclusions_verified is not True):
        raise ValueError("confirmation needs source document IDs, explicit exclusion IDs and verified exclusions")
    excluded = set(excluded_document_ids or [])
    if document_ids is not None and set(document_ids) & excluded:
        raise ValueError("selected documents overlap the supplied exclusion list")
    samples = []
    for row, tokens in enumerate(matrix):
        pair = build_context_pair(tokens, native_length=native_length, recent_history=recent_history,
                                  target_tokens=target_tokens, window_start=window_start)
        samples.append({
            "pair_id": f"row:{row}:start:{window_start}", "source_row": row,
            "document_id": document_ids[row] if document_ids is not None else None,
            "source_target_start": pair["full"]["source_target_start"],
            "target_sha256": target_digest(pair["full"]["target_ids"]),
        })
    return {
        "contract": CONTRACT, "status": "CPU_PREPARED_NO_MODEL_RESULTS",
        "array_path": str(Path(array_path).resolve()), "array_shape": list(matrix.shape),
        "native_length": native_length, "recent_history": recent_history,
        "target_tokens": target_tokens, "window_start": window_start, "split": split,
        "rows": len(samples), "documents": len(set(document_ids)) if document_ids is not None else None,
        "document_identity": "source_ids_provided" if document_ids is not None else "unknown",
        "excluded_document_ids": sorted(excluded), "exclusions_verified": exclusions_verified,
        "exclusion_scope": "All source documents used in method design or selection; supplied by caller.",
        "fresh_slice_is_independent_document": False,
        "samples": samples,
        "future_gpu_contract": {
            "conditions": [list(c) for c in CONDITIONS], "input_convention": "span[:-1]",
            "loss_convention": "Gather loss_positions and target_ids; do not apply another shift.",
            "position_ids": "Each input starts at zero; shared-token relative distances are preserved.",
            "scoring": "Teacher-forced final target_tokens only; no generation, padding or truncation.",
            "reuse_limit": "Existing prefix-average NLL cannot recover target-only full/recent losses.",
            "hold_fixed": ["checkpoint", "tokenizer", "dtype", "attention backend", "gain", "table within arm"],
            "required_score_fields": ["pair_id", "arm", "context", "target_sha256", "nll_sum", "target_count"],
        },
    }


def analyze_four_conditions(manifest: dict, records: list[dict], *,
                            draws: int = 20000, seed: int = 20260915) -> dict:
    """Use exact document-mean point estimates and paired document bootstrap.

    Delta_full = NCP_full - Native_full (negative is better).
    Use_arm = NLL_arm,recent - NLL_arm,full.
    Delta_use = Use_NCP - Use_Native = Delta_recent - Delta_full (positive
    indicates a larger benefit from the available earlier context).
    Delta_use alone is insufficient: it can rise from degraded recent NLL.
    """
    if manifest.get("contract") != CONTRACT:
        raise ValueError("manifest contract mismatch")
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be positive")
    samples = {sample["pair_id"]: sample for sample in manifest["samples"]}
    if not samples or len(samples) != len(manifest["samples"]):
        raise ValueError("empty or duplicate manifest pair IDs")
    by_pair = defaultdict(dict)
    for row in records:
        pair_id = row["pair_id"]
        condition = (row["arm"], row["context"])
        if pair_id not in samples or condition not in CONDITIONS or condition in by_pair[pair_id]:
            raise ValueError("extra pair, invalid condition or duplicate score")
        if row.get("target_sha256") != samples[pair_id]["target_sha256"]:
            raise ValueError("target identity mismatch")
        if type(row["target_count"]) is not int or row["target_count"] != manifest["target_tokens"]:
            raise ValueError("target count mismatch")
        total = float(row["nll_sum"])
        if not np.isfinite(total) or total < 0:
            raise ValueError("NLL sum must be finite and nonnegative")
        by_pair[pair_id][condition] = total / row["target_count"]
    if set(by_pair) != set(samples) or any(set(values) != set(CONDITIONS) for values in by_pair.values()):
        raise ValueError("every manifest pair needs all four conditions")
    identities_known = all(s["document_id"] is not None for s in samples.values())
    groups = defaultdict(list)
    for pair_id, sample in samples.items():
        group = sample["document_id"] if identities_known else pair_id
        groups[group].append([by_pair[pair_id][condition] for condition in CONDITIONS])
    # Document-equal estimand: repeated slices first average within a document.
    values = np.asarray([np.mean(rows, axis=0) for rows in groups.values()])
    nf, nr, cf, cr = values.T
    metrics = {"delta_full": cf - nf, "delta_recent": cr - nr,
               "use_native": nr - nf, "use_ncp": cr - cf,
               "delta_use": (cr - cf) - (nr - nf)}
    metrics.update({f"{arm}_{context}": values[:, i] for i, (arm, context) in enumerate(CONDITIONS)})
    summaries = {name: {"estimate": float(vector.mean()), "ci95": None} for name, vector in metrics.items()}
    if identities_known and len(groups) > 1:
        rng = np.random.default_rng(seed)
        # Bounded batches avoid allocating draws x documents for large panels.
        sampled = defaultdict(list)
        for start in range(0, draws, 256):
            indices = rng.integers(len(groups), size=(min(256, draws - start), len(groups)))
            for name, vector in metrics.items():
                sampled[name].append(vector[indices].mean(axis=1))
        for name, batches in sampled.items():
            summaries[name]["ci95"] = np.quantile(np.concatenate(batches), [0.025, 0.975]).tolist()
    return {
        "contract": CONTRACT, "status": "SCORE_ANALYSIS", "split": manifest["split"],
        "pairs": len(samples), "documents": len(groups) if identities_known else None,
        "estimand": "document-equal mean" if identities_known else "descriptive row mean; document IDs unavailable",
        "ci_method": "paired document percentile bootstrap" if identities_known and len(groups) > 1 else "unavailable",
        "draws": draws, "seed": seed, "metrics": summaries,
        "interpretation": "Negative delta_full improves native NLL; positive delta_use increases context benefit. Inspect delta_recent too.",
        "scope": "Teacher-forced same-target NLL; no claim of generated-task gain or new independent documents.",
    }


def cpu_canary() -> dict:
    """A successor-token oracle exposes a one-token shift without a real model."""
    pair = build_context_pair(np.arange(20), native_length=16, recent_history=4, target_tokens=3)
    scores = {}
    for name, context in pair.items():
        logits = np.zeros((len(context["input_ids"]), 20))
        logits[np.arange(len(logits)), context["input_ids"] + 1] = 9.0
        scores[name] = target_nll_from_logits(logits, context)
    shared = len(pair["recent"]["input_ids"])
    full_pos = pair["full"]["position_ids"][-shared:]
    recent_pos = pair["recent"]["position_ids"]
    assert np.array_equal(pair["full"]["target_ids"], pair["recent"]["target_ids"])
    assert np.array_equal(full_pos[:, None] - full_pos, recent_pos[:, None] - recent_pos)
    assert abs(scores["full"]["nll"] - scores["recent"]["nll"]) < 1e-12
    return {"status": "CPU_CANARY_PASS", "model_loaded": False, "gpu_used": False,
            "native_length": 16, "target_ids": pair["full"]["target_ids"].tolist(),
            "loss_positions": {name: c["loss_positions"].tolist() for name, c in pair.items()},
            "same_targets": True, "shared_relative_positions_equal": True, "scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("canary", help="Run a CPU-only off-by-one oracle check")
    prepare = commands.add_parser("prepare", help="Describe an existing token matrix; no tokenization or inference")
    prepare.add_argument("--tokens", type=Path, required=True)
    prepare.add_argument("--native-length", type=int, required=True)
    prepare.add_argument("--split", choices=("development", "confirmation"), required=True)
    prepare.add_argument("--document-ids", type=Path, help="JSON list of source document IDs, one per matrix row")
    prepare.add_argument("--excluded-document-ids", type=Path, help="JSON list of all previously used source document IDs")
    prepare.add_argument("--exclusions-verified", action="store_true")
    prepare.add_argument("--recent-history", type=int, default=512)
    prepare.add_argument("--target-tokens", type=int, default=256)
    prepare.add_argument("--window-start", type=int, default=0)
    prepare.add_argument("--out", type=Path, required=True)
    analyze = commands.add_parser("analyze", help="Analyze paired four-condition score JSONL")
    analyze.add_argument("--manifest", type=Path, required=True)
    analyze.add_argument("--scores", type=Path, required=True)
    analyze.add_argument("--draws", type=int, default=20000)
    analyze.add_argument("--seed", type=int, default=20260915)
    analyze.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "canary":
        result = cpu_canary()
    elif args.command == "prepare":
        result = build_manifest(args.tokens, native_length=args.native_length, split=args.split,
                                document_ids=json.loads(args.document_ids.read_text()) if args.document_ids else None,
                                excluded_document_ids=json.loads(args.excluded_document_ids.read_text()) if args.excluded_document_ids else None,
                                exclusions_verified=args.exclusions_verified, recent_history=args.recent_history,
                                target_tokens=args.target_tokens, window_start=args.window_start)
    else:
        result = analyze_four_conditions(json.loads(args.manifest.read_text()),
                                         [json.loads(line) for line in args.scores.read_text().splitlines() if line.strip()],
                                         draws=args.draws, seed=args.seed)
    if args.command != "canary":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
