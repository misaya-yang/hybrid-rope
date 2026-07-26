#!/usr/bin/env python3
"""Write a compact fact-only report from available 4K conversion artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def natural_lines(name: str, result: dict[str, Any] | None) -> list[str]:
    if not result or not result.get("natural_nll"):
        return []
    lines = [f"### {name}", "", "| length | mean NLL | tail NLL |", "| ---: | ---: | ---: |"]
    for length, metrics in result["natural_nll"].items():
        lines.append(
            f"| {length.removeprefix('L')} | "
            f"{fmt(metrics['mean_nll'])} | {fmt(metrics['tail_mean_nll'])} |"
        )
    return lines + [""]


def binding_lines(name: str, metrics: dict[str, Any] | None) -> list[str]:
    if not metrics:
        return []
    keys = (
        "full_vocab_exact",
        "candidate_exact",
        "mean_nll",
        "median_rank",
        "mean_source_deletion_nll_gap",
        "swap_follow_positive_fraction",
        "mean_swap_follow_score",
    )
    lines = [f"### {name}", ""]
    for key in keys:
        if key in metrics:
            lines.append(f"- `{key}`: {fmt(metrics[key])}")
    return lines + [""]


def causal_lines(name: str, result: dict[str, Any] | None) -> list[str]:
    if not result:
        return []
    lines = [f"### {name}", ""]
    for set_name, entry in result.get("results", {}).items():
        overall = entry["summary"]["overall"]
        lines.append(
            f"- `{set_name}`: exact={fmt(overall['next_token_exact_match'])}, "
            f"NLL={fmt(overall['mean_answer_nll'])}, "
            f"source-gap={fmt(overall['mean_source_deletion_nll_gap'])}, "
            f"swap+={fmt(overall['swap_follow_positive_fraction'])}"
        )
    return lines + [""]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()

    base = load(root / "base_evq" / "results.json")
    stage_a = load(root / "stage_a" / "results.json")
    stage_b1 = load(root / "stage_b1" / "results.json")
    stage_b2 = load(root / "stage_b2" / "results.json")
    base_causal = load(root / "causal_base_canary" / "results.json")
    post_causal = load(root / "causal_stage_b2_canary" / "results.json")
    gates = {
        name: load(root / "gates" / f"{name}.json")
        for name in ("stage_a", "stage_b1", "stage_b2", "causal_16k")
    }
    ruler = load(root / "ruler_stage_b2" / "results.json")

    lines = [
        "# OLMo-2 step-30K 4K-only EVQ conversion run",
        "",
        "## Scientific contract",
        "",
        "- One sequential EVQ-LoRA arm; no hyperparameter sweep.",
        "- Every optimizer sequence is exactly 4,096 tokens.",
        "- Attention operator, softmax, RoPE base, and inference length rule are unchanged.",
        "- Hidden-state distillation and native-RoPE training controls are excluded.",
        "- Failure of any admission gate stops all later GPU work.",
        "",
        "## Confirmed execution state",
        "",
    ]
    for name, result in (
        ("EVQ-injected base", base),
        ("Stage A", stage_a),
        ("Stage B1", stage_b1),
        ("Stage B2", stage_b2),
        ("16K base canary", base_causal),
        ("16K adapted canary", post_causal),
        ("RULER", ruler),
    ):
        status = "not run" if result is None else result.get("status", "present")
        lines.append(f"- {name}: `{status}`")
    lines += ["", "## Admission gates", ""]
    for name, gate in gates.items():
        status = "not reached" if gate is None else gate.get("status", "unknown")
        lines.append(f"- `{name}`: **{status}**")
        if gate:
            for check, passed in gate.get("checks", {}).items():
                lines.append(f"  - `{check}`: {bool(passed)}")
    lines += [""]

    lines += natural_lines("EVQ-injected base natural text", base)
    lines += natural_lines("Stage A natural text", stage_a)
    lines += binding_lines(
        "Stage B1 calibration",
        None if stage_b1 is None else stage_b1.get("binding_validation"),
    )
    lines += binding_lines(
        "Stage B2 held-out validation",
        None if stage_b2 is None else stage_b2.get("binding_validation"),
    )
    lines += binding_lines(
        "Stage B2 untouched final test",
        None if stage_b2 is None else stage_b2.get("binding_final_test"),
    )
    lines += causal_lines("16K EVQ-injected base canary", base_causal)
    lines += causal_lines("16K Stage-B2 canary", post_causal)

    if ruler:
        lines += [
            "### RULER single-needle",
            "",
            f"- cells: `{json.dumps(ruler['results']['cells'], sort_keys=True)}`",
            f"- macro average: {fmt(ruler['results']['macro_average'])}",
            "",
        ]
    lines += [
        "## Interpretation boundary",
        "",
        "This file reports completed artifacts only. Natural-text NLL, teacher-forced "
        "source-causal metrics, and autoregressive RULER accuracy are separate evidence "
        "tiers. A missing or stopped stage is not treated as a positive result.",
        "",
    ]
    atomic_text(args.output.resolve(), "\n".join(lines))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
