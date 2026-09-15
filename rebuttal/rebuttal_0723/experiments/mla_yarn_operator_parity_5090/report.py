#!/usr/bin/env python3
"""Render a terminal PASS/STOP receipt into an auditable Markdown report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rebuttal.rebuttal_0723.experiments.mla_scarcity_5090.protocol import (
    SPEC as BASE_SPEC,
)
from rebuttal.rebuttal_0723.experiments.mla_scarcity_5090.run_experiment import (
    code_fingerprint as base_code_fingerprint,
)
from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090.prepare import (
    sha256_file,
)
from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090.protocol import (
    OPERATORS,
    PRIMARY_LENGTHS,
    SEEDS,
    SPEC,
)
from rebuttal.rebuttal_0723.experiments.mla_yarn_operator_parity_5090.run_experiment import (
    code_fingerprint,
    validate_gate,
)


def _fmt(value: float) -> str:
    return f"{float(value):+.4f}"


def _plain(value: float) -> str:
    return f"{float(value):.4f}"


def _ci(record: dict[str, Any]) -> str:
    low, high = record["ci95"]
    return (
        f"{_fmt(record['mean'])} "
        f"[{_fmt(low)}, {_fmt(high)}]"
    )


def _criteria_table(criteria: dict[str, Any]) -> list[str]:
    lines = ["| Criterion | Pass |", "| --- | :---: |"]
    for name, passed in criteria.items():
        lines.append(
            f"| `{name}` | {'yes' if bool(passed) else '**no**'} |"
        )
    return lines


def _stop_report(
    gate: dict[str, Any],
    ready: dict[str, Any],
    provenance: dict[str, str],
) -> str:
    rows = [
        row for row in gate["rows"] if row["stage"] == "300m"
    ]
    lines = [
        "# MLA shared-index YaRN operator-parity experiment",
        "",
        "Status: **STOP after the registered seed-42 selection gate**.",
        "",
        "The unseen test split was not read and seeds 43/88 were not "
        "authorized. This is a valid negative terminal result.",
        "",
        "## Gate criteria",
        "",
        *_criteria_table(gate["criteria"]),
        "",
        "## Seed-42 selection result at 300M tokens",
        "",
        "| Length | K=8 shared EVQ advantage | K=8 operator interaction "
        "| K=32 shared EVQ advantage | Scarcity interaction |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {length:,} | {k8_adv} | {k8_j} | {k32_adv} | "
            "{scarcity} |".format(
                length=row["length"],
                k8_adv=_fmt(
                    row["k8"]["operators"]["shared_index_full"][
                        "evq_advantage"
                    ]
                ),
                k8_j=_fmt(row["k8"]["shared_interaction"]),
                k32_adv=_fmt(
                    row["k32"]["operators"]["shared_index_full"][
                        "evq_advantage"
                    ]
                ),
                scarcity=_fmt(row["scarcity_interaction"]),
            )
        )
    lines.extend(
        [
            "",
            "Positive EVQ advantage means lower EVQ NLL. The interaction is "
            "the shared-operator EVQ advantage minus the raw EVQ advantage.",
            "",
            "## Decision",
            "",
            f"`{gate['decision']}`. No confirmatory claim is supported.",
            "",
            "## Provenance",
            "",
            f"- Protocol SHA-256: `{gate['protocol_sha256']}`",
            f"- Evaluation code SHA-256: "
            f"`{gate['evaluation_code_sha256']}`",
            f"- Launcher SHA-256: `{ready['launcher_sha256']}`",
            f"- Gate receipt SHA-256: `{provenance['gate_sha256']}`",
            f"- READY receipt SHA-256: `{provenance['ready_sha256']}`",
            f"- Report renderer SHA-256: "
            f"`{provenance['report_code_sha256']}`",
            "",
            "Identity boundary: the shared-index transform is official YaRN "
            "only on native endpoint RoPE; on EVQ it is a fixed-index "
            "YaRN-component control.",
            "",
        ]
    )
    return "\n".join(lines)


def _complete_report(
    summary: dict[str, Any],
    gate: dict[str, Any],
    ready: dict[str, Any],
    provenance: dict[str, str],
) -> str:
    aggregate = {
        (row["stage"], row["length"]): row
        for row in summary["aggregates"]
    }
    supports = (
        summary["claim_gate"]
        == "SUPPORTS_SHARED_OPERATOR_SCARCITY_INTERACTION"
    )
    lines = [
        "# MLA shared-index YaRN operator-parity experiment",
        "",
        "Status: **complete three-seed test report**.",
        "",
        f"Claim decision: **`{summary['claim_gate']}`**.",
        "",
        f"Precision grade: **`{summary['precision_grade']}`**.",
        "",
        (
            "The registered multi-seed result supports the narrowly stated "
            "shared-operator scarcity interaction."
            if supports
            else (
                "The registered multi-seed result does not support the full "
                "shared-operator scarcity interaction claim."
            )
        ),
        "",
        "## Confirmatory criteria",
        "",
        *_criteria_table(summary["criteria"]),
        "",
        "### Seed-level precision checks",
        "",
        *_criteria_table(summary["precision_criteria"]),
        "",
        "| Estimand | Mean and seed-level 95% CI |",
        "| --- | ---: |",
        *[
            f"| `{name}` | {_ci(record)} |"
            for name, record in summary["seed_level_precision"].items()
        ],
        "",
        "## Primary 300M test effects",
        "",
        "| Length | K=8 raw advantage | K=8 shared advantage | K=8 "
        "operator interaction | K=32 shared advantage | Scarcity interaction |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for length in SPEC.eval_lengths:
        row = aggregate[("300m", length)]
        lines.append(
            "| {length:,} | {raw} | {shared} | {interaction} | "
            "{k32} | {scarcity} |".format(
                length=length,
                raw=_ci(
                    row["k8"]["operators"]["raw"]["evq_advantage"]
                ),
                shared=_ci(
                    row["k8"]["operators"]["shared_index_full"][
                        "evq_advantage"
                    ]
                ),
                interaction=_ci(row["k8"]["shared_interaction"]),
                k32=_ci(
                    row["k32"]["operators"]["shared_index_full"][
                        "evq_advantage"
                    ]
                ),
                scarcity=_ci(row["scarcity_interaction"]),
            )
        )
    lines.extend(
        [
            "",
            "Values are Native-minus-EVQ tail NLL; positive values favor EVQ. "
            "Intervals are t-based seed-level 95% intervals at n=3.",
            "",
            "## 300M operator decomposition",
            "",
        ]
    )
    for pairs in (8, 32):
        lines.extend(
            [
                f"### K={pairs}",
                "",
                "| Length | Operator | Native NLL | EVQ NLL | "
                "Native-EVQ NLL | EVQ PPL ratio |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for length in SPEC.eval_lengths:
            row = aggregate[("300m", length)][f"k{pairs}"][
                "operators"
            ]
            for operator in OPERATORS:
                current = row[operator]
                advantage = current["evq_advantage"]["mean"]
                lines.append(
                    "| {length:,} | `{operator}` | {native} | {evq} | "
                    "{advantage} | {ratio:.3f} |".format(
                        length=length,
                        operator=operator,
                        native=_plain(current["native_nll"]["mean"]),
                        evq=_plain(current["evq_nll"]["mean"]),
                        advantage=_fmt(advantage),
                        ratio=math.exp(-advantage),
                    )
                )
        lines.append("")
    lines.extend(
        [
            "## 200M direction check",
            "",
            "| Length | K=8 shared advantage | K=8 operator interaction | "
            "Scarcity interaction |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for length in PRIMARY_LENGTHS:
        row = aggregate[("200m", length)]
        lines.append(
            f"| {length:,} | "
            f"{_ci(row['k8']['operators']['shared_index_full']['evq_advantage'])} "
            f"| {_ci(row['k8']['shared_interaction'])} "
            f"| {_ci(row['scarcity_interaction'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed primary checks",
            "",
            "| Seed | K=8 shared advantage 16K | K=8 shared advantage 32K "
            "| Mean scarcity interaction | 4K EVQ-Native K=8/K=32 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in summary["per_seed_primary"]:
        advantages = record["k8_shared_advantages"]
        costs = record["in_domain_evq_minus_native"]
        lines.append(
            f"| {record['seed']} | {_fmt(advantages[0])} | "
            f"{_fmt(advantages[1])} | "
            f"{_fmt(record['mean_scarcity_interaction'])} | "
            f"{_fmt(costs['k8'])} / {_fmt(costs['k32'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            summary["identity_boundary"],
            "",
            summary["statistical_boundary"],
            "",
            "This mechanism experiment does not establish production-scale "
            "utility, official YaRN compatibility on EVQ, or replacement of "
            "inference-time range scaling.",
            "",
            "## Provenance",
            "",
            f"- Protocol SHA-256: `{summary['protocol_sha256']}`",
            f"- Training protocol SHA-256: "
            f"`{summary['base_training_protocol_sha256']}`",
            f"- Evaluation code SHA-256: "
            f"`{summary['evaluation_code_sha256']}`",
            f"- Training code SHA-256: "
            f"`{summary['base_training_code_sha256']}`",
            f"- Launcher SHA-256: `{ready['launcher_sha256']}`",
            f"- Summary SHA-256: `{provenance['summary_sha256']}`",
            f"- Gate receipt SHA-256: `{provenance['gate_sha256']}`",
            f"- READY receipt SHA-256: `{provenance['ready_sha256']}`",
            f"- Report renderer SHA-256: "
            f"`{provenance['report_code_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_markdown(
    *,
    gate: dict[str, Any],
    ready: dict[str, Any],
    summary: dict[str, Any] | None,
    provenance: dict[str, str],
) -> str:
    if ready.get("status") != "READY":
        raise ValueError("report requires a READY receipt")
    if gate.get("status") == "STOP":
        if summary is not None:
            raise ValueError("STOP gate cannot have a confirmatory summary")
        return _stop_report(gate, ready, provenance)
    if gate.get("status") != "PASS" or summary is None:
        raise ValueError("report requires terminal STOP or complete summary")
    return _complete_report(summary, gate, ready, provenance)


def generate_report(work_dir: Path, output: Path) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    output = output.resolve()
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError(output)
    gate = validate_gate(work_dir, require_pass=False)
    ready_path = work_dir / "operator_parity_ready.json"
    ready = json.loads(ready_path.read_text())
    summary_path = work_dir / "summary_mla_yarn_operator_parity.json"
    summary: dict[str, Any] | None = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        expected = {
            "status": "PASS",
            "protocol_sha256": SPEC.fingerprint(),
            "evaluation_code_sha256": code_fingerprint(),
            "base_training_protocol_sha256": BASE_SPEC.fingerprint(),
            "base_training_code_sha256": base_code_fingerprint(),
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                raise ValueError(f"summary identity mismatch: {key}")
        if tuple(
            record["seed"] for record in summary["per_seed_primary"]
        ) != SEEDS:
            raise ValueError("summary seed order/coverage mismatch")
    provenance = {
        "gate_sha256": sha256_file(
            work_dir / "operator_parity_gate.json"
        ),
        "ready_sha256": sha256_file(ready_path),
        "summary_sha256": (
            sha256_file(summary_path) if summary is not None else "not-read"
        ),
        "report_code_sha256": sha256_file(Path(__file__)),
    }
    text = render_markdown(
        gate=gate,
        ready=ready,
        summary=summary,
        provenance=provenance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".incomplete")
    temporary.write_text(text)
    temporary.replace(output)
    result = {
        "status": "PASS",
        "terminal_gate": gate["status"],
        "claim_gate": (
            summary["claim_gate"] if summary is not None else None
        ),
        "output": str(output),
        "output_sha256": sha256_file(output),
        **provenance,
    }
    receipt_temporary = receipt_path.with_name(
        receipt_path.name + ".incomplete"
    )
    receipt_temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    receipt_temporary.replace(receipt_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_report(args.work_dir, args.output)


if __name__ == "__main__":
    main()
