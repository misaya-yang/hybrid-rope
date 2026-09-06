#!/usr/bin/env python3
"""Build and analyse a frozen scale-orbit validation panel.

The script deliberately separates exact float32 orbit counting from continuous
finite-window Gram quantities.  ``build`` uses no language-model outcome.
``summarize`` only compares already completed, row-matched evaluator outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STATUS = "SCALE_ORBIT_VALIDATION_PANEL_FROZEN"
DEFAULT_GAIN_COEFFICIENT = 0.074


def float32_sha256(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(payload.tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".incomplete",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_table(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if (
        values.dtype != np.dtype("float32")
        or values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or not (values > 0).all()
        or not np.all(values[:-1] > values[1:])
    ):
        raise ValueError(f"invalid float32 frequency table: {path}")
    return np.ascontiguousarray(values, dtype="<f4")


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def float32_power_orbit_classes(values: np.ndarray, factor: float) -> int:
    """Count realized float32 power-orbit classes, not mathematical residues."""

    if factor != 4.0:
        raise ValueError("the realized exact-orbit audit is frozen to factor four")
    table = np.ascontiguousarray(values, dtype="<f4")
    lookup = {np.float32(value).tobytes(): index for index, value in enumerate(table)}
    dsu = DisjointSet(len(table))
    for index, value in enumerate(table):
        scaled = np.float32(value * np.float32(factor))
        match = lookup.get(scaled.tobytes())
        if match is not None:
            dsu.union(index, match)
    return len({dsu.find(index) for index in range(len(table))})


def float32_boundary_count(values: np.ndarray, factor: float) -> int:
    table = np.ascontiguousarray(values, dtype="<f4")
    existing = {np.float32(value).tobytes() for value in table}
    return sum(
        np.float32(value * np.float32(factor)).tobytes() not in existing
        for value in table
    )


def maximum_one_step_matching(values: np.ndarray, factor: float, tau: float) -> int:
    """Maximum one-step injection for interval costs on ordered log frequencies."""

    x = np.sort(-np.log(np.asarray(values, dtype=np.float64)))
    ideals = x - math.log(factor)
    tolerance = max(float(tau), 128.0 * np.finfo(np.float64).eps * max(1.0, float(x[-1])))
    target = 0
    matches = 0
    for ideal in ideals:
        while target < len(x) and x[target] < ideal - tolerance:
            target += 1
        if target < len(x) and x[target] <= ideal + tolerance:
            matches += 1
            target += 1
    return matches


def unique_scaled_frequencies(values: np.ndarray, factor: float, levels: int) -> np.ndarray:
    orbit = np.concatenate([
        np.asarray(values * np.float32(factor ** level), dtype="<f4")
        for level in range(levels + 1)
    ])
    return np.unique(orbit).astype(np.float64)


def continuous_gram(frequencies: np.ndarray, horizon: int) -> np.ndarray:
    delta = frequencies[:, None] - frequencies[None, :]
    return np.sinc(float(horizon) * delta / np.pi)


def discrete_gram(frequencies: np.ndarray, horizon: int) -> np.ndarray:
    delta = frequencies[:, None] - frequencies[None, :]
    wrapped = (delta + np.pi) % (2.0 * np.pi) - np.pi
    denominator = np.sin(wrapped / 2.0)
    numerator = np.sin((2 * int(horizon) + 1) * wrapped / 2.0)
    gram = np.divide(
        numerator,
        (2 * int(horizon) + 1) * denominator,
        out=np.ones_like(wrapped),
        where=np.abs(denominator) > 1e-15,
    )
    np.fill_diagonal(gram, 1.0)
    return gram


def ky_fan_lower_bound(gram: np.ndarray, dimension: int) -> dict[str, float]:
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    captured = float(np.clip(eigenvalues[:dimension], 0.0, None).sum())
    epsilon_squared = max(0.0, 1.0 - captured / gram.shape[0])
    off_diagonal = gram - np.eye(gram.shape[0])
    coherence = float(np.max(np.abs(off_diagonal))) if gram.shape[0] > 1 else 0.0
    return {
        "epsilon_squared_lower_bound": epsilon_squared,
        "epsilon_lower_bound": math.sqrt(epsilon_squared),
        "ky_fan_captured_energy": captured,
        "coherence": coherence,
    }


def phase_sup_cost(source: np.ndarray, target: np.ndarray, horizon: int) -> np.ndarray:
    delta = np.abs(source[:, None] - target[None, :])
    phase = float(horizon) * delta
    return np.where(phase >= np.pi, 2.0, 2.0 * np.sin(phase / 2.0))


def has_perfect_matching(cost: np.ndarray, threshold: float) -> bool:
    matched_source = [-1] * cost.shape[0]

    def augment(target: int, seen: list[bool]) -> bool:
        for source in np.flatnonzero(cost[:, target] <= threshold):
            source_index = int(source)
            if seen[source_index]:
                continue
            seen[source_index] = True
            if matched_source[source_index] < 0 or augment(matched_source[source_index], seen):
                matched_source[source_index] = target
                return True
        return False

    for target in range(cost.shape[1]):
        if not augment(target, [False] * cost.shape[0]):
            return False
    return True


def permutation_error_bounds(values: np.ndarray, factor: float, horizon: int) -> dict[str, float]:
    source = np.asarray(values, dtype=np.float64)
    target = source * float(factor)
    cost = phase_sup_cost(source, target, horizon)
    candidates = np.unique(cost)
    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if has_perfect_matching(cost, float(candidates[mid])):
            hi = mid
        else:
            lo = mid + 1
    return {
        "permutation_operator_error": float(candidates[lo]),
    }


def analyse_table(
    values: np.ndarray,
    *,
    factor: float,
    horizon: int,
    levels: int,
    tau_grid: Iterable[float],
) -> dict[str, Any]:
    x = -np.log(np.asarray(values, dtype=np.float64))
    span = float(x[-1] - x[0])
    alpha = math.log(factor)
    exact_capacity = min(len(values), math.floor(span / alpha) + 1)
    exact_q_lower_bound = math.ceil(len(values) / exact_capacity)
    orbit = unique_scaled_frequencies(values, factor, levels)
    continuous = continuous_gram(orbit, horizon)
    discrete = discrete_gram(orbit, horizon)
    approximate = []
    for tau in tau_grid:
        beta = alpha - float(tau)
        if beta <= 0:
            raise ValueError("every tau must be smaller than log(factor)")
        capacity = min(len(values), math.floor(span / beta) + 1)
        matches = maximum_one_step_matching(values, factor, float(tau))
        approximate.append({
            "tau": float(tau),
            "matched_channels": matches,
            "observed_one_step_leakage": 1.0 - matches / len(values),
            "theorem4_capacity": capacity,
            "theorem4_leakage_lower_bound": 1.0 / capacity,
        })
    return {
        "float32_sha256": float32_sha256(values),
        "pairs": len(values),
        "fast_frequency": float(values[0]),
        "slow_frequency": float(values[-1]),
        "log_span": span,
        "theorem3_capacity": exact_capacity,
        "theorem3_class_lower_bound": exact_q_lower_bound,
        "theorem3_boundary_fraction_lower_bound": exact_q_lower_bound / len(values),
        "realized_float32_power_orbit_classes": float32_power_orbit_classes(values, factor),
        "realized_float32_boundary_count": float32_boundary_count(values, factor),
        "scaled_orbit_modes": len(orbit),
        "continuous_gram": ky_fan_lower_bound(continuous, len(values)),
        "discrete_gram": ky_fan_lower_bound(discrete, len(values)),
        "permutation_upper_bounds": permutation_error_bounds(values, factor, horizon),
        "approximate_one_step_matching": approximate,
    }


def same_support_geometric(native: np.ndarray, slow_endpoint: np.float32) -> np.ndarray:
    x = np.linspace(
        -math.log(float(native[0])),
        -math.log(float(slow_endpoint)),
        len(native),
        dtype=np.float64,
    )
    result = np.exp(-x).astype("<f4")
    result[0] = native[0]
    result[-1] = slow_endpoint
    return result


def minimum_q_chain(native: np.ndarray, factor: float) -> tuple[np.ndarray, list[dict[str, int | float]]]:
    """Construct the minimum-class consecutive-chain panel for this fixed support."""

    alpha = math.log(factor)
    slow_endpoint = np.float32(native[-1] / np.float32(factor))
    span = -math.log(float(slow_endpoint)) + math.log(float(native[0]))
    capacity = min(len(native), math.floor(span / alpha) + 1)
    classes = math.ceil(len(native) / capacity)
    long_chains = len(native) - classes * (capacity - 1)
    remainder = span - (capacity - 1) * alpha
    if classes < 2 or long_chains < 2 or not 0.0 < remainder < alpha:
        raise RuntimeError("the frozen analytic chain construction is not valid for this support")

    starts: list[tuple[float, int]] = []
    for start in np.linspace(0.0, remainder, long_chains):
        starts.append((float(start), capacity))
    short_chains = classes - long_chains
    for index in range(short_chains):
        start = remainder + (index + 1) * (alpha - remainder) / (short_chains + 1)
        starts.append((float(start), capacity - 1))

    entries: list[tuple[np.float32, int, int]] = []
    chain_receipt: list[dict[str, int | float]] = []
    for chain_id, (start, length) in enumerate(starts):
        if chain_id == 0:
            root = np.float32(native[0])
        elif chain_id == long_chains - 1:
            root = np.float32(slow_endpoint * np.float32(factor ** (length - 1)))
        else:
            root = np.float32(math.exp(-(-math.log(float(native[0])) + start)))
        chain_receipt.append({"chain_id": chain_id, "length": length, "start_log_offset": start})
        value = root
        for level in range(length):
            entries.append((np.float32(value), chain_id, level))
            value = np.float32(value / np.float32(factor))

    entries.sort(key=lambda item: float(item[0]), reverse=True)
    result = np.asarray([item[0] for item in entries], dtype="<f4")
    if len(result) != len(native) or not np.all(result[:-1] > result[1:]):
        raise RuntimeError("exact-chain construction is not a strict K-table")
    if result[0] != native[0] or result[-1] != slow_endpoint:
        raise RuntimeError("exact-chain support endpoint drift")
    if float32_power_orbit_classes(result, factor) != classes:
        raise RuntimeError("exact-chain realized class count drift")
    if float32_boundary_count(result, factor) != classes:
        raise RuntimeError("exact-chain realized boundary count drift")
    return result, chain_receipt


def ulp_jittered_chain(
    chain: np.ndarray,
    factor: float,
) -> np.ndarray:
    # Infer levels from exact float32 power relations before perturbing.
    lookup = {np.float32(value).tobytes(): index for index, value in enumerate(chain)}
    inferred: dict[bytes, int] = {}
    for value in chain:
        current = np.float32(value)
        level = 0
        while np.float32(current * np.float32(factor)).tobytes() in lookup:
            current = np.float32(current * np.float32(factor))
            level += 1
        inferred[np.float32(value).tobytes()] = level

    jittered = chain.copy()
    positive_infinity = np.float32(np.inf)
    for index in range(1, len(chain) - 1):
        level = inferred[np.float32(chain[index]).tobytes()]
        value = np.float32(chain[index])
        for _ in range(level + 1):
            value = np.nextafter(value, positive_infinity, dtype=np.float32)
        jittered[index] = value
    if not np.all(jittered[:-1] > jittered[1:]):
        raise RuntimeError("ULP jitter introduced an order crossing")
    if float32_power_orbit_classes(jittered, factor) != len(jittered):
        raise RuntimeError("ULP jitter did not break every exact power-orbit class")
    return jittered


def validate_same_support(table: np.ndarray, native: np.ndarray, factor: float) -> None:
    expected_slow = np.float32(native[-1] / np.float32(factor))
    if table[0] != native[0] or table[-1] != expected_slow:
        raise RuntimeError("candidate does not share the frozen extended support")
    if not np.all(table[:-1] > table[1:]):
        raise RuntimeError("candidate is not strictly ordered")


def command_build(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    native = load_table(args.native.resolve())
    current_p2 = load_table(args.current_p2.resolve())
    if args.expected_native_sha256 and float32_sha256(native) != args.expected_native_sha256:
        raise RuntimeError("Native float32 hash drift")
    if args.expected_p2_sha256 and float32_sha256(current_p2) != args.expected_p2_sha256:
        raise RuntimeError("current-p2 float32 hash drift")
    validate_same_support(current_p2, native, args.factor)

    chain, chain_receipt = minimum_q_chain(native, args.factor)
    jitter = ulp_jittered_chain(chain, args.factor)
    geometric = same_support_geometric(native, current_p2[-1])
    for table in (geometric, chain, jitter):
        validate_same_support(table, native, args.factor)

    max_phase_delta = float(
        args.validation_length
        * np.max(np.abs(chain.astype(np.float64) - jitter.astype(np.float64)))
    )
    if max_phase_delta > args.max_jitter_phase_delta:
        raise RuntimeError(
            f"jitter control exceeds phase-distance contract: {max_phase_delta}"
        )

    tau_grid = [float(value) * math.log(args.factor) for value in args.tau_fractions]
    tables = {
        "native": native,
        "same_support_geometric_s4": geometric,
        "legacy_u_p2_log_s4": current_p2,
        "min_q_exact_chain_s4": chain,
        "min_q_ulp_jitter_s4": jitter,
    }
    args.output.mkdir(parents=True)
    manifest_tables: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    gain = 1.0 + float(args.gain_coefficient) * math.log(args.factor)
    for name, table in tables.items():
        path = args.output / f"{name}.npy"
        np.save(path, np.ascontiguousarray(table, dtype="<f4"), allow_pickle=False)
        metrics[name] = analyse_table(
            table,
            factor=args.factor,
            horizon=args.native_length,
            levels=args.levels,
            tau_grid=tau_grid,
        )
        manifest_tables[name] = {
            "path": path.name,
            "file_sha256": sha256_file(path),
            "float32_sha256": float32_sha256(table),
            "method": "native" if name == "native" else "external_table_static",
            "table_support": "native" if name == "native" else "native_div_factor",
            "attention_scaling": 1.0 if name == "native" else gain,
        }

    exact_q = metrics["min_q_exact_chain_s4"]["realized_float32_power_orbit_classes"]
    jitter_q = metrics["min_q_ulp_jitter_s4"]["realized_float32_power_orbit_classes"]
    if exact_q >= jitter_q:
        raise RuntimeError("primary discontinuity control did not separate exact class count")

    manifest = {
        "status": STATUS,
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "source_native_file_sha256": sha256_file(args.native.resolve()),
        "source_current_p2_file_sha256": sha256_file(args.current_p2.resolve()),
        "question": (
            "Do exact boundary count, approximate matching, and Fourier-orbit rank "
            "remain distinct on a near-identical table pair, and do their prospective "
            "rankings have any mature-model behavioral relevance?"
        ),
        "factor": float(args.factor),
        "native_length": int(args.native_length),
        "validation_length": int(args.validation_length),
        "levels": int(args.levels),
        "gain_coefficient": float(args.gain_coefficient),
        "nonnative_attention_scaling": gain,
        "tau_fractions_of_log_factor": [float(value) for value in args.tau_fractions],
        "construction_uses_lm_outcomes": False,
        "primary_contrast": ["min_q_exact_chain_s4", "min_q_ulp_jitter_s4"],
        "primary_pair_max_phase_delta_at_validation_length": max_phase_delta,
        "chain_construction": chain_receipt,
        "tables": manifest_tables,
        "interpretation_boundary": (
            "Theorem checks are algebraic/numerical. GPU scores test only the separate "
            "behavioral bridge and cannot prove the theorem, identify a universal z, or certify novelty."
        ),
    }
    atomic_json(args.output / "metrics.json", metrics)
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def paired_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    value_field: str,
    higher_is_better: bool,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    left_map = {tuple(row[field] for field in key_fields): row for row in left}
    right_map = {tuple(row[field] for field in key_fields): row for row in right}
    if left_map.keys() != right_map.keys() or not left_map:
        raise RuntimeError("paired bootstrap requires identical nonempty row identities")
    groups: dict[Any, list[float]] = {}
    for key in sorted(left_map):
        value = float(left_map[key][value_field]) - float(right_map[key][value_field])
        groups.setdefault(key[0], []).append(value if higher_is_better else -value)
    deltas = [np.asarray(values, dtype=np.float64) for values in groups.values()]
    rng = np.random.default_rng(seed)
    draws = np.mean([
        delta[rng.integers(0, len(delta), size=(replicates, len(delta)))].mean(axis=1)
        for delta in deltas
    ], axis=0)
    return {
        "orientation": "positive favors left",
        "aggregation": "paired bootstrap within task, then equal-task macro",
        "paired_rows": sum(len(delta) for delta in deltas),
        "strata": len(deltas),
        "mean_delta": float(np.mean([delta.mean() for delta in deltas])),
        "ci95": percentile_interval(draws),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def command_summarize(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text())
    primary_left, primary_right = manifest["primary_contrast"]
    root = args.results_root.resolve()
    summary: dict[str, Any] = {
        "status": "SCALE_ORBIT_VALIDATION_SUMMARY",
        "manifest_sha256": sha256_file(args.manifest),
        "primary_contrast": [primary_left, primary_right],
        "formal": {},
        "ruler": {},
        "claim_limit": manifest["interpretation_boundary"],
    }
    for family in ("formal", "ruler"):
        available = {}
        for name in manifest["tables"]:
            examples = root / family / name / "examples.jsonl"
            results = root / family / name / "results.json"
            if examples.is_file() and results.is_file():
                available[name] = {
                    "examples": load_jsonl(examples),
                    "results_sha256": sha256_file(results),
                    "examples_sha256": sha256_file(examples),
                }
        summary[family]["available"] = {
            name: {key: value for key, value in item.items() if key != "examples"}
            for name, item in available.items()
        }
        if primary_left not in available or primary_right not in available:
            summary[family]["primary_status"] = "INCOMPLETE"
            continue
        if family == "formal":
            left_rows = [row for row in available[primary_left]["examples"] if row["task"] == "pg19"]
            right_rows = [row for row in available[primary_right]["examples"] if row["task"] == "pg19"]
            for multiplier in sorted({int(row["multiplier"]) for row in left_rows}):
                summary[family][f"pg19_x{multiplier}"] = paired_bootstrap(
                    [row for row in left_rows if int(row["multiplier"]) == multiplier],
                    [row for row in right_rows if int(row["multiplier"]) == multiplier],
                    key_fields=("task", "row_sha256", "multiplier"),
                    value_field="nll",
                    higher_is_better=False,
                    seed=args.bootstrap_seed + multiplier,
                    replicates=args.bootstrap_replicates,
                )
        else:
            left_rows = available[primary_left]["examples"]
            right_rows = available[primary_right]["examples"]
            for length in sorted({int(row["nominal_length"]) for row in left_rows}):
                summary[family][f"L{length}"] = paired_bootstrap(
                    [row for row in left_rows if int(row["nominal_length"]) == length],
                    [row for row in right_rows if int(row["nominal_length"]) == length],
                    key_fields=("task", "nominal_length", "local_index"),
                    value_field="official_task_score",
                    higher_is_better=True,
                    seed=args.bootstrap_seed + length,
                    replicates=args.bootstrap_replicates,
                )
        summary[family]["primary_status"] = "COMPLETE"
    atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--native", type=Path, required=True)
    build.add_argument("--current-p2", type=Path, required=True)
    build.add_argument("--expected-native-sha256")
    build.add_argument("--expected-p2-sha256")
    build.add_argument("--factor", type=float, default=4.0)
    build.add_argument("--native-length", type=int, default=4096)
    build.add_argument("--validation-length", type=int, default=16384)
    build.add_argument("--levels", type=int, default=1)
    build.add_argument("--gain-coefficient", type=float, default=DEFAULT_GAIN_COEFFICIENT)
    build.add_argument(
        "--tau-fractions",
        type=float,
        nargs="+",
        default=(0.0, 1e-7, 1e-6, 1e-4, 0.01, 0.05, 0.1),
    )
    build.add_argument("--max-jitter-phase-delta", type=float, default=0.01)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(function=command_build)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--manifest", type=Path, required=True)
    summarize.add_argument("--results-root", type=Path, required=True)
    summarize.add_argument("--bootstrap-seed", type=int, default=202609041)
    summarize.add_argument("--bootstrap-replicates", type=int, default=10000)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.set_defaults(function=command_summarize)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
