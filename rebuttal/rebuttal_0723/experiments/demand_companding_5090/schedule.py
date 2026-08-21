#!/usr/bin/env python3
"""Demand-companding schedule construction and finite-table assertions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


LAMBDA_VALUES: tuple[float, ...] = (0.0, 0.1, 0.3)
DEFAULT_TAU = 4.0
DEFAULT_BASE = 500_000.0
DEFAULT_K = 32


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: Iterable[float], *, dtype: str = "<f8") -> str:
    array = np.asarray(list(values), dtype=dtype).copy(order="C")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _get_path(payload: Any, path: str) -> Any:
    value = payload
    for component in path.split(".") if path else ():
        if isinstance(value, Mapping):
            if component not in value:
                raise KeyError(component)
            value = value[component]
        else:
            raise TypeError(component)
    return value


def _first_present(payload: Mapping[str, Any], paths: tuple[str, ...]) -> tuple[Any, str] | tuple[None, None]:
    for path in paths:
        try:
            return _get_path(payload, path), path
        except (KeyError, TypeError):
            continue
    return None, None


def _find_mapping_with_demand(value: Any, path: str = "") -> tuple[Mapping[str, Any], str, str] | None:
    """Find a nested R0 case such as ``{"cases":[{"x":...,"m":...}]}``."""

    if isinstance(value, Mapping):
        # The current R0 owner uses ``demand_density``.  Keep the older
        # aliases below because the schedule package is also used with the
        # archived case-shaped R0 probes.
        for key in ("demand_density", "m", "demand", "p_dem", "density"):
            if key in value:
                return value, key, f"{path}.{key}".lstrip(".")
        for key, child in value.items():
            found = _find_mapping_with_demand(child, f"{path}.{key}".lstrip("."))
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_mapping_with_demand(child, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _as_vector(value: Any, *, label: str) -> np.ndarray:
    if isinstance(value, Mapping):
        if "values" in value:
            value = value["values"]
        elif "array" in value:
            value = value["array"]
    if isinstance(value, Mapping):
        numeric_keys: list[tuple[float, Any]] = []
        for key, item in value.items():
            try:
                numeric_keys.append((float(key), item))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} mapping keys must be numeric") from exc
        numeric_keys.sort(key=lambda pair: pair[0])
        value = [item for _, item in numeric_keys]
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{label} must be a one-dimensional vector of length >=2")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


@dataclass(frozen=True)
class DemandProfile:
    """Validated R0 demand profile in an increasing normalised coordinate."""

    delta: np.ndarray
    m: np.ndarray
    source_path: str
    source_sha256: str
    m_path: str
    delta_path: str | None
    delta_was_inferred: bool
    raw_delta_min: float
    raw_delta_max: float
    metadata: dict[str, Any]
    profile_path: str | None = None
    coordinate_path: str | None = None

    @property
    def count(self) -> int:
        return int(self.m.size)

    @property
    def m_sha256(self) -> str:
        return sha256_array(self.m)


def load_r0_profile(path: str | Path, *, m_path: str | None = None, delta_path: str | None = None) -> DemandProfile:
    """Load ``m`` from R0 JSON without silently rescaling its values."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("R0 JSON root must be an object")

    # This is the canonical R0 profile as of the 2026-08-20 analysis.  It is
    # intentionally tried before generic recursive discovery so a JSON file
    # containing several diagnostic profiles cannot silently select another
    # one.
    requested_m_path = m_path or "profiles.endpoint_log__mass__no_self.demand_density"
    record: Mapping[str, Any] = payload
    profile_path: str | None = None
    try:
        raw_m = _get_path(payload, requested_m_path)
        resolved_m_path = requested_m_path
        profile_path = requested_m_path.rsplit(".", 1)[0] if "." in requested_m_path else None
        if profile_path:
            candidate = _get_path(payload, profile_path)
            if isinstance(candidate, Mapping):
                record = candidate
    except (KeyError, TypeError):
        raw_m, resolved_m_path = _first_present(
            payload,
            (
                "m",
                "demand.m",
                "profile.m",
                "metrics.m",
                "attention_demand.m",
            ),
        )
        if raw_m is None or resolved_m_path is None:
            found = _find_mapping_with_demand(payload)
            if found is None:
                raise KeyError("R0 JSON must expose m at 'm' or a supported nested path")
            record, key, resolved_m_path = found
            raw_m = record[key]
            profile_path = resolved_m_path.rsplit(".", 1)[0] if "." in resolved_m_path else None
        elif "." in resolved_m_path:
            profile_path = resolved_m_path.rsplit(".", 1)[0]
            candidate = _get_path(payload, profile_path)
            if isinstance(candidate, Mapping):
                record = candidate
    m = _as_vector(raw_m, label="m")
    if (m <= 0).any():
        raise ValueError("m must be strictly positive")

    if delta_path is not None:
        raw_delta = _get_path(payload, delta_path)
        resolved_delta_path: str | None = delta_path
    else:
        raw_delta, resolved_delta_path = _first_present(
            record,
            (
                "phi_centers",
                "delta",
                "x",
                "profile.delta",
                "demand.delta",
                "grid",
                "phi",
                "coordinate",
                "coordinates",
                "positions",
            ),
        )
    delta_was_inferred = raw_delta is None
    if delta_was_inferred:
        raw_delta_array = np.linspace(0.0, 1.0, m.size, dtype=np.float64)
    else:
        raw_delta_array = _as_vector(raw_delta, label="delta")
        if raw_delta_array.size != m.size:
            raise ValueError("delta and m must have the same length")
    if not np.all(np.diff(raw_delta_array) > 0):
        raise ValueError("delta must be strictly increasing")
    raw_min = float(raw_delta_array[0])
    raw_max = float(raw_delta_array[-1])
    if not raw_max > raw_min:
        raise ValueError("delta must have a non-zero span")
    delta = (raw_delta_array - raw_min) / (raw_max - raw_min)
    delta[0] = 0.0
    delta[-1] = 1.0
    if not np.all(np.diff(delta) > 0):
        raise ValueError("normalised delta is not strictly increasing")

    metadata: dict[str, Any] = {}
    # ``measurement`` is source-provenance metadata.  It must be retained in
    # the manifest, but callers must not treat source head_dim/K as the target
    # 151M schedule configuration.
    measurement = payload.get("measurement")
    if isinstance(measurement, Mapping):
        metadata["measurement"] = dict(measurement)
        for key in ("base", "rope_base", "K", "k", "head_dim", "tau"):
            if key in measurement:
                metadata[key] = measurement[key]
    for key in ("base", "rope_base", "K", "k", "head_dim", "tau"):
        if key in record:
            metadata.setdefault(key, record[key])
        elif key in payload:
            metadata.setdefault(key, payload[key])
    if isinstance(record.get("metadata"), Mapping):
        metadata["r0_metadata"] = dict(record["metadata"])
    coordinate_source_path = resolved_delta_path
    if profile_path and resolved_delta_path and "." not in resolved_delta_path and not resolved_delta_path.startswith("["):
        coordinate_source_path = f"{profile_path}.{resolved_delta_path}"
    return DemandProfile(
        delta=delta,
        m=m,
        source_path=str(source),
        source_sha256=sha256_file(source),
        m_path=str(resolved_m_path),
        delta_path=resolved_delta_path,
        delta_was_inferred=delta_was_inferred,
        raw_delta_min=raw_min,
        raw_delta_max=raw_max,
        metadata=metadata,
        profile_path=profile_path,
        coordinate_path=coordinate_source_path,
    )


def demand_density(profile: DemandProfile, lambda_value: float) -> np.ndarray:
    """Compute rho proportional to ``((1-lambda)m + lambda) ** (1/3)``."""

    lam = float(lambda_value)
    if not math.isfinite(lam) or lam < 0.0 or lam > 1.0:
        raise ValueError("lambda must lie in [0, 1]")
    mixed = (1.0 - lam) * profile.m + lam
    if not np.isfinite(mixed).all() or (mixed < 0).any():
        raise ValueError("companded demand must be non-negative and finite")
    rho = np.cbrt(mixed)
    area = float(np.sum(0.5 * (rho[:-1] + rho[1:]) * np.diff(profile.delta)))
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("density integral must be positive and finite")
    rho = rho / area
    if not np.isfinite(rho).all() or (rho < 0).any():
        raise ValueError("normalised density must be non-negative and finite")
    return rho


def density_cdf(delta: np.ndarray, rho: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    if delta.shape != rho.shape or not np.all(np.diff(delta) > 0):
        raise ValueError("delta and rho must share a strictly increasing grid")
    if (rho < 0).any() or not np.isfinite(rho).all():
        raise ValueError("rho must be non-negative and finite")
    increments = 0.5 * (rho[:-1] + rho[1:]) * np.diff(delta)
    cdf = np.concatenate(([0.0], np.cumsum(increments, dtype=np.float64)))
    if not np.isfinite(cdf).all() or not cdf[-1] > 0:
        raise ValueError("CDF integral must be positive and finite")
    cdf /= cdf[-1]
    cdf[0] = 0.0
    cdf[-1] = 1.0
    if not np.all(np.diff(cdf) >= 0):
        raise ValueError("CDF must be non-decreasing")
    return cdf


def _strictify_quantiles(phi: np.ndarray) -> tuple[np.ndarray, int]:
    """Make finite generalized-inverse samples strictly increasing.

    A zero-density interval creates a CDF plateau.  The generalized inverse is
    therefore allowed to return the same coordinate for two requested
    quantiles.  A finite RoPE table still needs a strictly ordered frequency
    tensor, so only those ties are moved by a few floating-point ulps.  The
    count is returned and recorded in the schedule receipt.
    """

    values = np.asarray(phi, dtype=np.float64).copy()
    if values.size < 2:
        raise ValueError("at least two quantiles are required")
    tie_count = 0
    # Several ulps are needed because the subsequent exp/log-base mapping can
    # otherwise round two distinct coordinates to the same float64 frequency.
    minimum_step = 8.0 * np.finfo(np.float64).eps
    for index in range(1, values.size - 1):
        lower = values[index - 1]
        if values[index] <= lower:
            values[index] = lower + minimum_step * max(1.0, abs(lower))
            tie_count += 1
    if values[-1] <= values[-2]:
        raise ValueError("generalized inverse CDF exhausted the endpoint support")
    values[0] = 0.0
    values[-1] = 1.0
    if not np.all(np.diff(values) > 0.0) or values[-2] >= 1.0:
        raise ValueError("could not construct a strict finite quantile table")
    return values, tie_count


def quantile_phi_details(delta: np.ndarray, rho: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    """Return strict finite generalized-inverse quantiles and tie count."""

    if int(k) < 2:
        raise ValueError("K must be >=2")
    cdf = density_cdf(delta, rho)
    p = np.linspace(0.0, 1.0, int(k), dtype=np.float64)
    coordinates = np.asarray(delta, dtype=np.float64)
    # np.interp's duplicate-x behavior is implementation-dependent.  The
    # searchsorted form below is the left-continuous generalized inverse
    # inf{x: F(x)>=p}, including CDF plateaus from zero m values.
    phi = np.empty_like(p)
    for index, probability in enumerate(p):
        if probability <= 0.0:
            phi[index] = coordinates[0]
            continue
        if probability >= 1.0:
            phi[index] = coordinates[-1]
            continue
        upper = int(np.searchsorted(cdf, probability, side="left"))
        if upper <= 0:
            phi[index] = coordinates[0]
        elif upper >= cdf.size:
            phi[index] = coordinates[-1]
        elif cdf[upper] == probability:
            phi[index] = coordinates[upper]
        else:
            lower = upper - 1
            if cdf[upper] <= cdf[lower]:
                phi[index] = coordinates[upper]
            else:
                fraction = (probability - cdf[lower]) / (cdf[upper] - cdf[lower])
                phi[index] = coordinates[lower] + fraction * (coordinates[upper] - coordinates[lower])
    phi = (phi - coordinates[0]) / (coordinates[-1] - coordinates[0])
    phi, tie_count = _strictify_quantiles(phi)
    assert_phi(phi, k=int(k))
    return phi, tie_count


def quantile_phi(delta: np.ndarray, rho: np.ndarray, k: int) -> np.ndarray:
    """Return K endpoint-inclusive inverse-CDF quantiles in delta->phi order."""

    return quantile_phi_details(delta, rho, k)[0]


def cosh_phi(k: int, tau: float = DEFAULT_TAU) -> np.ndarray:
    """Endpoint-inclusive Cosh quantiles used by the fixed-table baseline."""

    if int(k) < 2:
        raise ValueError("K must be >=2")
    tau_f = float(tau)
    if not math.isfinite(tau_f) or tau_f < 0.0:
        raise ValueError("tau must be finite and non-negative")
    p = np.linspace(0.0, 1.0, int(k), dtype=np.float64)
    if tau_f < 1e-10:
        phi = p
    else:
        phi = 1.0 - np.arcsinh((1.0 - p) * math.sinh(tau_f)) / tau_f
    phi[0] = 0.0
    phi[-1] = 1.0
    assert_phi(phi, k=int(k))
    return phi


def geo_phi(k: int) -> np.ndarray:
    if int(k) < 2:
        raise ValueError("K must be >=2")
    phi = np.linspace(0.0, 1.0, int(k), dtype=np.float64)
    assert_phi(phi, k=int(k))
    return phi


def endpoint_anchored_omega(phi: np.ndarray, base: float, *, k: int | None = None) -> np.ndarray:
    """Convert normalised phi to positive inverse frequencies with fixed support."""

    phi_array = np.asarray(phi, dtype=np.float64)
    assert_phi(phi_array, k=int(phi_array.size))
    base_f = float(base)
    if not math.isfinite(base_f) or base_f <= 1.0:
        raise ValueError("base must be finite and >1")
    count = int(k if k is not None else phi_array.size)
    if count != int(phi_array.size) or count < 2:
        raise ValueError("k must equal the finite table size and be >=2")
    # The native sampled support is R=(K-1)/K log(base), not log(base).
    log_span = (count - 1) / count * math.log(base_f)
    omega = np.exp(-log_span * phi_array)
    assert_omega(omega, base=base_f, k=count)
    return omega


def assert_phi(phi: np.ndarray, *, k: int) -> None:
    values = np.asarray(phi, dtype=np.float64)
    if values.ndim != 1 or values.size != int(k):
        raise AssertionError(f"phi must have shape ({k},)")
    if not np.isfinite(values).all():
        raise AssertionError("phi contains non-finite values")
    if not math.isclose(float(values[0]), 0.0, abs_tol=1e-12):
        raise AssertionError("phi[0] is not anchored")
    if not math.isclose(float(values[-1]), 1.0, abs_tol=1e-12):
        raise AssertionError("phi[-1] is not anchored")
    if not np.all(np.diff(values) > 0.0):
        raise AssertionError("phi must be strictly increasing")
    if float(values[0]) < 0.0 or float(values[-1]) > 1.0:
        raise AssertionError("phi leaves support [0,1]")


def assert_omega(omega: np.ndarray, *, base: float, k: int) -> None:
    values = np.asarray(omega, dtype=np.float64)
    if values.ndim != 1 or values.size != int(k):
        raise AssertionError(f"omega must have shape ({k},)")
    if not np.isfinite(values).all() or not (values > 0.0).all():
        raise AssertionError("all frequencies must be positive and finite")
    if not np.all(np.diff(values) < 0.0):
        raise AssertionError("omega must be strictly decreasing")
    if not math.isclose(float(values[0]), 1.0, abs_tol=1e-12):
        raise AssertionError("omega[0] does not match support")
    expected_min = float(base) ** (-(int(k) - 1) / int(k))
    if not math.isclose(float(values[-1]), expected_min, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("omega[-1] does not match support")


def canonical_geo_inv_freq(k: int, base: float) -> np.ndarray:
    """Canonical sampled Geo/FMRoPE values ``base**(-j/K)`` in float64."""

    count = int(k)
    if count < 2:
        raise ValueError("K must be >=2")
    base_f = float(base)
    if not math.isfinite(base_f) or base_f <= 1.0:
        raise ValueError("base must be finite and >1")
    index = np.arange(count, dtype=np.float64)
    return 1.0 / (base_f ** (index / count))


def _segment_map(target: np.ndarray, start: int, end: int, lower: float, upper: float) -> np.ndarray:
    if end <= start:
        raise ValueError("segment must contain at least two points")
    source = np.asarray(target[start : end + 1], dtype=np.float64)
    local = (source - source[0]) / (source[-1] - source[0])
    mapped = float(lower) + (float(upper) - float(lower)) * local
    mapped[0] = float(lower)
    mapped[-1] = float(upper)
    if not np.all(np.diff(mapped) > 0.0):
        raise AssertionError("mapped segment is not strictly increasing")
    return mapped


def anchored_tail_phi(target_phi: np.ndarray, *, tail_count: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Compand the prefix while retaining every slow-tail Geo channel."""

    target = np.asarray(target_phi, dtype=np.float64)
    k = int(target.size)
    assert_phi(target, k=k)
    count = int(tail_count if tail_count is not None else max(2, k // 4))
    if count < 2 or count >= k - 1:
        raise ValueError("tail_count must leave at least two prefix points")
    split = k - count
    geo = geo_phi(k)
    output = np.empty_like(target)
    output[:split] = _segment_map(target, 0, split - 1, 0.0, float(geo[split - 1]))
    output[split:] = geo[split:]
    assert_phi(output, k=k)
    return output, {
        "mode": "anchored_tail",
        "retained_slow_tail_count": count,
        "retained_slow_tail_start_index": split,
        "retained_slow_tail_phi": output[split:].tolist(),
        "content_dimensions_removed": 0,
    }


def mid_only_phi(target_phi: np.ndarray, *, start_fraction: float = 0.25, end_fraction: float = 0.75) -> tuple[np.ndarray, dict[str, Any]]:
    """Compand only a middle segment; preserve both fast and slow Geo bands."""

    target = np.asarray(target_phi, dtype=np.float64)
    k = int(target.size)
    assert_phi(target, k=k)
    if not 0.0 < start_fraction < end_fraction < 1.0:
        raise ValueError("mid-only fractions must satisfy 0<start<end<1")
    geo = geo_phi(k)
    start = max(1, int(round((k - 1) * float(start_fraction))))
    end = min(k - 2, int(round((k - 1) * float(end_fraction))))
    if end <= start:
        raise ValueError("mid-only interval is empty for this K")
    output = geo.copy()
    output[start : end + 1] = _segment_map(target, start, end, float(geo[start]), float(geo[end]))
    assert_phi(output, k=k)
    return output, {
        "mode": "mid_only",
        "mid_start_index": start,
        "mid_end_index": end,
        "retained_fast_prefix_count": start,
        "retained_slow_tail_count": k - end - 1,
        "retained_slow_tail_start_index": end + 1,
        "retained_slow_tail_phi": output[end + 1 :].tolist(),
        "content_dimensions_removed": 0,
    }


def build_phi_schedules(profile: DemandProfile, *, k: int = DEFAULT_K, tau: float = DEFAULT_TAU, include_r1: bool = True) -> dict[str, dict[str, Any]]:
    """Build Geo/Cosh/R2 and R1 control phi schedules."""

    schedules: dict[str, dict[str, Any]] = {
        "geo": {"phi": geo_phi(k), "family": "geometric", "mode": "full_table", "lambda": None},
        "cosh_tau4": {"phi": cosh_phi(k, tau=tau), "family": "cosh", "mode": "full_table", "lambda": None, "tau": float(tau)},
    }
    for lambda_value in LAMBDA_VALUES:
        label = f"demand_lambda_{lambda_value:g}".replace(".", "p")
        target, tie_count = quantile_phi_details(
            profile.delta, demand_density(profile, lambda_value), k
        )
        schedules[label] = {
            "phi": target,
            "family": "demand_companding",
            "mode": "full_table",
            "lambda": float(lambda_value),
            "density_formula": "((1-lambda)*m + lambda)^(1/3)",
            "direction": "delta_increasing_to_phi_increasing",
            "quantile_method": "left_continuous_generalized_inverse_cdf",
            "zero_density_bins_allowed": True,
            "quantile_tie_break_count": int(tie_count),
        }
        if include_r1:
            tail_phi, tail_meta = anchored_tail_phi(target)
            mid_phi, mid_meta = mid_only_phi(target)
            schedules[f"anchored_tail_{label}"] = {"phi": tail_phi, "family": "demand_companding", "lambda": float(lambda_value), "density_formula": "((1-lambda)*m + lambda)^(1/3)", "direction": "delta_increasing_to_phi_increasing", **tail_meta}
            schedules[f"mid_only_{label}"] = {"phi": mid_phi, "family": "demand_companding", "lambda": float(lambda_value), "density_formula": "((1-lambda)*m + lambda)^(1/3)", "direction": "delta_increasing_to_phi_increasing", **mid_meta}
    return schedules


def schedule_receipt(name: str, entry: Mapping[str, Any], *, base: float, k: int) -> dict[str, Any]:
    """Materialise one schedule's finite frequency receipt."""

    phi = np.asarray(entry["phi"], dtype=np.float64)
    omega = endpoint_anchored_omega(phi, base, k=int(k))
    canonical_match = None
    if name == "geo":
        # Geo is the standard sampled endpoint grid.  Use the independent
        # canonical expression for the stored values and compare the mapped
        # result elementwise after the float32 cast used by the trainer.
        canonical = canonical_geo_inv_freq(int(k), float(base))
        if not np.array_equal(omega.astype(np.float32), canonical.astype(np.float32)):
            raise AssertionError("Geo schedule does not match canonical training_inv_freq elementwise")
        omega = canonical
        canonical_match = True
    metadata = {key: value for key, value in entry.items() if key != "phi"}
    family = metadata.pop("family")
    return {
        "name": name,
        "K": int(k),
        "base": float(base),
        "family": family,
        **metadata,
        "phi": phi.tolist(),
        "inv_freq": omega.tolist(),
        "phi_sha256": sha256_array(phi),
        "inv_freq_float64_sha256": sha256_array(omega),
        "inv_freq_float32_sha256": sha256_array(omega, dtype="<f4"),
        "support": {
            "phi_min": float(phi[0]),
            "phi_max": float(phi[-1]),
            "log_span": (int(k) - 1) / int(k) * math.log(float(base)),
            "omega_max": float(omega[0]),
            "omega_min": float(omega[-1]),
            "omega_min_formula": "base^(-((K-1)/K))",
        },
        "assertions": {
            "strict_phi_increasing": True,
            "strict_omega_decreasing": True,
            "endpoint_anchored": True,
            "support_anchored": True,
            "all_frequency_values_positive": True,
            "canonical_training_inv_freq_match": canonical_match,
            "content_dimensions_removed": int(metadata.get("content_dimensions_removed", 0)),
        },
    }


def build_schedule_receipts(profile: DemandProfile, *, base: float = DEFAULT_BASE, k: int = DEFAULT_K, tau: float = DEFAULT_TAU, include_r1: bool = True) -> dict[str, dict[str, Any]]:
    schedules = build_phi_schedules(profile, k=int(k), tau=float(tau), include_r1=include_r1)
    return {name: schedule_receipt(name, entry, base=float(base), k=int(k)) for name, entry in schedules.items()}
