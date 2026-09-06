from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scripts.analysis.maxent_dilation_allocation import (
    DEFAULT_LAMBDAS,
    run_contract,
)
from scripts.eval.target_free_formal_eval import validate_external_table_support
from scripts.lib.rope.schedules import (
    geometric_inv_freq,
    maxent_dilation_factors,
    maxent_dilation_inv_freq,
)


def native_table() -> torch.Tensor:
    return geometric_inv_freq(head_dim=128, base=500_000.0)


def test_lambda_zero_is_log_uniform_dilation() -> None:
    native = native_table()
    q = (torch.arange(native.numel(), dtype=torch.float64) + 0.5) / native.numel()
    expected_r = torch.pow(4.0, q)
    observed_r = maxent_dilation_factors(
        native.numel(), target_factor=4.0, lambda_=0.0
    )
    assert torch.allclose(observed_r, expected_r, atol=0.0, rtol=1e-14)
    assert torch.allclose(
        maxent_dilation_inv_freq(native, target_factor=4.0, lambda_=0.0),
        native / expected_r,
        atol=0.0,
        rtol=1e-14,
    )


def test_finite_lambda_matches_maxent_quantile_cdf() -> None:
    count = 64
    lambda_ = -2.0
    dilation = maxent_dilation_factors(
        count, target_factor=4.0, lambda_=lambda_
    )
    tau = dilation.log()
    cdf = torch.expm1(lambda_ * tau) / math.expm1(lambda_ * math.log(4.0))
    q = (torch.arange(count, dtype=torch.float64) + 0.5) / count
    assert torch.allclose(cdf, q, atol=1e-14, rtol=0.0)


def test_default_grid_passes_cpu_contract() -> None:
    reports = run_contract(
        native_table(), target_factor=4.0, lambdas=DEFAULT_LAMBDAS
    )
    assert len(reports) == len(DEFAULT_LAMBDAS)
    assert all(receipt["fast_endpoint_moved"] for _, receipt in reports)
    assert all(receipt["slow_endpoint_moved"] for _, receipt in reports)


def test_explicit_support_accepts_moved_endpoints() -> None:
    native = np.asarray([1.0, 0.1], dtype=np.float32)
    moved = np.asarray([0.95, 0.02], dtype=np.float32)
    validate_external_table_support(moved, native, support="explicit", factor=4.0)
    with pytest.raises(RuntimeError, match="support identity drift"):
        validate_external_table_support(moved, native, support="native", factor=4.0)
