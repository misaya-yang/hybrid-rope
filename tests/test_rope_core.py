#!/usr/bin/env python3
"""
Core unit tests for scripts/lib/rope/ — the mathematical heart of EVQ-Cosh.

Covers:
  - learnable_evq.py: EVQ frequency computation, τ=0 geometric recovery,
    boundary anchoring, Taylor-full continuity, gradient flow, numerical stability
  - schedules.py: build_inv_freq for all methods, canonical_method aliases,
    geometric_inv_freq properties
  - inject.py: find_rotary_modules, apply_inv_freq_inplace, clear_rotary_cache
  - attn_hist.py: accumulate_distance_histogram, fit_power_law
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path so we can import scripts.lib.rope
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from lib.rope.learnable_evq import (
    LearnableEVQRoPE,
    EVQRoPEWrapper,
    TauLogger,
    _inverse_softplus,
    estimate_tau_from_distance_prior,
    measure_distance_distribution,
    setup_optimizer_with_tau,
)
from lib.rope.schedules import (
    build_inv_freq,
    canonical_method,
    geometric_inv_freq,
    infer_shape_name,
    METHOD_ALIASES,
)
from lib.rope.inject import (
    apply_inv_freq_inplace,
    clear_rotary_cache,
    find_rotary_modules_with_inv_freq,
    hash_tensor_sha256,
)
from lib.rope.attn_hist import (
    accumulate_distance_histogram,
    fit_power_law,
    bootstrap_alpha_ci,
)


# ============================================================
# Helper: standalone EVQ formula (numpy reference implementation)
# ============================================================

def evq_cosh_inv_freq_numpy(head_dim: int, tau: float, base: float = 500000.0):
    """Reference implementation in numpy for cross-validation."""
    K = head_dim // 2
    idx = np.arange(K, dtype=np.float64)
    u = (idx + 0.5) / float(K)  # midpoint quantization
    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u) * np.sinh(tau))
    return np.float_power(float(base), -phi)


# ============================================================
# Tests: learnable_evq.py
# ============================================================

class TestEVQFrequencyMath:
    """Test the mathematical correctness of EVQ-Cosh frequency computation."""

    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    @pytest.mark.parametrize("tau", [0.5, 1.0, 1.414, 2.0, 3.0])
    def test_evq_matches_numpy_reference(self, head_dim, tau):
        """EVQ frequencies must match the numpy reference implementation."""
        rope = LearnableEVQRoPE(dim=head_dim, tau_init=tau, base=500000.0)
        freqs_torch = rope.get_frequencies().detach().cpu().numpy()
        freqs_numpy = evq_cosh_inv_freq_numpy(head_dim, tau, base=500000.0)
        np.testing.assert_allclose(freqs_torch, freqs_numpy, rtol=1e-6, atol=1e-10,
                                   err_msg=f"Mismatch at head_dim={head_dim}, tau={tau}")

    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_tau_zero_recovers_geometric(self, head_dim):
        """τ→0 must recover standard geometric RoPE frequencies."""
        rope = LearnableEVQRoPE(dim=head_dim, tau_init=1e-6, base=10000.0)
        freqs_evq = rope.get_frequencies().detach()

        # Standard geometric: ω_k = base^{-(k+0.5)/K}  (midpoint)
        K = head_dim // 2
        u = (torch.arange(K, dtype=torch.float64) + 0.5) / K
        freqs_geo = 10000.0 ** (-u)

        diff = (freqs_evq - freqs_geo).abs().max().item()
        assert diff < 1e-6, f"τ→0 recovery failed: max diff = {diff:.2e}"

    @pytest.mark.parametrize("tau", [0.5, 1.0, 1.414, 2.0, 5.0])
    def test_phi_monotonically_increasing(self, tau):
        """φ_k must be strictly monotonically increasing (frequencies decrease)."""
        rope = LearnableEVQRoPE(dim=64, tau_init=tau, base=500000.0)
        phi = rope.get_phi_schedule().detach()
        diffs = phi[1:] - phi[:-1]
        assert (diffs > 0).all(), f"φ not monotonic at tau={tau}: min diff = {diffs.min().item():.2e}"

    @pytest.mark.parametrize("tau", [0.5, 1.0, 1.414, 2.0, 5.0])
    def test_phi_in_unit_interval(self, tau):
        """φ_k must lie in [0, 1] for all k."""
        rope = LearnableEVQRoPE(dim=64, tau_init=tau, base=500000.0)
        phi = rope.get_phi_schedule().detach()
        assert phi.min().item() >= -1e-10, f"φ below 0: {phi.min().item()}"
        assert phi.max().item() <= 1.0 + 1e-10, f"φ above 1: {phi.max().item()}"

    @pytest.mark.parametrize("tau", [0.5, 1.0, 2.0, 5.0])
    def test_boundary_anchoring(self, tau):
        """Endpoints φ_0 ≈ 0 and φ_{K-1} ≈ 1 (boundary anchoring)."""
        rope = LearnableEVQRoPE(dim=64, tau_init=tau, base=500000.0)
        phi = rope.get_phi_schedule().detach()
        # Midpoint quantization means φ_0 is not exactly 0, but close
        # For u_0 = 0.5/K, φ_0 should be small
        assert phi[0].item() < 0.1, f"φ_0 too large: {phi[0].item()}"
        # φ_{K-1} should be close to 1 (for large τ, midpoint u_{K-1} < 1 pulls it down)
        assert phi[-1].item() > 0.7, f"φ_{{K-1}} too small: {phi[-1].item()}"

    @pytest.mark.parametrize("tau", [0.01, 0.1, 1.0, 5.0, 10.0])
    def test_frequencies_all_positive(self, tau):
        """All frequencies must be strictly positive."""
        rope = LearnableEVQRoPE(dim=128, tau_init=tau, base=500000.0)
        freqs = rope.get_frequencies().detach()
        assert (freqs > 0).all(), f"Non-positive frequency at tau={tau}"

    def test_frequencies_decreasing(self):
        """Frequencies ω_k = base^{-φ_k} must be decreasing (since φ is increasing)."""
        rope = LearnableEVQRoPE(dim=64, tau_init=1.414, base=500000.0)
        freqs = rope.get_frequencies().detach()
        diffs = freqs[1:] - freqs[:-1]
        assert (diffs < 0).all(), "Frequencies not monotonically decreasing"


class TestEVQNumericalStability:
    """Test numerical stability at extreme τ values."""

    @pytest.mark.parametrize("tau", [0.001, 1e-5, 1e-8])
    def test_small_tau_no_nan(self, tau):
        """Small τ (Taylor branch) must not produce NaN/Inf."""
        rope = LearnableEVQRoPE(dim=64, tau_init=tau, base=500000.0)
        freqs = rope.get_frequencies().detach()
        assert torch.isfinite(freqs).all(), f"Non-finite freq at tau={tau}"

    @pytest.mark.parametrize("tau", [5.0, 10.0, 15.0, 20.0])
    def test_large_tau_no_overflow(self, tau):
        """Large τ must not produce NaN/Inf (sinh overflow risk)."""
        rope = LearnableEVQRoPE(dim=64, tau_init=tau, base=500000.0)
        freqs = rope.get_frequencies().detach()
        assert torch.isfinite(freqs).all(), f"Non-finite freq at tau={tau}"

    def test_taylor_full_continuity_at_boundary(self):
        """φ must be continuous at the Taylor↔full transition (τ=1e-4)."""
        rope_lo = LearnableEVQRoPE(dim=128, tau_init=9e-5)
        rope_hi = LearnableEVQRoPE(dim=128, tau_init=1.1e-4)
        phi_lo = rope_lo._compute_phi(rope_lo.tau).detach()
        phi_hi = rope_hi._compute_phi(rope_hi.tau).detach()
        max_diff = (phi_lo - phi_hi).abs().max().item()
        assert max_diff < 1e-6, f"Taylor-full discontinuity: {max_diff:.2e}"


class TestEVQGradientFlow:
    """Test that gradients flow correctly through the EVQ computation."""

    def test_gradient_exists_normal_tau(self):
        """Gradient ∂L/∂ψ must be nonzero at normal τ."""
        rope = LearnableEVQRoPE(dim=64, tau_init=1.0, base=500000.0)
        freqs = rope.get_frequencies()
        freqs.sum().backward()
        assert rope.raw_tau.grad is not None
        assert abs(rope.raw_tau.grad.item()) > 0, "Zero gradient at tau=1.0"

    def test_gradient_exists_small_tau(self):
        """Gradient must flow through Taylor branch (τ < 1e-4)."""
        rope = LearnableEVQRoPE(dim=64, tau_init=1e-5, base=500000.0)
        freqs = rope.get_frequencies()
        freqs.sum().backward()
        assert rope.raw_tau.grad is not None
        assert abs(rope.raw_tau.grad.item()) > 0, "Zero gradient at small tau (Taylor branch)"

    def test_gradcheck_finite_differences(self):
        """Autograd gradcheck: analytic gradient must match finite differences."""
        n = 8
        u = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        A = 1.0 - u

        def freq_fn(psi):
            tau = F.softplus(psi)
            sinh_tau = torch.sinh(tau)
            phi = 1.0 - (1.0 / tau) * torch.arcsinh(A * sinh_tau)
            return torch.pow(10000.0, -phi)

        psi = torch.tensor(_inverse_softplus(1.0), dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(freq_fn, psi, eps=1e-6)

    def test_softplus_ensures_positive_tau(self):
        """τ = softplus(ψ) must always be positive regardless of ψ."""
        for psi_val in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]:
            rope = LearnableEVQRoPE(dim=64, tau_init=0.5)
            rope.raw_tau.data.fill_(psi_val)
            assert rope.tau.item() > 0, f"τ not positive for ψ={psi_val}"


class TestLearnableEVQRoPEModule:
    """Test the LearnableEVQRoPE nn.Module interface."""

    def test_single_parameter(self):
        """Module should have exactly 1 learnable parameter (raw_tau)."""
        rope = LearnableEVQRoPE(dim=64, max_seq_len=4096)
        params = list(rope.parameters())
        assert len(params) == 1
        assert params[0].numel() == 1

    def test_forward_shape(self):
        """forward() output shape should be (seq_len, n_freqs)."""
        rope = LearnableEVQRoPE(dim=128, max_seq_len=4096)
        angles = rope(seq_len=512)
        assert angles.shape == (512, 64)

    def test_cos_sin_shape(self):
        """get_cos_sin() should return matching shapes."""
        rope = LearnableEVQRoPE(dim=64, max_seq_len=2048)
        cos_e, sin_e = rope.get_cos_sin(seq_len=256)
        assert cos_e.shape == (256, 32)
        assert sin_e.shape == (256, 32)

    def test_cos_sin_identity(self):
        """cos²θ + sin²θ = 1 for all angles."""
        rope = LearnableEVQRoPE(dim=64, max_seq_len=1024, tau_init=1.414)
        cos_e, sin_e = rope.get_cos_sin(seq_len=128)
        identity = cos_e**2 + sin_e**2
        torch.testing.assert_close(identity, torch.ones_like(identity), atol=1e-6, rtol=1e-6)

    def test_extra_repr(self):
        """extra_repr should include key parameters."""
        rope = LearnableEVQRoPE(dim=64, max_seq_len=4096, base=500000.0, tau_init=1.5)
        repr_str = rope.extra_repr()
        assert "dim=64" in repr_str
        assert "n_freqs=32" in repr_str
        assert "base=500000.0" in repr_str

    def test_get_tau_value_matches_property(self):
        """get_tau_value() should match tau property."""
        rope = LearnableEVQRoPE(dim=64, tau_init=2.0)
        assert abs(rope.get_tau_value() - rope.tau.item()) < 1e-10


class TestEVQRoPEWrapper:
    """Test the drop-in EVQRoPEWrapper."""

    def test_wrapper_forward(self):
        """Wrapper forward should return (cos, sin) tuple."""
        wrapper = EVQRoPEWrapper(head_dim=64, max_seq_len=2048, tau_init=1.0)
        cos_e, sin_e = wrapper(seq_len=128)
        assert cos_e.shape == (128, 32)
        assert sin_e.shape == (128, 32)

    def test_wrapper_tau_property(self):
        """Wrapper should expose tau as a float property."""
        wrapper = EVQRoPEWrapper(head_dim=64, tau_init=1.5)
        assert isinstance(wrapper.tau, float)
        assert abs(wrapper.tau - 1.5) < 0.01


class TestInverseSoftplus:
    """Test the _inverse_softplus utility."""

    @pytest.mark.parametrize("x", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_roundtrip(self, x):
        """softplus(inverse_softplus(x)) == x."""
        psi = _inverse_softplus(x)
        recovered = F.softplus(torch.tensor(psi)).item()
        assert abs(recovered - x) < 1e-6, f"Roundtrip failed: {x} -> {psi} -> {recovered}"


class TestTauLogger:
    """Test the TauLogger utility."""

    def test_log_and_save(self, tmp_path):
        """TauLogger should record entries and save to JSON."""
        rope = LearnableEVQRoPE(dim=64, tau_init=1.0)
        logger = TauLogger(log_interval=1)
        logger.log(0, rope, loss=2.5)
        logger.log(1, rope, loss=2.3)

        assert len(logger.trajectory) == 2
        assert logger.trajectory[0]["step"] == 0
        assert abs(logger.trajectory[0]["loss"] - 2.5) < 1e-6

        path = tmp_path / "tau_log.json"
        logger.save(str(path))
        assert path.exists()

    def test_get_final_tau(self):
        """get_final_tau should return last logged tau."""
        rope = LearnableEVQRoPE(dim=64, tau_init=1.5)
        logger = TauLogger(log_interval=1)
        logger.log(0, rope)
        assert abs(logger.get_final_tau() - 1.5) < 0.01

    def test_empty_trajectory(self):
        """get_final_tau on empty trajectory should return nan."""
        logger = TauLogger()
        assert math.isnan(logger.get_final_tau())

    def test_convergence_std(self):
        """get_convergence_std should return a finite value."""
        rope = LearnableEVQRoPE(dim=64, tau_init=1.0)
        logger = TauLogger(log_interval=1)
        for i in range(20):
            logger.log(i, rope)
        std = logger.get_convergence_std()
        assert not math.isnan(std)
        # All same tau, std should be ~0
        assert std < 0.01


class TestSetupOptimizer:
    """Test the optimizer setup utility."""

    def test_separate_lr_groups(self):
        """setup_optimizer_with_tau should create 2 param groups."""
        model = nn.Sequential(
            nn.Linear(64, 64),
            EVQRoPEWrapper(head_dim=64, tau_init=1.0),
        )
        opt = setup_optimizer_with_tau(model, base_lr=1e-4, tau_lr_multiplier=10.0)
        assert len(opt.param_groups) == 2
        # tau group should have higher lr
        tau_group = opt.param_groups[1]
        assert tau_group["lr"] == 1e-3  # 1e-4 * 10
        assert tau_group["weight_decay"] == 0.0


class TestAlgorithm1:
    """Test Algorithm 1: D(Δ) → τ* estimation."""

    def test_power_law_prior(self):
        """Power-law D(Δ) should yield a finite positive τ*."""
        deltas = torch.arange(1, 2049, dtype=torch.float64)
        D = deltas ** (-1.5)
        D = D / D.sum()
        tau, alpha, beta, residual = estimate_tau_from_distance_prior(D)
        assert tau > 0, f"τ* should be positive, got {tau}"
        assert math.isfinite(tau), f"τ* should be finite, got {tau}"
        assert residual < 1.0, f"Residual too large: {residual}"

    def test_uniform_prior_finite_tau(self):
        """Uniform D(Δ) should yield a finite positive τ*."""
        D = torch.ones(2048, dtype=torch.float64) / 2048
        tau, _, _, _ = estimate_tau_from_distance_prior(D)
        assert math.isfinite(tau), f"Uniform prior τ* not finite: {tau}"
        assert tau > 0, f"Uniform prior τ* not positive: {tau}"


class TestMeasureDistanceDistribution:
    """Test the distance distribution measurement."""

    def test_output_shape(self):
        """Output should have shape (max_delta,)."""
        tokens = torch.randint(0, 100, (10000,))
        D = measure_distance_distribution(tokens, max_delta=256, sample_size=100)
        assert D.shape == (256,)

    def test_sums_to_one(self):
        """Output should be normalized to sum to 1."""
        tokens = torch.randint(0, 100, (10000,))
        D = measure_distance_distribution(tokens, max_delta=256, sample_size=500)
        assert abs(D.sum().item() - 1.0) < 1e-6

    def test_non_negative(self):
        """All entries should be non-negative."""
        tokens = torch.randint(0, 100, (10000,))
        D = measure_distance_distribution(tokens, max_delta=256, sample_size=100)
        assert (D >= 0).all()


# ============================================================
# Tests: schedules.py
# ============================================================

class TestCanonicalMethod:
    """Test method alias resolution."""

    def test_all_aliases_resolve(self):
        """Every alias in METHOD_ALIASES should resolve without error."""
        for alias in METHOD_ALIASES:
            result = canonical_method(alias)
            assert isinstance(result, str)

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            canonical_method("nonexistent_method")

    def test_evq_cosh_aliases(self):
        """evq_cosh, exact_cosh, cosh should all resolve to evq_cosh."""
        assert canonical_method("evq_cosh") == "evq_cosh"
        assert canonical_method("exact_cosh") == "evq_cosh"
        assert canonical_method("cosh") == "evq_cosh"

    def test_case_insensitive(self):
        """Method resolution should be case-insensitive."""
        assert canonical_method("Baseline") == "baseline"
        assert canonical_method("YARN") == "yarn"


class TestGeometricInvFreq:
    """Test geometric (standard RoPE) frequency computation."""

    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_output_shape(self, head_dim):
        """Output should have K = head_dim // 2 elements."""
        inv_freq = geometric_inv_freq(head_dim, base=10000.0)
        assert inv_freq.shape == (head_dim // 2,)

    def test_first_element(self):
        """First element should be base^0 = 1.0."""
        inv_freq = geometric_inv_freq(64, base=10000.0)
        # ω_0 = 1 / (base^(0/64)) = 1.0
        assert abs(inv_freq[0].item() - 1.0) < 1e-10

    def test_last_element(self):
        """Last element should be base^{-(d-2)/d}."""
        inv_freq = geometric_inv_freq(64, base=10000.0)
        expected = 1.0 / (10000.0 ** (62.0 / 64.0))
        assert abs(inv_freq[-1].item() - expected) < 1e-10

    def test_monotonically_decreasing(self):
        """Geometric frequencies must be monotonically decreasing."""
        inv_freq = geometric_inv_freq(128, base=500000.0)
        diffs = inv_freq[1:] - inv_freq[:-1]
        assert (diffs < 0).all()

    def test_odd_dim_raises(self):
        """Odd head_dim should raise ValueError."""
        with pytest.raises(ValueError, match="even"):
            geometric_inv_freq(63, base=10000.0)

    def test_all_positive(self):
        """All frequencies must be positive."""
        inv_freq = geometric_inv_freq(128, base=500000.0)
        assert (inv_freq > 0).all()


class TestBuildInvFreq:
    """Test build_inv_freq for different methods."""

    @pytest.mark.parametrize("method", [
        "baseline", "pi", "yarn", "sigmoid", "anchored_sigmoid",
        "evq_cosh", "evq_exp", "anchored_hybrid",
    ])
    def test_all_methods_return_correct_shape(self, method):
        """All methods should return K frequencies."""
        inv_freq = build_inv_freq(method, head_dim=64, base=500000.0, max_seq_len=8192)
        assert inv_freq.shape == (32,)

    @pytest.mark.parametrize("method", [
        "baseline", "pi", "yarn", "sigmoid", "anchored_sigmoid",
        "evq_cosh", "evq_exp",
    ])
    def test_all_positive(self, method):
        """All methods should produce positive frequencies."""
        inv_freq = build_inv_freq(method, head_dim=64, base=500000.0, max_seq_len=8192)
        assert (inv_freq > 0).all(), f"{method} produced non-positive frequencies"

    def test_baseline_matches_geometric(self):
        """Baseline method should match geometric_inv_freq exactly."""
        inv_freq_build = build_inv_freq("baseline", head_dim=64, base=500000.0, max_seq_len=8192)
        inv_freq_geo = geometric_inv_freq(64, base=500000.0)
        torch.testing.assert_close(inv_freq_build, inv_freq_geo)

    def test_pi_scales_down(self):
        """PI method should produce lower frequencies (divided by scale)."""
        inv_base = build_inv_freq("baseline", head_dim=64, base=500000.0, max_seq_len=8192)
        inv_pi = build_inv_freq("pi", head_dim=64, base=500000.0, max_seq_len=16384)
        # PI divides by scale (16384/8192 = 2), so PI freqs should be lower
        assert (inv_pi <= inv_base + 1e-10).all()


class TestInferShapeName:
    """Test shape name inference."""

    def test_baseline_to_geometric(self):
        assert infer_shape_name("baseline") == "geometric"

    def test_evq_cosh(self):
        assert infer_shape_name("evq_cosh") == "evq_cosh"

    def test_yarn(self):
        assert infer_shape_name("yarn") == "yarn"


# ============================================================
# Tests: inject.py
# ============================================================

class _FakeRotaryModule(nn.Module):
    """Fake rotary module with inv_freq buffer for testing injection."""

    def __init__(self, n_freqs: int):
        super().__init__()
        self.register_buffer("inv_freq", torch.ones(n_freqs))
        self.cos_cached = torch.zeros(10)
        self.sin_cached = torch.zeros(10)
        self.max_seq_len_cached = 1024


class _FakeModel(nn.Module):
    """Fake model containing rotary modules."""

    def __init__(self, n_layers: int = 2, n_freqs: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([_FakeRotaryModule(n_freqs) for _ in range(n_layers)])


class TestFindRotaryModules:
    """Test finding rotary modules in a model."""

    def test_finds_modules_with_inv_freq(self):
        model = _FakeModel(n_layers=3, n_freqs=32)
        modules = find_rotary_modules_with_inv_freq(model)
        assert len(modules) == 3

    def test_empty_model(self):
        model = nn.Linear(64, 64)
        modules = find_rotary_modules_with_inv_freq(model)
        assert len(modules) == 0

    def test_returns_name_and_module(self):
        model = _FakeModel(n_layers=1)
        modules = find_rotary_modules_with_inv_freq(model)
        name, mod = modules[0]
        assert isinstance(name, str)
        assert isinstance(mod, nn.Module)
        assert hasattr(mod, "inv_freq")


class TestClearRotaryCache:
    """Test cache clearing for rotary modules."""

    def test_clears_cached_attrs(self):
        mod = _FakeRotaryModule(32)
        assert mod.cos_cached is not None
        assert mod.sin_cached is not None
        assert mod.max_seq_len_cached == 1024

        clear_rotary_cache(mod)

        assert mod.cos_cached is None
        assert mod.sin_cached is None
        assert mod.max_seq_len_cached == 0


class TestApplyInvFreqInplace:
    """Test in-place frequency injection."""

    def test_changes_inv_freq(self):
        model = _FakeModel(n_layers=2, n_freqs=32)
        new_freqs = torch.randn(32)
        result = apply_inv_freq_inplace(model, new_freqs)
        assert result["patched_count"] == 2
        assert len(result["changed_modules"]) == 2

        # Verify values were actually changed
        for _, mod in find_rotary_modules_with_inv_freq(model):
            torch.testing.assert_close(
                mod.inv_freq.cpu().float(),
                new_freqs.cpu().float(),
            )

    def test_shape_mismatch_raises(self):
        model = _FakeModel(n_layers=1, n_freqs=32)
        wrong_freqs = torch.randn(16)  # wrong size
        with pytest.raises(RuntimeError, match="Shape mismatch"):
            apply_inv_freq_inplace(model, wrong_freqs)

    def test_no_rotary_raises(self):
        model = nn.Linear(64, 64)
        with pytest.raises(RuntimeError, match="No rotary modules"):
            apply_inv_freq_inplace(model, torch.randn(32))

    def test_clears_cache_after_injection(self):
        model = _FakeModel(n_layers=1, n_freqs=32)
        new_freqs = torch.randn(32)
        apply_inv_freq_inplace(model, new_freqs)

        mod = list(model.modules())
        for m in mod:
            if hasattr(m, "cos_cached"):
                assert m.cos_cached is None
            if hasattr(m, "max_seq_len_cached"):
                assert m.max_seq_len_cached == 0


class TestHashTensor:
    """Test tensor hashing utility."""

    def test_deterministic(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        h1 = hash_tensor_sha256(t)
        h2 = hash_tensor_sha256(t)
        assert h1 == h2

    def test_different_tensors_different_hash(self):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 4.0])
        assert hash_tensor_sha256(t1) != hash_tensor_sha256(t2)

    def test_hash_length(self):
        t = torch.tensor([1.0])
        h = hash_tensor_sha256(t)
        assert len(h) == 64  # SHA-256 hex digest


# ============================================================
# Tests: attn_hist.py
# ============================================================

class TestAccumulateDistanceHistogram:
    """Test attention distance histogram accumulation."""

    def test_output_shape(self):
        """Histogram should have max_distance+1 entries."""
        h, q_len, k_len, d = 2, 8, 16, 32
        q = torch.randn(h, q_len, d)
        k = torch.randn(h, k_len, d)
        positions = torch.arange(q_len)
        max_dist = 20
        hist = torch.zeros(max_dist + 1)
        accumulate_distance_histogram(q, k, positions, max_dist, hist)
        assert hist.shape == (max_dist + 1,)

    def test_non_negative(self):
        """All histogram entries should be non-negative."""
        h, q_len, k_len, d = 2, 8, 16, 32
        q = torch.randn(h, q_len, d)
        k = torch.randn(h, k_len, d)
        positions = torch.arange(q_len)
        max_dist = 20
        hist = torch.zeros(max_dist + 1)
        accumulate_distance_histogram(q, k, positions, max_dist, hist)
        assert (hist >= 0).all()

    def test_rank_check(self):
        """Should raise ValueError for wrong rank inputs."""
        with pytest.raises(ValueError, match="rank-3"):
            accumulate_distance_histogram(
                torch.randn(8, 32),  # 2D, should be 3D
                torch.randn(2, 8, 32),
                torch.arange(8),
                10,
                torch.zeros(11),
            )


class TestFitPowerLaw:
    """Test power-law fitting."""

    def test_known_power_law(self):
        """Fitting a known power-law should recover the exponent."""
        n = 500
        x = np.arange(1, n + 1, dtype=np.float64)
        alpha_true = 1.5
        hist = x ** (-alpha_true) + np.random.RandomState(42).normal(0, 1e-6, n)
        result = fit_power_law(hist, d_min=8, d_max=400)
        assert result["alpha"] is not None
        assert abs(result["alpha"] - alpha_true) < 0.1, f"Got alpha={result['alpha']}, expected ~{alpha_true}"
        assert result["r2"] > 0.95

    def test_too_few_points(self):
        """Should return None alpha when too few valid points."""
        hist = np.zeros(10)
        result = fit_power_law(hist, d_min=8, d_max=9)
        assert result["alpha"] is None

    def test_1d_check(self):
        """Should raise ValueError for non-1D input."""
        with pytest.raises(ValueError, match="1D"):
            fit_power_law(np.zeros((10, 2)))


class TestBootstrapAlphaCI:
    """Test bootstrap confidence interval estimation."""

    def test_single_sample(self):
        """Single sample should return point estimate."""
        hist = np.arange(1, 501, dtype=np.float64) ** (-1.5)
        result = bootstrap_alpha_ci([hist], n_bootstrap=10, d_min=8)
        assert result["alpha_mean"] is not None

    def test_empty_input(self):
        """Empty input should return None values."""
        result = bootstrap_alpha_ci([], n_bootstrap=10)
        assert result["alpha_mean"] is None


# ============================================================
# Cross-validation: EVQ formula matches AIHANDOFF.md specification
# ============================================================

class TestAIHandoffFormula:
    """Verify EVQ implementation matches the formula in AIHANDOFF.md exactly.

    From AIHANDOFF.md Part 2:
        K = head_dim // 2
        idx = torch.arange(K, dtype=torch.float64)
        u = (idx + 0.5) / float(K)                  # midpoint
        phi = 1.0 - (1.0/tau) * torch.arcsinh((1.0-u) * math.sinh(tau))
        inv_freq = torch.pow(base, -phi).float()
    """

    @pytest.mark.parametrize("head_dim,tau,base", [
        (64, 1.414, 500000.0),   # Phase 18 setting
        (64, 1.0, 10000.0),      # Standard base
        (128, 2.0, 500000.0),    # Large dim
        (32, 0.5, 500000.0),     # Small dim
    ])
    def test_matches_aihandoff_formula(self, head_dim, tau, base):
        """Implementation must match AIHANDOFF.md formula exactly."""
        # AIHANDOFF.md formula (reference)
        K = head_dim // 2
        idx = torch.arange(K, dtype=torch.float64)
        u = (idx + 0.5) / float(K)
        phi_ref = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
        inv_freq_ref = torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi_ref).float()

        # Implementation
        rope = LearnableEVQRoPE(dim=head_dim, tau_init=tau, base=base)
        inv_freq_impl = rope.get_frequencies().detach().float()

        torch.testing.assert_close(
            inv_freq_impl, inv_freq_ref,
            atol=1e-5, rtol=1e-5,
            msg=f"Mismatch at head_dim={head_dim}, tau={tau}, base={base}",
        )


# ============================================================
# Integration: EVQ vs GEO at known tau values
# ============================================================

class TestEVQvsGEO:
    """Verify that EVQ with specific tau values differs from geometric."""

    def test_tau_1414_differs_from_geometric(self):
        """EVQ(τ=1.414) should produce different frequencies than geometric."""
        rope = LearnableEVQRoPE(dim=64, tau_init=1.414, base=500000.0)
        freqs_evq = rope.get_frequencies().detach()
        freqs_geo = geometric_inv_freq(64, base=500000.0).float()
        diff = (freqs_evq.float() - freqs_geo).abs().max().item()
        assert diff > 0.01, "EVQ(τ=1.414) should differ significantly from geometric"

    def test_small_tau_approaches_geometric(self):
        """EVQ with very small τ should approach geometric."""
        rope = LearnableEVQRoPE(dim=64, tau_init=0.001, base=500000.0)
        freqs_evq = rope.get_frequencies().detach()
        K = 32
        u = (torch.arange(K, dtype=torch.float64) + 0.5) / K
        freqs_geo = 500000.0 ** (-u)
        diff = (freqs_evq - freqs_geo).abs().max().item()
        assert diff < 1e-4, f"Small τ should ≈ geometric, but diff = {diff:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
