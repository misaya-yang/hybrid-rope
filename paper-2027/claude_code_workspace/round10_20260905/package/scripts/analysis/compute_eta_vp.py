"""Compute the η integral for the video DiT training schedule.

Goal: verify the Appendix A4 prediction tau_video / tau_causal = sqrt(eta / 2),
with observed ratio 0.53 -> target eta = 2 * 0.53**2 = 0.5618.

Schedule discovery:
  - scripts/video_temporal/video_dit.py defines RectifiedFlowScheduler
    (x_t = (1-t) x_0 + t * eps, t ~ Uniform[0, 1]).
  - scripts/video_temporal/run_dit_temporal.py uses torch.rand (uniform t).
  -> The paper's "eta_VP" is actually eta_RF for Rectified Flow.
  -> For RF, signal coefficient alpha_t = 1 - t (not sqrt(alpha_cumprod)).

We report:
  (A) Primary: Rectified Flow with uniform w(t), alpha_t = 1 - t
      (this is what the DiT experiments actually trained).
  (B) Hypothetical VP schedules (linear, cosine, EDM) for comparison.

Usage: python scripts/analysis/compute_eta_vp.py
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


OBSERVED_RATIO = 0.53
TARGET_ETA = 2.0 * OBSERVED_RATIO ** 2  # approx 0.5618


def ratio_and_match(eta: float) -> tuple[float, float]:
    pred = float(np.sqrt(eta / 2.0))
    pct = 100.0 * (pred - OBSERVED_RATIO) / OBSERVED_RATIO
    return pred, pct


# --------------------------------------------------------------------------
# (A) Rectified Flow (actual training setup)
# --------------------------------------------------------------------------

def eta_rf(m: int) -> float:
    """Rectified Flow with t ~ U[0,1], alpha_t = 1 - t.

    eta_RF^(m) = E_t[(1-t)^m] / E_t[1] = integral_0^1 (1-t)^m dt = 1/(m+1).
    """
    return 1.0 / (m + 1.0)


# --------------------------------------------------------------------------
# (B) Hypothetical VP schedules
# --------------------------------------------------------------------------

def linear_vp(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
    """Standard DDPM linear beta schedule. Returns alpha_t = sqrt(alpha_cumprod[t])."""
    betas = np.linspace(beta_start, beta_end, T)
    alpha_cumprod = np.cumprod(1.0 - betas)
    alpha_t = np.sqrt(alpha_cumprod)
    w_t = np.ones(T)  # uniform timestep sampling (standard DDPM training)
    return alpha_t, w_t


def cosine_vp(T: int = 1000, s: float = 0.008):
    """Nichol & Dhariwal 2021 cosine schedule."""
    t = np.arange(T + 1) / T
    f = np.cos(((t + s) / (1.0 + s)) * np.pi / 2.0) ** 2
    alpha_cumprod = f / f[0]
    alpha_cumprod = alpha_cumprod[1:]  # T values, one per training step
    # Clip to avoid log(0) / divergence near t=T
    alpha_cumprod = np.clip(alpha_cumprod, 1e-8, 1.0)
    alpha_t = np.sqrt(alpha_cumprod)
    w_t = np.ones(T)
    return alpha_t, w_t


def edm_vp(N: int = 4000, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0):
    """Karras 2022 EDM. alpha_t = 1/sqrt(1+sigma^2); uniform in log-sigma.

    Note: EDM is VP-equivalent via the sigma <-> alpha change of variables.
    We grid over log(sigma) uniformly (Karras's training sampler is log-normal,
    but log-uniform is the simplest canonical reference).
    """
    log_sigma = np.linspace(np.log(sigma_min), np.log(sigma_max), N)
    sigma = np.exp(log_sigma)
    alpha_t = 1.0 / np.sqrt(1.0 + sigma ** 2)
    w_t = np.ones(N)  # uniform in log-sigma
    return alpha_t, w_t


def eta_discrete(alpha_t: np.ndarray, w_t: np.ndarray, m: int) -> float:
    return float(np.sum(w_t * alpha_t ** m) / np.sum(w_t))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_row(m: int, eta: float) -> None:
    pred, pct = ratio_and_match(eta)
    print(f"  m={m}: eta = {eta:.4f}, predicted ratio = {pred:.4f}, "
          f"observed = {OBSERVED_RATIO}, match = {pct:+.1f}%")


def main() -> None:
    print(f"Target: tau_video / tau_causal = {OBSERVED_RATIO} observed")
    print(f"        -> required eta = 2 * {OBSERVED_RATIO}^2 = {TARGET_ETA:.4f}")

    section("(A) Rectified Flow  [ACTUAL training schedule]")
    print("  scripts/video_temporal/video_dit.py:446 RectifiedFlowScheduler")
    print("  x_t = (1-t) x_0 + t eps,  t ~ U[0,1],  alpha_t = 1 - t")
    print("  eta^(m) = integral_0^1 (1-t)^m dt = 1/(m+1)")
    for m in (1, 2, 4):
        print_row(m, eta_rf(m))

    section("(B) Hypothetical VP schedules  [NOT what the DiT was trained on]")

    print("\n  B1. Linear VP (T=1000, beta in [1e-4, 2e-2], uniform w):")
    a, w = linear_vp()
    print(f"      mean alpha = {a.mean():.4f}, min alpha = {a.min():.4e}")
    for m in (2, 4):
        print_row(m, eta_discrete(a, w, m))

    print("\n  B2. Cosine VP (Nichol & Dhariwal 2021, s=0.008, uniform w):")
    a, w = cosine_vp()
    print(f"      mean alpha = {a.mean():.4f}, min alpha = {a.min():.4e}")
    for m in (2, 4):
        print_row(m, eta_discrete(a, w, m))

    print("\n  B3. EDM (Karras 2022, sigma in [0.002, 80], log-uniform w):")
    a, w = edm_vp()
    print(f"      mean alpha = {a.mean():.4f}, min alpha = {a.min():.4e}")
    for m in (2, 4):
        print_row(m, eta_discrete(a, w, m))

    section("Summary")
    print(f"  Observed tau_video/tau_causal = {OBSERVED_RATIO}")
    print(f"  -> target eta = {TARGET_ETA:.4f}")
    print()
    # Find best match across all computed schedules
    candidates = [
        ("RF m=1 (E[alpha])",      eta_rf(1)),
        ("RF m=2 (E[alpha^2])",    eta_rf(2)),
        ("RF m=4 (E[alpha^4])",    eta_rf(4)),
        ("Linear VP m=2",          eta_discrete(*linear_vp(), m=2)),
        ("Linear VP m=4",          eta_discrete(*linear_vp(), m=4)),
        ("Cosine VP m=2",          eta_discrete(*cosine_vp(), m=2)),
        ("Cosine VP m=4",          eta_discrete(*cosine_vp(), m=4)),
        ("EDM log-uniform m=2",    eta_discrete(*edm_vp(), m=2)),
        ("EDM log-uniform m=4",    eta_discrete(*edm_vp(), m=4)),
    ]
    closest = min(candidates, key=lambda x: abs(x[1] - TARGET_ETA))
    print(f"  Closest eta to target:  {closest[0]}  ->  eta = {closest[1]:.4f} "
          f"(gap {100*(closest[1]-TARGET_ETA)/TARGET_ETA:+.1f}%)")
    print()
    print("  Note: RF is the actual training setup. m=1 (linear signal expectation)")
    print("  gives the closest match; m=2 (canonical 'second-moment' reading used")
    print("  in the paper's VP-style formula) undershoots by ~23%; m=4 undershoots")
    print("  by ~40%. If the paper wants eta ~= 0.56, either (i) switch to the")
    print("  E[alpha] / m=1 interpretation, or (ii) revisit whether the RF signal")
    print("  coefficient (1-t) is the right 'alpha' for the relative-position kernel.")


if __name__ == "__main__":
    main()
