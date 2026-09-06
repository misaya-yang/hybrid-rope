#!/usr/bin/env python3
"""Dependency-spectrum allocation audit (CPU-only, numpy-only).

Independent analysis script for the ICML-2027 research audit. It does NOT
touch training code. It tests the candidate chain

    attention dependency demand K(r) -> p(x) -> rho*(x) -> theta_k

and tries hard to FALSIFY it:

Part 1 (Question C): single-channel utility U(theta) = E_{r~K}[1-cos(theta r)].
  For theta = theta* e^{-eps}, is the mismatch regret U(theta*)-U(theta)
  O(eps) or O(eps^2)? Tested per kernel over the GLOBAL RoPE-feasible band
  theta in [b^-1, 1] (b=5e5). Also measures the O(1) gap between the true
  optimum and the single-width theory point theta_th = pi/E[r], and the
  harmonic degeneracy (how many near-global maxima exist).

Part 2 (Question 4): finite spectral budget K in {8,16,32,64}.
  Objective (coverage):  Obj({theta_k}) = E_r[ max_k 1-cos(theta_k r) ]
  Methods compared on IDENTICAL fixed extrema [theta_lo, theta_hi]:
    - geometric (uniform in log-freq, midpoint grid)
    - EVQ-Cosh  (repo formula, swept tau incl. rule tau = 2K/sqrt(r_max))
    - analytic density candidates via inverse-CDF midpoint sampling:
        rho ~ p_dem            (demand matching)
        rho ~ p_dem^{1/3}      (high-rate quantization, quadratic regret)
        rho ~ p_dem^{1/2}      (the candidate rho* ~ sqrt(p c), c const)
    - direct discrete optimum: multi-restart Lloyd-style alternating
      optimization + coordinate ascent (numerical LOWER BOUND on optimum).
  Secondary: the ADDITIVE objective E_r[sum_k 1-cos(theta_k r)] is separable
  in k -> all channels collapse to one point; coverage alone cannot define
  an allocation without an interference/budget term.

Part 3 (Question B): single-width compression loss: allocation built from
  delta(E[r]) vs allocation built from the full scale distribution.

Run:  python3 dependency_spectrum_audit.py
Out:  <repo>/results/dependency_spectrum_audit_20260819/summary.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(20260819)

REPO = Path(__file__).resolve().parents[2]

N_R = 12000          # distance quadrature points
N_THETA = 40000      # dense x-grid for 1-D utility scans
EPS_GRID = np.logspace(-3.0, -0.3, 26)  # eps values for regret order fit
FIT_MASK = EPS_GRID <= 0.05             # asymptotic region for slope fit


def make_r_grid(r_min, r_max, n=N_R):
    r = np.geomspace(max(r_min, 1e-3), r_max, n)
    return r, np.gradient(r)


class Kernel:
    def __init__(self, name, r_min, r_max, pdf, note=""):
        self.name, self.r_min, self.r_max = name, r_min, r_max
        self.pdf, self.note = pdf, note
        if r_max <= r_min * (1 + 1e-9):          # true point mass
            self.r = np.array([r_max])
            self.w = np.array([1.0])
        else:
            r, dr = make_r_grid(r_min, r_max)
            w = pdf(r) * dr
            w = w / w.sum()
            self.r, self.w = r, w
        self.mean_r = float((self.w * self.r).sum())


def kernel_catalog(L=8192.0):
    W = 1024.0
    cat = [
        Kernel("delta_W1024", W, W, lambda r: np.ones_like(r),
               "point mass at W=1024 (interior benchmark)"),
        Kernel("delta_W2", 2.0, 2.0, lambda r: np.ones_like(r),
               "point mass at r=2 < pi -> boundary regime"),
        Kernel("uniform_1_W", 1.0, W, lambda r: np.ones_like(r),
               "uniform distances on [1, W]"),
        Kernel("triangular_W", 0.5 * W, 1.5 * W,
               lambda r: np.clip(1.0 - np.abs(r - W) / (0.5 * W), 0, None),
               "triangular, peak at W"),
        Kernel("gaussian_W", 1.0, 3.0 * W,
               lambda r: np.exp(-0.5 * ((r - W) / (0.3 * W)) ** 2),
               "Gaussian centered at W, sigma=0.3W"),
        Kernel("heavy_tail", 1.0, L,
               lambda r: 1.0 / (r * math.log(L)),
               "repo D(Delta)=1/(Delta ln L) power law"),
        Kernel("bimodal", 128.0, 4096.0,
               lambda r: 0.5 * np.exp(-0.5 * ((r - 128.0) / 6.4) ** 2)
               + 0.5 * np.exp(-0.5 * ((r - 4096.0) / 204.8) ** 2),
               "two narrow modes at 128 and 4096"),
        Kernel("local_plus_long", 1.0, 4096.0,
               lambda r: 0.85 * np.exp(-r / 32.0)
               + 0.15 * np.exp(-0.5 * ((r - 4096.0) / 100.0) ** 2),
               "85% local exp(scale 32) + 15% long mode at 4096"),
    ]
    return cat


# ---- EVQ-Cosh warp: exact copy of scripts/lib/rope/schedules.py::evq_cosh_phi

def evq_cosh_phi(K, tau):
    u = (np.arange(K) + 0.5) / K
    if abs(tau) < 1e-8:
        return u
    return 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u) * math.sinh(tau))


def inv_freq_from_phi(phi, theta_lo, theta_hi):
    return theta_hi * (theta_lo / theta_hi) ** np.asarray(phi)


def geometric_grid(K, theta_lo, theta_hi):
    return inv_freq_from_phi((np.arange(K) + 0.5) / K, theta_lo, theta_hi)


def evq_cosh_grid(K, tau, theta_lo, theta_hi):
    return inv_freq_from_phi(evq_cosh_phi(K, tau), theta_lo, theta_hi)


def quantile_grid(K, density_x, x_grid, theta_lo, theta_hi):
    """Inverse-CDF midpoint sampling of a density on x = ln(theta)."""
    pdf = np.clip(density_x, 0, None) * np.gradient(x_grid)
    tot = pdf.sum()
    if tot <= 0:
        return geometric_grid(K, theta_lo, theta_hi)
    cdf = np.clip(np.cumsum(pdf) / tot, 0, 1)
    u = (np.arange(K) + 0.5) / K
    return np.exp(np.interp(u, cdf, x_grid))


# ---- objectives

def u_of(theta, r):
    return 1.0 - np.cos(np.asarray(theta) * r)


def coverage(thetas, kern):
    """E_r[max_k 1-cos(theta_k r)]."""
    th = np.atleast_1d(np.asarray(thetas, dtype=float))
    best = np.full(kern.r.shape[0], -np.inf)
    for t in th:
        best = np.maximum(best, u_of(t, kern.r))
    return float((kern.w * best).sum())


def additive(thetas, kern):
    th = np.atleast_1d(np.asarray(thetas, dtype=float))
    return float(sum((kern.w * u_of(t, kern.r)).sum() for t in th))


def golden_max(f, a, b, tol=1e-10, max_iter=200):
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    x = (a + b) / 2
    return x, f(x)


def utility_scan(kern, xs, chunk=400):
    """Vectorized U(theta)=E[1-cos(theta r)] over an x=ln(theta) grid."""
    xs = np.asarray(xs)
    out = np.empty(xs.shape[0])
    for s in range(0, xs.shape[0], chunk):
        th = np.exp(xs[s:s + chunk])[:, None]
        out[s:s + chunk] = 1.0 - (kern.w[None, :] * np.cos(th * kern.r[None, :])).sum(axis=1)
    return out


def best_single_theta(kern, x_lo, x_hi):
    """Global argmax of U over the band: dense scan + local refinement.

    U is OSCILLATORY in x=ln(theta) for broad kernels (maxima whenever
    theta*r is an odd multiple of pi); unimodal search fails on it.
    """
    xs = np.linspace(x_lo, x_hi, N_THETA)
    vals = utility_scan(kern, xs)
    i = int(np.argmax(vals))
    dx = xs[1] - xs[0]
    a, b = max(x_lo, xs[i] - 2 * dx), min(x_hi, xs[i] + 2 * dx)
    x_ref, val_ref = golden_max(
        lambda x: float((kern.w * u_of(math.exp(x), kern.r)).sum()), a, b)
    if val_ref >= vals[i]:
        return math.exp(x_ref), float(val_ref)
    return math.exp(xs[i]), float(vals[i])


def near_optimal_maxima_count(kern, x_lo, x_hi, frac=0.99):
    """Number of local maxima of U within `frac` of the global max: the
    harmonic degeneracy of 'the' optimal frequency."""
    xs = np.linspace(x_lo, x_hi, N_THETA)
    vals = utility_scan(kern, xs)
    vmax = float(vals.max())
    is_max = (vals[1:-1] > vals[:-2]) & (vals[1:-1] >= vals[2:])
    near = is_max & (vals[1:-1] >= frac * vmax)
    return int(near.sum()), vmax


# ---- Part 1

def regret_order(kern, x_lo, x_hi):
    theta_star, u_star = best_single_theta(kern, x_lo, x_hi)
    at_hi = theta_star >= math.exp(x_hi) * (1 - 1e-9)
    at_lo = theta_star <= math.exp(x_lo) * (1 + 1e-9)
    regs = np.array([u_star - float((kern.w * u_of(theta_star * math.exp(-e),
                                                    kern.r)).sum())
                     for e in EPS_GRID])
    slope, _ = np.polyfit(np.log(EPS_GRID[FIT_MASK]),
                          np.log(np.maximum(regs[FIT_MASK], 1e-16)), 1)
    W = kern.mean_r
    theta_th = math.pi / W
    gap_th = u_star - float((kern.w * u_of(theta_th, kern.r)).sum())
    n_max, _ = near_optimal_maxima_count(kern, x_lo, x_hi)
    return {
        "kernel": kern.name,
        "theta_star": theta_star,
        "boundary_side": "hi" if at_hi else ("lo" if at_lo else "none"),
        "U_theta_star": u_star,
        "regret_slope_eps_to_0": float(slope),
        "regret_order": "O(eps^%.2f)" % slope,
        "theta_theory_pi_over_Er": theta_th,
        "theta_theory_feasible": bool(math.exp(x_lo) <= theta_th <= math.exp(x_hi)),
        "gap_theory_vs_optimum_O1": float(gap_th),
        "n_near_global_maxima_99pct": n_max,
    }


# ---- Part 2

def opt_core(kern, n=2400):
    """Subsampled (r,w) used INSIDE optimization loops; final coverage is
    always measured on the full kernel grid."""
    if kern.r.size <= n:
        return kern.r, kern.w
    idx = np.linspace(0, kern.r.size - 1, n).astype(int)
    w = kern.w[idx]
    return kern.r[idx], w / w.sum()


def lloyd_max_coverage(K, kern, x_lo, x_hi, init_x, iters=40):
    """Alternating assign/optimize ascent on E[max_k u]."""
    x = np.array(sorted(init_x), dtype=float)
    r, w = opt_core(kern)
    prev = -np.inf
    for _ in range(iters):
        U = u_of(np.exp(x)[:, None], r)
        assign = np.argmax(U, axis=0)
        for k in range(K):
            mask = assign == k
            if not mask.any():
                best_now = U.max(axis=0)
                j = int(np.argmax(w * (2.0 - best_now)))
                x[k] = min(max(math.log(math.pi / r[j]), x_lo), x_hi)
                U = u_of(np.exp(x)[:, None], r)
                assign = np.argmax(U, axis=0)
                mask = assign == k
                if not mask.any():
                    continue
            wk, rk = w[mask], r[mask]
            xk, _ = golden_max(lambda xx: float((wk * u_of(math.exp(xx), rk)).sum()),
                               x_lo, x_hi, max_iter=120)
            x[k] = xk
            U = u_of(np.exp(x)[:, None], r)
            assign = np.argmax(U, axis=0)
        cur = float((w * U.max(axis=0)).sum())
        if abs(cur - prev) < 1e-11:
            break
        prev = cur
    for k in range(K):  # one coordinate-ascent polish pass
        def obj(xx):
            xt = x.copy()
            xt[k] = xx
            best = np.full(r.shape[0], -np.inf)
            for j in range(K):
                best = np.maximum(best, u_of(math.exp(xt[j]), r))
            return float((w * best).sum())
        x[k], _ = golden_max(obj, x_lo, x_hi, max_iter=120)
    return np.exp(x), coverage(np.exp(x), kern)


def density_candidates(kern, K, x_lo, x_hi):
    """Demand density on log-frequency via matching theta = pi/r.

    x = ln theta; matched x = ln(pi) - ln r (distribution of ln r shifted).
    Powers tested: 1 (matching), 1/3 (high-rate quadratic regret),
    1/2 (the rho* ~ sqrt(p c) candidate with constant c).
    """
    x_dem = math.log(math.pi) - np.log(kern.r)
    order = np.argsort(x_dem)
    x_dem, w_sorted = x_dem[order], kern.w[order]
    x_grid = np.linspace(x_lo, x_hi, 4001)
    cdf_dem = np.interp(x_grid, x_dem, np.cumsum(w_sorted))
    p_dem = np.clip(np.gradient(np.clip(cdf_dem, 0, 1), x_grid), 0, None)
    out = {}
    for label, power in (("matching_p1", 1.0), ("highrate_p13", 1.0 / 3.0),
                         ("sqrt_p12", 0.5)):
        th = quantile_grid(K, p_dem ** power, x_grid,
                           math.exp(x_lo), math.exp(x_hi))
        out[label] = coverage(th, kern)
    return out


def finite_K_block(kern, K_list=(8, 16, 32, 64), n_restarts=6):
    if kern.r_max < math.pi * 1.01:
        return {"skipped": "r_max < pi: no feasible resolving band"}
    x_lo = math.log(math.pi / kern.r_max)
    x_hi = 0.0
    out = {}
    for K in K_list:
        tau_rule = 2.0 * K / math.sqrt(kern.r_max)  # repo rule d_head/sqrt(L)
        block = {}
        block["geometric"] = coverage(
            geometric_grid(K, math.exp(x_lo), math.exp(x_hi)), kern)
        evq = {}
        for tau in [0.5, 1.0, 1.414, 2.0, 3.0, 4.0, 6.0, tau_rule]:
            evq[round(float(tau), 3)] = coverage(
                evq_cosh_grid(K, float(tau), math.exp(x_lo), math.exp(x_hi)), kern)
        best_tau = max(evq, key=lambda t: evq[t])
        block["evq_cosh_best_tau"] = best_tau
        block["evq_cosh_best"] = evq[best_tau]
        block["evq_cosh_rule_tau"] = round(float(tau_rule), 3)
        block["evq_cosh_rule"] = evq[round(float(tau_rule), 3)]
        block.update(density_candidates(kern, K, x_lo, x_hi))
        starts = [np.log(geometric_grid(K, math.exp(x_lo), math.exp(x_hi)))]
        starts.append(np.log(evq_cosh_grid(K, 2.0, math.exp(x_lo), math.exp(x_hi))))
        for _ in range(n_restarts):
            starts.append(np.sort(RNG.uniform(x_lo, x_hi, K)))
        best_val = -np.inf
        for s in starts:
            _, val = lloyd_max_coverage(K, kern, x_lo, x_hi, s)
            if val > best_val:
                best_val = val
        block["direct_optimum"] = best_val
        block["gap"] = {m: best_val - block[m] for m in
                        ("geometric", "evq_cosh_best", "matching_p1",
                         "highrate_p13", "sqrt_p12")}
        block["rel_gap_sqrt"] = block["gap"]["sqrt_p12"] / max(best_val, 1e-12)
        out[K] = block
    return out


def additive_collapse_demo(kern, K=16):
    """Additive objective is separable: optimum = K copies of one theta."""
    if kern.r_max < math.pi * 1.01:
        return {"skipped": "r_max < pi"}
    x_lo = math.log(math.pi / kern.r_max)
    theta_star, _ = best_single_theta(kern, x_lo, 0.0)
    return {"K_copies_of_single_theta": additive(np.full(K, theta_star), kern),
            "geometric_grid_additive":
                additive(geometric_grid(K, math.exp(x_lo), 1.0), kern),
            "note": "additive objective prefers collapse; coverage alone "
                    "cannot define an allocation without interference term"}


def single_width_compression(kern, K=32):
    """Q3B: allocate as if the kernel were delta(E[r]) vs full distribution."""
    if kern.r_max < math.pi * 1.01:
        return {"skipped": "r_max < pi"}
    x_lo = math.log(math.pi / kern.r_max)
    _, val_full = lloyd_max_coverage(
        K, kern, x_lo, 0.0,
        np.log(geometric_grid(K, math.exp(x_lo), 1.0)))
    W = kern.mean_r
    th_sur = geometric_grid(K, math.exp(x_lo), 1.0)
    j = int(np.argmin(np.abs(np.log(th_sur) - math.log(math.pi / W))))
    th_sur[j] = min(max(math.pi / W, math.exp(x_lo)), 1.0)
    val_sur = coverage(th_sur, kern)
    return {"full_distribution": val_full,
            "single_width_delta_Er": val_sur,
            "compression_loss": val_full - val_sur}


# ----

def main():
    kernels = kernel_catalog()
    report = {"part1_regret_order": [], "part2_finite_K": {},
              "part2b_additive_collapse": {},
              "part3_single_width_compression": {}}
    # Part 1 uses the GLOBAL RoPE-feasible band theta in [b^-1, 1], b=5e5:
    # the regret-order question is about where theta* lands relative to the
    # true parameterization limits.
    X_LO_GLOBAL = -math.log(500000.0)
    for kern in kernels:
        report["part1_regret_order"].append(regret_order(kern, X_LO_GLOBAL, 0.0))
        print(f"[part1] {kern.name:18s} done", flush=True)
    for kern in kernels:
        report["part2_finite_K"][kern.name] = finite_K_block(kern)
        print(f"[part2] {kern.name:18s} done", flush=True)
        report["part2b_additive_collapse"][kern.name] = additive_collapse_demo(kern)
        report["part3_single_width_compression"][kern.name] = \
            single_width_compression(kern)
        print(f"[part3] {kern.name:18s} done", flush=True)

    outdir = REPO / "results/dependency_spectrum_audit_20260819"
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "summary.json").open("w") as f:
        json.dump(report, f, indent=1, default=float)

    print("\n=== PART 1: regret order around theta* (eps -> 0+) ===")
    for row in report["part1_regret_order"]:
        print(f"{row['kernel']:18s} theta*={row['theta_star']:.4g} "
              f"bnd={row['boundary_side']:4s} slope={row['regret_slope_eps_to_0']:+.2f} "
              f"U*={row['U_theta_star']:.3f} nmax99={row['n_near_global_maxima_99pct']:3d} "
              f"| pi/E[r]={row['theta_theory_pi_over_Er']:.3g} "
              f"feas={int(row['theta_theory_feasible'])} "
              f"O1gap={row['gap_theory_vs_optimum_O1']:+.3f}")
    print("\n=== PART 2: coverage gaps vs direct optimum ===")
    for kname, blocks in report["part2_finite_K"].items():
        for K in (8, 16, 32, 64):
            if K not in blocks:
                continue
            b = blocks[K]
            g = b["gap"]
            print(f"{kname:18s} K={K:2d} opt={b['direct_optimum']:.4f} "
                  f"gap[geo={g['geometric']:+.4f} evq={g['evq_cosh_best']:+.4f}"
                  f"(t={b['evq_cosh_best_tau']}) match={g['matching_p1']:+.4f} "
                  f"p13={g['highrate_p13']:+.4f} sqrt={g['sqrt_p12']:+.4f}]")
    print("\n=== PART 2b: additive objective collapse (K=16) ===")
    for kname, row in report["part2b_additive_collapse"].items():
        if "skipped" in row:
            print(f"{kname:18s} skipped ({row['skipped']})")
        else:
            print(f"{kname:18s} K-copies={row['K_copies_of_single_theta']:.4f} "
                  f"geometric={row['geometric_grid_additive']:.4f}")
    print("\n=== PART 3: single-width compression loss (K=32) ===")
    for kname, row in report["part3_single_width_compression"].items():
        if "skipped" in row:
            print(f"{kname:18s} skipped ({row['skipped']})")
        else:
            print(f"{kname:18s} full={row['full_distribution']:.4f} "
                  f"delta(E[r])={row['single_width_delta_Er']:.4f} "
                  f"loss={row['compression_loss']:+.4f}")
    print(f"\nWrote {outdir}/summary.json")


if __name__ == "__main__":
    main()
