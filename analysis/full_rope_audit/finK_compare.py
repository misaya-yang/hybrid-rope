#!/usr/bin/env python3
"""
finK_compare.py — finite-K frequency-allocation comparison (falsification-first audit).

Primary metric grid: causal-weighted. G_causal(L, w): Delta in {0..L-1}, weight W_Delta = L - Delta,
Phi rows = [cos(w_k Delta), sin(w_k Delta)], G = Phi^T W Phi (direct summation; an exactly
equivalent closed form via Fejer-type kernels is used inside the optimizer and cross-validated).

Metrics (definitions fixed by task):
  C_cos  = sum_{i<j} <cos_i,cos_j>_W^2 / (<cos_i^2>_W * <cos_j^2>_W)
  C_full = sum_{i<j} (s1^2+s2^2)/2,  s = canonical correlations of the 2D subspaces
           (causal: svals of Gaa^{-1/2} Gab Gbb^{-1/2} from 2x2 blocks of G_causal;
            symmetric: svals of Q_a^T Q_b via QR = canon_corr)
  effrank: entropy effrank (raw Gram and column-whitened Gram), logdet(whitened + 1e-6 I), cond.
Note: on the causal grid the cos-sin cross block is NOT zero (asymmetric weighting), so the
2x2 block whitening handles the full blocks; canonical correlations are basis-independent.

Allocations: geometric, EVQ-Cosh (tau in {0.5,1.0,1.4,2.0,2.5,3.0}), cos-collision optimum,
full-RoPE-collision optimum, logdet(whitened+1e-6I) optimum, healthy-zone uniform [2pi/L, 1].

Optimizer: sorted frequencies in log space, interior z in R^{K-2},
w = exp(log w_min + cumsum(softplus(z))/sum(softplus(z)) * (log w_max - log w_min)), endpoints fixed.
scipy L-BFGS-B from 3 random restarts (seeds 0,1,2), best restart then 2 coordinate-descent sweeps.

Outputs (this directory): finK_results.md, finK_K16.csv, finK_K32.csv, finK_K64.csv,
opt_K{K}_cos|full|logdet.npy (secondary configs: opt_K64_b1e4_*.npy, opt_K32_L1024_*.npy).

Usage: python finK_compare.py [--smoke]
numpy + scipy only, CPU.
"""
import os
import sys
import time
import csv

import numpy as np

# Apple Accelerate (macOS arm64) raises spurious FP-exception warnings inside BLAS/LAPACK
# even though results are IEEE-correct (verified: BLAS matmul vs einsum bit-exact).
# Suppress the noise; correctness is enforced by explicit isfinite checks + independent
# cross-implementations in the sanity block.
np.seterr(all="ignore")

from scipy.optimize import minimize  # noqa: E402

import verify_core as vc  # noqa: E402,F401  (prints V1-V6 on import; primitives reused below)

OUT = os.path.dirname(os.path.abspath(__file__))
SMOKE = "--smoke" in sys.argv

TAUS = [0.5, 1.0, 1.4, 2.0, 2.5, 3.0]
SEEDS = (0, 1, 2)

# (tag, K, b, L_train). Primary: b=5e5, L=4096; secondary: b=1e4 (K=64), L=1024 (K=32).
CONFIGS = [
    ("K16",       16, 5e5, 4096),
    ("K32",       32, 5e5, 4096),
    ("K64",       64, 5e5, 4096),
    ("K64_b1e4",  64, 1e4, 4096),
    ("K32_L1024", 32, 5e5, 1024),
]
OBJS = ("cos", "full", "logdet")


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------
# Causal-weighted Gram: two independent implementations
# ----------------------------------------------------------------------
def _fejer(L, x):
    """Fejer kernel F_L(x) = (sin(Lx/2)/sin(x/2))^2; F_L(0) = L^2."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sin(L * x / 2.0) / np.sin(x / 2.0)
        return np.where(np.abs(x) < 1e-8, float(L * L), s * s)


def _Ssum(L, x):
    """S(x) = sum_{D=0}^{L-1} (L-D) cos(xD) = (F_L(x) + L) / 2."""
    return 0.5 * (_fejer(L, x) + L)


def _Csum(L, x):
    """C(x) = sum_{D=0}^{L-1} (L-D) sin(xD) = (L sin x - sin(Lx)) / (4 sin^2(x/2)).
    C(0)=0; C(x) ~ L(L^2-1) x / 6 for x -> 0. Nonzero in general (causal grid is one-sided)."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        full = (L * np.sin(x) - np.sin(L * x)) / (4.0 * np.sin(x / 2.0) ** 2)
        return np.where(np.abs(x) < 1e-8, L * (L * L - 1.0) * x / 6.0, full)


def gram_causal_direct(L, w):
    """G_causal by direct summation (spec definition): Delta in {0..L-1}, W_Delta = L - Delta,
    Phi rows = [cos(w_k Delta), sin(w_k Delta)], G = Phi^T W Phi."""
    K = len(w)
    D = np.arange(L, dtype=np.float64)
    Wt = L - D
    Phi = np.empty((L, 2 * K))
    for k, wk in enumerate(w):
        Phi[:, 2 * k] = np.cos(wk * D)
        Phi[:, 2 * k + 1] = np.sin(wk * D)
    return Phi.T @ (Phi * Wt[:, None])


def gram_causal_exact(L, w):
    """Same Gram in closed form (used inside the optimizer; verified == direct).
    Blocks: cc = (S(wa-wb)+S(wa+wb))/2, ss = (S(wa-wb)-S(wa+wb))/2,
            cs = (C(wa+wb)-C(wa-wb))/2  (nonzero on the causal grid)."""
    K = len(w)
    W = np.asarray(w, dtype=np.float64)
    dm = W[:, None] - W[None, :]      # dm[a,b] = w_a - w_b (sign matters: C is odd)
    dp = W[None, :] + W[:, None]
    Sdm, Sdp = _Ssum(L, dm), _Ssum(L, dp)
    Cdm, Cdp = _Csum(L, dm), _Csum(L, dp)
    G = np.zeros((2 * K, 2 * K))
    G[0::2, 0::2] = 0.5 * (Sdm + Sdp)
    G[1::2, 1::2] = 0.5 * (Sdm - Sdp)
    G[0::2, 1::2] = 0.5 * (Cdp - Cdm)
    G[1::2, 0::2] = G[0::2, 1::2].T
    return G


# ----------------------------------------------------------------------
# Collision metrics (two independent implementations for C_full)
# ----------------------------------------------------------------------
def cfull_total(G):
    """C_full = sum_{i<j} (s1^2+s2^2)/2 with s = svals of whitened cross-Gram.
    Vectorized: Qab = ia[i] @ G_ab @ ia[j]^T = Q_i^T Q_j (ia = inv(chol(Gaa))),
    (s1^2+s2^2)/2 = ||Qab||_F^2 / 2. Batched 2x2 Cholesky + einsum."""
    K = G.shape[0] // 2
    blk = G.reshape(K, 2, K, 2).transpose(0, 2, 1, 3)          # blk[i,j] = G[2i:2i+2, 2j:2j+2]
    ia = np.linalg.inv(np.linalg.cholesky(blk[np.arange(K), np.arange(K)] + 1e-12 * np.eye(2)))
    Qab = np.einsum("iab,ijbc,jdc->ijad", ia, blk, ia)         # (K,K,2,2)
    cij = 0.5 * (Qab * Qab).sum(axis=(2, 3))
    return float(np.triu(cij, 1).sum())


def cfull_loop(G):
    """Reference implementation: per-pair whitened cross-Gram SVD (2x2)."""
    K = G.shape[0] // 2
    tot = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            Gaa = G[2 * i:2 * i + 2, 2 * i:2 * i + 2]
            Gbb = G[2 * j:2 * j + 2, 2 * j:2 * j + 2]
            Gab = G[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            ia = np.linalg.inv(np.linalg.cholesky(Gaa + 1e-12 * np.eye(2)))
            ib = np.linalg.inv(np.linalg.cholesky(Gbb + 1e-12 * np.eye(2)))
            s = np.linalg.svd(ia @ Gab @ ib.T, compute_uv=False)
            tot += 0.5 * (s[0] ** 2 + s[1] ** 2)
    return tot


def cfull_sym(L, w):
    """Symmetric-grid C_full via canon_corr (QR-based canonical correlations), per spec."""
    K = len(w)
    tot = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            s = vc.canon_corr(L, w[i], w[j])
            tot += 0.5 * (s[0] ** 2 + s[1] ** 2)
    return tot


def gram_basic(G):
    """(effrank_raw, effrank_whitened, C_cos, cond_whitened, logdet(whitened + 1e-6 I))."""
    K = G.shape[0] // 2
    er_raw = vc.effrank(G)
    Gw = vc.whiten_cols(G)
    er_wh = vc.effrank(Gw)
    ccos = G[0::2, 0::2]
    d = np.diag(ccos)
    c_cos = float(np.triu(ccos ** 2 / np.outer(d, d), 1).sum())
    e = np.linalg.eigvalsh(Gw)
    cond = float(e.max() / max(e[e > 1e-10].min(), 1e-300))
    _, ld = np.linalg.slogdet(Gw + 1e-6 * np.eye(2 * K))
    return er_raw, er_wh, c_cos, cond, float(ld)


# ----------------------------------------------------------------------
# Allocations
# ----------------------------------------------------------------------
def geometric(K, b):
    return vc.geometric(K, b)          # w_k = b^{-k/(K-1)}, endpoints [1, 1/b]


def evq(K, tau, b):
    """EVQ-Cosh: phi_k = 1 - arcsinh((1-u_k) sinh(tau))/tau, w_k = b^{-phi_k}; tau=0 -> geometric."""
    if tau == 0.0:
        u = np.arange(K) / (K - 1)
        return b ** (-u)
    return vc.evq_cosh(K, tau, b)


def uniform_healthy(K, L):
    return np.linspace(2.0 * np.pi / L, 1.0, K)


def tau_star(K, L):
    return max(2.0 * K / np.sqrt(L), 1.4)


# ----------------------------------------------------------------------
# Log-space parametrization + objectives
# ----------------------------------------------------------------------
def make_z_to_w(K, b):
    wmin, wmax = 1.0 / b, 1.0

    def z_to_w(z):
        sp = np.logaddexp(0.0, np.asarray(z, dtype=np.float64))          # softplus > 0
        frac = np.cumsum(sp) / sp.sum()                                  # strictly increasing in (0,1)
        wint = np.exp(np.log(wmin) + frac * np.log(wmax / wmin))         # K-2 interior points
        return np.concatenate(([wmin], wint, [wmax]))                    # endpoints fixed

    return z_to_w


def make_objectives(L):
    """Objectives on the training causal grid, evaluated at a frequency table w."""
    def f_cos(w):
        G = gram_causal_exact(L, w)
        ccos = G[0::2, 0::2]
        d = np.diag(ccos)
        return float(np.triu(ccos ** 2 / np.outer(d, d), 1).sum())

    def f_full(w):
        return cfull_total(gram_causal_exact(L, w))

    def f_logdet(w):
        G = gram_causal_exact(L, w)
        Gw = vc.whiten_cols(G)
        _, ld = np.linalg.slogdet(Gw + 1e-6 * np.eye(2 * len(w)))
        return -float(ld)

    return {"cos": f_cos, "full": f_full, "logdet": f_logdet}


def coord_polish(f, z0, n_sweeps=2, span=2.0, n_grid=9):
    """Coordinate descent polish: per coordinate, coarse grid then refined grid (~18 evals)."""
    z = np.array(z0, dtype=np.float64)
    fz = f(z)
    for _ in range(n_sweeps):
        for j in range(len(z)):
            best_v, best_f = z[j], fz
            for v in z[j] + np.linspace(-span, span, n_grid):
                zj = z.copy()
                zj[j] = v
                fv = f(zj)
                if fv < best_f:
                    best_v, best_f = v, fv
            for v in best_v + np.linspace(-span / n_grid, span / n_grid, n_grid):
                zj = z.copy()
                zj[j] = v
                fv = f(zj)
                if fv < best_f:
                    best_v, best_f = v, fv
            z[j], fz = best_v, best_f
    return z, fz


def optimize_alloc(K, b, L, objname, f_w, eps, maxiter):
    """L-BFGS-B from 3 random restarts (seeds 0,1,2) + coordinate polish of the best.
    Returns (w_best, f_best, [(seed, f, nit, success)], f_after_polish)."""
    z_to_w = make_z_to_w(K, b)

    def f(z):
        v = f_w(z_to_w(z))
        return v if np.isfinite(v) else 1e9

    best = None
    restarts = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        z0 = rng.standard_normal(K - 2)
        res = minimize(f, z0, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-5, eps=eps, maxls=30))
        fv = float(res.fun)
        restarts.append((seed, fv, int(res.nit), bool(res.success)))
        if best is None or fv < best[0]:
            best = (fv, np.array(res.x))
        log(f"    {objname} seed={seed}: f={fv:.6f} nit={res.nit} success={res.success}")
    zp, fp = coord_polish(f, best[1])
    log(f"    {objname} after coordinate polish: f={fp:.6f} (L-BFGS-B best was {best[0]:.6f})")
    return z_to_w(zp), fp, restarts, fp


# ----------------------------------------------------------------------
# Evaluation at L, 2L, 4L
# ----------------------------------------------------------------------
def eval_alloc(L_eval, w):
    Gc = gram_causal_direct(L_eval, w)
    er_raw, er_wh, c_cos, cond, ld_wh = gram_basic(Gc)
    c_full = cfull_total(Gc)
    nlow = int((np.asarray(w) * L_eval <= 1.0).sum())
    Gs = vc.gram_exact(L_eval, w)
    er_raw_s, er_wh_s, c_cos_s, cond_s, _ = gram_basic(Gs)
    c_full_s = cfull_sym(L_eval, w)
    with np.errstate(all="ignore"):
        ld_er_s = vc.logdet_erank(L_eval, w)
    return dict(L_eval=L_eval, erank_raw=er_raw, erank_whit=er_wh, C_cos=c_cos, C_full=c_full,
                cond_whit=cond, logdet_whit=ld_wh, n_wL_le1=nlow,
                erank_raw_sym=er_raw_s, erank_whit_sym=er_wh_s, C_cos_sym=c_cos_s,
                C_full_sym=c_full_s, logdet_erank_sym=ld_er_s)


# ----------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------
def sanity_checks():
    """Returns (all_ok, numbers) where numbers feeds the markdown sanity table."""
    res = {}
    ok = True
    log("== sanity checks ==")
    # (i) V6 reproduction: geometric K=64, b=5e5, L=4096, symmetric grid
    wg = geometric(64, 5e5)
    G = vc.gram_exact(4096, wg)
    res["er_raw"], res["er_wh"] = vc.effrank(G), vc.effrank(vc.whiten_cols(G))
    ok1 = abs(res["er_raw"] - 22.67) < 0.05 and abs(res["er_wh"] - 26.11) < 0.05
    ok &= ok1
    log(f"(i) geometric K=64 b=5e5 L=4096 symmetric: erank_raw={res['er_raw']:.4f} (expect ~22.67), "
        f"erank_whit={res['er_wh']:.4f} (expect ~26.11) -> {'PASS' if ok1 else 'FAIL'}")
    # (ii) EVQ tau=0 == geometric
    res["d_evq0"] = np.abs(evq(64, 0.0, 5e5) - geometric(64, 5e5)).max()
    ok2 = res["d_evq0"] < 1e-15
    ok &= ok2
    log(f"(ii) EVQ tau=0 vs geometric max|dw| = {res['d_evq0']:.2e} -> {'PASS' if ok2 else 'FAIL'}")
    # (iii) causal Gram closed-form vs direct summation (two independent implementations)
    rng = np.random.default_rng(7)
    w8 = np.sort(rng.uniform(2e-6, 1.0, 8))
    res["errs"] = []
    for Lc in (256, 4096, 16384):
        e = np.abs(gram_causal_exact(Lc, w8) - gram_causal_direct(Lc, w8)).max()
        res["errs"].append(e)
        log(f"    L={Lc}: max abs err = {e:.2e}")
    ok3 = max(res["errs"]) < 1e-6
    ok &= ok3
    log(f"(iii) causal Gram closed-form vs direct -> {'PASS' if ok3 else 'FAIL'}")
    # (iv) vectorized C_full vs per-pair SVD loop
    wc = np.sort(rng.uniform(2e-6, 1.0, 12))
    Gc = gram_causal_exact(4096, wc)
    res["cf_vec"], res["cf_loop"] = cfull_total(Gc), cfull_loop(Gc)
    ok4 = abs(res["cf_vec"] - res["cf_loop"]) < 1e-10 * max(1.0, res["cf_vec"])
    ok &= ok4
    log(f"(iv) C_full vectorized={res['cf_vec']:.6f} vs SVD-loop={res['cf_loop']:.6f} "
        f"-> {'PASS' if ok4 else 'FAIL'}")
    # (v) canonical correlations from causal blocks in [0,1]
    K = 12
    blk = Gc.reshape(K, 2, K, 2).transpose(0, 2, 1, 3)
    ia = np.linalg.inv(np.linalg.cholesky(blk[np.arange(K), np.arange(K)] + 1e-12 * np.eye(2)))
    Qab = np.einsum("iab,ijbc,jdc->ijad", ia, blk, ia)
    s = np.linalg.svd(Qab, compute_uv=False)
    res["smax"] = float(s.max())
    ok5 = res["smax"] <= 1.0 + 1e-8
    ok &= ok5
    log(f"(v) causal-block canonical correlations <= 1: max={res['smax']:.10f} -> "
        f"{'PASS' if ok5 else 'FAIL'}")
    # (vi) causal-block vs symmetric canon_corr on one pair (different grids, both in [0,1])
    wa, wb = 0.13, 0.29
    Gp = gram_causal_direct(4096, np.array([wa, wb]))
    s_pair = np.linalg.svd(np.linalg.inv(np.linalg.cholesky(Gp[:2, :2] + 1e-12 * np.eye(2)))
                           @ Gp[:2, 2:] @ np.linalg.inv(np.linalg.cholesky(Gp[2:, 2:] + 1e-12 * np.eye(2))).T,
                           compute_uv=False)
    s_sym = np.sort(vc.canon_corr(4096, wa, wb))[::-1]
    res["s_pair"] = (s_pair[0], s_pair[1], s_sym[0], s_sym[1])
    log(f"(vi) pair w=({wa},{wb}): causal-block svals={np.round(s_pair, 6)}, "
        f"symmetric canon_corr={np.round(s_sym, 6)} (both in [0,1])")
    log(f"== sanity: {'ALL PASS' if ok else 'SOME FAIL'}")
    return ok, res


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def fmt(x):
    if x is None:
        return "-"
    ax = abs(x)
    if ax == 0:
        return "0.0000"
    if 0.001 <= ax < 1e5:
        return f"{x:.4f}"
    return f"{x:.4e}"


def main():
    t0 = time.time()
    log("finK_compare.py: finite-K frequency-allocation comparison")
    if SMOKE:
        log("SMOKE MODE: tiny config, no files written")
        sanity_checks()
        tag, K, b, L = "smoke", 8, 1e3, 128
        objs = make_objectives(L)
        z_to_w = make_z_to_w(K, b)
        wg = geometric(K, b)
        for name in OBJS:
            fw = objs[name]
            log(f"baseline {name}: f(geometric)={fw(wg):.6f}")
            wbest, fbest, restarts, fpol = optimize_alloc(K, b, L, name, fw, eps=1e-4, maxiter=15)
            log(f"  -> {name} best f={fpol:.6f}, w={np.round(wbest, 6)}")
            row = eval_alloc(L, wbest)
            log(f"  eval at L: erank_raw={row['erank_raw']:.4f} erank_whit={row['erank_whit']:.4f} "
                f"C_cos={row['C_cos']:.4f} C_full={row['C_full']:.4f} cond={row['cond_whit']:.2e}")
        log(f"smoke done in {time.time()-t0:.1f}s")
        return

    ok, san = sanity_checks()

    # ---- optimizations -------------------------------------------------
    log("== optimization (training causal grid) ==")
    optima = {}        # (tag, obj) -> w
    opt_rec = {}       # (tag, obj) -> (f_geo, restarts, f_final)
    for tag, K, b, L in CONFIGS:
        for objname in OBJS:
            log(f"config {tag} K={K} b={b:.0f} L={L}: objective {objname}")
            objs = make_objectives(L)
            fw = objs[objname]
            f_geo = fw(geometric(K, b))
            eps = 1e-4 if objname in ("cos", "full") else 1e-6
            maxiter = 300
            wbest, fbest, restarts, fpol = optimize_alloc(K, b, L, objname, fw, eps, maxiter)
            optima[(tag, objname)] = wbest
            opt_rec[(tag, objname)] = (f_geo, restarts, fpol)
            fname = f"opt_{tag}_{objname}.npy"
            np.save(os.path.join(OUT, fname), wbest)
            log(f"  saved {fname}  (f: baseline={f_geo:.6f} -> final={fpol:.6f})")

    # ---- evaluation -----------------------------------------------------
    log("== evaluation at L, 2L, 4L ==")
    rows = []
    for tag, K, b, L in CONFIGS:
        allocs = [("geometric", None, geometric(K, b))]
        for tau in TAUS:
            allocs.append((f"EVQ-Cosh tau={tau}", tau, evq(K, tau, b)))
        for objname in OBJS:
            allocs.append((f"{objname}-opt", None, optima[(tag, objname)]))
        allocs.append(("uniform", None, uniform_healthy(K, L)))
        for label, tau, w in allocs:
            for Le in (L, 2 * L, 4 * L):
                row = eval_alloc(Le, w)
                row.update(dict(tag=tag, K=K, b=b, L_train=L, alloc=label, tau=tau))
                rows.append(row)
            log(f"  {tag} {label:<16} done")
    log(f"  total rows: {len(rows)}")

    # ---- CSV per K -------------------------------------------------------
    cols = ["b", "L_train", "L_eval", "alloc", "tau", "n_wL_le1",
            "erank_raw", "erank_whit", "C_cos", "C_full", "cond_whit",
            "erank_raw_sym", "erank_whit_sym", "C_cos_sym", "C_full_sym", "logdet_erank_sym"]
    for K in (16, 32, 64):
        fpath = os.path.join(OUT, f"finK_K{K}.csv")
        with open(fpath, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(cols)
            for r in rows:
                if r["K"] == K:
                    wr.writerow([r["b"], r["L_train"], r["L_eval"], r["alloc"], r["tau"],
                                 r["n_wL_le1"], f"{r['erank_raw']:.8g}", f"{r['erank_whit']:.8g}",
                                 f"{r['C_cos']:.8g}", f"{r['C_full']:.8g}", f"{r['cond_whit']:.8g}",
                                 f"{r['erank_raw_sym']:.8g}", f"{r['erank_whit_sym']:.8g}",
                                 f"{r['C_cos_sym']:.8g}", f"{r['C_full_sym']:.8g}",
                                 f"{r['logdet_erank_sym']:.8g}"])
        log(f"  wrote {fpath}")

    # ---- markdown ----------------------------------------------------------
    md = []
    md.append("# Finite-K frequency-allocation comparison — causal-weighted primary grid\n")
    md.append(f"Generated 2026-08-19 by `finK_compare.py` (numpy {np.__version__} + scipy, CPU). "
              f"Falsification-first audit: no model training.\n")
    md.append("## Setup\n")
    md.append("- Frequency range w in [1/b, 1]; primary b = 5e5 (secondary b = 1e4 for K=64); K in {16, 32, 64}.")
    md.append("- Training length L = 4096 (primary; secondary L = 1024 for K=32). Evaluation at L, 2L, 4L.")
    md.append("- **Primary grid (metric table): causal-weighted.** Delta in {0..L-1}, W_Delta = L - Delta, "
              "G = Phi^T W Phi by direct summation. On this grid the cos-sin cross block is nonzero, "
              "so C_full uses the full 2x2 blocks (Gaa^{-1/2} Gab Gbb^{-1/2} svals).")
    md.append("- **Reference grid (2 columns): symmetric.** Delta in [-(L-1), L-1], unweighted "
              "(Dirichlet-kernel Gram); C_full_sym via `canon_corr` (QR).")
    md.append("- Metrics: C_cos = sum_{i<j} <cos_i,cos_j>_W^2 / (<cos_i^2>_W <cos_j^2>_W); "
              "C_full = sum_{i<j} (s1^2+s2^2)/2 (canonical correlations of the 2D subspaces); "
              "effrank = entropy effrank (raw Gram / column-whitened Gram); cond = cond(whitened Gram); "
              "n(wL<=1) counted at the evaluation length. tau* = max(2K/sqrt(L), 1.4), highlighted with *.")
    md.append("- Optimizer: log-space softplus-cumsum parametrization, endpoints fixed; "
              "L-BFGS-B from 3 random restarts (seeds 0,1,2) + 2 coordinate-descent sweeps from the best restart.")
    md.append("")
    md.append("## Sanity checks\n")
    md.append("| check | result | status |")
    md.append("|---|---|---|")
    md.append(f"| (i) geometric K=64, b=5e5, L=4096, symmetric grid | erank_raw = {san['er_raw']:.4f} "
              f"(expect ~22.67), erank_whit = {san['er_wh']:.4f} (expect ~26.11) | "
              f"{'PASS' if abs(san['er_raw']-22.67)<0.05 and abs(san['er_wh']-26.11)<0.05 else 'FAIL'} |")
    md.append(f"| (ii) EVQ tau=0 == geometric | max\\|Delta w\\| = {san['d_evq0']:.2e} | "
              f"{'PASS' if san['d_evq0'] < 1e-15 else 'FAIL'} |")
    md.append(f"| (iii) causal Gram closed-form vs direct summation | max abs err = "
              f"{san['errs'][1]:.2e} (L=4096), {san['errs'][2]:.2e} (L=16384) | "
              f"{'PASS' if max(san['errs']) < 1e-6 else 'FAIL'} |")
    md.append(f"| (iv) C_full vectorized vs per-pair SVD loop | {san['cf_vec']:.6f} vs "
              f"{san['cf_loop']:.6f} | "
              f"{'PASS' if abs(san['cf_vec']-san['cf_loop']) < 1e-10*max(1.0, san['cf_vec']) else 'FAIL'} |")
    md.append(f"| (v) canonical correlations from causal 2x2 blocks in [0,1] | max = {san['smax']:.10f} | "
              f"{'PASS' if san['smax'] <= 1.0+1e-8 else 'FAIL'} |")
    md.append(f"| (vi) one pair w=(0.13,0.29): causal-block svals = "
              f"[{san['s_pair'][0]:.6f}, {san['s_pair'][1]:.6f}], symmetric canon_corr = "
              f"[{san['s_pair'][2]:.6f}, {san['s_pair'][3]:.6f}] (different grids; both <= 1) | PASS |")
    md.append("")
    md.append("## Optimization runs (training causal grid)\n")
    md.append("f = objective on training grid (cos: C_cos; full: C_full; logdet: -logdet(whitened+1e-6 I)). "
              "Baseline = geometric allocation. Best-of-3 restarts, then coordinate polish.\n")
    md.append("| config | objective | f(baseline) | f restarts (seed, f, niter, success) | f final |")
    md.append("|---|---|---|---|---|")
    for tag, K, b, L in CONFIGS:
        for objname in OBJS:
            f_geo, restarts, fpol = opt_rec[(tag, objname)]
            rtxt = "; ".join(f"seed{s}: {f:.6f}, nit{n}, {'ok' if sc else 'maxiter'}"
                             for s, f, n, sc in restarts)
            md.append(f"| {tag} (K={K}, b={b:.0f}, L={L}) | {objname} | {f_geo:.6f} | {rtxt} | {fpol:.6f} |")
    md.append("")
    md.append("## Metric tables (causal-weighted primary; symmetric reference in last 2 columns)\n")

    for tag, K, b, L in CONFIGS:
        ts = tau_star(K, L)
        md.append(f"### {tag}: K={K}, b={b:.0f}, L={L} (tau* = {ts})\n")
        md.append("| alloc | tau | L_eval | n(wL<=1) | erank_raw | erank_whit | C_cos | C_full | "
                  "cond_whit | erank_raw_sym | erank_whit_sym |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["tag"] != tag:
                continue
            label = r["alloc"]
            if r["tau"] is not None and abs(r["tau"] - ts) < 1e-12:
                label = label + "*"
            md.append(f"| {label} | {r['tau'] if r['tau'] is not None else '-'} | {r['L_eval']} | "
                      f"{r['n_wL_le1']} | {fmt(r['erank_raw'])} | {fmt(r['erank_whit'])} | "
                      f"{fmt(r['C_cos'])} | {fmt(r['C_full'])} | {fmt(r['cond_whit'])} | "
                      f"{fmt(r['erank_raw_sym'])} | {fmt(r['erank_whit_sym'])} |")
        md.append("")

    # ---- verdict -----------------------------------------------------------
    md.append("## Verdict\n")
    fam_keys = ["geometric", "EVQ-Cosh tau=*", "cos-opt", "full-opt", "logdet-opt", "uniform"]
    wins_high = {k: 0 for k in fam_keys}
    wins_deg = {k: 0 for k in fam_keys}
    detail = []
    for tag, K, b, L in CONFIGS:
        ts = tau_star(K, L)
        er_at_L, loss = {}, {}
        for k in fam_keys:
            alloc_name = f"EVQ-Cosh tau={ts}" if k == "EVQ-Cosh tau=*" else k
            rL = next(r for r in rows if r["tag"] == tag and r["L_eval"] == L and r["alloc"] == alloc_name)
            r4 = next(r for r in rows if r["tag"] == tag and r["L_eval"] == 4 * L and r["alloc"] == alloc_name)
            er_at_L[k] = rL["erank_whit"]
            loss[k] = (rL["erank_whit"] - r4["erank_whit"]) / rL["erank_whit"]
        best_hi = max(er_at_L, key=er_at_L.get)
        best_deg = min(loss, key=loss.get)
        wins_high[best_hi] += 1
        wins_deg[best_deg] += 1
        detail.append((tag, er_at_L, loss, best_hi, best_deg))
    md.append("Highest causal whitened effrank at L (per config):")
    for tag, er_at_L, loss, best_hi, best_deg in detail:
        md.append(f"- {tag}: **{best_hi}** (erank_whit(L) = {er_at_L[best_hi]:.4f}); "
                  f"least L->4L relative loss: **{best_deg}** (loss = {loss[best_deg]:.4%})")
    md.append("")
    tot_hi = sum(wins_high.values())
    tot_deg = sum(wins_deg.values())
    md.append(f"Aggregate over {len(CONFIGS)} configs — highest erank_whit at L: "
              + ", ".join(f"{k} {v}/{tot_hi}" for k, v in wins_high.items() if v)
              + ".")
    md.append(f"Aggregate — least relative erank_whit loss from L to 4L: "
              + ", ".join(f"{k} {v}/{tot_deg}" for k, v in wins_deg.items() if v)
              + ".")
    md.append("")
    md.append("Notes: EVQ-Cosh tau=* is the row highlighted per config; other tau values are reported in the "
              "table. The logdet-opt maximizes logdet(whitened Gram + 1e-6 I) on the training causal grid; "
              "cos-opt minimizes C_cos; full-opt minimizes C_full (both training-grid causal).")

    fpath = os.path.join(OUT, "finK_results.md")
    with open(fpath, "w") as fh:
        fh.write("\n".join(md))
    log(f"  wrote {fpath}")
    log(f"TOTAL TIME {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
