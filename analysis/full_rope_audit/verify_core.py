"""
verify_core.py — Full-RoPE 2D subspace collision theory: core identity verification.

Independent audit script (does not modify paper/training code).
Verifies, each with TWO independent implementations:
  V1: Gram block structure on symmetric grid via Dirichlet kernel (exact formula)
  V2: canonical correlations = whitened cross-Gram singular values = Q^T Q singular values
  V3: phase-optimized cosine overlap = largest canonical correlation (phase invariance gap)
  V4: logdet identity: det G(2 blocks) = det G_aa det G_bb (1-s1^2)(1-s2^2)
  V5: low-frequency collapse: V_w -> span{1, Delta} at rate (wL)^2 in principal angle
  V6: effective-rank loss of geometric allocations (raw vs whitened Gram)
numpy only, <2s.
"""
import numpy as np

# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------
def dirichlet(L, x):
    """D_L(x) = sum_{d=-(L-1)}^{L-1} cos(x d); exact closed form."""
    if abs(x) < 1e-15:
        return 2 * L - 1
    return np.sin((L - 0.5) * x) / np.sin(x / 2)

def gram_exact(L, w):
    """2K x 2K Gram, columns ordered cos(w0), sin(w0), cos(w1), sin(w1), ...
    Exact: G_ab = 1/2 * diag(D(wa-wb) + D(wa+wb), D(wa-wb) - D(wa+wb)); cross = 0."""
    K = len(w)
    G = np.zeros((2 * K, 2 * K))
    for a in range(K):
        for b in range(K):
            dp = dirichlet(L, w[a] - w[b])
            dm = dirichlet(L, w[a] + w[b])
            G[2 * a, 2 * b] = 0.5 * (dp + dm)          # cos-cos
            G[2 * a + 1, 2 * b + 1] = 0.5 * (dp - dm)  # sin-sin
            G[2 * a, 2 * b + 1] = 0.0                  # cos-sin (exact by symmetry)
            G[2 * b + 1, 2 * a] = 0.0
    return G

def gram_direct(L, w):
    """Same Gram by direct summation over the Delta grid (reference impl)."""
    D = np.arange(-(L - 1), L)
    Phi = np.zeros((len(D), 2 * len(w)))
    for k, wk in enumerate(w):
        Phi[:, 2 * k] = np.cos(wk * D)
        Phi[:, 2 * k + 1] = np.sin(wk * D)
    return Phi.T @ Phi

def basis(L, w):
    D = np.arange(-(L - 1), L)
    return D, np.stack([np.cos(w * D), np.sin(w * D)]).T  # (2L-1) x 2

def orthobasis(C):
    Q, _ = np.linalg.qr(C)
    return Q[:, :2]

def canon_corr(L, wa, wb):
    """Canonical correlations between V_wa and V_wb (svals of Q_a^T Q_b)."""
    _, Ca = basis(L, wa)
    _, Cb = basis(L, wb)
    Qa, Qb = orthobasis(Ca), orthobasis(Cb)
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)

def whitened_cross(L, wa, wb):
    """Singular values of Gaa^{-1/2} Gab Gbb^{-1/2} (whitened cross-Gram)."""
    G = gram_exact(L, np.array([wa, wb]))
    Gaa, Gab, Gbb = G[:2, :2], G[:2, 2:], G[2:, 2:]
    ia = np.linalg.inv(np.linalg.cholesky(Gaa + 1e-12 * np.eye(2)))
    ib = np.linalg.inv(np.linalg.cholesky(Gbb + 1e-12 * np.eye(2)))
    return np.linalg.svd(ia @ Gab @ ib.T, compute_uv=False)

def effrank(G):
    e = np.linalg.eigvalsh((G + G.T) / 2)
    e = e[e > 1e-10]
    p = e / e.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def logdet_erank(L, w):
    _, ld = np.linalg.slogdet(gram_exact(L, w))
    return float(np.exp(ld / (2 * len(w))))

def whiten_cols(G):
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)

# ----------------------------------------------------------------------
# V1: exact block structure vs direct summation
# ----------------------------------------------------------------------
rng = np.random.default_rng(0)
w = np.sort(rng.uniform(0.001, 3.0, 5))
L = 64
G1, G2 = gram_exact(L, w), gram_direct(L, w)
err_v1 = np.abs(G1 - G2).max()
print(f"V1 Gram exact-vs-direct max abs err: {err_v1:.2e}  {'PASS' if err_v1 < 1e-10 else 'FAIL'}")

# cos-sin cross terms exactly zero even for a single frequency (self-block)
_, C = basis(37, 0.7)
err_cross = abs(np.dot(C[:, 0], C[:, 1]))
print(f"   self cos-sin cross term: {err_cross:.2e}  {'PASS' if err_cross < 1e-10 else 'FAIL'}")

# ----------------------------------------------------------------------
# V2: canonical correlations = whitened cross-Gram singular values
# ----------------------------------------------------------------------
errs = []
for wa, wb in [(0.05, 0.09), (0.5, 0.7), (1.0, 1.05), (2.3, 2.31), (0.01, 2.9)]:
    s_q = canon_corr(64, wa, wb)
    s_w = whitened_cross(64, wa, wb)
    errs.append(np.abs(s_q - s_w).max())
err_v2 = max(errs)
print(f"V2 canon-corr vs whitened cross-Gram svals max err: {err_v2:.2e}  "
      f"{'PASS' if err_v2 < 1e-8 else 'FAIL'}")

# ----------------------------------------------------------------------
# V3: phase-optimized cosine overlap = sigma_1 (phase invariance gap)
# ----------------------------------------------------------------------
def phase_overlap(L, wa, wb, n=128):
    D = np.arange(-(L - 1), L)
    best = 0.0
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for a in th:
        ca = np.cos(wa * D + a)
        ca /= np.linalg.norm(ca)
        for b in th:
            cb = np.cos(wb * D + b)
            cb /= np.linalg.norm(cb)
            best = max(best, abs(np.dot(ca, cb)))
    return best

errs = []
for wa, wb in [(0.5, 0.7), (1.0, 1.4), (0.05, 0.09)]:
    s1 = canon_corr(64, wa, wb)[0]
    po = phase_overlap(64, wa, wb)
    errs.append(abs(s1 - po))
err_v3 = max(errs)
print(f"V3 max_phase cos-overlap vs sigma_1 max err: {err_v3:.2e}  "
      f"{'PASS' if err_v3 < 2e-2 else 'FAIL'} (grid-res limited)")

# phase NON-invariance of the cosine-only kernel itself:
D = np.arange(-63, 64)
c0 = np.cos(0.5 * D); c0 /= np.linalg.norm(c0)
c1 = np.cos(0.5 * D + 1.3); c1 /= np.linalg.norm(c1)
print(f"   cos-only overlap across phase shift 1.3 rad: {abs(np.dot(c0, c1)):.3f} "
      f"(<1 => cos-only kernel is phase-dependent; subspace metric gives 1.000)")

# ----------------------------------------------------------------------
# V4: logdet identity for 2 blocks
# ----------------------------------------------------------------------
for wa, wb in [(0.05, 0.09), (0.5, 0.7), (2.3, 2.31)]:
    G = gram_exact(64, np.array([wa, wb]))
    Gaa, Gab, Gbb = G[:2, :2], G[:2, 2:], G[2:, 2:]
    s1, s2 = canon_corr(64, wa, wb)
    _, ld_full = np.linalg.slogdet(G)
    _, ld_aa = np.linalg.slogdet(Gaa)
    _, ld_bb = np.linalg.slogdet(Gbb)
    ld_rhs = ld_aa + ld_bb + np.log(1 - s1**2) + np.log(1 - s2**2)
    rel = abs(ld_full - ld_rhs) / abs(ld_full)
    print(f"V4 logdet identity (w={wa},{wb}): LHS={ld_full:.4f} RHS={ld_rhs:.4f} "
          f"rel err {rel:.2e}  {'PASS' if rel < 1e-10 else 'FAIL'}")

# ----------------------------------------------------------------------
# V5: low-frequency collapse rate: V_w -> span{1, Delta}, angle ~ (wL)^2
# ----------------------------------------------------------------------
def angles_to_lim(L, w):
    """Canonical correlations between V_w and V_0 = span{1, Delta}."""
    D = np.arange(-(L - 1), L)
    Qw = orthobasis(np.stack([np.cos(w * D), np.sin(w * D)]).T)
    Q0 = orthobasis(np.stack([np.ones_like(D, float), D.astype(float)]).T)
    return np.linalg.svd(Qw.T @ Q0, compute_uv=False)

print("V5 collapse rate: (1 - s_i^2) / (wL)^4 should approach constants as wL->0")
print(f"   {'wL':>6} | {'(1-s1^2)/(wL)^4':>15} | {'(1-s2^2)/(wL)^4':>15}")
Lv = 4096
prev = {}
for c in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    s1, s2 = np.sort(angles_to_lim(Lv, c / Lv))[::-1]
    r1, r2 = (1 - s1**2) / c**4, (1 - s2**2) / c**4
    prev[c] = (r1, r2)
    print(f"   {c:6.2f} | {r1:15.6f} | {r2:15.6f}")
# L-independence of the ratios (depends on wL only):
print("   L-sweep at wL=0.2 (ratios should be ~L-independent):")
for Ll in [256, 512, 1024, 2048, 4096]:
    s1, s2 = np.sort(angles_to_lim(Ll, 0.2 / Ll))[::-1]
    print(f"     L={Ll:5d}: {(1-s1**2)/0.2**4:.6f}  {(1-s2**2)/0.2**4:.6f}")
r1s = np.array([v[0] for v in list(prev.values())[:5]])
r2s = np.array([v[1] for v in list(prev.values())[:5]])
sp1 = (r1s.max() - r1s.min()) / r1s.mean()
sp2 = (r2s.max() - r2s.min()) / r2s.mean()
print(f"   ratio drift over c in [0.02,0.5]: s1 {sp1:.2%}, s2 {sp2:.2%} "
      f"{'PASS (both converge to constants; (wL)^4 rate confirmed)' if max(sp1, sp2) < 0.1 else 'CHECK'}")
print(f"   constants: (1-s1^2)/(wL)^4 -> ~{r1s[0]:.6f} (sin-direction, =1/525), "
      f"(1-s2^2)/(wL)^4 -> ~{r2s[0]:.6f} (cos-direction, =1/45)")

# ----------------------------------------------------------------------
# V6: effective-rank loss of geometric allocations (raw + column-whitened)
# ----------------------------------------------------------------------
def geometric(K, b):
    return b ** (-np.arange(K) / (K - 1))   # endpoints fixed: [1, 1/b]

def evq_cosh(K, tau, b):
    u = np.arange(K) / (K - 1)
    phi = 1 - np.arcsinh((1 - u) * np.sinh(tau)) / tau
    return b ** (-phi)

print("\nV6 effective rank of positional basis, L=4096:")
print(f"   {'alloc':>22} | {'K':>3} | {'b':>7} | {'n(wL<=1)':>7} | "
      f"{'erank_raw':>9} | {'erank_whit':>10} | {'logdet_er':>9} | {'cond':>9}")
for b in [1e4, 5e5]:
    for K in [16, 32, 64]:
        for name, wtab in [("geometric", geometric(K, b)),
                           ("evq t=1.4", evq_cosh(K, 1.4, b)),
                           ("evq t=2.5", evq_cosh(K, 2.5, b))]:
            G = gram_exact(4096, wtab)
            nlow = int((wtab * 4096 <= 1).sum())
            er_raw = effrank(G)
            er_wh = effrank(whiten_cols(G))
            er_ld = logdet_erank(4096, wtab)
            e = np.linalg.eigvalsh(G)
            cond = e.max() / max(e[e > 1e-10].min(), 1e-300)
            print(f"   {name:>22} | {K:3d} | {b:7.0f} | {nlow:7d} | "
                  f"{er_raw:9.2f} | {er_wh:10.2f} | {er_ld:9.2f} | {cond:9.2e}")
print(f"   (full rank baseline = 2K; raw Gram reflects energy scales, "
      f"whitened Gram reflects subspace geometry only)")
