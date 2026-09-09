#!/usr/bin/env python3
# Numerics for rebuttal/STRONG_MODEL_THEORY_VERDICT_20260720.md
# Part 1: main checks (A-J). Part 2: follow-ups (KL fix, video tau, grid conventions).
# Pure NumPy; run: python3 rebuttal/strong_model_verdict_numerics_20260720.py
import numpy as np, json, math, csv, io, sys
np.set_printoptions(precision=4, suppress=True)

# ---------- shared objects ----------
def exact_kernel_mat(K, L, b, phis=None):
    """K(phi_i,phi_j) = (1/2L)[sin((w1-w2)L)/(w1-w2) + sin((w1+w2)L)/(w1+w2)],
    the uniform-prior [0,L] cos-product Gram (both sinc terms)."""
    if phis is None:
        phis = (np.arange(K) + 0.5) / K  # midpoint grid
    w = b ** (-phis)
    W1, W2 = np.meshgrid(w, w, indexing="ij")
    diff, s = W1 - W2, W1 + W2
    t1 = np.where(np.abs(diff) < 1e-15, L, np.sin(diff * L) / np.where(diff == 0, 1, diff))
    t2 = np.sin(s * L) / s
    return (t1 + t2) / (2 * L), phis

def fit_surrogate(Kmat, phis):
    """Paper protocol: alpha = mean(K_ii)*dphi ; beta = LS of off-diag vs min."""
    Kc = len(phis); dphi = 1.0 / Kc
    alpha = np.mean(np.diag(Kmat)) * dphi
    M = np.minimum.outer(phis, phis)
    off = ~np.eye(Kc, dtype=bool)
    beta = float(np.dot(Kmat[off], M[off]) / np.dot(M[off], M[off]))
    return alpha, beta

def rho_tau(phi, tau):
    if tau < 1e-9: return np.ones_like(phi)
    return tau * np.cosh(tau * (1 - phi)) / np.sinh(tau)

def evq_phis_midpoint(K, tau):
    u = (np.arange(K) + 0.5) / K
    if tau < 1e-9: return u
    return 1 - np.arcsinh((1 - u) * np.sinh(tau)) / tau

def collision_C(Kmat):
    d = np.sqrt(np.diag(Kmat))
    R = Kmat / np.outer(d, d)
    iu = np.triu_indices(len(Kmat), 1)
    return float(np.sum(R[iu] ** 2)), R

def eff_rank(R):
    ev = np.linalg.eigvalsh(R); ev = np.clip(ev, 0, None)
    p = ev / ev.sum(); p = p[p > 1e-15]
    return float(np.exp(-np.sum(p * np.log(p))))

def qfun(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    out[small] = x[small] ** 4 / 45.0
    xs = x[~small]
    out[~small] = 0.5 + np.sin(2 * xs) / (4 * xs) - (np.sin(xs) / xs) ** 2
    return out

def Q1(L, b, n=20000):
    phi = (np.arange(n) + 0.5) / n
    eta = (1 - phi) ** 2 / 2 - 1 / 6
    return float(np.mean(eta * qfun(L * b ** (-phi))))

SEC = lambda t: print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

# =====================================================================
SEC("A. Q2: surrogate fit (alpha,beta) -> tau_surr vs deployed tau=d/sqrt(L)")
rows = []
for d in (32, 64, 128):
    for L in (128, 256, 512, 1024, 2048, 4096):
        K = d // 2
        Km, ph = exact_kernel_mat(K, L, 500_000)
        a, bta = fit_surrogate(Km, ph)
        rows.append((d, L, a, bta, math.sqrt(bta / a), d / math.sqrt(L)))
print(f"{'d':>4}{'L':>6}{'alpha':>9}{'1/d':>8}{'beta':>9}{'tau_surr':>10}{'tau_dep':>9}{'ratio':>8}")
for d, L, a, bta, ts, td in rows:
    print(f"{d:>4}{L:>6}{a:>9.4f}{1/d:>8.4f}{bta:>9.4f}{ts:>10.3f}{td:>9.3f}{ts/td:>8.2f}")
# beta ~ L^p and tau_surr ~ L^p/2 exponents at d=64
d64 = [(L, bta, ts) for d, L, a, bta, ts, td in rows if d == 64]
lL = np.log([r[0] for r in d64]); lb = np.log([r[1] for r in d64]); lt = np.log([r[2] for r in d64])
pb = np.polyfit(lL, lb, 1)[0]; pt = np.polyfit(lL, lt, 1)[0]
print(f"\nfit d=64,b=500K: beta ~ L^{pb:.3f}; tau_surr ~ L^{pt:.3f}")
# d-exponent of tau_surr at fixed L
for L in (512, 2048):
    sub = [(d, ts) for d, LL, a, bta, ts, td in rows if LL == L]
    e = np.polyfit(np.log([s[0] for s in sub]), np.log([s[1] for s in sub]), 1)[0]
    print(f"tau_surr ~ d^{e:.3f} at L={L}")

SEC("B. Q2: density distance surrogate-optimal vs deployed (d=64,b=500K)")
phig = (np.arange(200000) + 0.5) / 200000
for L in (2048, 4096):
    K = 32
    Km, ph = exact_kernel_mat(K, L, 500_000)
    a, bta = fit_surrogate(Km, ph)
    ts, td = math.sqrt(bta / a), 64 / math.sqrt(L)
    r_s, r_d = rho_tau(phig, ts), rho_tau(phig, td)
    l1 = float(np.mean(np.abs(r_s - r_d)))
    print(f"L={L}: tau_surr={ts:.2f} tau_dep={td:.2f} factor={ts/td:.2f}  "
          f"cosh(ts)={math.cosh(ts):.0f}:1 vs cosh(td)={math.cosh(td):.2f}:1  ||rho_s-rho_d||_1={l1:.3f}")

SEC("C. Q8/Q2/Q6: reproduce 12-config validation table; find which tau it used")
configs = ([(32, L, 500_000) for L in (128, 256, 512, 1024, 2048, 4096)] +
           [(32, 2048, 10_000), (32, 2048, 100_000)] +
           [(16, 32, 100), (16, 32, 1000), (16, 32, 10_000), (16, 32, 50_000)])
paper_geo = [226.3, 192.6, 163.2, 133.9, 109.0, 87.0, 34.8, 79.4, 22.6, 42.3, 55.4, 63.3]
paper_evq = [17.7, 27.6, 43.3, 59.0, 67.5, 66.3, 19.2, 47.0, 14.2, 28.0, 39.8, 47.2]
print(f"{'K':>3}{'L':>6}{'b':>8}{'tau=d/√L':>9} | {'C_geo':>8}{'(paper)':>9} | {'C_evq':>8}{'(paper)':>9} | "
      f"{'er_g':>6}{'er_e':>6}")
mono_lin_best, mono_exp_best, evq_red, tsur_red = [], [], [], []
for i, (K, L, b) in enumerate(configs):
    d = 2 * K; tau = d / math.sqrt(L)
    Kg, _ = exact_kernel_mat(K, L, b)
    Cg, Rg = collision_C(Kg)
    Ke, _ = exact_kernel_mat(K, L, b, evq_phis_midpoint(K, tau))
    Ce, Re = collision_C(Ke)
    evq_red.append(1 - Ce / Cg)
    # surrogate's own optimum on same config
    a_, b_ = fit_surrogate(Kg, (np.arange(K) + .5) / K)
    ts = math.sqrt(b_ / a_)
    Kts, _ = exact_kernel_mat(K, L, b, evq_phis_midpoint(K, ts))
    Cts, _ = collision_C(Kts)
    tsur_red.append((ts, 1 - Cts / Cg))
    # generic monotone controls (non-cosh): linear tilt and exponential
    bl = be = np.inf
    for s in np.linspace(0.05, 1.95, 39):  # rho = 1 + s(1/2-phi) >0
        u = (np.arange(K) + 0.5) / K       # inverse CDF of linear density
        # F(x) = x + s x(1-x)/2 ... solve quadratic: (s/2)x^2 - (1+s/2)x + u = 0 sign... use numeric
        xs = np.linspace(0, 1, 4001); F = xs + s * xs * (1 - xs) / 2
        pl = np.interp(u, F, xs)
        Cl, _ = collision_C(exact_kernel_mat(K, L, b, pl)[0]); bl = min(bl, Cl)
    for s in np.linspace(0.25, 12, 48):    # rho ∝ exp(-s phi); F=(1-e^-sx)/(1-e^-s)
        u = (np.arange(K) + 0.5) / K
        pe = -np.log(1 - u * (1 - math.exp(-s))) / s
        Cx, _ = collision_C(exact_kernel_mat(K, L, b, pe)[0]); be = min(be, Cx)
    mono_lin_best.append(1 - bl / Cg); mono_exp_best.append(1 - be / Cg)
    print(f"{K:>3}{L:>6}{b:>8}{tau:>9.3f} | {Cg:>8.1f}{paper_geo[i]:>9.1f} | {Ce:>8.1f}{paper_evq[i]:>9.1f} | "
          f"{eff_rank(Rg):>6.1f}{eff_rank(Re):>6.1f}")
print("\n-> reduction comparison per config:")
print(f"{'cfg':>3}{'EVQ@dep':>9}{'EVQ@tau_surr':>13}{'lin-tilt*':>10}{'exp*':>7}")
for i in range(12):
    print(f"{i:>3}{evq_red[i]*100:>8.0f}%{tsur_red[i][1]*100:>11.0f}%  "
          f"{mono_lin_best[i]*100:>7.0f}%{mono_exp_best[i]*100:>6.0f}%   (tau_surr={tsur_red[i][0]:.1f})")

SEC("C2. held-out prior: exponential D, mean L/4 (kernel a^2/(a^2+w^2) form)")
def exp_kernel_mat(K, L, b, phis=None):
    if phis is None: phis = (np.arange(K) + 0.5) / K
    w = b ** (-phis); a = 4.0 / L
    W1, W2 = np.meshgrid(w, w, indexing="ij")
    return 0.5 * (a**2 / (a**2 + (W1 - W2)**2) + a**2 / (a**2 + (W1 + W2)**2)), phis
for (K, L, b) in [(32, 128, 500_000), (32, 2048, 500_000), (32, 4096, 500_000)]:
    d = 2 * K; tau = d / math.sqrt(L)
    Cg, _ = collision_C(exp_kernel_mat(K, L, b)[0])
    Ce, _ = collision_C(exp_kernel_mat(K, L, b, evq_phis_midpoint(K, tau))[0])
    print(f"K={K} L={L} b={b}: geo {Cg:.1f} -> evq {Ce:.1f}  ({100*(1-Ce/Cg):.0f}% reduction)")

SEC("D. collision curve C(tau), rep config d=64 L=512 b=500K + d-scaling of argmin")
for d, L in [(32, 512), (64, 512), (128, 512), (64, 2048)]:
    K = d // 2
    taus = np.concatenate([np.linspace(0, 16, 161)])
    Cs = [collision_C(exact_kernel_mat(K, L, 500_000, evq_phis_midpoint(K, t))[0])[0] for t in taus]
    j = int(np.argmin(Cs)); td = d / math.sqrt(L)
    Cdep = collision_C(exact_kernel_mat(K, L, 500_000, evq_phis_midpoint(K, td))[0])[0]
    print(f"d={d:>3} L={L}: argmin_tau C = {taus[j]:.1f} (C={Cs[j]:.3f}, c={taus[j]/td:.2f});"
          f" C(tau_dep={td:.2f})={Cdep:.1f}; C(0)={Cs[0]:.1f}")

SEC("E. Q5: quantization load integrals for cosh density")
print("closed forms: I1=∫1/rho = sinh(t)·atan(sinh t)/t², I2=∫1/rho² = sinh²t·tanh t/t³")
for t in (0.5, 1.0, 1.414, 2.0, 4.0, 5.657):
    I1 = math.sinh(t) * math.atan(math.sinh(t)) / t**2
    I2 = math.sinh(t)**2 * math.tanh(t) / t**3
    print(f"tau={t:<6}: ∫1/rho={I1:.4f}  ∫1/rho²={I2:.4f}  (uniform=1; Jensen: I2>=I1²={I1**2:.4f}>=1)")
print("\nnon-circular weight w(phi)=q(L b^-phi) (phase-variance):  D_w[rho]=∫w/rho² vs D_w[1]=∫w")
for (L, b, t) in [(2048, 500_000, 1.414), (4096, 500_000, 1.0), (128, 500_000, 5.657),
                  (512, 500_000, 2.828), (32, 10_000, 1.4)]:
    w = qfun(L * b ** (-phig)); r = rho_tau(phig, t)
    Dw_r = float(np.mean(w / r**2)); Dw_1 = float(np.mean(w))
    print(f"L={L:<5} b={b:<7} tau={t}: ∫w/rho²={Dw_r:.4f}  ∫w={Dw_1:.4f}  ratio={Dw_r/Dw_1:.3f}"
          f"  ({'REDUCES' if Dw_r < Dw_1 else 'increases'})")

SEC("F. Q4: order check — schedule-KL is O(tau^4); linear allocation score O(tau^2)")
L, b, K = 512, 500_000, 32
j = np.arange(1, L + 1)
def logits(tau):
    ph = evq_phis_midpoint(K, tau); w = b ** (-ph)
    return np.cos(np.outer(j, w)).sum(axis=1) / math.sqrt(2 * K)
z0 = logits(0); p0 = np.full(L, 1 / L)
th_list = np.array([0.01, 0.02, 0.04, 0.08, 0.16])
kls, dUs = [], []
U0 = float(np.mean(qfun(L * b ** (-evq_phis_midpoint(2000, 0)))))
for th in th_list:
    t = math.sqrt(th)
    z = logits(t)
    # KL(p0 || softmax(z0 + (z - z0))) with diffuse base: shift so z0 flat-> use raw z vs z0
    dz = z - z0
    a1 = np.log(np.mean(np.exp(dz - dz.max()))) + dz.max()
    kls.append(a1 - np.mean(dz))          # KL(p0||p_th) for uniform p0
    Ut = float(np.mean(qfun(L * b ** (-evq_phis_midpoint(2000, t)))))
    dUs.append(Ut - U0)
kls, dUs = np.array(kls), np.array(dUs)
sK = np.polyfit(np.log(th_list), np.log(np.abs(kls)), 1)[0]
sU = np.polyfit(np.log(th_list), np.log(np.abs(dUs)), 1)[0]
print(f"KL(p0||p_theta) ~ theta^{sK:.2f}  (theta=tau², so KL=O(tau^{2*sK:.1f}))")
print(f"Delta U_alloc  ~ theta^{sU:.2f}  (linear score: O(tau^{2*sU:.1f})),  Q1({L},{b})={Q1(L,b):.5f}")
print("Q1 table:", {f"L={LL}": round(Q1(LL, 500_000), 5) for LL in (128, 512, 2048, 8192)},
      " b=10K,L=4096:", round(Q1(4096, 10_000), 5), " video b=100,L=32:", round(Q1(32, 100), 5),
      " b=50K,L=32:", round(Q1(32, 50_000), 5))

SEC("G. Q3: Phase16 manifest — does the sweep identify the d-exponent?")
try:
    import os as _os
    _manifest = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "..", "data", "curated", "phase16_99run_manifest.csv")
    with open(_manifest) as f:
        rd = list(csv.DictReader(f))
    agg = {}
    for r in rd:
        Ltr = int(r["seq_len"]); d = int(r["head_dim"]); tau = float(r["tau"])
        ppl = json.loads(r["ppl_json"].replace('""', '"'))
        ext = [math.log(v) for k, v in ppl.items() if int(k) > Ltr]
        agg.setdefault((Ltr, d), {}).setdefault(tau, []).append(np.mean(ext))
    print(f"{'L':>5}{'d':>5}{'thry':>6} | tau grid: mean extrapolation logPPL (n seeds)  -> best tau (c=best/thry)")
    best = {}
    for (Ltr, d), tm in sorted(agg.items()):
        thry = d / math.sqrt(Ltr)
        items = sorted(tm.items())
        s = "  ".join(f"{t:g}:{np.mean(v):.3f}(n{len(v)})" for t, v in items)
        nz = [(t, np.mean(v)) for t, v in items if t > 0]
        tb = min(nz, key=lambda x: x[1])[0]
        best[(Ltr, d)] = tb
        print(f"{Ltr:>5}{d:>5}{thry:>6.2f} | {s} -> {tb:g} (c={tb/thry:.2f})")
    print("\nbest-multiple c per config (rows L, cols d):")
    for Ltr in (256, 512, 1024):
        print(f"  L={Ltr}: " + "  ".join(f"d={d}: c={best[(Ltr,d)]/(d/math.sqrt(Ltr)):.2f}"
              for d in (32, 64, 128) if (Ltr, d) in best))
    # naive exponents (caveat: grid is proportional to d/sqrt(L) by design)
    for Ltr in (256, 512, 1024):
        ds = [d for d in (32, 64, 128) if (Ltr, d) in best]
        e = np.polyfit(np.log(ds), np.log([best[(Ltr, dd)] for dd in ds]), 1)[0]
        print(f"  naive d-exponent at L={Ltr}: {e:.2f}")
    for d in (32, 64, 128):
        Ls = [Ltr for Ltr in (256, 512, 1024) if (Ltr, d) in best]
        e = np.polyfit(np.log(Ls), np.log([best[(Ll, d)] for Ll in Ls]), 1)[0]
        print(f"  naive L-exponent at d={d}: {e:.2f}")
except Exception as ex:
    print("manifest analysis failed:", ex)

SEC("H. Q7: inverse-CDF amplification sinh(tau)/tau at deployed taus")
for t in (1.0, 1.414, 2.828, 4.0, 5.657, 8.0):
    print(f"tau={t:<6}: sinh(tau)/tau = {math.sinh(t)/t:>8.2f}   tau/(2 ln 500K)={t/(2*math.log(500_000)):.3f}")

SEC("I. Q1: minimizer of the EXACT quadratic form rho^T K rho (mass=1, rho>=0)")
def proj_simplex(v, s):
    """Euclidean projection of v onto {x>=0, sum x = s}."""
    u = np.sort(v)[::-1]; css = np.cumsum(u)
    idx = np.arange(1, len(v) + 1)
    cond = u - (css - s) / idx > 0
    r = idx[cond][-1]; th = (css[r - 1] - s) / r
    return np.maximum(v - th, 0)
try:
    for (d, L) in [(64, 512), (64, 2048)]:
        K = d // 2
        Km, ph = exact_kernel_mat(K, L, 500_000)
        dphi = 1.0 / K
        lam = np.linalg.eigvalsh(Km)[-1]
        step = 0.9 / lam
        r = np.ones(K)
        for _ in range(200000):
            r_new = proj_simplex(r - step * 2 * (Km @ r), K)  # sum r = K  <=> sum r*dphi = 1
            if np.max(np.abs(r_new - r)) < 1e-12: r = r_new; break
            r = r_new
        phistar = math.log(L) / math.log(500_000)
        # best cosh fit (L2 over tau grid)
        taus = np.linspace(0.01, 20, 400)
        errs = [np.mean((rho_tau(ph, t) - r) ** 2) for t in taus]
        tb = taus[int(np.argmin(errs))]
        mass_active = r[ph <= phistar].sum() * dphi
        print(f"d={d} L={L}: QF minimizer rho* (first/last 8 of {K}):")
        print("  rho*:", np.round(r[:8], 2), "...", np.round(r[-8:], 2))
        print(f"  mass on active phi<=log_b L={phistar:.2f}: {mass_active:.2f}; "
              f"best cosh-fit tau={tb:.1f}, L2 err={min(errs):.3f} "
              f"(vs cosh(tau_dep) L2 err={np.mean((rho_tau(ph, d/math.sqrt(L)) - r)**2):.3f}); "
              f"monotone? {bool(np.all(np.diff(r) <= 1e-6))}; n_zero={int(np.sum(r < 1e-8))}")
except Exception as ex:
    print("QP failed:", ex)

SEC("J. cosh family on exact quadratic form: argmin_tau rho_tau^T K rho_tau")
for (d, L) in [(64, 512), (64, 2048)]:
    K = d // 2
    Km, ph = exact_kernel_mat(K, L, 500_000)
    taus = np.linspace(0, 20, 201)
    vals = [rho_tau(ph, t) @ Km @ rho_tau(ph, t) for t in taus]
    j0 = int(np.argmin(vals))
    a_, b_ = fit_surrogate(Km, ph)
    print(f"d={d} L={L}: argmin_tau <rho,K rho> = {taus[j0]:.1f}; tau_surr(fit)={math.sqrt(b_/a_):.2f}; "
          f"tau_dep={d/math.sqrt(L):.2f}")
print("\nDone.")

# ============================ PART 2: FOLLOW-UPS ============================
"""Follow-ups: correct KL order check; video-row tau; endpoint-grid conventions."""
import numpy as np, math
def exact_kernel_mat(K, L, b, phis=None):
    if phis is None: phis = (np.arange(K) + 0.5) / K
    w = b ** (-phis)
    W1, W2 = np.meshgrid(w, w, indexing="ij")
    diff, s = W1 - W2, W1 + W2
    t1 = np.where(np.abs(diff) < 1e-15, L, np.sin(diff * L) / np.where(diff == 0, 1, diff))
    t2 = np.sin(s * L) / s
    return (t1 + t2) / (2 * L), phis
def evq_mid(K, tau):
    u = (np.arange(K) + 0.5) / K
    return u if tau < 1e-9 else 1 - np.arcsinh((1 - u) * np.sinh(tau)) / tau
def evq_end(K, tau):
    u = np.arange(K) / K
    return u if tau < 1e-9 else 1 - np.arcsinh((1 - u) * np.sinh(tau)) / tau
def collision_C(Km):
    d = np.sqrt(np.diag(Km)); R = Km / np.outer(d, d)
    iu = np.triu_indices(len(Km), 1)
    return float(np.sum(R[iu] ** 2))

print("== 1. corrected KL order: p0 = softmax(z0), p_th = softmax(z_th) ==")
L, b, K = 512, 500_000, 32
j = np.arange(1, L + 1)
rng = np.random.default_rng(0)
# content-bearing logits: random per-channel amplitudes (fixed across tau)
A = rng.normal(size=K); B = rng.normal(size=K)
def logits(tau):
    ph = evq_mid(K, tau); w = b ** (-ph)
    return (A * np.cos(np.outer(j, w)) + B * np.sin(np.outer(j, w))).sum(axis=1) / math.sqrt(2 * K)
z0 = logits(0.0)
p0 = np.exp(z0 - z0.max()); p0 /= p0.sum()
ths = np.array([0.005, 0.01, 0.02, 0.04, 0.08])
kls = []
for th in ths:
    z = logits(math.sqrt(th))
    lp = z - (np.log(np.sum(np.exp(z - z.max()))) + z.max())
    lp0 = np.log(p0)
    kls.append(float(np.sum(p0 * (lp0 - lp))))
sK = np.polyfit(np.log(ths), np.log(kls), 1)[0]
print(f"  KL values {[f'{k:.2e}' for k in kls]}  slope in theta = {sK:.2f}  => KL = O(tau^{2*sK:.1f})")

print("\n== 2. video rows (K=16, L=32): which tau reproduces paper C_evq? ==")
paper = {100: (22.6, 14.2), 1000: (42.3, 28.0), 10_000: (55.4, 39.8), 50_000: (63.3, 47.2)}
for bb, (pg, pe) in paper.items():
    Cg = collision_C(exact_kernel_mat(16, 32, bb)[0])
    row = [f"b={bb}: C_geo={Cg:.1f} (paper {pg})"]
    for lab, tau in [("tau=5.657", 5.657), ("0.53x=3.0", 0.53 * 32 / math.sqrt(32)),
                     ("tau=1.5", 1.5), ("tau=2.0", 2.0)]:
        Ce = collision_C(exact_kernel_mat(16, 32, bb, evq_mid(16, tau))[0])
        row.append(f"{lab}: {Ce:.1f}")
    print("  " + " | ".join(row) + f"  (paper EVQ {pe})")

print("\n== 3. endpoint vs midpoint conventions on text config 0 (K=32,L=128,b=500K) ==")
for lab, gphis, ephis in [
    ("mid-geo / mid-EVQ", (np.arange(32) + .5) / 32, evq_mid(32, 5.657)),
    ("end-geo / end-EVQ", np.arange(32) / 32, evq_end(32, 5.657)),
]:
    Cg = collision_C(exact_kernel_mat(32, 128, 500_000, gphis)[0])
    Ce = collision_C(exact_kernel_mat(32, 128, 500_000, ephis)[0])
    print(f"  {lab}: C_geo={Cg:.1f} C_evq={Ce:.1f}  (paper 226.3 / 17.7)")

print("\n== 4. weighted-load control: does a GENERIC monotone density also reduce ∫q/rho²? ==")
def qfun(x):
    x = np.asarray(x, float); out = np.empty_like(x)
    s = np.abs(x) < 1e-6; out[s] = x[s] ** 4 / 45
    xs = x[~s]; out[~s] = 0.5 + np.sin(2 * xs) / (4 * xs) - (np.sin(xs) / xs) ** 2
    return out
phig = (np.arange(200000) + .5) / 200000
for (LL, tau) in [(2048, 1.414), (128, 5.657)]:
    w = qfun(LL * 500_000. ** (-phig)); W = w.mean()
    r_c = 1.414 * np.cosh(1.414 * (1 - phig)) / np.sinh(1.414) if False else None
    rc = tau * np.cosh(tau * (1 - phig)) / np.sinh(tau)
    s = 1.9; rl = 1 + s * (0.5 - phig)              # linear tilt (steepest positive)
    se = 2 * tau; re_ = se * np.exp(-se * phig) / (1 - math.exp(-se))  # exponential
    print(f"  L={LL}: ∫q={W:.4f} | cosh: {np.mean(w/rc**2):.4f} | lin s=1.9: {np.mean(w/rl**2):.4f}"
          f" | exp s={se:.1f}: {np.mean(w/re_**2):.4f}")
