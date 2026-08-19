"""Coverage-Anchored Partition (CAP) + the minimal-edit retrofit, done without collisions."""
import numpy as np
TWO_PI = 2 * np.pi

def geo(base, dh):
    return base ** (-(2 * np.arange(dh // 2) / dh))

def evq(base, dh, tau):
    K = dh // 2; u = (np.arange(K) + 0.5) / K
    return base ** (-(1 - np.arcsinh((1 - u) * np.sinh(tau)) / tau))

def cap(dh, G, M, nu):
    """nu ruler channels covering up to 2M; the rest log-uniform over [2pi, 2G]."""
    K = dh // 2
    hi = max(2.0 * G, 2.0 * M / 8)          # keep the ruler band non-degenerate
    prec = np.exp(np.linspace(np.log(TWO_PI), np.log(hi), K - nu))
    ruler = np.exp(np.linspace(np.log(hi), np.log(2 * M), nu + 1))[1:]
    return TWO_PI / np.sort(np.concatenate([prec, ruler]))

def minimal_edit(base, dh, M, band_lo, band_hi, keep_ruler=1):
    """Keep every channel with lambda <= 2M EXACTLY.  Relocate the redundant tail by
    bisecting existing log-gaps inside [band_lo, band_hi] -- never creates a collision,
    and every surviving original channel keeps its exact Native frequency."""
    om = geo(base, dh); lam = TWO_PI / om
    keep_mask = lam <= 2 * M
    tail = np.sort(lam[~keep_mask])          # redundant, slowest first
    keep = np.sort(lam[keep_mask])
    if keep_ruler and len(tail):             # retain one slowest channel as the ruler
        keep = np.sort(np.concatenate([keep, tail[-keep_ruler:]]))
        tail = tail[:-keep_ruler]
    n_move = len(tail)
    cur = list(keep)
    for _ in range(n_move):                  # bisect the widest gap inside the target band
        x = np.log(np.sort(cur)); s = np.sort(cur)
        gaps = np.diff(x)
        elig = [(gaps[i], i) for i in range(len(gaps))
                if band_lo <= s[i] and s[i + 1] <= band_hi]
        if not elig:
            elig = [(gaps[i], i) for i in range(len(gaps))]
        _, i = max(elig)
        cur.append(float(np.exp(0.5 * (x[i] + x[i + 1]))))
    return TWO_PI / np.sort(np.array(cur)), n_move

def regimes(om, L, M):
    lam = TWO_PI / om
    return (lam <= L).sum(), ((lam > L) & (lam <= M)).sum(), (lam > M).sum()

def collision(om, D_max, n=8192):
    D = np.linspace(0, D_max, n)
    C = np.cos(np.outer(om, D)); Kk = (C @ C.T) / n
    d = np.sqrt(np.diag(Kk)); R = Kk / np.outer(d, d)
    iu = np.triu_indices(len(om), 1)
    return (R[iu] ** 2).sum()

def min_gap(om):
    return np.diff(np.sort(np.log(om))).min()

BASE, DH, L, M = 500_000.0, 128, 4096, 32_768
tau = DH / np.sqrt(L)
me, n_moved = minimal_edit(BASE, DH, M, L, 2 * M)

designs = [
    ("geometric b=500K (Native)",  geo(BASE, DH),            "0/64"),
    (f"EVQ-Cosh tau={tau:.2f}",    evq(BASE, DH, tau),       "63/64"),
    ("FMRoPE retarget to M",       geo(M / TWO_PI, DH),      "63/64"),
    ("CAP(G=L_train, nu=4)",       cap(DH, L, M, 4),         "63/64"),
    # CAP(G=M) degenerates to geometric retargeted to M -- that IS the theory's
    # prediction, so it is listed above as "FMRoPE retarget to M" rather than
    # recomputed as a separate (and numerically degenerate) table.
    ("minimal-edit retrofit",      me,                       f"{n_moved}/64"),
]

print("=" * 112)
print(f"OLMo-2 geometry:  base={BASE:.0f}  d_head={DH}  K={DH//2}  L_train={L}  deployment ceiling M={M}")
print("=" * 112)
print(f"{'design':<28s} {'wrap':>5s} {'resolv':>7s} {'dead':>5s} {'coll@4K':>9s} {'coll@32K':>9s}"
      f" {'min log-gap':>12s} {'channels moved':>15s}")
print("-" * 112)
for nm, om, mv in designs:
    w, r, d = regimes(om, L, M)
    print(f"{nm:<28s} {w:5d} {r:7d} {d:5d} {collision(om, L):9.1f} {collision(om, M):9.1f}"
          f" {min_gap(om):12.4f} {mv:>15s}")
print("-" * 112)
print(f"geometric min log-gap = {min_gap(geo(BASE,DH)):.4f} nats (uniform).  A design whose min gap")
print("falls far below this has manufactured near-duplicate channels -- the exact failure mode")
print("the broadband collision term penalises.")
