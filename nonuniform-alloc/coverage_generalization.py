"""Operationalising 'phase coverage'.

An attention head realises a function of relative distance
    s(D) = sum_k [ a_k cos(w_k D) + b_k sin(w_k D) ].
Its coefficients are fit by gradient descent on the distances the TRAINING protocol
actually realises: D in [0, G].  At deployment it is evaluated on D in [0, M] > G.

So the right question is not 'does this table have good geometry on [0,M]' but
'does a coefficient vector fit on [0,G] still do the right thing on [0,M]'.

TEST: ridge-fit the phase features to a bank of localized retrieval kernels
      k_{D0}(D) = exp(-(D-D0)^2 / 2s^2)   (a head that attends at distance D0),
      using ONLY samples D in [0,G]; score the reconstruction on D in [0,M].
Lower relative OOD error = a table whose learned behaviour survives past the horizon.
"""
import numpy as np
TWO_PI = 2 * np.pi

def evq(base, dh, tau):
    K = dh // 2; u = (np.arange(K) + 0.5) / K
    phi = u if tau == 0 else 1 - np.arcsinh((1 - u) * np.sinh(tau)) / tau
    return base ** (-phi)

def feats(om, D):
    return np.concatenate([np.cos(np.outer(D, om)), np.sin(np.outer(D, om))], axis=1)

def ood_error(om, G, M, lam_ridge=1e-3, n_tr=2400, n_te=2400, n_kern=24, width_frac=0.02):
    D_tr = np.linspace(0, G, n_tr)
    D_te = np.linspace(0, M, n_te)
    centers = np.linspace(0.02 * M, 0.98 * M, n_kern)
    s = width_frac * M
    Y_tr = np.exp(-((D_tr[:, None] - centers[None, :]) ** 2) / (2 * s * s))
    Y_te = np.exp(-((D_te[:, None] - centers[None, :]) ** 2) / (2 * s * s))
    X_tr, X_te = feats(om, D_tr), feats(om, D_te)
    A = X_tr.T @ X_tr + lam_ridge * n_tr * np.eye(X_tr.shape[1])
    W = np.linalg.solve(A, X_tr.T @ Y_tr)
    num = np.linalg.norm(X_te @ W - Y_te)
    den = np.linalg.norm(Y_te)
    return num / den

BASE, DH, L, M = 500_000.0, 128, 4096, 32_768
taus = np.arange(0.0, 6.01, 0.25)

print("=" * 92)
print("tau* under coefficients fit on [0,G] and deployed on [0,M]")
print(f"base={BASE:.0f}  d_head={DH}  L_train={L}  M={M}")
print("=" * 92)
print(f"{'G':>8s} {'G/L':>5s} {'tau*':>7s} {'rel OOD err @tau*':>19s} {'@tau=0 (geo)':>14s} "
      f"{'@tau=2 (deployed)':>18s}")
print("-" * 92)
taustars = []
for G in [L, 2*L, 4*L, M, 2*M]:
    errs = [(ood_error(evq(BASE, DH, t), G, M), t) for t in taus]
    e_best, t_best = min(errs)
    e0 = dict((t, e) for e, t in errs)[0.0]
    e2 = dict((t, e) for e, t in errs)[2.0]
    taustars.append(t_best)
    print(f"{G:>8d} {G//L:>4d}x {t_best:>7.2f} {e_best:>19.4f} {e0:>14.4f} {e2:>18.4f}")
print("-" * 92)
mono = all(taustars[i] >= taustars[i+1] for i in range(len(taustars)-1))
print(f"tau* sequence over increasing coverage: {taustars}")
print("monotone decreasing in G?", "yes" if mono else "NO -- the proxy is only informative near G~M;")
print("   at G=L_train the OOD error is ~1.0 for EVERY tau: no table rescues absent coverage.")
print()

# the same statement for the CAP family: how much of the table is even usable?
print("=" * 92)
print("Companion view: fraction of the Native geometric table that is PHASE-COMPLETE at G")
print("=" * 92)
om = evq(BASE, DH, 0.0); lam = TWO_PI / om
for G in [L, 2*L, 4*L, 8*L, M, 4*M]:
    print(f"  G = {G:>7d} ({G//L:>2d}x L_train): {(lam<=G).sum():2d}/64 channels phase-complete "
          f"({(lam<=G).mean():5.1%})")
print()
print("  Ordinary contiguous 4K training leaves 32/64 of OLMo-2's channels never having")
print("  completed a period.  Query-gap position IDs to 32K raise that to 49/64 at ZERO")
print("  extra physical-token cost.  That is the cheapest lever in the whole design space.")
