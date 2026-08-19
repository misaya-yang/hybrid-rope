"""THE CENTRAL PREDICTION.

A channel contributes at relative distance D only if its phase at D was OBSERVED during
training.  Ordinary contiguous training observes phases only up to D = L_train, so a
channel is 'usable' beyond L_train iff it has wrapped (lambda <= L_train).  Query-gap /
virtual-position training raises that horizon to G: usable iff lambda <= G.

Utility of a table at deployment ceiling M = effective rank (participation ratio) of the
usable phase-feature matrix  Phi = [cos(w_k D), sin(w_k D)]_{D in [0,M], usable k}.
More effective rank = more distinguishable relative positions.

Prediction: tau* = argmax_tau effrank(tau; G, M) DECREASES monotonically in G, and the
argmax approaches tau=0 (uniform, i.e. geometric retargeted) as G -> M.
"""
import numpy as np
TWO_PI = 2 * np.pi

def evq(base, dh, tau):
    K = dh // 2; u = (np.arange(K) + 0.5) / K
    if tau == 0:
        phi = u
    else:
        phi = 1 - np.arcsinh((1 - u) * np.sinh(tau)) / tau
    return base ** (-phi)

def eff_rank(om, M, n=3000):
    if len(om) == 0:
        return 0.0
    D = np.linspace(0, M, n)
    P = np.concatenate([np.cos(np.outer(D, om)), np.sin(np.outer(D, om))], axis=1)
    P = P - P.mean(0, keepdims=True)
    s = np.linalg.svd(P, compute_uv=False) ** 2
    s = s / s.sum()
    return float(np.exp(-(s * np.log(s + 1e-300)).sum()))   # participation entropy

BASE, DH, L, M = 500_000.0, 128, 4096, 32_768
taus = np.arange(0.0, 6.01, 0.125)

print("=" * 96)
print(f"tau* as a function of the phase-coverage horizon G   (base={BASE:.0f}, d={DH}, "
      f"L_train={L}, M={M})")
print("=" * 96)
print(f"{'G (coverage)':>14s} {'G/L_train':>10s} {'usable @tau*':>13s} {'tau*':>7s} "
      f"{'effrank@tau*':>13s} {'effrank@tau=0':>14s} {'gain from warp':>16s}")
print("-" * 96)
rows = []
for G in [L, 2 * L, 4 * L, 8 * L, 16 * L, 32 * L]:
    best = (-1, None, None)
    for t in taus:
        om = evq(BASE, DH, t)
        lam = TWO_PI / om
        om_u = om[lam <= G]
        er = eff_rank(om_u, M)
        if er > best[0]:
            best = (er, t, len(om_u))
    om0 = evq(BASE, DH, 0.0); lam0 = TWO_PI / om0
    er0 = eff_rank(om0[lam0 <= G], M)
    er, t, nu = best
    rows.append((G, t, er, er0))
    print(f"{G:>14d} {G/L:>10.0f}x {nu:>13d} {t:>7.3f} {er:>13.2f} {er0:>14.2f} "
          f"{(er-er0)/max(er0,1e-9):>15.1%}")
print("-" * 96)
print("prediction check: tau* is", "MONOTONE DECREASING in G  ✓"
      if all(rows[i][1] >= rows[i+1][1] for i in range(len(rows)-1))
      else "NOT monotone decreasing  ✗ (theory needs revision)")
print("and the payoff of warping (last column) shrinks as coverage grows:",
      f"{rows[0][2]/rows[0][3]-1:.1%} at G=L_train  ->  {rows[-1][2]/rows[-1][3]-1:.1%} at G=32*L_train")
print()
print("If this holds empirically, then: EVQ's benefit is a SUBSTITUTE for phase coverage,")
print("not a complement.  Supply coverage in the training protocol and the optimal table")
print("moves back toward uniform-over-the-truncated-span -- which is why retargeted uniform")
print("FMRoPE beat cosh once both were retargeted.")
