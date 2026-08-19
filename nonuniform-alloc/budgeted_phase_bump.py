"""CORRECTION to the naive 'free budget' reading, and the design it forces.

A channel with lambda >> M is NOT function-free.  Its rotation is ~identity inside any
realizable context, so it behaves exactly like a NoPE channel: it carries CONTENT
matching, position-independently.  (That is why Oka et al. find that replacing
low-frequency RoPE dims with NoPE is near-free -- it is a no-op, not a deletion.)

So the dead band is CONVERTIBLE, not free, and the exchange rate is exact:
    in-window distortion   eps_k = max_{D<=L} |cos(w'_k D) - cos(w_k D)| ~= 1 - cos(w'_k L)
    long-range resolution  = w'_k * M  radians swept across the deployment ceiling
"""
import numpy as np
TWO_PI = 2 * np.pi
def geo(base, dh): return base ** (-(2 * np.arange(dh // 2) / dh))
def evq(base, dh, tau):
    K = dh // 2; u = (np.arange(K) + 0.5) / K
    return base ** (-(1 - np.arcsinh((1 - u) * np.sinh(tau)) / tau))
def w_ceiling(eps, L): return np.arccos(np.clip(1 - eps, -1, 1)) / L

BASE, DH, L, M = 500_000.0, 128, 4096, 32_768
om_nat = geo(BASE, DH); lam_nat = TWO_PI / om_nat
GEO_SPACING = np.diff(np.sort(np.log(om_nat))).mean()      # 0.2050 nats

print("=" * 104)
print(f"A. Exchange rate  (OLMo-2: base={BASE:.0f}, d_head={DH}, L_train={L}, M={M})")
print("=" * 104)
print(f"{'eps in-window':>14s} {'lambda floor':>13s} {'phase @M':>11s} {'as x pi':>8s}"
      f" {'ruler slots*':>13s} {'dead ch. movable':>18s}")
print("-" * 104)
for eps in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    w = w_ceiling(eps, L); lf = TWO_PI / w
    # * how many distinct channels fit between the budget floor and 4M at geometric spacing
    slots = max(0, int(np.floor(np.log(4 * M / lf) / GEO_SPACING)) + 1) if lf < 4 * M else 0
    movable = int((lam_nat > lf).sum())
    print(f"{eps:>14.3f} {lf:>13.0f} {w*M:>10.3f}r {w*M/np.pi:>8.2f} {slots:>13d} {movable:>18d}")
print("-" * 104)
print("* 'ruler slots' = how many channels fit between the eps floor and 4M without going")
print("  below the native geometric log-spacing (0.2050 nats).  This is the HARD LIMIT on")
print("  how much of the dead tail can be converted into useful long-range structure.")
print()
print("  => the convertible budget is NOT the 18-22 dead channels.  At any eps a mature")
print("     model would tolerate, only a HANDFUL fit in the useful band; the rest must stay")
print("     where they are (still doing content work) or be packed into near-duplicates.")
print("     One ruler is enough for unambiguity -- but you cannot buy RESOLUTION this way.")
print()

print("=" * 104)
print("B. What the model already has, and what it is merely never trained on")
print("=" * 104)
for Dmax in [4096, 8192, 16384, 32768]:
    wrapped = (lam_nat <= Dmax).sum()
    usable_if_covered = (lam_nat <= 2 * Dmax).sum()     # monotone ruler needs lambda >= 2D
    print(f"  at D={Dmax:>6d}: {wrapped:2d}/64 channels have wrapped (alias-safe with NO extra"
          f" training);  {usable_if_covered:2d}/64 are phase-complete once coverage reaches {2*Dmax}")
print()
print("  Native ALREADY owns 10 regime-II channels that could serve as rulers to 32K.")
print("  Ordinary contiguous 4K training simply never shows them a phase beyond 4K.")
print("  => the first thing to try on a mature model is NOT a new table.  It is coverage.")
print()

print("=" * 104)
print("C. Worst-case in-window cosine distortion of each candidate intervention")
print("=" * 104)
def worst(om_new, n=4096):
    D = np.linspace(0, L, n)
    return np.abs(np.cos(np.outer(om_new, D)) - np.cos(np.outer(om_nat, D))).max()
def n_moved(om_new): return int((~np.isclose(np.sort(om_new), np.sort(om_nat))).sum())
cands = [("do nothing (coverage only)", om_nat),
         ("EVQ-Cosh tau=2.00",          evq(BASE, DH, DH / np.sqrt(L))),
         ("FMRoPE literal base=M",      geo(float(M), DH))]
# BPB: bump only the channels slower than the eps floor, packing them just above it
for eps in [0.02, 0.05]:
    lf = TWO_PI / w_ceiling(eps, L)
    om = om_nat.copy(); lam = TWO_PI / om
    idx = np.where(lam > lf)[0]
    if len(idx):
        tgt = np.exp(np.linspace(np.log(lf), np.log(lam[idx].max()), len(idx)))
        om[idx] = TWO_PI / tgt
    cands.append((f"BPB eps={eps} (bump {len(idx)} ch.)", np.sort(om)[::-1]))
print(f"{'intervention':<32s} {'moved':>6s} {'worst |dcos| in-window':>24s} {'theorem applies?':>18s}")
print("-" * 104)
for nm, om in cands:
    print(f"{nm:<32s} {n_moved(om):>6d} {worst(om):>24.4f} "
          f"{('YES' if n_moved(om) else 'no -- table unchanged'):>18s}")
print("-" * 104)
print("EVQ and FMRoPE both reach |dcos| = 2.0: a full sign reversal on some channel at some")
print("IN-WINDOW distance.  That is the coordinate-transplant cost, and it is why 63/64-pair")
print("interventions destroy mature-model in-window behaviour no matter how they are trained.")
