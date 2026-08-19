"""Spectral-regime accounting for RoPE frequency tables.

Definitions (all in terms of channel wavelength lambda_k = 2*pi/omega_k):
  n_k(D)   = D / lambda_k          -- number of full periods the channel completes
                                      over relative distance D.
  Regime I   (wrapped)   : n_k(L_train) >= 1        phase at any D is an alias of a
                                                    trained phase  -> extrapolation-safe,
                                                    distance-ambiguous
  Regime II  (resolving) : n_k(L_max) >= 1 > n_k(L_train)
                                                    monotone, high distance resolution
                                                    inside the window; phase beyond
                                                    L_train is UNOBSERVED
  Regime III (dead)      : n_k(L_max) < 1           near-constant over every realizable
                                                    context -> mutually redundant,
                                                    softmax-invariant
"""
import numpy as np

TWO_PI = 2 * np.pi


def geo_table(base, d_head, convention="std"):
    K = d_head // 2
    k = np.arange(K)
    phi = (2 * k / d_head) if convention == "std" else ((k + 0.5) / K)
    return base ** (-phi)


def evq_table(base, d_head, tau, grid="midpoint"):
    K = d_head // 2
    k = np.arange(K)
    u = (k + 0.5) / K if grid == "midpoint" else k / (K - 1)
    phi = 1.0 - np.arcsinh((1 - u) * np.sinh(tau)) / tau
    return base ** (-phi)


def regimes(omega, L_train, L_max):
    lam = TWO_PI / omega
    wrapped = lam <= L_train
    dead = lam > L_max
    resolving = ~wrapped & ~dead
    return wrapped, resolving, dead, lam


def report(name, omega, L_train, L_max):
    w, r, d, lam = regimes(omega, L_train, L_max)
    K = len(omega)
    print(f"{name:<34s} K={K:3d}  wrapped {w.sum():3d} ({w.mean():5.1%})   "
          f"resolving {r.sum():3d} ({r.mean():5.1%})   dead {d.sum():3d} ({d.mean():5.1%})"
          f"   lam range [{lam.min():.1f}, {lam.max():.3g}]")
    return w.mean(), r.mean(), d.mean()


CONFIGS = [
    # name,                base,     d_head, L_train, L_max
    ("OLMo-2 1.485B",      500_000,  128,     4096,   32_768),
    ("LLaMA-3-8B",         500_000,  128,     8192,   131_072),
    ("Qwen2.5 (b=1e6)",  1_000_000,  128,    32_768,  131_072),
    ("DeepSeek-V3 MLA",    500_000,   64,     4096,   131_072),   # d_rope=64 -> 32 pairs
    ("paper 454M MHA",     500_000,   64,     2048,   16_384),
]

print("=" * 118)
print("A. Geometric tables: how the channel budget is spent")
print("=" * 118)
for nm, b, dh, lt, lm in CONFIGS:
    report(f"{nm}  geo", geo_table(b, dh), lt, lm)

print()
print("=" * 118)
print("B. Same configs under EVQ-Cosh at the deployed tau = d_head/sqrt(L_train)")
print("=" * 118)
for nm, b, dh, lt, lm in CONFIGS:
    tau = dh / np.sqrt(lt)
    report(f"{nm}  evq(tau={tau:.2f})", evq_table(b, dh, tau), lt, lm)

print()
print("=" * 118)
print("C. OLMo-2: where does EVQ take budget FROM and give it TO?")
print("=" * 118)
b, dh, lt, lm = 500_000, 128, 4096, 32_768
tau = dh / np.sqrt(lt)
g = geo_table(b, dh); e = evq_table(b, dh, tau)
for nm, om in (("geo", g), (f"evq tau={tau:.3f}", e)):
    w, r, d, lam = regimes(om, lt, lm)
    print(f"  {nm:<16s} wrapped={w.sum():2d}  resolving={r.sum():2d}  dead={d.sum():2d}")
print()
print("  per-decade channel counts (log10 lambda bin -> geo / evq):")
lg, le = np.log10(TWO_PI / g), np.log10(TWO_PI / e)
edges = np.arange(0, 8.5, 1.0)
hg, _ = np.histogram(lg, edges); he, _ = np.histogram(le, edges)
for i in range(len(edges) - 1):
    marker = ""
    lo, hi = 10 ** edges[i], 10 ** edges[i + 1]
    if hi <= lt: marker = " <- wrapped"
    elif lo >= lm: marker = " <- DEAD"
    elif lo >= lt and hi <= lm: marker = " <- resolving"
    print(f"    lambda 1e{edges[i]:.0f}-1e{edges[i+1]:.0f}: geo {hg[i]:3d}   evq {he[i]:3d}"
          f"   delta {he[i]-hg[i]:+3d}{marker}")
