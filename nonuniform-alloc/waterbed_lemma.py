"""Why a one-parameter monotone warp cannot get both windows.

rho_tau(phi) = tau*cosh(tau*(1-phi))/sinh(tau)  is strictly DECREASING in phi.
phi is log-frequency exponent: phi=0 -> fastest channel, phi=1 -> slowest.
Regime I (wrapped) sits at small phi, III (dead) at large phi, II (resolving) between.
Claim: d rho_tau(phi) / d tau < 0 for every phi above a single crossover phi_c(tau),
so ANY tau that thins the dead band also thins the resolving band.
"""
import numpy as np

def rho(tau, phi):
    if tau == 0:
        return np.ones_like(phi)
    return tau * np.cosh(tau * (1 - phi)) / np.sinh(tau)

def phi_of_lambda(lam, base):
    # lambda = 2*pi*base**phi  ->  phi = log(lam/2pi)/log(base)
    return np.log(lam / (2 * np.pi)) / np.log(base)

BASE, DH, LTR, LMAX = 500_000.0, 128, 4096, 32_768
phi_L  = phi_of_lambda(LTR,  BASE)   # boundary wrapped | resolving
phi_M  = phi_of_lambda(LMAX, BASE)   # boundary resolving | dead
print(f"OLMo-2 (b={BASE:.0f}, L_train={LTR}, L_max={LMAX}):")
print(f"  regime I  (wrapped)   phi in [0.000, {phi_L:.3f})   = {phi_L:6.1%} of the exponent axis")
print(f"  regime II (resolving) phi in [{phi_L:.3f}, {phi_M:.3f})   = {phi_M-phi_L:6.1%}")
print(f"  regime III(dead)      phi in [{phi_M:.3f}, 1.000]   = {1-phi_M:6.1%}")
print()

print("  tau |  mass in I  |  mass in II  |  mass in III  | II relative to geometric")
print("  " + "-" * 74)
grid = np.linspace(0, 1, 200001)
def mass(tau, lo, hi):
    m = (grid >= lo) & (grid < hi)
    return np.trapezoid(rho(tau, grid[m]), grid[m])
base_II = phi_M - phi_L
for tau in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
    mI, mII, mIII = mass(tau, 0, phi_L), mass(tau, phi_L, phi_M), mass(tau, phi_M, 1.0)
    print(f"  {tau:4.1f} | {mI:10.3f}  | {mII:11.3f}  | {mIII:12.3f}  | {mII/base_II:+.2f}x")

print()
print("  => every tau>0 that reduces the dead mass ALSO reduces the resolving mass.")
print("     the cosh family has one knob and three regimes; it cannot separate II from III.")
print()

# minimal unambiguous ruler
print("=" * 78)
print("Minimal ruler: one channel of wavelength lambda gives a strictly monotone,")
print("unambiguous distance code on [0, lambda/2].  To cover [0, M] you need lambda >= 2M.")
print("LeRoPE's learned dominant band sits at lambda = 2.205 * L_train  (Karypis et al.).")
print("2.205 vs the predicted 2.0: the minimal monotone ruler for its own training window.")
print("=> the learned optimum is a MINIMAL ruler, not a broad low-frequency band.")
print()
for M in [4096, 8192, 16384, 32768, 131072]:
    lam = 2 * M
    phi = phi_of_lambda(lam, BASE)
    k_geo = phi * (DH // 2)
    print(f"  cover [0,{M:6d}] -> ruler lambda={lam:7d}  (phi={phi:.3f}, i.e. geometric channel #{k_geo:.1f}/64)")
print()
print("  a geometric b=500K table devotes 22/64 channels to lambda > 32768.")
print("  the unambiguity job needs ONE of them.  the other 21 are redundant copies.")
