#!/usr/bin/env python3
"""EVQ vs three-band unification: the one-question numerical table.

All numbers computed from experiments/curvature_20260910/tables.py (the ground
truth table algebra), Qwen2.5-3B config: theta=1e6, W=32768, head_dim=128,
K=64, band [23,40] (n=17).  Nothing here is typed by hand from memory.
"""
import math
import sys

import numpy as np

sys.path.insert(0, "/Users/yang/projects/hybrid-rope")
from experiments.curvature_20260910 import tables as T

CFG = T.QWEN25_3B
K = T.K
NATIVE_GAP = math.log(CFG["theta"]) / K   # log(nu_j / nu_{j+1}) on the native grid


def stats(m):
    m = np.asarray(m, dtype=np.float64)
    nu = T.m_to_inv_freq(m, CFG["theta"])
    gaps = np.log(nu[:-1] / nu[1:])       # a_j = ln(nu_j/nu_{j+1})
    return dict(
        held=int(np.sum(m < 1e-12)), plateau=int(np.sum(m > 1 - 1e-12)),
        sum_m=float(m.sum()), span=float(m[-1] - m[0]),
        m0=float(m[0]), m63=float(m[-1]),
        gap_min=float(gaps.min()), gap_max=float(gaps.max()),
        gap_min_at=int(np.argmin(gaps)), gap_max_at=int(np.argmax(gaps)),
        gaps=gaps)


def show(title, m):
    s = stats(m)
    print(f"{title:22s} held={s['held']:2d} plat={s['plateau']:2d} "
          f"sum_m={s['sum_m']:9.5f} span={s['span']:8.6f} "
          f"m0={s['m0']:+.6f} m63={s['m63']:+.6f} "
          f"gap_min={s['gap_min']:.6f}@{s['gap_min_at']:2d} "
          f"gap_max={s['gap_max']:.6f}@{s['gap_max_at']:2d}")
    return s


print(f"native gap ln(theta)/K = {NATIVE_GAP:.6f}; ln S = {T.LN_S:.6f}; "
      f"ln(theta)/ln(S) = {math.log(CFG['theta'])/T.LN_S:.4f}\n")

print("== 1. THE NINE TABLES (Qwen cfg, band [23,40]) ==")
rows = {}
rows["mrpro_n17"] = show("mrpro_n17", T.m_mrpro(17))
rows["beta_b1 (BM)"] = show("beta_b1 (BM)", T.m_incr_beta(1.0))
rows["yarn_lin"] = show("yarn_lin", T.m_yarn())
rows["evq_deploy_t1"] = show("evq_deploy_t1", T.m_evq_deployed(1.0))
rows["evq_deploy_t1.414"] = show("evq_deploy_t1.414", T.m_evq_deployed(1.414))
rows["evq_shift_t1"] = show("evq_shift_t1", T.m_evq_shift(1.0))
rows["power_p2"] = show("power_p2", T.m_power_shift(2))
rows["power_p4"] = show("power_p4", T.m_power_shift(4))
rows["native"] = show("native", T.m_native())

print("\n== 1b. what each family conserves: span vs sum m ==")
m_mr = T.m_mrpro(17)
m_bm = T.m_incr_beta(1.0)
print(f"MrPro:  sum_m = 23 + sum_q m_q, sum_q m_q = {m_mr[23:41].sum():.6f} "
      f"(= (n+2)/3 = {19/3:.6f} for the quadratic ramp, n=17)")
print(f"BM:     sum_q m_q = {m_bm[23:41].sum():.6f} (shape moves sum m: "
      f"{rows['mrpro_n17']['sum_m']:.4f} -> {rows['beta_b1 (BM)']['sum_m']:.4f})")
print(f"band span m_40 - m_23 = {m_mr[40] - m_mr[23]:.6f} (exact 1 for every shape)")

m_e = T.m_evq_deployed(1.414)
print(f"EVQ-deployed tau=1.414: span m63-m0 = {m_e[-1]-m_e[0]:.6f} "
      f"(short of 1 by {1-(m_e[-1]-m_e[0]):.6f} lnS = "
      f"{(1-(m_e[-1]-m_e[0]))*T.LN_S:.6f} nats)")
m_s = T.m_evq_shift(1.0)
print(f"EVQ-shift tau=1: sum_m = {m_s.sum():.6f} (redistribution inside native "
      f"span: sum m ~ 0, not the budget); span = {m_s[-1]-m_s[0]:.6f}")

print("\n== 1c. EVQ family limits and where it crosses MrPro's sum m ==")
print(f"{'tau':>8s} {'held':>4s} {'plat':>4s} {'sum_m':>9s} {'span':>9s} "
      f"{'m0':>9s} {'m63':>9s} {'gap_min':>9s} {'gap_max':>9s}")
for tau in (0.0, 0.25, 0.5, 0.75, 1.0, 1.414, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0):
    s = stats(T.m_evq_deployed(tau))
    print(f"{tau:8.3f} {s['held']:4d} {s['plateau']:4d} {s['sum_m']:9.5f} "
          f"{s['span']:9.6f} {s['m0']:9.6f} {s['m63']:9.6f} "
          f"{s['gap_min']:9.6f} {s['gap_max']:9.6f}")

# bisect tau where EVQ-deployed sum_m == MrPro's 29.3333
target = float(T.m_mrpro(17).sum())
lo, hi = 0.0, 32.0
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if T.m_evq_deployed(mid).sum() > target:
        lo = mid
    else:
        hi = mid
tau_star = 0.5 * (lo + hi)
s = stats(T.m_evq_deployed(tau_star))
print(f"\nEVQ-deployed with MrPro's sum_m: tau* = {tau_star:.4f}, "
      f"span = {s['span']:.6f} (MrPro span = 1.000000), "
      f"gap range [{s['gap_min']:.6f}, {s['gap_max']:.6f}] "
      f"vs MrPro [{rows['mrpro_n17']['gap_min']:.6f}, "
      f"{rows['mrpro_n17']['gap_max']:.6f}]")

print("\n== 2. three-band as weak limit of a companding (Gaussian-width sweep) ==")
# MrRoPE = companding whose slope-density is the atomic measure sum_q eps_q
# delta(v_q).  A smooth companding with slope concentrated on the band slots
# converges to it as the width -> 0.  (EVQ has no width knob: its density is
# fixed-form, full support for every finite tau.)
eps = np.diff(m_mr[22:41])          # increments eps_q at slots 23..40
v = np.arange(23, 41) / K           # jump LOCATIONS: boundary between slot q-1, q
u = np.linspace(1 / (2 * K), 1 - 1 / (2 * K), K)
print(f"{'sigma':>8s} {'max|Delta m|':>14s} {'held':>5s} {'plat':>5s} {'sum_m':>9s}")
for sig in (1.0, 0.4, 0.2, 0.1, 0.05, 0.02, 0.008, 0.003):
    def phi(uu):
        return sum(e * 0.5 * (1 + math.erf((uu - vv) / (sig * math.sqrt(2))))
                   for e, vv in zip(eps, v))
    ph = np.array([phi(uu) for uu in u])
    m_phi = ph / ph[-1] * 1.0            # rescale so endpoint = 1 (companding
    m_phi = m_phi - m_phi[0]             #  normalised on the sampled grid)
    m_phi = m_phi / m_phi[-1]
    d = np.abs(m_phi - m_mr).max()
    s = stats(m_phi)
    print(f"{sig:8.2f} {d:14.3e} {s['held']:5d} {s['plateau']:5d} {s['sum_m']:9.5f}")

print("\n== 3. KKT numerics: turns r_j and native marginal at m=0 ==")
r = CFG["window"] * T.native_inv_freq(CFG["theta"]) / (2 * math.pi)
for j in (0, 4, 8, 12, 16, 20, 23, 24, 39, 40):
    print(f"slot {j:2d}: r_j = W*w_j/2pi = {r[j]:9.3f} turns")

# smooth native cost C(e) = e^2/3, e = r(1 - S^-m): analytic first derivative
# dC/dm = (2/3) e * de/dm, de/dm = r lnS S^-m  ->  at m=0: e=0 -> dC/dm = 0 exactly.
# The one-sided difference quotient C(h)/h -> 0 linearly in h (second-order cost).
print("\nd/dm C(e) at m=0 under the quadratic drift model:")
for j in (0, 8, 16):
    e0 = 0.0
    analytic = (2.0 / 3.0) * e0 * (r[j] * T.LN_S * 1.0)   # = 0 exactly at m=0
    print(f"  slot {j:2d}: analytic dC/dm|_0 = {analytic:.3e};  "
          + "  one-sided C(h)/h for h="
          + ", ".join(f"{h:g}->{((r[j]*(1-4.0**-h))**2/3.0)/h:.2e}"
                      for h in (1e-3, 1e-5, 1e-7)))


print("\n== 5. the leak arm (pre-declared discriminator construction) ==")
for eta in (0.005, 0.02):
    m_leak = np.zeros(K)
    m_leak[:23] = eta
    q = np.arange(18, dtype=np.float64)          # q = 0..17, m_q = q(q+1)/306
    m_leak[23:41] = eta + (1 - eta) * q * (q + 1) / (17 * 18)
    m_leak[41:] = 1.0
    nu = T.m_to_inv_freq(m_leak, CFG["theta"])
    s = stats(m_leak)
    print(f"leak eta={eta:5.3f}: sum_m={s['sum_m']:.4f} span={s['span']:.4f} "
          f"nu strictly descending: {bool((np.diff(nu) < 0).all())}, "
          f"band start m_23={m_leak[23]:.4f}, "
          f"gap range [{s['gap_min']:.6f},{s['gap_max']:.6f}] "
          f"vs MrPro [0.215867,0.369900]")

print("\n== 5a2. leak variant (b): band-adjacent, span EXACTLY 1 ==")
for eta in (0.005, 0.02):
    m_leak = np.zeros(K)
    m_leak[17:23] = eta
    q = np.arange(18, dtype=np.float64)
    m_leak[23:41] = eta + (1 - eta) * q * (q + 1) / (17 * 18)
    m_leak[41:] = 1.0
    nu = T.m_to_inv_freq(m_leak, CFG["theta"])
    s = stats(m_leak)
    print(f"leakB eta={eta:5.3f}: sum_m={s['sum_m']:.4f} span={s['span']:.6f} "
          f"nu strictly descending: {bool((np.diff(nu) < 0).all())}, "
          f"gap range [{s['gap_min']:.6f},{s['gap_max']:.6f}]")

print("\n== 5b. shift-EVQ profile detail (tau=1) ==")
print(f"  m range [{m_s.min():+.4f}, {m_s.max():+.4f}], argmin={int(m_s.argmin())}, "
      f"argmax={int(m_s.argmax())}; slots with m>0: {int((m_s > 0).sum())}")
print(f"  exponent span: phi63-phi0 = {T.evq_phi(1.0)[-1] - T.evq_phi(1.0)[0]:.6f} "
      f"vs native 63/64 = {63/64:.6f}")
print(f"  sum(phi - u) = {(T.evq_phi(1.0) - T.uniform_phi()).sum():.6f} "
      f"(mean shift {np.mean(T.evq_phi(1.0) - T.uniform_phi()):.6f} exponent units)")

