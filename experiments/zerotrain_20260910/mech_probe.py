#!/usr/bin/env python3
"""Mechanism probe: what a frequency table does to the RoPE phase code at the
TEST distance, computed per slot, for the nine measured OLMo arms.

No GPU, no fitting.  Everything here is arithmetic on the tables.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.zerotrain_20260910.why import MEASURED, OLMO, table_of  # noqa: E402

K = 64
TH = OLMO["theta"]
W = OLMO["window"]
D = 16384.0
SCALE = 4.0


def profile(name):
    m = table_of(name)
    omega = TH ** (-np.arange(K) / K)
    nu = omega * SCALE ** (-m)
    t_W = omega * W / (2 * math.pi)          # turns in the trained window, native
    T_D = nu * D / (2 * math.pi)             # turns at the TEST distance, table m
    T_D_native = omega * D / (2 * math.pi)   # turns at the test distance, native
    return dict(m=np.asarray(m, float), omega=omega, nu=nu,
                t_W=t_W, T_D=T_D, T_D_native=T_D_native)


def main():
    names = [n for n, _ in MEASURED]
    acc = {n: a for n, a in MEASURED}
    P = {n: profile(n) for n in names}

    print("=== per-slot table dump (OLMo: theta=5e5 W=4096 D=16384) ===")
    omega = P[names[0]]["omega"]
    t_W = P[names[0]]["t_W"]
    hdr = f"{'j':>3} {'t_W':>8} {'4*t_W':>8} | " + " ".join(
        f"{n[:9]:>9}" for n in names)
    print(hdr)
    for j in range(0, 36):
        row = f"{j:3d} {t_W[j]:8.3f} {4 * t_W[j]:8.2f} | " + " ".join(
            f"{P[n]['m'][j]:9.4f}" for n in names)
        print(row)
    print()

    print("=== test-distance turn counts T_j = nu_j*D/2pi ===")
    print(f"{'j':>3} {'native':>9} | " + " ".join(f"{n[:9]:>9}" for n in names))
    for j in range(0, 36):
        print(f"{j:3d} {P[names[0]]['T_D_native'][j]:9.2f} | " + " ".join(
            f"{P[n]['T_D'][j]:9.2f}" for n in names))
    print()

    print("=== summary per arm ===")
    print(f"{'arm':22s} {'acc':>6} {'S=Sm':>7} {'mean m':>7} "
          f"{'#m>0':>5} {'#m=1':>5} {'sum nu':>8} {'sum 1/nu':>10}")
    for n in names:
        m = P[n]["m"]
        nu = P[n]["nu"]
        print(f"{n:22s} {acc[n]:6.4f} {m.sum():7.3f} {m.mean():7.4f} "
              f"{int((m > 1e-12).sum()):5d} {int((m > 1 - 1e-12).sum()):5d} "
              f"{nu.sum():8.4f} {(1 / nu).sum():10.3f}")


if __name__ == "__main__":
    main()
