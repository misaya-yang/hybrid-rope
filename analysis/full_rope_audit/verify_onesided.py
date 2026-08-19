"""
verify_onesided.py — Full-RoPE completion of the paper's own collision kernel.

The paper's exact kernel (paper-2027/appendix/a1_proofs.tex:334) is
    K(phi_i, phi_j) = (1/2L)[sin((w1-w2)L)/(w1-w2) + sin((w1+w2)L)/(w1+w2)]
= cosine-cosine block of the full 2x2 Gram under the one-sided uniform prior
D(d) = 1/L on [0, L]. The full-RoPE 2x2 block G_ab = [[cc, cs], [sc, ss]] is:

    cc = 1/2 [sinc((wa-wb)L) + sinc((wa+wb)L)]                      (paper's K)
    ss = 1/2 [sinc((wa-wb)L) - sinc((wa+wb)L)]
    cs = 1/2 [(1-cos((wb-wa)L))/((wb-wa)L) + (1-cos((wa+wb)L))/((wa+wb)L)]   (cos_a sin_b)
    sc = 1/2 [(1-cos((wa-wb)L))/((wa-wb)L) + (1-cos((wa+wb)L))/((wa+wb)L)]   (sin_a cos_b)

Verified against quadrature with grid refinement (error -> 0). The paper's kernel
is exactly the cc block; the sin-direction and cross blocks are omitted.
Low-frequency limit (wL << 1): cc -> 1, ss -> wa wb L^2/3, cs -> wb L/2
=> V_w -> span{1, d} on the one-sided grid as well (collapse persists).
"""
import numpy as np

def sinc(x):
    return np.sinc(x / np.pi)

def block_closed(wa, wb, L):
    dw, sw = wa - wb, wa + wb
    cc = 0.5 * (sinc(dw * L) + sinc(sw * L))
    ss = 0.5 * (sinc(dw * L) - sinc(sw * L))
    cs = 0.5 * ((1 - np.cos((wb - wa) * L)) / (wb - wa) + (1 - np.cos(sw * L)) / sw) / L
    sc = 0.5 * ((1 - np.cos(dw * L)) / dw + (1 - np.cos(sw * L)) / sw) / L
    return np.array([[cc, cs], [sc, ss]])

def block_quad(wa, wb, L, npts):
    d = np.linspace(0, L, npts)
    cc = np.trapezoid(np.cos(wa * d) * np.cos(wb * d), d) / L
    ss = np.trapezoid(np.sin(wa * d) * np.sin(wb * d), d) / L
    cs = np.trapezoid(np.cos(wa * d) * np.sin(wb * d), d) / L
    sc = np.trapezoid(np.sin(wa * d) * np.cos(wb * d), d) / L
    return np.array([[cc, cs], [sc, ss]])

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    maxerr = 0.0
    for L in [64, 256, 1024]:
        for _ in range(15):
            wa, wb = np.sort(rng.uniform(0.001, 3.0, 2))
            if abs(wa - wb) < 1e-12:
                continue
            bf = block_closed(wa, wb, L)
            bq = block_quad(wa, wb, L, 128 * L + 1)
            maxerr = max(maxerr, np.abs(bf - bq).max())
    print(f"one-sided full 2x2 closed forms vs quadrature: max err {maxerr:.2e} "
          f"{'PASS' if maxerr < 5e-7 else 'FAIL'}")
    # paper's kernel is the cc block
    L = 512
    wa, wb = 0.7, 0.9
    K_paper = 0.5 * (np.sin((wa - wb) * L) / ((wa - wb) * L) + np.sin((wa + wb) * L) / ((wa + wb) * L))
    print(f"paper kernel a1:334 vs cc block: |diff| = {abs(K_paper - block_closed(wa, wb, L)[0, 0]):.2e}")
    # low-frequency collapse on one-sided grid
    for wa, wb in [(0.0005, 0.0005), (0.0005, 0.001)]:
        b = block_quad(wa, wb, 512, 2049)
        print(f"w=({wa},{wb}), wL~({wa*512:.2f},{wb*512:.2f}): cc={b[0,0]:.4f} (->1), "
              f"ss={b[1,1]:.2e} vs wa*wb*L^2/3={wa*wb*512**2/3:.2e}, cs={b[0,1]:.2e} vs wb*L/2={wb*512/2:.2e}")
