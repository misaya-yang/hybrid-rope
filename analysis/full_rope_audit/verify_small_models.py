"""
verify_small_models.py — small-model (small L, small K) global verification of the
full-RoPE identifiability theory. Exhaustive enumeration, exact closed forms.

Verified claims:
V-S1: K=2 global optimality on the one-sided uniform-prior grid: the global minimum of
      C_full = (s1^2+s2^2)/2 over frequency pairs is exactly 0, attained only at pairs
      where (w2-w1)L/2pi and (w1+w2)L/2pi are both integers (parity-lattice pairs).
      L=16/24/32 exhaustive. Min value ~3e-33.
V-S2: Parity-lattice exactness: all channels on w_k = pi*a_k/L with a_k of the SAME
      parity give exactly orthogonal 2D subspaces (Gram = (1/2) I_2K, effrank = 2K).
      K=3@L=16 (odd class a={1,3,5}): er = 6.0000/6. K=5@L=32 (a={1,3,5,7,9}): 10.0000/10.
      Capacity of the largest same-parity class: K* = ceil(floor(L/pi)/2).
V-S3: Mixed parity -> rank loss: K=4@L=16 exhaustive best er = 7.60/8 (not full rank).
V-S4: Endpoint-constrained (w1, wK pinned): L=64, K=4, w1=1/512, w4=1: exhaustive best
      er = 7.988/8 — boundary-layer cost ~0.15%.
V-S5: Collapse constants converge from below at small L (wL=0.2):
      (1-s1^2)/(wL)^4 -> 1/525: 0.001392@L=8 -> 0.001851@L=64
      (1-s2^2)/(wL)^4 -> 1/45:  0.016899@L=8 -> 0.021691@L=64
REFUTED: "oversampling saturation at 2K*+2" — exhaustive K=7@L=32 achieves 12.96/14 > 12.
         Overflow channels at fractional offsets (1/4, 3/4, 1/2 lattice units) contribute
         partial independent dimensions; rank degrades GRADUALLY, not by saturation.
         (Earlier L=16 numbers suggesting saturation ~8 were grid-resolution artifacts.)
"""
import numpy as np
from itertools import combinations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_core import effrank, whiten_cols

def block(L, wa, wb):
    """One-sided uniform-prior [0,L] closed-form 2x2 Gram block, diagonal handled."""
    if abs(wa - wb) < 1e-14:
        return np.array([[0.5 * (1 + np.sinc(2 * wa * L / np.pi)), 0.0],
                         [0.0, 0.5 * (1 - np.sinc(2 * wa * L / np.pi))]])
    dw, sw = wa - wb, wa + wb
    return np.array([
        [0.5 * (np.sinc(dw * L / np.pi) + np.sinc(sw * L / np.pi)),
         0.5 * ((1 - np.cos((wb - wa) * L)) / (wb - wa) + (1 - np.cos(sw * L)) / sw) / L],
        [0.5 * ((1 - np.cos(dw * L)) / dw + (1 - np.cos(sw * L)) / sw) / L,
         0.5 * (np.sinc(dw * L / np.pi) - np.sinc(sw * L / np.pi))]])

def erank_of(w, L):
    K = len(w)
    G = np.zeros((2 * K, 2 * K))
    for i in range(K):
        for j in range(K):
            G[2*i:2*i+2, 2*j:2*j+2] = block(L, w[i], w[j])
    return effrank(whiten_cols(G))

def best_table(L, K, grid_div, wlo=None, whi=None):
    """Exhaustive best-erank K-tuple from the quarter-lattice candidate grid."""
    J = int(grid_div * L / (2 * np.pi)) + 2
    ws = 2 * np.pi * np.arange(1, J + 1) / (grid_div * L)
    best = (0.0, None)
    for c in combinations(range(len(ws)), K):
        w = np.sort(ws[list(c)])
        er = erank_of(w, L)
        if er > best[0]:
            best = (er, w)
    return best

if __name__ == "__main__":
    print("V-S1 K=2 global optimality (one-sided grid):")
    for L in [16, 24, 32]:
        J = int(4 * L / (2 * np.pi)) + 2
        ws = 2 * np.pi * np.arange(1, J + 1) / (4 * L)
        best = (1e9, None); nz = 0
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                b = block(L, ws[i], ws[j])
                aa, bb = block(L, ws[i], ws[i]), block(L, ws[j], ws[j])
                Sw = np.linalg.inv(np.linalg.cholesky(aa)) @ b @ np.linalg.inv(np.linalg.cholesky(bb)).T
                s12 = np.linalg.svd(Sw, compute_uv=False)
                cf = 0.5 * (s12[0]**2 + s12[1]**2)
                if cf < best[0]: best = (cf, (ws[i], ws[j]))
                if cf < 1e-12: nz += 1
        w1, w2 = best[1]
        print(f"  L={L}: min C_full={best[0]:.2e} at w=({w1:.4f},{w2:.4f}) "
              f"(dL/2pi={(w2-w1)*L/(2*np.pi):.2f}, sL/2pi={(w1+w2)*L/(2*np.pi):.2f}) "
              f"#zero-pts={nz}")
    print("V-S2/V-S3 parity lattice and mixed parity:")
    for L, K, a in [(16, 3, [1, 3, 5]), (32, 5, [1, 3, 5, 7, 9]), (16, 4, [1, 2, 3, 5])]:
        w = np.pi * np.array(a) / L
        print(f"  L={L} K={K} a={a}: er = {erank_of(w, L):.4f}/{2*K}")
    print("V-S3b exhaustive K=4 @ L=16 (mixed parity forced):", end=" ")
    er, w = best_table(16, 4, 8)
    print(f"best er = {er:.4f}/8, w={np.round(w, 4)}")
    print("V-S4 endpoint-constrained L=64 K=4 (w1=1/512, w4=1 pinned):", end=" ")
    L, J = 64, int(8 * 64 / (2 * np.pi)) + 2
    ws = 2 * np.pi * np.arange(1, J + 1) / (8 * L)
    best = (0.0, None)
    for ci, cj in combinations(range(len(ws)), 2):
        w = np.sort([1 / 512, ws[ci], ws[cj], 1.0])
        er = erank_of(w, L)
        if er > best[0]:
            best = (er, w)
    er, w = best
    print(f"best er = {er:.4f}/8, w={np.round(w, 5)}")
    print("V-S5 collapse constants from below (wL=0.2):")
    for L in [8, 16, 32, 64]:
        D = np.arange(-(L - 1), L); w = 0.2 / L
        Qw = np.linalg.qr(np.stack([np.cos(w * D), np.sin(w * D)]).T)[0][:, :2]
        Q0 = np.linalg.qr(np.stack([np.ones(2 * L - 1), D.astype(float)]).T)[0][:, :2]
        s1, s2 = np.sort(np.linalg.svd(Qw.T @ Q0, compute_uv=False))[::-1]
        print(f"  L={L:3d}: (1-s1^2)/(wL)^4={(1-s1**2)/0.2**4:.6f} (->1/525)  "
              f"(1-s2^2)/(wL)^4={(1-s2**2)/0.2**4:.6f} (->1/45)")
    print("REFUTED-1 oversampling saturation at 2K*+2: exhaustive K=7 @ L=32:", end=" ")
    er, w = best_table(32, 7, 4)
    print(f"best er = {er:.4f}/14 (> 2K*+2 = 12); w*L/2pi = {np.round(w*32/(2*np.pi), 2)}")
