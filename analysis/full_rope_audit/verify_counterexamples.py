"""
verify_counterexamples.py — independent re-verification of the counterexample search
(counterexamples_results.md), with a CLEAN causal-weighted implementation.

Findings:
1. The symmetric-grid counterexamples from the search agent are exact (cross-checked 1e-14).
2. The search agent's causal-grid C_cos values were corrupted (symmetric closed form mixed
   into the causal computation); its specific causal examples are NOT violations under the
   correct computation. However, the claim survives: with a clean implementation, Type-1
   violations (C_cos(A) < C_cos(B) but effrank(A) < effrank(B)) occur in ~8% of random
   within-range pairs on the causal grid, with large margins (best: effrank 3.35 vs 5.98).
3. Type-2 length reversals (C_L ordering flips at 2L/4L) are common on the symmetric grid
   (~15% two-length reversal rate; 5.6% full three-length non-monotone for C_cos).
"""
import numpy as np

L = 128

def causal_gram(L, w):
    d = np.arange(L, dtype=float)
    W = (L - d); W /= W.sum()
    F = np.concatenate([np.cos(np.outer(d, w)), np.sin(np.outer(d, w))], axis=1)
    G = (F * W[:, None]).T @ F
    return 0.5 * (G + G.T)

def metrics(L, w):
    G = causal_gram(L, w); K = len(w)
    Ccos = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            Ccos += G[2*i, 2*j]**2 / (G[2*i, 2*i] * G[2*j, 2*j])
    Cfull = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            aa = G[2*i:2*i+2, 2*i:2*i+2]; bb = G[2*j:2*j+2, 2*j:2*j+2]
            ab = G[2*i:2*i+2, 2*j:2*j+2]
            S = (np.linalg.inv(np.linalg.cholesky(aa)) @ ab
                 @ np.linalg.inv(np.linalg.cholesky(bb)).T)
            s = np.linalg.svd(S, compute_uv=False)
            Cfull += 0.5 * (s[0]**2 + s[1]**2)
    dd = np.sqrt(np.diag(G)); Gw = G / np.outer(dd, dd)
    ev = np.linalg.eigvalsh(Gw); ev = ev[ev > 1e-12]; p = ev / ev.sum()
    return Ccos, Cfull, float(np.exp(-(p * np.log(p)).sum()))

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # 1) agent's causal K=3 example under correct computation
    A = [0.0017641, 0.043555, 0.08598]; B = [0.076853, 0.14755, 1.0273]
    cA, fA, eA = metrics(L, A); cB, fB, eB = metrics(L, B)
    print(f"agent causal example (clean): A C_cos={cA:.4f} C_full={fA:.4f} effr={eA:.3f}; "
          f"B C_cos={cB:.4f} C_full={fB:.4f} effr={eB:.3f}")
    print(f"  -> agent's C_cos values (0.00486/0.00489) were corrupted; violation is "
          f"{'NOT' if cA > cB else ''} present under correct computation")
    # 2) within-range search
    viol = 0; best = None
    for _ in range(40000):
        wA = np.sort(rng.uniform(2e-6, 1.0, 3)); wB = np.sort(rng.uniform(2e-6, 1.0, 3))
        cA, fA, eA = metrics(L, wA); cB, fB, eB = metrics(L, wB)
        if cA < cB and eA < eB:
            viol += 1
            if best is None or eB - eA > best[0]:
                best = (eB - eA, wA, wB, cA, cB, eA, eB)
    print(f"within-range causal violations: {viol}/40000 = {viol/40000:.2%}")
    if best:
        m, wA, wB, cA, cB, eA, eB = best
        print(f"  best: A={np.round(wA, 5)} C_cos={cA:.4f} effr={eA:.3f}")
        print(f"        B={np.round(wB, 5)} C_cos={cB:.4f} effr={eB:.3f} (margin {m:.3f})")
