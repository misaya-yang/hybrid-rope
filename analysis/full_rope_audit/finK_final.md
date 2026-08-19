# Finite-K comparison — final numbers (K=64, b=5e5, causal-weighted grid; independently computed from opt_*.npy)

er_whit = entropy effrank of column-whitened [cos,sin] Gram; C_cos = Σ_{i<j} (cos-overlap)²;
C_full = Σ_{i<j} (σ₁²+σ₂²)/2 with σ = canonical correlations of the 2D subspaces.

| alloc | metric | L=4096 | L=8192 | L=16384 | rel Δ L→4L |
|---|---|---|---|---|---|
| geometric | er_whit | 17.54 | 22.47 | 28.60 | **+63%** |
| geometric | C_cos | 344.0 | 270.2 | 205.3 | −40% |
| EVQ τ=1.4 | er_whit | 28.08 | 35.10 | 43.15 | +54% |
| EVQ τ=2.0 | er_whit | 39.15 | 47.86 | 57.19 | +46% |
| EVQ τ=2.5 | er_whit | 50.60 | 60.55 | 70.52 | +39% |
| cos-opt | er_whit | 121.00 | 123.67 | 124.30 | +2.7% |
| full-opt | er_whit | 124.32 | 124.46 | 124.49 | +0.1% |
| logdet-opt | er_whit | 124.33 | 124.46 | 124.49 | +0.1% |
| uniform [2π/L,1] | er_whit | 127.78 | 127.94 | 127.99 | +0.2% |

Key facts:
1. The three static optima (cos / full / logdet) converge to the same solution class:
   depopulate the low band (n(ωL<1) = 1), er_whit ≈ 121–124/128, near-uniform in ω over the
   healthy zone; one boundary-degenerate pair piles at the fixed low endpoint (min gap 3.6e-15).
   cos-opt achieves C_cos = 0.83 at L (near-orthogonal cosines) yet its 2D subspaces still have
   C_full = 1.29 (the degenerate low pair contributes ~1).
2. EVQ effrank is monotone in τ (28.1 → 39.2 → 50.6 for τ = 1.4/2.0/2.5): no interior static
   optimum below the healthy-zone bound. Static optimality = empty the sub-cycle band.
3. **effrank improves with L for every allocation** (geometric +63%, EVQ +39–54%, optima ~0%).
   The static identifiability metric moves in the OPPOSITE direction from model degradation
   under extrapolation. Confirmed on the symmetric grid too (geometric er_whit:
   20.8 → 26.1 → 32.6 → 40.5 for L = 2048/4096/8192/16384).
4. Robustness: K=16/32 same pattern (verify_core.py V6, symmetric grid).
