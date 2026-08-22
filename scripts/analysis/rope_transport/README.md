# RoPE transport analysis — attention, allocation, and LoRA in one frame

- **Status:** CPU-only diagnostic; no checkpoint, no forward pass, no training
- **Evidence role:** internal analysis. It is a statement about frequency
  tables under an isotropic content model, never a task or capability result.

## 1. Why the three objects have different scopes

For one head the pre-softmax logit at relative distance `D` is

```
l(q, k, D) = q^T R_Omega(D) k = sum_k A_k cos(w_k D + psi_k)
```

The frequency table is a **basis**; content is only **coordinates** in it.
That separation fixes what each object can and cannot do.

| Object | Acts on | Scope |
| --- | --- | --- |
| Pretrained attention | coordinates `(A_k, psi_k)` | fits an element of `span{cos(w_k D), sin(w_k D)}` on `[0, L]`; behaviour beyond `L` is the forced analytic continuation, not a choice |
| Non-geometric allocation | the basis `Omega` | changes in-window representability **and** out-of-window continuation |
| LoRA on Q/K | coordinates only | a fixed, position-independent content map `q -> Mq`, `k -> Nk`; it can never leave `span(Omega')` |

So "can LoRA repair a table change" is exactly: **is the identity map between
two restricted bases realisable by a fixed content map?** The transplant
obstruction answers this for all `D` in `R` and all content, where the answer is
no. It says nothing about the restriction to `[0, L]`, which is the only regime
that matters for retention. This module measures that restriction.

## 2. What is computed

Under isotropic content, `E_{q,k}[(q^T A k - q^T B k)^2] = ||A - B||_F^2`
exactly, so squared Frobenius distance between rotation operators *is* expected
squared logit error. Every quantity below is therefore a logit-space error, not
a proxy.

- `D0 = E_{D~w} ||R_Omega'(D) - R_Omega(D)||_F^2` — hard table swap, no repair.
- `D* = min_{M,N} E_{D~w} ||M^T R_Omega'(D) N - R_Omega(D)||_F^2` — the best
  fixed position-independent Q/K maps. This is the exact `(A, B)` class of the
  obstruction theorem and a **strict superset of any Q/K LoRA**: any rank, plus
  cross-pair mixing that no practical adapter has.
- Both are divided by `d = ||R(D)||_F^2` to give a fraction of logit energy.
  **`1.0` is the reference value of a model emitting no positional signal at
  all**, so a table with relative `D0 > 1` is worse in-window than deleting RoPE.
- A rank sweep on `rank(M - I) <= r`. A LoRA of rank `r` on `q_proj` induces
  exactly `rank(M - I) <= r`, and that budget is **shared by every head in the
  layer**. The sweep therefore converts directly into an adapter-capacity
  statement.
- Per-pair attribution of the surviving residual.
- Per-pair in-window uniqueness: how much of a rotary pair is *not* already
  explained by the rest of the table on `[0, L]`. This is the per-channel form
  of low-frequency collapse and it is the free displacement budget.
- Phase safety at each target length: the fraction of channels whose deployed
  phase arc stays inside the arc seen during training.

## 3. The two axes, and why they are the right ones

Retrofit is a two-objective problem and the objectives are not exchangeable:

- **cost** = `D*`, the in-window function that no adapter can restore;
- **benefit** = phase-safe fraction at the target length, the part of the table
  that is in-distribution when the context grows.

`dominance` in the receipt reports strict Pareto dominance on those two axes.
A table that raises cost without raising benefit is dominated by doing nothing.

## 4. Solver notes

`D*` is bilinear in `(M, N)`, so each half-step is solved in closed form
(reduced-rank regression when a rank cap is set) and is monotone. The problem is
not jointly convex and has a trivial `M, N -> 0` basin whose value is exactly
`d` — a model that emits no positional signal. The search therefore runs from
the identity and from the rank-matching block permutation and keeps the better
result, so **`D*` is always an upper bound on the achievable repair and never
an underestimate**. Two unit tests pin the cases where the theorem guarantees
exact compensation exists (frequency permutation and sign alias); the solver
must drive those to zero, and does.

`repairability = 1 - D*/D0` saturates near `0.5` for any large displacement,
because an unrepairable pair is optimally shrunk to zero. Read `D*` itself, not
`repairability`.

## 5. Running it

```bash
python scripts/analysis/rope_transport/run_analysis.py \
  --manifest /path/to/target_manifest.json \
  --output   /path/to/receipt.json \
  --head-dim 128 --rope-base 500000 \
  --native-length 4096 --target-length 16384 \
  --phase-safety-lengths 8192 16384 \
  --support-points 2048 --ranks 1 2 4 8 16 32 64 128 \
  --emit-tables /path/to/derived_tables
```

Tests:

```bash
python -m pytest scripts/analysis/rope_transport/tests -q
```

The runner refuses to start if `CUDA_VISIBLE_DEVICES` is set to a real device
and asserts that `torch` was never imported.

## 6. Claim boundary

- Isotropic content is a **worst-case-symmetric** model. The realised `q, k`
  of a trained checkpoint occupy a lower-dimensional manifold, on which the true
  repair can only be **better** than reported here. The refinement is one
  forward pass collecting post-projection `q, k` and reweighting the same
  expectations; it is not implemented here and would need explicit GPU
  authorisation.
- The distance weight is part of the contract. `causal` is the exact pair count
  at each distance in a packed sequence; a measured attention-distance profile
  can be supplied through the `empirical` family.
- Nothing here is a task metric. `D*` does not predict RULER, 2Wiki, or NLL by
  itself; it bounds what an adapter can restore, which is a different object.
- Frozen tables are loaded from a manifest and never reconstructed. Derived
  operators are labelled `derived` and carry their rule, source table, and hash.
