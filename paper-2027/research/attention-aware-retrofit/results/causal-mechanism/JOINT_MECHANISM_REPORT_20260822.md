# Basis, content map, and softmax act jointly: a 2x2 factorial on a mature 1.5B model

- **Date:** 2026-08-22
- **Model:** OLMo-2-0425-1B-Instruct (1.485B, 16L/16H, d_head 128, K 64, base 5e5, native 4096)
- **Protocol:** RULER core-4, zero training, single seed. GPU cells produced by the
  parallel session; every analysis in this note is CPU-only, no checkpoint loaded.
- **Code:** `scripts/analysis/rope_transport/` (24 unit tests pass)

> **Evidence-owner update:** final identities, matched controls, runtime parity,
> and held-out 2Wiki results are owned by
> `LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md`. This note remains a
> mechanism draft. The evaluated adapter is a 2,097,152-parameter headwise
> rank-16 Q/K map installed after Q/K normalization; it is not a layer-global
> rank-64 adapter. The two observed lengths do not establish a scaling trend,
> and `G=5` below is a first-order interpretation of an inherited coefficient,
> not a measured OLMo-2 logit gap. The zero-training result does not depend on
> either hypothesis. Subsequent verification established bitwise-exact Native
> 4K dispatch and retained 2Wiki performance at 8K/16K.

## 1. The joint frame

For one head the pre-softmax logit at relative distance `D` is

```
l(q, k, D) = q^T R_Omega(D) k = sum_k A_k cos(w_k D + psi_k)
```

Three objects act on three disjoint parts of that expression, and the softmax
couples them:

| object | acts on | role |
| --- | --- | --- |
| non-geometric allocation | `w_k` | the **basis**: which distances are separable over the deployed range |
| LoRA / any fixed content map | `A_k, psi_k` | per-channel **amplitude and phase** |
| softmax | the sum, against `ln n` | converts logits to weights; needs a gap of order `ln n` |

**YaRN's `mscale` is a member of the content-map class, not a fourth mechanism.**
Scaling the rotary embedding by `c` is exactly `M = N = c I`, giving `logit -> c^2 l`.
It is the simplest element of the static Q/K linear-map class quantified by the
transplant obstruction. A low-rank LoRA is a restricted member of that class
and cannot express this full-rank scalar map exactly unless its rank reaches
the full dimension.

## 2. What the scalar exposes about adapter parameterization

For a map applied after Q/K normalization within one head, the gain update is

```
M - I = (c - 1) I ,   rank = d = 128 ,   all 128 singular values equal
```

so the best rank-`r` approximation captures exactly `r / d` of its squared
Frobenius energy. The evaluated adapter is an independent per-head rank-16 map,
so it cannot exactly represent this full-rank update; for the identity target
alone the corresponding fraction is `16/128 = 12.5%`. A layer-global LoRA rank
must not be divided into independent per-head ranks, and a scalar applied before
Q/K normalization is not equivalent because the normalization cancels scale.

This rank fact excludes exact representation of the scalar identity update by
the tested low-rank residual. It does not exclude task-distribution-specific
approximations, different update directions, or compensation through other
modules, and therefore does not prove that additional optimization can never
help.

Empirically, the tested 2.10M-parameter, 50-step headwise adapter did not improve
the same table, while the deterministic scalar did:

| arm | 8K RULER |
| --- | ---: |
| `budgeted_s2_p2`, no adapter | 0.4000 |
| `budgeted_s2_p2` + LoRA, 50 steps | 0.3875 |
| `budgeted_s2_p2` + scalar gain | **0.5825** |

An affine-with-gain map, `M = c I + B A`, is therefore a plausible future
parameterization: it prevents the low-rank residual from spending capacity on
the full-rank scalar. It is not a unique consequence of the data, and the
current best method needs no trained `BA` term.

## 3. The 2x2 factorial

| basis \\ gain | `mscale = 1` | `mscale = 1 + 0.1 ln s` |
| --- | ---: | ---: |
| **8K** native | 0.0000 | **0.0000** |
| **8K** budgeted s2 p2 | 0.4000 | **0.5825** |
| **8K** budgeted s2 p1 | 0.3875 | 0.5525 |
| **16K** native | 0.0000 | **0.0000** |
| **16K** budgeted s4 p2 | 0.1150 | **0.4000** |
| **16K** budgeted s4 p1 | 0.1375 | 0.4025 |

On this assay, gain alone does not lift Native from the score floor:
`native + mscale` scores `0.0000` at both lengths, identical to Native. This does
not mean its logits are unchanged. It shows that the scalar is not sufficient
for these long-context tasks and pays off only in the tested cells that also
change the frequency basis.

Interaction, taking native as the origin:

| cell | main effect (basis) | observed joint | interaction | ratio |
| --- | ---: | ---: | ---: | ---: |
| 8K, p2 | 0.4000 | 0.5825 | **+0.1825** | 0.46 |
| 8K, p1 | 0.3875 | 0.5525 | +0.1650 | 0.43 |
| 16K, p2 | 0.1150 | 0.4000 | **+0.2850** | **2.48** |
| 16K, p1 | 0.1375 | 0.4025 | +0.2650 | 1.93 |

The interaction is positive at both observed lengths. It is about `0.46` times
the bare-table main effect at 2x and `2.48` times that effect at 4x, so the 4x
cell would be badly understated by an additive account. Two lengths do not
establish a monotone law in extrapolation ratio.

## 4. The clean cell

At 16K, with **the same attention scaling on both sides** (`1.1386`, so this is
not a mscale artefact):

| operator | 16K RULER |
| --- | ---: |
| official YaRN s=4 | 0.0125 |
| `budgeted_s4_p2` | **0.4000** |

At 8K, official YaRN s=2 scores 0.5375 against `budgeted_s2_p2`'s 0.5825.

## 5. A gain-measurement hypothesis, not a closed-form result

Softmax over `n -> s n` costs `ln s` of retrieval log-odds. With native logit
gap `G` and background variance `V`, a logit-side gain `alpha` restores it when

```
(alpha - 1) G - (alpha^2 - 1) V / 2 = ln s
```

Under the stated approximation, to first order
`alpha = 1 + ln(s) / G`; because `mscale` acts on both rotated Q and K,
`mscale = sqrt(alpha)`. Expanding the inherited form
`mscale = 1 + 0.1 ln s` to first order corresponds to an *effective toy-model*
gap of about `G = 5` nats.

This does not measure OLMo-2's gap or derive the coefficient. A useful follow-up
would estimate task-conditioned gaps and distractor statistics on a frozen
natural calibration set, then validate a preregistered coefficient on separate
tasks. RoPE base alone is insufficient to infer those statistics.

Two falsifiable consequences:

- **cross-length consistency (parameter-free):** the same `G` forces
  `mscale^2(s=4) - 1 = 2 [ mscale^2(s=2) - 1 ]`. Note this is only a real test
  when the constant is swept **freely at each length**; YaRN's own one-parameter
  family satisfies it to first order by construction.
- **the second-order term** `-(alpha^2 - 1) V / 2` motivates measuring whether
  diffuse distractor logits require more gain. The current receipts do not
  establish that VT is the most diffuse task, so no task-specific coefficient
  prediction follows yet.

## 6. What did not work, stated plainly

Four CPU-computable scalars were built and tested prospectively against these
cells. **None predicts RULER**, across 16 cells:

```
Spearman(RULER, -D*)              = -0.588
Spearman(RULER, -rms sidelobe)    = -0.588
Spearman(RULER, gain alpha)       = +0.168
Spearman(RULER, joint scalar)     = -0.047
```

`D*` and the positional-kernel sidelobe are both *anti*-correlated: they measure
how far the table moved, and moving is necessary. `phase excess risk` (a fifth,
reported separately) is decisively falsified by `one_turn_floor_s2`, which is
optimal on it and scores `0.0000`.

The honest conclusion is that **no scalar summary of the frequency table alone
predicts downstream retrieval**, which is what the joint frame in section 1
says should happen: every one of these quantities is computed under isotropic
content, i.e. in a model whose trained weights have been erased. `D*` survives
as what it was derived to be -- an upper bound on adapter repair -- and section 2
uses it correctly in that role.

## 7. Next

1. Complete frozen RULER-13 breadth evaluation without changing `p=2` or the
   inherited coefficient `0.10`.
2. Keep the completed bitwise Native 4K dispatch and held-out 2Wiki result as
   the practical retention evidence.
3. If coefficient derivation remains valuable, measure `G` and `V` on a frozen
   natural calibration set and validate once on held-out tasks; do not tune on
   RULER.
4. Consider `M = c I + BA` only if a trained residual is later needed after the
   zero-training method's breadth is known.
5. Use a second checkpoint or model, rather than repeated deterministic-method
   seeds, to test whether the interaction generalizes.
