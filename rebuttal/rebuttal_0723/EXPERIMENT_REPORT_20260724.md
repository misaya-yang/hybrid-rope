# EVQ-Cosh rebuttal experiment report — 2026-07-24

Status: **final**. The Reviewer 27bE experiments, the 100M-token FMRoPE
diagnostic, the EVQ-Cosh x FMRoPE arm, and the separately registered
500M-token undertraining check are complete.

No paper table value is changed by this report. These are fresh rebuttal
experiments and must not be numerically merged with historical Primary-II rows.

## Evaluation contract

- Model: the repository's historical “125M” decoder, exact tied-embedding
  parameter count **151,898,880**.
- Training/evaluation data: deterministic FineWeb-Edu shard-000 prefixes;
  validation windows come from disjoint pinned shard 004.
- Metric: teacher-forced causal NLL. The primary extrapolation metric is mean
  NLL over the last 128 targets in each fixed held-out window.
- Pairing: within each registered comparison, trainable initialization, token
  order, optimizer, batch protocol, and evaluation anchors match; only the
  immutable frequency schedule changes.
- Execution: one RTX 5090, BF16 autocast, fused AdamW, SDPA, and
  `torch.compile`. These are execution details, not scientific variables.

## 1. Direct 100M-token FMRoPE diagnostic

This is a single-seed (`42`), `L_train=256`, 99,942,400-token diagnostic with
32 paired held-out anchors. The three trainable initialization hashes and row
orders match.

Important identity boundary: this first completed run used **Std-Geo on the
native endpoint grid** as its Geo arm. It is not the submission's corrected
Paper-Geo midpoint baseline. The FMRoPE and EVQ rows retain their stated
formulas; the later 500M replication uses Paper-Geo explicitly.

### Mean tail NLL

| condition | 256 | 512 | 1,024 | 2,048 |
| --- | ---: | ---: | ---: | ---: |
| Std-Geo raw, base 500K | 3.8815 | 4.6092 | 5.1424 | 5.6932 |
| Std-Geo + inference-only official YaRN equations | 3.8815 | **3.8483** | **3.8780** | **3.9611** |
| FMRoPE, target base = evaluation length | 3.9251 | 3.9324 | 4.0881 | 4.4393 |
| EVQ-Cosh, tau 4 / base 500K, raw | 3.9057 | 4.1660 | 4.6515 | 5.2496 |

At 2K, EVQ improves over raw Std-Geo by `-0.4436 NLL` and wins all 32
paired windows. FMRoPE target retargeting improves over its fixed-base
condition by `-0.8473 NLL`. Inference-only YaRN equations are strongest in
this diagnostic.

**Consequence.** EVQ provides a real raw-extrapolation benefit over an
unscaled geometric baseline, but this experiment does not support performance
superiority over FMRoPE or YaRN. FMRoPE requires a declared target length;
raw EVQ does not, so the rows also represent different deployment contracts.

## 2. Independently selected tau

Protocol: `L_train=128`, base 500K, `d_head=64`, 14,974,976 training tokens.
Sixteen selection anchors and 32 test anchors are disjoint. Selection uses
only mean tail NLL across 1K/2K/4K/8K.

| tau | selection NLL |
| ---: | ---: |
| 0 (Paper-Geo) | 6.3465 |
| 1 | 6.3345 |
| 2 | 6.1376 |
| 3 | 6.0761 |
| 4 | 6.0475 |
| **5 (selected)** | **5.9898** |
| `d/sqrt(L) = 5.657` | 6.0017 |
| 6 | 6.0916 |
| 7 | 6.2328 |

The operating rule is not the exact tuned optimum, but it is within
`0.0119 NLL` (about 0.20%) of the independently selected tau. The degradation
at tau 6 and 7 shows a bounded basin rather than monotonic benefit from larger
tau.

## 3. Three-seed allocation-shape attribution

All non-cosh schedules are analytic and fixed. Uniform matches the EVQ-rule
endpoints/span; power and exponential additionally match the RMS interior
deformation. Values are three-seed mean tail NLL.

| schedule | 128 | 1K | 2K | 4K | 8K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paper-Geo | 5.5167 | 5.9381 | 6.1594 | 6.3687 | 6.5354 |
| Uniform span-matched | 5.4987 | 5.7725 | 5.9942 | 6.2967 | 6.6067 |
| **EVQ-Cosh rule** | 5.4651 | 5.6819 | 5.8541 | 6.1457 | 6.2970 |
| Power matched | 5.4732 | 5.6870 | 5.8603 | 6.0960 | 6.3560 |
| Exponential matched | **5.4447** | **5.6497** | **5.7986** | **6.0221** | **6.2593** |

EVQ-rule improves over Paper-Geo by `-0.256/-0.305/-0.223/-0.238 NLL`
at 1K/2K/4K/8K; every seed has the same direction. EVQ also beats the
uniform span-matched schedule, especially at 8K.

However, exponential matching is numerically better than EVQ at the reported
means, including all three seeds at 4K. With only three seeds, the paired 95%
intervals for EVQ-minus-exponential still include zero. The valid conclusion
is therefore:

> Non-uniform allocation shape is an effective RoPE design axis; Cosh is a
> closed-form, zero-learned-parameter instance with a useful operating rule,
> not a uniquely optimal analytic family.

## 4. Held-out base and head dimension

This suite is held out from the submitted calibration: base 1M,
`d_head=128`, `L_train=512`, 49,995,776 tokens, and three seeds. Values below
are mean tail NLL. Delta is EVQ minus Paper-Geo; negative favors EVQ.

| length | Paper-Geo | EVQ rule | delta | paired 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 512 | **4.0429** | 4.1123 | +0.0694 | `[+0.0503,+0.0885]` |
| 1K | 5.2780 | **4.4763** | -0.8018 | `[-0.9351,-0.6684]` |
| 2K | 5.4949 | **4.8308** | -0.6640 | `[-0.7310,-0.5971]` |
| 4K | 5.7181 | **5.2852** | -0.4329 | `[-0.6761,-0.1897]` |
| 8K | 5.9710 | **5.6840** | -0.2871 | `[-0.4387,-0.1354]` |
| 16K | 6.0922 | **5.8805** | -0.2117 | `[-0.3029,-0.1205]` |

All three seeds agree at every length: EVQ pays a small in-domain cost at 512
and improves every extrapolation length through 16K. This is strong evidence
that the allocation effect is not confined to base 500K or `d_head=64`.
It is still a 151.9M model and does not answer the reviewer's separate
production-scale request.

## 5. EVQ-Cosh x FMRoPE diagnostic

One new matched arm trains EVQ-Cosh at base 256 and tau 4, then either keeps
base 256 or substitutes the declared evaluation length while preserving the
Cosh quantile shape.

| condition | 256 | 512 | 1K | 2K |
| --- | ---: | ---: | ---: | ---: |
| EVQ + fixed base 256 | 3.9903 | 4.1050 | 4.6584 | 5.0403 |
| EVQ + target base = length | 3.9903 | **4.0074** | **4.1298** | **4.4026** |

Target retargeting improves the combined checkpoint by
`-0.098/-0.529/-0.638 NLL` at 512/1K/2K. Relative to the pure FMRoPE target
row, the combination is worse by `+0.075/+0.042 NLL` at 512/1K and better by
only `-0.037 NLL` at 2K. It remains worse than inference-only YaRN.

**Consequence.** EVQ allocation and FMRoPE-style range retargeting are
mechanically compatible, but this result does not establish superlinear
synergy or universal superiority of the combination.

## 6. 500M-token undertraining check

The 151.9M model was trained from the same initialization and token order for
499,974,144 tokens per arm (`3.29` tokens per parameter), using seed 42,
`L_train=256`, and 32 paired held-out anchors. This is a longer-training
diagnostic, not a saturation claim. Unlike Section 1, the geometric arm here
is the corrected **Paper-Geo midpoint** schedule.

The YaRN-derived row applies the repository's official-YaRN equations to the
Paper-Geo checkpoint in virtual frequency coordinates. It is useful as an
inference-scaling control, but it is **not** official native-grid YaRN and is
not a continuation-training result.

### Mean tail NLL

| condition | 256 | 512 | 1,024 | 2,048 |
| --- | ---: | ---: | ---: | ---: |
| Paper-Geo raw, base 500K | **3.3270** | 4.9644 | 5.7279 | 6.3065 |
| Paper-Geo + YaRN-derived virtual ramp | **3.3270** | **3.2860** | **3.3689** | **3.4251** |
| FMRoPE, fixed base 256 | 3.3459 | 4.9618 | 5.8350 | 6.3357 |
| FMRoPE, target base = evaluation length | 3.3459 | 3.3336 | 3.5613 | 4.6063 |
| EVQ-Cosh, tau 4 / base 500K, raw | 3.3502 | 4.6043 | 5.5070 | 6.1765 |

EVQ-minus-Paper-Geo paired tail-NLL differences are
`+0.023/-0.360/-0.221/-0.130` at 256/512/1K/2K. EVQ wins 31/32, 32/32,
and 28/32 held-out
windows at the three extrapolation lengths. Thus the raw EVQ benefit survives
the fivefold training increase, but its magnitude narrows with distance and
does not remove long-range degradation.

FMRoPE target retargeting improves over its fixed-base checkpoint by
`-1.628/-2.274/-1.729 NLL` at 512/1K/2K. It beats raw EVQ by
`1.271/1.946/1.570 NLL` at those lengths, winning all 32 paired windows.
The YaRN-derived control is stronger still, beating FMRoPE by
`0.048/0.192/1.181 NLL`.

**Consequence.** More training does not reverse the raw EVQ advantage over
matched Paper-Geo, but it makes the deployment-contract distinction decisive:
target-aware inference scaling is much stronger than leaving either trained
frequency schedule unscaled. FMRoPE is therefore a strong target-length-aware
baseline, not the overall winner in this suite; the YaRN-derived row has the
lowest extrapolation NLL. The result gives no support for claiming that EVQ
replaces range scaling.

### Reproducibility receipt

- All three arms share initial trainable hash
  `fb17648236fc6b976795f6f4985dc055421b0122fae594382c7cb99937872452`
  and row-order hash
  `c6fc5b4d7dea51134ad609ca87864e54b8cf46c169838cd08e1a20b9bcf47452`.
- Protocol hash:
  `d92da1b8a4ba92cb9bee5bcd28c2d73282bc0bedfc38adf164dc9693a42d188d`.
- Code hash:
  `351419fcda6d3bd32e57dec122fdd7cf892b17b96c3b1d9bfa845949e6797e14`.
- Data-manifest hash:
  `9029c05140d27195560663353c0852ec6cdbb6c20fd81cbdedae912f64a99e6a`.
- Result JSON hash:
  `da7434beb2c34fe93723b41c9e680632443c20908b4daa2b9843ad18a0cf2cda`.
- In-domain identity controls: **PASS**.

## 7. Rebuttal-ready claim decisions

1. **Keep:** the tau rule is a basin selector/default, not a global optimum.
   The independent sweep places it close to the selected optimum.
2. **Keep and strengthen:** benefits transfer to held-out base 1M and
   `d_head=128`, with three-seed paired evidence through 16K.
3. **Narrow:** allocation shape matters, but the current evidence does not
   show that Cosh is uniquely best among matched analytic schedules.
4. **Concede:** the small-model FMRoPE/YaRN diagnostic does not support a broad
   performance-novelty claim over inference-time range scaling.
5. **State the deployment boundary:** raw EVQ improves over raw Paper-Geo,
   whereas FMRoPE and YaRN-derived evaluation assume a declared target length.
   Do not present these as interchangeable contracts or claim EVQ replaces
   inference-time scaling.
6. **Keep separate:** these results answer mechanism attribution and
   held-out-configuration robustness. They do not replace the planned
   larger-model/RULER evidence.
