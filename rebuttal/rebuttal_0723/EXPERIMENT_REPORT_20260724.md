# EVQ-Cosh rebuttal experiment report — 2026-07-24

Status: **final through the native-grid/attention-shape diagnostic**. The
Reviewer 27bE experiments, the 100M-token FMRoPE diagnostic, the EVQ-Cosh x
FMRoPE arm, the separately registered 500M-token undertraining check, and the
three-seed native Std-RoPE/real-shape study are complete. The MLA scarcity
study in Section 8 is a verified design, not a completed new result.

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

## 7. Native Std-RoPE and real-shape diagnostic

This experiment removes the Paper-Geo midpoint ambiguity. All schedules use
the **native Std-RoPE endpoint grid** as the reference and have exactly the
same frequency span. Every non-Geo schedule also has the same RMS deformation
from native Geo (`0.255704`). Training uses the Section 2 protocol and three
seeds.

The four alternatives are:

- native EVQ-Cosh, rescaled to the Std-RoPE span;
- an exponential schedule matched to the same span and deformation;
- a schedule minimizing an exact cosine-feature Gram collision objective
  under a uniform distance prior;
- a schedule minimizing the same collision objective under a causal-attention
  distance prior measured from the Std-RoPE seed-42 checkpoint on the
  **selection anchors only**.

The collision objectives are schedule diagnostics, not language-model losses.
The registered test anchors were not used to derive any schedule.

### Three-seed mean tail NLL

| schedule | 128 | 256 | 512 | 1K | 2K | 4K | 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native Std-RoPE | 5.5046 | 5.5026 | 5.6563 | 5.8456 | 6.0530 | 6.2566 | 6.3928 |
| Native EVQ-Cosh | **5.4645** | 5.4473 | **5.5434** | 5.6966 | 5.8457 | 6.0664 | 6.2935 |
| Native exponential matched | 5.4698 | **5.4464** | 5.5450 | **5.6883** | 5.8423 | 6.0342 | 6.2165 |
| Exact-kernel uniform prior | 5.4759 | 5.4514 | 5.7355 | 5.8295 | 5.9365 | 6.0724 | 6.1727 |
| Attention-prior exact kernel | 5.4826 | 5.5099 | 5.5959 | 5.7212 | **5.8254** | **5.9504** | **6.0823** |

Native EVQ-minus-Std-RoPE differences are
`-0.113/-0.149/-0.207/-0.190/-0.099 NLL` at 512/1K/2K/4K/8K. The
seed-level paired 95% intervals are,
respectively, `[-0.200,-0.026]`, `[-0.183,-0.115]`,
`[-0.241,-0.174]`, `[-0.260,-0.121]`, and `[-0.141,-0.058]`.
All three seeds agree at every one of these lengths. Therefore the raw EVQ
effect is not an artifact of comparing against Paper-Geo midpoint.

The attention-derived schedule is not geometrically close to Cosh: it forms
two bands separated by a large interior gap, and its phi-space RMS distance
from native EVQ is `0.1804` despite their shared deformation radius
`0.2557`. It is slightly worse than EVQ at 512/1K, approximately tied at 2K,
then better by `-0.116 NLL` at 4K and `-0.211 NLL` at 8K. The 8K paired
interval for attention-minus-EVQ is `[-0.357,-0.065]`; seeds 137 and 256,
which did not supply the attention prior, agree with seed 42.

The exact uniform-prior kernel schedule also trades short-range quality for
extreme extrapolation: it is worse than EVQ at 512/1K but has a lower mean at
8K. The matched exponential is almost tied with EVQ through 2K and improves
the 8K mean by `-0.0769 NLL`, with paired interval
`[-0.1384,-0.0155]`.

**Answer to the shape question.** Cosh is not a numerical approximation to
the attention-derived optimum. It is a smooth compromise that is consistently
better than native Geo and competitive at short-to-moderate extrapolation.
The stronger extreme-length result from a very different attention-derived
shape supports the broader **allocation-shape mechanism**, but not uniqueness
or optimality of Cosh. Because the attention prior comes from one small
undertrained model, this is mechanistic evidence, not a claim that it is the
universal “true attention shape.”

### Reproducibility receipt

- Fifteen train receipts and fifteen test receipts are present and PASS.
- Within each seed, all five arms have identical trainable initialization and
  row order; all runs use the same data manifest and test anchors.
- Data-manifest SHA-256:
  `0b89bf512e837d374bf8019fee37fdb44c0a29c4b933669637f111a20e921a45`.
- New-arm code SHA-256:
  `97a349da8d8115429cf98c3a09b6af7624441a66ba6559fa85ebc3c77f16cb27`.
  The native Std-RoPE runs use the immediately preceding code receipt; the
  training/evaluation implementation is unchanged and the new fingerprint
  adds only the frozen derived schedules and their registration.
- Aggregate result SHA-256:
  `76f410fc3a4530c30a3727b99daec233743028d64717480853266dfd627576cd`.
- Selection-only attention-prior SHA-256:
  `104b20e6b03937794d13ee01e547b3230b75daf031b0fe77eb52397736c8ee1b`.
- Full checkpoints were hash-validated against both train and evaluation
  receipts, then removed after result retrieval to conserve disk space.

## 8. Registered next experiment: scarce rotary channels in MLA

The existing three-seed MLA artifact supports a narrower and potentially
stronger practical positioning. At 16 independent frequency pairs
(`K=16`, equivalently `d_rope=32` rotated scalar dimensions), the 2x/16K
PPL values are:

| condition | PPL@16K | NLL difference vs native Geo |
| --- | ---: | ---: |
| Native Geo | 138.81 | 0 |
| EVQ-Cosh | **95.59** | -0.3731 |
| Legacy MLA wavelength scaler | 117.88 | -0.1639 |

EVQ-minus-legacy-scaler is `-0.2092 NLL`, with seed-level paired 95%
interval `[-0.3228,-0.0956]`. All three seeds agree. However, the scaler is
the historical matched-scale wavelength blend without official YaRN attention
`mscale`; it must not be relabeled as official YaRN or as a matched-range
control. The existing advantage is strongest at 2x and is not monotonic at
longer extrapolation.

The next 5090 study should test the claim:

> As the rotary-frequency budget becomes scarce, internal allocation shape
> contributes more than changing spectral range alone.

Use a fixed 50.1M-parameter MLA model, `L_train=4K`, base 500K, a fixed
FineWeb-Edu tensor, no passkey mixture, and fixed `tau=1.414`. The model keeps
`d_rope=64`, `d_nope=0`, and a 32-pair rotary capacity in every arm. Cross two
**active** frequency budgets, `K=8` and `K=32`; the 24 inactive K=8 pairs have
zero frequency, so their rotation is exactly the identity. This preserves
parameter count, trainable initialization, projection paths, and tensor shapes.
It avoids the legacy implementation's confound in which changing `d_rope`
also changes the latent/non-latent key projections and model size. Compare
three schedules:

1. native Geo, `phi_k=k/K`;
2. range-matched uniform, with exactly the EVQ minimum, maximum, and
   log-frequency span but uniform interior spacing;
3. EVQ-Cosh.

Run 300M tokens per arm with snapshots at 100M/200M/300M. Seed 42 is a six-run
gate on 16 selection windows; expand to seeds 43 and 88 only if the gate
passes. Confirmatory reporting uses 32 disjoint test windows at
4K/8K/16K/32K and tail NLL as the primary statistic. Official native-grid
YaRN, including attention `mscale`, is an inference-only secondary control:
K=32 is the full native-grid operator, while K=8 applies the same official
equations to the active native grid and identity-pads the fixed capacity. Both
must remain distinct from the explicitly labeled EVQ virtual-coordinate
generalization.

For each frequency budget define:

`A_K = NLL(Native) - NLL(RangeMatched)` and
`S_K = NLL(RangeMatched) - NLL(EVQ)`.

The scarcity interaction is `I = S_K8 - S_K32`. Claim “shape matters more
than range under scarcity” only if, at 300M tokens:

- `S_K8 > 0.05 NLL`, `I > 0`, and all three seeds agree;
- `S_K8 > A_K8`;
- the 1x EVQ cost is no more than `+0.02 NLL`;
- the effect does not reverse from 200M to 300M.

Stop after the seed-42 gate if these conditions are clearly missed. The
discarded compiled probes measured 427.5K–428.0K tokens/s. The six real runs
later averaged 406.3K tokens/s and required 73.81 training minutes in total;
the short probe therefore overestimated sustained throughput by 5.3%.

The model-free preflight uses
`K(delta)=mean_k cos(delta*omega_k)` over active frequencies only. At
4K-to-8K, the range-minus-EVQ RMS-collision improvement is `+0.1281` for K=8
and `+0.0624` for K=32, giving the expected positive scarcity interaction.
At 8K-to-16K, however, the K=8 sign reverses to `-0.0372`. This proxy is not an
LM loss, but it predeclares the same non-monotonic risk seen in the historical
MLA result: 8K is the primary endpoint, and all 16K/32K results must be retained
even if unfavorable.

The implementation is now code-ready in `mla_scarcity_5090/`: sixteen focused
tests cover fixed parameter/init identity, exact inactive-pair rotation,
real-model forward/backward, model-free diagnostic identity, full preflight
receipt construction, gate/statistics, cleanup recovery, and compile-cache
identity. The launcher enforces a READY receipt and an 8 GiB free-space floor.
Its first GPU action verifies native CUDA-architecture support, BF16,
`torch.compile`, and a Flash-only SDPA path, then records compiled steady-state
throughput, memory, and ETA for K=8/K=32. The READY receipt budgets a
conservative maximum of fifteen simultaneous checkpoints, below 4 GiB. Full
checkpoints are deleted only after their raw evaluation JSONs and hashes pass.
The 100M diagnostic weight is pruned immediately; 200M/300M weights are
evaluated and deleted per run. Raw rows, receipts, schedule sidecars, and the
shared TorchInductor cache remain.

### 8.1 Execution and terminal decision

All six seed-42 arms completed on one RTX 5090. Each arm consumed exactly
299,892,736 tokens. BF16, Flash-only SDPA, `torch.compile(default)`, fused
AdamW, micro/global batch 32, and the shared compile cache were active. Mean
training throughput was 406,311 tokens/s; all losses remained finite; each run
peaked at 32,171,420,160 allocated CUDA bytes. The six training loops required
4,428.5 seconds (73.81 minutes), excluding the small evaluation and checkpoint
serialization overhead.

The originally registered selection gate returned `PASS`: at 8K/K=8,
range-matched uniform collapsed to NLL 6.2005, while EVQ recovered to 4.6179,
producing a nominal shape gain of 1.5827 NLL and a positive scarcity
interaction. This gate was incomplete as a practical decision rule, however:
native RoPE was substantially better at 3.9505 NLL, and EVQ was also worse
than native at K=32.

Before reading the disjoint test anchors, a terminal decision was frozen:
do not run seeds 43/88, regardless of the test result. The additional practical
criteria were:

- EVQ must beat native at the primary 2x endpoint for K=8;
- EVQ must beat native at the primary 2x endpoint for K=32;
- the positive interaction must not be created only by catastrophic failure of
  the range-only control.

All three criteria failed on selection. The decision receipt SHA-256 is
`2a72bbb2656c99a8a74dfc32bd949d5e1fa1aca36c5efc7a9d093db9cbabb370`.
Only then were the 32 disjoint test anchors read.

### 8.2 Disjoint test result: raw substrates

The following values are 300M-checkpoint tail NLL on the disjoint test split.
`A_K` is native-minus-range NLL, `S_K` is range-minus-EVQ NLL, and
`EVQ-Native < 0` is the practical win condition.

| K | length | Native | Range | EVQ | `A_K` | `S_K` | EVQ-Native |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 4K | 3.9033 | 3.8969 | 3.8933 | +0.0064 | +0.0036 | **-0.0100** |
| 8 | 8K | **3.9915** | 6.2195 | 4.6437 | -2.2280 | +1.5758 | **+0.6522** |
| 8 | 16K | **6.7859** | 6.7283 | 6.8992 | +0.0576 | -0.1709 | **+0.1133** |
| 8 | 32K | 6.9244 | **6.7060** | 6.7552 | +0.2184 | -0.0492 | **-0.1692** |
| 32 | 4K | 3.8704 | **3.8642** | 3.8687 | +0.0063 | -0.0046 | -0.0017 |
| 32 | 8K | **4.9115** | 4.9240 | 4.9497 | -0.0125 | -0.0257 | +0.0382 |
| 32 | 16K | 5.9648 | 6.0178 | **5.9626** | -0.0530 | +0.0552 | -0.0022 |
| 32 | 32K | **6.1597** | 6.1952 | 6.2158 | -0.0355 | -0.0206 | +0.0560 |

The test split reproduces the selection diagnosis almost exactly. At K=8/8K,
the selection and test EVQ-minus-native gaps are +0.6674 and +0.6522 NLL,
respectively; the shape gains are +1.5827 and +1.5758. Thus this is not
selection-anchor overfitting. It is a real, highly non-monotonic phase pattern:
the range control fails specifically at 8K, EVQ partially repairs that bad
control, but native remains much better. The +0.6522 NLL gap means EVQ PPL is
1.92 times native PPL at the registered primary endpoint.

At K=32, all raw differences are much smaller and allocation-shape signs vary
with length. The test result therefore does **not** reproduce a practical raw
EVQ advantage under scarcity, and it does not justify twelve confirmatory
training runs. The earlier historical 432M/K=16 result remains a valid result
for its stated protocol, but this fresh 50M fixed-architecture experiment does
not generalize it.

The 200M checkpoint has the same primary direction: EVQ-minus-native is
+0.6111 NLL for K=8 and +0.0028 for K=32 at 8K. Training longer to 300M does
not repair the primary raw gap. At 4K, EVQ is slightly better than native
(-0.0100/-0.0017 NLL for K=8/K=32), so the failure is not training divergence
or an in-domain collapse.

### 8.3 Inference-time YaRN diagnostic

The retained 300M weights were also evaluated with the registered
inference-time operators before cleanup. K=32 native uses full standard
official YaRN. K=8 native applies the same official equations to the active
native grid and identity-pads the fixed capacity. EVQ uses the explicitly
labeled virtual-coordinate YaRN-derived transform; it is **not** official
YaRN, so the cross-arm delta is a deployment diagnostic rather than an
official-YaRN parity claim.

| K | length | Native + YaRN NLL | EVQ + derived YaRN NLL | EVQ-Native | PPL change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 4K | 3.9033 | **3.8933** | -0.0100 | -1.0% |
| 8 | 8K | 3.9064 | **3.8926** | -0.0138 | -1.4% |
| 8 | 16K | 4.0900 | **3.9739** | -0.1161 | -11.0% |
| 8 | 32K | 4.9827 | **4.2030** | -0.7797 | -54.1% |
| 32 | 4K | 3.8704 | **3.8687** | -0.0017 | -0.2% |
| 32 | 8K | 3.8638 | **3.8622** | -0.0017 | -0.2% |
| 32 | 16K | 3.8980 | **3.8944** | -0.0036 | -0.4% |
| 32 | 32K | 3.9504 | **3.9418** | -0.0086 | -0.9% |

This secondary result is directionally interesting: under the respective
YaRN transforms, the scarce K=8 EVQ substrate gains grow with extrapolation,
whereas K=32 is nearly tied. It is consistent with the historical observation
that a scarce substrate can interact with aggressive range scaling. It cannot
rescue the failed raw-scarcity claim, prove a pure superlinear interaction, or
restore the submitted official-YaRN identity: the two substrates receive
different, clearly labeled operator mappings, and the result is single seed.

### 8.4 Provenance, cleanup, and revised gate rule

The sanitized aggregate is
`mla_scarcity_seed42_result_20260724.json`. Its exact-number parity was checked
against all retrieved raw evaluation JSONs at terminal time. That retrieval
contained 99 files (1.87 MB), with six train receipts, 12 selection
evaluations, 12 test raw evaluations, four YaRN evaluations, and six cleanup
receipts. Its manifest SHA-256 is
`b8af9d91431056f4b5a89fbf4d1863bc3d6846734cb427720abb1c5d044b605a`.
The 99-file bundle is no longer discoverable in the current local workspace;
only the sanitized aggregate, recorded hashes and report remain locally
verifiable. Do not claim current per-file access to the historical bundle.

All twelve 200M/300M checkpoints were deleted only after the corresponding
test JSON and checkpoint SHA-256 checks passed. The terminal audit recorded
zero checkpoints and zero incomplete files. Seeds 43/88 were not started, and
the paid instance was shut down after result retrieval.

Future mechanism gates must add two conditions that this run exposed:

1. the proposed method must beat the meaningful native baseline at the primary
   endpoint, not merely beat a matched control;
2. an interaction cannot pass when it is dominated by catastrophic failure of
   one control at one oscillatory length.

For strongly oscillatory RoPE evaluations, report every registered length and
consider an aggregate across log-spaced lengths; never promote a favorable
single-length sign as a monotonic extrapolation law.

### 8.5 Registered shared-operator follow-up

The next experiment is now implemented but has not run. It removes the operator
identity confound by training only Native and EVQ at K=8/K=32, then applying
the same official native-index correction coefficients and the same `mscale`
to both substrates. On EVQ this remains an explicitly labeled shared-index
component control, not official YaRN.

The seed-42 gate uses fresh selection windows disjoint from every previously
observed 32K window. It requires EVQ to beat native under the shared operator at
both 16K and 32K, a mean advantage and interaction of at least 0.05 NLL,
positive scarcity interaction, bounded 4K cost for both K values, no native
control degradation above 0.05 NLL for either K, and agreeing 200M/300M
directions. Seeds 43/88 and the fresh test split are inaccessible before PASS.
Only raw/shared-full are evaluated at 200M for this direction check; all six
registered operator components are evaluated at 300M, reducing total registered
inference work by one third without changing a primary estimand.
Evaluation batches are 8/4/2/1 windows at 4K/8K/16K/32K, so each forward sees
at most 32K tokens—the same token volume as the already validated single 32K
path—while retaining independent per-window NLL. This cuts forward-launch count
by 53%.

The native identity check uses exact registered checkpoint-table equality and
exact official mask/`mscale` parity. Because the registered schedule is stored
in FP32 while the equation audit is evaluated in FP64, runtime output may
differ by at most one FP32 ULP; this tolerance is not used to infer method
identity. The target-server data manifest and READY receipts do not yet exist,
so this section records implementation status, not a result or launch approval.
Both terminal paths now render a hash-backed Markdown report automatically.
Monitoring reads one compact artifact snapshot every five minutes and does not
stream logs; live PID/GPU health remains a separate one-time post-launch check.

The 0.05-NLL gate is a practical-effect floor (about 4.9% lower PPL), not a
significance threshold. Final inference uses training seeds as the independent
units and reports t-based seed-level intervals separately. All per-anchor paired
effects are retained for audit, but windows are explicitly labeled repeated
measurements rather than pseudo-seeds. The earlier 99-file retrieval bundle is
not currently discoverable in the local workspace, so its aggregate means and
recorded hashes remain evidence, but it is not used to retrofit a variance-based
MDE.

## 9. Rebuttal-ready claim decisions

1. **Keep:** the tau rule is a basin selector/default, not a global optimum.
   The independent sweep places it close to the selected optimum.
2. **Keep and strengthen:** benefits transfer to held-out base 1M and
   `d_head=128`, with three-seed paired evidence through 16K.
3. **Keep and clarify:** native-grid EVQ beats true Std-RoPE from 512 through
   8K in all three seeds, so the effect is not caused by Paper-Geo midpoint.
4. **Narrow:** allocation shape matters, but the analytic and
   attention-derived controls show that Cosh is not uniquely optimal.
5. **Concede:** the small-model FMRoPE/YaRN diagnostic does not support a broad
   performance-novelty claim over inference-time range scaling.
6. **State the deployment boundary:** raw EVQ improves over raw Paper-Geo,
   whereas FMRoPE and YaRN-derived evaluation assume a declared target length.
   Do not present these as interchangeable contracts or claim EVQ replaces
   inference-time scaling.
7. **Do not promote the fresh raw scarcity claim:** the seed-42 test shows that
   the registered K=8 interaction came from collapse of the range control, while
   raw EVQ remained worse than native at 2x. The YaRN-derived K=8 trend is
   supporting-only and operator-qualified; no additional seeds were run.
8. **Keep separate:** these results answer mechanism attribution and
   held-out-configuration robustness. They do not replace the planned
   larger-model/RULER evidence.
