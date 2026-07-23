# LongAlpaca LoRA Temporal-Holdout NLL Audit

## Material Passport

- Status: `ANALYZED`, supporting evidence; not yet a multi-seed paper claim.
- Result artifact: `data/curated/lora_longalpaca_temporal_s42_20260712.json`.
- Result SHA256: `0335415a2245e1fb31149705342e975a016ddddb557a79c364fc4a98c3f89001`.
- Evidence checks: artifact identity, matched run protocols, adapter/frequency hashes,
  NLL sums and token counts, PPL conversion, macro aggregation, and pack alignment
  were independently recomputed.
- Experimental values changed in the paper: none.

> **2026-07-13 identity correction:** 该实验匹配了model/data/order/LoRA/optimizer/evaluator，但不是同quantizer的纯density-shape control：Geo使用native endpoint frequencies，EVQ使用midpoint quantization。以下“matched”均只指training pipeline matched；方法差异包含quantizer shift与cosh shape，不能只归因于density shape。事实边界以 `FULL_PAPER_INTEGRITY_AUDIT_20260713.md` 为准。

## Result to use in rebuttal

This is a fresh matched-training-pipeline seed-42 comparison after 300 LoRA steps on the frozen
LongAlpaca-12k training tensor. Evaluation uses a separately frozen 2026 corpus
from arXiv, the Federal Register, and Stack Overflow.

| Prefix | Geo base NLL | Geo+LoRA NLL | EVQ+LoRA NLL | Geo+LoRA − base | EVQ+LoRA − Geo+LoRA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 2.07296 | 1.91947 | 2.30933 | -0.15348 | +0.38986 |
| 16K | 5.01426 | 4.69096 | 3.18088 | -0.32330 | **-1.51008** |
| 32K | 7.30849 | 6.89919 | 4.85134 | -0.40929 | **-2.04786** |

The clean reviewer-facing statement is:

> On a frozen 2026 temporal holdout spanning arXiv, the Federal Register, and
> Stack Overflow, EVQ+LoRA changes NLL relative to matched Geo+LoRA by +0.390 at
> 8K, -1.510 at 16K, and -2.048 nats/token at 32K. The 16K and 32K direction is
> consistent across all 3 domains and all 24 packs. Geo uses native endpoint
> frequencies while EVQ uses midpoint quantization, so this is not a pure
> density-shape comparison.

Do not describe the 8K difference as a “48% tradeoff.” PPL is `exp(NLL)`, so a
percentage computed in PPL space visually inflates the in-window difference and
is not commensurate with the much larger long-window NLL reductions. PPL may be
reported descriptively, but NLL is the primary comparison.

## Why this is useful

- The 16K and 32K EVQ-minus-Geo+LoRA deltas are negative for 24/24 independent
  packs and 3/3 domains; the 8K delta is positive for 24/24 packs.
- The position-bucket transition occurs outside the 8K training window:
  EVQ-minus-Geo+LoRA is `+0.201` at 0–4K and `+0.578` at 4–8K, then `-2.868` at
  8–12K, `-3.950` at 12–16K, `-2.550` at 16–24K, and `-2.621` at 24–32K.
- Geo+LoRA also improves on the untouched Geo base at every prefix and in every
  pack. That is a general fine-tuning effect; the EVQ-minus-Geo+LoRA delta is
  evidence for the combined schedule intervention, not an isolated cosh-density effect.
- The test text is not the LongAlpaca training corpus. The result therefore
  supports cross-domain temporal transfer rather than training-set-only PPL
  improvement.

## Exact experiment identity

Both adapters use seed 42, BF16, LoRA rank 64 on q/k/v/o, batch 2 with gradient
accumulation 4, 8K sequences, 300 optimizer steps, learning rate `1e-4`, the
same split seed, tokenizer, model bytes, data bytes, objective, optimizer,
scheduler, checkpointing, and compile mode. Their recorded run protocols differ
in `method` and `tau`; scientifically, the method switch also changes native
endpoint Geo to midpoint EVQ, so it is more than a same-grid tau-only contrast.

| Identity | SHA256 |
| --- | --- |
| Frozen LongAlpaca training manifest | `1a610863e6602091a524e45c3ebc5907b6d0deae78851d3b0fd23ef9585e587f` |
| Base-model manifest | `0196fe3f3dcd932e337a7a0e91625fd12667cecbcca221020ea39428c6178210` |
| Shared seed-42 training code bundle | `cacaf7dfdba9adc03319433ad28adb7e64df3fc6bff0da6903a0e8ad88b1e0fb` |
| Geo adapter | `0e7efa6e83e74166a3ae5a0db6997124f881badcd7ddbdcae49cb6d211ebec9a` |
| EVQ adapter | `8ea0423473793bb8f2ccfed46f25e67c44cd75011620542664246fa598d0c780` |
| Geo frequency artifact | `09fab0f4da1c96bf31cf5d54dd45a935dc63f1ef8d7b1669dd5361087e4bb794` |
| EVQ frequency artifact | `49b205926ae989cdf973ceaa612e0dab937bf21fd63f095f72485ba3d4a4bacf` |
| Temporal collection manifest | `187c8ff86f0e52229b0ca6c6a55b506c4e536c959a95bb5bbe06399c1611879e` |

The LongAlpaca source receipt is a best-effort reconstruction from recovered
public bytes: raw SHA256
`090b348755c728cfb4b13fa443b54a0cb4e5128a92f142c6a52c5b4ff0f55e1f`,
with the upstream revision unresolved. Of 8,000 selected rows, 7,628 passed the
minimum-token filter; the fixed split contains 7,476 train and 152 validation
rows.

## Temporal-holdout provenance

Each domain contains eight disjoint 32K packs; 8K and 16K are exact causal
prefixes of those same packs. The 24 packs score 194,794 tokens at 8K, 389,672
at 16K, and 779,400 at 32K per arm. Documents were selected by verifiable 2026
timestamps, and the collection was frozen before any temporal evaluation.

The collection was finalized in domain-specific runs after the original
one-shot fetch encountered source/network failures. These helpers only
orchestrated the canonical fetch/write functions and did not observe model
outputs or select examples by score:

| Domain artifact finalized | Operator-helper SHA256 |
| --- | --- |
| arXiv, `2026-07-12T10:44:42Z` | `6a27c5a671eab6233dae0c1d924cfaa39ac3b2b8913fc0abce81a90cbf73e0b5` |
| Federal Register, `2026-07-12T11:22:25Z` | `e2a37802eab170dc081d4eb1e6804942e4aa0a4ff1834b93945881feec610b8a` |
| Stack Overflow and collection, `2026-07-12T11:25:39Z` | `9da2a2994f34707b289f8ba9f21de23799fee8a538fc340467800c558461d9ff` |

The collection manifest was written at `2026-07-12T11:25:39.457202Z`.
Its `frozen_through=2026-07-12T23:59:59Z` field is a query eligibility bound,
not the actual acquisition-completion time. The manifest's canonical
preparation-code hash records the final on-disk generator, so the helper hashes
above are required to describe the split-run construction accurately.

## Claim boundary and next gate

This result is single-seed supporting evidence. It measures teacher-forced NLL
on absolute positions in packs made by concatenating shorter documents. It does
not by itself prove long-range retrieval, QA, generation accuracy, or universal
zero-overlap generalization. Timestamp separation is strong evidence of a fresh
holdout, not proof of zero phrase-level pretraining overlap.

The three-arm design also has no Base-EVQ arm. It supports the end-to-end
EVQ+LoRA versus Geo+LoRA pipeline comparison, not a pure density-shape claim or
a claim that LoRA alone learned the
mechanism. The cost-controlled follow-up trains only EVQ+LoRA seeds 43 and 44 on
the same frozen data and evaluates each against the already completed Geo+LoRA
seed-42 reference. Afterward, report EVQ's three-seed stability and show the
fixed Geo-42 reference separately; do not label it a three-seed paired
Geo-versus-EVQ comparison.
