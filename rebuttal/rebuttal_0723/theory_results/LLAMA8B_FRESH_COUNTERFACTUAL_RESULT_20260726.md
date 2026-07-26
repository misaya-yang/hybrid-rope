# LLaMA-3-8B fresh EVQ counterfactual result

Date: 2026-07-26
Status: **training and all registered evaluations complete**
Evidence tier: **post-submission remote-raw-backed / single seed**
Reviewer routing: `R27bE.2`, `R27bE.5`, `AC.2`

## Direct verdict

The fresh LLaMA-3-8B EVQ-LoRA arm learns strong counterfactual binding at its
physical 8K training length, but that capability does not transfer reliably to
2x or 4x length. On `niah_single_1`, strict first-generated-number exact is
`20/20`, `6/20`, and `0/20` at 8K, 16K, and 32K. Across the full 13-task RULER
matrix, official macro is `32.69%`, `4.20%`, and `0.51%`, while normalized
full-output exact is zero at every length.

Temporal natural-text PPL exhibits a different boundary. Relative to the
untouched Native base, the adapted arm is worse at 8K and 16K but better at
32K:

| Prefix length | Native-base NLL / PPL | Fresh EVQ-CF NLL / PPL | Adapted-minus-Native NLL |
| ---: | ---: | ---: | ---: |
| 8K | `2.0729 / 7.95` | `4.5061 / 90.57` | `+2.4332` |
| 16K | `5.0144 / 150.56` | `5.5117 / 247.56` | `+0.4973` |
| 32K | `7.3089 / 1493.50` | `6.4482 / 631.55` | `-0.8607` |

The 32K PPL reduction therefore does **not** establish 32K generation
capability: the same adapter obtains `0/20` strict `niah_single_1` exact and
`0.51%` full-suite official RULER macro at 32K.

## Experiment identity

- Base model: Meta-Llama-3-8B-Instruct.
- Frequency substrate: full EVQ-Cosh injection.
- Frequency settings: base `500,000`, head dimension `128`, midpoint
  quantization, `tau=1.414`.
- Frequency artifact SHA-256:
  `49b205926ae989cdf973ceaa612e0dab937bf21fd63f095f72485ba3d4a4bacf`.
- Adapter initialization: fresh LoRA; no parent adapter.
- Seed: `42`.
- Physical training length: `8,192`.
- Long-context backward passes: none.
- Virtual position IDs: none.
- Training steps: `300`.
- Batch pattern: `counterfactual, counterfactual, natural`.
- Effective allocation: approximately 200 counterfactual batches and 100
  natural-replay batches.
- Micro-batch / accumulation / global batch: `2 / 4 / 8`.
- Learning rate / warmup: `1e-4 / 60 steps`.
- Weight decay: `0.01`.
- Optimizer: fused AdamW, betas `(0.9, 0.95)`.
- Precision: BF16 autocast.
- Attention: Flash-only SDPA; no math-attention fallback.
- LoRA: rank `64`, alpha `128`, dropout `0.05`, targets
  `q_proj/k_proj/v_proj/o_proj`.
- Trainable parameters: `54,525,952`.
- Counterfactual margin / weight: `1.0 / 0.5`.

This is not the staged OLMo protocol. In particular, it does not contain the
OLMo 611-step, approximately 20M-token Stage-A natural-text adaptation before
counterfactual routing.

## Counterfactual data and objective

The paired training set contains 576 source rows over:

- `niah_single_1`
- `niah_single_2`
- `niah_single_3`
- `niah_multikey_1`
- `niah_multikey_2`
- `niah_multikey_3`

Each source row has a sourced and value-swapped variant at physical length 8K.
The loss combines answer-only CE, including EOS, with a source-preference
counterfactual margin. The frozen calibration set contains 24 disjoint source
rows, 48 variants, and 526 supervised answer tokens. Train/calibration source
overlap is zero.

Natural replay uses 128 rows drawn from the existing physical-8K 13-family
training view. The registered training view contains 1,376 training rows and 52
validation rows; its objective is answer-only CE.

## Training execution

| Field | Result |
| --- | ---: |
| Training time | `2244.40 s` (`37.41 min`) |
| Steady training throughput | `8759.93 tokens/s` |
| Probe throughput | `9329.68 tokens/s` |
| Peak allocated GPU memory | `93,432,949,760 bytes` |
| Compile mode | `max-autotune-no-cudagraphs` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| PyTorch / CUDA | `2.8.0+cu128 / 12.8` |

## Frozen counterfactual calibration

| Metric | Initial | Final |
| --- | ---: | ---: |
| Mean answer NLL | `8.1805` | `1.7700` |
| Token exact | `1.90%` | `63.88%` |
| Sequence exact | `0.00%` | `47.92%` |
| Source-preference-positive fraction | `48.48%` | `68.63%` |
| Mean source-preference margin | `1.6821` | `9.4925` |

Interpretation: the adapter learned the registered paired binding objective,
but the final calibration is materially below the `96%–98%` token-exact level
of the earlier OLMo routing arms. This comparison is diagnostic only because
the models, task coverage, physical lengths, data volumes, and staged training
protocols differ.

## Full 13-task RULER result

Each cell has `n=20`; each length therefore has 260 examples and the complete
matrix has 780 examples. Values below are official task-specific scores.

| Task | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| `niah_single_1` | `100.00%` | `30.00%` | `0.00%` |
| `niah_single_2` | `95.00%` | `0.00%` | `0.00%` |
| `niah_single_3` | `10.00%` | `0.00%` | `0.00%` |
| `niah_multikey_1` | `95.00%` | `5.00%` | `0.00%` |
| `niah_multikey_2` | `0.00%` | `0.00%` | `0.00%` |
| `niah_multikey_3` | `0.00%` | `0.00%` | `0.00%` |
| `niah_multivalue` | `58.75%` | `5.00%` | `0.00%` |
| `niah_multiquery` | `62.50%` | `1.25%` | `0.00%` |
| `vt` | `1.00%` | `0.00%` | `0.00%` |
| `cwe` | `1.00%` | `0.00%` | `0.00%` |
| `fwe` | `1.67%` | `13.33%` | `6.67%` |
| `qa_1` | `0.00%` | `0.00%` | `0.00%` |
| `qa_2` | `0.00%` | `0.00%` | `0.00%` |
| **Official macro** | **`32.69%`** | **`4.20%`** | **`0.51%`** |
| **Normalized full-output exact macro** | **`0.00%`** | **`0.00%`** | **`0.00%`** |

The official metric is not strict full-string exact. The independently
recomputed strict first-generated-number result for `niah_single_1` is:

| Length | Strict first-number exact |
| ---: | ---: |
| 8K | `20/20` |
| 16K | `6/20` |
| 32K | `0/20` |

## Temporal natural-text NLL/PPL

Protocol:

- timestamp-selected 2026 temporal holdout;
- three domains: `arxiv_2026`, `federal_register_2026`,
  `stackoverflow_2026`;
- eight 32K packs per domain;
- nested 8K/16K/32K prefixes;
- document-first tokens excluded by the frozen mask;
- teacher-forced NLL/PPL only.

| Prefix | Native-base PPL | Fresh EVQ-CF PPL | Adapted / Native |
| ---: | ---: | ---: | ---: |
| 8K | `7.95` | `90.57` | `11.395x` |
| 16K | `150.56` | `247.56` | `1.644x` |
| 32K | `1493.50` | `631.55` | `0.423x` |

At 32K, the adapted PPL is approximately `57.71%` lower than the untouched
Native base. The result is nevertheless not a matched Native-LoRA control and
cannot identify whether the long-position change comes from EVQ, specialized
counterfactual training, natural replay, or their interaction.

## Evidence classification and rebuttal use

### What this run establishes

1. A fresh LLaMA-3-8B EVQ-LoRA can learn source-dependent counterfactual
   binding at the 8K training length.
2. Some same-family RULER behavior survives at 16K, but only weakly.
3. The 32K temporal PPL improvement can coexist with essentially absent 32K
   autoregressive task capability.

### What this run does not establish

- It is not a matched Native-versus-EVQ counterfactual comparison.
- It is not a replication of the staged OLMo protocol.
- It is one training seed.
- It is task-family-adapted, not unseen-task transfer.
- It does not demonstrate downstream QA capability.
- It does not demonstrate reliable 2x or 4x RULER capability.
- It does not provide training-seed uncertainty or statistical significance.
- It does not attribute the result purely to frequency allocation.

### Recommended rebuttal role

Do not use this as the headline 8B result. The existing matched LLaMA
Native/EVQ NLL and 16K task-family RULER package is stronger for attribution.
If this run is mentioned, use it only as a bounded feasibility statement:

> In a separate single-seed LLaMA-3-8B experiment, fresh EVQ-LoRA trained with
> paired source-content counterfactual supervision reaches 20/20 strict
> `niah_single_1` exact at its 8K training length, but falls to 6/20 at 16K and
> 0/20 at 32K. We therefore do not use this arm as evidence of reliable
> beyond-training-length capability.

Do not infer capability from its 32K PPL reduction.

## Remote provenance

The result artifacts were retained on the experiment instance and identified
by the hashes below. The instance was stopped after the final status, metrics,
and hashes were captured.

| Artifact | SHA-256 |
| --- | --- |
| Training result JSON | `52f95be5265b084d4e37191dc7db8fbf58229d3984339dbddff3c42f46da87ec` |
| Adapter safetensors | `2768ada3d3761d9d520d6abb13ae16c53b5913b48d74230e83356f1efb4b7d7e` |
| RULER result JSON | `e8c34077d4cb17adc536b83d1b9280727667b2817e86998ebd11258fa19fbc8a` |
| RULER predictions JSONL | `0b7136ca13e6f225637c47c3f1206a852045c4aa07b3c6e5077cc43af94d63e1` |
| Temporal result JSON | `49312845973049db2243c1492675945990b0ec620246f4bab1ad3677ac6a287d` |
| Training log | `48f90b7c10c62bc3c761081e3dc5ba5ee88d5bead3372d585b19186805af19da` |
| Evaluation queue log | `a1d02ae28a37948751f5ac55fce534f51cc5b0f6a6716818d966859c7bf98611` |
| Temporal retry log | `f8cb07f8c9af1ef6135692f6cd5e159d310a50bfea08a1620423c8014c75f2ca` |
| READY receipt | `8ee7d25c3ab24d7378d1f7edd33b5525166617f0df15490162808e2aa7e7d864` |

The result report records the final numeric outputs and remote hashes; the
adapter and bulk training arrays were intentionally not copied into the
repository.
