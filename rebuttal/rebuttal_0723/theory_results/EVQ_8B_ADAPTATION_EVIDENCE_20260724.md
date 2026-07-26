# EVQ on a mature 8B model — adaptation and capability-conversion evidence

Date: 2026-07-24

Status: **tracked-report and curated-JSON backed / supporting / single seed**

Paper claim: **false**

## Reviewer or AC concern addressed

- `R27bE.2` / `R27bE.5` / `AC.2`: whether the intervention is confined to
  small models trained from scratch and how far the evidence reaches toward
  downstream capability.

This record is not an exact-range attribution experiment. Its role is
mature-model applicability and endpoint separation.

## Experiment identity

- Base model: Meta-Llama-3-8B-Instruct.
- Training seed: 42.
- Adaptation: 300 optimizer steps, BF16, rank-64 / alpha-128 LoRA on Q/K/V/O,
  8,192-token sequences, learning rate \(10^{-4}\).
- Shared between arms: frozen training rows and order, model and tokenizer
  bytes, seed, objective, optimizer, scheduler, batch/accumulation, LoRA
  capacity, checkpointing, compile mode, and evaluator.
- Native-LoRA control: the exact original Llama endpoint frequency grid. Source
  reports also call this arm `Geo+LoRA`; it is not Paper-Geo midpoint or
  uniform FMRoPE.
- EVQ-LoRA arm: EVQ-Cosh at \(\tau=1.414\), using midpoint quantization.

The contrast therefore measures the end-to-end effect of assigning different
frequency substrates under a fixed adaptation pipeline, including the LoRA
co-adaptation they induce. It does not isolate Cosh interior shape from the
endpoint-versus-midpoint quantizer shift, nor establish transfer beyond this
matched adaptation dataset and readout.

## Level 1: temporal language-model probability

Evaluation uses 24 frozen 2026 temporal packs from arXiv, the Federal Register,
and Stack Overflow. Negative NLL delta means EVQ-LoRA is better.

| Prefix | Native-LoRA NLL | EVQ-LoRA NLL | EVQ minus Native |
| --- | ---: | ---: | ---: |
| 8K | 1.91947 | 2.30933 | +0.38986 |
| 16K | 4.69096 | 3.18088 | **-1.51008** |
| 32K | 6.89919 | 4.85134 | **-2.04786** |

The 16K and 32K direction holds in all 24 packs and all three domains; the 8K
direction is worse for EVQ in all 24 packs. Descriptive PPL is
6.817/108.958/991.475 for Native-LoRA and 10.068/24.068/127.911 for EVQ-LoRA
at 8K/16K/32K. NLL is the primary comparison because PPL-space percentages
inflate and make the in-window cost incommensurate with the long-window gain.

## Level 2: remote-source routing and causal use at true 16K

Ten frozen passkey cases contain exactly 16,384 tokens.

| Measure | Native-LoRA | EVQ-LoRA |
| --- | ---: | ---: |
| Median target-block hit@16 over 32 frozen retrieval heads | 18.75% | **64.06%** |
| Cases won by EVQ on hit@16 | — | **10/10** |
| NLL change after removing the gold block from every head | -0.0095 | **+1.5055** |
| Dense first-token median correct-token rank | 33,774.5 | **2,043.0** |

The all-head deletion result establishes causal use of remote answer
information in this diagnostic: removing the source lowers EVQ's geometric-mean
correct-token probability by a factor of \(\exp(1.5055)=4.51\), while the
Native-LoRA control is effectively unchanged.

## Level 3: autoregressive readout and downstream accuracy

The stronger remote signal does not yet complete the capability chain:

- true-16K dense generation exact match remains 0% in the ten-case diagnostic;
- the correct first token remains near rank 2,043 rather than top-1;
- a matched short retrieval micro-tune and sparse-attention variants do not
  convert the signal;
- on the registered 303-example Qasper/NarrativeQA generation-only gate,
  EVQ-LoRA task-macro F1 is 0.1126 versus 0.2110 for Native-LoRA. The aggregate
  deficit is concentrated at or below the 8K adaptation length; above 8K all
  arms are near the task floor and EVQ-versus-Native is unresolved.

The supported interpretation is:

> A fixed EVQ substrate can alter long-position probability and causal
> remote-source use in a mature 8B model after a low-cost matched LoRA
> adaptation. Stable conversion of that signal into top-1 generation and
> aggregate downstream accuracy remains open.

This is an applicability result and a capability-conversion decomposition. It
is not downstream superiority, exact retrieval, or pure interior-allocation
identification.

## Safe reviewer-facing paragraph

> The intervention is not confined to small models trained from scratch. In a
> matched seed-42 adaptation of Llama-3-8B-Instruct, 300 steps of rank-64
> Q/K/V/O LoRA change EVQ-minus-Native NLL by
> \(+0.390/-1.510/-2.048\) at 8K/16K/32K, with the long-range direction holding
> across all 24 packs and three domains. At true 16K, median target-block
> hit@16 rises from 18.75% to 64.06%, and removing the remote gold block from
> every head worsens EVQ NLL by 1.5055 while leaving the Native control
> essentially unchanged. This establishes mature-model probability and causal
> source-use effects under the matched adaptation protocol. The correct first
> token nevertheless remains around rank 2,000 rather than top-1, so we
> separate this substrate evidence from the still-open conversion to reliable
> generation and task accuracy.

## Provenance

- Temporal NLL source report:
  `../../pre_rebuttal/LORA_LONGALPACA_TEMPORAL_NLL_20260712.md`,
  SHA256
  `19271a2c428cffb5c24ff043b48522b27b0fd329672d1dabc0b99cd32eefc17b`.
- Curated temporal aggregate:
  `../../../data/curated/lora_longalpaca_temporal_s42_20260712.json`,
  SHA256
  `0335415a2245e1fb31149705342e975a016ddddb557a79c364fc4a98c3f89001`.
- Routing, causal deletion, rank, and generation report:
  `../../../docs/exp/2026-07-14_lora_retrieval_conversion_probe.md`,
  SHA256
  `5e05495ca8c8d295a4e3d3c4dc2c31bcbad81295a79947e7031010a08ad85a25`.
- Registered QA gate:
  `../../../docs/exp/2026-07-15_lora_qa16k_three_arm_results.md`,
  SHA256
  `024241e20eb30f99ad9f057a40683a69eb1b52a90f48e56481a45be8e1248bc0`.

## Claim boundary

- Single training seed.
- Native-LoRA and EVQ-LoRA are matched in adaptation capacity and data, but
  their frequency grids differ in both native-endpoint versus midpoint
  quantization and Cosh deformation.
- Teacher-forced NLL, routing, causal source dependence, correct-token rank,
  autoregressive exact match, and QA F1 remain distinct endpoints.
- This record does not close controlled full-pretraining scale transfer; that
  is the role of the pending OLMo-2 1B experiment.
