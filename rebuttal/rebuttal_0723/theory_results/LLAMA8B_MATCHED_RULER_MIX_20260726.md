# LLaMA-3-8B Matched 8K RULER-Family Adaptation

Date: 2026-07-26  
Status: complete at 8K and 16K; Native-LoRA 32K is partial after an evaluation-only OOM  
Evidence owner: `llama8b_matched_ruler_mix_20260726.json`

## Rebuttal routing

1. **Reviewer or AC concern addressed:** `R27bE.2`, `R27bE.5`, and `AC.2`
   ask whether EVQ remains useful in a mature, production-scale model and under
   stronger evaluation.
2. **Existing evidence:** the prior LLaMA-3-8B LoRA result established
   long-position NLL and remote-source effects, but did not establish broad
   autoregressive capability.
3. **Smallest missing evidence:** compare Native and EVQ under the same
   physical-8K task-family adaptation and the same 13-task autoregressive
   RULER evaluation.
4. **Smallest executed plan:** continue the matched Native and EVQ seed-42
   LongAlpaca adapters with one identical 8K RULER-family plus natural-replay
   recipe, then evaluate untouched Native, Native-LoRA, and EVQ-LoRA.
5. **Stop condition:** do not expand the claim if EVQ does not retain a
   non-zero advantage outside 8K, or if the result is confined to supervised
   generator families.

## Protocol

Both adapted arms use Meta-Llama-3-8B-Instruct, rank-64 Q/K/V/O LoRA, seed
`20420726`, 1,376 training rows, three epochs, 516 optimizer steps, global
batch eight, and 33,816,576 processed input tokens. Physical training length is
8,192; virtual position IDs and long-context backward passes are not used.

The training view contains 96 rows for each of the 13 RULER task families plus
128 natural-instruction replay rows. Evaluation uses different generated rows,
20 examples per task and length, greedy decoding, and the official RULER
metrics. Exact train/evaluation row overlap is zero. This is nevertheless
**task-family supervised adaptation**, not zero-shot or unseen-task transfer.

The execution stack is BF16, Flash-only SDPA, fused AdamW,
`max-autotune-no-cudagraphs`, and fused linear cross-entropy. Native and EVQ
training throughput is effectively identical.

## Training receipt

| Arm | Validation NLL, initial → final | Validation PPL, initial → final | Training time | Throughput |
| --- | ---: | ---: | ---: | ---: |
| Native-LoRA | 0.7805 → 0.3855 | 2.1827 → 1.4703 | 4,033.7 s | 8,383.5 tok/s |
| EVQ-LoRA | 1.7885 → 0.4092 | 5.9804 → 1.5056 | 4,031.4 s | 8,388.3 tok/s |

Both arms learn the supervised 8K validation view. Native begins much closer
to that distribution and ends slightly better; EVQ has to adapt from a much
larger initial operator mismatch.

## Autoregressive RULER

The first number is the official RULER macro; the second is the evaluator's
normalized-exact macro.

| Arm | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| Untouched Native | 91.82% / 5.77% | 0% / 0% | 0% / 0% |
| Native-LoRA | **94.44%** / 17.69% | 0.295% / 0% | 0% / 0% on 10/13 completed tasks |
| EVQ-LoRA | 77.60% / **21.54%** | **14.03% / 1.54%** | 0% / 0% |

All 13 Native-LoRA cells at 8K and 16K completed before the later failure and
were recomputed from the saved per-example predictions. At 16K, Native-LoRA's
only non-zero official cells are CWE (`0.5%`) and FWE (`3.33%`); every
normalized-exact cell is zero. EVQ-LoRA retains non-zero official scores in
multiple retrieval/counting cells and has normalized-exact successes in FWE
and QA2.

The 32K Native-LoRA shard for `niah_multivalue`, `niah_multikey_1`, and
`niah_single_3` did not start because four concurrent 8B evaluators exhausted
the 96GB GPU. The other ten Native-LoRA 32K cells are all zero. EVQ-LoRA and
untouched Native completed all 13 cells and are also zero. This is an
evaluation-orchestration failure, not a training failure, but a 13-task
Native-LoRA 32K macro must not be reported.

## External-domain temporal NLL

The completed temporal comparison contains eight 32K packs from each of
ArXiv-2026, Federal Register-2026, and Stack Overflow-2026. It compares
EVQ-LoRA with untouched Native; the matched Native-LoRA temporal arm did not
run after the RULER queue failed.

| Arm | 8K NLL / PPL | 16K NLL / PPL | 32K NLL / PPL |
| --- | ---: | ---: | ---: |
| Untouched Native | **2.0729 / 7.95** | 5.0144 / 150.56 | 7.3089 / 1,493.50 |
| EVQ-LoRA | 2.5068 / 12.27 | **3.4689 / 32.10** | **5.0994 / 163.93** |

This independently reproduces the expected trade-off: EVQ is worse in-window
but substantially more stable at 2x and 4x in teacher-forced language
modeling. It does not turn the zero 32K RULER result into capability evidence.

## Interpretation

The clean result is:

> Under identical physical-8K RULER-family adaptation, Native-LoRA maximizes
> the in-window official score, whereas EVQ-LoRA retains non-zero 16K
> autoregressive capability that both untouched Native and matched Native-LoRA
> lose.

This is the strongest current mature-8B evidence that the EVQ frequency
substrate changes **length transfer**, rather than merely lowering temporal
NLL. It also imposes three mandatory limits:

- it is single seed;
- training explicitly uses the same 13 RULER generator families;
- EVQ does not produce usable 32K RULER capability.

The comparison is matched across the full adaptation chain, but it remains a
combined frequency-substrate plus LoRA result. Native uses Llama's endpoint
geometric grid, while EVQ uses the midpoint Cosh grid; it is not pure
interior-shape attribution.

## Provenance

The local raw archive contains 60 hashed files (training receipts, per-example
predictions, evaluator outputs, manifests, and logs), totaling about 2.2MB;
adapter weights were not copied. Its checksum-manifest SHA-256 is
`f470e5cf9efd0e4c05dfbae0c21cb6ce8ed5b01db54ed38a9323caa9dd015700`.
Five key remote/local files were independently hashed after transfer and match.

