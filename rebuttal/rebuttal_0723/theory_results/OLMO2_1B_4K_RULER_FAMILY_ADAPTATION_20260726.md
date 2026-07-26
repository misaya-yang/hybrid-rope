# OLMo-2 1B: 4K RULER-family adaptation and 4K–16K capability

Status: **complete single-seed supporting result**  
Concerns: `R27bE.2`, `R27bE.5`, `AC.2`

## Decision record

1. **Concern addressed.** Can a mature approximately 1B EVQ model convert
   4K-only adaptation into autoregressive capability across the complete
   13-task RULER matrix at 4K, 8K, and 16K?
2. **Existing evidence.** Two earlier EVQ adapters learned one official-RULER
   NIAH family but had weak coverage outside NIAH. Separate adaptation with no
   explicitly added RULER rows did not recover broad RULER capability.
3. **Smallest missing evidence.** Add short, explicit 4K supervision for the
   five uncovered RULER task families while replaying NIAH and natural
   instruction rows, then rerun the unchanged complete matrix.
4. **Smallest executable plan.** Continue the stronger seed-20260725 EVQ
   adapter for 276 steps on a fixed 4K mixture and evaluate 13 tasks x 3
   lengths x 20 examples autoregressively.
5. **Stop condition.** Stop after this one continuation seed and complete
   matrix. Do not open another mechanism search unless the held-out matrix
   improves without erasing the previously learned NIAH family.

## Protocol

- Base model: `OLMo-2-0425-1B-Instruct`.
- Frequency substrate: endpoint EVQ-Cosh, \(\tau=2\), base \(500{,}000\);
  frequency SHA256
  `917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607`.
- Parent: seed-20260725, rank-64/alpha-128 Q/K/V/O LoRA trained at physical
  4K; adapter SHA256
  `95ceeb70117c73233915760a9756b9b2a98416ec188b125054da8ced75cad16a`.
- Continuation data: 480 official-generator synthetic training rows
  (96 each for VT, CWE, FWE, SQuAD QA, and HotpotQA), 128 paired NIAH replay
  rows, and 128 LongAlign assistant replay rows. A separate 20-row internal
  validation split contains four rows per added task.
- Every backward pass remains at physical length 4,096. No virtual 8K/16K
  positions or long-sequence backward pass is used.
- Training: three deterministic passes, 276 optimizer steps, global batch 8,
  BF16, fused AdamW, learning rate \(2\times10^{-5}\), 14 warmup steps,
  Flash-only SDPA, and TorchInductor
  `max-autotune-no-cudagraphs`.
- Training processed 9,041,760 physical input tokens. The complete run,
  including first compile, took 431.6 seconds; steady-state throughput was
  approximately 30.3K physical token/s on an RTX 5090.
- Evaluation: official task-specific autoregressive RULER scoring on all
  13 tasks at 4K/8K/16K, \(n=20\) per cell, for 780 generations. VT/CWE/FWE
  and NIAH use `string_match_all`; QA uses `string_match_part`.
- Training and evaluation use different generated rows. Exact serialized-row
  overlap is zero. They still share the official RULER generator families.

The added task validation NLL falls from 3.0372 to 0.9881. This is an internal
optimization check, not the capability result below.

## Complete-matrix result

Scores are fractions of the official task-specific RULER metric. “Legacy”
is the mean of the two previously completed EVQ NIAH-style adapter seeds on
the same 780-example matrix. The new arm is one continuation seed.

| Task group | Arm | 4K | 8K | 16K |
| --- | --- | ---: | ---: | ---: |
| All 13 tasks | Legacy two-seed mean | 24.89% | 15.19% | 4.08% |
| All 13 tasks | New 4K task-family adaptation | **37.51%** | **21.29%** | **6.13%** |
| Eight NIAH tasks | Legacy two-seed mean | 30.23% | 16.56% | 1.25% |
| Eight NIAH tasks | New 4K task-family adaptation | **34.38%** | 16.09% | 1.09% |
| VT/CWE/FWE/QA | Legacy two-seed mean | 16.33% | 13.00% | 8.60% |
| VT/CWE/FWE/QA | New 4K task-family adaptation | **42.53%** | **29.60%** | **14.20%** |

The macro over all 39 cells rises from 14.72% for the legacy two-seed mean to
21.65% for the new single continuation seed: \(+6.93\) percentage points, or
approximately \(+47\%\) relative. This comparison diagnoses the value of
broader task-family adaptation; it is not a training-seed confidence interval.

## Per-task result

| Task | Legacy mean 4K / 8K / 16K | New 4K / 8K / 16K |
| --- | ---: | ---: |
| NIAH single 1 | 100 / 62.5 / 0 | 100 / 45 / 0 |
| NIAH single 2 | 40 / 12.5 / 0 | 50 / 20 / 0 |
| NIAH single 3 | 5 / 0 / 0 | 10 / 0 / 0 |
| NIAH multikey 1 | 45 / 27.5 / 5 | 65 / 35 / 5 |
| NIAH multikey 2 | 22.5 / 10 / 0 | 10 / 5 / 0 |
| NIAH multikey 3 | 0 / 0 / 0 | 0 / 0 / 0 |
| NIAH multivalue | 18.13 / 10.63 / 2.5 | 22.5 / 17.5 / 1.25 |
| NIAH multiquery | 11.25 / 9.38 / 2.5 | 17.5 / 6.25 / 2.5 |
| VT | 0 / 0 / 0 | **68 / 31 / 0** |
| CWE | 0 / 0 / 0.5 | **38 / 22 / 1** |
| FWE | 39.17 / 30 / 27.5 | **66.67 / 60 / 50** |
| SQuAD QA | 15 / 15 / 5 | 15 / 10 / 10 |
| HotpotQA | 27.5 / 20 / 10 | 25 / 25 / 10 |

The gain is not uniform. VT, CWE, and especially FWE improve strongly and
retain nonzero 8K capability after only 4K adaptation. QA remains low and is
essentially unchanged. Sixteen-k capability remains task-dependent: FWE
reaches 50%, while VT remains at zero.

## Interpretation and claim boundary

This result closes a narrower practical question: a mature EVQ model can learn
multiple long-context task families using only 4K backward passes, then retain
measurable autoregressive capability on different 8K and 16K rows. It also
shows that adding the missing task supervision need not erase the earlier NIAH
family.

It does **not** establish any of the following:

- unseen-task or benchmark-independent transfer;
- an EVQ-versus-Native causal advantage, because no matched Native arm received
  this same continuation mixture;
- a clean pretrained-model comparison to official YaRN;
- broad downstream QA success;
- training-seed uncertainty or production-scale SOTA.

The safe rebuttal wording is:

> With all adaptation sequences capped at 4K, a short RULER-family-matched
> continuation of an approximately 1B EVQ Instruct model reaches 37.5%,
> 21.3%, and 6.1% macro score across the complete 13-task RULER matrix at
> 4K, 8K, and 16K. The 8K result includes 31% VT, 22% CWE, and 60% FWE.
> Training and evaluation rows are disjoint, but task generator families are
> shared; we therefore present this as task-adapted capability and length
> transfer, not unseen-task generalization or pure EVQ attribution.

## Provenance

- Final adapter SHA256:
  `b18f6cfee8aad7d9004938beec0a075834cc05c69c6433aaed1943c7a603a653`
- Fixed training-view manifest SHA256:
  `1bd1d35a40bbc450757c920028971df093885726153571ab02af47130681ef17`
- Synthetic-source manifest SHA256:
  `0e5d0deb2710843cf5bc1c9a39745b619a37b2fe85ce469420b039bcf5125f24`
- Formal evaluation manifest SHA256:
  `886d94731af31f30698204caa9f30045887c9333d6327e2e1c33555ed57da68c`
- Training result SHA256:
  `2c72b79c06b12632e1984760f10c16238aa8cf7a2bff5458b9c6ce0410893657`
- Evaluation result SHA256:
  `9e96ba9134aad63ff77af42ae2678068af7658c290e1716ba73cb80773f9d17f`
- 780-example prediction stream SHA256:
  `bf51a499ffdaeafeb03f7afa95c705dac4cf437a05485264fe65fe59ce3ce660`
- Legacy seed-20260725/20260726 result SHA256:
  `1e076e3015aebae2eba98cc0085892cb68f79fa09fc4d239afa312a66ab4b07c`
  / `b7b82aa622450dde14fabac99d48a4fe1edf5cb48e500b8fc79f50f6fe345acb`
- Preparation/training/launcher code SHA256:
  `bcffa9cb8129acf703e8e79a7c885fa2c5d9ce33fbaee615677ea783c65d5828`
  / `c4343f845e37aef16573de78cc5fe6bb078050f1cc74ae0b6e6e5e3d1a175744`
  / `f24f25039537ee68e2a28982806e789aadc97f8db2f9937cb0baf10310d81de6`

The curated numeric record is
`olmo2_1b_4k_ruler_family_adaptation_20260726.json`. Raw predictions and
training receipts remain in the experiment artifact store and are not copied
into the reviewer package.
