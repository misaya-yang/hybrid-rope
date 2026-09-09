# OLMo-2 1B released-RoPE baseline calibration

Date: 2026-07-25

Status: **paired step-1,000 Geo/EVQ evaluation complete / single training trajectory**

Evidence tier: **post-submission raw-hash-backed / reviewer-usable**

Source-hierarchy note: the sibling
`olmo2_1b_released_rope_baseline_20260725.json` is an earlier
`released_native_rope_baselines_only` snapshot. Its statement that no EVQ
result is included describes that JSON's deliberately native-only scope; it
does not contradict or own the later paired EVQ result. This report, together
with the retained Geo raw-result, EVQ checkpoint, EVQ raw/per-token, paired
comparison, and evaluation-anchor hashes below, is the canonical owner for the
completed Geo/EVQ comparison. Do not downgrade this result to conditional
because the native-only JSON does not duplicate the later EVQ fields.

## Reviewer or AC concern addressed

`R27bE.2`, `R27bE.5`, `AC.2`, and `AC.4`: test whether EVQ changes
long-context language modeling at approximately 1.5B parameters under a
matched from-scratch step-1,000 training budget, and calibrate that result
against the released native-RoPE trajectory.

## Existing evidence

The repository already has smaller controlled allocation studies and mature
8B LoRA evidence. The missing scale-transfer evidence is the pre-specified EVQ
branch from the public OLMo-2 step-0 initialization.

## Smallest missing evidence

The registered missing evidence was a paired evaluation of the EVQ step-1,000
checkpoint and released native-RoPE step-1,000 checkpoint on the exact same
document rows. That comparison is now complete.

## Protocol

- Model: OLMo-2 1B, 1,484,916,736 actual parameters.
- Architecture: full RoPE, `d_head=128`, `base=500K`, 4,096-token training
  context.
- Released checkpoints: steps 1,000/2,000/5,000 from the pinned early-training
  repository revisions. Under the pinned 512-sequence global batch, these
  correspond to 2.097B/4.194B/10.486B counted input tokens. The public branch
  labels are `tokens3B`/`tokens5B`/`tokens11B`.
- EVQ checkpoint: trained from the same public step-0 initialization for
  exactly 1,000 optimizer steps and 2,097,152,000 counted input tokens, with
  the same 512-sequence global batch, 4,096-token training context, pinned
  official recipe, and reconstructed seed-6198 data-order prefix.
- Trainer boundary: the EVQ branch uses the reviewed Hugging Face single-GPU
  loop, while the released Geo checkpoint was produced by AI2's distributed
  OLMo trainer. The valid identity is `same-initialization, same-recipe`, not
  a bitwise paired training trajectory.
- Evaluation: native RoPE with no inference-time range scaling; BF16 autocast,
  Flash-only SDPA, and `torch.compile`. The EVQ evaluation uses the same
  contract with the fixed EVQ frequency table.
- Natural-text anchors:
  - 128 document-disjoint PG-19 rows, each 16,384 tokens;
  - a separate 64 document-disjoint PG-19 set, each 32,768 tokens.
- Metric: PPL is \(\exp(\text{mean per-token NLL})\), not the arithmetic mean
  of per-document PPL. Tail PPL uses the final 1,024 predicted tokens.
- RULER calibration: the pinned upstream generators, four fixed tasks, three
  lengths, and the first 100 deterministic examples per cell.

## Natural-text PPL

### Same 64-document 32K anchor

| released step | 2K | 4K | 8K | 16K | 32K |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 185.61 | 163.57 | 162.01 | 180.69 | 224.83 |
| 2,000 | 67.42 | 66.30 | 75.49 | 93.87 | 125.38 |
| 5,000 | **30.76** | **31.80** | **44.43** | **72.83** | **117.47** |

The 128-document 16K anchor independently gives 16K PPL
182.73/95.10/73.71 at steps 1,000/2,000/5,000.

### Matched step-1,000 Geo versus EVQ

The registered comparison uses the same 128 document-disjoint PG-19 rows and
the same token prefixes at every length:

| Schedule, 2.097B tokens | 2K PPL | 4K PPL | 8K PPL | 16K PPL |
| --- | ---: | ---: | ---: | ---: |
| Released Geo step 1,000 | **177.99** | **161.19** | 163.88 | 182.73 |
| EVQ step 1,000 | 191.36 | 167.45 | **156.87** | **159.64** |
| EVQ relative change | +7.51% | +3.88% | **-4.28%** | **-12.64%** |

The paired full-token NLL deltas, defined as EVQ minus Geo, are
`+0.0724/+0.0381/-0.0437/-0.1351` at 2K/4K/8K/16K. Thus the matched EVQ
trajectory pays a short-context modeling cost but crosses over by 8K and
widens its advantage at 16K.

The document-paired bootstrap intervals exclude zero for all four full-NLL
comparisons:

| Length | Mean paired NLL delta | Document bootstrap 95% CI | Documents favoring EVQ |
| ---: | ---: | ---: | ---: |
| 2K | +0.0724 | `[+0.0664, +0.0779]` | 3/128 |
| 4K | +0.0381 | `[+0.0332, +0.0428]` | 7/128 |
| 8K | **-0.0437** | `[-0.0494, -0.0380]` | **122/128** |
| 16K | **-0.1351** | `[-0.1420, -0.1281]` | **126/128** |

The final-1,024-token PPL gives the same and stronger long-range direction:

| Schedule | 2K tail | 4K tail | 8K tail | 16K tail |
| --- | ---: | ---: | ---: | ---: |
| Released Geo step 1,000 | **138.49** | 144.55 | 168.70 | 214.63 |
| EVQ step 1,000 | 151.92 | **144.27** | **148.43** | **172.60** |

At 16K, the tail-NLL delta is `-0.2179`, its bootstrap 95% interval is
`[-0.2296, -0.2063]`, and all 128 documents favor EVQ. These intervals measure
held-out document sampling, not training-seed uncertainty.

### Mean NLL on the same 64 documents

| released step | 4K | 8K | 16K | 32K |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 5.0972 | 5.0877 | 5.1968 | 5.4153 |
| 2,000 | 4.1942 | 4.3240 | 4.5420 | 4.8313 |
| 5,000 | **3.4594** | **3.7939** | **4.2881** | **4.7662** |

### Tail PPL on the same 64 documents

| released step | 4K | 8K | 16K | 32K |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 144.51 | 163.08 | 211.26 | 309.92 |
| 2,000 | 66.46 | 88.96 | **120.55** | **183.55** |
| 5,000 | **33.57** | **78.82** | 136.70 | 222.14 |

## RULER calibration at released step 5,000

This is a bounded diagnostic subset, not an official full-suite RULER score.
Each cell has 100 examples.

| task | 4K | 8K | 16K |
| --- | ---: | ---: | ---: |
| single needle | **95.0%** | 40.0% | 0.0% |
| multi-key needle | 0.0% | 0.0% | 0.0% |
| variable tracking | 0.2% | 0.0% | 0.0% |
| frequent-word extraction | 4.0% | 1.0% | 0.0% |

The 12-cell macro average is 11.68%. The single-needle result is the clearest
calibration: the model retrieves reliably at its 4K training limit, degrades
at 8K, and fails at 16K.

### Early-checkpoint floor gate

A denser 4K/5K/6K/7K/8K/9K/10K sweep was frozen for the released step-1,000
and step-2,000 checkpoints. Each planned cell contained 100 evaluated examples
from a 500-example frozen pool.

- Step 1,000 scored 0/100 at 4K and 0/100 at 5K; the next 35 examples at 6K
  were also all zero.
- Step 2,000 scored 0/100 at 4K; the next 46 examples at 5K were also all
  zero.

Both runs were stopped after the complete in-window 4K single-needle gate was
zero. Continuing the longer cells could not identify a meaningful length
breakpoint and would only spend GPU time measuring the same task floor. This
negative result separates language-model PPL from learned retrieval ability:
step 2,000 has 4K PPL around 66 on PG-19 but no measurable single-needle RULER
capability under this protocol.

## Interpretation

The baseline is no longer at the “cannot model language” floor. At step 2,000
it reaches 16K PPL 95.10 on 128 held-out documents; at step 5,000 it reaches
73.71. A future EVQ improvement at the matched step therefore cannot be
dismissed solely as comparison against a random or nonfunctional model.

At the same time, later training improves short-context modeling faster than
long-context extrapolation. On the 64-document anchor, the 32K-to-4K PPL ratio
grows from 1.37 at step 1,000 to 1.89 at step 2,000 and 3.69 at step 5,000.
The final-1K-token 32K PPL even rises from 183.55 at step 2,000 to 222.14 at
step 5,000 while full-sequence PPL improves. RULER independently exposes the
same length break.

The step-1,000 result is positive scale-transfer evidence for the
language-modeling endpoint: with the same initialization family, parameter
count, training context, and counted-token budget, changing the fixed
frequency allocation shifts modeling quality away from 2K/4K and toward
8K/16K. It is stronger than an unmatched maturity comparison and cannot be
explained by giving EVQ more training tokens. The different trainer stacks
remain a residual implementation confound, so this row must not be promoted
to a bitwise paired trajectory or multi-seed causal estimate.

It is not yet a capability result. The released step-1,000 model is at the
registered 4K RULER floor, and no RULER score has been produced for the new
EVQ step-1,000 checkpoint. A further continuation or capability evaluation
must be motivated separately rather than inferred from PPL.

## Provenance

- Native-only released-baseline snapshot (not the owner of the later EVQ
  comparison): `olmo2_1b_released_rope_baseline_20260725.json`.
- Released Geo step-1,000 raw result SHA256:
  `34ad3eebb266c0bb8b47fefb026422942b04f9b4d6b8fe920ec7da1bc3cbdd7a`.
- EVQ step-1,000 checkpoint SHA256:
  `b15d9aa805d336d031b45ed0985db86052cf30583fdf5aee1976a3fec607b7ac`.
- EVQ step-1,000 raw result / per-token NLL SHA256:
  `f068cd61ed33b8b2184bd1c35c5dfd264e484538d6bc05ac408e3dd1931fbe29`
  / `bf7849cc139c3da153198d153fc3cd9fe7046ae44c09e3fdab646deffaf997bd`.
- Paired comparison SHA256:
  `d1fa790f495b29e9f8e7a536e323e87d0feb9b8c076b6f1c8ff3e64da8453095`.
- Evaluation manifest / 128-document anchor SHA256:
  `7f871db06206df6a483bbafab352db551c560356d7cc1a85a0e8bd0a9663c7e6`
  / `3052ce4e574a4317bad7bdfb78363e29d0fe32b44a6433132f05e743579e2f1a`.
- Corrected isolated Flash-only evaluator / READY receipt SHA256:
  `8e81bfef11f5522f7b4389043095c7886c24b370880fa2ab395a04e5ec29b5c9`
  / `c7487f0e980b1662590a9606028bfe6a6092c1214040bbef528aed14d6092a92`.
- PG-19 16K anchor SHA256:
  `3052ce4e574a4317bad7bdfb78363e29d0fe32b44a6433132f05e743579e2f1a`.
- PG-19 32K anchor SHA256:
  `692db521d8c19ddd8f2668b9f06142e25a4c60646032abff7747394cbde08626`.
- RULER manifest SHA256:
  `80c64fe8b7af3a169ef3f12e6d1a2228fe7462aa23bde081b94fd584b37561d1`.
- RULER aggregate SHA256:
  `d1db7d481d18b704dc8b699135dfcf2552969b696b001e7160a421310ae58eef`.
- RULER 1,200-example output SHA256:
  `570b827eac52578a0420e7a087a0eef13cf874c3f7a2e39a11fafccfa94ae82c`.
- Dense 4K–10K RULER manifest SHA256:
  `fbb4f410b256bd502b0149954a8e6f8c1cd28a75cd2da9cc3068435ca20fc3bd`.
- Early-stop step-1,000/step-2,000 output SHA256:
  `36de2a8f31759cdb25a86e19df8da1c83cc03747b46a5fcf3698de76c34f5844`
  / `b803ab49dccecbd7baf95c086c56a5f72bfb5ac7baafbbee22c43e2a7455fc34`.
- All six raw PPL result hashes are retained in the curated JSON.

## Claim boundary

- The matched Geo/EVQ result is one from-scratch training trajectory, not a
  multi-seed scale result.
- The EVQ run and released Geo control are same-initialization/same-recipe
  trajectories with an exactly reconstructed data-order prefix, but they use
  different trainer stacks. They are not bitwise paired training runs.
- The result identifies the complete fixed EVQ schedule against the released
  native Geo schedule; it does not isolate interior shape at matched sampled
  endpoints.
- EVQ was evaluated through 16K on the registered 128-document anchor. A
  matched EVQ 32K result has not been run.
- The three released checkpoints provide one training trajectory, not
  multi-seed evidence.
- The step-5,000 RULER run calibrates evaluation sensitivity; it is not the
  matched control for EVQ step 1,000.
- Step 1,000 and step 2,000 RULER are early-stop floor gates, not complete
  seven-length aggregates.
- No RULER or downstream-capability result is claimed for the new EVQ
  step-1,000 checkpoint.
- Natural-text PPL, tail PPL, RULER retrieval, and downstream task accuracy
  remain distinct endpoints.
