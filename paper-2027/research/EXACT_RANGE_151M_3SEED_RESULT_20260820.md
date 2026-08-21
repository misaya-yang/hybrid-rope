# Exact-range interior allocation at 151.9M: three-seed result

- **Date:** 2026-08-20
- **Status:** complete; raw-result hashes and per-seed contrasts frozen
- **Evidence tier:** post-submission, three independent training seeds,
  raw-hash-receipted
- **Paper role:** primary fixed-support identification of normalized interior
  allocation `z`
- **Supersedes for outward use:** the author-confirmed, locally unpromoted
  aggregate in
  `../../rebuttal/rebuttal_0723/theory_results/MATCHED_RANGE_COSH_500M_3SEED_20260724.md`
- **Paper integration:** the active manuscript uses this frozen three-seed
  result; this owner remains authoritative for future wording changes

## 1. Decision

The registered replication succeeds.

> With sampled frequency extrema and log-span fixed, changing only the 30
> interior frequencies changes trained-model behavior across independent
> training seeds.

At the fixed training range, anchored EVQ-Cosh minus FMRoPE
tail NLL is `-0.28073/-0.17599/-0.14571` at `512/1K/2K`. All three training
seeds favor anchored EVQ-Cosh at every OOD length. At the 256-token training length, anchored EVQ-Cosh
has a small, consistent `+0.02619` NLL cost.

When both grids are target-matched, the ordering reverses in all three seeds:
Anchored EVQ-Cosh minus FMRoPE is `+0.06032/+0.22720/+0.45959` at `512/1K/2K`. The result
therefore identifies interior allocation as a training-time variable but does
not establish additive gain over target-aware range transport.

## 2. Scientific question and protocol

The experiment asks whether interior allocation remains causally active after
every scalar support quantity has been pinned.

- Model: `151,898,880` parameters, 12 layers, hidden size 768, 12 heads,
  `d_head=64`, `K=32` rotary pairs.
- Training seeds: `42`, `137`, and `256`.
- Training length: 256 tokens.
- Training budget per arm: `499,974,144` tokens, `7,629` optimizer steps.
- Evaluation: 32 frozen anchors at `256/512/1024/2048`; primary metric is the
  paired final-128-token teacher-forced NLL.
- Baseline: paper-faithful FMRoPE grid at training base 256.
- Intervention: anchored EVQ-Cosh at `tau=4`.
- Held fixed within each seed pair: architecture, trainable initialization,
  row order, optimizer, LR schedule, global batch, token budget, evaluation
  anchors, sampled maximum and minimum frequencies, and log-frequency span.
- Changed variable: the locations of the `K-2=30` interior frequencies.

Seed 42 used micro-batch 64 with accumulation 4. Seeds 137 and 256 used the
runtime-equivalent micro-batch 128 with accumulation 2; all arms retain global
batch 256 and the registered optimizer-step/token contract. This is scientific
protocol identity, not bitwise trainer identity.

The regenerated manifest container has a different whole-file SHA-256 from
the historical seed-42 receipt because it is a separately generated receipt.
This report uses the matched scientific contract and per-seed paired results,
not byte identity of manifest metadata, as the aggregation criterion. Within
each new seed pair, every frozen scientific hash listed in Section 7 matches.

## 3. Primary fixed-range result

Negative values favor anchored EVQ-Cosh. The 95% intervals use a
two-sided Student-t interval across the three independent training seeds
(`df=2`, `t=4.3026527`). With only three seeds, they are descriptive
uncertainty summaries, not a basis for significance language.

| Length | Seed 42 | Seed 137 | Seed 256 | Mean | SD across seeds | 95% t interval | Anchored EVQ-Cosh wins | `exp(mean delta)-1` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | +0.03276 | +0.02903 | +0.01680 | +0.02619 | 0.00835 | `[+0.00545,+0.04693]` | 0/3 | +2.65% |
| 512 | -0.47750 | -0.27050 | -0.09418 | **-0.28073** | 0.19186 | `[-0.75735,+0.19589]` | **3/3** | **-24.48%** |
| 1,024 | -0.20499 | -0.19088 | -0.13211 | **-0.17599** | 0.03865 | `[-0.27202,-0.07997]` | **3/3** | **-16.14%** |
| 2,048 | -0.11284 | -0.18083 | -0.14347 | **-0.14571** | 0.03405 | `[-0.23029,-0.06114]` | **3/3** | **-13.56%** |

The 512-token effect varies substantially in magnitude, but not sign. The
1K and 2K contrasts are both directionally consistent and materially less
heterogeneous. The defensible primary statement is the complete three-length
vector plus `3/3` direction consistency, not a claim that every individual
length is statistically established from three seeds.

### Anchor-level direction

The per-seed fractions of the 32 paired evaluation anchors favoring anchored EVQ-Cosh are:

| Length | Seed 42 | Seed 137 | Seed 256 |
| ---: | ---: | ---: | ---: |
| 256 | 31.25% | 34.38% | 40.62% |
| 512 | 100.00% | 93.75% | 65.62% |
| 1,024 | 84.38% | 78.12% | 75.00% |
| 2,048 | 68.75% | 84.38% | 78.12% |

Evaluation anchors are paired observations within a seed, not independent
training seeds. They support the within-seed direction but must not replace the
three training seeds as the uncertainty unit.

## 4. Target-matched deployment boundary

Positive values favor FMRoPE.

| Length | Seed 42 | Seed 137 | Seed 256 | Mean | SD across seeds | 95% t interval | Anchored EVQ-Cosh wins | `exp(mean delta)-1` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | +0.03276 | +0.02903 | +0.01680 | +0.02619 | 0.00835 | `[+0.00545,+0.04693]` | 0/3 | +2.65% |
| 512 | +0.06109 | +0.05257 | +0.06729 | **+0.06032** | 0.00739 | `[+0.04196,+0.07867]` | 0/3 | +6.22% |
| 1,024 | +0.18182 | +0.15560 | +0.34417 | **+0.22720** | 0.10214 | `[-0.02654,+0.48094]` | 0/3 | +25.51% |
| 2,048 | +0.27857 | +0.39924 | +0.70096 | **+0.45959** | 0.21757 | `[-0.08087,+1.00006]` | 0/3 | +58.34% |

At 1K and 2K the training-seed intervals are wide, but every seed and nearly
every anchor favors target-matched FMRoPE. This is a deployment boundary, not
a failure of the fixed-support identification: fixing support asks whether
`z` matters, while target matching allows support to move with the declared
evaluation length.

## 5. Relationship to the historical aggregate

The previous author-confirmed values were
`-0.3159/-0.1949/-0.1674` at `512/1K/2K`. They lacked local per-seed values,
source hashes, and confidence intervals. The new raw-hash-receipted means are
`-0.28073/-0.17599/-0.14571`, respectively 11.1%, 9.7%, and 13.0% smaller in
absolute magnitude. The direction and `3/3` conclusion are unchanged.

For all future analysis and manuscript work, use this report and its companion
JSON. Retain the older files only as historical records of the unpromoted
author-confirmed summary; do not average, splice, or choose between their
numbers.

## 6. Paper-facing interpretation

### Supported

- At fixed sampled support, interior allocation is a separately identifiable
  training-time variable.
- The fixed-range OOD direction replicates across three training seeds.
- The result is a small in-window cost followed by a consistent OOD gain: an
  effective-context shift rather than uniform dominance.
- Target-aware support control remains the stronger deployment lever in the
  target-matched condition.

### Not supported

- EVQ-Cosh is the unique or universal allocation optimum.
- Allocation is additive or synergistic with target-aware FMRoPE in this
  protocol.
- This 151.9M control alone establishes mature-scale capability.
- Evaluation anchors can be counted as independent seeds.
- Three seeds justify generic statistical-significance language.

### Recommended manuscript sentence

> Across three paired training seeds, moving only the 30 interior frequencies
> changes anchored EVQ-Cosh minus FMRoPE OOD NLL by
> `-0.281/-0.176/-0.146` at `2x/4x/8x`, with all three seed contrasts favoring
> anchored EVQ-Cosh at every OOD length, at an in-domain cost of `+0.026` NLL.

The abstract, introduction, identification figure, experiment paragraph, and
identification appendix are synchronized to this three-seed result. The
target-matched row remains an appendix/discussion boundary rather than an
abstract result.

## 7. Completion and provenance receipt

The multiseed wrapper exited `0`. All four new trainings, both per-seed
evaluations, and both identity controls completed without OOM, NaN, traceback,
or failed assertions. The four new arms consumed `10,941.55` training seconds
(`3.04` GPU-hours of summed arm time) on an RTX 5090. The GPU was idle after
completion.

### Result sources

| Seed | Source role | SHA-256 |
| ---: | --- | --- |
| 42 | historical comparison JSON | `801792f062fe680afe1570fbf23ed710d77be0659c03a3924361b79ccbf334b1` |
| 137 | evaluation results JSON | `6a5ab42b783b2ea98428e379ab49ba7264401c3ec8533a44a0a40c0d65196e48` |
| 256 | evaluation results JSON | `23a0dd0698f2f72089767602df3449975a83278292692c0518d7608a3f6c8947` |

### Shared new-run data receipts

| Object | SHA-256 / revision |
| --- | --- |
| FineWeb-Edu revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| GPT-NeoX tokenizer revision | `c292233c833e336628618a88a648727eb3dff0a7` |
| Tokenized train array | `3dde6bd292685e189fe6d172693ce7cd93963f60eeeff80d48cc5c4371837bea` |
| Consumed train prefix | `66ee82396750d2c2fe9ab0a678092383a46ad28290983d091bd83895d5f83e60` |
| Validation tokens | `be564fa673da684d989752576c2d9cebf7d4ae3235a4751a1a281712623cb3e0` |
| Evaluation anchors | `a6716d6e8c348494c517540c167f6a2b43a16eb4377af2c395d12e34734493b3` |
| Regenerated manifest | `2c4b1c0ec6993a4065a666dd04c26f1c3439e812de25e49cbf5d21b106ab9433` |
| Scientific protocol embedded in manifest | `d92da1b8a4ba92cb9bee5bcd28c2d73282bc0bedfc38adf164dc9693a42d188d` |

### Within-seed paired receipts

| Seed | Initial trainable | Row order | Protocol | FMRoPE frequency | Anchored EVQ-Cosh frequency |
| ---: | --- | --- | --- | --- | --- |
| 137 | `c88c6abc...97dcb` | `92d60450...ef5ff` | `8135dc4d...b2f48` | `06adcd40...d6b04` | `e0b20171...8a0eb` |
| 256 | `214007eb...e78d` | `c19e9371...611b` | `2b776fd1...ee0f` | `06adcd40...d6b04` | `e0b20171...8a0eb` |

For each new seed, initialization, row order, consumed train prefix,
validation data, optimizer steps, token counts, protocol hash, and code hash
match exactly across the two arms. Only the immutable frequency table differs.

## 8. Next manuscript integration gate

1. Use the companion JSON as the numeric owner.
2. Replace the seed-42 headline in the abstract, introduction, main experiment,
   identification figure, and appendix as one atomic manuscript change.
3. Show per-seed points plus the three-seed mean; do not hide the 512-length
   heterogeneity behind a bar alone.
4. Keep the 1K/2K stability and the `3/3` direction visible.
5. Update the appendix batch-geometry description for seed 42 versus the two
   new runtime-equivalent seeds.
6. Rebuild, package, and visually inspect only after the manuscript text and
   figure agree with this owner.
