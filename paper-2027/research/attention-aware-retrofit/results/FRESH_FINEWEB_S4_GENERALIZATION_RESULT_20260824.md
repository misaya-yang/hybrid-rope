# Fresh FineWeb-Edu zero-training session-s4 generalization

- **Date:** 2026-08-24/25
- **Status:** completed fresh-distribution natural-NLL confirmation
- **Decision:** retain the zero-training Native/s4 session policy; treat the
  third axis as a necessary coarse allocation degree of freedom at 4x, without
  claiming that one detailed interior profile is uniquely responsible
- **Data:** FineWeb-Edu `sample/10BT/002_00000.parquet`, absent from the prior
  local `000/001/004` shard set

## Decision first

The completed Native/s4 policy generalizes strongly to an independently
downloaded FineWeb-Edu shard. It preserves the exact Native 4K route, improves
every one of 128 paired documents at both 8K and 16K, lies only `0.0913` NLL
behind the target-aware s2 oracle at 8K, and is the oracle at 16K by
construction.

This result also separates the roles of the method components on fresh natural
text. A same-support geometric long table collapses at 16K, while a coarse
fixed-index ramp control matches the derived allocation to within `0.001` NLL.
Applying the same derived s4 table statically at 4K incurs a visible cost; the
session route removes that cost exactly. The practical method is therefore:

> exact Native inside the model window, plus one zero-parameter, session-static
> long profile with a non-geometric interior allocation.

## Data and protocol

The source is the official FineWeb-Edu 10BT shard `002_00000.parquet`, LFS
SHA-256 `547ae182d132c9f06b6ce63149567208ea9f57630bfd9b1a2938e504f0c9ebd7`.
The previous local data owner used shards `000`, `001`, and `004`; the four
historical calibration-document text hashes are additionally excluded.

Each selected document supplies the same first 4K/8K/16K tokens to every arm.
The endpoint is teacher-forced NLL over the final 1,024 tokens. Methods use the
released OLMo-2-0425-1B-Instruct checkpoint, BF16, Flash-only attention, and no
weight updates:

1. Native at every length;
2. zero-training session-s4: Native at 4K, frozen s4 at 8K/16K;
3. target-aware Native/s2/s4 oracle.

Official Transformers YaRN factor four was also run as an external reference.
It is not a variable-isolated comparator and owns no mechanism conclusion in
this report.

The first 32 eligible 16K documents were exploratory. After observing that
matrix, the next 128 eligible documents were frozen as a disjoint confirmation
set. A further 512-document set skipping the first 160 eligible documents was
prepared before reading the 128-document metrics and used for a larger
Native/session-only confirmation.

## Fresh-32 exploratory matrix

| Method | 4K NLL | 8K NLL | 16K NLL |
| --- | ---: | ---: | ---: |
| Native | `2.6110` | `6.8906` | `7.1947` |
| target-aware oracle | `2.6110` | **`2.7228`** | `2.6713` |
| **session-s4** | **`2.6110`** | `2.8071` | **`2.6713`** |

## Disjoint holdout-128 confirmation

| Method | 4K NLL | 8K NLL | 16K NLL |
| --- | ---: | ---: | ---: |
| Native | `2.7538` | `7.0023` | `7.2703` |
| target-aware oracle | `2.7538` | **`2.7749`** | `2.7868` |
| **session-s4** | **`2.7538`** | `2.8662` | **`2.7868`** |

Paired document contrasts for session-s4 are:

| Contrast | 4K | 8K | 16K |
| --- | ---: | ---: | ---: |
| session-s4 minus Native | `0.0000` | `-4.1361` | `-4.4835` |
| 95% paired-row bootstrap | `[0,0]` | `[-4.2539,-4.0219]` | `[-4.6140,-4.3572]` |
| rows favouring session-s4 | exact `128/128` | `128/128` | `128/128` |
| session-s4 minus oracle | `0.0000` | `+0.0913` | `0.0000` |

The bootstrap resamples paired source documents 20,000 times with seed
`20260824`; it conditions on this checkpoint and deterministic shard subset.

## Allocation and routing controls

This is the third-axis identification layer. In the paper's exact table
decomposition

\[
x_k=-\log\omega_k=a+Rz_k,\qquad z_0=0,\ z_{K-1}=1,
\]

the controls hold sampled support `(a,R)`, attention gain, Native/long route,
checkpoint, rows, and runtime fixed. Only interior allocation `z` changes.

| Long allocation | 4K NLL | 8K NLL | 16K NLL |
| --- | ---: | ---: | ---: |
| geometric | `2.7538` | **`2.8316`** | `7.3189` |
| derived s4 | `2.7538` | `2.8662` | **`2.7868`** |
| coarse fixed-index ramp | `2.7538` | `2.8663` | `2.7876` |

Geometric allocation is locally competitive at 8K but catastrophically loses
the 16K endpoint. The ramp-minus-derived differences are only `+0.00005` at 8K
and `+0.00072` at 16K, extending the prior RULER profile-detail negative to
fresh natural NLL.

The disjoint holdout-512 controls confirm both statements more sharply. At 8K,
geometric-minus-derived is `-0.02882`, with paired interval
`[-0.03217,-0.02550]`: geometric is slightly better locally. At 16K the sign
reverses catastrophically to `+4.4560`, interval `[+4.3804,+4.5317]`, and
geometric loses on all `512/512` rows. Allocation is therefore not a generic
short-range quality bonus; it determines whether the fixed finite support
survives farther extrapolation.

On the same holdout-512 rows, ramp-minus-derived is `+0.00051` at 8K and
`+0.00056` at 16K. Both paired intervals contain zero
(`[-0.00028,+0.00131]` and `[-0.00020,+0.00133]`). The coarse ramp control
matches the detailed profile on fresh natural text as well as in the earlier
RULER owner. This is a profile-detail negative, not a reduction of our method
to YaRN and not evidence that `z` is irrelevant: geometric versus
non-geometric at 16K is the isolated third-axis effect, whereas ramp versus
derived tests only fine-shape uniqueness.

Applying derived s4 statically at every length gives 4K NLL `2.8774`, a
`+0.1236` regression versus the exact-Native session route, while its 8K/16K
values are identical. This is a separate routing estimand: the long table is
held fixed and only the short-request dispatch changes. It directly identifies
routing as the in-window retention mechanism rather than attributing retention
to the long table or to `z`.

The target-aware s2 versus session-s4 contrast at 8K changes the installed long
table and support together, so it is a practical operating-point comparison,
not a pure allocation or routing effect.

## Holdout-512 confirmation

The larger disjoint Native/session-only matrix was frozen before the
holdout-128 metrics were read:

| Method | 4K NLL | 8K NLL | 16K NLL |
| --- | ---: | ---: | ---: |
| Native | `2.6886` | `6.9099` | `7.1629` |
| **session-s4** | **`2.6886`** | **`2.8145`** | **`2.7237`** |

Session-s4 is exactly Native on all `512/512` 4K rows. It improves all
`512/512` paired documents at 8K and 16K. Mean deltas and paired-document
bootstrap intervals are `-4.0953` `[-4.1494,-4.0421]` at 8K and `-4.4393`
`[-4.5085,-4.3713]` at 16K. Even the least favorable paired deltas remain
negative (`-2.3006/-2.0216`). This larger confirmation makes the practical
policy persistence result insensitive to the initial 32-document discovery
set and the 128-document mechanism matrix.

## Claim boundary

This is natural-text teacher-forced tail NLL on deterministic subsets of one
new FineWeb-Edu shard. It strongly tests row-level and source-shard
generalization for one checkpoint, but it is not autoregressive capability,
checkpoint-population uncertainty, or proof of universal task dominance. The
existing Qasper, 2Wiki, PG-19, and RULER owners retain their endpoint roles.
