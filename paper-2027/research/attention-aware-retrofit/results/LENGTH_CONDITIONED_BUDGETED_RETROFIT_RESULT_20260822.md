# Length-conditioned uniqueness-budgeted retrofit

- **Date:** 2026-08-22
- **Status:** core-4 RULER and runtime parity complete; held-out 2Wiki generation
  complete with one physical-budget invariant pending recheck; RULER-13 data
  preparation interrupted by platform shutdown before evaluation
- **Evidence role:** prospective mature-checkpoint zero-training retrofit result;
  not yet manuscript evidence
- **Checkpoint:** released OLMo-2-0425-1B-Instruct, weight SHA-256
  `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f`
- **Learned parameters / training tokens / search on 2Wiki:** `0 / 0 / none`

## Decision

Freeze one practical method for held-out evaluation:

1. Requests of at most 4096 tokens call the Native rotary module directly,
   with Native attention scaling one.
2. A request with context factor `s > 1` uses the frozen
   uniqueness-budgeted table

   ```text
   move_k = (1 - normalized_uniqueness_k)^2
   omega'_k = omega_k * (1 - move_k) + (omega_k / s) * move_k
   ```

   and the matched length-only attention amplitude
   `a(s) = 1 + 0.1 log(s)`.
3. The request branch is fixed from the total context budget before prefill and
   remains fixed for cached decoding.

The amplitude is the same deterministic factor used by the official reference;
it is not a learned component or a new contribution. Using it on both operators
makes the frequency-allocation comparison matched.

The frozen frequency identities are:

| target | table SHA-256 (float32 tensor) | amplitude |
| --- | --- | ---: |
| 8K (`s=2`) | `f94a34381cfb3d05621db41bd3778779b16812246c392bd645013a1a06f80814` | 1.0693147181 |
| 16K (`s=4`) | `a435d75441444bcea39b73d9cf530005249dc5afdc3cfb5a60fda10ef33312d3` | 1.1386294361 |

`p=2` is frozen for both lengths. At 16K, `p=1` scored 0.4025 versus
0.4000 for `p=2`; the 0.0025 difference does not justify a length-specific
shape, while `p=2` was stronger at 8K (0.5825 versus 0.5525).

## Core-4 RULER result

All rows use the same released checkpoint, frozen data manifest
`6f64aa6bb44821f52015c868b1f13d07a17a5f710b7645a9e0fb3fb86ac09d37`,
20 rows per task, greedy decoding, and official task-specific scoring.

| operator | 8K | 16K |
| --- | ---: | ---: |
| Native | 0.0000 | 0.0000 |
| Native + matched amplitude | 0.0000 | 0.0000 |
| budgeted frequency only | 0.4000 | 0.1150 |
| official Transformers YaRN | 0.5375 | 0.0125 |
| **budgeted frequency + matched amplitude** | **0.5825** | **0.4000** |

Per-task rows for the frozen method:

| length | single-key | multikey-2 | multikey-3 | variable tracking |
| --- | ---: | ---: | ---: | ---: |
| 8K | 1.00 | 0.80 | 0.50 | 0.03 |
| 16K | 1.00 | 0.55 | 0.00 | 0.05 |

The two missing-factor controls establish positive non-additive interaction on
this assay. The interaction
`joint - frequency_only - scaling_only + native` is `+0.1825` at 8K and
`+0.2850` at 16K. Scaling does not lift Native from the score floor at either
length, while the budgeted table without scaling is much weaker. This is
consistent with a two-part mechanism in which non-geometric allocation changes
long-range ordering and length-only amplitude counters attention dilution after
that ordering becomes usable. It does not show that scaling leaves Native
logits unchanged or that the interaction follows a monotone law with length.

The frozen coefficient `0.10` was not selected on these tasks; it is the
matched reference value. A post-freeze sensitivity check changes only that
coefficient:

| coefficient in `1 + c log(s)` | 8K | 16K |
| ---: | ---: | ---: |
| 0.08 | 0.5375 | 0.3150 |
| **0.10 (frozen)** | **0.5825** | **0.4000** |
| 0.12 | 0.5850 | 0.4275 |

Both tested neighbouring coefficients retain the reported positive scores.
Three discrete points do not establish a continuous basin or an optimum; the
higher 16K score at 0.12 is sensitivity only and does not alter the frozen
method.

This does **not** establish a universal operator or a pooled task effect. It is
one deterministic checkpoint on a four-task, 20-row-per-cell subset.

## Held-out natural downstream result

The method was frozen before reading LongBench 2Wiki scores. All 200 official
rows were evaluated with normalized token F1; normalized exact is auxiliary.

| operator / budget | token F1 | normalized exact | truncated rows |
| --- | ---: | ---: | ---: |
| Native 4K | 0.2656 | 0.205 | 175 |
| official Transformers YaRN 8K | 0.2774 | 0.210 | 54 |
| **frozen method 8K** | **0.2679** | **0.205** | 54 |
| official Transformers YaRN 16K | 0.2585 | 0.205 | 1 |
| **frozen method 16K** | **0.2639** | **0.210** | 1 |

The frozen method is within `+0.0023/-0.0017` token F1 of the Native 4K score at
8K/16K, under materially different truncation rates. It is slightly below the
official reference at 8K and slightly above it at 16K; these differences are
descriptive, not significance claims. No natural multi-hop QA collapse is
observed.

A post-run code audit found that the untruncated fast path checked raw prompt
tokens before adding chat-template overhead. Before manuscript promotion, the
raw rows must be rechecked for `input_tokens + 32 <= nominal_length`; any
violating cell must be rerun with the corrected check. This limitation does not
affect the RULER or runtime-parity results.

## Practical preservation receipt

The length-conditioned runtime has zero learned parameters. On a physical
4096-token natural row:

- Native versus short branch final hidden state: bitwise equal, max delta 0;
- Native versus short branch logits: bitwise equal, max delta 0;
- parameter count before versus after installation: identical.

For both the 8K and 16K configurations, the forced long branch is bitwise equal
to directly installing the corresponding frozen table and amplitude on a
512-token parity probe. Unit tests separately cover short, long, mixed-batch,
cache-boundary, zero-parameter, amplitude-formula, and exact table-rebuild
contracts (`7/7`). `scripts/analysis/export_uniqueness_budgeted_tables.py`
rebuilds both
frozen tables from the Native endpoint grid and the registered 4K causal
measure; its emitted `.npy` file hashes are
`2087ee670210d0fa38e8f0cf9ecd37d4c7a9c6748a54276e5e3bc5e9f3436ab2`
(8K) and
`f812acc0a815db2eea2d2ce2ba21ba9be55f62e17baa961b37179d41a405b1e0`
(16K), matching the evaluated assets.

## What was falsified and what changed

The earlier CPU axes (`D*`, trajectory coverage, and unseen-phase risk) do not
rank downstream operators. The one-turn floor is the decisive counterexample:
near-minimal `D*=0.0192`, zero phase risk, and exactly 0 RULER. Those axes remain
diagnostics or bounds, not method selectors.

The positive result did not come from another table-space search. It came from
restoring the second half of the attention problem that the table-only analysis
omitted. This supersedes the judgement that mature retrofit must next be solved
by a trained adapter. The 50-step generic headwise Q/K LoRA preserved 31 of the
32 zero-training successes (0.3875 versus 0.4000) but supplied no evidence that
training was necessary. The frozen zero-training method is therefore the
current practical candidate.

## Interrupted breadth extension

The frozen RULER-13 extension did not produce an experimental result. It was
registered as three arms at each length: Native, the official Transformers
YaRN operator, and the frozen `p=2`, `c=0.10` method. The nine tasks outside
core-4 were locked as confirmation-only before generation; they were not
allowed to change the method.

The external RULER word asset was recovered from its Git LFS object and
verified at SHA-256
`affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca`.
At the last live receipt, the fresh 8K/16K build had generated at least `18/26`
cells. The platform then shut the instance down before the final manifest was
written. No full-13 GPU arm had started. The partial data and prior completed
raw receipts remain on the persistent volume, but their post-shutdown state is
unverified.

On restart, first inspect the existing output. If it lacks a complete verified
26-cell manifest, generate a fresh versioned output instead of treating the
partial directory as complete. Compare the four core-4 cell hashes with the
completed selection manifest: reuse old core-4 predictions only if every cell
hash is identical. Report the nine-task confirmation macro separately from the
complete 13-task macro, at 8K and 16K separately, regardless of outcome.

The compact machine-path-free receipt is
`../evidence/LENGTH_CONDITIONED_BUDGETED_RESULTS_20260822.json`, SHA-256
`f92332a4d9c0d6c92e2bb2296c876a32001aee88b7057ff422db9c2b9fb1a311`.

## Raw receipt hashes

Core-4 result JSON SHA-256:

- Native 8K: `f1ee3b62456715a0bbd2af5799f99c39323206051adb643364b93a13205cb77a`
- Native 16K: `8e074f0a94c01fcf39eee8b2f877fd7848416a784be9cdda8b4f534b67fa1edd`
- Native + amplitude 8K: `abae9c40f32b8f5e873088bb87375edb070066c3054e4c4f1a37411c74183249`
- Native + amplitude 16K: `60655f4d858d75009a13083abc90b8d1ee4ffbd5280620e85d3b019ac9736b8e`
- budgeted frequency 8K: `9055e7d08ffd106d4ac36fe3bd83fb08241f1e77b13a0efd5100f22de30fcfba`
- budgeted frequency 16K: `7da663bffff33eb2ba5176109bedc512ac42b09f63a996fc2a3d77bc55b020fe`
- official reference 8K: `974dc70c47b4ffa8df617b31505d5775a9a5dc85a886767907df846b51296a88`
- official reference 16K: `44b723012ca1673c24b33bd22c8cfc9c1e639755e8e7cf10bff4c3585d5d658d`
- frozen method 8K: `df634b0dee59dc0bc971eb236d2751a095b1fbfeea96ab6cadd3c257e16c73e4`
- frozen method 16K: `eeeb4722e960850ee4f9e0187097736bfdaa205634682be84cf3b5b0e6b0210c`

2Wiki result JSON SHA-256:

- Native 4K: `473072306044b3fe9a069a5671456168ff076e6e6dfd061848142a30a70f9ea2`
- official reference 8K: `b30513d3ff2a276f908fdbbaf908e1d36dcac58cdadf1219eb83dd2a5ca9ebdd`
- official reference 16K: `9f97a3aa581994188a0527e237380591251d85715ee8d4b6d46c04206a73bfc2`
- frozen method 8K: `43133cabeb7849e69019b8b9c711bc397a633b6e5acdebf37ec4cfa0563d0b69`
- frozen method 16K: `8f717f1c5aa8294fbbd6a1989ff31f5c761c5b812965698dba3e7b2df5e0068b`

The owner intentionally records hashes rather than private machine paths. Raw
examples and runtime receipts remain outside the repository.
