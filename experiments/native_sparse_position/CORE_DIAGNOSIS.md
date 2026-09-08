# Current evidence and next discriminating test

These are development experiments, not a completed method or a benchmark claim.
The September 2026 research question remains open.

## Phase summaries: negative first test

The supplied PSR design was implemented as farthest-first clustering of native
RoPE phases, B=64 and R=4. Count-matched contiguous and random partitions control
for the number and sizes of summaries. Two fixed long prefixes, three sampled
layers per model, eight terminal queries, and all query heads were inspected.

| Native checkpoint | Selector | Remote probability retained | Relative attention-value error |
|---|---|---:|---:|
| Qwen3.5-0.8B | RoPE mean | .783644 | .018219 |
| Qwen3.5-0.8B | NoPE mean | .774587 | .019021 |
| Qwen3.5-0.8B | PSR | .791685 | .017753 |
| Qwen3.5-0.8B | Contiguous R=4 | .792752 | .017686 |
| Qwen3.5-0.8B | Random R=4 | .786830 | .018039 |
| Qwen2.5-3B | RoPE mean | .606341 | .120433 |
| Qwen2.5-3B | NoPE mean | .292119 | .150254 |
| Qwen2.5-3B | PSR | .609223 | .119691 |
| Qwen2.5-3B | Contiguous R=4 | .612167 | .120067 |
| Qwen2.5-3B | Random R=4 | .609048 | .119043 |

Means give equal weight to the six sampled prefix/layer cells within each model.
The sample is too small for population claims. PSR does not establish an advantage
over its matched controls; increasing the number of summaries explains the
observed mean-summary improvement at least as well. No selector is deployed here.
Raw pre-RoPE capture and native rotary reconstruction were bitwise verified.

## Exact ranking: local fidelity is not yet an answer bottleneck

To remove summary error, the next experiment ranks remote B=64 blocks by exact
logsumexp over all keys, either with native RoPE Q/K or their pre-rotation Q/K.
Both read the original native RoPE K/V, keeping 32 remote blocks per query head,
2048 recent tokens and a 64-token sink. This scans all keys and is an oracle
diagnostic, with **no efficiency claim**.

The document prefix is independent of the question. Every method receives its own
copy of that cache; sparse attention starts at the first question token in all 36
Qwen2 attention layers and continues through the answer. Qualification with every
key visible gives bitwise equality to dense logits for both gather paths.

On 16 previously inspected 16K event questions, all 48 completions terminated.
RoPE and NoPE oracle outputs are identical at the raw-token level on 15/16 pairs.
The remaining pair changes a wrong violet assignment to the correct orange one.
Manual whole-answer content correctness is Dense 13/16, RoPE 13/16, NoPE 14/16;
these are annotations, **not exact-string scores**. Strict full answer plus EOS
is 0/16 for every method because outputs violate the required concise format.
Case-insensitive whole-string plus EOS is respectively 2/16, 3/16, 3/16.

This fails to establish that the large Qwen2 local fidelity gap is binding for
these generated answers. It also does not establish equivalence, positional
redundancy, or a NoPE method improvement. Prior causal prefix states and the reader
still carry positional information. “NoPE” here means withdrawing only the
selector's explicit final rotary transform.

Evidence lives under `results/position_observability_20260908/`: the two
`psr_*activation_01/result.json` files and `oracle_continuation_01/`, including
unmodified outputs, source snapshot, status, summary and manual annotations.

## Next: naturally distributed evidence and a recurrent/global hybrid

Use 24 untruncated natural QA examples: eight each from HotpotQA, Qasper and
MultiFieldQA-en, selected by source-row SHA order. Inputs must fit 8192..16256
tokens in both native tokenizers; answers never participate in selection. The
custom document-before-question template enables the same question-blind cache
intervention. This is a development subset, not an official full LongBench run.

Run the native Qwen3.5-0.8B hybrid and the existing Qwen2.5-3B with the identical
three-arm contract. In Qwen3.5 only its six full-attention layers change; all 18
Gated DeltaNet blocks, Q/K normalization and output gates remain native. Preserve
the full generated token stream, EOS, exact answer, normalized whole-answer exact,
and F1 of the **entire** completion with a fixed 128-token cap. No substring,
first-line, or first-number answer extraction is permitted.

The test asks whether a downstream effect emerges outside the artificial record
task, and whether its relation to fidelity differs in a recurrent/global model.
Weak dense task performance limits interpretation; a small null or negative
result changes the research direction rather than triggering a blind budget sweep.

## Completed natural oracle results and truncation resolution

All 24 questions completed in each method on both models. At the original
128-token cap Qwen3.5 Dense/R/N terminated 24/23/23 times, and Qwen2.5-3B
21/23/22. Only these eight capped trajectories were replayed, checking exact
identity of all original 128 generated tokens, and extended with a 384-token cap.
All eight then terminated. Earlier terminated completions were reused unchanged;
`natural_*_extension_01` is the complete 384-cap view, with provenance to its parent.

| Model | Dense whole-answer F1 | RoPE oracle | NoPE oracle | R/N identical token sequences |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | .376799 | .343573 | .368684 | 15/24 |
| Qwen2.5-3B | .234505 | .254752 | .248081 | 14/24 |

Every cell has 24/24 EOS at the extended cap. Trimmed exact plus EOS is 3/2/3
for Qwen3.5 and 1/1/1 for Qwen2.5-3B. These small differences do not establish a
consistent advantage. In particular the largest Qwen3.5 R/N difference is
`qasper_196`: both give the correct 0.6103, but R includes a surrounding sentence.
Length-sensitive F1 cannot be interpreted as a retrieval gain in that pair.

The saved-key phase/content decomposition also completed without new forwards.
For a block mean pre-rotation key, rotate that constant key at the original
positions to define a phase-only component. The difference to the saved native
key is the content residual (including BF16 rounding). Score variances decompose
exactly into phase variance, residual variance and twice their covariance.
Equal-file mean values, uniformly over eligible query/block pairs:

| Model / partition | Native variance | Phase | Residual | Twice covariance |
|---|---:|---:|---:|---:|
| Qwen3.5 / whole block | 2.952803 | .023275 | 2.928698 | .000830 |
| Qwen3.5 / PSR | 2.457900 | .013475 | 2.443798 | .000626 |
| Qwen3.5 / contiguous | 2.363582 | .011247 | 2.352000 | .000335 |
| Qwen2.5 / whole block | 4.899932 | 1.937103 | 2.963351 | -.000522 |
| Qwen2.5 / PSR | 3.886419 | 1.352520 | 2.533197 | .000702 |
| Qwen2.5 / contiguous | 3.886309 | 1.350432 | 2.537490 | -.001613 |

The phase term is tiny on the sampled Qwen3.5 queries. On Qwen2.5 it is substantial,
but PSR does not reduce it below the contiguous control. This is a diagnostic
explanation to investigate, not proof of what dominates all model attention.

The next two mean-summary arms reuse the same natural inputs, reader and physical
budget. They complete the distinction between withdrawal of explicit rotation and
summary error; existing dense/oracle outputs are reused. No frequency/budget sweep
is scheduled. The native sparse checkpoint MiniCPM4.1 is being prepared separately
because the deployed shared-GQA selector has a normalization mechanism absent from
these per-query-head oracles.

## Native shared-head selection: a different causal channel

The actual InfLLM-V2 selector sums scores across the query heads in one KV group,
then pools and selects shared blocks. Its implementation estimates the softmax
normalizer using a coarser 128-token/64-stride set of key means, while numerator
scores use 32-token/16-stride means. This is absent from our per-query-head oracle.
See the official paper, Algorithm 1 and Section 3.4:
https://arxiv.org/html/2509.24663v1 .

For fixed fine-window scores `s[h,c]`, define exact fine partition `Z[h]` and
coarse estimate `Zhat[h]`. Then the implemented shared score is exactly

`sum_h exp(s[h,c])/Zhat[h] = sum_h (Z[h]/Zhat[h]) * softmax(s[h,:])[c]`.

A common multiplicity factor cancels when ranking shared blocks. A head-dependent
factor does not: it changes head weights. Within-head top-k is invariant to it,
which explains why per-head diagnostics cannot identify this mechanism. Changing
RoPE treatment changes both fine scores and these weights. This identity is
algebra, not a novel theorem by itself or evidence that exact normalization is
better for a checkpoint trained with an approximation.

The discriminating native test is to retain the same fine scores and reader while
replacing only the coarse normalizer with the exact fine normalizer, comparing
actual shared selections and generated answers. First qualify the original native
kernel and baseline, then decide whether measured head-weight distortion warrants
that intervention. Do not mistake a PyTorch reconstruction for the official fused
kernel without checking its causal/masking/precision contract.

A preliminary FP64 probe on the already saved Qwen2.5 activations uses fully causal
fine/coarse means. With 8 query heads per KV group, effective head count is 7.049
(RoPE) versus 7.283 (NoPE); mean maximum weight .215 versus .186 (uniform .125).
The induced shared fine-window distribution changes by TV .060 versus .053, while
its top-32 window overlap with exact normalization remains .956 versus .977.
This is a small frozen-trajectory diagnostic; native-kernel parity and final task
consequences remain untested. It is not sufficient to justify a performance claim.
