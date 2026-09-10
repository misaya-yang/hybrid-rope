# Position in modern sparse attention: claim discussion

2026-09-09. The user clarified that the intended research direction is better
positional computation for heterogeneous modern sparse-attention architectures.
Continually accumulating Agent context is a potentially important setting, not
authorization to replace that objective with a dense-model extrapolation patch.
GPU remains off by the user's current instruction; theory, design and code
preparation are authorized. No method or acceptance probability is established.

## Candidate central question

Within an already supported context window and at a fixed sparse-memory/read
budget, can positional computation better preserve the distinction between
repeated occurrences, their temporal relations, and the information a later
query needs? A transferable design principle should improve actual answers
across distinct sparse interfaces; one universal frequency table is not assumed.

Sparse selection, pooled/compressed entries and recurrent state are distinct.
The RoPE relative-rotation identity remains valid for retained token pairs.
The open issue is whether the entire information path preserves the distinctions
needed by the task, not whether sparsity algebraically breaks that identity.

## Fresh sources and limits

- Qwen3.8-Next §2.1.2 explicitly pools content before partial RoPE, assigning
  block-start positions. Post-RoPE mean cancellation therefore cannot be its
  assumed defect. Table 3 reports 1M QSA RULER 93.00 versus 8-needle MRCR 26.44;
  these are different benchmarks, showing uneven capability, not positional
  causation. https://arxiv.org/html/2608.30320v1
- The same report §2.1.1 retains RoPE after a NoPE comparison showed more
  nontermination after post-training. This is evidence in that architecture,
  not a universal need for RoPE.
- GLM-5.3-Flash differs from GLM-5.3: the Flash config uses KDA/sparse layers,
  index_kpool=4 and qk_rope_head_dim=0. Do not generalize the zero rotary width
  to every indexer or to the whole GLM family.
  https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json
- YaRN §3.3 already discusses dynamic scaling and caching before rotation;
  Jet-Long already combines dynamic scaling and local protection. Cache reuse
  and unknown-length support alone are not a new main contribution.
  https://arxiv.org/html/2309.00071v2
  https://arxiv.org/html/2607.07740v1

## Recovered proposal and what was actually tested

The original Pro plan is available at
`/Users/yang/Downloads/native_sparse_position_research_plan_20260908.md`.
Its PSR proposal targeted position-compatible router summaries. The local
`experiments/native_sparse_position/CORE_DIAGNOSIS.md` records small activation
comparisons where PSR did not separate from matched contiguous summaries, and
16 generation cases where selector RoPE versus NoPE outputs mostly agreed.
Those observations neither prove the intended benefit nor reject the broad
research question. A separate compressed-memory scratch assay failed its
query-dependent capability check; it did not test a successful modern model's
positional bottleneck. RefCarry has substantial operator overlap with TAPE,
documented separately. Do not restart these configurations merely because this
discussion returns to their parent question.

## Relationship to growing context

Two independent axes must be measured: the current prefix length and the way
the prefix was processed. Multiple independent full-prefill lengths do not
prove good incremental execution; chunk-invariant execution does not prove
good answers at those lengths. A sparse-position method may be tested on both,
but the dynamic cache schedule prototype is only a separate diagnostic candidate.

For paired mechanism tests, append a fixed external document/event stream and
ask questions on temporary branches. Generated answers do not enter the shared
future stream. This keeps inputs identical across interventions. Full Agent
rollouts are a subsequent end-to-end test with potentially divergent histories.

The decisive controlled task changes the requested occurrence/order relation
while holding content inventory and difficulty controls fixed. A method must
improve real answers over matched original positional/sparse computation, with
source-availability and ordinary-content controls. A position-only mechanism
claim needs an intervention that separates it from simply selecting more
evidence, adding model capacity or adapting the backbone. The precise causal
intervention must follow the actual architecture, not an assumed universal
compression collision. Practical success requires independent natural tasks,
cross-interface transfer and all memory/latency overheads included.
