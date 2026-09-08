# Position after sparse attention: research decision, 2026-09-08

Status: problem selection and experiment preparation. No new method or neural
capability result is established. Latest user instruction overrides the earlier
six-arm-first plan: strongest central experiment first, completeness and broad
comparisons second. Both supplied Pro documents are hypotheses, not constraints.

## An actual contemporary disagreement

Kimi K3 uses KDA with NoPE in every global MLA layer; it attributes implicit
position/recency information to recurrence. Qwen3.8-Next instead retains RoPE:
its report states that the NoPE variant is close in pretraining but has more
endless generation after post-training. These are different model families,
not a controlled comparison and not proof that scalar versus diagonal decay
causes the discrepancy.

Sources read directly:
- Kimi K3, §§2.1.2 and 3.4, https://arxiv.org/html/2607.24653v1
- Qwen3.8-Next, §2.1.1, https://arxiv.org/html/2608.30320v1 (2026-08-31)
- DeepSeek V4, §2.3.3, https://arxiv.org/html/2606.19348v1: compressed attention
  still uses partial RoPE, including an inverse rotation on outputs because K=V.

Thus 1M feasibility has changed the importance of pure extrapolation, but it
has not settled positional design. Sparse attention, recurrent linear attention,
and compressed attention are distinct mechanisms. Closed-model internals do not
justify the universal claim that every model uses the same sparse architecture.

## What is already taken

- ScoPE explicitly uses structured sparsity as an implicit position mechanism:
  https://aclanthology.org/2026.acl-long.1650/ (ACL 2026).
- RNoPE already combines local RoPE with global NoPE:
  https://arxiv.org/html/2501.18795v1 .
- Long-Context Generalization with Sparse Attention studies sparse distributions
  and positional choices including NAPE: https://arxiv.org/abs/2506.16640 .
- LeRoPE and MHA2MLA already address frequency learning/selection and partial
  rotation; a rotary-budget win alone is not a compelling novelty argument.

Do not repackage any of those as a new topology-adaptive universal PE.

## Stronger question to test

When can implicit order information in a sparse/recurrent hybrid replace explicit
position **without sacrificing order-dependent behavior and controlled generation**?
The useful outcome would be a causal, reproducible condition that tells an
architecture designer when NoPE is safe, where an explicit positional channel is
still required, and what failure that channel repairs. A new frequency table is
not required for that outcome; a universal best encoding is not assumed.

Distinguish three roles:
1. Access: is the relevant source available to the query after selection/compression?
2. Identification: can available representations distinguish competing occurrences,
   their order, and the current sequence state?
3. Use: does the full model exploit that distinction in the answer and terminal EOS?

A graph path proves access, not robust identification; distinct real-valued hidden
states do not prove useful margins under finite precision. A good NLL does not
certify order handling or stopping. Conversely, removing RoPE from a checkpoint
trained with it is a distribution shift and cannot establish that a NoPE model
could not learn the same task.

## First experiment selection, before costly training

The first paid experiment must isolate one of these failures, not populate a
method leaderboard. Prefer an available modern hybrid checkpoint at its native
window, with tasks the unchanged checkpoint can actually perform. Freeze a small
set of complete-text counterfactual families before the intervention: same content
retrieval, competing-record order, and a finite requested output with exact EOS.
Use matched distractor/padding controls and preserve all legitimate model paths.

The first result sought is a reproducible separation between source availability
and usable order, with final output consequences. If probing only finds that an
untrained/ablated model is bad, or only that arbitrary position scrambling breaks
co-adaptation, reject that evidence as a basis for the paper. A local injected
payload diagnostic is a mechanism test, not the primary complete-text result.

After a concrete effect exists, intervene on the identified channel and require
recovery of the same failed outputs while retaining content access and current-value
accuracy. Only then spend on learned RoPE/NoPE counterparts, architecture breadth,
and competitive methods. If the effect requires changing a trained positional
scheme globally, a matched adaptation or training control is necessary before
making a general architectural claim; checkpoint surgery alone is insufficient.

No architecture-specific cause of the Kimi/Qwen disagreement is selected yet.
Candidate mechanisms such as recurrent order decay, loss of positional information
under normalization, or sparse selection are distinguishable hypotheses, not facts.
Do not invent a cure before finding which, if any, is present in real trajectories.

## Consequences for the two supplied plans and manuscript

Rotary-budget tables, fixed-pair code, same-target validation and training runtime
are reusable infrastructure. The six-arm experiment is retained, not scheduled;
it becomes a supporting controlled test only if rotary width matters to the
selected central failure. No need to complete it merely because a protocol exists.

KLD's exact derivative interpretation is useful for one potential memory-erasure
failure; it does not itself solve global positional identification or termination.
Its equal-state ordinary-memory control remains mandatory if that route is chosen.
Do not combine the two documents into two parallel main methods.

The compiled `paper-2027/main.pdf` is a provisional budget-focused restructuring,
not a final submission claim. The final paper should open with a reproduced modern
failure and the architectural decision it changes, state a conditional mechanism,
then present a focused repair and decisive evidence. Prior EVQ/MLA results can
support the story only where their actual estimands match. Acceptance confidence
must come from significance of the question, novelty against close prior work,
and causal/full-model evidence—not from an asserted probability or theorem count.
