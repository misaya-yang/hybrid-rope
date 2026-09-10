# Dynamic position rules and cached state: current research decision

2026-09-09. Research resumed under the user's overnight authorization. This is
an operator result and experiment decision, not a validated method or a paper
acceptance assessment. No Cosh construction is used.

## Evidence that changes the next decision

The prior BM cross-cache experiment already isolated more than the recent
conversation acknowledged. On two retrospective Qwen3B cases, success followed
the prefix-formation table after all cached keys were rebuilt from pre-RoPE K
under the reader's table. Switching the reader did not repair the failing
prefix. This rules out that specific decode-only rescue, not all repairs.
See [original results](ROPE_BM_CROSS_CACHE_20260908.md).

Local protection and dynamic scaling are not new proposals. Jet-Long already
combines these, stores base-rotated keys, and applies remote correction rotations.
Its current official Qwen3 implementation chooses one group size from the maximum
position in the current call, then applies it to all query rows in that call.
This introduces a separate question: are prefix hidden states independent of
the prefill partition after phase alignment has already been made exact?

Source: [Jet-Long paper](https://arxiv.org/html/2607.07740v1), Sections 3.1–3.3;
[official implementation](https://github.com/jet-ai-projects/jet-long/blob/e5d4dce8fc9b9ffd31d975940f26fb9dd92528b6/jetlm/modeling/jetlong/modeling_qwen3_jetlong.py),
lines 145–158 and 380–403. The fused implementation uses the same call-wide
schedule at lines 145–155. Code was cloned read-only into a temporary directory;
no upstream changes or messages were made.

## Three different meanings of consistency

1. **Phase consistency:** for fixed raw Q/K, every key is rotated according to
   the same intended query-time rule. This is the familiar Dynamic NTK problem.
2. **State consistency:** raw K and V agree with those produced by a reference
   computation. Correcting a rotation does not generally restore these states.
3. **Partition consistency:** the same token prefix produces the same logits
   regardless of whether it arrived in one call, several chunks, or single tokens.

The [2023 Transformers issue](https://github.com/huggingface/transformers/issues/25104)
already discusses cached rotation inconsistency and correcting raw keys. Its
author did not report a convincing downstream gain. Therefore phase correction
alone cannot be claimed as new, nor does mathematical consistency imply better
task performance. The current question explicitly removes that old error first.

## Exact counterexample, with no pretrained-model claim

Let the native window be W, and let a call end at position E−1. In a call-wide
grouped rule, G=ceil(E/W). For a remote pair (t,s), the phase uses
floor(t/G)−floor(s/G). For local pairs, it uses t−s. All keys in the check are
cached **before** rotation and rotated afresh for each read.

If a prefix was formed at G=1 and later a call uses G=3, correcting every cached
K to G=3 reconstructs the score for the OLD raw keys. At the next layer, those
raw keys and values were projected from different first-layer hidden states.
No identity of rotation matrices removes this difference in general.

[Executable counterexample](../../experiments/rope_operator_family/cache_schedule_counterexample.py)
uses a fixed random two-layer, four-dimensional causal attention stack with
residuals and RMS normalization, W=4 and 12 input vectors. It is deliberately
untrained. Run from the repository root:

```bash
python3 -m experiments.rope_operator_family.cache_schedule_counterexample
```

Calling the file directly from its existing directory is unsupported because
that directory contains `operator.py`, which shadows Python's standard library.

[Saved output](../../experiments/rope_operator_family/cache_schedule_counterexample_result.json):

| Comparison | First-layer last hidden difference | Second-layer last hidden difference |
|---|---:|---:|
| Call-wide G: full vs chunks ending 4/8/12 | 0 | 0.00063994 |
| Call-wide G: full vs chunks ending 3/7/9/12 | 0 | 0.00091070 |
| Query-position G: all four partitions | 0 | 0 |
| Fixed final G: all four partitions | 0 | 0 |

First-layer raw V is identical across schedules. Second-layer cached V differs
by up to 0.028493 in the 4/8/12 comparison. Appending a suffix changes old hidden
states by up to 0.032716 under call-wide G; the query-position rule leaves them
unchanged. These magnitudes have no language-model accuracy interpretation.

## A constructively defined reference, not a performance guarantee

Set G_t=max(1,ceil((t+1)/W)) using the query's logical position, independent of
call size. For every remote key s, use floor(s/G_t) for that query. Retain the
same local window as the original bifocal rule. This is the canonical execution
of the phase-corrected rule when tokens arrive individually; it does not choose
a new frequency profile or tune on answers.

**Partition-invariance proof.** At layer zero, token embeddings and logical
positions agree across partitions. If the input hidden states of layer l agree
for all s≤t, projected raw Q/K/V agree. The row's G_t, local/remote membership,
causal visible set, and softmax therefore agree. Its attention residual and any
tokenwise MLP/normalization agree. Induct over layers and positions. This assumes
deterministic evaluation, no dropout, contiguous logical positions and the same
mask; numerical kernel differences are a separate finite-precision issue.

Two corollaries: prefixes within W reproduce native behavior in real arithmetic;
extending a prefix does not revise its cached states. This does NOT reproduce
the full-call horizon-conditioned model at arbitrary longer L. It deliberately
defines a different, prefix-consistent model. Static scaling is another valid
partition-consistent control, but depends on a preselected horizon.

Efficient implementation need not run attention token by token: group query rows
by G_t. Within each group, ordinary rectangular causal attention has the correct
bottom-right alignment. Local and remote regions use the same unified softmax.
The groups partition query rows, so the attention pair count stays quadratic,
not quadratic multiplied by the number of groups. Additional rotations/kernel
launches still require measurement; there is no speedup claim.

## Decision experiment

Use an already available pretrained Qwen checkpoint and fixed tokenized inputs.
First measure full-call versus cached/chunked execution for the same corrected
dynamic rule, with static and native controls. Preserve full outputs. Avoid
using a kernel mismatch, padding error, or BF16 correction drift as evidence.
Then compare the query-position reference on the same inputs and execution
partitions. The decisive unknown is whether the state discrepancy changes
meaningful task outcomes and whether canonical execution preserves useful quality.

The toy proof justifies this specific check, not a new benchmark campaign.
If only tiny logit differences appear and task behavior is unaffected, do not
promote this into a paper. If partition effects are material, assess independent
inputs and actual incremental workloads, and compare existing consistent/static
alternatives before claiming a useful repair. No claim that this explains BM's
static-table failures: that experiment merely motivated separating raw state
from phase alignment.

## Current execution state

The user initially authorized GPU experiments and on/off control. The supplied
current SSH endpoint connected successfully. PyTorch 2.8.0+cu128 and Transformers
4.57.6 were inspected; one direct bundled Flash Attention call succeeded with
GQA, rectangular causal attention and a local window. This was an API smoke test,
not a numerical-equivalence or model-quality result.

The user then deliberately converted the instance to CPU-only and instructed
that work before sleep be restricted to theory, experiment design and code
preparation. **Do not restart the GPU during this phase.** The scheduled probe
exited at CUDA initialization, before its numerical checks or model evaluation.
There is no pretrained-model result from that run. The prototype and probe code
are prepared but remain unvalidated; the probe's long-input schema needs to be
adapted to the archived `ids`/`references` fields before use.

## Proposed paper-level claim, pending discussion and evidence

The practical target is reliable context extension for an append-only stream
whose final length is unknown, with reusable historical KV states. Partition
invariance alone is an implementation property, not a sufficient paper result.

The proposed empirical claim is: a prefix-consistent positional computation can
retain useful long-range answer quality as a context grows, without repeatedly
re-encoding the history, improving the quality/cost frontier over deployable
static and dynamic extension alternatives. This is a hypothesis, not a finding.
It must show an advantage over strong static controls; merely restoring a
property already satisfied by static RoPE is insufficient.

The mechanism has two logically separate claims. (1) Position transformations
change the states written at earlier layers; rephasing raw cached K cannot in
general recover those different K/V contents. This has a toy counterexample and
two retrospective BM examples, not a population-level estimate. (2) A
prefix-consistent construction preserves useful quality. This is wholly untested.
The row-wise schedule is a minimal constructive candidate for (2), not a novel
mathematical theorem or an assumed winning method.

### Discriminating experiment, before a broader benchmark

Use fresh, fixed streams and final queries, with identical final token IDs and
weights. Vary only the prefill partition, crossing the native-window boundary.
All dynamic readers rephase raw K correctly. Compare call-wide dynamic execution,
the row-wise construction, static scaling selected on separate development
streams, and full-history recomputation under the dynamic final-horizon rule.
Also include a static factor supplied with the actual final length as a stronger,
privileged control. Full recomputation is a reference, not an accuracy upper bound.

Measure whole generated answers and terminal EOS on tasks with an explicit exact
answer contract; use appropriate semantic/task scoring plus raw answers for
natural tasks. Keep numerical drift relative to native/static kernel controls
separate from task errors. Count all processing, rotations, cache conversion and
recomputation in runtime and peak-memory comparisons.

| Observation | Decision value |
| --- | --- |
| Dynamic partitions change answers, row-wise execution restores quality at comparable cached cost | Supports both an operational problem and this repair; independent confirmation is warranted |
| Partitions differ numerically but task answers remain comparable | No useful failure established; do not elevate numerical consistency to the headline |
| Full recomputation helps, but row-wise execution remains poor | Supports a state-formation issue, rejects this proposed repair; no automatic coefficient sweep |
| A simple static policy matches quality and cost across held-out streams | Online consistency is not enough to justify this method; prefer the simpler policy |
| Row-wise quality improves without the predicted state effect | Retain any independently confirmed empirical benefit, but revise the mechanism claim |
| All methods fail, including the reference | This comparison does not demonstrate recoverable headroom; no general impossibility conclusion |

### What a submission-grade result would additionally require

Frozen held-out incremental workloads should include exact multi-record
retrieval and natural document/repository question answering, with queries at
multiple stream lengths, followed by transfer to another architecture/checkpoint
family. Separate a realistic fixed deployment policy from per-example hindsight
tuning. Compare with current leading relevant methods, including Jet-Long and
MrRoPE-Pro where applicable, without calling an independently implemented
simplification an official reproduction. Report paired uncertainty over
independent documents/streams, quality versus end-to-end latency, and memory.

A useful planning target is quality within a predeclared small non-inferiority
margin of full recomputation with substantial cumulative latency reduction, or
clear task-quality improvement at matched cached-inference cost over strong
deployable baselines. Exact numerical thresholds must follow task variance and
workload economics, not be selected after seeing favorable outputs. A constant
factor/kernel speedup alone or a synthetic-only win does not prove the intended
claim. No acceptance probability is currently justified.
