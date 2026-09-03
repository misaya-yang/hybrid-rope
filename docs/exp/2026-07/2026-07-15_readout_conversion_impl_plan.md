# Track Z readout conversion: feasibility audit and implementation plan

Date: 2026-07-15

Status: code-grounded implementation plan, not a result report. No paper
number changes here. The July 14 probe remains single-seed, ten-case supporting
evidence only. All GPU commands below are pending explicit approval and must
load the matched step-300 Geo/EVQ adapters read-only.

The binding labels and decision rules are those in
`docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:12-23`. In particular,
anything that uses the true gold block or true first gold token is
`oracle-diagnostic`; choosing one global scalar or layer on dev is `ZT-cal`,
not `ZT-0`.

## Feasibility audit

### 1. Persisted logits: no

Neither full-vocabulary first-step logits nor per-answer-position logits are
present in the saved ten-case causal artifacts.

- `_answer_nll` obtains a full logit vector at every gold answer position, but
  immediately reduces it to cross-entropy and strict rank and returns only
  aggregates (`experiments/lora_evq_v2/eval_sparse_conversion.py:1536-1575`).
- `_first_step_logits` returns one tensor only to the in-process dense/full
  parity check (`experiments/lora_evq_v2/eval_sparse_conversion.py:1931-1948`,
  `experiments/lora_evq_v2/eval_sparse_conversion.py:2328-2340`).
- The saved row contains answer-token count, depth, aggregate rank, generation,
  and NLL, not logits
  (`results/lora_sparse_conversion_s42_20260714/server_26521/rank_16k_v1/geo.json:177-180`,
  `results/lora_sparse_conversion_s42_20260714/server_26521/rank_16k_v1/geo.json:218-229`).
  Read-only inspection also found no tensor-logit file in the saved bundle.

Minimal recompute for the required first artifact: the same ten unique 16K
cases, both unchanged adapters, one dense prefill per case and two short cache
branches (`dense` and `gold_drop_all`) over all three answer tokens. Cache
sharing gives 20 matched-arm prefills:

- prompt-equivalent input: `10 cases x 2 arms x 16,384 = 327,680` tokens;
- actual model input including two three-token branches:
  `20 x (16,383 + 2 x 3) = 327,780` tokens;
- if real-model cache cloning fails, the no-sharing fallback would be 40
  prefills / 655,440 actual input tokens and requires approval because it
  changes the registered compute accounting.

The new `readout-trace` path shares the prefill, writes immutable BF16 tensors,
and refuses overwrite
(`experiments/lora_evq_v2/eval_sparse_conversion.py:2035-2058`,
`experiments/lora_evq_v2/eval_sparse_conversion.py:2061-2169`). It has not been
run on GPU.

### 2. `gold_drop_all` across answer positions: executable, not persisted

`gold_drop_all` removes the registered gold block on every attention head
(`experiments/lora_evq_v2/eval_sparse_conversion.py:825-837`). The mode and
gold span are configured once before `_answer_nll` teacher-forces every token,
so the intervention is applied at every answer position, not only at the first
token (`experiments/lora_evq_v2/eval_sparse_conversion.py:1545-1562`).

What is reusable today is the intervention implementation and the frozen KV
layout, not a saved `z_ablate(j)`. The old JSON reduces all positions to
first/mean/max rank and NLL. The new trace collector materializes
`[answer_position, decoder_layer, vocabulary]` for both branches and verifies
the final-layer logit-lens parity before saving.

### 3. Needle depth: stored directly

Depth is present per saved rank row
(`results/lora_sparse_conversion_s42_20260714/server_26521/rank_16k_v1/geo.json:177-180`)
and per Phase-0 case alongside the exact answer span
(`results/lora_sparse_conversion_s42_20260714/phase0_geo.json:156-163`). Z3 Part
A can therefore stratify the existing 16K cases without reconstructing depth.

If a case must be regenerated, depth is deterministically derivable from the
frozen builder's length/depth/trial grid and insertion calculation
(`experiments/lora_evq_v2/prepare_seed42_capability_data.py:438-453`,
`experiments/lora_evq_v2/prepare_seed42_capability_data.py:471-510`). The saved
`example_id` also encodes the registered depth, but the explicit field is the
authoritative value.

### 4. Association-swap builder: absent before this implementation

The existing frozen passkey builder makes one passkey needle and one query
(`experiments/lora_evq_v2/prepare_seed42_capability_data.py:455-481`,
`experiments/lora_evq_v2/prepare_seed42_capability_data.py:500-519`). The only
existing counterfactual helper replaces or removes one answer span; it does not
construct two labeled key/value associations
(`experiments/lora_evq_v2/eval_sparse_conversion.py:1578-1585`). No existing
builder satisfied the Z1 token-multiset and position controls.

`build_association_swap_pair` is therefore new
(`experiments/lora_evq_v2/eval_sparse_conversion.py:174-316`). Its deterministic
contract is:

- two distinct, equal-token-length key labels and two distinct,
  equal-token-length answers;
- the answer first tokens must have the same pre-registered frequency bucket;
- one fixed token-level template and filler stream;
- records placed in two blocks symmetric about the registered depth;
- query-A and query-B prompts have exactly the same length and token multiset;
- `mirror=False/True` swaps which answer occupies the left/right slot, so the
  planned 256 pairs balance physical position rather than treating it as a
  nuisance after collection.

The planned frequency bucket is fixed as
`floor(log2(training_token_count + 1))`, computed once on CPU from the frozen
LoRA training-token artifact. It is a matching control, not a selected
hyperparameter. The count artifact is not present in this checkout; see Open
questions.

### 5. Teacher-forced rank and separate generation path: confirmed, with one correction

The rank metric is teacher-forced against the registered gold string:
`_answer_nll` loops over gold `answer_ids`, ranks the current gold label, then
sets the next input token to that label
(`experiments/lora_evq_v2/eval_sparse_conversion.py:1555-1562`). Strict rank is
`1 + count(logit > gold_logit)`, so ties do not outrank the target
(`experiments/lora_evq_v2/eval_sparse_conversion.py:1475-1482`).

Free generation is separate: `_generate` greedily feeds back `argmax`, checks
EOS, and stops at `max_new_tokens`
(`experiments/lora_evq_v2/eval_sparse_conversion.py:1890-1927`). Passkey rows
set that cap to 32 (`experiments/lora_evq_v2/eval_sparse_conversion.py:2179-2191`).
The probe's “full 32 tokens without EOS and began with the same token” is an
observed outcome (`docs/exp/2026-07/2026-07-14_lora_retrieval_conversion_probe.md:205-221`),
not a harness setting: current code neither disables EOS nor fixes the first
generated token. Z0 therefore needs a separate, explicit gold-`g_1` forced
branch; that branch is `oracle-diagnostic`.

## Reuse versus build

| Function or artifact | Decision | Evidence / concrete action |
| --- | --- | --- |
| Frozen step-300 adapter identity and hashes | reuse | Saved receipts identify global step/max steps 300 and adapter hashes (`results/lora_sparse_conversion_s42_20260714/server_26521/rank_16k_v1/geo.json:3-25`, `results/lora_sparse_conversion_s42_20260714/server_26521/rank_16k_v1/geo.json:36-55`). Load only through existing validation and `PeftModel.from_pretrained` (`experiments/lora_evq_v2/eval_sparse_conversion.py:973-1008`); never save adapters. |
| Frozen passkey file and hash checks | reuse | `load_passkey_rows` checks schema, filename, SHA256, size, row count, prompt hash, and exact length (`experiments/lora_evq_v2/eval_sparse_conversion.py:501-533`). No builder writes into this root. |
| Ten causal cases and registered depths | reuse | Reuse trials 0/1 at five depths; `_select_passkey_rows` enforces the expected Cartesian count (`experiments/lora_evq_v2/eval_sparse_conversion.py:1197-1212`). |
| Saved ten-case rank/NLL JSON | reuse | Re-analysis for existing aggregate checks only. It cannot produce the linchpin because logits are absent. |
| Dense prefill, decode cache, original rotary indices | reuse | `_prefill` is dense and `_decode_logits` consumes the existing cache (`experiments/lora_evq_v2/eval_sparse_conversion.py:1438-1472`). |
| `gold_drop_all` | reuse | Existing all-head mask (`experiments/lora_evq_v2/eval_sparse_conversion.py:825-837`). Always report its measurements as `oracle-diagnostic`. |
| Dense/full parity | reuse | Existing first-step parity path (`experiments/lora_evq_v2/eval_sparse_conversion.py:1931-1948`, `experiments/lora_evq_v2/eval_sparse_conversion.py:2328-2340`); new per-layer traces additionally check last-layer parity. |
| Full-vocab, all-position, all-layer traces | extend | New cache-branch collector and immutable trace manifest (`experiments/lora_evq_v2/eval_sparse_conversion.py:1952-2169`). Raw tensors may contain gold token IDs; JSON manifests do not contain answer text. |
| Causal-delta strict rank | new | Vectorized `rank_t[z_full-z_ablate]` (`experiments/lora_evq_v2/eval_sparse_conversion.py:1485-1505`). |
| Per-layer causal logit lens | new | Apply final norm and unembedding separately to full/ablated states, then subtract (`experiments/lora_evq_v2/eval_sparse_conversion.py:1508-1532`). |
| Association-swap pair builder | new | Deterministic matched builder (`experiments/lora_evq_v2/eval_sparse_conversion.py:174-316`). |
| Frequency-bucket receipt and 128-dev/128-test swap dataset | new | CPU-only build under the new result root; hash source training manifest/token counts, seed, template, split, depth allocation, and every prompt. Never change frozen eval files. |
| Swap trace collector | extend | Reuse `_prefill`, cache branching, and per-layer lens. Add a command consuming only the newly frozen swap manifest; save schema `evq_cosh.readout_association_swap_trace.v1`. This is the remaining Step-1 collection task. |
| Forced-`g_1` suffix generation | extend | Add an explicit first-token override to a dedicated diagnostic path; do not alter ordinary `_generate` semantics. |
| Linchpin aggregation and figure | new | `scripts/analysis/readout_conversion.py:85-198` validates matched Geo/EVQ traces and computes causal rank/swap score; `scripts/analysis/readout_conversion.py:200-347` aggregates and plots; `scripts/analysis/readout_conversion.py:350-418` enforces planned counts and writes a sanitized summary. |
| Focused verification | new | Causal strict-rank, swap controls, logit-lens delta, end-to-end figure, and sanitization tests (`tests/test_readout_conversion.py:28-207`). |
| Z2/Z3 execution and every S-track training run | gated | Do not implement or launch beyond the scaffolds and decision branches below without approval. |

## Output and data contracts

All new artifacts go below
`results/readout_conversion_s42_20260715/`. Collectors write to a sibling
`.incomplete` directory and refuse an existing final or incomplete path. No
command writes into `results/lora_sparse_conversion_s42_20260714/`, the model
directory, adapter directories, the training-data root, or the frozen passkey
root.

Proposed layout:

```text
results/readout_conversion_s42_20260715/
  raw/
    causal_native_geo/{manifest.json,records/*.pt}
    causal_evq_cosh/{manifest.json,records/*.pt}
    swap_native_geo/{manifest.json,records/*.pt}
    swap_evq_cosh/{manifest.json,records/*.pt}
  linchpin/
    linchpin_causal_delta_rank.png
    summary.json
```

Raw `.pt` records are analysis inputs and may contain gold/candidate token IDs,
but never decoded answer strings. `summary.json` contains only aggregate ranks,
logit deltas, swap scores/CIs, counts, input hashes, labels, and the figure
name. The analysis deliberately ignores unknown raw fields; its focused test
injects sentinel passkey text into raw records and proves it is absent from the
summary (`tests/test_readout_conversion.py:106-207`).

The pre-existing Phase-1 writer serializes `references`, including answer text
(`experiments/lora_evq_v2/eval_sparse_conversion.py:2280-2285`). Therefore the
old raw rank JSON is not a sanitized publication artifact even though the
sanitized causal manifest contains no passkey content
(`results/lora_sparse_conversion_s42_20260714/rtx5090_causal_manifest.json:1-10`).

## Track Z Step 1: concrete tasks

Token accounting below is for the required matched Geo+EVQ measurement. A
“16K prefill equivalent” is 16,384 input tokens. Cache reuse means full and
ablated one-token branches share the same dense prompt prefill. No budget,
block-count, sparse-threshold, temperature, or beam scan is planned.

### Z0-A — dense teacher-forced/readout baseline — `ZT-0`

Coding task: sanitize and re-aggregate the already saved dense rank/NLL rows;
when trace collection is approved, also derive dense per-position rank/NLL from
the same full logits without selecting a layer or scalar.

- manipulated variable: answer position `j` under unchanged dense inference.
- prediction: the dense first-token versus suffix profile identifies whether
  the readout gap is concentrated at `g_1` before using any oracle ablation.
- metric: per-position dense gold rank, NLL, margin, ordinary 32-token EM,
  containment, generated-token count, and EOS termination.
- kill condition: if no first-token/suffix contrast exists, do not attribute
  failure to a single first-token gate; Z1 still runs because identity and
  continuation are separate questions.
- budget in input tokens: 0 for saved aggregate re-analysis; 0 incremental for
  per-position metrics because the full branch is shared with Z0-B.

### Z0-B — full versus all-head gold-block ablation — `oracle-diagnostic`

Coding task: use `readout-trace` for the existing ten causal cases first;
compute `d_j(t)=z_full(t)-z_ablate(t)` for every answer position and layer.
Expand only within the Z0 cap after resolving the 100-versus-256 frozen-set
question.

- manipulated variable: full context versus removing the true gold 128-token
  block on every head, stratified by answer position `j`.
- prediction: a first-token bottleneck has a distinct `g_1` causal profile;
  continuation-only use produces larger suffix effects; uniformly weak ranks
  reject the single-token story.
- metric: per-position `dNLL_j`, gold causal logit delta, strict causal-delta
  rank, causal margin, and share of total NLL gain at `g_1` versus suffix.
- kill condition: several later positions retain median causal rank above 10,
  or no position carries answer-specific delta, kills “only `g_1` is stuck.”
- budget in input tokens: first artifact exactly 327,780 actual matched-arm
  tokens (327,680 prompt-equivalent); the binding Z0 ceiling remains 256
  samples, with its prompt work shared with Z0-C.

### Z0-C — force only the true first gold token — `oracle-diagnostic`

Coding task: add a dedicated forced-`g_1` branch that reuses the full prefill,
then greedily decodes the remaining suffix under the existing 32-token cap and
normal EOS behavior. Ordinary generation remains unchanged.

- manipulated variable: free greedy first token versus explicitly supplied
  true `g_1`; all later tokens are generated normally.
- prediction: if generation format is gated only by the first token, suffix EM
  after forced `g_1` approaches saturation.
- metric: suffix exact match, full-string exact match with the forced token,
  suffix containment, generated length, and EOS termination.
- kill condition: suffix EM below 80% kills the “only the first token is
  stuck” story, exactly as required by the source plan.
- budget in input tokens: at the 256-sample cap and two matched arms, at most
  `512 x (16,383 + 32) = 8,404,480` actual tokens (8,388,608 16K-prompt
  equivalents); if restricted to the current 100 frozen 16K rows, 3,283,000
  actual tokens.

### Z1-A — association swap under full/gold ablation — `oracle-diagnostic`

Coding task: CPU-build and freeze 128 dev + 128 test pairs using the new
builder; balance five depths and mirror orientation by a fixed seed-42 rule;
run both unchanged adapters; extend the trace command to save two prompt
conditions per pair under the association-swap schema.

- manipulated variable: only which labeled key the query selects, with token
  multiset, answer first-token frequency bucket, template, depth, and mirrored
  physical position matched; compare full versus true-gold-block ablation.
- prediction: precise value identity makes `d(A)-d(B)` reverse when the query
  selects B; generic position/format signal does not follow the swap.
- metric: per-layer causal rank, gold-versus-decoy causal margin, and paired
  swap-follow score
  `[d_xA(A)-d_xA(B)]-[d_xB(A)-d_xB(B)]`, with a paired test CI.
- kill condition: test swap-follow CI lower bound is not above 0, or gold remains
  near random absolute rank in `d`, stops the precise-readable-identity claim
  and sends execution to S1.
- budget in input tokens: 256 pairs x 2 prompt conditions x 2 matched arms =
  1,024 shared prefills; `1,024 x (16,383 + 2 short branches) = 16,778,240`
  actual tokens (16,777,216 prompt-equivalent). No dev/test sample increase.

### Z1-B — analytic scalar feasibility with the true gold block — `oracle-diagnostic`

Coding task: from saved full/ablated full-vocabulary logits, intersect the
competitor half-lines for `s_alpha=z+alpha*d` per sample. This is analytic, not
an alpha sweep.

- manipulated variable: the mathematical scalar `alpha` over the complete
  feasible interval induced by the oracle gold-block delta.
- prediction: a non-empty interval means the answer is present but
  underweighted; an empty interval means scalar causal contrast cannot make it
  top-1.
- metric: feasible fraction, interval bounds, and oracle top-1 upper bound; do
  not report a per-sample-best alpha as deployable.
- kill condition: if at least 90% of samples have no feasible alpha, stop all
  causal-delta amplification.
- budget in input tokens: 0; pure re-analysis of Z1-A logits.

### Z1-C — fixed selector delta at `alpha=1` — `ZT-0`

Coding task: only if Z1-B is feasible, add the already frozen block selector as
one additional answer-side cache branch and evaluate exactly `alpha=1`.

- manipulated variable: full logits versus one fixed selector-derived causal
  delta at the pre-registered scalar `alpha=1`.
- prediction: a selector that preserves the oracle direction converts some
  samples without calibration.
- metric: first-token top-1, full EM, causal margin, and swap-follow on test.
- kill condition: if the oracle succeeds but this fixed selector fails, stop
  the deployable result here and diagnose selector recall without scanning
  thresholds or block counts.
- budget in input tokens: no new 16K prefill; at most 1,024 additional one-token
  decode inputs, one per Z1 prompt-condition/arm cache.

### Z1-D — one global dev-selected scalar — `ZT-cal`

Coding task: only after Z1-C, choose one global alpha from the analytic dev
interval endpoints, freeze it, and evaluate test once. There is no grid or
temperature sweep.

- manipulated variable: one global scalar selected on the independent 128-pair
  dev set and frozen before opening the 128-pair test set.
- prediction: a shared under-scaling mechanism yields material test top-1/EM;
  sample-specific incompatible intervals do not.
- metric: frozen-alpha test top-1, full EM, causal margin, swap-follow, and dev
  versus test interval coverage.
- kill condition: no material test top-1/EM improvement stops scalar contrast.
- budget in input tokens: 0; pure re-analysis of already saved selector logits.

### Linchpin figure — `oracle-diagnostic`

Coding task: run `scripts/analysis/readout_conversion.py` over exactly one Geo
and one EVQ causal manifest plus exactly one Geo and one EVQ swap manifest. The
script validates case-for-case pairing and file hashes, computes strict
per-`(j,l,d)` ranks and swap-follow, streams raw tensors, and emits the figure
plus sanitized JSON.

- manipulated variable: answer position `j`, decoder layer `l`, needle depth
  `d`, and matched query association, under full-minus-gold-ablated logits.
- prediction: the joint trajectory discriminates first-token gating,
  continuation, mid-layer overwrite, depth OOD, and generic versus precise
  identity signal as specified in the source plan.
- metric: median `rank_t[z_full(t)-z_ablate(t)]` heatmaps; first-position test
  swap overlay; gold causal delta; swap mean, median, positive fraction, and
  deterministic paired-bootstrap CI.
- kill condition: the figure is invalid if Geo/EVQ cases, layers, depths, or
  swap pairs are unmatched; scientifically, use the Z0/Z1 kill rules above.
- budget in input tokens: 0; pure CPU re-analysis after trace collection.

## Explicit decision gate

Run all Z0 diagnostics and the preregistered Z1 association swap before choosing
a branch. Z0 decides whether a forced-first-token wrapper is meaningful; it
does not replace the identity test.

- Swap-follow negative: if the paired test CI lower bound is `<= 0`, or gold
  causal rank stays near random, stop Track Z immediately and open only the S1
  planning gate. Do not try temperature, beam width, top-p, sparse threshold,
  sparse budget, block count, or forced-attention variants.
- Swap-follow positive, oracle scalar infeasible: if the swap CI lower bound is
  `> 0` but at least 90% of oracle intervals are empty, precise identity exists
  in the measured delta but scalar amplification is not viable. Stop Z1
  amplification and enter S1; do not publish an oracle upper bound as a result.
- Swap-follow positive, oracle scalar feasible: run the fixed selector at
  `alpha=1` as `ZT-0`, then at most one global dev-selected alpha as `ZT-cal`.
  If the layer trajectory is mid-layer-good/final-bad, unlock Z2; if final
  causal rank is already near top-1, unlock the Z3 wrapper branch. No other
  Track-Z expansion is authorized by this document.
- Independently, Z0 negative (forced-`g_1` suffix EM below 80% or several later
  positions above median rank 10) kills the first-token-only wrapper even when
  swap-follow is positive.

## Gated outline beyond Step 1

This section points to the binding source plan and does not redesign it.

| Gate | Next measurement | Label | Binding source |
| --- | --- | --- | --- |
| Z1 oracle-feasible and a mid-layer peak is visible | Z2 layer trajectory and three-layer/64-sample causal confirmation | `oracle-diagnostic` | `docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:217-237` |
| Z1 oracle-feasible; depth pattern warrants it | Z3 Part A fixed-length position stratification | `oracle-diagnostic` | `docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:239-261` |
| Z1 identity positive and a fixed candidate rule is frozen without dev selection | Z3 Part B fixed extractive/trie wrapper | `ZT-0` | `docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:263-285` |
| Same wrapper but one global layer/scalar is chosen on independent dev | Z3 Part B calibrated wrapper | `ZT-cal` | `docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:263-285` |
| Track Z stops | S1 three-arm length-distribution experiment; plan only, no launch | `supervised` | `docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:287-325` |
| S1 shows the source plan's margin/NLL branch condition | S2 one fixed loss contrast; no loss sweep | `supervised` | `docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:327-350` |
| Z1/Z2 locates the missing pathway | S3 one mechanism-selected adapter locus plus parameter-matched Q/K control | `supervised` | `docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:352-376` |
| One frozen recipe already has non-zero EM | S4 `{Geo,EVQ} x {8K,16K}` on three new seeds | `supervised` | `docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:378-394` |

No statement that “EVQ is more convertible” is allowed before the S4
four-arm, three-new-seed interaction CI excludes zero
(`docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:384-394`).

## Runnable command skeletons (do not execute without approval)

The checkout currently contains receipts and saved JSON, not the actual base
model, adapter tensor directories, model/training manifests, or frozen
`passkey.jsonl`. Supply those as read-only mounted paths. These commands create
new directories only.

```bash
python -m experiments.lora_evq_v2.eval_sparse_conversion readout-trace \
  --model-name <read-only-model> \
  --model-manifest <read-only-model-manifest.json> \
  --training-data-manifest <read-only-training-manifest.json> \
  --adapter-dir <read-only-step300-geo-adapter> \
  --substrate native_geo \
  --data-root <read-only-frozen-passkey-root> \
  --lengths 16384 \
  --trials 0,1 \
  --output results/readout_conversion_s42_20260715/raw/causal_native_geo

python -m experiments.lora_evq_v2.eval_sparse_conversion readout-trace \
  --model-name <read-only-model> \
  --model-manifest <read-only-model-manifest.json> \
  --training-data-manifest <read-only-training-manifest.json> \
  --adapter-dir <read-only-step300-evq-adapter> \
  --substrate evq_cosh \
  --data-root <read-only-frozen-passkey-root> \
  --lengths 16384 \
  --trials 0,1 \
  --output results/readout_conversion_s42_20260715/raw/causal_evq_cosh
```

After the separately frozen Z1 swap manifests and their matched trace
directories exist:

```bash
python scripts/analysis/readout_conversion.py \
  --causal-manifest results/readout_conversion_s42_20260715/raw/causal_native_geo/manifest.json \
  --causal-manifest results/readout_conversion_s42_20260715/raw/causal_evq_cosh/manifest.json \
  --swap-manifest results/readout_conversion_s42_20260715/raw/swap_native_geo/manifest.json \
  --swap-manifest results/readout_conversion_s42_20260715/raw/swap_evq_cosh/manifest.json \
  --output-dir results/readout_conversion_s42_20260715/linchpin
```

## Open questions and feasibility mismatches

1. **External read-only inputs are absent locally.** The code requires CUDA and
   validates real adapter/model/data artifacts
   (`experiments/lora_evq_v2/eval_sparse_conversion.py:973-1008`), while this
   checkout has only result receipts/hashes. No real trace collection is
   runnable until the read-only paths are supplied. This does not authorize a
   GPU run.
2. **Z0 says 256 fixed samples, but the frozen 16K passkey grid has only 100.**
   The builder has five depths and 20 trials at each length, hence 100 rows at
   16K (`experiments/lora_evq_v2/prepare_seed42_capability_data.py:438-453`,
   `experiments/lora_evq_v2/prepare_seed42_capability_data.py:471-473`), versus
   the source plan's 256 (`docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:175-179`).
   Do not alter the frozen file. Approval must choose either the 100-row frozen
   evaluation or a separate, newly versioned 156-row readout-only extension;
   the latter changes the artifact manifest but remains under the 256 cap.
3. **No first-token frequency-count artifact was saved.** The swap builder
   refuses unmatched/unregistered buckets. Compute and hash counts from frozen
   training token IDs on CPU, or stop Z1 if the source data is unavailable; do
   not replace this with an unverified tokenizer-frequency heuristic.
4. **The Z1 heading's `ZT-0 diagnostic` conflicts with its gold-block
   intervention.** The binding label definition says true-gold-block use is
   `oracle-diagnostic`
   (`docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:15-20`). This
   implementation labels Z1-A and the linchpin causal delta
   `oracle-diagnostic`; only fixed selector `alpha=1` is `ZT-0`.
5. **“Fixed first token, no EOS” is observational, not configured.** Current
   generation checks EOS and chooses its own first token
   (`experiments/lora_evq_v2/eval_sparse_conversion.py:1907-1917`). The explicit
   forced-gold branch is new and must be labeled `oracle-diagnostic`.
6. **Existing raw JSON is not sanitized.** It includes serialized `references`
   (`experiments/lora_evq_v2/eval_sparse_conversion.py:2280-2285`) even though
   the probe describes a text-free rank diagnostic. Never copy it into the new
   summary; use only aggregate fields or the new tensor/manifest contract.
7. **Real CUDA cache cloning and per-layer parity remain unverified.** A tiny
   CPU Llama cache can be deep-copied branch-independently, but the actual BF16
   PEFT/CUDA model was intentionally not run. The collector fails closed on
   final-layer parity. If cache sharing fails, the input budget doubles and
   requires approval before a no-sharing rerun.
8. **Raw swap traces are large.** Keeping BF16 `full_logits` and
   `ablated_logits` for two query conditions, 32 layers, and full vocabulary is
   roughly 16-17 GB over 256 pairs and both arms. This does not change the
   input-token budget. Before Z1 collection, choose between retaining both raw
   tensors for later analytic alpha work or retaining baseline plus delta with
   an explicitly versioned equivalent schema; do not silently downcast or
   discard fields.
9. **The actual linchpin figure is still pending.** The analysis skeleton is
   runnable and CPU-tested on synthetic matched traces, but no experimental
   figure or summary has been fabricated from absent logits.
