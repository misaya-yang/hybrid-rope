# Work-machine next experiment plan (2026-09-01)

This is the frozen execution handoff subordinate to `INDEX.md` §6, not a
second research agenda. The tree was reported complete on the work machine on
2026-09-02; its raw remote receipts have not been imported into this checkout.
The two-day synthesis records the session-level outcomes. The commands below
are retained for provenance and are no longer a current action queue.

## 1. Frozen conclusion entering the work machine

The current evidence supports one engineering representative and closes two
branches:

- carry forward **normalized-index s2** as one request-wide static table;
- stop physical-x coordinate superiority: K32 does not reproduce its long
  advantage and K128 independently favors index;
- keep Native-Q/K P3, finite-cell correction, `G(x;K)`, residuals, new gain,
  and boundary search closed;
- do not call the method universal or SOTA before natural likelihood, natural
  QA, broader baselines, and another scale/checkpoint confirmation.

The latest untouched K32 full-RULER-13 result is `CLEAR_ADVANCE`:

```text
                    32K        64K
Native            .547821    .220513
normalized-index  .559167    .514551
official YaRN     .559423    .453654
```

At 64K, index-minus-YaRN is `+.060897`, paired 95% CI
`[.027627,.095835]`. At 32K they are unresolved. Natural QA rows inside RULER
do not uniformly favor index, so natural-task confirmation is mandatory.

## 2. Immediate work-machine experiment: packed-natural NLL

Owner:
[`K32_PACKED_NATURAL_NLL_CONFIRMATION_PREFLIGHT_20260901`](preflights/K32_PACKED_NATURAL_NLL_CONFIRMATION_PREFLIGHT_20260901.md).
Its full-RULER entrance is satisfied; no protocol field may change.

Prepared model-free data identity:

- source file: `001_00000.parquet`;
- source SHA-256:
  `3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693`;
- fixed start row: `650000`;
- consumed range: `650000..652017`;
- 32 non-overlapping packed 64K streams and paired 32K suffixes;
- data manifest SHA-256:
  `d1970c519a3c1840cbda06daa30642f3a8dbc26d4fc00ef648dd330a3d1258ac`;
- rows SHA-256:
  `8be1fee7037968d02b6e1839ec69e33f4e512c9f9f10624d68f6acc857848487`;
- model evaluation status: `NOT_RUN`.

If the prepared external artifact is unavailable, regenerate it exactly rather
than selecting replacement rows:

```bash
python scripts/data/prepare_qwen_k32_natural_nll.py \
  --source "$SOURCE_ROOT/sample/10BT/001_00000.parquet" \
  --expected-source-sha256 3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693 \
  --start-row 650000 \
  --checkpoint "$QWEN_K32_CHECKPOINT" \
  --output "$NLL_ROOT/data"
```

Verify its manifest/rows hashes match the values above, then run the frozen
budgeted tree:

```bash
scripts/eval/run_qwen_k32_experiment_tree.sh
```

The runner first audits target boundaries, executes a metric-blind timing
canary and refuses the primary if the 20%-buffered worst-case NLL cost exceeds
eight RMB. Stage A is Native/index only. The same frozen YaRN arm runs next as
the planned comparator when Stage A passes, or as a non-rescuing positive
control when it fails. The evaluator hard-binds checkpoint/config/table hashes
and gains, uses `use_cache=False`, and keeps one static table for every complete
forward.

Decision:

1. index 32K PPL retention must be at least `.875`;
2. 64K index-minus-Native NLL CI upper endpoint must be below zero;
3. report index-minus-YaRN NLL as favored only if its CI upper endpoint is
   below zero, YaRN-favored only if its lower endpoint is above zero, otherwise
   unresolved.

If either gate 1 or 2 fails, record whether YaRN passes the same two gates and
stop. `INDEX_SPECIFIC_FAILURE` means YaRN resolves the protocol while index
does not; `SHARED_OR_UNRESOLVED_NATURAL_FAILURE` means it does not. Neither
outcome permits another split, gain, physical-x rescue, or profile change.

## 3. Conditional natural-task confirmation

Open only if the packed-natural resolver passes and index is not NLL-dominated
by YaRN. Compare exactly `{Native, normalized-index, official YaRN}` on a
model-free, deterministic 30-row far-evidence panel:

- 2WikiMultihopQA;
- Qasper;
- HotpotQA.

For each task, take the first ten admissible official LongBench rows in
canonical source-row-hash order. Put the complete released context before
token 28,672, insert unrelated natural packed text, and put the query suffix
near 64K. The evidence-to-query lower bound is at least 32,768 tokens; inserted
filler is rejected when it contains a normalized reference answer. Qasper
yes/no/unanswerable rows are excluded because random filler cannot provide a
meaningful answer-leak check. This is a derived far-evidence test with
official-style token F1, not official LongBench.

The frozen LongBench main/data archive hashes are respectively
`3824b6dae9738eb70f7c89cf1a1e13d37cb626b7dab779780d1a96f393c2280a`
and `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`.

Run a two-row metric-blind generation canary covering the 32- and 128-token
decode budgets. The buffered NLL estimate plus buffered QA estimate must be at
most ten RMB; otherwise QA does not start. A positive utilization result
requires the paired task-stratified index-minus-Native macro interval lower
endpoint above zero and positive point deltas on at least two of three tasks.
Report index-minus-YaRN separately.

Use the repository's complete answer/F1 contract and fixed context budgets;
do not select examples, prompts, truncation, or decoding from RULER or NLL
outcomes. Report per-task F1/exact rows and paired uncertainty. RULER's two QA
rows already warn that a macro retrieval gain may not transfer to natural QA.

Stop if this gate fails or index is materially below YaRN across natural QA.
That outcome means the method is a synthetic-retrieval operating point, not a
general deployment profile.

## 4. Conditional baseline and generality stage

Only after both natural gates pass:

1. add deterministic static PI, static NTK, Resonance, and official YaRN
   baselines under matched checkpoint/reference/target contracts;
2. use new rows and fixed factors `2/4/8`, without per-factor profile tuning;
3. confirm the same normalized-index construction on one second checkpoint
   with a different Native window or rotary budget;
4. report curves across every interior length, not only the maximum endpoint.

This stage may establish breadth or baseline competitiveness. It cannot turn
one checkpoint pair into K causality, nor can a finite scale grid establish a
continuous guarantee.

## 5. Work-machine preflight checklist

- read `docs/overview/RTX5090_BLACKWELL_PROFILE.md` before GPU launch;
- verify repository SHA, clean/understood worktree, checkpoint/config/weight
  hashes, source/data/table hashes, free space, output path, and shutdown plan;
- require the first real forward, finite NLL, throughput/memory receipt, and
  exact active table/gain identity;
- preserve raw per-stream/per-task rows, run manifests, code hashes, runtime
  receipt, and all negative endpoints;
- keep the protected 1.485B asset untouched;
- no experiment in this plan authorizes model training or manuscript SOTA
  wording by itself.
