# P2: same-generation Qwen s2 baseline completion

## Hypothesis and entrance

P0 confirmed a protocol-specific Gemma reference; P1 now completes useful
reference-correct s2/s4 curves with frozen parameters. That rules out the old
16K negative as sufficient evidence against K128 transport, but does not
privilege physical x over index. P2 therefore asks a smaller question: does
the existing Qwen K32 Native/long crossing persist in a matched-s2 comparison
with a deterministic YaRN baseline, and what does the same law do on its
closest available K64 same-generation checkpoint?

Use only the existing Qwen2.5-0.5B-Instruct (K32) and
Qwen2.5-1.5B-Instruct (K64) artifacts. No model download, model selection,
reference search, boundary fitting or new profile is registered.

## Construction and fixed budget

- Retain each Qwen config/reference length 32768. Do not transfer Gemma's
  calibrated 4096 reference to this family. Target 65536 fixes s=2.
- Freeze the existing physical/index law, boundaries and c=.074. YaRN uses
  its published fixed beta bounds and c=.1, exported from the installed HF
  CPU initializer. All profiles remain unchanged across 32K/64K requests.
- Reuse existing input rows only after checking checkpoint, Native tensor,
  tokenizer, cell identities and decoding contract. Existing results are
  historical paired evidence, not fresh independent replication.
- Initial GPU work: K32 YaRN2 at 32K/64K, four fixed tasks, 20 rows per task
  and length (160 generations). This fills a missing baseline, not a sweep.
- Conditional next work: K64 YaRN2 at 32K/64K (160 generations), followed
  by its unique C2-s2 profile (160). K64 physical/index are identical by
  construction and must not be run as nominally independent alternatives.
- Reuse Native scores only if their complete scientific identity can be
  verified. Otherwise register the required matched Native rerun explicitly
  before using a retention ratio; never splice an unverifiable comparator.

## Confounds and decision rule

Native 1x and YaRN2 at 2x must both resolve nonzero capability before
interpreting a checkpoint's transport behavior. A failed resolver stops
interpretation for that checkpoint, without changing its reference or scale.
Report the existing .875 Native macro-retention margin, all four task
scores, and paired row-bootstrap intervals. This is not yet a Qwen natural
NLL double gate. Small contrasts do not establish rankings.

K32 physical/index provide a non-degenerate coordinate contrast; K64 does
not. Differences in model size, head count and training/checkpoint remain
confounds, so this triangulation cannot identify causal K or a trend of
physical-minus-index with K. It can establish only matched-s2 behavior of
the fixed methods on these two checkpoints. If the mainline underperforms,
record which premise failed; do not add G(x;K), residuals or gain tuning.

The initial 480-generation budget ends after these registered panels.
Independent seed replication or Native-only mechanistic work requires an
evidence-based registration after these results, not automatic queue growth.
No s8, PI/NTK/Resonance sweep or SOTA claim is opened here.

## Evidence and lifecycle

Existing inputs/results are routed by `K32_MATCHED_S2_RECEIPT_20260901`,
`K32_FINITE_K_COUPLING_GPU_RECEIPT_20260901` and
`LOW_DIM_COUPLING_GPU_RECEIPT_20260901` in the evidence directory. Raw
checkpoint/data/code/table hashes, memory and finite first-step receipts are
recorded at launch. No shutdown is scheduled under the current user goal.
The protected 1.485B checkpoint is untouched.

## Bounded historical-execution replay amendment

Before adding any new comparison claim, the source audit recovered the exact
K64 historical runner and found no greedy/scorer change. The exact K32
historical runner was not recovered; its 480 raw rows and scores were verified
but cannot establish full source-level equality. Therefore register only
24 diagnostic replays: Native/physical/index, both lengths, first row of each
of the four tasks. Compare full decoded predictions and scores with the old
rows, including old failures. Do not select easy/successful rows or tune after
a mismatch. The total budget is now 480 panel generations plus 24 replays.
Agreement is bounded decoded-output parity, not token-level parity (old IDs
were not saved) or proof that all old execution paths were identical. A
mismatch prevents upgrading historical results to a verified current-run
comparator and requires attribution, not automatic profile modification.
