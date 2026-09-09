# Frozen follow-up: full-vector mixture versus product envelopes

Registered before the 32-row run, 2026-09-08 around 23:34 UTC.

The single development input niah_multiquery_32768_0 has identical Dense,
SupportRepair, exact RoPE max and exact RoPE logmass generated token streams,
including terminal EOS. RoPEMean, Quest, native-pair PCA envelopes, random-pair
PCA envelopes and two contiguous Quest subpages do not recover all four numbers.
PostMetric4 and PreMetric4 recover the complete numerical string; the control
with contiguous groups matched to the post-metric group counts does not.
The development prompt requested commas but Dense emits spaces, so this is a
raw-Dense-trajectory recovery, not a strict prescribed-string success.

The follow-up uses every one of the 32 previously generated, not yet evaluated,
RULER multiquery rows in primary_ruler_03/frozen_long.jsonl, SHA256
046d3e90b399c6084d688aba96b19140a6667fe866b94180e86acd38c709eaee.
The fixed seed is 20260911. Only the missing native assistant-header newline was
restored from primary_ruler_02; keys, placements, order, expected answers and
background are unchanged. All rows are within Qwen2.5-3B's 32768 input+128 cap.
The essay background is shared; these are independent key/placement samples,
not 32 independent natural documents and not the complete official RULER suite.

Methods: Dense, RoPEMean, Quest, QuestSplit32, PostMetric4, PreMetric4,
MatchedContiguous4. B=64, 16 remote blocks per query head, 2048 local and 64 sink
tokens. Every question/answer token uses the tested path; the common prefix is
question-blind. No weights, native frequencies, reader K/V or generation settings
change. All methods generate greedily without constraints, max 128, stop on EOS.

PostMetric4 selects four farthest-first centers in actual post-RoPE key space,
assigns every key to the nearest center, then stores the actual group mean and
log-count. PreMetric4 changes only the clustering metric to actual pre-RoPE keys;
it still stores and scores actual post-RoPE means. MatchedContiguous4 preserves
the per-block PostMetric4 group counts but uses contiguous groups. Summaries are
FP32: 4D+4 floats per physical KV block, compared with Quest 2D and QuestSplit32
4D floats. The counts must not be omitted from memory comparisons. No parameter,
metric weight, representative count, layer split or example filtering is tuned.

Primary endpoint: entire decoded generated body, removing only a final EOS token,
exactly equals the four expected numeric strings in query order with single spaces;
terminal EOS is required. No strip, substring, digit extraction or F1 supplies
primary correctness. Trimmed exact is recorded separately. Report all 32 rows,
paired wins/losses versus each baseline, and all incomplete/capped generations.

This is a known clustering-family mechanism test, not a claimed new clustering
algorithm. In particular, equal post/pre performance does not establish a
position-specific benefit. A quality improvement at fixed gathered KV budget is
not a speedup: this Python reference includes cache construction, per-head GQA
duplication, and unoptimized indexing; build and continuation timing stay separate.
A positive development example alone does not establish a publishable method.
