# Llama Plan B frozen scoring contract

Revision: `llama-planb-ruler-derived-v1`  
Frozen before the first non-engineering P result.

The primary `partial_score` is the pinned RULER-derived scorer: case-insensitive
substring recall, averaged over all gold answers for synthetic tasks and
maximized over accepted references for QA. `correct` is retained as an alias
for this score.

`strict_score` is a whole-question measure. Synthetic tasks receive 1 only
when every gold substring is recalled. This metric explicitly allows extra
text, does not enforce gold order, and does not establish key-to-value binding.
QA uses standard SQuAD-style normalized exact match.

`full_string_exact` compares the complete generated answer-token sequence after
lower-casing and punctuation/whitespace normalization. It preserves gold order
and rejects extra answer tokens. For QA it is max-over-reference.
`full_string_exact_and_eos` additionally requires a terminal configured EOS or
EOT token. This is the complete-output capability receipt and is never replaced
by substring recall.

QA also records standard max-over-reference normalized `qa_em` and token-overlap
`qa_f1`. Every row separately records `eos_seen`, `cap_hit`, `format_ok`, and
`validity_status`. A valid but wrong, unterminated, or cap-hit output remains a
valid scored row; an implementation or identity error must be marked invalid
rather than converted to score zero.
