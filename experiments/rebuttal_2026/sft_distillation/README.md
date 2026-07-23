# Verified short-context SFT distillation data

This package prepares one shared short-context capability dataset for the
from-scratch Paper-Geo and EVQ-Cosh 1.5B models. It is a rebuttal-only
competence adaptation path: it teaches general extraction, state tracking,
aggregation, and multi-hop reasoning before an out-of-distribution official
RULER evaluation.

The program, not DeepSeek, creates the facts, solver, and unique answer.
DeepSeek may write only the fictional report framing, unrelated distractors,
transitions, and an equivalent question. Canonical fact sentences are inserted
after the API response and must survive byte-for-byte.

## Frozen distributions

Tasks:

- `information_retrieval`: 30%
- `relation_state_tracking`: 25%
- `aggregation_statistics`: 20%
- `multi_hop_qa`: 25%

Context lengths are measured with the pinned
`EleutherAI/gpt-neox-20b` tokenizer:

- `[512, 1024)`: 20%
- `[1024, 2048)`: 30%
- `[2048, 3072)`: 25%
- `[3072, 3990)`: 25%

The final upper bound leaves room for the question and answer inside a 4096
token SFT sequence.

Train, validation, and test use disjoint `world_id` and `template_family`
sets. The generator never reads RULER, NIAH/passkey, Paul Graham, SQuAD, or
HotpotQA data. It also rejects benchmark names and `A points to B`-style
language in generated samples.

## Environment

The API credential and model are read only from the process environment:

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_MODEL=...
```

Do not put either value in a command, config, shell history, or repository
file. `DEEPSEEK_MODEL` is checked against the authenticated `/models` response
before generation.

On macOS, a user who does not want to use a shell can double-click:

```text
setup_deepseek_env_macos.command
```

It asks for the credential in a hidden system prompt and registers both
variables with the current graphical login session through `launchctl`. It
does not write the credential to this repository, a dotfile, or command
history. New terminal processes can import the registered values without
placing the credential itself on the command line:

```bash
export DEEPSEEK_API_KEY="$(launchctl getenv DEEPSEEK_API_KEY)"
export DEEPSEEK_MODEL="$(launchctl getenv DEEPSEEK_MODEL)"
```

Monetary accounting is enabled by supplying the current rates separately:

```bash
export DEEPSEEK_INPUT_CACHE_HIT_USD_PER_MILLION=...
export DEEPSEEK_INPUT_CACHE_MISS_USD_PER_MILLION=...
export DEEPSEEK_OUTPUT_USD_PER_MILLION=...
```

Rates are deliberately not embedded in code because they are model- and
date-dependent. Token usage is always recorded even when monetary rates are
unset.

## Commands

Use an ignored output directory:

```bash
OUT="$PWD/outputs/rebuttal_sft_distillation"
```

Protocol-only dry run; no API request:

```bash
python -m experiments.rebuttal_2026.sft_distillation.generate \
  --action generate \
  --stage audit \
  --num-samples 100 \
  --output-dir "$OUT" \
  --local-files-only \
  --dry-run
```

Verify authentication, model availability, chat completion, and JSON output:

```bash
python -m experiments.rebuttal_2026.sft_distillation.generate \
  --action verify-api \
  --output-dir "$OUT"
```

Generate the 100-row audit set:

```bash
python -m experiments.rebuttal_2026.sft_distillation.generate \
  --action generate \
  --stage audit \
  --num-samples 100 \
  --concurrency 4 \
  --output-dir "$OUT" \
  --local-files-only
```

Interrupted work resumes without repaying cached successful requests:

```bash
python -m experiments.rebuttal_2026.sft_distillation.generate \
  --action generate \
  --stage audit \
  --num-samples 100 \
  --concurrency 4 \
  --output-dir "$OUT" \
  --local-files-only \
  --resume
```

The audit run writes `reports/audit_annotations_template.jsonl`. A human must
set all four booleans to `true` for all 100 rows. Approval is fail-closed:
`reports/audit_manual_review_queue.jsonl` contains the full context, question,
oracle answer, evidence, and structured facts for each row, so no database or
source-code lookup is needed during review.

```bash
python -m experiments.rebuttal_2026.sft_distillation.generate \
  --action approve-audit \
  --output-dir "$OUT" \
  --annotations "$OUT/reports/audit_annotations_completed.jsonl" \
  --reviewer-id internal-audit
```

Only then can the exact 3000/400/400 pilot run:

```bash
python -m experiments.rebuttal_2026.sft_distillation.generate \
  --action generate \
  --stage pilot \
  --num-samples 3800 \
  --concurrency 4 \
  --output-dir "$OUT" \
  --local-files-only \
  --resume
```

## Outputs

```text
raw/audit.jsonl
raw/train.jsonl
raw/validation.jsonl
raw/test.jsonl
messages/audit.jsonl
messages/train.jsonl
messages/validation.jsonl
messages/test.jsonl
reports/audit_report.json
reports/pilot_audit_report.json
reports/*_manual_review_queue.jsonl
audit_dataset_manifest.json
audit_gate.json
dataset_manifest.json
work/state.sqlite3
cache/
```

`raw/*.jsonl` retains facts, oracle, evidence, teacher metadata, hashes, token
counts, and validation receipts. `messages/*.jsonl` contains only:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

The final manifest binds Paper-Geo and EVQ-Cosh to the same dataset identity
and identical train-file hash/order.

## Full-parameter SFT handoff

This repository does not yet contain the authorized 1.5B full-parameter SFT
trainer. Do not silently reuse the historical 8B LoRA trainer. Once the 1.5B
entrypoint exists, both arms must receive the identical arguments below; only
their input checkpoint/frequency identity may differ:

```bash
COMMON_DATA_ARGS=(
  --train-jsonl "$OUT/messages/train.jsonl"
  --validation-jsonl "$OUT/messages/validation.jsonl"
  --dataset-manifest "$OUT/dataset_manifest.json"
  --sequence-length 4096
  --full-parameter-sft
)

python "$FULL_SFT_ENTRYPOINT" \
  --input-checkpoint "$PAPER_GEO_CHECKPOINT" \
  --run-name paper_geo_sft \
  "${COMMON_DATA_ARGS[@]}"

python "$FULL_SFT_ENTRYPOINT" \
  --input-checkpoint "$EVQ_COSH_CHECKPOINT" \
  --run-name evq_cosh_sft \
  "${COMMON_DATA_ARGS[@]}"
```

Before either launch, the trainer must record the final
`dataset_identity_sha256`, `messages/train.jsonl` SHA-256, and ordered sample
count from `dataset_manifest.json`.
