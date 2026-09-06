# Target-free retrofit evaluation command draft

This file is a command draft only. None of the commands below has been
executed in this preparation. They require an explicitly authorised GPU run,
the local model checkpoint, and a complete token manifest.

```bash
export DATA_ROOT="$TARGET_FREE_DATA_ROOT"
export TOKENIZER_ROOT="$DATA_ROOT/olmo2_tokenizer_config"
export TOKEN_MANIFEST="$DATA_ROOT/tokenized/olmo2_target_free/token_manifest.json"
export CHECKPOINT="$OLMO_CHECKPOINT_ROOT"
export OUT_ROOT="$DATA_ROOT/eval/target_free_20260822"
export RULER_DATA_ROOT="$DATA_ROOT/ruler_eval_core4"
```

## One-time data build after the GPU machine is started

```bash
python3 scripts/data_prep/target_free_context_builder.py \
  --download-root "$DATA_ROOT" \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --native-context-length "$NATIVE_CONTEXT_LENGTH" \
  --output "$DATA_ROOT/tokenized/olmo2_target_free" \
  --pg19-anchor-count "$PG19_ANCHOR_COUNT" \
  --allow-full-tokenization
```

The command uses the checkpoint's tokenizer files and chat template only; it
does not load checkpoint weights. `NATIVE_CONTEXT_LENGTH` and
`PG19_ANCHOR_COUNT` are explicit runtime inputs, not repository defaults.

## Formal four-arm matrix

The eventual formal evaluator must write independent cells for every task and
each multiplier (`1x`, `2x`, `4x`). It must not pool 8K/16K or merge task
families. The exact entrypoint is intentionally left as a future evaluator
owner; the command contract is frozen here.

```bash
# 1. Native: direct native rotary path; retention is the valid native-window control.
python3 scripts/eval/target_free_formal_eval.py \
  --checkpoint "$CHECKPOINT" --token-manifest "$TOKEN_MANIFEST" \
  --method native --tasks qasper narrativeqa multifieldqa_en hotpotqa 2wikimqa gov_report pg19 \
  --multipliers 1 2 4 --limit-per-cell 20 --output "$OUT_ROOT/native"

# 2. One official Transformers YaRN deployment, factor=4.
python3 scripts/eval/target_free_formal_eval.py \
  --checkpoint "$CHECKPOINT" --token-manifest "$TOKEN_MANIFEST" \
  --method official_yarn --factor 4 \
  --tasks qasper narrativeqa multifieldqa_en hotpotqa 2wikimqa gov_report pg19 \
  --multipliers 1 2 4 --limit-per-cell 20 --output "$OUT_ROOT/official_yarn_factor4"

# 3. Existing target-aware s2/s4 operators, explicitly labelled oracle ceiling.
python3 scripts/eval/target_free_formal_eval.py \
  --checkpoint "$CHECKPOINT" --token-manifest "$TOKEN_MANIFEST" \
  --method target_aware --factors 2 4 --label oracle_ceiling \
  --tasks qasper narrativeqa multifieldqa_en hotpotqa 2wikimqa gov_report pg19 \
  --multipliers 1 2 4 --limit-per-cell 20 --output "$OUT_ROOT/target_aware_oracle"

# 4. Target-free anchored operator: no L_target and no request-budget argument.
python3 scripts/eval/target_free_formal_eval.py \
  --checkpoint "$CHECKPOINT" --token-manifest "$TOKEN_MANIFEST" \
  --method target_free_anchored \
  --tasks qasper narrativeqa multifieldqa_en hotpotqa 2wikimqa gov_report pg19 \
  --multipliers 1 2 4 --limit-per-cell 20 --output "$OUT_ROOT/target_free_anchored"
```

Required formal endpoints are: PG-19 paired tail-NLL over the same final 512
tokens; single-document QA; multi-document QA; and GovReport summarization.
QA uses the fixed 64-token generation reserve. GovReport uses the official
LongBench generation budget from the pinned `dataset2maxlen.json`.

RULER is an engineering smoke check only and must be reported separately from
the formal endpoints:

```bash
for method in native official_yarn target_aware target_free_anchored; do
  python3 scripts/eval/target_free_ruler_smoke.py \
    --checkpoint "$CHECKPOINT" --data-root "$RULER_DATA_ROOT" \
    --method "$method" --lengths 8192 16384 --limit-per-cell 20 \
    --output "$OUT_ROOT/ruler_smoke/$method"
done
```
