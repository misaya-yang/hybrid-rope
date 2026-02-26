# Mid-Run Hypotheses (From Partial Signal)

## Observed pattern

- Retrieval-like tasks are strong (`passage_retrieval_en/zh` high running scores).
- Code similarity task (`lcc`) is still running and volatile.
- Prior runs showed tradeoff: some multi-hop QA drops while retrieval tasks rise.

## Hypothesis 1: Length-bucket sensitivity

- Cause: anchored schedule helps ultra-long retrieval but can hurt 4k-12k mixed reasoning windows.
- Test:
  - compute per-sample delta vs baseline by buckets `<=2k`, `2-4k`, `4-8k`, `8-16k`, `>16k`.
  - accept if gain is monotonic or at least non-negative for `>8k`.

## Hypothesis 2: Retrieval vs multi-hop tradeoff

- Cause: frequency remapping improves token lookup but degrades compositional aggregation.
- Test:
  - grouped metrics: `retrieval*` vs `hotpotqa/2wikimqa/musique`.
  - check whether retrieval gains outweigh multi-hop losses after FDR correction.

## Hypothesis 3: Decoding mismatch with baseline

- Cause: small decode mismatch (max_new/stop/template) can shift QA/coding metrics.
- Test:
  - rerun anchored with baseline-locked decode (`temperature=0`, same `max_new_tokens_policy`, same template path).
  - compare per-task deltas; if gaps shrink significantly, attribute to decode mismatch.

## Hypothesis 4: Prompt/template alignment drift

- Cause: slight formatting mismatch affects tasks requiring exact answer format.
- Test:
  - run strict parity audit (`prompt_source=official`, `chat_template=auto`, `truncate_mode=middle`).
  - inspect `template_leakage_rate`, `parse_fail_rate`, answer-format violations.

## Hypothesis 5: Train-set distribution mismatch

- Cause: WikiText-heavy fine-tuning may underrepresent instruction QA structures.
- Test:
  - controlled retrain with `LongAlpaca/LongQA + synthetic long QA`, keeping all hyperparams fixed.
  - compare only one factor at a time (dataset composition).
