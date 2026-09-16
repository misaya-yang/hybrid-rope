# Llama-3-70B NF4 S4/32K scale-transfer suite

This thin wrapper reuses the frozen Llama-3-8B prompts, tables, evaluator and
scorers with the prequantized `unsloth/llama-3-70b-Instruct-bnb-4bit`
checkpoint.  The vocabulary/tokenizer graph, chat template and BOS/EOS are
identical; only the 70B runtime declares a left-padding token, which is unused
by the formal batch-1 execution.

No benchmark assets are regenerated:

- Full-13 RULER uses `limit_per_cell=10` on the completed 13x200 panel;
- NIAH-8 is the retrieval-family view of those same 130 raw rows;
- PPL uses the first five frozen 32K documents;
- QA uses the existing complete Natural-QA631 panel.

Before this wrapper, run the existing `benchmark_prefill_chunks` utility with
the 70B path, the existing TailSpline table, one clean 32K row, the classic LM
array, chunks `0,8192`, and `max_new_tokens=4`.  A failed canary is
infrastructure evidence, not a model result.

```bash
cd /root/autodl-tmp/hybrid-rope
bash experiments/llama70b_scale_20260916/run_direct_reuse_suite.sh
bash experiments/llama70b_scale_20260916/run_direct_reuse_suite.sh --execute
```

Completion requires `complete.txt`, the Full-13 report, the QA report and the
PPL summary.  The archived 8B reports remain the exact-asset reference; the
70B-NF4 versus 8B-BF16 precision confound must remain explicit.  Shutdown and
browser-side GPU allocation are orchestration responsibilities outside this
runner.

