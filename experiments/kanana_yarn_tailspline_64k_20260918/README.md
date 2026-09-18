# Kanana 64K: official runtime YaRN vs TailSpline S=2

This is a frozen, paired zero-training runtime comparison on
`kakaocorp/kanana-1.5-8b-instruct-2505` at 64K.

- Official arm: Transformers YaRN with factor 4.4, original length 32768,
  beta-fast 64 and beta-slow 2, matching the model's published runtime recipe.
- Candidate arm: exact finite-grid TailSpline, S=2, canonical public 32-turn to
  1-turn transition, with the fixed `1 + 0.1 ln 2` gain.
- Pilot: `niah_multiquery` and `vt`, ten source-order unpadded prompts per task.
- If the absolute task-equal pilot difference is at most 10 percentage points,
  run only the remaining eleven tasks and merge them with the pilot into
  Full-13 x 10. A directionally larger pilot stops for inspection.
- The two arms always use the same prompts, tokenizer, decoder and scorer.

CPU preparation:

```bash
bash experiments/kanana_yarn_tailspline_64k_20260918/prepare_parallel_server.sh
```

The server wrapper uses at most twelve single-threaded CPU workers. It generates
the thirteen upstream task sources in parallel, then performs one shared
tokenization/conversion pass and freezes both runtime tables.

GPU execution is dry-run by default:

```bash
bash experiments/kanana_yarn_tailspline_64k_20260918/run_server.sh
bash experiments/kanana_yarn_tailspline_64k_20260918/run_server.sh --execute
```

To insert a same-prompt canonical MrRoPE-Pro S=2 pilot before resuming the
YaRN/TailSpline Full-13 continuation:

```bash
bash experiments/kanana_yarn_tailspline_64k_20260918/run_mrpro_then_resume.sh
```
