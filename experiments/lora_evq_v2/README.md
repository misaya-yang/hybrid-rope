# EVQ-Cosh LoRA v2

This package is a supporting LoRA experiment for LLaMA-3-8B style checkpoints. It is not part of the primary NeurIPS claim tier.

Set local paths with environment variables instead of editing scripts:

```bash
export EVQ_LORA_BASE_DIR=/path/to/local/work
export EVQ_LORA_MODEL=$EVQ_LORA_BASE_DIR/models/Meta-Llama-3-8B-Instruct
export EVQ_LORA_TRAIN_DATA=$EVQ_LORA_BASE_DIR/data/longalign_10k/longalign_10k.jsonl
export EVQ_LORA_WIKITEXT=$EVQ_LORA_BASE_DIR/data/wikitext2/wikitext2_test.txt

python download_model_data.py --verify_only
bash run.sh dryrun
```

The default local cache is `experiments/lora_evq_v2/local/`, which is excluded from reviewer archives.

## Clean positional-distillation pilot (prepared, not run)

The seed-42 A-stage pilot separates frequency-injection shock from LongAlign
content adaptation. It uses a frozen native-Geo view of LLaMA-3-8B as teacher,
an EVQ or Geo student, q/k-only LoRA (`r=64`, `alpha=128`), and no token labels.
The loss matches final hidden states in equal-weight 0-2K, 2-4K, and 4-8K
position buckets.

The operator entrypoint is:

```bash
export EVQ_LORA_BASE_DIR=/path/to/local/work
export EVQ_LORA_MODEL=$EVQ_LORA_BASE_DIR/models/Meta-Llama-3-8B-Instruct
export EVQ_LORA_WIKITEXT=$EVQ_LORA_BASE_DIR/data/wikitext2/wikitext2_test.txt
export EVQ_POSITIONAL_DISTILL_DIR=$EVQ_LORA_BASE_DIR/positional_distill_s42

bash scripts/2026-07/01_lora_positional_distill_seed42.sh prepare
bash scripts/2026-07/01_lora_positional_distill_seed42.sh train
bash scripts/2026-07/01_lora_positional_distill_seed42.sh eval
```

`prepare` freezes 2,400 train and 128 validation sequences from plain
FineWeb-Edu text. If the server cannot stream Hugging Face data, set
`EVQ_POSITIONAL_TEXT_JSONL` to a local JSONL whose every row has a plain `text`
field. Chat/instruction rows are rejected.

`train` creates matched `geo_distill_s42` and `evq_distill_s42` checkpoints.
The implementation keeps one 8B model in memory, switches to native Geo with
the adapter disabled for the teacher pass, then switches to the student
schedule for backward. Defaults remain batch 2 and gradient accumulation 4 to
use a 96GB card without changing the effective batch.

`eval` evaluates Base-Geo, Base-EVQ, Geo-Distill, and EVQ-Distill at
8K/16K/32K, measures held-out representation recovery, and writes
`positional_distill_summary.json`. The fixed gates are documented in
`docs/superpowers/specs/2026-07-10-llama8b-positional-distillation-design.md`.
Quick RULER is deliberately outside the A-stage pass/fail gates.

These commands have not been run as part of repository preparation. Do not
quote a result until the raw JSON, frozen-data manifest, checkpoint metadata,
GPU record, and seed scope have been verified.
