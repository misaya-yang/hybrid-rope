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
