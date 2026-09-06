#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

cd /root/autodl-tmp/phase21b

GEO_INIT=/root/autodl-tmp/evq_phase17c_2048_continue/seed42/geo_454m_1024_to_2048_continue/model.pt
EVQ_INIT=/root/autodl-tmp/evq_phase17c_2048_continue/seed42/evq1.41421_454m_1024_to_2048_continue/model.pt
QA_DATA=/root/autodl-tmp/datasets/scrolls_quality
BASE=/root/autodl-tmp/results/phase21b/quality_454m

# ════════════════════════════════════════════════════════════════
#  Step 0: Download QuALITY if missing
# ════════════════════════════════════════════════════════════════
if [ ! -f "$QA_DATA/quality/validation.jsonl" ]; then
  echo "============================================================"
  echo "  Downloading SCROLLS QuALITY dataset..."
  echo "============================================================"
  mkdir -p $QA_DATA/quality
  python -c "
from datasets import load_dataset
import json, os
out_dir = '$QA_DATA/quality'
for split in ['train', 'validation', 'test']:
    print(f'Downloading {split}...')
    ds = load_dataset('tau/scrolls', 'quality', split=split, trust_remote_code=True)
    path = os.path.join(out_dir, f'{split}.jsonl')
    with open(path, 'w') as f:
        for item in ds:
            f.write(json.dumps(item) + '\n')
    print(f'  Saved {len(ds)} samples to {path}')
"
  echo "  Download complete."
fi

# ════════════════════════════════════════════════════════════════
#  Step 1a: Finetune Geo on QuALITY @4K (2× training length)
# ════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  Step 1a: Finetune Geo on QuALITY @4K"
echo "============================================================"
if [ ! -f "$BASE/finetune/geo_seed42/model.pt" ]; then
  rm -rf $BASE/finetune/geo_seed42/
  python phase21b_scrolls_finetune.py \
    --init_ckpt $GEO_INIT \
    --rope geo --base 500000 \
    --task quality --seq_len 4096 --yarn 0 \
    --lr 1e-5 --steps 2000 --warmup 100 --dropout 0.1 \
    --micro_batch_size 1 --grad_accum 4 --seed 42 \
    --eval_every 2000 --eval_samples 200 \
    --data_dir $QA_DATA \
    --output_dir $BASE/finetune/geo_seed42/
else
  echo "  [SKIP] Geo finetune already exists"
fi

# ════════════════════════════════════════════════════════════════
#  Step 1b: Finetune EVQ on QuALITY @4K
# ════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  Step 1b: Finetune EVQ on QuALITY @4K"
echo "============================================================"

# Clean up GPU memory between runs
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()" 2>/dev/null

if [ ! -f "$BASE/finetune/evq_seed42/model.pt" ]; then
  rm -rf $BASE/finetune/evq_seed42/
  python phase21b_scrolls_finetune.py \
    --init_ckpt $EVQ_INIT \
    --rope evq --tau 1.41421 --base 500000 \
    --task quality --seq_len 4096 --yarn 0 \
    --lr 1e-5 --steps 2000 --warmup 100 --dropout 0.1 \
    --micro_batch_size 1 --grad_accum 4 --seed 42 \
    --eval_every 2000 --eval_samples 200 \
    --data_dir $QA_DATA \
    --output_dir $BASE/finetune/evq_seed42/
else
  echo "  [SKIP] EVQ finetune already exists"
fi

GEO_PT=$BASE/finetune/geo_seed42/model.pt
EVQ_PT=$BASE/finetune/evq_seed42/model.pt

# Verify models exist
if [ ! -f "$GEO_PT" ]; then
  echo "ERROR: Geo model not found at $GEO_PT"
  exit 1
fi
if [ ! -f "$EVQ_PT" ]; then
  echo "ERROR: EVQ model not found at $EVQ_PT"
  exit 1
fi

# ════════════════════════════════════════════════════════════════
#  Step 2: Distractor-padded eval at 4K/8K/16K/32K
# ════════════════════════════════════════════════════════════════

run_qa_eval() {
  local label=$1 model_pt=$2 rope=$3 tau=$4 tlen=$5 outdir=$6
  echo ""
  echo "============================================================"
  echo "  $label"
  echo "============================================================"

  # Skip if already done
  if [ -f "$outdir/result.json" ]; then
    echo "  [SKIP] Already completed"
    cat $outdir/result.json
    return 0
  fi

  EXTRA_ARGS=""
  if [ "$rope" = "evq" ]; then
    EXTRA_ARGS="--tau $tau"
  fi

  # Clear GPU cache between evals
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

  python phase21b_quality_eval.py \
    --model_pt $model_pt \
    --tier 454m --rope $rope --base 500000 \
    --target_len $tlen \
    --eval_samples 200 \
    --data_dir $QA_DATA \
    --output_dir $outdir \
    $EXTRA_ARGS
}

# @4K — in-distribution baseline
run_qa_eval "Eval [1/8] Geo @4K"   $GEO_PT geo 0       4096  $BASE/eval/4k/geo/
run_qa_eval "Eval [2/8] EVQ @4K"   $EVQ_PT evq 1.41421 4096  $BASE/eval/4k/evq/

# @8K — 2× extrapolation
run_qa_eval "Eval [3/8] Geo @8K"   $GEO_PT geo 0       8192  $BASE/eval/8k/geo/
run_qa_eval "Eval [4/8] EVQ @8K"   $EVQ_PT evq 1.41421 8192  $BASE/eval/8k/evq/

# @16K — 4× extrapolation
run_qa_eval "Eval [5/8] Geo @16K"  $GEO_PT geo 0       16384 $BASE/eval/16k/geo/
run_qa_eval "Eval [6/8] EVQ @16K"  $EVQ_PT evq 1.41421 16384 $BASE/eval/16k/evq/

# @32K — 8× extrapolation
run_qa_eval "Eval [7/8] Geo @32K"  $GEO_PT geo 0       32768 $BASE/eval/32k/geo/
run_qa_eval "Eval [8/8] EVQ @32K"  $EVQ_PT evq 1.41421 32768 $BASE/eval/32k/evq/

# ════════════════════════════════════════════════════════════════
#  Summary
# ════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  QuALITY QA (454M) — COMPLETE RESULTS"
echo "================================================================"
echo "Finetune results:"
echo "=== Geo ==="
cat $BASE/finetune/geo_seed42/result.json 2>/dev/null
echo ""
echo "=== EVQ ==="
cat $BASE/finetune/evq_seed42/result.json 2>/dev/null
echo ""
echo "Distractor-padded eval:"
for len in 4k 8k 16k 32k; do
  for cfg in geo evq; do
    echo "--- $len / $cfg ---"
    cat $BASE/eval/$len/$cfg/result.json 2>/dev/null || echo "  [FAILED]"
    echo ""
  done
done
