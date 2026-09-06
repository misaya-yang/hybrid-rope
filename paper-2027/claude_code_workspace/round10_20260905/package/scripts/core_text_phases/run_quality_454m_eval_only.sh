#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

cd /root/autodl-tmp/phase21b

QA_DATA=/root/autodl-tmp/datasets/scrolls_quality
BASE=/root/autodl-tmp/results/phase21b/quality_454m
GEO_PT=$BASE/finetune/geo_seed42/model.pt
EVQ_PT=$BASE/finetune/evq_seed42/model.pt

# Clean ALL old eval results
rm -rf $BASE/eval/

run_qa_eval() {
  local label=$1 model_pt=$2 rope=$3 tau=$4 tlen=$5 outdir=$6
  echo ""
  echo "============================================================"
  echo "  $label"
  echo "============================================================"

  EXTRA_ARGS=""
  if [ "$rope" = "evq" ]; then
    EXTRA_ARGS="--tau $tau"
  fi

  python phase21b_quality_eval.py \
    --model_pt $model_pt \
    --tier 454m --rope $rope --base 500000 \
    --target_len $tlen \
    --eval_samples 200 \
    --data_dir $QA_DATA \
    --output_dir $outdir \
    $EXTRA_ARGS

  # Check if result saved
  if [ -f "$outdir/result.json" ]; then
    echo "  [OK] Result saved"
  else
    echo "  [FAILED] No result saved"
  fi
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

# Summary
echo ""
echo "================================================================"
echo "  QuALITY QA (454M) — COMPLETE EVAL RESULTS"
echo "================================================================"
for len in 4k 8k 16k 32k; do
  for cfg in geo evq; do
    echo "--- $len / $cfg ---"
    cat $BASE/eval/$len/$cfg/result.json 2>/dev/null || echo "  [FAILED]"
    echo ""
  done
done
