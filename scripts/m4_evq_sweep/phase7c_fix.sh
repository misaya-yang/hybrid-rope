#!/bin/bash
# Phase 7C Fix: τ LR sensitivity with separate work_dirs per lr_mult
set -e
SWEEP=/root/autodl-tmp/hybrid-rope/scripts/m4_evq_sweep
WORK=/root/autodl-tmp/evq_phase7/tau_lr_sensitivity
cd $SWEEP

echo ">>> 7C Fix: τ LR Sensitivity (separate work_dirs)"

# lr_mult=1 is already done in $WORK/results_checkpoint.json
# Run lr_mult=5,20,50,100 with separate work_dirs
for mult in 5 20 50 100; do
    echo ""
    echo "  lr_mult=$mult"
    python run_evq_sweep.py --tier 125m --learnable --tau_init 1.0 \
        --tau_lr_mult $mult --seeds 42 --base 500000 \
        --dataset fineweb-edu --seq_len 128 --train_tokens 15000000 \
        --eval_lengths 128,256,512,1024,2048,4096,8192 \
        --work_dir $WORK/lr_mult_$mult --resume
done

# Consolidate all results
echo ""
echo ">>> Consolidating 7C results..."
python3 -c "
import json, os, glob

results = {}

# lr_mult=1 from original run
orig = '$WORK/results_checkpoint.json'
if os.path.exists(orig):
    d = json.load(open(orig))
    for name, exp in d.get('experiments', {}).items():
        results['lr_mult_1'] = {
            'lr_mult': 1,
            'tau_final': exp.get('tau'),
            'ppl': exp.get('ppl'),
        }

# lr_mult=5,20,50,100 from separate dirs
for mult in [5, 20, 50, 100]:
    cp = f'$WORK/lr_mult_{mult}/results_checkpoint.json'
    if os.path.exists(cp):
        d = json.load(open(cp))
        for name, exp in d.get('experiments', {}).items():
            results[f'lr_mult_{mult}'] = {
                'lr_mult': mult,
                'tau_final': exp.get('tau'),
                'ppl': exp.get('ppl'),
            }

out = '$WORK/consolidated_results.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved: {out}')
print()
print(f\"{'lr_mult':>8}  {'tau_final':>10}  {'PPL@128':>8}  {'PPL@8K':>8}\")
print('-' * 45)
for k in sorted(results, key=lambda x: results[x]['lr_mult']):
    r = results[k]
    p = r.get('ppl', {})
    print(f\"{r['lr_mult']:>8}  {r['tau_final']:>10.4f}  {p.get('128','?'):>8}  {p.get('8192','?'):>8}\")
"

echo ">>> 7C Fix COMPLETE"
