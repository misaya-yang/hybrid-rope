#!/bin/bash
# Phase 7 runner — all experiments on RTX 5090
set -e

SWEEP=/root/autodl-tmp/hybrid-rope/scripts/m4_evq_sweep
WORK=/root/autodl-tmp/evq_phase7
cd $SWEEP

echo "============================================================"
echo "  PHASE 7 START  $(date)"
echo "============================================================"

# ================================================================
# 7A: YaRN Verification + Ablation (~3 min)
# ================================================================
echo ""
echo ">>> 7A: YaRN Verification"
python phase7a_yarn_verify.py

# ================================================================
# 7B: Multi-Seed τ=2.5 and τ=5.0 (~20 min)
# ================================================================
echo ""
echo ">>> 7B: Multi-Seed (τ=2.5,5.0 × seed=137,256)"
python run_evq_sweep.py --tier 125m --taus 2.5,5.0 --seeds 137,256 \
    --base 500000 --dataset fineweb-edu --seq_len 128 \
    --train_tokens 15000000 \
    --eval_lengths 128,256,512,1024,2048,4096,8192 \
    --work_dir $WORK/multiseed --resume

# ================================================================
# 7C: τ LR Sensitivity Ablation (~25 min)
# ================================================================
echo ""
echo ">>> 7C: τ LR Sensitivity"
for mult in 1 5 20 50 100; do
    echo "  lr_mult=$mult"
    python run_evq_sweep.py --tier 125m --learnable --tau_init 1.0 \
        --tau_lr_mult $mult --seeds 42 --base 500000 \
        --dataset fineweb-edu --seq_len 128 --train_tokens 15000000 \
        --eval_lengths 128,256,512,1024,2048,4096,8192 \
        --work_dir $WORK/tau_lr_sensitivity --resume
done

# ================================================================
# 7E: NTK-Aware Baseline (~20 min)
# ================================================================
echo ""
echo ">>> 7E: NTK-Aware Baseline"
# NTK-train FineWeb (base *= s^(dim/(dim-2)), s=64, dim=64)
python -c "
import math
s = 8192/128
dim = 64
base = 500000 * (s ** (dim / (dim - 2)))
print(f'NTK base = {base:.0f}')
"

# NTK-train: use scaled base with geometric RoPE
NTK_BASE=$(python3 -c "print(f'{500000 * (64 ** (64/62)):.0f}')")
echo "  NTK base = $NTK_BASE"

python run_evq_sweep.py --tier 125m --taus 0.0 --seeds 42 \
    --base $NTK_BASE --dataset fineweb-edu --seq_len 128 \
    --train_tokens 15000000 \
    --eval_lengths 128,256,512,1024,2048,4096,8192 \
    --work_dir $WORK/ntk_baseline --resume

python run_evq_sweep.py --tier 125m --taus 0.0 --seeds 42 \
    --base $NTK_BASE --dataset tinystories --seq_len 128 \
    --train_tokens 15000000 \
    --eval_lengths 128,256,512,1024,2048,4096,8192 \
    --work_dir $WORK/ntk_baseline_ts --resume

# NTK-infer: eval Geometric checkpoint with NTK frequencies
# (handled inline — load geo ckpt, replace inv_freq, eval)
python -c "
import sys, os, json, torch, math
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.insert(0, '$SWEEP')
from run_evq_sweep import GPT, evq_cosh_inv_freq, TIER_CONFIGS, DEVICE, DTYPE, eval_model

CFG = TIER_CONFIGS['125m'].copy()
CFG['seq_len'] = 128
CFG['max_position_embeddings'] = 128
BASE = 500000.0
s = 8192/128
ntk_base = BASE * (s ** (64/62))

for ds, val_p, ckpt_p in [
    ('fineweb', '/root/autodl-tmp/evq_128tok/val_fineweb-edu_5000000.pt',
     '/root/autodl-tmp/evq_128tok/125m_tau0.00_seed42/model.pt'),
]:
    print(f'\n  NTK-infer ({ds})')
    val = torch.load(val_p, weights_only=True)
    ntk_inv = 1.0 / (ntk_base ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    ntk_inv = ntk_inv.float()
    geo_inv = evq_cosh_inv_freq(64, 0.0, BASE)
    model = GPT(CFG, geo_inv).to(DEVICE)
    ckpt = torch.load(ckpt_p, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
    model.blocks[0].attn.rope.inv_freq.copy_(ntk_inv)
    model.blocks[0].attn.rope._build(128)
    ppl = eval_model(model, val, [128, 2048, 4096, 8192], 10)
    print(f'  NTK-infer {ds}: {ppl}')
    out = '/root/autodl-tmp/evq_phase7/ntk_baseline/ntk_infer_{}.json'.format(ds)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'method': 'NTK-infer', 'ppl': ppl, 'ntk_base': ntk_base}, f, indent=2)
    del model; torch.cuda.empty_cache()
"

# ================================================================
# 7D: PPL@128 Completion (~2 min, eval-only)
# ================================================================
echo ""
echo ">>> 7D: PPL@128 Completion"
python -c "
import sys, os, json, torch
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.insert(0, '$SWEEP')
from run_evq_sweep import GPT, evq_cosh_inv_freq, TIER_CONFIGS, DEVICE, eval_model

CFG = TIER_CONFIGS['125m'].copy()
CFG['seq_len'] = 128
CFG['max_position_embeddings'] = 128
BASE = 500000.0

val_fw = torch.load('/root/autodl-tmp/evq_128tok/val_fineweb-edu_5000000.pt', weights_only=True)

# Missing PPL@128 for tau=0.5, 2.0, 2.5 (from minisweep)
checkpoints = {
    'tau0.50': ('/root/autodl-tmp/evq_minisweep/125m_tau0.50_seed42/model.pt', 0.5),
    'tau2.00': ('/root/autodl-tmp/evq_minisweep/125m_tau2.00_seed42/model.pt', 2.0),
    'tau2.50': ('/root/autodl-tmp/evq_minisweep/125m_tau2.50_seed42/model.pt', 2.5),
}
results = {}
for name, (ckpt_path, tau) in checkpoints.items():
    inv = evq_cosh_inv_freq(64, tau, BASE)
    model = GPT(CFG, inv).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
    ppl = eval_model(model, val_fw, [128], 10)
    results[name] = {'tau': tau, 'ppl_128': ppl.get('128')}
    print(f'  {name}: PPL@128 = {ppl.get(\"128\")}')
    del model; torch.cuda.empty_cache()

out = '/root/autodl-tmp/evq_phase7/ppl128_completion/results.json'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print('  Saved:', out)
"

# ================================================================
# 7F: 350M Context Extension (512→2K) — ~3h
# ================================================================
echo ""
echo ">>> 7F: 350M Context Extension"
python phase7f_context_ext.py

echo ""
echo "============================================================"
echo "  PHASE 7 COMPLETE  $(date)"
echo "============================================================"
