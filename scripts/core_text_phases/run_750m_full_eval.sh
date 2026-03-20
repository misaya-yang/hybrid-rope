#!/bin/bash
# ============================================================
# 750M Full Evaluation Pipeline: GEO vs EVQ
# ============================================================
# Evaluates both models on:
#   1. PPL at 4K/8K/16K/32K
#   2. Passkey NLL gap (multi-length, multi-depth)
#   3. Multi-needle NIAH
#   4. YaRN overlay (no fine-tune) at 8K/16K/32K
#   5. YaRN overlay (400-step fine-tune) at 8K/16K/32K
#   6. LongBench 13 tasks NLL at 4K/8K
# ============================================================

set -euo pipefail
PYTHON=/root/miniconda3/bin/python
WORK=/root/autodl-tmp/eval_750m_final
SCRIPT_DIR=/root/autodl-tmp/scripts/core_text_phases
SWEEP_SCRIPT=$SCRIPT_DIR/run_evq_sweep.py
LONGBENCH_SCRIPT=$SCRIPT_DIR/eval_longbench_nll.py
PASSKEY_SCRIPT=$SCRIPT_DIR/eval_passkey_scratch.py

# Model checkpoints
EVQ_DIR=/root/autodl-tmp/evq_750m_tau1p0_seed88_tok1470m_pk2p_bs7/750m_tau1.00_seed88
GEO_DIR=/root/autodl-tmp/evq_750m_tau1p0_seed88_tok1470m_pk2p_bs7/750m_tau0.00_seed88
EVQ_MODEL=$EVQ_DIR/model.pt
GEO_MODEL=$GEO_DIR/model.pt
EVQ_INVFREQ=$EVQ_DIR/inv_freq.npy
GEO_INVFREQ=$GEO_DIR/inv_freq.npy

# Val data
VAL_DATA=/root/autodl-tmp/evq_750m_clean/val_fineweb-edu_5000000.pt

mkdir -p "$WORK"

echo "============================================================"
echo "  750M Full Evaluation: GEO vs EVQ"
echo "  $(date)"
echo "============================================================"

# Verify checkpoints exist
for f in "$EVQ_MODEL" "$GEO_MODEL"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Checkpoint not found: $f"
        exit 1
    fi
done
echo "Checkpoints OK."

# ──────────────────────────────────────────────────────────────
# Phase 1: PPL at 4K/8K/16K/32K (both models)
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Phase 1/6: PPL Evaluation"
for TAG in geo evq; do
    if [ "$TAG" = "geo" ]; then MODEL=$GEO_MODEL; INV=$GEO_INVFREQ; else MODEL=$EVQ_MODEL; INV=$EVQ_INVFREQ; fi
    echo "  [$TAG] PPL eval..."
    $PYTHON -u -c "
import torch, numpy as np, math, sys, json
sys.path.insert(0, '$SCRIPT_DIR')
from run_evq_sweep import GPT, TIER_CONFIGS, get_device_and_dtype, eval_model

DEVICE, DTYPE = get_device_and_dtype()
cfg = TIER_CONFIGS['750m'].copy()
inv_freq = torch.from_numpy(np.load('$INV')).float()
model = GPT(cfg, inv_freq).to(DEVICE)
model.load_state_dict(torch.load('$MODEL', map_location=DEVICE, weights_only=True))
val = torch.load('$VAL_DATA', weights_only=True)

# Eval at multiple lengths
ppl = eval_model(model, val, [4096, 8192, 16384, 32768], eval_chunks=10)
print(json.dumps(ppl, indent=2))
with open('$WORK/ppl_${TAG}.json', 'w') as f:
    json.dump({'model': '$TAG', 'ppl': ppl}, f, indent=2)
print('  [$TAG] PPL saved.')
del model; torch.cuda.empty_cache()
" 2>&1 | tee "$WORK/ppl_${TAG}.log"
done

# ──────────────────────────────────────────────────────────────
# Phase 2: Passkey NLL gap
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Phase 2/6: Passkey NLL Gap"
for TAG in geo evq; do
    if [ "$TAG" = "geo" ]; then MODEL=$GEO_MODEL; INV=$GEO_INVFREQ; else MODEL=$EVQ_MODEL; INV=$EVQ_INVFREQ; fi
    echo "  [$TAG] Passkey eval..."
    $PYTHON -u -c "
import torch, numpy as np, json, sys
sys.path.insert(0, '$SCRIPT_DIR')
from run_evq_sweep import GPT, TIER_CONFIGS, get_device_and_dtype
from eval_passkey_scratch import eval_passkey_nll_gap
from transformers import AutoTokenizer

DEVICE, DTYPE = get_device_and_dtype()
cfg = TIER_CONFIGS['750m'].copy()
inv_freq = torch.from_numpy(np.load('$INV')).float()
model = GPT(cfg, inv_freq).to(DEVICE)
model.load_state_dict(torch.load('$MODEL', map_location=DEVICE, weights_only=True))
model.eval()

tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
val = torch.load('$VAL_DATA', weights_only=True)

results = eval_passkey_nll_gap(
    model, tok, val,
    lengths=[4096, 8192, 16384],
    depths=[0.10, 0.20, 0.50, 0.80, 0.90],
    num_trials=10,
)
g = results.get('global', {})
print(f'  Retrieval: {g.get(\"retrieval_rate\", \"?\")}, AR: {g.get(\"ar_exact_match_rate\", \"?\")}')
with open('$WORK/passkey_${TAG}.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
del model; torch.cuda.empty_cache()
" 2>&1 | tee "$WORK/passkey_${TAG}.log"
done

# ──────────────────────────────────────────────────────────────
# Phase 3: Multi-needle NIAH
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Phase 3/6: Multi-needle NIAH"
for TAG in geo evq; do
    if [ "$TAG" = "geo" ]; then MODEL=$GEO_MODEL; INV=$GEO_INVFREQ; else MODEL=$EVQ_MODEL; INV=$EVQ_INVFREQ; fi
    echo "  [$TAG] Multi-needle eval..."
    $PYTHON -u -c "
import torch, numpy as np, json, sys
sys.path.insert(0, '$SCRIPT_DIR')
from run_evq_sweep import GPT, TIER_CONFIGS, get_device_and_dtype
from eval_multi_needle import eval_multi_needle_passkey
from transformers import AutoTokenizer

DEVICE, DTYPE = get_device_and_dtype()
cfg = TIER_CONFIGS['750m'].copy()
inv_freq = torch.from_numpy(np.load('$INV')).float()
model = GPT(cfg, inv_freq).to(DEVICE)
model.load_state_dict(torch.load('$MODEL', map_location=DEVICE, weights_only=True))
model.eval()

tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
val = torch.load('$VAL_DATA', weights_only=True)

results = eval_multi_needle_passkey(
    model, tok, val,
    lengths=[4096, 8192],
    n_needles=5,
    num_trials=10,
)
print(json.dumps({k: v for k, v in results.items() if k != 'details'}, indent=2, default=str))
with open('$WORK/multineedle_${TAG}.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
del model; torch.cuda.empty_cache()
" 2>&1 | tee "$WORK/multineedle_${TAG}.log"
done

# ──────────────────────────────────────────────────────────────
# Phase 4: YaRN overlay (NO fine-tune) — inference-time only
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Phase 4/6: YaRN Overlay (no fine-tune)"
for TAG in geo evq; do
    if [ "$TAG" = "geo" ]; then MODEL=$GEO_MODEL; INV=$GEO_INVFREQ; else MODEL=$EVQ_MODEL; INV=$EVQ_INVFREQ; fi
    echo "  [$TAG] YaRN inference-only eval..."
    $PYTHON -u -c "
import torch, numpy as np, math, json, sys
sys.path.insert(0, '$SCRIPT_DIR')
from run_evq_sweep import GPT, TIER_CONFIGS, get_device_and_dtype, eval_model

DEVICE, DTYPE = get_device_and_dtype()
cfg = TIER_CONFIGS['750m'].copy()
inv_freq_orig = torch.from_numpy(np.load('$INV')).float()
model = GPT(cfg, inv_freq_orig).to(DEVICE)
model.load_state_dict(torch.load('$MODEL', map_location=DEVICE, weights_only=True))
model.eval()
val = torch.load('$VAL_DATA', weights_only=True)

# YaRN frequency modification
def yarn_inv_freq(base_inv, scale):
    K = len(base_inv)
    idx = torch.arange(K, dtype=torch.float64)
    start, end = int(0.20 * K), int(0.90 * K)
    ramp = torch.zeros(K, dtype=torch.float64)
    ramp[end:] = 1.0
    if end > start:
        r = (idx[start:end] - start).float() / (end - start)
        ramp[start:end] = (r * r * (3.0 - 2.0 * r)).double()
    temperature = 1.0 + 0.07 * math.log2(scale) if scale > 1 else 1.0
    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    return (base_inv.double() / yarn_scale).float()

results = {}
for scale in [2, 4, 8]:
    target_L = 4096 * scale
    y_inv = yarn_inv_freq(inv_freq_orig, scale)
    model.blocks[0].attn.rope.inv_freq.copy_(y_inv)
    model.blocks[0].attn.rope._build(target_L + 100)
    ppl = eval_model(model, val, [target_L], eval_chunks=10)
    results[f'yarn_s{scale}_L{target_L}'] = ppl
    print(f'  YaRN scale={scale} L={target_L}: PPL={ppl}')

# Restore original
model.blocks[0].attn.rope.inv_freq.copy_(inv_freq_orig)
model.blocks[0].attn.rope._build(cfg['max_position_embeddings'])

with open('$WORK/yarn_noft_${TAG}.json', 'w') as f:
    json.dump({'model': '$TAG', 'yarn_no_finetune': results}, f, indent=2)
del model; torch.cuda.empty_cache()
" 2>&1 | tee "$WORK/yarn_noft_${TAG}.log"
done

# ──────────────────────────────────────────────────────────────
# Phase 5: YaRN + 400-step fine-tune → eval
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Phase 5/6: YaRN + 400-step Fine-tune"
for TAG in geo evq; do
    if [ "$TAG" = "geo" ]; then MODEL=$GEO_MODEL; INV=$GEO_INVFREQ; else MODEL=$EVQ_MODEL; INV=$EVQ_INVFREQ; fi
    echo "  [$TAG] YaRN fine-tune (400 steps, scale=2, target=8K)..."
    $PYTHON -u -c "
import torch, numpy as np, math, json, sys, time
sys.path.insert(0, '$SCRIPT_DIR')
from run_evq_sweep import GPT, TIER_CONFIGS, get_device_and_dtype, eval_model
from eval_passkey_scratch import eval_passkey_nll_gap
from transformers import AutoTokenizer
from contextlib import nullcontext

DEVICE, DTYPE = get_device_and_dtype()
USE_AUTOCAST = DEVICE == 'cuda' and DTYPE != torch.float32
cfg = TIER_CONFIGS['750m'].copy()
inv_freq_orig = torch.from_numpy(np.load('$INV')).float()
tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
val = torch.load('$VAL_DATA', weights_only=True)

# YaRN inv_freq
def yarn_inv_freq(base_inv, scale):
    K = len(base_inv)
    idx = torch.arange(K, dtype=torch.float64)
    start, end = int(0.20 * K), int(0.90 * K)
    ramp = torch.zeros(K, dtype=torch.float64)
    ramp[end:] = 1.0
    if end > start:
        r = (idx[start:end] - start).float() / (end - start)
        ramp[start:end] = (r * r * (3.0 - 2.0 * r)).double()
    temperature = 1.0 + 0.07 * math.log2(scale) if scale > 1 else 1.0
    yarn_scale = (scale ** ramp) * (temperature ** (0.5 * ramp))
    return (base_inv.double() / yarn_scale).float()

# Fine-tune data: use 8K mixed data
ft_data_path = '/root/autodl-tmp/data/8k_mixed/train_slimpajama-mixed_500000000_8192.pt'
ft_data = torch.load(ft_data_path, weights_only=True)[:2000]  # 2000 chunks enough for 400 steps
print(f'  Fine-tune data: {ft_data.shape}')

for scale, target_L, ft_steps in [(2, 8192, 400), (4, 16384, 400)]:
    print(f'\\n  === YaRN scale={scale} target={target_L} ft_steps={ft_steps} ===')

    y_inv = yarn_inv_freq(inv_freq_orig, scale)

    # Build model with YaRN frequencies
    cfg_ft = cfg.copy()
    cfg_ft['max_position_embeddings'] = target_L
    model = GPT(cfg_ft, y_inv).to(DEVICE)
    model.load_state_dict(torch.load('$MODEL', map_location=DEVICE, weights_only=True))

    # Fine-tune: 400 steps, lr=2e-5, AdamW
    # YaRN official: lr=2e-5, AdamW, NO weight decay, 20-step warmup
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.95), weight_decay=0.0)
    WARMUP = 20
    model.train()

    if target_L == 8192:
        data_chunks = ft_data  # already 8192
    else:
        # For 16K: concatenate pairs of 8K chunks
        n_pairs = ft_data.shape[0] // 2
        data_chunks = ft_data[:n_pairs*2].view(n_pairs, target_L)

    t0 = time.time()
    base_lr = 2e-5
    for step in range(ft_steps):
        # Linear warmup (20 steps per YaRN paper)
        if step < WARMUP:
            cur_lr = base_lr * step / max(WARMUP, 1)
        else:
            cur_lr = base_lr
        for g in opt.param_groups:
            g['lr'] = cur_lr

        idx = step % data_chunks.shape[0]
        batch = data_chunks[idx:idx+1].to(DEVICE)
        ctx = torch.amp.autocast('cuda', dtype=DTYPE) if USE_AUTOCAST else nullcontext()
        with ctx:
            logits = model(batch[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f'    step {step}/{ft_steps} loss={loss.item():.4f} lr={cur_lr:.2e}')

    ft_time = time.time() - t0
    print(f'  Fine-tune done in {ft_time/60:.1f} min')

    # Eval PPL
    model.eval()
    ppl = eval_model(model, val, [4096, target_L], eval_chunks=10)
    print(f'  PPL after YaRN+FT: {ppl}')

    # Eval passkey
    pk = eval_passkey_nll_gap(model, tok, val,
        lengths=[4096, target_L] if target_L <= 16384 else [4096, 8192],
        depths=[0.10, 0.50, 0.90], num_trials=10)
    g = pk.get('global', {})
    print(f'  Passkey: ret={g.get(\"retrieval_rate\")}, AR={g.get(\"ar_exact_match_rate\")}')

    # Save
    tag_s = f'yarn_ft_s{scale}_L{target_L}'
    with open(f'$WORK/{tag_s}_${TAG}.json', 'w') as f:
        json.dump({
            'model': '$TAG', 'scale': scale, 'target_L': target_L,
            'ft_steps': ft_steps, 'ft_time_sec': round(ft_time, 1),
            'ppl': ppl, 'passkey_global': g
        }, f, indent=2, default=str)

    del model, opt
    torch.cuda.empty_cache()
" 2>&1 | tee "$WORK/yarn_ft_${TAG}.log"
done

# ──────────────────────────────────────────────────────────────
# Phase 6: LongBench 13 tasks NLL
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Phase 6/6: LongBench NLL (qa6 + sum4 = 10 tasks)"
for TAG in geo evq; do
    if [ "$TAG" = "geo" ]; then MODEL=$GEO_MODEL; INV=$GEO_INVFREQ; ROPE=geo; else MODEL=$EVQ_MODEL; INV=$EVQ_INVFREQ; ROPE=hybrid; fi
    echo "  [$TAG] LongBench NLL eval..."
    $PYTHON -u "$LONGBENCH_SCRIPT" \
        --model_path "$MODEL" \
        --tier 750m \
        --rope_type "$ROPE" \
        --inv_freq_path "$INV" \
        --tasks all \
        --max_context_len 8192 \
        --method_name "${TAG}_750m" \
        --output_dir "$WORK/longbench_${TAG}" \
        2>&1 | tee "$WORK/longbench_${TAG}.log"
done

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ALL EVALUATIONS COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Results in: $WORK/"
ls -la "$WORK/"*.json 2>/dev/null | wc -l
echo " JSON result files generated."
