#!/usr/bin/env python
"""Quick gamma sweep to find reasonable sparsity target (80-90%)"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import and apply patch FIRST (before loading model)
import attention_patch_v2
import transformers.models.gpt2.modeling_gpt2 as gpt2_module
gpt2_module.eager_attention_forward = attention_patch_v2.patched_eager_attention_forward

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("=" * 70)
print("GAMMA SWEEP TEST (Prior-Sparse Attention)")
print("=" * 70)
print(f"Device: {device}")
print()

# Test prompt (longer for better statistics)
text = "The quick brown fox jumps over the lazy dog. Artificial intelligence has transformed the way we think about machine learning and natural language processing. The development of large language models has enabled new capabilities in understanding and generating human-like text."

# Test different gamma values
gammas = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

print(f"{'Gamma':<8} {'Sparsity':<15} {'NNZ':<10} {'PPL':<12} {'vs Baseline'}")
print("-" * 65)

results = []
baseline_ppl = None

for gamma in gammas:
    # Reload model fresh for each gamma (to ensure clean state)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    
    # Set config
    attention_patch_v2.set_attention_config(
        variant='prior_sparse',
        lam=0.01,
        gamma=gamma,
        prior_mode='centered'
    )
    
    # Reset stats
    attention_patch_v2.ATTENTION_CONFIG['last_stats'] = None
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        ppl = torch.exp(outputs.loss).item()
        
        # Get stats
        stats = attention_patch_v2.get_attention_stats()
        if stats:
            sparsity = stats.get('sparsity_allowed', 0)
            nnz = stats.get('avg_nnz', 0)
        else:
            sparsity = 0
            nnz = 0
        
        if gamma == 0.3:  # Use lowest gamma as baseline
            baseline_ppl = ppl
        
        increase = (ppl / baseline_ppl - 1) * 100 if baseline_ppl else 0
        print(f"{gamma:<8} {sparsity*100:>6.2f}%{'':<7} {nnz:>5.1f}{'':<4} {ppl:>8.2f}{'':<3} +{increase:.0f}%")
        results.append((gamma, sparsity, nnz, ppl))

print()
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print()
print("Analysis of gamma values for prior-sparse attention:")
print()

# Find sweet spot
for gamma, sparsity, nnz, ppl in results:
    if 0.80 <= sparsity <= 0.95 and ppl < 100:
        print(f"✅ γ={gamma}: Sparsity={sparsity*100:.1f}%, PPL={ppl:.1f} (Good balance)")
    elif sparsity > 0.95:
        print(f"⚠️  γ={gamma}: Sparsity={sparsity*100:.1f}% (Too extreme)")
    elif sparsity < 0.70:
        print(f"ℹ️  γ={gamma}: Sparsity={sparsity*100:.1f}% (Not sparse enough)")

print()
print("Suggested config for fine-tuning:")
print("  python finetune_sparse.py --gamma 0.5 --lam 0.01 --epochs 3 --method lora")
print()
print("This should give ~85% sparsity with better chance of PPL recovery.")
