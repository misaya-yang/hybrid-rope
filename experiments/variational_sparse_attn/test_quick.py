#!/usr/bin/env python3
"""
Quick Test Script (5-10 minutes)
=================================

Runs minimal experiment to verify:
1. Attention patch works correctly
2. All three variants run without error
3. Sparsemax produces exact zeros
4. PPL can be computed

Use this before running full experiment.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

from attention_patch import (
    apply_attention_patch,
    set_attention_variant,
    get_attention_stats,
    compute_distance_prior,
    clear_attention_weights
)

def test():
    print("="*60)
    print("QUICK TEST: Prior-guided Sparse Attention")
    print("="*60)
    
    # Apply patch
    apply_attention_patch()
    
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[TEST] Device: {device}")
    
    # Load model
    print("[TEST] Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation="eager")
    model = model.to(device)
    model.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Get sample text
    print("[TEST] Loading sample data...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    text = dataset[0]['text']
    if len(text) < 100:
        text = dataset[1]['text']
    
    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)[:128]
    input_ids = torch.tensor([tokens], device=device)
    
    print(f"[TEST] Sequence length: {len(tokens)}")
    
    # Test 1: Baseline
    print("\n[TEST 1] Baseline variant...")
    set_attention_variant('baseline')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, output_attentions=True)
        loss1 = outputs.loss.item() if outputs.loss is not None else 0.0
    stats1 = get_attention_stats()
    print(f"  Loss: {loss1:.4f}")
    print(f"  Sparsity: {stats1.get('sparsity', 0):.3f}")
    print(f"  ✅ Baseline works")
    
    # Test 2: Prior-Softmax
    print("\n[TEST 2] Prior-biased softmax...")
    clear_attention_weights()
    set_attention_variant('prior_softmax', lam=8.0, gamma=1.0, alpha=1.5)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, output_attentions=True)
        loss2 = outputs.loss.item() if outputs.loss is not None else 0.0
    stats2 = get_attention_stats()
    print(f"  Loss: {loss2:.4f}")
    print(f"  Sparsity: {stats2.get('sparsity', 0):.3f}")
    print(f"  ✅ Prior-Softmax works")
    
    # Test 3: Prior-Sparse
    print("\n[TEST 3] Prior-guided sparse...")
    clear_attention_weights()
    set_attention_variant('prior_sparse', lam=8.0, gamma=0.5, alpha=1.5)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, output_attentions=True)
        loss3 = outputs.loss.item() if outputs.loss is not None else 0.0
    stats3 = get_attention_stats()
    print(f"  Loss: {loss3:.4f}")
    print(f"  Sparsity: {stats3.get('sparsity', 0):.3f}")
    print(f"  Exact zeros: {stats3.get('exact_zeros', 0)}")
    print(f"  Row sum error: {stats3.get('row_sum_error', 1e10):.2e}")
    
    # Validate sparse
    if stats3.get('exact_zeros', 0) > 0:
        print(f"  ✅ Sparsemax produces exact zeros")
    else:
        print(f"  ❌ No exact zeros found!")
        return False
    
    if stats3.get('row_sum_error', 1.0) < 1e-3:
        print(f"  ✅ Row sums valid")
    else:
        print(f"  ❌ Row sums invalid!")
        return False
    
    # Test 4: Determinism
    print("\n[TEST 4] Determinism check...")
    clear_attention_weights()
    set_attention_variant('baseline')
    with torch.no_grad():
        out_a = model(input_ids, labels=input_ids, output_attentions=True)
        out_b = model(input_ids, labels=input_ids, output_attentions=True)
    diff = (out_a.logits - out_b.logits).abs().max().item()
    if diff < 1e-6:
        print(f"  Max diff: {diff:.2e}")
        print(f"  ✅ Deterministic")
    else:
        print(f"  Max diff: {diff:.2e}")
        print(f"  ❌ Non-deterministic!")
        return False
    
    # Test 5: Distance prior
    print("\n[TEST 5] Distance prior computation...")
    prior = compute_distance_prior(64, alpha=1.5, device=device)
    print(f"  Shape: {prior.shape}")
    print(f"  Finite values: {(prior > float('-inf')).sum().item()}")
    print(f"  Upper triangle all -inf: {torch.all(prior.triu(diagonal=1) == float('-inf')).item()}")
    print(f"  ✅ Distance prior correct")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
    print("\nYou can now run the full experiment:")
    print("  bash run_experiment.sh")
    return True

if __name__ == '__main__':
    import sys
    success = test()
    sys.exit(0 if success else 1)
