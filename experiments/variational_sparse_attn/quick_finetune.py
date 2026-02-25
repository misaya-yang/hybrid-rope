#!/usr/bin/env python3
"""
Quick Fine-tune Verification (5-10 minutes on M4)
=================================================
Lightweight script to verify sparse attention fine-tuning works.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import time

# Import our patch
import sys
sys.path.insert(0, '.')
from attention_patch_v2 import apply_attention_patch, set_attention_config, get_attention_stats

apply_attention_patch()

def quick_test():
    print("="*60)
    print("QUICK FINE-TUNE TEST (M4 Max)")
    print("="*60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[ENV] Device: {device}")
    
    # Load model with LoRA
    print("\n[1/5] Loading model...")
    model = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation="eager")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=8,  # Smaller rank for speed
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model = model.to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"      Trainable: {trainable:.1f}M parameters")
    
    # Prepare tiny dataset
    print("\n[2/5] Preparing data (500 samples)...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:500]')
    text = '\n\n'.join([item['text'] for item in dataset if len(item['text'].strip()) > 0])
    tokens = tokenizer.encode(text, add_special_tokens=False)[:50000]  # ~100 samples
    
    # Simple batch
    seq_len = 128
    batch_size = 2
    batches = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        batches.append({
            'input_ids': torch.tensor(tokens[i:i+seq_len]),
            'labels': torch.tensor(tokens[i:i+seq_len])
        })
        if len(batches) >= 50:  # Only 50 batches
            break
    
    print(f"      {len(batches)} batches created")
    
    # Optimizer
    print("\n[3/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Train for just a few steps
    print("\n[4/5] Training (10 steps)...")
    model.train()
    
    set_attention_config('prior_sparse', lam=0.01, gamma=2.0, prior_mode='centered')
    
    losses = []
    start = time.time()
    
    for step in range(min(10, len(batches))):
        batch = batches[step]
        input_ids = batch['input_ids'].unsqueeze(0).to(device)
        labels = batch['labels'].unsqueeze(0).to(device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        stats = get_attention_stats()
        
        if stats:
            print(f"  Step {step+1}/10: Loss={loss.item():.4f}, "
                  f"Sparsity={stats.get('sparsity_allowed', 0):.2%}, "
                  f"NNZ={stats.get('avg_nnz_allowed', 0):.1f}")
        else:
            print(f"  Step {step+1}/10: Loss={loss.item():.4f}")
    
    elapsed = time.time() - start
    print(f"\n      Training speed: {elapsed/10:.2f}s/step")
    
    # Evaluation
    print("\n[5/5] Evaluating...")
    model.eval()
    
    eval_losses = []
    with torch.no_grad():
        for i in range(5):  # 5 eval batches
            batch = batches[-(i+1)]  # Use last batches
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            labels = batch['labels'].unsqueeze(0).to(device)
            
            outputs = model(input_ids, labels=labels)
            eval_losses.append(outputs.loss.item())
    
    avg_train_loss = np.mean(losses)
    avg_eval_loss = np.mean(eval_losses)
    train_ppl = np.exp(avg_train_loss)
    eval_ppl = np.exp(avg_eval_loss)
    
    print(f"\n      Train Loss: {avg_train_loss:.4f}, PPL: {train_ppl:.2f}")
    print(f"      Eval Loss:  {avg_eval_loss:.4f}, PPL: {eval_ppl:.2f}")
    
    # Check if loss is decreasing
    if losses[-1] < losses[0]:
        print("\n✅ LOSS IS DECREASING - Training works!")
    else:
        print("\n⚠️  Loss not decreasing - may need more steps or lr tuning")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nReady for full training!")
    print("Run: python finetune_sparse.py --epochs 3 --method lora")

if __name__ == '__main__':
    quick_test()
