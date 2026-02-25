#!/usr/bin/env python3
"""
Fine-tuning Framework for Prior-Guided Sparse Attention
=========================================================

Supports:
- Full fine-tuning (GPT-2 Small/Medium, ~2-10GB)
- LoRA fine-tuning (GPT-2 Large/XL, ~10-30GB with LoRA)
- Sparse-aware training with gradient propagation through sparsemax
- Mixed precision (bf16/fp16) for speed
- Gradient checkpointing for memory efficiency

Usage:
    # M4 Max (36GB) - GPT-2 Small full fine-tune
    python finetune_sparse.py --model gpt2 --method full --epochs 3
    
    # A100 (40/80GB) - GPT-2 XL with LoRA
    python finetune_sparse.py --model gpt2-xl --method lora --sparse_gamma 0.5
    
    # Ablation: train dense baseline for comparison
    python finetune_sparse.py --model gpt2 --method full --no_sparse
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    get_linear_schedule_with_warmup,
    default_data_collator
)
from datasets import load_dataset

# Import our sparse attention patch
from attention_patch_v2 import (
    apply_attention_patch,
    set_attention_config,
    get_attention_stats,
    clear_attention_state,
    ATTENTION_CONFIG
)

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARN] PEFT not installed. LoRA mode unavailable.")
    print("  Install: pip install peft")

# Apply patch immediately
apply_attention_patch()


class WikiTextDataset(Dataset):
    """WikiText-2 dataset for language modeling."""
    
    def __init__(self, tokenizer, split='train', max_length=512, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset
        print(f"[DATA] Loading WikiText-2 ({split})...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        
        # Concatenate all text
        full_text = '\n\n'.join([item['text'] for item in dataset if len(item['text'].strip()) > 0])
        
        # Tokenize
        print(f"[DATA] Tokenizing...")
        self.tokens = tokenizer.encode(full_text, add_special_tokens=False)
        
        if max_samples:
            self.tokens = self.tokens[:max_samples * max_length]
        
        print(f"[DATA] Total tokens: {len(self.tokens):,}")
        print(f"[DATA] Num sequences: {len(self.tokens) // max_length:,}")
    
    def __len__(self):
        return max(1, len(self.tokens) // self.max_length)
    
    def __getitem__(self, idx):
        start = idx * self.max_length
        end = start + self.max_length
        
        input_ids = self.tokens[start:end]
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(input_ids, dtype=torch.long),
        }


def setup_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    method: str = 'full',
    lora_rank: int = 16,
    lora_alpha: int = 32,
    use_gradient_checkpointing: bool = True
) -> Tuple[nn.Module, GPT2Tokenizer]:
    """Setup model with optional LoRA."""
    
    print(f"[MODEL] Loading {model_name}...")
    
    # Load base model
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        attn_implementation="eager",  # Required for our patch
        torch_dtype=torch.float32,  # Start with fp32, we'll cast later
    )
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA if requested
    if method == 'lora':
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT not installed. Run: pip install peft")
        
        print(f"[LoRA] Applying LoRA (r={lora_rank}, alpha={lora_alpha})...")
        
        # Target attention layers for sparse attention adaptation
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["c_attn", "c_proj"],  # GPT-2 attention layers
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Gradient checkpointing for memory efficiency
    if use_gradient_checkpointing:
        print("[OPT] Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[MODEL] Trainable params: {param_count:.1f}M / {total_params:.1f}M")
    
    return model, tokenizer


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    variant: str,
    lam: float,
    gamma: float,
    prior_mode: str,
    use_amp: bool = False,
    max_grad_norm: float = 1.0,
    log_interval: int = 50,
) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    
    # Setup AMP if requested
    scaler = GradScaler() if use_amp else None
    
    total_loss = 0.0
    total_tokens = 0
    sparse_stats = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Set attention config for this forward pass
        set_attention_config(
            variant=variant,
            lam=lam,
            gamma=gamma,
            alpha=1.5,
            prior_mode=prior_mode,
            clear_weights=True
        )
        
        # Forward pass with AMP
        if use_amp and device.type == 'cuda':
            with autocast():
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
        else:
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Optimizer step
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        scheduler.step()
        optimizer.zero_grad()
        
        # Stats
        batch_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        
        # Collect sparse stats
        stats = get_attention_stats()
        if stats:
            sparse_stats.append({
                'sparsity': stats.get('sparsity_allowed', 0),
                'nnz': stats.get('avg_nnz_allowed', 0),
            })
        
        # Logging
        if step % log_interval == 0:
            avg_loss = total_loss / max(1, total_tokens)
            ppl = np.exp(avg_loss)
            lr = scheduler.get_last_lr()[0]
            
            desc = f"Epoch {epoch} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}"
            if sparse_stats:
                recent_sparsity = np.mean([s['sparsity'] for s in sparse_stats[-10:]])
                recent_nnz = np.mean([s['nnz'] for s in sparse_stats[-10:]])
                desc += f" | Sparsity: {recent_sparsity:.2%} | NNZ: {recent_nnz:.1f}"
            
            pbar.set_description(desc)
    
    # Epoch stats
    avg_loss = total_loss / max(1, total_tokens)
    avg_ppl = np.exp(avg_loss)
    
    result = {
        'loss': avg_loss,
        'ppl': avg_ppl,
        'tokens': total_tokens,
    }
    
    if sparse_stats:
        result['avg_sparsity'] = np.mean([s['sparsity'] for s in sparse_stats])
        result['avg_nnz'] = np.mean([s['nnz'] for s in sparse_stats])
    
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    variant: str,
    lam: float,
    gamma: float,
    prior_mode: str,
    desc: str = "Eval"
) -> Dict[str, float]:
    """Evaluate model."""
    
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    sparse_stats = []
    
    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        set_attention_config(
            variant=variant,
            lam=lam,
            gamma=gamma,
            alpha=1.5,
            prior_mode=prior_mode,
            clear_weights=True
        )
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        batch_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        
        stats = get_attention_stats()
        if stats:
            sparse_stats.append({
                'sparsity': stats.get('sparsity_allowed', 0),
                'nnz': stats.get('avg_nnz_allowed', 0),
            })
    
    avg_loss = total_loss / max(1, total_tokens)
    avg_ppl = np.exp(avg_loss)
    
    result = {
        'loss': avg_loss,
        'ppl': avg_ppl,
    }
    
    if sparse_stats:
        result['avg_sparsity'] = np.mean([s['sparsity'] for s in sparse_stats])
        result['avg_nnz'] = np.mean([s['nnz'] for s in sparse_stats])
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 with Sparse Attention')
    
    # Model args
    parser.add_argument('--model', type=str, default='gpt2',
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                       help='Model to fine-tune')
    parser.add_argument('--method', type=str, default='lora',
                       choices=['full', 'lora'],
                       help='Fine-tuning method')
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank (for method=lora)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha (for method=lora)')
    
    # Sparse attention args
    parser.add_argument('--variant', type=str, default='prior_sparse',
                       choices=['baseline', 'prior_softmax', 'prior_sparse'],
                       help='Attention variant')
    parser.add_argument('--lam', type=float, default=0.01,
                       help='Prior strength (lambda)')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Sparsemax temperature (gamma)')
    parser.add_argument('--prior_mode', type=str, default='centered',
                       choices=['raw', 'centered', 'clipped', 'standardized'],
                       help='Prior processing mode')
    parser.add_argument('--no_sparse', action='store_true',
                       help='Use dense baseline (for comparison)')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm')
    
    # Optimization args
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision (fp16)')
    parser.add_argument('--no_gradient_checkpointing', action='store_true',
                       help='Disable gradient checkpointing')
    
    # Data args
    parser.add_argument('--max_train_samples', type=int, default=None,
                       help='Max training samples (for debugging)')
    parser.add_argument('--max_eval_samples', type=int, default=10000,
                       help='Max eval tokens')
    
    # Output args
    parser.add_argument('--output_dir', type=str, default='outputs/finetune',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[ENV] Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"[ENV] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("[ENV] Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("[ENV] Using CPU")
    
    # Override for dense baseline
    if args.no_sparse:
        args.variant = 'baseline'
        args.lam = 0.0
        args.gamma = 1.0
        print("[CONFIG] Running DENSE baseline (no sparse attention)")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.model}_{args.method}_{args.variant}_gamma{args.gamma}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"[MAIN] Output directory: {output_dir}")
    print(f"[CONFIG] Model: {args.model}, Method: {args.method}")
    print(f"[CONFIG] Variant: {args.variant}, λ={args.lam}, γ={args.gamma}")
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(
        args.model,
        device,
        method=args.method,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing=not args.no_gradient_checkpointing
    )
    
    # Setup datasets
    train_dataset = WikiTextDataset(
        tokenizer, split='train',
        max_length=args.max_length,
        max_samples=args.max_train_samples
    )
    
    eval_dataset = WikiTextDataset(
        tokenizer, split='validation',
        max_length=args.max_length,
        max_samples=args.max_eval_samples
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # MPS doesn't play well with multiprocessing
        collate_fn=default_data_collator
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=default_data_collator
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"[TRAIN] Total steps: {total_steps}, Warmup: {warmup_steps}")
    
    # Training loop
    best_eval_ppl = float('inf')
    history = []
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        
        # Train
        train_stats = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch,
            args.variant, args.lam, args.gamma, args.prior_mode,
            use_amp=args.use_amp,
            max_grad_norm=args.max_grad_norm
        )
        
        print(f"  Train - Loss: {train_stats['loss']:.4f}, PPL: {train_stats['ppl']:.2f}")
        if 'avg_sparsity' in train_stats:
            print(f"  Train - Sparsity: {train_stats['avg_sparsity']:.2%}, NNZ: {train_stats['avg_nnz']:.1f}")
        
        # Evaluate
        eval_stats = evaluate(
            model, eval_loader, device,
            args.variant, args.lam, args.gamma, args.prior_mode,
            desc=f"Eval Epoch {epoch}"
        )
        
        print(f"  Eval  - Loss: {eval_stats['loss']:.4f}, PPL: {eval_stats['ppl']:.2f}")
        if 'avg_sparsity' in eval_stats:
            print(f"  Eval  - Sparsity: {eval_stats['avg_sparsity']:.2%}, NNZ: {eval_stats['avg_nnz']:.1f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train': train_stats,
            'eval': eval_stats
        })
        
        # Save best model
        if eval_stats['ppl'] < best_eval_ppl:
            best_eval_ppl = eval_stats['ppl']
            print(f"  *** New best PPL: {best_eval_ppl:.2f} ***")
            
            if args.method == 'lora':
                model.save_pretrained(output_dir / 'best_model')
            else:
                model.save_pretrained(output_dir / 'best_model')
                tokenizer.save_pretrained(output_dir / 'best_model')
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    final_stats = evaluate(
        model, eval_loader, device,
        args.variant, args.lam, args.gamma, args.prior_mode,
        desc="Final Eval"
    )
    
    print(f"Final PPL: {final_stats['ppl']:.2f}")
    print(f"Best PPL: {best_eval_ppl:.2f}")
    
    # Save final model
    if args.method == 'lora':
        model.save_pretrained(output_dir / 'final_model')
    else:
        model.save_pretrained(output_dir / 'final_model')
        tokenizer.save_pretrained(output_dir / 'final_model')
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save summary
    summary = {
        'config': vars(args),
        'final_ppl': final_stats['ppl'],
        'best_ppl': best_eval_ppl,
        'final_sparsity': final_stats.get('avg_sparsity', 0),
        'final_nnz': final_stats.get('avg_nnz', 0),
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[MAIN] Results saved to: {output_dir}")
    print("="*70)
    
    return best_eval_ppl


if __name__ == '__main__':
    main()
