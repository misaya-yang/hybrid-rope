#!/usr/bin/env python3
"""
NeurIPS Phase 1: Prior-Softmax Adaptation Experiment

Goal: Validate that prior-guided attention is a "controllable regularizer"
      under light-weight fine-tuning.

Groups:
  A) Baseline: Vanilla softmax (full fine-tune 3k steps)
  B) Prior-Softmax: logits += λ * log D(Δ), λ ∈ {0.01, 0.05, 0.1}
  C) Prior-Softmax + LoRA: Only q/k/v/o_proj LoRA, λ ∈ {0.01, 0.05, 0.1}

Metrics (every 200 steps):
  - Valid PPL (true forward, no estimation)
  - Attention entropy (per layer, averaged)
  - Average attention distance
  - Gradient norm (for monitoring stability)

Stop Conditions:
  - PPL > baseline * 5.0
  - 2k steps without PPL improvement
  - Entropy collapse (< 0.1)

Output:
  - CSV: results/phase1_results.csv
  - Plot: figures/phase1_ppl_curves.png
  - JSON: results/phase1_summary.json
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    default_data_collator
)
from datasets import load_dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import GPT2 module for eager_attention_forward patching
from transformers.models.gpt2 import modeling_gpt2 as gpt2_module

# Experiment Configuration
EXPERIMENT_CONFIG = {
    'model_name': 'gpt2',
    'dataset': 'wikitext',
    'dataset_config': 'wikitext-2-raw-v1',
    'max_length': 512,  # As specified
    'batch_size': 4,    # Adjusted for M4 Max 36GB
    'num_workers': 0,
    'seed': 42,
    
    # Training schedule
    'max_steps': 3000,
    'eval_every': 200,
    'warmup_steps': 300,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    
    # Early stopping
    'patience_steps': 2000,  # Stop if no improvement for 2k steps
    'explosion_threshold': 5.0,  # PPL > baseline * 5
    
    # Group B & C: Lambda values
    'lambda_values': [0.01, 0.05, 0.1],
    
    # LoRA config for Group C
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_target_modules': ['c_attn', 'c_proj'],  # q/k/v/o_proj in GPT-2
}

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[ENV] Device: {device}")
print(f"[ENV] PyTorch: {torch.__version__}")

# Paths
BASE_DIR = Path('/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/experiments/neurips_strict')
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = BASE_DIR / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


class AttentionMonitor:
    """Monitor attention statistics during forward pass."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.entropies = []
        self.distances = []
        self.attention_weights = None
    
    def compute_entropy(self, attn_weights: torch.Tensor) -> float:
        """Compute average attention entropy (per position)."""
        # attn_weights: [batch, heads, seq_len, seq_len]
        # entropy = -sum(p * log(p))
        eps = 1e-10
        entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)  # [batch, heads, seq_len]
        return entropy.mean().item()
    
    def compute_avg_distance(self, attn_weights: torch.Tensor) -> float:
        """Compute average attention distance (weighted by position)."""
        seq_len = attn_weights.size(-1)
        positions = torch.arange(seq_len, device=attn_weights.device).float()  # [seq_len]
        
        # attn_weights: [batch, heads, seq_len_q, seq_len_k]
        # Weighted average position
        avg_pos = (attn_weights * positions.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        query_positions = torch.arange(seq_len, device=attn_weights.device).float()
        distances = torch.abs(query_positions.unsqueeze(0) - avg_pos)  # [batch, heads, seq_len]
        
        return distances.mean().item()
    
    def record(self, attn_weights: torch.Tensor):
        """Record statistics from attention weights."""
        self.entropies.append(self.compute_entropy(attn_weights))
        self.distances.append(self.compute_avg_distance(attn_weights))
        self.attention_weights = attn_weights.detach().cpu()
    
    def get_stats(self) -> Dict[str, float]:
        """Get average statistics."""
        if not self.entropies:
            return {'entropy': 0.0, 'avg_distance': 0.0}
        return {
            'entropy': np.mean(self.entropies),
            'avg_distance': np.mean(self.distances),
        }


class PriorSoftmaxAttention(nn.Module):
    """
    Prior-guided softmax attention.
    
    Adds λ * log D(Δ) to attention logits before softmax,
    where D(Δ) is the distance-based prior (lower for distant tokens).
    """
    
    def __init__(self, lambda_val: float = 0.01, alpha_decay: float = 1.0):
        super().__init__()
        self.lambda_val = lambda_val
        self.alpha_decay = alpha_decay
        self.monitor = AttentionMonitor()
    
    def compute_distance_prior(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute -log(Δ + 1) prior with causal masking.
        Returns: [seq_len, seq_len] tensor of log-priors.
        """
        positions = torch.arange(seq_len, device=device).float()
        # Δ[i,j] = |i - j|
        delta = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
        # log D(Δ) = -log(Δ + 1)
        log_prior = -torch.log(delta + 1.0)
        # Causal mask: only attend to previous positions
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        log_prior = log_prior * causal_mask
        # Fill upper triangle with very negative values
        log_prior = log_prior.masked_fill(causal_mask == 0, -1e9)
        return log_prior
    
    def forward(self, query, key, value, attention_mask=None, **kwargs):
        """
        Modified attention with prior bias.
        
        Args:
            query: [batch, heads, seq_len, head_dim]
            key: [batch, heads, seq_len, head_dim]
            value: [batch, heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Compute standard attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(head_dim)
        # scores: [batch, heads, seq_len, seq_len]
        
        # Add distance prior bias
        log_prior = self.compute_distance_prior(seq_len, scores.device)
        # Expand to match batch and heads
        log_prior = log_prior.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply prior bias: λ * log D(Δ)
        scores = scores + self.lambda_val * log_prior
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Record statistics
        self.monitor.record(attn_weights)
        
        # Apply to values
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


# Global storage for original attention function
_original_eager_attention_forward = None

def patch_model_for_prior(model, lambda_val: float, use_prior: bool = True):
    """
    Patch GPT-2 model to use prior-softmax attention.
    
    Args:
        model: GPT-2 model
        lambda_val: Prior strength
        use_prior: If False, use vanilla attention (for baseline)
    """
    global _original_eager_attention_forward
    
    # Store original on first call
    if _original_eager_attention_forward is None:
        _original_eager_attention_forward = gpt2_module.eager_attention_forward
    
    if not use_prior:
        # Restore original if no prior
        gpt2_module.eager_attention_forward = _original_eager_attention_forward
        return None
    
    prior_module = PriorSoftmaxAttention(lambda_val=lambda_val)
    
    def patched_eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
        """
        Patched eager_attention_forward with distance prior support.
        """
        # Compute attention scores (copied from original)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if module.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)
            
        # Apply prior: logits += λ * log D(Δ)
        # D(Δ) is distance-based prior that decays with position distance
        seq_len = query.size(-2)
        device = query.device
        
        # Create distance matrix: |i - j|
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))  # [seq_len, seq_len]
        
        # Distance prior: closer positions get higher prior weight
        # D(Δ) = 1 / (1 + Δ) -> log D(Δ) = -log(1 + Δ)
        log_distance_prior = -torch.log1p(distance)  # [seq_len, seq_len]
        
        # Apply causal mask (future positions get -inf prior)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device), 
            diagonal=1
        )
        log_distance_prior = log_distance_prior + causal_mask
        
        # Add prior to attention scores: logits += λ * log D(Δ)
        # Expand to match attn_weights shape [batch, heads, seq_len, seq_len]
        prior_bias = lambda_val * log_distance_prior.unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights + prior_bias
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax normalization
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        # Record statistics via monitor
        if hasattr(model, '_attention_monitor'):
            model._attention_monitor.record(attn_weights)
        
        # Apply dropout
        attn_weights = nn.functional.dropout(
            attn_weights, p=module.attn_dropout.p, training=module.training
        )
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights
    
    # Apply patch globally
    gpt2_module.eager_attention_forward = patched_eager_attention_forward
    
    return prior_module


def unpatch_model_for_prior():
    """Remove the prior patch and restore original attention."""
    global _original_eager_attention_forward
    if _original_eager_attention_forward is not None:
        gpt2_module.eager_attention_forward = _original_eager_attention_forward


def prepare_dataset(tokenizer, max_length: int = 512):
    """Prepare WikiText-2 dataset."""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            max_length=max_length,
            return_special_tokens_mask=False
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        num_proc=1,
    )
    
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated['input_ids'])
        total_length = (total_length // max_length) * max_length
        
        result = {
            k: [t[i:i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated.items()
        }
        result['labels'] = result['input_ids'].copy()
        return result
    
    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=1,
    )
    
    return lm_dataset['train'], lm_dataset['validation']


def evaluate(model, dataloader, device) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            # Count tokens (excluding padding -1)
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = np.exp(avg_loss)
    
    return {'loss': avg_loss, 'ppl': ppl}


def train_group(
    group_name: str,
    use_prior: bool,
    lambda_val: Optional[float],
    use_lora: bool,
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    baseline_ppl: Optional[float] = None
) -> Dict:
    """
    Train one experimental group.
    
    Returns:
        Dict with training history and final metrics
    """
    print(f"\n{'='*60}")
    print(f"Training Group: {group_name}")
    print(f"  Use Prior: {use_prior}, λ={lambda_val}, LoRA: {use_lora}")
    print(f"{'='*60}\n")
    
    # Load fresh model
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    
    # Apply LoRA if needed (Group C)
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=config['lora_r'],
                lora_alpha=config['lora_alpha'],
                target_modules=config['lora_target_modules'],
                lora_dropout=0.05,
                bias='none',
                task_type='CAUSAL_LM'
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            print("[WARN] peft not installed, falling back to full fine-tune")
            use_lora = False
    
    # Patch attention for prior (Group B & C)
    monitor = AttentionMonitor()
    model._attention_monitor = monitor
    prior_module = None
    if use_prior:
        prior_module = patch_model_for_prior(model, lambda_val, use_prior=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Training state
    history = {
        'steps': [],
        'train_loss': [],
        'val_ppl': [],
        'entropy': [],
        'avg_distance': [],
    }
    
    best_ppl = float('inf')
    best_step = 0
    steps_without_improvement = 0
    global_step = 0
    
    model.train()
    
    for epoch in range(100):  # Large number, we'll stop by steps
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= config['max_steps']:
                break
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Reset monitor
            if prior_module is not None:
                prior_module.monitor.reset()
            
            # Forward
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            
            global_step += 1
            
            # Evaluate every N steps
            if global_step % config['eval_every'] == 0:
                # Get attention stats
                attn_stats = monitor.get_stats() if use_prior else {}
                
                # Validation
                val_metrics = evaluate(model, val_loader, device)
                val_ppl = val_metrics['ppl']
                
                # Record
                history['steps'].append(global_step)
                history['train_loss'].append(loss.item())
                history['val_ppl'].append(val_ppl)
                history['entropy'].append(attn_stats.get('entropy', 0.0))
                history['avg_distance'].append(attn_stats.get('avg_distance', 0.0))
                
                # Check for improvement
                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                    best_step = global_step
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += config['eval_every']
                
                # Progress print
                rel_increase = (val_ppl / baseline_ppl - 1) * 100 if baseline_ppl else 0
                print(f"Step {global_step:4d} | "
                      f"Train Loss: {loss.item():.4f} | "
                      f"Val PPL: {val_ppl:7.2f} ({rel_increase:+.1f}%) | "
                      f"Entropy: {attn_stats.get('entropy', 0):.3f} | "
                      f"Dist: {attn_stats.get('avg_distance', 0):.1f}")
                
                # Stop condition 1: PPL explosion
                if baseline_ppl and val_ppl > baseline_ppl * config['explosion_threshold']:
                    print(f"[STOP] PPL explosion: {val_ppl:.2f} > {baseline_ppl * config['explosion_threshold']:.2f}")
                    return {
                        'group': group_name,
                        'history': history,
                        'best_ppl': best_ppl,
                        'best_step': best_step,
                        'stopped_early': True,
                        'stop_reason': 'ppl_explosion'
                    }
                
                # Stop condition 2: No improvement for 2k steps
                if steps_without_improvement >= config['patience_steps']:
                    print(f"[STOP] No improvement for {config['patience_steps']} steps")
                    return {
                        'group': group_name,
                        'history': history,
                        'best_ppl': best_ppl,
                        'best_step': best_step,
                        'stopped_early': True,
                        'stop_reason': 'no_improvement'
                    }
                
                # Stop condition 3: Entropy collapse
                if attn_stats.get('entropy', 1.0) < 0.1:
                    print(f"[STOP] Entropy collapse: {attn_stats['entropy']:.4f}")
                    return {
                        'group': group_name,
                        'history': history,
                        'best_ppl': best_ppl,
                        'best_step': best_step,
                        'stopped_early': True,
                        'stop_reason': 'entropy_collapse'
                    }
                
                model.train()
        
        if global_step >= config['max_steps']:
            break
    
    return {
        'group': group_name,
        'history': history,
        'best_ppl': best_ppl,
        'best_step': best_step,
        'stopped_early': False,
        'stop_reason': 'completed'
    }


def run_phase1():
    """Run complete Phase 1 experiment."""
    config = EXPERIMENT_CONFIG
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    print("[DATA] Loading WikiText-2...")
    train_dataset, val_dataset = prepare_dataset(tokenizer, config['max_length'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=default_data_collator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        collate_fn=default_data_collator
    )
    
    print(f"[DATA] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Store results
    all_results = []
    baseline_ppl = None
    
    # Group A: Baseline (vanilla softmax)
    print("\n" + "="*60)
    print("GROUP A: BASELINE (Vanilla Softmax)")
    print("="*60)
    result_a = train_group(
        group_name='A_baseline',
        use_prior=False,
        lambda_val=None,
        use_lora=False,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        baseline_ppl=None  # No baseline yet
    )
    baseline_ppl = result_a['best_ppl']
    all_results.append(result_a)
    
    print(f"\n[BASELINE] Final PPL: {baseline_ppl:.2f}")
    
    # Group B: Prior-Softmax with different lambda
    for lam in config['lambda_values']:
        result_b = train_group(
            group_name=f'B_prior_softmax_lam{lam}',
            use_prior=True,
            lambda_val=lam,
            use_lora=False,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            baseline_ppl=baseline_ppl
        )
        all_results.append(result_b)
    
    # Group C: Prior-Softmax + LoRA
    for lam in config['lambda_values']:
        result_c = train_group(
            group_name=f'C_prior_softmax_lora_lam{lam}',
            use_prior=True,
            lambda_val=lam,
            use_lora=True,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            baseline_ppl=baseline_ppl
        )
        all_results.append(result_c)
    
    # Save results
    save_results(all_results, baseline_ppl)
    
    return all_results, baseline_ppl


def save_results(results: List[Dict], baseline_ppl: float):
    """Save results to CSV and JSON."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed history as JSON
    json_path = RESULTS_DIR / f'phase1_history_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'baseline_ppl': baseline_ppl,
            'results': results
        }, f, indent=2, default=str)
    print(f"\n[SAVED] History: {json_path}")
    
    # Save summary CSV
    csv_path = RESULTS_DIR / f'phase1_summary_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Group', 'Lambda', 'LoRA', 'Best PPL', 'Relative Increase',
            'Best Step', 'Stopped Early', 'Stop Reason'
        ])
        
        for r in results:
            group = r['group']
            lam = 'N/A' if 'lam' not in group else group.split('lam')[1].split('_')[0]
            use_lora = 'Yes' if 'lora' in group else 'No'
            rel = (r['best_ppl'] / baseline_ppl - 1) * 100
            
            writer.writerow([
                group,
                lam,
                use_lora,
                f"{r['best_ppl']:.2f}",
                f"{rel:+.1f}%",
                r['best_step'],
                r['stopped_early'],
                r['stop_reason']
            ])
    
    print(f"[SAVED] Summary: {csv_path}")
    
    # Print final summary table
    print("\n" + "="*80)
    print("PHASE 1 FINAL RESULTS")
    print("="*80)
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    print(f"Acceptance Threshold (≤+5%): {baseline_ppl * 1.05:.2f}")
    print("-"*80)
    print(f"{'Group':<30} {'Best PPL':<12} {'Increase':<12} {'Status'}")
    print("-"*80)
    
    acceptable_count = 0
    for r in results:
        group = r['group']
        ppl = r['best_ppl']
        rel = (ppl / baseline_ppl - 1) * 100
        status = "✅ ACCEPT" if ppl <= baseline_ppl * 1.05 else "❌ REJECT"
        if ppl <= baseline_ppl * 1.05:
            acceptable_count += 1
        print(f"{group:<30} {ppl:<12.2f} {rel:+6.1f}%      {status}")
    
    print("-"*80)
    print(f"\nConclusion: {acceptable_count}/{len(results)-1} prior configurations within +5% threshold.")
    
    if acceptable_count > 0:
        print("✓ Prior-softmax is a CONTROLLABLE regularizer under light adaptation.")
    else:
        print("✗ Prior direction needs adjustment (all λ values too aggressive).")


if __name__ == '__main__':
    print("="*80)
    print("NeurIPS Phase 1: Prior-Softmax Adaptation Experiment")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Max steps: {EXPERIMENT_CONFIG['max_steps']}")
    print(f"Eval every: {EXPERIMENT_CONFIG['eval_every']} steps")
    print(f"Early stop patience: {EXPERIMENT_CONFIG['patience_steps']} steps")
    print("="*80)
    
    results, baseline = run_phase1()
    
    print("\n" + "="*80)
    print(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
