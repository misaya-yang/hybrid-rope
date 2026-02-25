#!/usr/bin/env python3
"""
NeurIPS Phase 2: Entmax Annealing Experiment

Goal: Demonstrate that sparse attention can be achieved through
      gradual annealing with distillation stabilization.

Strategy:
  - Use entmax(α) where α=1 is softmax, α=2 is near-sparsemax
  - Anneal α: 1.0 → 1.2 → 1.4 → 1.6 → 1.8 over training
  - Distillation loss: L = CE(sparse) + β * KL(softmax || entmax)
  - Only train attention LoRA (q/k/v/o proj)

Training Schedule:
  Phase 1 (1k steps): α=1.0 (standard softmax baseline)
  Phase 2 (1k steps): α=1.2 (light sparsification)
  Phase 3 (1k steps): α=1.4 (medium sparsification)
  Phase 4 (1k steps): α=1.6 (high sparsification)
  Phase 5 (1k steps): α=1.8 (near-sparsemax)
  
  Total: 5k steps max (early stop if PPL > baseline * 1.2)

Metrics (every 200 steps):
  - Valid PPL (true forward)
  - Sparsity (fraction of near-zero weights, threshold=1e-6)
  - Avg NNZ (non-zeros per row)
  - Attention entropy
  - KL divergence between teacher (softmax) and student (entmax)

Stop Conditions:
  - PPL > baseline * 1.20 (20% degradation)
  - Entropy collapse (< 0.05)
  - 2k steps without PPL improvement

Output:
  - CSV: results/phase2_results.csv
  - Plot: figures/phase2_alpha_vs_ppl.png
  - Plot: figures/phase2_pareto_sparsity_ppl.png
  - JSON: results/phase2_summary.json

Acceptance:
  - ∃ phase where sparsity ≥ 0.70 AND PPL ≤ baseline * 1.05
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

sys.path.insert(0, '/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/experiments/variational_sparse_attn')

# Configuration
EXPERIMENT_CONFIG = {
    'model_name': 'gpt2',
    'max_length': 512,
    'batch_size': 4,
    'seed': 42,
    
    # Annealing schedule
    'alpha_schedule': [
        (0, 1000, 1.0),      # Phase 1: softmax
        (1000, 2000, 1.2),   # Phase 2: light sparse
        (2000, 3000, 1.4),   # Phase 3: medium sparse
        (3000, 4000, 1.6),   # Phase 4: high sparse
        (4000, 5000, 1.8),   # Phase 5: near-sparsemax
    ],
    
    # Distillation weights
    'beta_values': [0.1, 0.5],
    
    # Training
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'eval_every': 200,
    'warmup_steps': 200,
    
    # Early stopping
    'max_steps': 5000,
    'patience_steps': 2000,
    'degradation_threshold': 1.20,  # Stop if PPL > baseline * 1.20
    
    # LoRA
    'lora_r': 16,
    'lora_alpha': 32,
}

device = "mps" if torch.backends.mps.is_available() else "cpu"
BASE_DIR = Path('/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/experiments/neurips_strict')
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = BASE_DIR / 'figures'


def entmax_bisect(inputs: torch.Tensor, alpha: float = 1.5, dim: int = -1, n_iter: int = 50) -> torch.Tensor:
    """
    Entmax bisection algorithm.
    
    For α=1: returns softmax
    For α=2: returns sparsemax (exact zeros)
    For 1<α<2: returns intermediate sparsity
    """
    if alpha == 1.0:
        return F.softmax(inputs, dim=dim)
    
    # Normalize inputs
    inputs = inputs - inputs.max(dim=dim, keepdim=True)[0]
    
    # Initialize bounds
    tau_min = inputs.min(dim=dim, keepdim=True)[0] - 1
    tau_max = inputs.max(dim=dim, keepdim=True)[0]
    
    # Bisection to find threshold τ
    for _ in range(n_iter):
        tau = (tau_min + tau_max) / 2
        # p_i = max(0, (α-1) * x_i - τ)^(1/(α-1))
        p = torch.clamp((alpha - 1) * inputs - tau, min=0) ** (1 / (alpha - 1))
        p_sum = p.sum(dim=dim, keepdim=True)
        
        # Update bounds based on whether sum > 1
        tau_min = torch.where(p_sum > 1, tau, tau_min)
        tau_max = torch.where(p_sum < 1, tau, tau_max)
    
    # Final projection
    tau = (tau_min + tau_max) / 2
    p = torch.clamp((alpha - 1) * inputs - tau, min=0) ** (1 / (alpha - 1))
    p = p / p.sum(dim=dim, keepdim=True)
    
    return p


class EntmaxAttention(nn.Module):
    """Entmax attention with teacher (softmax) for distillation."""
    
    def __init__(self, alpha: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.entropy = 0.0
        self.sparsity = 0.0
        self.nnz = 0.0
        self.kl_div = 0.0
    
    def forward(self, query, key, value, attention_mask=None):
        """
        Forward with entmax, recording statistics.
        Returns: (output, attn_weights, teacher_attn) for distillation
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Compute scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Teacher: softmax (baseline)
        teacher_attn = F.softmax(scores, dim=-1)
        
        # Student: entmax(α)
        student_attn = entmax_bisect(scores, alpha=self.alpha, dim=-1)
        
        # Record statistics
        with torch.no_grad():
            # Entropy
            eps = 1e-10
            entropy = -(student_attn * torch.log(student_attn + eps)).sum(dim=-1).mean()
            self.entropy = entropy.item()
            
            # Sparsity (fraction < 1e-6)
            self.sparsity = (student_attn < 1e-6).float().mean().item()
            
            # NNZ (non-zeros per row)
            self.nnz = (student_attn >= 1e-6).float().sum(dim=-1).mean().item()
            
            # KL divergence (teacher || student)
            kl = (teacher_attn * (torch.log(teacher_attn + eps) - torch.log(student_attn + eps))).sum(dim=-1).mean()
            self.kl_div = kl.item()
        
        # Apply to values
        output = torch.matmul(student_attn, value)
        
        return output, student_attn, teacher_attn


def patch_gpt2_for_entmax(model, alpha: float = 1.5):
    """Patch GPT-2 to use entmax attention."""
    entmax_module = EntmaxAttention(alpha=alpha)
    
    for layer_idx, block in enumerate(model.transformer.h):
        attn_module = block.attn
        
        # Store reference to entmax module for updating alpha
        attn_module._entmax_module = entmax_module
        
        def make_forward(entmax_mod):
            def forward(hidden_states, attention_mask=None, **kwargs):
                # Get q, k, v
                query, key, value = attn_module.c_attn(hidden_states).split(
                    attn_module.split_size, dim=2
                )
                query = attn_module._split_heads(query)
                key = attn_module._split_heads(key)
                value = attn_module._split_heads(value)
                
                # Use entmax attention
                attn_output, student_attn, teacher_attn = entmax_mod(
                    query, key, value, attention_mask=attention_mask
                )
                
                # Store for loss computation
                attn_module._student_attn = student_attn
                attn_module._teacher_attn = teacher_attn
                
                # Merge heads
                attn_output = attn_module._merge_heads(attn_output)
                attn_output = attn_module.c_proj(attn_output)
                attn_output = attn_module.attn_dropout(attn_output)
                
                return attn_output, student_attn
            return forward
        
        attn_module._attn = make_forward(entmax_module).__get__(attn_module, type(attn_module))
    
    return entmax_module


def update_alpha(model, new_alpha: float):
    """Update alpha for all entmax modules."""
    for block in model.transformer.h:
        if hasattr(block.attn, '_entmax_module'):
            block.attn._entmax_module.alpha = new_alpha


def compute_distillation_loss(model, beta: float = 0.1) -> torch.Tensor:
    """
    Compute distillation loss across all attention layers.
    L_distill = β * Σ KL(teacher || student)
    """
    total_kl = 0.0
    num_layers = 0
    
    for block in model.transformer.h:
        attn_module = block.attn
        if hasattr(attn_module, '_student_attn') and hasattr(attn_module, '_teacher_attn'):
            teacher = attn_module._teacher_attn
            student = attn_module._student_attn
            
            eps = 1e-10
            kl = (teacher * (torch.log(teacher + eps) - torch.log(student + eps))).sum(dim=-1).mean()
            total_kl += kl
            num_layers += 1
    
    if num_layers == 0:
        return torch.tensor(0.0, device=device)
    
    return beta * (total_kl / num_layers)


def get_attention_stats(model) -> Dict[str, float]:
    """Get average attention statistics across layers."""
    entropies = []
    sparsities = []
    nnzs = []
    
    for block in model.transformer.h:
        if hasattr(block.attn, '_entmax_module'):
            mod = block.attn._entmax_module
            entropies.append(mod.entropy)
            sparsities.append(mod.sparsity)
            nnzs.append(mod.nnz)
    
    return {
        'entropy': np.mean(entropies) if entropies else 0.0,
        'sparsity': np.mean(sparsities) if sparsities else 0.0,
        'nnz': np.mean(nnzs) if nnzs else 0.0,
    }


def get_current_alpha(step: int, schedule: List[Tuple[int, int, float]]) -> float:
    """Get alpha for current step."""
    for start, end, alpha in schedule:
        if start <= step < end:
            return alpha
    return schedule[-1][2]  # Return last alpha


def prepare_dataset(tokenizer, max_length: int = 512):
    """Prepare WikiText-2 dataset."""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_length)
    
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text'], num_proc=1)
    
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
    
    lm_dataset = tokenized.map(group_texts, batched=True, batch_size=1000, num_proc=1)
    return lm_dataset['train'], lm_dataset['validation']


def evaluate(model, dataloader, device) -> Dict[str, float]:
    """Evaluate model (teacher mode, no distillation)."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return {'loss': avg_loss, 'ppl': np.exp(avg_loss)}


def train_entmax_annealing(beta: float, config: Dict, baseline_ppl: float) -> Dict:
    """
    Train with entmax annealing and distillation.
    
    Returns:
        Dict with complete training history
    """
    print(f"\n{'='*60}")
    print(f"Entmax Annealing Experiment (β={beta})")
    print(f"{'='*60}\n")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    
    # Apply LoRA
    try:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            target_modules=['c_attn', 'c_proj'],
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except ImportError:
        print("[WARN] peft not installed, using full fine-tune")
    
    # Patch for entmax
    entmax_module = patch_gpt2_for_entmax(model, alpha=1.0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Data
    train_dataset, val_dataset = prepare_dataset(tokenizer, config['max_length'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, collate_fn=default_data_collator)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           collate_fn=default_data_collator)
    
    # Training state
    history = {
        'steps': [],
        'alpha': [],
        'train_loss': [],
        'val_ppl': [],
        'sparsity': [],
        'nnz': [],
        'entropy': [],
        'kl_div': [],
    }
    
    best_ppl = float('inf')
    best_step = 0
    steps_without_improvement = 0
    global_step = 0
    
    model.train()
    
    for epoch in range(100):
        for batch in train_loader:
            if global_step >= config['max_steps']:
                break
            
            # Update alpha based on schedule
            alpha = get_current_alpha(global_step, config['alpha_schedule'])
            update_alpha(model, alpha)
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            outputs = model(input_ids, labels=labels)
            ce_loss = outputs.loss
            
            # Distillation loss
            distill_loss = compute_distillation_loss(model, beta=beta)
            
            # Total loss
            total_loss = ce_loss + distill_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            
            global_step += 1
            
            # Evaluate
            if global_step % config['eval_every'] == 0:
                # Get stats before eval (in train mode)
                attn_stats = get_attention_stats(model)
                
                # Validation
                val_metrics = evaluate(model, val_loader, device)
                val_ppl = val_metrics['ppl']
                
                # Record
                history['steps'].append(global_step)
                history['alpha'].append(alpha)
                history['train_loss'].append(ce_loss.item())
                history['val_ppl'].append(val_ppl)
                history['sparsity'].append(attn_stats['sparsity'])
                history['nnz'].append(attn_stats['nnz'])
                history['entropy'].append(attn_stats['entropy'])
                history['kl_div'].append(distill_loss.item() / beta if beta > 0 else 0)
                
                # Check improvement
                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                    best_step = global_step
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += config['eval_every']
                
                # Progress
                rel = (val_ppl / baseline_ppl - 1) * 100
                phase_name = {1.0: 'Softmax', 1.2: 'Light', 1.4: 'Medium', 
                             1.6: 'High', 1.8: 'Near-Sparse'}[alpha]
                
                print(f"Step {global_step:4d} | α={alpha:.1f} ({phase_name:8s}) | "
                      f"Val PPL: {val_ppl:7.2f} ({rel:+6.1f}%) | "
                      f"Sparse: {attn_stats['sparsity']*100:5.1f}% | "
                      f"NNZ: {attn_stats['nnz']:4.1f} | "
                      f"Entropy: {attn_stats['entropy']:.3f}")
                
                # Stop conditions
                if val_ppl > baseline_ppl * config['degradation_threshold']:
                    print(f"[STOP] PPL degradation: {val_ppl:.2f} > {baseline_ppl * config['degradation_threshold']:.2f}")
                    return {'beta': beta, 'history': history, 'best_ppl': best_ppl, 
                            'best_step': best_step, 'stopped': True, 'reason': 'degradation'}
                
                if steps_without_improvement >= config['patience_steps']:
                    print(f"[STOP] No improvement for {config['patience_steps']} steps")
                    return {'beta': beta, 'history': history, 'best_ppl': best_ppl,
                            'best_step': best_step, 'stopped': True, 'reason': 'no_improvement'}
                
                if attn_stats['entropy'] < 0.05:
                    print(f"[STOP] Entropy collapse: {attn_stats['entropy']:.4f}")
                    return {'beta': beta, 'history': history, 'best_ppl': best_ppl,
                            'best_step': best_step, 'stopped': True, 'reason': 'entropy_collapse'}
                
                model.train()
        
        if global_step >= config['max_steps']:
            break
    
    return {'beta': beta, 'history': history, 'best_ppl': best_ppl,
            'best_step': best_step, 'stopped': False, 'reason': 'completed'}


def run_phase2(baseline_ppl: float):
    """Run complete Phase 2 experiment."""
    config = EXPERIMENT_CONFIG
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    all_results = []
    
    for beta in config['beta_values']:
        result = train_entmax_annealing(beta, config, baseline_ppl)
        all_results.append(result)
    
    # Save results
    save_phase2_results(all_results, baseline_ppl)
    
    return all_results


def save_phase2_results(results: List[Dict], baseline_ppl: float):
    """Save Phase 2 results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON
    json_path = RESULTS_DIR / f'phase2_history_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump({'baseline_ppl': baseline_ppl, 'results': results}, f, indent=2, default=str)
    
    # CSV summary
    csv_path = RESULTS_DIR / f'phase2_summary_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Beta', 'Best PPL', 'Rel Increase', 'Best Step', 
                        'Max Sparsity', 'Min NNZ', 'Stopped', 'Reason'])
        
        for r in results:
            hist = r['history']
            max_sparse = max(hist['sparsity']) if hist['sparsity'] else 0
            min_nnz = min(hist['nnz']) if hist['nnz'] else 0
            rel = (r['best_ppl'] / baseline_ppl - 1) * 100
            
            writer.writerow([
                r['beta'],
                f"{r['best_ppl']:.2f}",
                f"{rel:+.1f}%",
                r['best_step'],
                f"{max_sparse*100:.1f}%",
                f"{min_nnz:.1f}",
                r['stopped'],
                r['reason']
            ])
    
    print(f"\n[SAVED] Phase 2 results to {csv_path}")
    
    # Acceptance check
    print("\n" + "="*80)
    print("PHASE 2 ACCEPTANCE CHECK")
    print("="*80)
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    print(f"Target: sparsity ≥ 70% AND PPL ≤ {baseline_ppl * 1.05:.2f}")
    print("-"*80)
    
    sweet_spot_found = False
    for r in results:
        hist = r['history']
        for i, (ppl, sparse) in enumerate(zip(hist['val_ppl'], hist['sparsity'])):
            if sparse >= 0.70 and ppl <= baseline_ppl * 1.05:
                step = hist['steps'][i]
                alpha = hist['alpha'][i]
                print(f"✓ SWEET SPOT FOUND at step {step}, α={alpha}")
                print(f"    PPL: {ppl:.2f} (+{(ppl/baseline_ppl-1)*100:.1f}%), Sparsity: {sparse*100:.1f}%")
                sweet_spot_found = True
                break
    
    if not sweet_spot_found:
        print("✗ No sweet spot found (need longer training or lower β)")


if __name__ == '__main__':
    # Load baseline from Phase 1
    baseline_file = list(RESULTS_DIR.glob('phase1_summary_*.csv'))
    if baseline_file:
        with open(baseline_file[-1], 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Group'] == 'A_baseline':
                    baseline_ppl = float(row['Best PPL'])
                    break
    else:
        print("[ERROR] Run Phase 1 first to get baseline")
        sys.exit(1)
    
    print("="*80)
    print("NeurIPS Phase 2: Entmax Annealing Experiment")
    print("="*80)
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    print(f"Annealing: 1.0 → 1.2 → 1.4 → 1.6 → 1.8")
    print("="*80)
    
    results = run_phase2(baseline_ppl)
