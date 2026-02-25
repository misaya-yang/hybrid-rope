#!/usr/bin/env python3
"""
Prior-Guided Variational Sparse Attention Validation on GPT-2
===========================================================

Validates three attention variants:
1. Baseline: Standard softmax attention
2. Prior-Biased: Softmax with distance prior
3. Sparse: Sparsemax with distance prior

Usage:
    conda activate aidemo
    python scripts/simple_validation.py
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from entmax import sparsemax
import numpy as np

# Force eager attention implementation to get attention weights
def load_model_with_eager_attention():
    """Load GPT-2 with eager attention implementation (required for getting attention weights)"""
    model = GPT2LMHeadModel.from_pretrained(
        'gpt2',
        attn_implementation="eager"  # Critical: SDPA doesn't return attention weights
    )
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def distance_prior(seq_len, device='cpu', alpha=1.5):
    """
    Compute distance-based prior D(Δ) ∝ (Δ+1)^(-α)
    
    Args:
        seq_len: Sequence length
        device: Device to place tensor on
        alpha: Power-law decay factor
        
    Returns:
        log_prior: [seq_len, seq_len] tensor with causal masking
    """
    # Distance matrix: Δ(i,j) = i - j (position difference)
    positions = torch.arange(seq_len, device=device)
    delta = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
    
    # Power-law: D(Δ) ∝ (|Δ|+1)^(-α)
    # Use log for numerical stability: log(D) = -α * log(|Δ|+1)
    log_prior = -alpha * torch.log(torch.abs(delta) + 1)
    
    # Apply causal mask (future positions get -inf)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    log_prior = log_prior.masked_fill(causal_mask == 0, float('-inf'))
    
    return log_prior


def apply_prior_biased_attention(attn_logits, log_prior, lam=8.0):
    """
    Apply distance prior to attention logits
    
    Args:
        attn_logits: [batch, heads, seq_len, seq_len] attention logits (before softmax)
        log_prior: [seq_len, seq_len] log prior matrix
        lam: Prior weight (λ)
        
    Returns:
        modified_logits: [batch, heads, seq_len, seq_len]
    """
    # Add prior to logits: logits' = logits + λ * log_prior
    # Expand prior to match attention shape
    log_prior_expanded = log_prior.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    modified_logits = attn_logits + lam * log_prior_expanded
    return modified_logits


def apply_sparse_attention(attn_logits, log_prior, lam=8.0, gamma=0.3):
    """
    Apply sparsemax with temperature scaling and distance prior
    
    Args:
        attn_logits: [batch, heads, seq_len, seq_len] attention logits
        log_prior: [seq_len, seq_len] log prior matrix
        lam: Prior weight (λ)
        gamma: Temperature for sparsemax (γ)
        
    Returns:
        attn_weights: [batch, heads, seq_len, seq_len] sparse attention weights
    """
    # Apply prior and temperature scaling
    modified_logits = apply_prior_biased_attention(attn_logits, log_prior, lam)
    scaled_logits = modified_logits / gamma
    
    # Apply sparsemax over last dimension (keys)
    batch, heads, seq_len, _ = scaled_logits.shape
    
    # Flatten to apply sparsemax
    flat_logits = scaled_logits.view(-1, seq_len)  # [batch*heads*seq_len, seq_len]
    
    # Apply sparsemax
    flat_weights = sparsemax(flat_logits, dim=-1)
    
    # Reshape back
    attn_weights = flat_weights.view(batch, heads, seq_len, seq_len)
    
    return attn_weights


def compute_sparsity(attn_weights):
    """
    Compute sparsity (% of zero weights)
    
    Args:
        attn_weights: [batch, heads, seq_len, seq_len]
        
    Returns:
        sparsity: float, percentage of zero weights
    """
    total = attn_weights.numel()
    zeros = (attn_weights == 0).sum().item()
    return 100.0 * zeros / total


def compute_attention_distance(attn_weights):
    """
    Compute average attention distance (how far back tokens attend)
    
    Args:
        attn_weights: [batch, heads, seq_len, seq_len]
        
    Returns:
        avg_distance: float, average attention distance
    """
    seq_len = attn_weights.shape[-1]
    device = attn_weights.device
    
    # Distance from current position
    positions = torch.arange(seq_len, device=device)
    distances = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
    
    # Weight by attention weights
    # Only consider causal (lower triangular) part
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    masked_weights = attn_weights * causal_mask.unsqueeze(0).unsqueeze(0)
    
    # Normalize per query position
    weight_sums = masked_weights.sum(dim=-1, keepdim=True) + 1e-8
    normalized_weights = masked_weights / weight_sums
    
    # Compute weighted average distance
    avg_distance = (normalized_weights * distances.unsqueeze(0).unsqueeze(0)).sum()
    
    return avg_distance.item()


def analyze_attention_pattern(attn_weights, tokenizer, input_ids, layer_name=""):
    """
    Analyze attention pattern statistics
    
    Args:
        attn_weights: [batch, heads, seq_len, seq_len]
        tokenizer: Tokenizer for decoding
        input_ids: [batch, seq_len] input token ids
        layer_name: Name of the layer for printing
    """
    batch, heads, seq_len, _ = attn_weights.shape
    
    print(f"\n{'='*60}")
    print(f"Layer: {layer_name}")
    print(f"Shape: {attn_weights.shape}")
    
    # Sparsity
    sparsity = compute_sparsity(attn_weights)
    print(f"Sparsity: {sparsity:.2f}%")
    
    # Non-zero count per position
    nonzero_per_query = (attn_weights[0] != 0).sum(dim=-1).float().mean().item()
    print(f"Avg non-zero weights per query: {nonzero_per_query:.2f} / {seq_len}")
    
    # Attention distance
    avg_dist = compute_attention_distance(attn_weights)
    print(f"Average attention distance: {avg_dist:.2f} tokens")
    
    # Distribution statistics
    flat_weights = attn_weights[0, 0].flatten()  # First batch, first head
    print(f"Weight distribution: min={flat_weights.min():.4f}, max={flat_weights.max():.4f}")
    print(f"                   mean={flat_weights.mean():.4f}, std={flat_weights.std():.4f}")
    
    # Check for uniform-like distribution (baseline should be smooth)
    entropy = -(flat_weights * (flat_weights + 1e-10).log()).sum().item()
    print(f"Attention entropy: {entropy:.4f}")
    
    # Sample a few positions to show attention pattern
    print("\nSample attention patterns (first head):")
    for q_pos in [min(3, seq_len-1), min(5, seq_len-1), min(10, seq_len-1), seq_len-1]:
        if q_pos < seq_len:
            weights = attn_weights[0, 0, q_pos, :q_pos+1]  # Only causal part
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0, :q_pos+1])
            
            # Find top-3 attended positions
            top_k = min(3, len(weights))
            top_indices = weights.argsort(descending=True)[:top_k]
            
            print(f"  Position {q_pos} ({tokens[-1] if tokens else 'N/A'}):")
            for idx in top_indices:
                idx_item = idx.item()
                print(f"    -> {idx_item} ({tokens[idx_item]}): {weights[idx_item]:.4f}")


def validate_variants(model, tokenizer, text, lam=8.0, gamma=0.3, alpha=1.5):
    """
    Validate three attention variants on the same input
    
    Args:
        model: GPT-2 model with eager attention
        tokenizer: GPT-2 tokenizer
        text: Input text
        lam: Prior weight (λ)
        gamma: Sparsemax temperature (γ)
        alpha: Power-law decay factor
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=32)
    input_ids = inputs['input_ids']
    seq_len = input_ids.shape[1]
    device = input_ids.device
    
    print(f"\n{'='*80}")
    print("Prior-Guided Variational Sparse Attention Validation")
    print(f"{'='*80}")
    print(f"Input text: {text[:60]}...")
    print(f"Sequence length: {seq_len}")
    print(f"Parameters: λ={lam}, γ={gamma}, α={alpha}")
    
    # Compute distance prior
    log_prior = distance_prior(seq_len, device=device, alpha=alpha)
    print(f"\nDistance prior computed:")
    print(f"  Shape: {log_prior.shape}")
    print(f"  Finite values: {(log_prior > float('-inf')).sum().item()} / {log_prior.numel()}")
    
    # Get baseline attention
    print(f"\n{'='*80}")
    print("1. BASELINE (Standard Softmax Attention)")
    print(f"{'='*80}")
    
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        baseline_attentions = outputs.attentions  # Tuple of [batch, heads, seq, seq]
    
    # Analyze last layer
    last_layer_attn = baseline_attentions[-1]  # [batch, heads, seq, seq]
    analyze_attention_pattern(last_layer_attn, tokenizer, input_ids, "Layer 11 (Last)")
    baseline_sparsity = compute_sparsity(last_layer_attn)
    
    # Get attention logits (before softmax) from first layer for detailed analysis
    # We need to manually compute attention to get logits
    print(f"\n{'='*80}")
    print("2. PRIOR-BIASED (Softmax + Distance Prior)")
    print(f"{'='*80}")
    
    # For prior-biased, we modify the attention computation
    # We'll do this for the first layer as demonstration
    with torch.no_grad():
        hidden_states = model.transformer.wte(input_ids) + model.transformer.wpe(
            torch.arange(seq_len, device=device).unsqueeze(0)
        )
        
        # First layer
        block = model.transformer.h[0]
        
        # Get Q, K, V
        attn = block.attn
        qkv = attn.c_attn(hidden_states)
        q, k, v = qkv.split(attn.split_size, dim=2)
        
        # Reshape for multi-head
        batch_size = 1
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # [B, H, T, T]
        
        if attn.scale_attn_weights:
            scores = scores / (head_dim ** 0.5)
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply prior
        modified_scores = apply_prior_biased_attention(scores, log_prior, lam)
        
        # Softmax
        prior_biased_attn = F.softmax(modified_scores, dim=-1)
        prior_biased_attn = prior_biased_attn.masked_fill(torch.isnan(prior_biased_attn), 0)
        
        analyze_attention_pattern(prior_biased_attn, tokenizer, input_ids, "Layer 0 (Prior-Biased)")
        prior_sparsity = compute_sparsity(prior_biased_attn)
    
    # Sparse attention
    print(f"\n{'='*80}")
    print("3. SPARSE (Sparsemax + Distance Prior)")
    print(f"{'='*80}")
    
    with torch.no_grad():
        # Use same scores from above
        sparse_attn = apply_sparse_attention(scores, log_prior, lam, gamma)
        sparse_attn = sparse_attn.masked_fill(torch.isnan(sparse_attn), 0)
        
        analyze_attention_pattern(sparse_attn, tokenizer, input_ids, "Layer 0 (Sparse)")
        sparse_sparsity = compute_sparsity(sparse_attn)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Variant':<20} {'Sparsity':<15} {'Avg Distance':<15} {'Non-zero/Query':<15}")
    print("-" * 80)
    
    def get_stats(attn):
        sparsity = compute_sparsity(attn)
        dist = compute_attention_distance(attn)
        nonzero = (attn[0] != 0).sum(dim=-1).float().mean().item()
        return sparsity, dist, nonzero
    
    b_sparsity, b_dist, b_nonzero = get_stats(last_layer_attn)
    p_sparsity, p_dist, p_nonzero = get_stats(prior_biased_attn)
    s_sparsity, s_dist, s_nonzero = get_stats(sparse_attn)
    
    print(f"{'Baseline':<20} {b_sparsity:<15.2f}% {b_dist:<15.2f} {b_nonzero:<15.2f}")
    print(f"{'Prior-Biased':<20} {p_sparsity:<15.2f}% {p_dist:<15.2f} {p_nonzero:<15.2f}")
    print(f"{'Sparse':<20} {s_sparsity:<15.2f}% {s_dist:<15.2f} {s_nonzero:<15.2f}")
    
    print(f"\nKey Findings:")
    print(f"  - Sparse attention achieves {s_sparsity:.1f}% sparsity (zeros)")
    print(f"  - Prior-biased attention shifts focus to closer tokens (dist={p_dist:.2f})")
    print(f"  - Sparse attention reduces non-zero connections by {(b_nonzero - s_nonzero):.1f} per query")
    
    return {
        'baseline': {'sparsity': b_sparsity, 'distance': b_dist},
        'prior_biased': {'sparsity': p_sparsity, 'distance': p_dist},
        'sparse': {'sparsity': s_sparsity, 'distance': s_dist}
    }


def main():
    """Main validation function"""
    print("Loading GPT-2 model with eager attention...")
    model, tokenizer = load_model_with_eager_attention()
    model.eval()
    
    # Test on a long-range dependency example
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In 1492, Christopher Columbus sailed across the Atlantic Ocean and discovered the Americas. This historic voyage changed the course of world history.",
        "The capital of France is Paris. It is known for the Eiffel Tower, delicious cuisine, and rich cultural heritage.",
    ]
    
    results = []
    for text in test_texts:
        result = validate_variants(
            model, tokenizer, text,
            lam=8.0,   # Prior weight
            gamma=0.3,  # Sparsemax temperature
            alpha=1.5   # Power-law decay
        )
        results.append(result)
    
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
