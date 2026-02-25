"""
Attention patching for variational sparse attention experiments - V2
Supports multiple prior modes: raw, centered, clipped, standardized
"""
import torch
import torch.nn.functional as F
from entmax import sparsemax
from typing import Optional, Dict, Any
import transformers.models.gpt2.modeling_gpt2 as gpt2_module
import numpy as np

# Store original function
_original_eager_attention_forward = gpt2_module.eager_attention_forward

# Global configuration storage
ATTENTION_CONFIG: Dict[str, Any] = {
    'variant': 'baseline',  # 'baseline', 'prior_softmax', 'prior_sparse'
    'lam': 0.0,             # Prior strength
    'gamma': 1.0,           # Temperature for sparsemax
    'alpha': 1.5,           # Distance prior exponent
    'prior_mode': 'centered',  # 'raw', 'centered', 'clipped', 'standardized'
    'clip_value': 5.0,      # For clipped mode
    'attention_weights': None,  # Store for analysis
    'capture_stats': True,  # Whether to compute detailed stats
}


def compute_distance_prior(seq_len: int, alpha: float, device: torch.device, 
                           mode: str = 'centered', clip_value: float = 5.0) -> torch.Tensor:
    """
    Compute distance-based prior log D(Δ) = -α * log(Δ + 1).
    
    Args:
        seq_len: Sequence length
        alpha: Distance decay exponent
        device: Target device
        mode: 'raw', 'centered', 'clipped', 'standardized'
        clip_value: Max magnitude for clipped mode
    
    Returns:
        [seq_len, seq_len] tensor with -inf for future positions (causal mask)
    """
    indices = torch.arange(seq_len, device=device)
    delta = indices.unsqueeze(1) - indices.unsqueeze(0)  # [seq_len, seq_len]
    
    # Causal mask: only attend to current and past positions
    log_prior = torch.where(
        delta >= 0,
        -alpha * torch.log(delta + 1.0),
        torch.tensor(float('-inf'), device=device)
    )
    
    if mode == 'raw':
        return log_prior
    
    # For normalization, work only on allowed positions (finite values)
    allowed_mask = torch.isfinite(log_prior)
    
    if mode == 'clipped':
        # Clip to [-clip_value, 0]
        log_prior = torch.where(
            allowed_mask,
            torch.clamp(log_prior, min=-clip_value, max=0.0),
            log_prior
        )
        return log_prior
    
    if mode == 'centered' or mode == 'standardized':
        # Compute statistics over allowed positions only
        allowed_values = log_prior[allowed_mask]
        
        if len(allowed_values) > 0:
            mean_val = allowed_values.mean()
            
            if mode == 'centered':
                # Subtract mean to center around 0
                log_prior = torch.where(
                    allowed_mask,
                    log_prior - mean_val,
                    log_prior
                )
            elif mode == 'standardized':
                std_val = allowed_values.std() + 1e-8
                log_prior = torch.where(
                    allowed_mask,
                    (log_prior - mean_val) / std_val,
                    log_prior
                )
    
    return log_prior


def compute_sparsity_stats(attn_weights: torch.Tensor) -> Dict[str, float]:
    """
    Compute sparsity statistics, distinguishing allowed vs masked regions.
    
    Args:
        attn_weights: [batch, heads, seq_len, seq_len] tensor
    
    Returns:
        Dictionary with sparsity metrics
    """
    seq_len = attn_weights.shape[-1]
    
    # Create causal mask (allowed positions: j <= i)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attn_weights.device)).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
    
    # Total elements
    total_elements = attn_weights.numel()
    
    # Allowed region elements
    allowed_elements = causal_mask.expand_as(attn_weights).sum().item()
    
    # Exact zeros in total
    total_zeros = (attn_weights == 0).sum().item()
    
    # Exact zeros in allowed region
    allowed_zeros = ((attn_weights == 0) & causal_mask.expand_as(attn_weights)).sum().item()
    
    # Sparsity ratios
    sparsity_total = total_zeros / total_elements if total_elements > 0 else 0.0
    sparsity_allowed = allowed_zeros / allowed_elements if allowed_elements > 0 else 0.0
    
    # Average NNZ per token (in allowed region)
    # For each query position, count non-zeros among allowed keys
    allowed_weights = attn_weights * causal_mask.expand_as(attn_weights).float()
    nnz_per_row = (allowed_weights > 0).sum(dim=-1).float()  # [batch, heads, seq_len]
    avg_nnz_allowed = nnz_per_row.mean().item()
    
    # Baseline NNZ (dense softmax with causal mask) = seq_len * (seq_len+1) / 2 / seq_len ≈ seq_len/2
    baseline_nnz = (seq_len + 1) / 2
    nnz_ratio = avg_nnz_allowed / baseline_nnz if baseline_nnz > 0 else 1.0
    
    # Row sum error (should be 1.0 for each row in allowed region)
    row_sums = attn_weights.sum(dim=-1)
    row_sum_error = (row_sums - 1.0).abs().max().item()
    
    # Entropy (optional, in nats)
    eps = 1e-10
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1).mean().item()
    
    return {
        'sparsity_total': sparsity_total,
        'sparsity_allowed': sparsity_allowed,
        'exact_zeros_total': total_zeros,
        'exact_zeros_allowed': allowed_zeros,
        'avg_nnz_allowed': avg_nnz_allowed,
        'baseline_nnz': baseline_nnz,
        'nnz_ratio': nnz_ratio,
        'row_sum_error': row_sum_error,
        'entropy': entropy,
        'seq_len': seq_len,
    }


def patched_eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    """
    Patched version of eager_attention_forward with prior support.
    """
    config = ATTENTION_CONFIG
    variant = config['variant']
    
    # Compute attention weights
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    # Apply distance prior if requested
    if variant in ['prior_softmax', 'prior_sparse'] and config['lam'] > 0:
        seq_len = query.size(-2)
        lam = config['lam']
        alpha = config['alpha']
        prior_mode = config.get('prior_mode', 'centered')
        clip_value = config.get('clip_value', 5.0)
        
        # Compute prior
        prior = compute_distance_prior(seq_len, alpha, query.device, prior_mode, clip_value)
        
        # Expand to match batch/heads dimensions
        batch_size, num_heads, _, _ = attn_weights.shape
        prior = prior.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        
        # Add prior bias
        attn_weights = attn_weights + lam * prior

    # Apply attention mask if provided
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Apply normalization
    if variant == 'prior_sparse':
        gamma = config['gamma']
        # Temperature scaling
        attn_weights = gamma * attn_weights
        # Sparsemax normalization
        attn_weights = sparsemax(attn_weights, dim=-1)
    else:
        # Standard softmax (baseline or prior_softmax)
        attn_weights = F.softmax(attn_weights, dim=-1)

    # Store attention weights for analysis
    if config.get('capture_stats', True):
        config['attention_weights'] = attn_weights.detach().clone()
        config['last_stats'] = compute_sparsity_stats(attn_weights)

    # Downcast and apply dropout
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    # Compute output
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


def apply_attention_patch():
    """Apply the attention patch to GPT2."""
    gpt2_module.eager_attention_forward = patched_eager_attention_forward
    print("[PATCH] eager_attention_forward patched (v2)")


def remove_attention_patch():
    """Remove the attention patch."""
    gpt2_module.eager_attention_forward = _original_eager_attention_forward
    print("[PATCH] Restored original eager_attention_forward")


def set_attention_config(
    variant: str = 'baseline',
    lam: float = 0.0,
    gamma: float = 1.0,
    alpha: float = 1.5,
    prior_mode: str = 'centered',
    clip_value: float = 5.0,
    clear_weights: bool = True
):
    """Configure attention behavior."""
    global ATTENTION_CONFIG
    ATTENTION_CONFIG['variant'] = variant
    ATTENTION_CONFIG['lam'] = lam
    ATTENTION_CONFIG['gamma'] = gamma
    ATTENTION_CONFIG['alpha'] = alpha
    ATTENTION_CONFIG['prior_mode'] = prior_mode
    ATTENTION_CONFIG['clip_value'] = clip_value
    if clear_weights:
        ATTENTION_CONFIG['attention_weights'] = None
        ATTENTION_CONFIG['last_stats'] = None


def get_attention_stats() -> Dict[str, float]:
    """Get statistics from last forward pass."""
    return ATTENTION_CONFIG.get('last_stats', {})


def clear_attention_state():
    """Clear stored attention state."""
    ATTENTION_CONFIG['attention_weights'] = None
    ATTENTION_CONFIG['last_stats'] = None
