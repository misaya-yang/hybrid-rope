"""
Attention patching for variational sparse attention experiments.
Patches eager_attention_forward to inject distance prior and sparsemax.
"""
import torch
import torch.nn.functional as F
from entmax import sparsemax
from typing import Optional, Dict, Any
import transformers.models.gpt2.modeling_gpt2 as gpt2_module

# Store original function
_original_eager_attention_forward = gpt2_module.eager_attention_forward

# Global configuration storage
ATTENTION_CONFIG: Dict[str, Any] = {
    'variant': 'baseline',  # 'baseline', 'prior_softmax', 'prior_sparse'
    'lam': 8.0,             # Prior strength
    'gamma': 0.5,           # Temperature for sparsemax
    'alpha': 1.5,           # Distance prior exponent
    'attention_weights': None,  # Store for analysis
}


def compute_distance_prior(seq_len: int, alpha: float, device: torch.device) -> torch.Tensor:
    """
    Compute distance-based prior log D(Δ) = -α * log(Δ + 1).
    Returns a [seq_len, seq_len] tensor with -inf for future positions (causal mask).
    """
    indices = torch.arange(seq_len, device=device)
    delta = indices.unsqueeze(1) - indices.unsqueeze(0)  # [seq_len, seq_len]
    
    # Causal mask: only attend to current and past positions
    # delta > 0 means future positions, which should be masked out
    prior = torch.where(
        delta >= 0,
        -alpha * torch.log(delta + 1.0),
        torch.tensor(float('-inf'), device=device)
    )
    return prior


def patched_eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    """
    Patched version of eager_attention_forward that supports:
    - Baseline (standard softmax)
    - Prior-biased softmax
    - Prior-biased sparsemax
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
    if variant in ['prior_softmax', 'prior_sparse']:
        seq_len = query.size(-2)
        lam = config['lam']
        alpha = config['alpha']
        
        # Compute prior and expand to match batch/heads dimensions
        prior = compute_distance_prior(seq_len, alpha, query.device)
        # prior shape: [seq_len, seq_len]
        # Expand to [batch_size, num_heads, seq_len, seq_len]
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
    if config['attention_weights'] is None:
        # Store a copy for external analysis
        config['attention_weights'] = attn_weights.detach().clone()

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
    print("[PATCH] eager_attention_forward patched successfully")


def remove_attention_patch():
    """Remove the attention patch and restore original behavior."""
    gpt2_module.eager_attention_forward = _original_eager_attention_forward
    print("[PATCH] eager_attention_forward restored to original")


def set_attention_variant(
    variant: str,
    lam: float = 8.0,
    gamma: float = 0.5,
    alpha: float = 1.5,
    clear_weights: bool = True
):
    """
    Set the attention variant and parameters.
    
    Args:
        variant: 'baseline', 'prior_softmax', or 'prior_sparse'
        lam: Prior strength (λ)
        gamma: Temperature for sparsemax (γ)
        alpha: Distance prior exponent (α)
        clear_weights: Whether to clear stored attention weights
    """
    global ATTENTION_CONFIG
    ATTENTION_CONFIG['variant'] = variant
    ATTENTION_CONFIG['lam'] = lam
    ATTENTION_CONFIG['gamma'] = gamma
    ATTENTION_CONFIG['alpha'] = alpha
    if clear_weights:
        ATTENTION_CONFIG['attention_weights'] = None


def get_attention_weights() -> Optional[torch.Tensor]:
    """Get the stored attention weights from the last forward pass."""
    return ATTENTION_CONFIG['attention_weights']


def clear_attention_weights():
    """Clear the stored attention weights."""
    ATTENTION_CONFIG['attention_weights'] = None


def get_attention_stats() -> Dict[str, float]:
    """
    Compute statistics on stored attention weights.
    Returns dict with sparsity, exact_zeros, row_sum_error, etc.
    """
    weights = ATTENTION_CONFIG['attention_weights']
    if weights is None:
        return {
            'sparsity': 0.0,
            'exact_zeros': 0,
            'row_sum_error': float('inf'),
            'mean_val': 0.0,
            'std_val': 0.0,
        }
    
    # Sparsity: proportion of exact zeros
    exact_zeros = (weights == 0).sum().item()
    total_elements = weights.numel()
    sparsity = exact_zeros / total_elements
    
    # Row sum error (should be ~1.0 for valid attention)
    row_sums = weights.sum(dim=-1)
    row_sum_error = (row_sums - 1.0).abs().mean().item()
    
    return {
        'sparsity': sparsity,
        'exact_zeros': exact_zeros,
        'row_sum_error': row_sum_error,
        'mean_val': weights.mean().item(),
        'std_val': weights.std().item(),
    }
