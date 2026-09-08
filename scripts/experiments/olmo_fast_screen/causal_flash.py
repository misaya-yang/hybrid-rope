"""Mask-free cached causal attention through PyTorch's CUDA Flash operator.

The private operator uses BLHD tensors and lower-right causal alignment for
cached chunks. That CUDA alignment must be qualified on the serving runtime;
the CPU tests only verify the HF integration against an independent reference.
"""
import inspect
import math

import torch
from transformers import AttentionInterface, AttentionMaskInterface, DynamicCache


ATTENTION_NAME = 'hybrid_rope_causal_flash'


def flash_causal(query, key, value, scale):
    """Attend BHLD Q/K/V with lower-right causality; return BLHD, without fallback."""
    if torch.is_grad_enabled():
        raise ValueError('causal Flash is inference-only')
    if any(x.ndim != 4 for x in (query, key, value)):
        raise ValueError('Q/K/V must have shape B,H,L,D')
    batch, heads, q_len, dim = query.shape
    if (batch != 1 or key.shape[0] != 1 or value.shape[0] != 1
            or key.shape != value.shape or key.shape[-1] != dim
            or key.shape[1] == 0 or heads == 0 or heads % key.shape[1]
            or q_len == 0 or key.shape[-2] < q_len):
        raise ValueError('requires B1, matching K/V, GQA heads and 0 < Q length <= K length')
    if (any(x.device != query.device or x.dtype != query.dtype for x in (key, value))
            or query.device.type != 'cuda' or query.dtype not in (torch.float16, torch.bfloat16)):
        raise ValueError('requires CUDA Q/K/V with one shared FP16 or BF16 dtype')
    if dim % 8 or dim > 256 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('requires Flash head width divisible by 8 and <= 256, and positive finite scale')
    # Keep compact GQA keys/values. Flash implements the query-to-KV head mapping.
    q, k, v = (x.transpose(1, 2) for x in (query, key, value))
    return torch.ops.aten._flash_attention_forward(
        q, k, v, None, None, q_len, key.shape[-2], 0.0, True, False,
        scale=scale)[0]


def _unpadded_mask(attention_mask, length=None):
    if attention_mask is not None:
        if (not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2
                or attention_mask.shape[0] != 1
                or (length is not None and attention_mask.shape[1] != length)
                or not torch.all(attention_mask == 1)):
            raise ValueError('only an unpadded B1 all-ones 2D attention mask is supported')


def _mask(batch_size, attention_mask=None, kv_offset=0, **kwargs):
    if batch_size != 1 or kv_offset != 0:
        raise ValueError('requires B1 full-history attention from position zero')
    # HF may wrap the causal function in a packing check even for a plain
    # sequence. The model pre-hook already rejects packing/custom input masks.
    _unpadded_mask(attention_mask)
    return None


def _attention(module, query, key, value, attention_mask, dropout=0.0,
               scaling=None, is_causal=None, sliding_window=None, **kwargs):
    if module.training or torch.is_grad_enabled() or dropout != 0:
        raise ValueError('causal Flash is inference-only with dropout zero')
    if attention_mask is not None or is_causal is False or sliding_window is not None:
        raise ValueError('requires mask-free, full-history causal attention')
    scale = query.shape[-1] ** -0.5 if scaling is None else scaling
    return flash_causal(query, key, value, scale), None


def _validate_inputs(module, args, kwargs):
    # Binding also validates masks/positions supplied as positional arguments.
    bound = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
    tokens = bound.get('input_ids')
    if tokens is None:
        tokens = bound.get('inputs_embeds')
    if tokens is None or tokens.shape[0] != 1 or tokens.shape[1] == 0:
        raise ValueError('requires one nonempty unpadded sequence')
    if module.training or torch.is_grad_enabled():
        raise ValueError('causal Flash is inference-only')
    cache = bound.get('past_key_values')
    if cache is not None and not isinstance(cache, DynamicCache):
        raise ValueError('only full-history DynamicCache is supported')
    start = cache.get_seq_length() if cache is not None else 0
    expected = torch.arange(start, start + tokens.shape[1], device=tokens.device)
    for name, positions in (('position_ids', expected.unsqueeze(0)), ('cache_position', expected)):
        supplied = bound.get(name)
        if supplied is not None and (supplied.shape != positions.shape or not torch.equal(supplied, positions)):
            raise ValueError('cache/query positions must be contiguous from zero')
    _unpadded_mask(bound.get('attention_mask'), start + tokens.shape[1])


# These global entries never capture a model or a context instance.
AttentionInterface.register(ATTENTION_NAME, _attention)
AttentionMaskInterface.register(ATTENTION_NAME, _mask)


class CausalFlash:
    """Temporarily select causal Flash for an eval-mode Qwen2/OLMo2 model.

    Forward calls must run under no_grad/inference_mode, with B1, no padding,
    full-history DynamicCache and contiguous global positions from zero.
    """
    def __init__(self, model):
        self.model = model
        self.core = getattr(model, 'model', model)
        self.saved = None
        self.handle = None

    def __enter__(self):
        if self.saved is not None:
            raise RuntimeError('CausalFlash context is already active')
        if self.model.config.model_type not in ('qwen2', 'olmo2'):
            raise ValueError('only Qwen2 and OLMo2 are reviewed')
        if (getattr(self.model.config, 'use_sliding_window', False)
                or any(getattr(layer.self_attn, 'sliding_window', None) is not None
                       for layer in self.core.layers)):
            raise ValueError('sliding-window attention is not supported')
        configs = {id(module.config): module.config for module in self.model.modules()
                   if getattr(module, 'config', None) is not None}
        self.saved = [(config, config._attn_implementation) for config in configs.values()]
        try:
            for config, _ in self.saved:
                config._attn_implementation = ATTENTION_NAME
            self.handle = self.core.register_forward_pre_hook(_validate_inputs, with_kwargs=True)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self.saved is not None:
            for config, original in self.saved:
                config._attn_implementation = original
            self.saved = None
