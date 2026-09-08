"""Inference-only token chunking for Qwen2's positionwise gated MLP."""
from contextlib import contextmanager
import torch


def qualify_chunking(model, token_ids, chunk_size):
    """Check the actual first MLP's BF16 arithmetic on an uneven token chunk."""
    if chunk_size <= 0:
        return None
    if model.config.model_type != 'qwen2':
        raise ValueError('chunk qualification is scoped to Qwen2')
    ids = torch.tensor([token_ids[:chunk_size+17]], device=model.device)
    if ids.shape[1] <= chunk_size:
        raise ValueError('qualification input must exercise multiple chunks')
    with torch.inference_mode():
        hidden = model.get_input_embeddings()(ids)
        forward = model.model.layers[0].mlp.forward
        expected = forward(hidden)
        actual = chunk_forward(forward, hidden, chunk_size)
        delta = actual.float()-expected.float()
        relative_l2 = float(delta.norm()/expected.float().norm().clamp_min(1e-12))
        if not torch.isfinite(actual).all() or relative_l2 > .01:
            raise ValueError('chunked MLP numerical qualification failed')
        return dict(tokens=ids.shape[1], max_abs_error=float(delta.abs().max()),
            relative_l2_error=relative_l2, bitwise_equal=bool(torch.equal(actual,expected)),
            scope='First actual positionwise MLP arithmetic, not whole-generation equivalence')


def chunk_forward(forward, hidden_states, chunk_size):
    if hidden_states.ndim != 3:
        raise ValueError('expected batch, sequence, hidden dimensions')
    if hidden_states.shape[1] <= chunk_size:
        return forward(hidden_states)
    if torch.is_grad_enabled():
        raise RuntimeError('MLP chunking is inference-only')
    result = torch.empty_like(hidden_states)
    for start in range(0, hidden_states.shape[1], chunk_size):
        result[:, start:start+chunk_size] = forward(hidden_states[:, start:start+chunk_size])
    return result


@contextmanager
def tokenwise_mlp_chunks(model, chunk_size):
    if chunk_size == 0:
        yield
        return
    if chunk_size < 1 or model.config.model_type != 'qwen2':
        raise ValueError('positive token chunks currently reviewed for Qwen2 only')
    modules = [layer.mlp for layer in model.model.layers]
    if any(type(mlp).__name__ != 'Qwen2MLP' for mlp in modules):
        raise ValueError('unreviewed MLP implementation')
    original = []
    try:
        for mlp in modules:
            forward = mlp.forward
            original.append((mlp, forward))
            def wrapped(hidden_states, _forward=forward):
                return chunk_forward(_forward, hidden_states, chunk_size)
            mlp.forward = wrapped
        yield
    finally:
        for mlp, forward in original:
            mlp.forward = forward
