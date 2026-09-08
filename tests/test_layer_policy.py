import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.experiments.cross_audit.tables import tensor_sha
from scripts.experiments.olmo_fast_screen.layer_policy import LayerTablePolicy
from scripts.experiments.olmo_fast_screen.runtime import install


def test_uniform_layer_policy_is_bitwise_stock_for_prefill_and_cached_decode():
    torch.manual_seed(912)
    config = Qwen2Config(vocab_size=31, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32768, rope_theta=1e6)
    model = Qwen2ForCausalLM(config).eval()
    values = model.model.rotary_emb.inv_freq.detach().numpy().copy()
    tables = {name: dict(values_float32=f.tolist(), gain=1.13, tensor_sha256=tensor_sha(f))
              for name, f in [('M', values), ('B', values*np.float32(.7))]}
    ids = torch.tensor([[1, 4, 8, 2]])
    positions = torch.tensor([[0, 17, 35000, 130999]])
    with torch.inference_mode():
        for name in tables:
            install(model, tables[name])
            expected = model(ids, position_ids=positions, use_cache=True)
            follow = model(torch.tensor([[3]]), position_ids=torch.tensor([[131000]]),
                past_key_values=expected.past_key_values, use_cache=True)
            with LayerTablePolicy(model, tables, [name, name]):
                actual = model(ids, position_ids=positions, use_cache=True)
                decoded = model(torch.tensor([[3]]), position_ids=torch.tensor([[131000]]),
                    past_key_values=actual.past_key_values, use_cache=True)
            assert torch.equal(actual.logits, expected.logits)
            assert torch.equal(decoded.logits, follow.logits)


def test_mixed_policy_assigns_different_rotations_and_removes_hooks():
    config = Qwen2Config(vocab_size=31, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32768, rope_theta=1e6)
    model = Qwen2ForCausalLM(config).eval()
    f = model.model.rotary_emb.inv_freq.detach().numpy().copy()
    tables = {name: dict(values_float32=v.tolist(), gain=1.13, tensor_sha256=tensor_sha(v))
              for name, v in [('M', f), ('B', f*np.float32(.7))]}
    seen = []
    with LayerTablePolicy(model, tables, ['M', 'B']):
        handles = [layer.self_attn.register_forward_pre_hook(
            lambda m, a, k: seen.append(k['position_embeddings'][0].clone()), with_kwargs=True)
            for layer in model.model.layers]
        with torch.inference_mode():
            model(torch.tensor([[1, 2]]), position_ids=torch.tensor([[0, 130999]]))
        for handle in handles:
            handle.remove()
    assert len(seen) == 2 and not torch.equal(seen[0], seen[1])
    assert not model.model.rotary_emb._forward_hooks
    assert all(not layer.self_attn._forward_pre_hooks for layer in model.model.layers)
