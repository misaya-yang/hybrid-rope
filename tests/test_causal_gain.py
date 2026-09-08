import math

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.experiments.olmo_fast_screen.causal_gain import CausalGain, query_factor


def test_length_law_has_native_prefix_and_original_terminal_multiplier():
    gain = 1+.1*math.log(4)
    factor = query_factor(torch.tensor([[0, 7, 15, 31]]), native_length=8, installed_gain=gain)
    effective = factor.double()*gain**2
    expected = torch.tensor([[1., 1., (1+.1*math.log(2))**2, gain**2]], dtype=torch.float64)
    assert torch.allclose(effective, expected, atol=1e-7, rtol=0)
    assert factor[0, -1] == 1


def test_hook_changes_q_not_k_and_decoding_uses_absolute_visible_count():
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=31, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=8)).eval()
    ids = torch.arange(32).remainder(31).unsqueeze(0)
    attn = model.model.layers[0].self_attn
    captured = {}
    def collect(key):
        def hook(module, args, output):
            captured[key] = output.clone()
        return hook
    qh = attn.q_proj.register_forward_hook(collect('raw_q'))
    kh = attn.k_proj.register_forward_hook(collect('raw_k'))
    gain = 1+.1*math.log(4)
    with torch.inference_mode():
        original = model(ids, use_cache=True)
        q, k = captured['raw_q'].clone(), captured['raw_k'].clone()
        with CausalGain(model, native_length=8, installed_gain=gain):
            after = attn.q_proj.register_forward_hook(collect('scaled_q'))
            result = model(ids, use_cache=True)
            factor = query_factor(torch.arange(32).unsqueeze(0), native_length=8, installed_gain=gain)
            assert torch.equal(captured['raw_k'], k)
            assert torch.equal(captured['scaled_q'], q*factor.unsqueeze(-1))
            assert not torch.equal(result.logits, original.logits)
            model(torch.tensor([[1]]), position_ids=torch.tensor([[32]]),
                past_key_values=result.past_key_values, use_cache=True)
            expected = query_factor(torch.tensor([[32]]), native_length=8, installed_gain=gain)
            assert torch.equal(captured['scaled_q'], captured['raw_q']*expected.unsqueeze(-1))
            after.remove()
    qh.remove(); kh.remove()
    assert not model.model.rotary_emb._forward_hooks
