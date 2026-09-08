"""CPU integration tests; these do not qualify the CUDA Flash causal convention."""
import copy
import gc
import weakref

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Olmo2Config, Olmo2ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

from scripts.experiments.olmo_fast_screen import causal_flash as cf


def lower_right_reference(query, key, value, scale):
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    q_len, k_len = query.shape[-2], key.shape[-2]
    query_positions = torch.arange(k_len - q_len, k_len, device=query.device)
    visible = torch.arange(k_len, device=query.device)[None, :] <= query_positions[:, None]
    logits = (query.double() @ key.double().transpose(-1, -2)) * scale
    output = logits.masked_fill(~visible, -torch.inf).softmax(-1) @ value.double()
    return output.to(query.dtype).transpose(1, 2).contiguous()


def tiny_model(config_type=Qwen2Config, model_type=Qwen2ForCausalLM):
    config = config_type(vocab_size=97, hidden_size=64, intermediate_size=96,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, rope_theta=10000.)
    config._attn_implementation = 'sdpa'
    return model_type(config).eval()


@pytest.mark.parametrize('config_type,model_type', [
    (Qwen2Config, Qwen2ForCausalLM), (Olmo2Config, Olmo2ForCausalLM)])
def test_cached_chunks_match_full_causal_forward(monkeypatch, config_type, model_type):
    torch.manual_seed(29)
    model = tiny_model(config_type, model_type)
    ids = torch.randint(0, 97, (1, 13))
    calls = []

    def reference(q, k, v, scale):
        calls.append((q.shape[-2], k.shape[-2], q.shape[1], k.shape[1], scale))
        return lower_right_reference(q, k, v, scale)

    monkeypatch.setattr(cf, 'flash_causal', reference)
    original_sdpa = ALL_ATTENTION_FUNCTIONS['sdpa']
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits
        with cf.CausalFlash(model):
            assert model.config._attn_implementation == cf.ATTENTION_NAME
            assert all(layer.self_attn.config._attn_implementation == cf.ATTENTION_NAME
                       for layer in model.model.layers)
            full = model(ids, use_cache=False).logits
            cache = None
            outputs = []
            for begin, end in ((0, 4), (4, 9), (9, 12), (12, 13)):
                result = model(ids[:, begin:end], past_key_values=cache, use_cache=True,
                               position_ids=torch.arange(begin, end)[None],
                               attention_mask=torch.ones(1, end, dtype=torch.long))
                outputs.append(result.logits)
                cache = result.past_key_values
                assert cache.get_seq_length() == end
            torch.testing.assert_close(torch.cat(outputs, dim=1), full, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(full, expected, rtol=3e-5, atol=3e-6)
        restored = model(ids, use_cache=False).logits
    assert torch.equal(restored, expected)
    assert model.config._attn_implementation == 'sdpa'
    assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original_sdpa
    assert (5, 9, 4, 2, .25) in calls
    assert (1, 13, 4, 2, .25) in calls


@pytest.mark.parametrize('bad_mask', [torch.tensor([[1, 0, 1]]),
                                     torch.ones(1, 2), torch.ones(1, 1, 3, 3),
                                     {'full_attention': None}])
def test_rejects_padding_custom_masks_and_restores(monkeypatch, bad_mask):
    model = tiny_model()
    monkeypatch.setattr(cf, 'flash_causal', lambda *args: pytest.fail('invalid mask reached attention'))
    with pytest.raises(ValueError, match='unpadded'), torch.inference_mode():
        with cf.CausalFlash(model):
            model(torch.tensor([[1, 2, 3]]), attention_mask=bad_mask)
    assert model.config._attn_implementation == 'sdpa'
    assert not model.model._forward_pre_hooks


@pytest.mark.parametrize('kwargs', [dict(position_ids=torch.tensor([[0, 2, 3]])),
                                  dict(cache_position=torch.tensor([1, 2, 3]))])
def test_rejects_noncontiguous_positions(monkeypatch, kwargs):
    model = tiny_model()
    monkeypatch.setattr(cf, 'flash_causal', lambda *args: pytest.fail('invalid positions reached attention'))
    with pytest.raises(ValueError, match='contiguous'), torch.inference_mode(), cf.CausalFlash(model):
        model(torch.tensor([[1, 2, 3]]), **kwargs)


def test_distinct_configs_restored_even_on_exception():
    model = tiny_model()
    attn = model.model.layers[0].self_attn
    attn.config = copy.deepcopy(model.config)
    attn.config._attn_implementation = 'eager'
    with pytest.raises(RuntimeError, match='caller failure'):
        with cf.CausalFlash(model):
            assert attn.config._attn_implementation == cf.ATTENTION_NAME
            raise RuntimeError('caller failure')
    assert model.config._attn_implementation == 'sdpa'
    assert attn.config._attn_implementation == 'eager'
    assert not model.model._forward_pre_hooks


def test_inference_only_and_batch_one(monkeypatch):
    model = tiny_model()
    monkeypatch.setattr(cf, 'flash_causal', lower_right_reference)
    with pytest.raises(ValueError, match='inference-only'), cf.CausalFlash(model):
        model(torch.tensor([[1, 2, 3]]))
    with pytest.raises(ValueError, match='one nonempty'), torch.inference_mode(), cf.CausalFlash(model):
        model(torch.ones(2, 3, dtype=torch.long))


def test_free_registry_functions_do_not_retain_model():
    model = tiny_model()
    reference = weakref.ref(model)
    with cf.CausalFlash(model):
        pass
    assert ALL_ATTENTION_FUNCTIONS[cf.ATTENTION_NAME] is cf._attention
    assert ALL_MASK_ATTENTION_FUNCTIONS[cf.ATTENTION_NAME] is cf._mask
    del model
    gc.collect()
    assert reference() is None


def test_flash_helper_rejects_cpu_without_fallback():
    q = torch.zeros(1, 4, 2, 16, dtype=torch.bfloat16)
    k = torch.zeros(1, 2, 5, 16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='CUDA'), torch.inference_mode():
        cf.flash_causal(q, k, k, .25)
