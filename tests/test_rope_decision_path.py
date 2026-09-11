"""Small CPU checks of the actual Transformers rotary/decoder path."""
import numpy as np
import pytest
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM, Qwen2Config, Qwen2ForCausalLM

from experiments.curvature_20260910.model import FrozenRoPE


def frozen_model(kind, dtype=torch.float32):
    torch.manual_seed(913)
    conf, cls = (Olmo2Config, Olmo2ForCausalLM) if kind == 'olmo2' else (Qwen2Config, Qwen2ForCausalLM)
    cfg = conf(vocab_size=96, hidden_size=256, intermediate_size=320,
               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
               max_position_embeddings=512, eos_token_id=2, pad_token_id=0,
               attention_dropout=0.0)
    model = cls(cfg).to(dtype=dtype).eval().requires_grad_(False)
    f = FrozenRoPE.__new__(FrozenRoPE)
    f.model, f.rotary = model, model.model.rotary_emb
    f.device, f.dtype = 'cpu', dtype
    f.native_inv_freq = f.rotary.inv_freq.float().numpy().copy()
    f._grad_patched = False
    return f


@pytest.mark.parametrize('kind', ['olmo2', 'qwen2'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_stock_forward_and_greedy_parity(kind, dtype):
    f = frozen_model(kind, dtype)
    # Real model loading reinstalls FP32 frequency tables after dtype conversion.
    freq = (500000.**(-np.arange(64)/64)).astype(np.float32)
    f.install(freq, 1.138629436111989)
    ids = torch.tensor([[7, 13, 22, 31, 41, 8, 11, 5]])
    with torch.no_grad():
        before = f.model(ids, use_cache=False).logits
        generated = f.model.generate(ids, max_new_tokens=6, do_sample=False)
        c0, s0 = f.rotary(torch.zeros(1,8,256,dtype=dtype), torch.arange(8)[None])
    f.enable_grad_path()
    with torch.no_grad():
        after = f.model(ids, use_cache=False).logits
        generated_after = f.model.generate(ids, max_new_tokens=6, do_sample=False)
        c1, s1 = f.rotary(torch.zeros(1,8,256,dtype=dtype), torch.arange(8)[None])
    assert c0.dtype == c1.dtype
    assert torch.equal(c0,c1) and torch.equal(s0,s1)
    assert torch.equal(before,after)
    assert torch.equal(generated,generated_after)
    leaf = f.install(freq, 1.138629436111989, track_grad=True)
    loss = f.logits(ids,2,want_grad=True).square().mean()
    g, = torch.autograd.grad(loss,leaf)
    assert torch.isfinite(g).all() and g.abs().max() > 0


@pytest.mark.parametrize('kind', ['olmo2', 'qwen2'])
def test_full_model_margin_directional_derivative(kind):
    f = frozen_model(kind)
    f.enable_grad_path()
    ids = torch.tensor([[7,13,22,31,41,8,11,5,37,14,25,44]])
    base = torch.tensor(10000.**(-np.arange(64)/64),dtype=torch.float32)
    direction = torch.zeros(64)
    direction[:8] = torch.linspace(-.7,.9,8)

    def margin(a):
        f.rotary.inv_freq = base * torch.exp(-np.log(4)*a*direction)
        f.rotary.attention_scaling = torch.exp(a*.2) * 1.13
        logits = f.model(ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        return logits[9]-logits[12]

    a = torch.tensor(0.,requires_grad=True)
    val = margin(a)
    analytic, = torch.autograd.grad(val,a)
    h = .002
    with torch.no_grad():
        numeric = (margin(torch.tensor(h))-margin(torch.tensor(-h)))/(2*h)
    assert abs(float(analytic)) > 1e-5
    torch.testing.assert_close(analytic,numeric,rtol=.035,atol=2e-4)


def test_frequency_nll_gradient_has_next_token_alignment():
    f = frozen_model('qwen2')
    ids = torch.tensor([[7,13,22,31,41,8,11,5]])
    value, _ = f.grad_wrt_freq(ids,keep=4)
    expected = float(f.nll(ids,keep=4))
    assert value == pytest.approx(expected, abs=1e-6)
