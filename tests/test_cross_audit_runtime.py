import math
import numpy as np
import pytest
import torch
from scripts.experiments.cross_audit.tables import native_table, transform, install_static, verify_static


@pytest.mark.parametrize('method', ['mrpro', 'mruni'])
def test_mr_prefix_products_and_endpoints(method):
    source = native_table(128, 500000.)
    out, gain, meta = transform(source, dim=128, base=500000., reference_length=4096, scale=4., method=method)
    lo, hi = meta['low'], meta['high']
    n = hi-lo
    factors = [1.] * 64
    for j in range(lo, hi):
        factors[j] = 4 ** (1/n if method == 'mruni' else 2*(j-lo+1)/(n*(n+1)))
    expected = np.array([source[j] / math.prod(factors[:j]) for j in range(64)])
    np.testing.assert_allclose(out, expected, rtol=1e-6)
    assert out[lo] == source[lo]
    assert out[hi] == source[hi]/4
    assert gain == 1+.1*math.log(4)


@pytest.mark.parametrize('kind', ['olmo2', 'qwen2'])
def test_static_table_and_cached_decode_equal_full_prefill(kind):
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(7)
    cfg = AutoConfig.for_model(kind, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        vocab_size=41, max_position_embeddings=16, rope_theta=500000.)
    cfg._attn_implementation = 'eager'  # tiny CPU correctness test only
    model = AutoModelForCausalLM.from_config(cfg).eval()
    values = native_table(16, 500000.) / 4
    install_static(model, values, 1.138629436111989)
    ids = torch.tensor([[1, 5, 9, 3, 8, 2]])
    with torch.no_grad():
        full = model(ids, use_cache=False).logits[:, -1]
        prefix = model(ids[:, :-1], use_cache=True)
        cached = model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1]
    torch.testing.assert_close(cached, full, atol=1e-6, rtol=1e-5)
    verify_static(model, values, 1.138629436111989)
    model.model.rotary_emb.inv_freq[0] *= .5
    with pytest.raises(RuntimeError, match='drift'):
        verify_static(model, values, 1.138629436111989)


def test_chunked_shifted_ce_value_and_gradients_match_dense():
    from transformers import Olmo2Config, Olmo2ForCausalLM
    from scripts.experiments.cross_audit.training import causal_loss
    torch.manual_seed(19)
    cfg=Olmo2Config(hidden_size=16,intermediate_size=24,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,vocab_size=23)
    cfg._attn_implementation='eager'
    model=Olmo2ForCausalLM(cfg)
    ids=torch.tensor([[1,2,3,4,5,6]])
    labels=ids.clone();labels[:,:3]=-100
    dense=model(ids,use_cache=False).logits
    expected=torch.nn.functional.cross_entropy(dense[:,:-1].reshape(-1,23),labels[:,1:].reshape(-1))
    expected.backward()
    gradients={n:p.grad.clone() for n,p in model.named_parameters()}
    model.zero_grad()
    got,n=causal_loss(model,ids,labels,chunk_size=2)
    got.backward()
    assert n==3
    torch.testing.assert_close(got,expected)
    for name,p in model.named_parameters():
        torch.testing.assert_close(p.grad,gradients[name],atol=2e-7,rtol=1e-5)


def test_native_kl_normalizes_positions_not_sequence_or_batch():
    from transformers import Olmo2Config, Olmo2ForCausalLM
    from scripts.experiments.cross_audit.training import native_kl
    cfg=Olmo2Config(hidden_size=16,intermediate_size=24,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,vocab_size=23)
    cfg._attn_implementation='eager';model=Olmo2ForCausalLM(cfg)
    ids=torch.tensor([[1,2,3,4,5,6]]);pos=torch.tensor([0,2,4])
    p=torch.softmax(torch.randn(3,23),-1)
    got,n=native_kl(model,ids,pos,p)
    logits=model(ids,use_cache=False).logits[0,pos]
    expected=(p*(p.log()-logits.log_softmax(-1))).sum(-1).mean()
    torch.testing.assert_close(got,expected)
    assert n==3
    doubled,_=native_kl(model,ids,torch.cat([pos,pos]),torch.cat([p,p]))
    torch.testing.assert_close(doubled,got)
    with pytest.raises(ValueError,match='prediction positions'):
        native_kl(model,ids,torch.tensor([6]),p[:1])


@pytest.mark.parametrize('regime',['full','lora'])
def test_complete_training_step_is_finite_and_changes_trainable_weights(regime):
    from transformers import Olmo2Config, Olmo2ForCausalLM
    from scripts.experiments.cross_audit.training import configure_training,update
    cfg=Olmo2Config(hidden_size=16,intermediate_size=24,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,vocab_size=23)
    cfg._attn_implementation='eager';model=Olmo2ForCausalLM(cfg)
    model,wrapper=configure_training(model,regime,rank=2)
    trainable={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    ids=torch.tensor([[1,2,3,4,5,6]]);labels=ids.clone();labels[:,:3]=-100
    receipt=update(model,optimizer,ids,[(ids,labels)],(ids,torch.tensor([0,2]),torch.full((2,23),1/23)))
    assert receipt['answer_prediction_tokens']==3
    assert receipt['native_prediction_positions']==2
    assert any(not torch.equal(p,trainable[n]) for n,p in model.named_parameters() if n in trainable)
