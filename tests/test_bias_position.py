import math
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Olmo2Config, Olmo2ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from scripts.experiments.olmo_fast_screen.bias_position import BiasPositionTerm, augment, phases
from scripts.experiments.olmo_fast_screen.runtime import install


def complex_rotate(x,freq,pos,gain):
    half=x.shape[-1]//2
    z=torch.complex(x[...,:half],x[...,half:])
    z=z*torch.exp(1j*pos[:,None]*freq[None,:])[None,None,:,:]
    return gain*torch.cat((z.real,z.imag),dim=-1)


def test_augmented_logits_match_relative_bias_formula_with_gqa_and_long_positions():
    torch.manual_seed(8)
    dim=128;gain=1.1386
    read=torch.exp(-torch.arange(64,dtype=torch.float64)/7)
    prior=read.clone();prior[24:40]*=1.3
    indexes=torch.cat((torch.arange(24,40),torch.arange(88,104)))
    pos=torch.tensor([0,100,2048,16384,131070,131071,131072],dtype=torch.float64)
    q=torch.randn(1,4,3,dim,dtype=torch.float64);k=torch.randn(1,2,7,dim,dtype=torch.float64)
    v=torch.randn_like(k);qb=torch.randn(4,dim,dtype=torch.float64);kb=torch.randn(2,dim,dtype=torch.float64)
    qr=complex_rotate(q,read,pos[-3:],gain);kr=complex_rotate(k,read,pos,gain)
    qa,ka,va=augment(qr,kr,v,qb[:,indexes],kb[:,indexes],
        phases(prior[24:40],pos,gain,torch.float64),phases(read[24:40],pos,gain,torch.float64),'bias')
    assert qa.shape[-1]==ka.shape[-1]==va.shape[-1]==192
    expected=qr@kr.repeat_interleave(2,dim=1).transpose(-1,-2)/math.sqrt(dim)
    bqc=torch.complex(qb[:,:64],qb[:,64:]);bkc=torch.complex(kb[:,:64],kb[:,64:])
    for h in range(4):
        for t in range(3):
            for j in range(7):
                delta=pos[j]-pos[-3+t]
                correction=(bqc[h].conj()*bkc[h//2]*(torch.exp(1j*prior*delta)-torch.exp(1j*read*delta))).real.sum()
                expected[0,h,t,j]+=gain**2*correction/math.sqrt(dim)
    actual=qa@ka.repeat_interleave(2,dim=1).transpose(-1,-2)/math.sqrt(dim)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    mask=pos[None,:]<=pos[-3:,None]
    weights=expected.masked_fill(~mask,-torch.inf).softmax(-1)
    out=torch.nn.functional.scaled_dot_product_attention(qa,ka.repeat_interleave(2,dim=1),
        va.repeat_interleave(2,dim=1),attn_mask=mask,scale=1/math.sqrt(dim))
    torch.testing.assert_close(out[...,:dim],weights@v.repeat_interleave(2,dim=1),rtol=1e-10,atol=1e-10)
    assert torch.count_nonzero(out[...,dim:])==0


def tables(dim):
    freq=(10000**(-torch.arange(dim//2,dtype=torch.float32)/(dim//2))).tolist()
    prior=list(freq)
    for j in range(2,dim//2-2):prior[j]*=1.2
    return dict(values_float32=freq,gain=1.1),dict(values_float32=prior,gain=1.1)


def tiny_model(config_cls,model_cls):
    config=config_cls(vocab_size=97,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,rope_theta=10000.)
    config._attn_implementation='sdpa'
    return model_cls(config).eval()


def test_qwen_padding_cache_shape_and_registry_restoration():
    torch.manual_seed(5)
    model=tiny_model(Qwen2Config,Qwen2ForCausalLM)
    for layer in model.model.layers:
        layer.self_attn.q_proj.bias.data.normal_(std=.4)
        layer.self_attn.k_proj.bias.data.normal_(std=.4)
    read,prior=tables(16);install(model,read)
    ids=torch.randint(0,97,(1,17));original=ALL_ATTENTION_FUNCTIONS['sdpa']
    with torch.inference_mode():
        baseline=model(ids,use_cache=False).logits
        with BiasPositionTerm(model,read_table=read,prior_table=prior,mode='pad_control'):
            padded=model(ids,use_cache=False).logits
        torch.testing.assert_close(padded,baseline,rtol=2e-5,atol=2e-6)
        with BiasPositionTerm(model,read_table=read,prior_table=prior):
            first=model(ids,use_cache=True)
            assert first.past_key_values.layers[0].keys.shape[-1]==16
            second=model(torch.tensor([[3]]),past_key_values=first.past_key_values,
                position_ids=torch.tensor([[17]]),attention_mask=torch.ones(1,18,dtype=torch.long),use_cache=True)
            assert second.past_key_values.get_seq_length()==18
            assert torch.isfinite(second.logits).all()
        assert not torch.equal(first.logits,baseline)
    assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original
    with pytest.raises(ValueError,match='contiguous'),torch.inference_mode():
        with BiasPositionTerm(model,read_table=read,prior_table=prior):
            model(ids,position_ids=torch.arange(1,18)[None],use_cache=False)
    assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original


def test_biasless_olmo_uses_exact_original_path():
    model=tiny_model(Olmo2Config,Olmo2ForCausalLM)
    read,prior=tables(16);install(model,read);ids=torch.tensor([[3,4,5,6]])
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    with torch.inference_mode():
        expected=model(ids,use_cache=False).logits
        with BiasPositionTerm(model,read_table=read,prior_table=prior) as policy:
            assert policy.identity
            actual=model(ids,use_cache=False).logits
    assert torch.equal(actual,expected)
    assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original
