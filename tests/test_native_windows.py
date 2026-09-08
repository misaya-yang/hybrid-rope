import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,Olmo2Config,Olmo2ForCausalLM
from scripts.experiments.olmo_fast_screen.native_windows import native_bank,rephase_bank


@pytest.mark.parametrize('config_type,model_type',[(Qwen2Config,Qwen2ForCausalLM),(Olmo2Config,Olmo2ForCausalLM)])
def test_native_windows_remove_only_cross_window_formation_and_rephase_all_keys(config_type,model_type):
    torch.manual_seed(17)
    config=config_type(vocab_size=97,hidden_size=64,intermediate_size=96,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=8,rope_theta=10000.)
    config._attn_implementation='sdpa';model=model_type(config).eval()
    freq=1/(10000**(torch.arange(0,16,2,dtype=torch.float32)/16))
    native=dict(values_float32=freq.tolist(),gain=1.)
    target=dict(values_float32=(freq/2).tolist(),gain=1.1)
    ids=list(range(1,20));changed=ids.copy();changed[0]=30
    with torch.inference_mode():
        bank,bounds=native_bank(model,ids,native,8)
        other,other_bounds=native_bank(model,changed,native,8)
        assert bounds==other_bounds==[[0,8],[8,16],[16,19]]
        for (raw,values),(raw2,values2) in zip(bank,other):
            assert torch.equal(raw[...,8:,:],raw2[...,8:,:])
            assert torch.equal(values[...,8:,:],values2[...,8:,:])
        assert not torch.equal(bank[0][0][...,:8,:],other[0][0][...,:8,:])
        cache=rephase_bank(model,bank,target)
        assert cache.get_seq_length()==len(ids)
        pos=torch.arange(len(ids),dtype=torch.float64)
        phase=torch.exp(1j*pos[:,None]*(freq.double()/2)[None,:])
        for (raw,values),layer in zip(bank,cache.layers):
            z=torch.complex(raw.double()[...,:8],raw.double()[...,8:])*phase[None,None]
            expected=(1.1*torch.cat((z.real,z.imag),dim=-1)).float()
            torch.testing.assert_close(layer.keys,expected,rtol=2e-6,atol=2e-6)
            assert torch.equal(layer.values,values)
        out=model(torch.tensor([[20]]),position_ids=torch.tensor([[19]]),
            attention_mask=torch.ones(1,20,dtype=torch.long),past_key_values=cache,use_cache=True)
        assert out.past_key_values.get_seq_length()==20
        assert torch.isfinite(out.logits).all()
