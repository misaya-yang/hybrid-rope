import math
import numpy as np
import torch
from scripts.experiments.scale_transport.math import replay,estimate_beta,bounded_isotonic


def test_rotary_replay_matches_direct_rotation_and_causal_gqa():
    torch.manual_seed(718)
    H,Q,D,L=4,3,8,17
    q=torch.randn(H,Q,D);k=torch.randn(2,L,D);v=torch.randn(2,L,D);wo=torch.randn(13,H*D)
    freq=torch.tensor([1.,.3,.07,.01]);pos=torch.tensor([10,13,16]);gain=1.13
    def rot(x,p):
        ph=p[:,None]*freq;co=torch.cat([ph.cos(),ph.cos()],-1);si=torch.cat([ph.sin(),ph.sin()],-1)
        return x*co+torch.cat([-x[...,D//2:],x[...,:D//2]],-1)*si
    qr=rot(q,pos);kr=rot(k,torch.arange(L)).repeat_interleave(2,0);vr=v.repeat_interleave(2,0)
    logits=qr@kr.transpose(-1,-2)*(gain*gain/math.sqrt(D));logits.masked_fill_(pos[:,None]<torch.arange(L),-torch.inf)
    expected=(logits.softmax(-1)@vr).transpose(0,1).reshape(Q,H*D)@wo.T
    actual,profile=replay(q,k,v,wo,freq,gain,pos,response=True)
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    assert torch.isfinite(profile).all() and (profile>=0).all() and (profile[:,0]==0).all()


def test_quantile_transport_and_visibility_null():
    a=np.zeros(33);b=np.zeros(65);a[[2,6,14]]=1;b[[4,12,28]]=1
    assert abs(estimate_beta(a,b)['beta']-1)<1e-12
    assert abs(estimate_beta(a,a)['beta'])<1e-12
    a=np.r_[0.,np.ones(1024)];b=np.r_[0.,np.ones(2048)]
    assert estimate_beta(a,b)['beta']>.99 # stationary cutoff is NOT causal dilation evidence


def test_projection_is_constrained_optimum_in_small_case():
    from scipy.optimize import minimize
    rng=np.random.default_rng(172)
    for _ in range(10):
        native=np.exp(-np.arange(6)*.3);lo=native/4;hi=native
        target=rng.uniform(lo,hi);weights=rng.uniform(.1,2,6)
        got=bounded_isotonic(target,weights,lo,hi)
        objective=lambda x:np.sum(weights*(x-target)**2)
        ref=minimize(objective,native/2,bounds=list(zip(lo,hi)),constraints=[{'type':'ineq','fun':lambda x:x[:-1]-x[1:]}],method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
        assert ref.success and abs(objective(got)-ref.fun)<1e-8


def test_actual_qwen_attention_hook_layout_and_output():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
    config._attn_implementation='eager'
    model=Qwen2ForCausalLM(config).eval();seen=[]
    def hook(module,args,kwargs,result):
        hidden=kwargs['hidden_states'];L=hidden.shape[1];positions=torch.arange(L-57,L,8)
        q=module.q_proj(hidden[:,positions]).reshape(8,4,8).transpose(0,1)
        k=module.k_proj(hidden).reshape(L,2,8).transpose(0,1)
        v=module.v_proj(hidden).reshape(L,2,8).transpose(0,1)
        got,_=replay(q,k,v,module.o_proj.weight,model.model.rotary_emb.inv_freq,1.,positions)
        torch.testing.assert_close(got,result[0][0,positions],atol=1e-6,rtol=1e-5);seen.append(True)
    handle=model.model.layers[0].self_attn.register_forward_hook(hook,with_kwargs=True)
    with torch.inference_mode():
        result=model(torch.randint(0,64,(1,64)),use_cache=False,logits_to_keep=16)
    handle.remove()
    assert seen==[True] and result.logits.shape==(1,16,64)
