import math
import numpy as np
import torch
from experiments.rotary_budget.build_tables import ARMS,build_table,active_table
from experiments.rotary_budget.eval_inputs import make_example
from experiments.native_rope_evq_150m.model import GPT,apply_rope,RotaryEmbedding

def test_tables_endpoints_and_reference():
    ref=(500000.**(-np.arange(32)/32)).astype(np.float32)
    np.testing.assert_array_equal(build_table('G32'),ref)
    for arm in ('G32','E32','G16','E16'):
        f=build_table(arm); f=f[f>0]
        np.testing.assert_array_equal(f[[0,-1]],ref[[0,-1]])
    np.testing.assert_array_equal(active_table(2,True),active_table(2))
    np.testing.assert_array_equal(active_table(16,True,0),active_table(16))
    np.testing.assert_array_equal(build_table('P16')[:16],ref[:16])
    np.testing.assert_array_equal(build_table('U16')[:16],ref[::2])

def test_pair_identity_norm_translation_and_initialization():
    config=dict(num_layers=1,hidden_size=64,num_heads=1,head_dim=64,intermediate_size=128,vocab_size=80,max_position_embeddings=64)
    states=[]
    for arm in ARMS:
        torch.manual_seed(42)
        model=GPT(config,torch.from_numpy(build_table(arm)))
        states.append([p.detach().clone() for p in model.parameters()])
        rope=RotaryEmbedding(64,64,torch.from_numpy(build_table(arm)))
        cos,sin=rope(64)
        x=torch.randn(1,1,64,64); y=apply_rope(x,cos,sin)
        torch.testing.assert_close(x.square().sum(-1),y.square().sum(-1),rtol=1e-6,atol=1e-5)
        if arm.endswith('16'):
            assert torch.equal(x[...,16:32],y[...,16:32])
            assert torch.equal(x[...,48:64],y[...,48:64])
        q=torch.randn(64); k=torch.randn(64)
        dot=lambda i,j:(apply_rope(q,cos[i],sin[i])*apply_rope(k,cos[j],sin[j])).sum()
        torch.testing.assert_close(dot(2,9),dot(22,29),atol=1e-5,rtol=1e-5)
    for state in states[1:]:
        assert all(torch.equal(a,b) for a,b in zip(state,states[0]))

def test_same_targets_shift_and_remote_boundary():
    doc=np.arange(9000); donor=np.arange(9000)+100000
    for length in (512,2048,4096,8192):
        x,y,meta=make_example(doc,length)
        np.testing.assert_array_equal(y,np.arange(7937,8193))
        np.testing.assert_array_equal(x[-256:]+1,y)
    intact,y,_=make_example(doc,4096)
    replaced,yr,_=make_example(doc,4096,donor)
    np.testing.assert_array_equal(replaced[-512:],intact[-512:])
    np.testing.assert_array_equal(yr,y)
    np.testing.assert_array_equal(replaced[:-512],donor[:3584])

def test_summary_rejects_mismatched_targets(tmp_path):
    import json
    from experiments.rotary_budget.summarize_budget import summarize
    rows=[]
    for length in (512,2048):
        for doc in range(512):
            rows.append(dict(arm='G32',training_seed=42,document_index=doc,context_length=length,condition='intact',
                target_token_count=256,sum_nll=256.,source_document_id=doc,target_sha256=str(doc),
                target_start=7937,target_end=8193,checkpoint_tokens=499974144,
                full_window_sum_nll=2048.,full_window_target_count=2048))
    p=tmp_path/'rows.jsonl';p.write_text('\n'.join(map(json.dumps,rows)))
    r=summarize([p]);assert r['status']=='PARTIAL' and not r['complete_seeds']
    rows[-1]['target_sha256']='wrong'
    p.write_text('\n'.join(map(json.dumps,rows)))
    import pytest
    with pytest.raises(ValueError,match='Targets'):summarize([p])
