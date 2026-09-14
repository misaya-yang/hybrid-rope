"""Independent dense checks of conditional replay and expanded-frequency scores."""
import math

import torch

from .select import phase_rotate, prepare_record, replay
from .worker import save, sha


def run(worker,job):
    torch.manual_seed(20260909)
    q=torch.randn(16,4,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(2,64,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    qp=torch.arange(60,64,device='cuda');kp=torch.arange(64,device='cuda')
    table=worker.tables['MrPro'];freq=torch.tensor(table['values_float32'],device='cuda')
    def scores(f):
        qr=phase_rotate(q,qp,f,table['gain']).float()
        kr=phase_rotate(k,kp,f,table['gain']).repeat_interleave(8,0).float()
        return [REDACTED_EMAIL](-1,-2)/math.sqrt(128)
    base=scores(freq);valid=kp[None]<=qp[:,None];base.masked_fill_(~valid[None],-torch.inf)
    p=base.softmax(-1);vv=v.repeat_interleave(8,0).float();out=p@vv
    selected=torch.arange(0,64,2,device='cuda');targets=kp%7==0
    record=dict(q_raw=q,k_raw=k[:,selected],v_raw=v[:,selected],query_positions=qp,key_positions=selected,
        selected_baseline_logits=base[...,selected],baseline_lse=base.logsumexp(-1),baseline_output=out,
        full_target_mass=p[...,targets].sum(-1),selected_target=targets[selected],valid=valid[:,selected])
    data=prepare_record(record,table)
    changed=freq.clone();changed[30]*=.7
    result=replay(data,changed)
    direct=base.clone();new=scores(changed);direct[...,selected]=new[...,selected]
    direct.masked_fill_(~valid[None],-torch.inf)
    direct_output=direct.softmax(-1)@vv
    replay_output=out+result['output_delta']
    conditional_error=(direct_output-replay_output).abs().max().item()
    assert conditional_error<3e-5,conditional_error
    # Expanded dimensions vs the arithmetic mean of two full logits in FP32.
    a=torch.randn(3,128,device='cuda');b=torch.randn(9,128,device='cuda')
    c=torch.randn(3,128,device='cuda');d=torch.randn(9,128,device='cuda')
    idx=torch.tensor(list(range(24,40))+list(range(88,104)),device='cuda')
    c[:,torch.tensor([x for x in range(128) if x not in idx.tolist()],device='cuda')]=a[:,torch.tensor([x for x in range(128) if x not in idx.tolist()],device='cuda')]
    d[:,torch.tensor([x for x in range(128) if x not in idx.tolist()],device='cuda')]=b[:,torch.tensor([x for x in range(128) if x not in idx.tolist()],device='cuda')]
    aa=a.clone();bb=b.clone();aa[:,idx]*=math.sqrt(.5);bb[:,idx]*=math.sqrt(.5)
    expanded=torch.cat((aa,c[:,idx]*math.sqrt(.5)),-1)@torch.cat((bb,d[:,idx]*math.sqrt(.5)),-1).T
    dual_error=(expanded-(a@b.T+c@d.T)/2).abs().max().item()
    assert dual_error<2e-5,dual_error
    worker.apply({'table':table})
    tokens=torch.tensor([worker.screen[0]['prompt_ids'][:512]],device='cuda')
    with torch.inference_mode():
        reference=worker.model(tokens,use_cache=False,logits_to_keep=8).logits.float()
        errors={}
        for kind in ('layer','group'):
            worker.apply(dict(operator=kind,layer=17,group=1,table=table,replacement=table))
            got=worker.model(tokens,use_cache=False,logits_to_keep=8).logits.float()
            errors[kind]=float((got-reference).abs().max())
            assert torch.equal(got,reference),(kind,errors[kind])
        worker.apply(dict(operator='dual_frequency',table=table,second_table=table))
        got=worker.model(tokens,use_cache=False,logits_to_keep=8).logits.float()
        errors['dual_frequency_bf16_rms']=float((got-reference).square().mean().sqrt())
        errors['dual_frequency_relative_rms']=float((got-reference).square().mean().sqrt()/reference.square().mean().sqrt())
        assert errors['dual_frequency_relative_rms']<.03,errors
    worker.apply({'table':table})
    result=dict(status='PASS',conditional_dense_max_error=conditional_error,dual_frequency_score_max_error=dual_error,
        model_identity_errors=errors,source_sha256=sha(__file__),scope='Algebra/integration checks; no efficacy claim')
    save(worker.root/'implementation_checks.json',result)
    return result
