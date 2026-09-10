"""E7 local-output constrained direction, with signed cross-slot derivatives."""
import json
import math

import numpy as np
import torch

from .select import prepare_record, replay
from .worker import read_rows, save, sha


def binding_window(worker):
    lines=[]
    for row in read_rows(worker.prepared/'prompts.jsonl'):
        if row['row_id'].startswith('niah_multikey_2_') and row['row_id'].endswith(('_0','_1')):
            lines.extend(line for line in row['prompt_text'].splitlines() if line.startswith('One of the special magic numbers for '))
    if not lines:raise ValueError('no literal primitive binding records')
    ids=worker.tokenizer(lines,add_special_tokens=False)['input_ids']
    longest=max(range(len(ids)),key=lambda i:len(ids[i]))
    return len(ids[longest]),dict(records=len(lines),max_tokens=len(ids[longest]),max_record=lines[longest],
        rule='Maximum complete primitive binding record length over fixed calibration row IDs 0/1')


def run(worker,job):
    selection=json.loads((worker.root/'selection/selection.json').read_text())
    proposed=json.loads((worker.root/'selection/proposals.json').read_text())
    mr=np.array(worker.tables['MrPro']['values_float32'],dtype=np.float64)
    bm=np.array(worker.tables['MrProBM']['values_float32'],dtype=np.float64)
    direction=bm[24:40]-mr[24:40]
    w,w_record=binding_window(worker)
    save(worker.root/'binding_window.json',w_record)
    hess=np.zeros((16,16));budgets={0:[],1:[]};count=0
    with torch.inference_mode():
        for path in sorted((worker.root/'calibration').glob('natural_*/layer_*.pt')):
            record=torch.load(path,map_location='cpu',weights_only=False)
            data=prepare_record(record,worker.tables['MrPro'])
            q=data['q_raw'].float();k=data['k_raw'].repeat_interleave(8,0).float()
            delta=data['query_positions'][:,None]-data['key_positions'][None,:]
            local=(delta>=0)&(delta<=w)
            a=q[...,24:40];b=q[...,88:104];x=k[...,24:40];y=k[...,88:104]
            c=a[:,:,None,:]*x[:,None,:,:]+b[:,:,None,:]*y[:,None,:,:]
            d=a[:,:,None,:]*y[:,None,:,:]-b[:,:,None,:]*x[:,None,:,:]
            phase=delta[...,None].float()*data['freq'][24:40]
            dz=(-c*phase.sin()[None]+d*phase.cos()[None])*delta[None,:,:,None]*torch.tensor(direction,device='cuda',dtype=torch.float32)
            dz*=worker.tables['MrPro']['gain']**2/math.sqrt(128)
            dz*=local[None,:,:,None]
            # d softmax(z)V = sum_k p_k (V_k - O) dz_k, including denominator.
            weighted=dz*data['oldp'][...,None]
            j=torch.einsum('hqsk,hsd->khqd',weighted,data['v16'])
            j-=torch.einsum('hqk,hqd->khqd',weighted.sum(2),data['baseline_output'])
            jo=j.permute(0,2,1,3).reshape(16,q.shape[1],-1)
            weight=worker.model.model.layers[record['layer']].self_attn.o_proj.weight.float()
            jo=jo@weight.T
            original=data['baseline_output'].permute(1,0,2).reshape(q.shape[1],-1)@weight.T
            denom=original.square().sum().clamp_min(1e-12)
            jflat=jo.reshape(16,-1)/denom.sqrt()
            hess+=(jflat@jflat.T).cpu().double().numpy();count+=1
            step_cost=[]
            for name in selection['e1']:
                rr=replay(data,proposed[name],max_distance=w)
                projected=rr['output_delta'].permute(1,0,2).reshape(q.shape[1],-1)@weight.T
                step_cost.append(float(projected.square().sum()/denom))
            budgets[record['metadata']['split']].append(min(step_cost))
    hess/=count
    eig,u=np.linalg.eigh((hess+hess.T)/2);eig=np.maximum(eig,0)
    budget=min(np.mean(x) for x in budgets.values())
    ones=np.ones(16)
    def solution(lam):return u@((u.T@ones)/(1+lam*eig))
    def energy(x):return float(x@hess@x)
    if energy(ones)<=budget:
        x=ones;lam=0.
    else:
        low,high=0.,1.
        while energy(solution(high))>budget:high*=2
        for _ in range(70):
            mid=(low+high)/2
            if energy(solution(mid))>budget:low=mid
            else:high=mid
        lam=high;x=solution(lam)
    max_phase=float(np.max(np.abs(direction*x))*w)
    radial=min(1.,.25/max_phase) if max_phase else 1.
    x*=radial
    values=mr.copy();values[24:40]+=direction*x
    if np.min(values)<0 or not np.isfinite(values).all():raise ValueError('E7 projection not a finite nonnegative table')
    spec=dict(operator='static',table=dict(values_float32=values.astype(np.float32).tolist(),gain=worker.tables['MrPro']['gain']))
    result=dict(status='COMPLETE',window=w_record,records=count,hessian=hess.tolist(),budget=float(budget),
        budget_split_means={str(k):float(np.mean(v)) for k,v in budgets.items()},x=x.tolist(),lambda_value=lam,
        predicted_local_output_nmse=energy(x),bm_predicted_local_output_nmse=energy(ones),
        max_local_phase_delta=float(np.max(np.abs(direction*x))*w),radial_cap=radial,spec=spec,
        source_sha256=sha(__file__),scope='Signed local linear response only; full long-prefix effect must be measured')
    save(worker.root/'selection/E7_projection.json',result)
    save(worker.root/'queue/019_E7_local_projection.json',dict(id='E7_local_projection',spec=spec,panel='small',nll_docs=4))
    save(worker.root/'queue/040_E9_distance.json',dict(id='E9_distance',spec=dict(operator='distance',window=w,table=worker.tables['MrPro']),panel='small',nll_docs=4))
    save(worker.root/'queue/041_E10_dual_frequency.json',dict(id='E10_dual_frequency',spec=dict(operator='dual_frequency',table=worker.tables['MrPro']),panel='small',nll_docs=4))
    return result
