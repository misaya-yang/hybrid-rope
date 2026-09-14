"""Separate ideal relative-phase response from absolute-coordinate rounding."""
import json
import math
import numpy as np
import torch

from .select import prepare_record
from .worker import save


def run(worker,job):
    projection=json.loads((worker.root/'selection/E7_projection.json').read_text())
    previous=json.loads((worker.root/'selection/E7_nonlinear_check.json').read_text())
    old_records={(r['row'],r['layer']):r for r in previous['records']}
    results=[]
    with torch.inference_mode():
        for path in sorted((worker.root/'calibration').glob('natural_*/layer_*.pt')):
            r=torch.load(path,map_location='cpu',weights_only=False);d=prepare_record(r,worker.tables['MrPro'])
            q=d['q_raw'].float();k=d['k_raw'].repeat_interleave(8,0).float()
            a,b=q[...,24:40],q[...,88:104];x,y=k[...,24:40],k[...,88:104]
            c=a[:,:,None]*x[:,None]+b[:,:,None]*y[:,None]
            sine=a[:,:,None]*y[:,None]-b[:,:,None]*x[:,None]
            dist=d['query_positions'][:,None]-d['key_positions'][None]
            old=d['freq'][24:40];new=torch.tensor(projection['spec']['table']['values_float32'][24:40],device='cuda')
            phase=dist[...,None].float()*old
            movement=dist[...,None].float()*(new-old)
            dc=-2*(phase+movement/2).sin()*(movement/2).sin()
            ds=2*(phase+movement/2).cos()*(movement/2).sin()
            ideal=(c*dc[None]+sine*ds[None]).sum(-1)*d['gain']**2/math.sqrt(128)
            def absolute_score(freq):
                qp=d['query_positions'].float()[:,None]*freq
                kp=d['key_positions'].float()[:,None]*freq
                qa=(a*qp.cos()-b*qp.sin())*d['gain'];qb=(b*qp.cos()+a*qp.sin())*d['gain']
                ka=(x*kp.cos()-y*kp.sin())*d['gain'];kb=(y*kp.cos()+x*kp.sin())*d['gain']
                return (qa@ka.transpose(-1,-2)+qb@kb.transpose(-1,-2))/math.sqrt(128)
            fp32=absolute_score(new)-absolute_score(old)
            w=worker.model.model.layers[r['layer']].self_attn.o_proj.weight.float()
            base=d['baseline_output'].permute(1,0,2).reshape(q.shape[1],-1)@w.T
            denom=base.square().sum().clamp_min(1e-12)
            values={}
            for name,delta in [('ideal_relative',ideal),('fp32_absolute',fp32)]:
                delta.masked_fill_((dist>projection['window']['max_tokens'])[None]|(~d['valid'])[None],0)
                weights=d['oldp']*delta.exp()
                out=(d['residual_output']+torch.einsum('hqs,hsd->hqd',weights,d['v16']))/(d['residual_mass']+weights.sum(-1))[...,None]
                changed=(out-d['baseline_output']).permute(1,0,2).reshape(q.shape[1],-1)@w.T
                values[name]=float(changed.square().sum()/denom)
            results.append(dict(row=r['row_id'],layer=r['layer'],bf16_absolute=old_records[(r['row_id'],r['layer'])]['local_only'],**values))
    means={name:float(np.mean([r[name] for r in results])) for name in ('ideal_relative','fp32_absolute','bf16_absolute')}
    result=dict(status='COMPLETE',means=means,linear_prediction=projection['predicted_local_output_nmse'],records=results,
        scope='Same frozen baseline states and local key support; numerical diagnostic, not attribution of full-model generation failure')
    save(worker.root/'selection/E7_precision_breakdown.json',result)
    return result
