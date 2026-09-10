"""Check E7's nonlinear finite response before interpreting its linear constraint."""
import json
import numpy as np
import torch

from .select import prepare_record,replay
from .worker import save


def run(worker,job):
    projection=json.loads((worker.root/'selection/E7_projection.json').read_text())
    freq=projection['spec']['table']['values_float32'];window=projection['window']['max_tokens']
    results=[]
    with torch.inference_mode():
        for path in sorted((worker.root/'calibration').glob('natural_*/layer_*.pt')):
            record=torch.load(path,map_location='cpu',weights_only=False)
            data=prepare_record(record,worker.tables['MrPro'])
            w=worker.model.model.layers[record['layer']].self_attn.o_proj.weight.float()
            q=data['q_raw'].shape[1]
            original=data['baseline_output'].permute(1,0,2).reshape(q,-1)@w.T
            denom=original.square().sum().clamp_min(1e-12)
            values={}
            for name,cap in [('local_only',window),('all_selected_keys',None)]:
                rr=replay(data,freq,max_distance=cap)
                difference=rr['output_delta'].permute(1,0,2).reshape(q,-1)@w.T
                values[name]=float(difference.square().sum()/denom)
            results.append(dict(row=record['row_id'],layer=record['layer'],**values))
    result=dict(status='COMPLETE',records=results,
        local_finite_mean=float(np.mean([r['local_only'] for r in results])),
        full_selected_finite_mean=float(np.mean([r['all_selected_keys'] for r in results])),
        linear_prediction=projection['predicted_local_output_nmse'],budget=projection['budget'],
        scope='Fixed real baseline states; checks nonlinear/precision approximation, not new whole-model states')
    save(worker.root/'selection/E7_nonlinear_check.json',result)
    return result
