"""Repair two recorded underflow cases without replacing prior model outcomes."""
import importlib
import json
import math
import torch
from .worker import read_rows, save, sha


def run(worker, job):
    mod=importlib.reload(importlib.import_module('experiments.nongeometric_screen.select'))
    folder=worker.root/'selection'
    proposals=json.loads((folder/'proposals.json').read_text())
    records=read_rows(folder/'conditional_rows.jsonl');repairs=[]
    with torch.inference_mode():
        for i,r in enumerate(records):
            if all(math.isfinite(r[k]) for k in ('gain','nmse','target_mass')):continue
            raw=torch.load(worker.root/'calibration'/r['row_id']/f"layer_{r['layer']:02d}.pt",map_location='cpu',weights_only=False)
            d=mod.prepare_record(raw,worker.tables['MrPro'])
            fixed=mod.replay(d,proposals[r['proposal']]);fixed.pop('output_delta')
            if not all(math.isfinite(fixed[k]) for k in ('gain','nmse','target_mass')):raise ValueError('repair nonfinite')
            records[i]={**r,**fixed};repairs.append(dict(original=r,repaired=records[i]))
    summary={name:mod.aggregate([r for r in records if r['proposal']==name]) for name in proposals}
    bm=[r for r in records if r['proposal']=='BM']
    layers=sorted(range(36),key=lambda layer:mod.aggregate([r for r in bm if r['layer']==layer])['robust_gain'],reverse=True)
    result=dict(status='COMPLETE',repairs=repairs,summary=summary,bm_layer_ranking=layers,
        source_sha256=sha(mod.__file__),scope='Numerical repair of fixed-state proposals; original selection and all model outcomes retained')
    save(folder/'replay_repair.json',result)
    return dict(status='COMPLETE',repaired=len(repairs),bm_layer_ranking=layers)
