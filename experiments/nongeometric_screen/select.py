"""Finite, precision-matched proposal replay on captured real-prefix states."""
from __future__ import annotations

from collections import defaultdict
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from .worker import save, sha


def phase_rotate(raw, pos, frequencies, gain):
    phase = pos.float()[:, None] * frequencies[None, :]
    cos = (phase.cos() * gain).to(raw.dtype)
    sin = (phase.sin() * gain).to(raw.dtype)
    a, b = raw[..., :64], raw[..., 64:]
    return torch.cat((a*cos - b*sin, b*cos + a*sin), dim=-1)


def prepare_record(record, table):
    data = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in record.items()}
    data['freq'] = torch.tensor(table['values_float32'], device='cuda', dtype=torch.float32)
    data['gain'] = table['gain']
    data['oldp'] = (data['selected_baseline_logits'] - data['baseline_lse'][..., None]).exp()
    data['oldp'].masked_fill_(~data['valid'][None], 0)
    data['v16'] = data['v_raw'].repeat_interleave(8, 0).float()
    data['residual_output'] = data['baseline_output'] - torch.einsum('hqs,hsd->hqd', data['oldp'], data['v16'])
    data['residual_mass'] = (1-data['oldp'].sum(-1)).clamp_min(0)
    data['residual_target'] = (data['full_target_mass']-data['oldp'][..., data['selected_target']].sum(-1)).clamp_min(0)
    return data


def replay(data, frequencies, head_mask=None, max_distance=None):
    freq = torch.as_tensor(frequencies, dtype=torch.float32, device='cuda')
    q = phase_rotate(data['q_raw'], data['query_positions'], freq, data['gain'])
    k = phase_rotate(data['k_raw'], data['key_positions'], freq, data['gain'])
    logits = torch.einsum('ghqd,gsd->ghqs', q.reshape(2,8,-1,128).float(), k.float()).reshape(16,q.shape[1],-1) / math.sqrt(128)
    delta = logits - data['selected_baseline_logits']
    delta.masked_fill_(~data['valid'][None], 0)
    if max_distance is not None:
        distances=data['query_positions'][:,None]-data['key_positions'][None,:]
        delta.masked_fill_((distances>max_distance)[None],0)
    if head_mask is not None:
        delta[~head_mask] = 0
    # Stabilize actual new log probabilities. Scaling by delta alone can
    # underflow every term when a formerly negligible key moves by >100 nats.
    log_weights = data['selected_baseline_logits'] - data['baseline_lse'][..., None] + delta
    log_weights.masked_fill_(~data['valid'][None], -torch.inf)
    shift = log_weights.max(-1).values.clamp_min(0)
    weights = (log_weights - shift[..., None]).exp()
    residual_scale = (-shift).exp()
    den = data['residual_mass'] * residual_scale + weights.sum(-1)
    if not torch.isfinite(den).all() or (den <= 0).any():
        raise ValueError('invalid conditional replay normalizer')
    out = (data['residual_output']*residual_scale[..., None]+torch.einsum('hqs,hsd->hqd', weights, data['v16'])) / den[..., None]
    target = (data['residual_target']*residual_scale+weights[...,data['selected_target']].sum(-1)) / den
    useful = data['full_target_mass']
    change = target.clamp_min(1e-20).log()-useful.clamp_min(1e-20).log()
    head_gain = (change * useful).sum(-1) / useful.sum(-1).clamp_min(1e-20)
    gain = (change * useful).sum() / useful.sum().clamp_min(1e-20)
    nmse = (out-data['baseline_output']).square().sum() / data['baseline_output'].square().sum().clamp_min(1e-12)
    return dict(gain=float(gain), nmse=float(nmse), head_gain=head_gain.cpu().tolist(),
        target_mass=float(target.mean()), target_support=float(useful.sum()),
        output_delta=out-data['baseline_output'], max_selected_logit_delta=float(delta.abs().max()))


def proposals(tables):
    native = np.array(tables['Native']['values_float32'], dtype=np.float64)
    base = np.array(tables['MrPro']['values_float32'], dtype=np.float64)
    m = np.log(native/base)/math.log(4)
    result = {}
    for slot in range(24,40):
        for direction in (-1,1):
            new = base.copy()
            new[slot] = native[slot] * 4 ** (-m[slot+direction])
            result[f'E1_s{slot}_{"less" if direction<0 else "more"}'] = new.astype(np.float32).tolist()
    for high in (39,41):
        n=high-23
        t=np.clip(np.arange(64)-23,0,n)
        result[f'E2_boundary{high}'] = (native/4**(t*(t+1)/(n*(n+1)))).astype(np.float32).tolist()
    factor = 1e6**(1/64)
    for name, mult in [('more',1/factor),('less',factor)]:
        new=base.copy();new[40:]*=mult
        result[f'E2_tail_{name}'] = new.astype(np.float32).tolist()
    result['BM'] = tables['MrProBM']['values_float32']
    for slot in range(40,64):
        new=base.copy();new[slot]=0
        result[f'E8_zero{slot}'] = new.astype(np.float32).tolist()
    return result


def aggregate(records):
    # Task macro first, then the two physical lengths, with split agreement kept.
    def score_for(rs):
        tasks=defaultdict(list)
        for r in rs:
            if r['task']!='natural' and r['target_support']>0:
                tasks[(r['task'],r['length'])].append(r['gain'])
        vals=[float(np.mean(v)) for v in tasks.values()]
        return float(np.mean(vals)) if vals else 0.
    natural=[r['nmse'] for r in records if r['task']=='natural']
    halves=[score_for([r for r in records if r['split']==s]) for s in (0,1)]
    return dict(target_gain=score_for(records),split_gains=halves,robust_gain=min(halves),
        natural_output_nmse=float(np.mean(natural)) if natural else None)


def run(worker, job):
    folder=worker.root/'selection';folder.mkdir(exist_ok=True)
    prop=proposals(worker.tables)
    save(folder/'proposals.json',prop)
    records=defaultdict(list)
    raw=[]
    max_identity=0.
    with torch.inference_mode():
        files=sorted((worker.root/'calibration').glob('*/layer_*.pt'))
        for index,path in enumerate(files):
            original=torch.load(path,map_location='cpu',weights_only=False)
            data=prepare_record(original,worker.tables['MrPro'])
            identity=replay(data,worker.tables['MrPro']['values_float32'])
            max_identity=max(max_identity,identity['max_selected_logit_delta'])
            if identity['nmse']>1e-8 or identity['max_selected_logit_delta']>0.01:
                raise ValueError('identity selected-key replay mismatch: '+str(path))
            for name,freq in prop.items():
                r=replay(data,freq)
                r.pop('output_delta')
                r.update(proposal=name,row_id=original['row_id'],layer=original['layer'],
                    task=original['metadata']['task'],length=original['metadata']['length'],split=original['metadata']['split'])
                records[name].append(r);raw.append(r)
            if index%36==35:
                save(worker.root/'live.json',dict(job=job['id'],phase='conditional_proposal_replay',files=index+1,total=len(files)))
            del data,original
    summary={name:aggregate(rs) for name,rs in records.items()}
    bm_damage=summary['BM']['natural_output_nmse']
    e1=sorted((name for name in prop if name.startswith('E1_') and summary[name]['natural_output_nmse']<=bm_damage),
        key=lambda name:summary[name]['robust_gain'],reverse=True)[:2]
    if len(e1)<2:
        raise ValueError('insufficient E1 proposals under predeclared BM local-output ceiling')
    e2=max((name for name in prop if name.startswith('E2_')),key=lambda name:summary[name]['robust_gain'])
    e8=max((name for name in prop if name.startswith('E8_') and summary[name]['natural_output_nmse']<=bm_damage),
        key=lambda name:summary[name]['robust_gain'])
    specs={name:dict(operator='static',table=dict(values_float32=prop[name],gain=worker.tables['MrPro']['gain'])) for name in e1+[e2,e8]}
    # E4 uses signed response fingerprints; select only a pair whose correlation
    # is positive in both disjoint calibration row-index halves.
    pair_scores=[]
    for j,k in itertools.combinations(range(24,40),2):
        correlations=[]
        for split in (0,1):
            a=[r['gain'] for r in records[f'E1_s{j}_more'] if r['split']==split and r['task']!='natural' and r['target_support']>0]
            b=[r['gain'] for r in records[f'E1_s{k}_more'] if r['split']==split and r['task']!='natural' and r['target_support']>0]
            correlations.append(float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else -1.)
        if min(correlations)>0:
            pair_scores.append((min(correlations),j,k,correlations))
    pair=max(pair_scores) if pair_scores else None
    if pair:
        _,j,k,_=pair
        native=np.array(worker.tables['Native']['values_float32']);freq=np.array(worker.tables['MrPro']['values_float32'])
        m=np.log(native/freq)/math.log(4);shared=float((m[j]+m[k])/2)
        freq[[j,k]]=native[[j,k]]*4**(-shared)
        specs[f'E4_pair{j}_{k}']=dict(operator='static',table=dict(values_float32=freq.astype(np.float32).tolist(),gain=worker.tables['MrPro']['gain']))
    # BM layer proposal: rank fixed-state effect, validate each from prefill.
    layer_rank=sorted(range(36),key=lambda layer:aggregate([r for r in records['BM'] if r['layer']==layer])['robust_gain'],reverse=True)[:2]
    for layer in layer_rank:
        specs[f'E5_layer{layer}']=dict(operator='layer',layer=layer,table=worker.tables['MrPro'],replacement=worker.tables['MrProBM'])
    # E6 uses the E2 direction, preserving each full shared KV group.
    groups=[]
    for layer in range(36):
        rs=[r for r in records[e2] if r['layer']==layer and r['task']!='natural' and r['target_support']>0]
        for group in (0,1):
            differences=[]
            for split in (0,1):
                val=[np.mean(r['head_gain'][group*8:(group+1)*8])-np.mean(r['head_gain'][(1-group)*8:(2-group)*8]) for r in rs if r['split']==split]
                differences.append(float(np.mean(val)) if val else 0.)
            groups.append((min(differences),layer,group,differences))
    group=max(groups)
    for g in (group[2],1-group[2]):
        specs[f'E6_layer{group[1]}_group{g}']=dict(operator='group',layer=group[1],group=g,table=worker.tables['MrPro'],replacement=specs[e2]['table'])
    with (folder/'conditional_rows.jsonl').open('w') as f:
        for r in raw:f.write(json.dumps(r)+'\n')
    save(folder/'summary.json',summary)
    receipt=dict(status='COMPLETE',e1=e1,e2=e2,e8=e8,e4_pair=pair,e5_layers=layer_rank,e6_group=group,
        specs=specs,max_identity_logit_delta=max_identity,source_sha256=sha(__file__),
        scope='Conditional selected-key replay; omitted key changes and upstream state changes are not modeled. Whole-model tests are mandatory.')
    save(folder/'selection.json',receipt)
    for n,(name,spec) in enumerate(specs.items(),10):
        save(worker.root/'queue'/f'{n:03d}_{name}.json',dict(id=name,spec=spec,panel='small',nll_docs=4,nll_lengths=[8192,32768]))
    return receipt
