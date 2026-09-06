#!/usr/bin/env python3
"""Direction 2: fixed-weight support x allocation factorial on frozen 151.9M weights.

STATUS (2026-09-06): 已收官——两方向审计时代（项目重置前）脚本，其方向被
Round 12 的谱预算框架取代；保留为历史记录，不再运行。服务器产物在
/root/autodl-tmp/claude_audit_prep_20260905/（本地镜像 results_20260905/ 已清理）。

No new training, no table optimization. For each available training seed and
each frozen weight W (geometric-trained W_G, anchored-Cosh-trained W_C), the
full cross of two supports (training support; existing target-matched support
at L_eval=1024) and two allocations (z_G geometric; z_C anchored Cosh tau=4)
is evaluated as tail-128 NLL on the frozen 32 FineWeb-Edu anchors. Cells whose
identity-exact receipt already exists are reused; only missing cells are run.

The tables are the exact registered runtime_frequency constructions of the
frozen protocol (all with declared mscale 1.0); nothing is re-derived ad hoc.
Seed 42 checkpoints are absent on this machine; those cells are reported
MISSING_ASSET and the factorial is stated over the two available training
seeds, tightening wording rather than retraining (per audit Section 4.3).

Reports all cell values, allocation contrasts within fixed W/S, runtime
interactions I_W, and the original diagonal policy contrasts. Never promotes
the best cell to a deployed model. Workspace: paper-2027/claude_code_workspace.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def write_json(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def _repo_root():
    import os
    marker='rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/protocol.py'
    here=Path(__file__).resolve()
    for parent in (here.parent,*here.parents):
        if (parent/marker).is_file():
            return parent
    env=os.environ.get('HYBRID_ROPE_ROOT')
    if env and (Path(env)/marker).is_file():
        return Path(env).resolve()
    raise ImportError('set HYBRID_ROPE_ROOT to a repo root containing rebuttal/rebuttal_0723')


ROOT=_repo_root()
sys.path.insert(0,str(ROOT))
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.protocol import (
    SPEC,runtime_frequency)
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment import (
    _load_checkpoint,_target_sha256,build_eval_windows,set_runtime_rope,sha256_file,tensor_sha256,
    code_fingerprint)

DATA_MANIFEST_SHA='2c4b1c0ec6993a4065a666dd04c26f1c3439e812de25e49cbf5d21b106ab9433'
L_EVAL=1024
TAIL=128
WEIGHTS={'G':'fmrope_base256','C':'anchored_cosh_tau4_fmrope_range'}
TABLE_SOURCES={
    ('S_train','G'):('fmrope_base256','fixed_train_base'),
    ('S_target','G'):('fmrope_base256','target_matched_base'),
    ('S_train','C'):('anchored_cosh_tau4_fmrope_range','fixed_train_range'),
    ('S_target','C'):('anchored_cosh_tau4_fmrope_range','target_matched_range')}
DIAGONAL_CONDITION={'G':'fixed_train_base','C':'fixed_train_range'}
TARGET_DIAGONAL_CONDITION={'G':'target_matched_base','C':'target_matched_range'}
AVAILABLE_SEEDS=(137,256)


def load_data(root):
    import numpy as np
    manifest_path=root/'data'/'data_manifest.json'
    if sha256_file(manifest_path)!=DATA_MANIFEST_SHA: raise ValueError('3-seed data manifest drift')
    manifest=json.loads(manifest_path.read_text())
    validation_path=Path(manifest['validation']['path'])
    anchors_path=Path(manifest['anchors']['path'])
    if sha256_file(validation_path)!=manifest['validation']['sha256']: raise ValueError('validation array drift')
    if sha256_file(anchors_path)!=manifest['anchors']['sha256']: raise ValueError('anchor array drift')
    validation=np.load(validation_path,mmap_mode='r',allow_pickle=False)
    anchors=np.load(anchors_path,allow_pickle=False)
    if len(anchors)!=SPEC.eval_anchor_count: raise ValueError('anchor count drift')
    return validation,anchors,manifest


def tables():
    import torch
    result={}
    for key,(arm,condition) in TABLE_SOURCES.items():
        inv,mscale,meta=runtime_frequency(arm,condition,L_EVAL)
        if mscale!=1.0: raise ValueError(f'declared mscale for {key} is {mscale}, expected 1.0')
        result[key]={'inv':inv.contiguous(),'mscale':float(mscale),
                     'sha256':tensor_sha256(inv.contiguous()),
                     'source_arm':arm,'source_condition':condition,'meta':meta}
    return result


def cells(seeds):
    for seed in seeds:
        for weight in ('G','C'):
            for support in ('S_train','S_target'):
                for shape in ('G','C'):
                    yield {'seed':seed,'weight':weight,'support':support,'shape':shape}


def audit_cells(a):
    root=a.root.resolve()
    ledger=[]
    table_receipts={f'{s}/{z}':v['sha256'] for (s,z),v in tables().items()}
    import numpy as np
    validation,anchors,manifest=load_data(root)
    cur_spec,cur_code=SPEC.fingerprint(),code_fingerprint()
    for cell in cells(a.seeds):
        entry={**cell,'status':'PENDING_NEW','provenance':None}
        run_dir=root/f'seed_{cell["seed"]}'/'runs'/WEIGHTS[cell['weight']]
        if not (run_dir/'model.pt').is_file():
            entry.update(status='MISSING_ASSET',
                         reason=f'no trained checkpoint at {run_dir}; audit forbids retraining to fill the figure')
            ledger.append(entry);continue
        entry['checkpoint_sha256']=sha256_file(run_dir/'model.pt')
        entry['table_sha256']=table_receipts[f'{cell["support"]}/{cell["shape"]}']
        diag=cell['support']=='S_train' and cell['shape']==cell['weight']
        target_diag=cell['support']=='S_target' and cell['shape']==cell['weight']
        if diag or target_diag:
            condition=DIAGONAL_CONDITION[cell['weight']] if diag else TARGET_DIAGONAL_CONDITION[cell['weight']]
            results_path=root/f'seed_{cell["seed"]}'/'evaluation'/'results.json'
            if results_path.is_file():
                results=json.loads(results_path.read_text())
                if (results.get('data_manifest_sha256')==DATA_MANIFEST_SHA
                        and results.get('eval_tail_tokens')==TAIL
                        and results.get('eval_anchor_count')==SPEC.eval_anchor_count):
                    records=[r for r in results.get('records',[])
                             if r['arm']==WEIGHTS[cell['weight']] and r['condition']==condition
                             and r['length']==L_EVAL]
                    if len(records)==len(anchors):
                        _,targets=build_eval_windows(np.asarray(validation),anchors,length=L_EVAL)
                        by_anchor={int(r['anchor']):r for r in records}
                        verified=(len(by_anchor)==len(records) and all(
                            int(anchors[i]) in by_anchor
                            and _target_sha256(targets[i,-TAIL:])==by_anchor[int(anchors[i])]['tail_target_sha256']
                            for i in range(len(anchors))))
                        if verified:
                            entry.update(status='REUSED',provenance={
                                'results_sha256':sha256_file(results_path),
                                'condition':condition,'records':[
                                    {'anchor':rec['anchor'],'tail_nll':rec['tail_nll'],
                                     'full_nll':rec['full_nll'],'tail_target_sha256':rec['tail_target_sha256']}
                                    for rec in sorted(records,key=lambda r:r['anchor'])]})
                        else: entry['reason']='existing receipt failed tail-target verification; re-evaluate'
                    else: entry['reason']='existing receipt lacks the full 32-anchor record; re-evaluate'
                else: entry['reason']='existing receipt identity differs (manifest/tail/anchors); re-evaluate'
            else: entry['reason']='no archived evaluation receipt; evaluate'
        else:
            entry['reason']='off-diagonal cell not covered by any archived receipt'
        if entry['status']=='PENDING_NEW':
            # A new evaluation must reload this checkpoint; _load_checkpoint refuses
            # when the recorded protocol/code fingerprints differ from the current
            # repo snapshot. Detect that here instead of crashing mid-run, and mark
            # the cell blocked rather than guessing the missing value.
            meta=json.loads((run_dir/'train_meta.json').read_text())
            if meta.get('protocol_sha256')!=cur_spec or meta.get('code_sha256')!=cur_code:
                entry.update(status='BLOCKED_IDENTITY_DRIFT',
                             recorded_protocol_sha256=meta.get('protocol_sha256'),
                             recorded_code_sha256=meta.get('code_sha256'),
                             current_spec_fingerprint=cur_spec,
                             current_code_fingerprint=cur_code,
                             reason='checkpoint protocol/code fingerprint differs from the current repo snapshot; the audit forbids loading it or fabricating the missing cell')
        ledger.append(entry)
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.output,{
        'status':'FIXED_WEIGHT_FACTORIAL_LEDGER_V1','L_eval':L_EVAL,'tail_tokens':TAIL,
        'seeds_declared':list(a.seeds),'seeds_with_checkpoints':list(AVAILABLE_SEEDS),
        'data_manifest_sha256':DATA_MANIFEST_SHA,'table_sha256':table_receipts,
        'script_sha256':sha(__file__),'cells':ledger,
        'seed42_note':'seed-42 checkpoints live on another machine and are not present; per the audit, missing assets tighten wording instead of triggering retraining',
        'scope':'Explanatory fixed-factor NLL factorial. No deployment selection, no zero-training ranking, no generation claim.'})
    counts={}
    for c in ledger: counts[c['status']]=counts.get(c['status'],0)+1
    print(json.dumps({'status':'AUDITED','cells':len(ledger),'by_status':counts}))


def run_missing(a):
    import torch
    import torch.nn.functional as F
    import numpy as np
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    if not a.authorized: raise ValueError('GPU runs require explicit --authorized')
    ledger_receipt=json.loads(a.ledger.read_text())
    if ledger_receipt['status']!='FIXED_WEIGHT_FACTORIAL_LEDGER_V1': raise ValueError('unknown ledger')
    root=a.root.resolve()
    validation,anchors,manifest=load_data(root)
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    table_cache=tables()
    done,start=0,time.monotonic()
    results={}
    for cell in ledger_receipt['cells']:
        if cell['status']!='PENDING_NEW': continue
        key=f'seed{cell["seed"]}_W{cell["weight"]}_{cell["support"]}_z{cell["shape"]}'
        model_key=(cell['seed'],cell['weight'])
        if model_key not in results:
            model,metadata=_load_checkpoint(root/f'seed_{cell["seed"]}',WEIGHTS[cell['weight']],spec=SPEC)
            model=model.to('cuda');results[model_key]=model
        model=results[model_key]
        table=table_cache[(cell['support'],cell['shape'])]
        set_runtime_rope(model,table['inv'],length=L_EVAL,mscale=table['mscale'])
        model.eval()
        inputs_np,targets_np=build_eval_windows(np.asarray(validation),anchors,length=L_EVAL)
        inputs=torch.from_numpy(inputs_np).to('cuda')
        targets=torch.from_numpy(targets_np).to('cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(inputs)
        token_nll=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                                  targets.reshape(-1),reduction='none').reshape(targets.shape)
        if not torch.isfinite(token_nll).all(): raise RuntimeError('nonfinite factorial NLL')
        full=token_nll.mean(dim=1);tail=token_nll[:,-TAIL:].mean(dim=1)
        records=[{'anchor':int(anchor),'full_nll':float(full[i].cpu()),'tail_nll':float(tail[i].cpu()),
                  'tail_target_sha256':_target_sha256(targets_np[i,-TAIL:])}
                 for i,anchor in enumerate(anchors.tolist())]
        write_json(a.output/f'{key}.json',{
            'status':'FIXED_WEIGHT_FACTORIAL_CELL_V1',**cell,
            'table_source':TABLE_SOURCES[(cell['support'],cell['shape'])],
            'table_sha256':table['sha256'],'mscale':table['mscale'],
            'checkpoint_sha256':sha256_file(root/f'seed_{cell["seed"]}'/'runs'/WEIGHTS[cell['weight']]/'model.pt'),
            'data_manifest_sha256':DATA_MANIFEST_SHA,'L_eval':L_EVAL,'tail_tokens':TAIL,
            'records':records,'mean_tail_nll':float(tail.mean().cpu()),
            'mean_full_nll':float(full.mean().cpu()),'script_sha256':sha(__file__),
            'seconds':round(time.monotonic()-start,1)})
        done+=1
        print(json.dumps({'cell':key,'mean_tail_nll':round(float(tail.mean().cpu()),6),
                          'seconds':round(time.monotonic()-start,1)}),flush=True)
        del inputs,targets,logits,token_nll
    write_json(a.output/'run_summary.json',{'status':'FIXED_WEIGHT_FACTORIAL_RUN_COMPLETE_V1',
        'new_cells':done,'ledger_sha256':sha256_file(a.ledger),'script_sha256':sha(__file__),
        'scope':'Newly evaluated missing cells only; reused cells remain in the ledger with archived provenance.'})
    print(json.dumps({'status':'RUN_COMPLETE','new_cells':done}))


def report(a):
    ledger_receipt=json.loads(a.ledger.read_text())
    if ledger_receipt['status']!='FIXED_WEIGHT_FACTORIAL_LEDGER_V1': raise ValueError('unknown ledger')
    value={}
    for cell in ledger_receipt['cells']:
        key=(cell['seed'],cell['weight'],cell['support'],cell['shape'])
        if cell['status']=='REUSED':
            records=cell['provenance']['records']
            value[key]={'mean_tail_nll':sum(r['tail_nll'] for r in records)/len(records),
                        'provenance':'reused_archived_receipt','status':'REUSED'}
        elif cell['status']=='PENDING_NEW':
            name=f'seed{cell["seed"]}_W{cell["weight"]}_{cell["support"]}_z{cell["shape"]}'
            path=a.run_dir/f'{name}.json'
            if not path.is_file():
                value[key]={'mean_tail_nll':None,'provenance':'MISSING_NEW_RECEIPT','status':'INCOMPLETE'};continue
            receipt=json.loads(path.read_text())
            value[key]={'mean_tail_nll':receipt['mean_tail_nll'],'provenance':'new_evaluation',
                        'status':'EVALUATED','receipt_sha256':sha(path)}
        else:
            value[key]={'mean_tail_nll':None,'provenance':cell.get('reason','blocked'),'status':cell['status']}
    def cell(seed,w,s,z):
        v=value.get((seed,w,s,z),{})
        return v.get('mean_tail_nll')
    contrasts,interactions,diagonal=[],[],[]
    for seed in ledger_receipt['seeds_declared']:
        for w in ('G','C'):
            d_train=cell(seed,w,'S_train','C')-cell(seed,w,'S_train','G') if None not in (cell(seed,w,'S_train','C'),cell(seed,w,'S_train','G')) else None
            d_target=cell(seed,w,'S_target','C')-cell(seed,w,'S_target','G') if None not in (cell(seed,w,'S_target','C'),cell(seed,w,'S_target','G')) else None
            for support,delta in (('S_train',d_train),('S_target',d_target)):
                if delta is not None:
                    contrasts.append({'seed':seed,'weight':w,'support':support,'delta_zC_minus_zG':delta})
            interactions.append({'seed':seed,'weight':w,'delta_S_train':d_train,'delta_S_target':d_target,
                                 'I_W':(d_target-d_train) if d_train is not None and d_target is not None else None})
        d_s_train=cell(seed,'C','S_train','C')-cell(seed,'G','S_train','G') if None not in (cell(seed,'C','S_train','C'),cell(seed,'G','S_train','G')) else None
        d_s_target=cell(seed,'C','S_target','C')-cell(seed,'G','S_target','G') if None not in (cell(seed,'C','S_target','C'),cell(seed,'G','S_target','G')) else None
        diagonal.append({'seed':seed,'D_S_train':d_s_train,'D_S_target':d_s_target,
                         'note':'diagonal contrasts pair each weight with its own training shape; supports are arm-specific constructions'})
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.output,{
        'status':'FIXED_WEIGHT_FACTORIAL_REPORT_V1','ledger_sha256':sha(a.ledger),
        'cells':{f'seed{s}_W{w}_{sup}_z{z}':value[(s,w,sup,z)]
                 for s in ledger_receipt['seeds_declared'] for w in ('G','C')
                 for sup in ('S_train','S_target') for z in ('G','C') if (s,w,sup,z) in value},
        'allocation_contrasts_within_fixed_W_S':contrasts or None,
        'interpretation_note':'I_W != 0 is an interaction, not a crossover; sign reversal of the allocation effect across supports is the reversal.',
        'runtime_interactions_I_W':interactions,'diagonal_policy_contrasts':diagonal,
        'seed_coverage':{'declared':ledger_receipt['seeds_declared'],'with_checkpoints':ledger_receipt['seeds_with_checkpoints'],
                         'note':'three seeds are not a large independent sample; training seed is the repetition unit'},
        'script_sha256':sha(__file__),
        'scope':'Causal decomposition of an existing claim. A good cell is not promoted to deployment; results do not rank zero-training methods or assert generation ability.'})
    print(json.dumps({'status':'REPORTED','cells':len(value)},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('audit-cells')
    s.add_argument('--root',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    s.add_argument('--seeds',nargs='+',type=int,default=list(AVAILABLE_SEEDS))
    s=sub.add_parser('run-missing')
    s.add_argument('--root',type=Path,required=True);s.add_argument('--ledger',type=Path,required=True)
    s.add_argument('--output',type=Path,required=True);s.add_argument('--authorized',action='store_true')
    s=sub.add_parser('report')
    s.add_argument('--ledger',type=Path,required=True);s.add_argument('--run-dir',type=Path,required=True)
    s.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='audit-cells': audit_cells(a)
    elif a.action=='run-missing': run_missing(a)
    else: report(a)


if __name__=='__main__':
    main()
