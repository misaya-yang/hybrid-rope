"""E0 complete-step probe and E2/E3 matched full/LoRA training.

Each update: one alternating 8K/16K CPT prefix, one matched near/far answer
pair, one original-Native replay row. CE means and KL weight are explicit.
This new recipe is a proposal, not an assertion of historical-recipe identity.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import time

import numpy as np
import torch
from .contracts import read_rows, sha_file, sha_json, write_json
from .runtime import cuda_runtime, load_model, verify_prepared, versions
from .tables import verify_static
from .training import configure_training,update


def paired_sft(rows,seed):
    pairs={}
    for r in rows:
        key=(r['group_id'],r['world'])
        if r['layout'] in pairs.setdefault(key,{}):raise ValueError('duplicate SFT layout')
        pairs[key][r['layout']]=r
    if any(set(v)!= {'near','far'} for v in pairs.values()):raise ValueError('SFT near/far pair incomplete')
    keys=sorted(pairs);random.Random(seed).shuffle(keys)
    return [[pairs[k]['near'],pairs[k]['far']] for k in keys]


def scheduled_lr(step,steps,lr):
    warmup=max(1,math.ceil(.05*steps))
    if step<=warmup:return lr*step/warmup
    fraction=(step-warmup)/max(1,steps-warmup)
    return lr*(.1+.9*.5*(1+math.cos(math.pi*fraction)))


def save_final_checkpoint(model,tokenizer,dest,*,arm,entry,contract_sha256,step,input_tokens):
    """Save the trained model and evaluation identity, without optimizer/RNG state."""
    model.save_pretrained(dest,safe_serialization=True)
    tokenizer.save_pretrained(dest)
    write_json(dest/'deployment.json',dict(arm=arm,table_sha256=entry['tensor_sha256'],amplitude=entry['amplitude']))
    write_json(dest/'checkpoint_manifest.json',dict(contract_sha256=contract_sha256,
        step=step,input_tokens=input_tokens,files={p.name:sha_file(p) for p in dest.iterdir() if p.is_file()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for a in ('prepared','arm','cpt','cpt-sha256','teacher-cache','out','regime'):p.add_argument('--'+a,required=True)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--lr',type=float,required=True);p.add_argument('--probe',action='store_true')
    p.add_argument('--kl-weight',type=float,default=1.);p.add_argument('--cpt-weight',type=float,default=1.)
    p.add_argument('--sft-weight',type=float,default=1.);p.add_argument('--reserve-gib',type=float,default=3.)
    a=p.parse_args()
    if a.steps<=0 or a.lr<=0 or a.regime not in ('full','lora') or a.arm not in ('YaRN','MrPro','Z'):
        raise ValueError('invalid matched training contract')
    if a.probe and a.steps!=4:raise ValueError('cost probe is exactly four complete updates, covering 8K and 16K twice')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);root=Path(a.prepared)
    assets=verify_prepared(root)
    if sha_file(a.cpt)!=a.cpt_sha256:raise ValueError('CPT hash drift')
    cpt=np.load(a.cpt,mmap_mode='r',allow_pickle=False)
    if cpt.ndim!=2 or cpt.shape[1]!=16385:raise ValueError('CPT requires 16385-token rows')
    cache=Path(a.teacher_cache);cm=json.loads((cache/'manifest.json').read_text())
    expected_weights={k:v for k,v in assets['assets'].items() if k.endswith('.safetensors')}
    if cm['status']!='COMPLETE' or cm['mode']!='cache' or cm['arm']!='Native' or cm['model_weights']!=expected_weights:
        raise ValueError('cache is not this original Native teacher')
    if sha_file(cache/'rows.jsonl')!=cm['rows_sha256']:raise ValueError('cache index drift')
    replay=list(read_rows(cache/'rows.jsonl'))
    for r in replay:
        if sha_file(cache/r['path'])!=r['sha256']:raise ValueError('cache tensor drift')
    sft=paired_sft(list(read_rows(root/'sft_rows.jsonl')),a.seed)
    order=list(range(len(cpt)));random.Random(a.seed).shuffle(order)
    replay_order=list(range(len(replay)));random.Random(a.seed).shuffle(replay_order)
    hardware=cuda_runtime();torch.manual_seed(a.seed)
    model,tok,values,entry=load_model(assets['model'],root,a.arm)
    model,wrapper=configure_training(model,a.regime)
    if a.regime=='full':model.float()  # FP32 master weights/Adam; BF16 compute below
    model.train();model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=a.lr,betas=(.9,.95),weight_decay=0.,fused=True)
    weight_bytes=sum(p.numel()*p.element_size() for p in params)
    # One final trained checkpoint plus metadata/evaluation headroom. Base is read-only.
    storage_bound=weight_bytes+int(a.reserve_gib*2**30)
    if shutil.disk_usage(out).free<storage_bound:
        raise RuntimeError(f'insufficient checkpoint disk: need {storage_bound/2**30:.2f} GiB free')
    contract=dict(vars(a),arm_identity=entry,hardware=hardware,software=versions(),
        trainable_parameters=sum(p.numel() for p in params),trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
        order_sha256=sha_json(order),sft_order_sha256=sha_json([[r['row_id'] for r in pair] for pair in sft]),
        replay_order_sha256=sha_json(replay_order),teacher_manifest_sha256=sha_file(cache/'manifest.json'),
        checkpoint_steps=[a.steps],checkpoint_storage_bound_bytes=storage_bound,
        retention_policy='final_only; per-step metrics; no intermediate weights or optimizer/RNG resume',
        loss='CPT CE mean + answer-token-weighted paired SFT CE mean + vocabulary-summed position-mean Native KL',
        selection='fixed final update; steps.jsonl records training losses, not intermediate generation quality')
    write_json(out/'contract.json',contract)
    start=time.monotonic();records=[];input_tokens=0;prediction_tokens=0
    with (out/'steps.jsonl').open('x') as f:
        for step in range(1,a.steps+1):
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic()
            lr=scheduled_lr(step,a.steps,a.lr)
            for g in optimizer.param_groups:g['lr']=lr
            length=8192 if step%2 else 16384
            c=torch.tensor(np.asarray(cpt[order[(step-1)%len(order)],:length+1],dtype=np.int64)[None],device='cuda')
            pair=sft[(step-1)%len(sft)];supervised=[]
            for r in pair:
                ids=r['prompt_ids']+r['target_ids'];labels=[-100]*len(r['prompt_ids'])+r['target_ids']
                supervised.append((torch.tensor([ids],device='cuda'),torch.tensor([labels],device='cuda')))
            rr=replay[replay_order[(step-1)%len(replay_order)]]
            replay_input=(torch.tensor([rr['input_ids']],device='cuda'),torch.tensor(rr['positions'],device='cuda'),
                          torch.from_numpy(np.load(cache/rr['path'],allow_pickle=False)))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=update(model,optimizer,c,supervised,replay_input,
                              kl_weight=a.kl_weight,cpt_weight=a.cpt_weight,sft_weight=a.sft_weight)
            verify_static(model,values,entry['amplitude'])
            input_tokens+=result['cpt_input_tokens']+result['sft_input_tokens']
            prediction_tokens+=result['cpt_prediction_tokens']+result['answer_prediction_tokens']
            torch.cuda.synchronize()
            result.update(step=step,lr=lr,physical_cpt_length=length,cumulative_input_tokens=input_tokens,
                cumulative_prediction_tokens=prediction_tokens,step_seconds=time.monotonic()-t,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                cpt_row=order[(step-1)%len(order)],sft_row_ids=[r['row_id'] for r in pair],replay_id=rr['id'])
            if step==a.steps:
                save_start=time.monotonic()
                temporary=tempfile.TemporaryDirectory(prefix='discardable_probe_',dir=out) if a.probe else None
                dest=Path(temporary.name)/f'step_{step}' if temporary else out/f'step_{step}'
                save_final_checkpoint(wrapper or model,tok,dest,arm=a.arm,entry=entry,
                    contract_sha256=sha_file(out/'contract.json'),step=step,input_tokens=input_tokens)
                result['save_seconds']=time.monotonic()-save_start
                if temporary:temporary.cleanup()  # only this job's declared disposable probe save
            f.write(json.dumps(result)+'\n');f.flush();records.append(result)
            print(json.dumps(result),flush=True)
    write_json(out/'manifest.json',dict(status='COMPLETE',mode='DISCARDABLE_COST_PROBE' if a.probe else 'MATCHED_TRAINING',
        wall_seconds=time.monotonic()-start,steps=a.steps,input_tokens=input_tokens,prediction_tokens=prediction_tokens,
        unique_cpt_rows=min(a.steps,len(order)),cpt_passes=a.steps/len(order),arm=a.arm,regime=a.regime,
        full_cost_includes=['CPT','SFT','Native replay','backward','AdamW','final model save'],
        retention_policy=contract['retention_policy'],
        contract_sha256=sha_file(out/'contract.json'),steps_sha256=sha_file(out/'steps.jsonl'),
        limits='Native KL is sampled; separate Native generation/NLL and long controls still required'))


if __name__=='__main__':main()
