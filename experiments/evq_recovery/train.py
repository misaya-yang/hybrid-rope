"""Resumable same-data adaptation. Dry-run by default; no wall-clock termination rule."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .acquire import file_hash
from .data import JsonlIndex,write_json
from .runtime import load,training_step


def order_index(step,size,seed):
    epoch,offset=divmod(step,size)
    indices=list(range(size));random.Random(seed+epoch).shuffle(indices)
    return indices[offset]


def learning_rate(step,total,peak):
    warmup=max(1,min(128,math.ceil(total*.05)))
    if step<warmup:return peak*(step+1)/warmup
    return peak*(.1+.45*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup))))


def validate_assets(root,plan):
    manifest=json.loads((root/'data/data_manifest.json').read_text())
    if manifest['status']!='CPU_DATA_READY':raise ValueError('data preparation incomplete')
    for name in ['cpt_train.npy','sft_train.jsonl','native_train.jsonl']:
        path=root/'data'/name
        if file_hash(path)!=manifest['files'][name]['sha256']:raise ValueError(f'data changed: {name}')
    table=json.loads((root/'tables.json').read_text())
    if not all(a in table for a in plan['arms']):raise ValueError('missing grid')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=['Native','Cosh','Exponential','Hybrid','YaRN'],required=True)
    p.add_argument('--regime',choices=['lora','full'],default='lora')
    p.add_argument('--until-cpt-tokens',type=int,default=33554432)
    p.add_argument('--resume',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Two real 16K updates; discard adapter, report memory/time only.')
    a=p.parse_args();root=a.root.resolve();plan=json.loads((root/'plan.json').read_text())
    validate_assets(root,plan)
    cpt=np.load(root/'data/cpt_train.npy',mmap_mode='r')
    sft=JsonlIndex(root/'data/sft_train.jsonl');native=JsonlIndex(root/'data/native_train.jsonl')
    length=cpt.shape[1]-1
    if length!=16384 or min(len(sft),len(native))==0:raise ValueError('actual long inputs and supervision required')
    if a.until_cpt_tokens%length or a.until_cpt_tokens>plan['schedule_cpt_tokens']:
        raise ValueError('token endpoint outside fixed optimizer schedule')
    total_steps=plan['schedule_cpt_tokens']//length
    stop=2 if a.smoke else a.until_cpt_tokens//length
    summary=dict(arm=a.arm,regime=a.regime,cpt_length=length,cpt_windows=len(cpt),
                 sft_rows=len(sft),native_rows=len(native),updates=stop,
                 cpt_tokens=stop*length,mode='SMOKE' if a.smoke else 'TRAIN',execute=a.execute)
    print(json.dumps(summary),flush=True)
    if not a.execute:return
    lock=(root/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Dense quadratic fallback at 16K is not an acceptable execution path.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    started=time.monotonic()
    model,wrapper,_,table=load(plan,root,a.arm,a.resume,True,a.regime)
    parameters=[x for x in model.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'],betas=(.9,.95),weight_decay=0.)
    base_step=0
    contract_hash=file_hash(root/'plan.json')+file_hash(root/'tables.json')+file_hash(root/'data/data_manifest.json')
    if a.resume:
        state=json.loads((a.resume/'state.json').read_text())
        if state['contract_hash']!=contract_hash:raise ValueError('resumed data/plan/table drift')
        base_step=state['step']
        optimizer.load_state_dict(torch.load(a.resume/'optimizer.pt',map_location='cuda',weights_only=True))
    if base_step>=stop:raise ValueError('requested endpoint already complete')
    out=root/'runs'/f'{a.arm}_{a.regime}'
    if a.smoke:out=root/'smokes'/f'{a.arm}_{a.regime}'
    if out.exists() and not a.resume:raise FileExistsError(f'preserve existing run: {out}')
    out.mkdir(parents=True,exist_ok=bool(a.resume))
    write_json(out/'contract.json',dict(plan=plan,table=table,contract_hash=contract_hash,
        regime=a.regime,trainable_parameters=sum(x.numel() for x in parameters),
        trainable_names=[n for n,x in model.named_parameters() if x.requires_grad],
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
        comparison='same initial weights/adapter seed, data ordering, losses and token schedule across grids'))
    checkpoints={x//length for x in plan['checkpoint_cpt_tokens']}|{stop}
    with (out/'steps.jsonl').open('a') as log:
        for step in range(base_step,stop):
            tick=time.monotonic();lr=learning_rate(step,total_steps,plan['learning_rate'])
            for group in optimizer.param_groups:group['lr']=lr
            c=order_index(step,len(cpt),plan['seed'])
            s=order_index(step,len(sft),plan['seed']+100)
            n=order_index(step,len(native),plan['seed']+200)
            record=training_step(model,optimizer,cpt[c],sft[s],native[n],amp=True,
                sft_weight=plan['sft_weight'],native_weight=plan['native_weight'],chunk_size=128)
            record.update(step=step+1,lr=lr,cpt_index=c,sft_index=s,native_index=n,
                          cpt_tokens=(step+1)*length,seconds=time.monotonic()-tick,
                          peak_allocated=torch.cuda.max_memory_allocated())
            log.write(json.dumps(record)+'\n');log.flush()
            if (step+1)%16==0 or a.smoke:print(json.dumps(record),flush=True)
            if step+1 in checkpoints and not a.smoke:
                checkpoint=out/f'checkpoint-{(step+1)*length}'
                checkpoint.mkdir(exist_ok=False)
                (wrapper if wrapper else model).save_pretrained(checkpoint,safe_serialization=True)
                torch.save(optimizer.state_dict(),checkpoint/'optimizer.pt')
                write_json(checkpoint/'state.json',dict(arm=a.arm,regime=a.regime,step=step+1,
                    contract_hash=contract_hash,seconds=time.monotonic()-started,
                    model_path=plan['model_path'],table_sha256=table['sha256']))
                write_json(out/'latest.json',dict(checkpoint=str(checkpoint),cpt_tokens=(step+1)*length))
    write_json(out/'completion.json',dict(**summary,status='GPU_SMOKE_ONLY' if a.smoke else 'TRAINING_ENDPOINT_COMPLETE_NEEDS_EVALUATION',
        seconds=time.monotonic()-started,peak_allocated=torch.cuda.max_memory_allocated(),
        final_step=stop,adapter_saved=not a.smoke))


if __name__=='__main__':main()
