#!/usr/bin/env python3
"""One full 8K,8K,16K-cycle disposable 5090 probe; dry-run unless --execute."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from experiments.evq_recovery.data import JsonlIndex
from .runtime import load_model,step,validate_cuda
from .train import verify_bound
from .length_schedule import SCHEDULE,length_at,occurrence_before

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=['Native','Cosh'],required=True);p.add_argument('--warmup-cycles',type=int,default=0);p.add_argument('--timed-cycles',type=int,default=1);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if a.warmup_cycles<0 or a.timed_cycles<1:raise ValueError('warmup cycles must be nonnegative and timed cycles positive')
 plan=json.loads((a.root/'plan.json').read_text());manifest,tables=verify_bound(plan,Path(plan['data_manifest']),a.root/'tables.json');summary={'status':'DRY_RUN_VALIDATED','updates':len(SCHEDULE)*(a.warmup_cycles+a.timed_cycles),'discarded':True,'evidence_scope':'engineering memory/throughput qualification; not scientific evidence','arm':a.arm,'length_schedule':list(SCHEDULE),'warmup_cycles':a.warmup_cycles,'timed_cycles':a.timed_cycles,'high_watermark':16384};print(json.dumps(summary))
 if not a.execute:return
 gpu=validate_cuda()
 cpt={length:np.load(manifest['cpt_train_by_length'][str(length)]['path'],mmap_mode='r') for length in plan['train_lengths']};sft={length:JsonlIndex(manifest['sft_train_by_length'][str(length)]['path']) for length in plan['train_lengths']};replay=JsonlIndex(manifest['replay_train']['path']);model,wrapper=load_model(plan,tables[a.arm]);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['learning_rate'],betas=tuple(plan['betas']),weight_decay=0.,fused=True)
 native=np.asarray(tables['Native']['values_float32'],dtype=np.float32);records=[]
 si={length:max(range(len(sft[length])),key=lambda i:len(sft[length][i]['input_ids'])) for length in plan['train_lengths']};ri=max(range(len(replay)),key=lambda i:len(replay[i]['input_ids']))
 for i in range(len(SCHEDULE)*a.warmup_cycles):
  length=length_at(i);j=occurrence_before(i,length);step(model,wrapper,opt,cpt[length][j],sft[length][si[length]],replay[ri],native_table=native,weights=plan['loss_weights'],chunk_size=128,amp=True)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
 for i in range(len(SCHEDULE)*a.timed_cycles):
  length=length_at(i);j=occurrence_before(i,length);record=step(model,wrapper,opt,cpt[length][j],sft[length][si[length]],replay[ri],native_table=native,weights=plan['loss_weights'],chunk_size=128,amp=True);record['scheduled_length']=length;records.append(record)
 torch.cuda.synchronize()
 elapsed=time.monotonic()-started;tokens=sum(x['cpt_prediction_tokens']+x['sft_prediction_tokens']+x['replay_prediction_tokens'] for x in records)
 print(json.dumps({'status':'GPU_PROBE_PASS','discarded':True,'evidence_scope':'engineering memory/throughput qualification; not scientific evidence','asset_identity_policy':'user_attested_clone/no_sha_validation','records':records,'elapsed_seconds_excluding_load':elapsed,'supervised_tokens_per_second':tokens/elapsed,'peak_bytes':torch.cuda.max_memory_allocated(),'activation_checkpointing':bool(getattr(model,'_activation_checkpointing',False)),'compile_mode':model.__dict__.get('_compile_mode'),'optimizer_fused':True,'gpu':gpu}))
if __name__=='__main__':main()
