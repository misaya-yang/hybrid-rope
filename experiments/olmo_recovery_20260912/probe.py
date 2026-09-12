#!/usr/bin/env python3
"""Two-update disposable 5090 memory probe; dry-run unless --execute."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from experiments.evq_recovery.data import JsonlIndex
from .runtime import load_model,step,trainable_sha,validate_cuda
from .train import verify_bound

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=['Native','Cosh','OfficialYaRN'],required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();plan=json.loads((a.root/'plan.json').read_text());manifest,tables=verify_bound(plan,Path(plan['data_manifest']),a.root/'tables.json');summary={'status':'DRY_RUN_VALIDATED','updates':2,'discarded':True,'evidence_scope':'engineering memory/throughput qualification; not scientific evidence','arm':a.arm,'train_length':16384};print(json.dumps(summary))
 if not a.execute:return
 gpu=validate_cuda()
 cpt=np.load(manifest['cpt_train']['path'],mmap_mode='r');sft=JsonlIndex(manifest['sft_train']['path']);replay=JsonlIndex(manifest['replay_train']['path']);model,wrapper=load_model(plan,tables[a.arm]);initial_sha=trainable_sha(model);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['learning_rate'],betas=tuple(plan['betas']),weight_decay=0.)
 native=np.asarray(tables['Native']['values_float32'],dtype=np.float32);records=[]
 si=max(range(len(sft)),key=lambda i:len(sft[i]['input_ids']));ri=max(range(len(replay)),key=lambda i:len(replay[i]['input_ids']));started=time.monotonic()
 for i in range(2):records.append(step(model,wrapper,opt,cpt[i],sft[si],replay[ri],native_table=native,weights=plan['loss_weights'],chunk_size=128,amp=True))
 elapsed=time.monotonic()-started;tokens=sum(x['cpt_prediction_tokens']+x['sft_prediction_tokens']+x['replay_prediction_tokens'] for x in records)
 print(json.dumps({'status':'GPU_PROBE_PASS','discarded':True,'evidence_scope':'engineering memory/throughput qualification; not scientific evidence','records':records,'elapsed_seconds_excluding_load':elapsed,'supervised_tokens_per_second':tokens/elapsed,'peak_bytes':torch.cuda.max_memory_allocated(),'gpu':gpu,'trainable_initial_sha256':initial_sha,'trainable_final_sha256':trainable_sha(model)}))
if __name__=='__main__':main()
