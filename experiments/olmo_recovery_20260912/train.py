#!/usr/bin/env python3
"""Resumable matched OLMo multiscale recovery; default validates without GPU work."""
import argparse,json,math,os,random,shutil
from pathlib import Path
import numpy as np,torch
from experiments.evq_recovery.data import JsonlIndex
from scripts.experiments.cross_audit.tables import check_table
from .runtime import load_model,step,validate_cuda
from .length_schedule import SCHEDULE,length_at,micro_sequences_for_tokens,occurrence_before,tokens_before

def lr_for_tokens(processed,total,peak):
 warm=max(1,math.ceil(total*.05))
 if processed<=warm:return peak*processed/warm
 return peak*(.1+.45*(1+math.cos(math.pi*(processed-warm)/max(1,total-warm))))
def order(i,n,seed):
 epoch,pos=divmod(i,n);x=list(range(n));random.Random(seed+epoch).shuffle(x);return x[pos]
def checked_sft(pool,index,length):
 row=pool[index];prompt=int(row['target_start']);total=len(row['input_ids'])
 minimum=7*length//8
 if prompt<minimum or total>length:raise ValueError(f'SFT row violates locked {length} physical pool: prompt={prompt}, total={total}')
 return row
def verify_bound(plan,manifest_path,tables_path):
 model=Path(plan['model_path']);config=json.loads((model/'config.json').read_text())
 if config.get('model_type')!=plan['model_type'] or config.get('hidden_size')//config.get('num_attention_heads')!=128:
  raise ValueError('model config type/head dimension differs from plan')
 index=model/'model.safetensors.index.json'
 expected=sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.is_file() else ['model.safetensors']
 if any(not (model/name).is_file() for name in expected):raise FileNotFoundError('one or more model weight shards are missing')
 manifest=json.loads(Path(manifest_path).read_text())
 for family in ('cpt_train_by_length','sft_train_by_length'):
  if set(manifest[family])!={'8192','16384'}:raise ValueError(family+' must contain exact 8K/16K pools')
  for length,entry in manifest[family].items():
   path=Path(entry['path'])
   if not path.is_file() or path.stat().st_size==0:raise ValueError(f'training data missing/empty: {family}/{length}')
 entry=manifest['replay_train'];path=Path(entry['path'])
 if not path.is_file() or path.stat().st_size==0:raise ValueError('replay training data missing/empty')
 tables=json.loads(Path(tables_path).read_text())
 if set(tables)!=set(plan['arms']):raise ValueError('table arm set drift')
 for name,entry in tables.items():
  values=np.asarray(entry['values_float32'],dtype=np.float32);check_table(values,128)
 for length in plan['train_lengths']:
  cpt=np.load(manifest['cpt_train_by_length'][str(length)]['path'],mmap_mode='r')
  if cpt.ndim!=2 or cpt.shape[1]!=length+1:raise ValueError(f'CPT must contain real {length}+1 next-token windows')
 if plan['length_schedule']!=list(SCHEDULE):raise ValueError('length schedule drift')
 return manifest,tables
def semantic_contract(plan,arm,accumulation):
 return {'arm':arm,'model_type':plan['model_type'],'cosh_contract':plan['cosh_contract'],'train_lengths':plan['train_lengths'],'length_schedule':plan['length_schedule'],'rank':plan['rank'],'alpha':plan['alpha'],'dropout':plan['dropout'],'modules':plan['modules'],'seed':plan['seed'],'learning_rate':plan['learning_rate'],'betas':plan['betas'],'weight_decay':plan['weight_decay'],'loss_weights':plan['loss_weights'],'schedule_cpt_tokens':plan['schedule_cpt_tokens'],'gradient_accumulation':accumulation}
def rng_state():
 return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()}
def restore_rng(state):
 torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda']);np.random.set_state(state['numpy']);random.setstate(state['python'])
def recoverable_checkpoint(path):
 if path.is_dir() and (path/'state.json').is_file():return path
 previous=path.with_name(path.name+'.old')
 if previous.is_dir() and (previous/'state.json').is_file():return previous
 raise FileNotFoundError(f'no complete checkpoint at {path} or {previous}')
def save_bundle(path,wrapper,opt,state):
 temporary=path.with_name(path.name+'.incomplete');old=path.with_name(path.name+'.old')
 if temporary.exists():shutil.rmtree(temporary)
 temporary.mkdir(parents=True);wrapper.save_pretrained(temporary,safe_serialization=True);torch.save({'optimizer':opt.state_dict(),'rng':rng_state()},temporary/'training.pt');(temporary/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 if old.exists():shutil.rmtree(old)
 if path.exists():os.replace(path,old)
 os.replace(temporary,path)
 if old.exists():shutil.rmtree(old)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=['Native','Cosh'],required=True);p.add_argument('--until-cpt-tokens',type=int,default=33554432);p.add_argument('--gradient-accumulation',type=int);p.add_argument('--resume',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if a.until_cpt_tokens<=0:raise ValueError('training endpoint must be positive')
 if a.resume:a.resume=recoverable_checkpoint(a.resume)
 plan_path=a.root/'plan.json';tables_path=a.root/'tables.json';plan=json.loads(plan_path.read_text());manifest_path=Path(plan['data_manifest']);micro_stop=micro_sequences_for_tokens(a.until_cpt_tokens)
 accumulation=plan['gradient_accumulation']
 if a.gradient_accumulation is not None and a.gradient_accumulation!=accumulation:raise ValueError('gradient accumulation is locked by plan')
 if a.until_cpt_tokens>plan['schedule_cpt_tokens'] or micro_stop%accumulation:raise ValueError('endpoint must align with the locked length cycle and accumulation')
 manifest,tables=verify_bound(plan,manifest_path,tables_path)
 contract=semantic_contract(plan,a.arm,accumulation)
 print(json.dumps({'status':'DRY_RUN_VALIDATED' if not a.execute else 'STARTING','arm':a.arm,'cpt_tokens':a.until_cpt_tokens,'micro_sequences':micro_stop,'gradient_accumulation':accumulation}))
 if not a.execute:return
 gpu=validate_cuda()
 cpt={length:np.load(manifest['cpt_train_by_length'][str(length)]['path'],mmap_mode='r') for length in plan['train_lengths']};sft={length:JsonlIndex(manifest['sft_train_by_length'][str(length)]['path']) for length in plan['train_lengths']};replay=JsonlIndex(manifest['replay_train']['path']);model,wrapper=load_model(plan,tables[a.arm],checkpoint=a.resume);model._frozen_native_teacher_cache_path=str(a.root/'native_teacher_trajectories.jsonl');opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['learning_rate'],betas=tuple(plan['betas']),weight_decay=0.,fused=True)
 start=0;prior_totals={'cpt':0,'sft':0,'replay':0};out=a.root/'runs'/a.arm
 if a.resume:
  state=json.loads((a.resume/'state.json').read_text())
  if state.get('semantic_contract') is not None and state['semantic_contract']!=contract:raise ValueError('resume semantic training contract mismatch')
  if state.get('arm',a.arm)!=a.arm or state.get('gradient_accumulation',accumulation)!=accumulation:raise ValueError('resume arm/accumulation mismatch')
  adapter=json.loads((a.resume/'adapter_config.json').read_text())
  targets=set(adapter.get('target_modules',[]))
  if int(adapter.get('r',-1))!=plan['rank'] or int(adapter.get('lora_alpha',-1))!=plan['alpha'] or float(adapter.get('lora_dropout',-1))!=plan['dropout'] or targets!=set(plan['modules']):raise ValueError('resume adapter semantic configuration mismatch')
  start=state['micro_sequences']
  if start>micro_stop:raise ValueError('resume checkpoint is beyond the requested endpoint')
  if state['cpt_tokens']!=tokens_before(start):raise ValueError('resume real CPT-token accounting mismatch')
  prior_totals=state.get('prediction_tokens',prior_totals)
  training=torch.load(a.resume/'training.pt',map_location='cpu',weights_only=False);opt.load_state_dict(training['optimizer']);restore_rng(training['rng'])
 elif out.exists():raise FileExistsError(out)
 out.mkdir(parents=True,exist_ok=bool(a.resume));native=np.asarray(tables['Native']['values_float32'],dtype=np.float32)
 (out/'status.json').write_text(json.dumps({'status':'RUNNING','arm':a.arm,'start_micro_sequence':start,'target_cpt_tokens':a.until_cpt_tokens,'asset_identity_policy':'user_attested_clone/no_sha_validation','gpu':gpu},indent=2)+'\n')
 checkpoints=set(plan['checkpoint_cpt_tokens'])|{a.until_cpt_tokens}
 with (out/'steps.jsonl').open('a') as log:
  started=__import__('time').monotonic();totals=dict(prior_totals)
  for begin in range(start,micro_stop,accumulation):
   records=[];update=begin//accumulation;done_after=tokens_before(begin+accumulation);lr=lr_for_tokens(done_after,plan['schedule_cpt_tokens'],plan['learning_rate'])
   for group in opt.param_groups:group['lr']=lr
   for offset in range(accumulation):
    i=begin+offset;length=length_at(i);j=occurrence_before(i,length);record=step(model,wrapper,opt,cpt[length][order(j,len(cpt[length]),plan['seed']+length)],checked_sft(sft[length],order(j,len(sft[length]),plan['seed']+100+length),length),replay[order(i,len(replay),plan['seed']+200)],native_table=native,weights=plan['loss_weights'],chunk_size=128,amp=True,zero_grad=offset==0,do_step=offset==accumulation-1,gradient_scale=1/accumulation);record['scheduled_length']=length;records.append(record);totals['cpt']+=record['cpt_prediction_tokens'];totals['sft']+=record['sft_prediction_tokens'];totals['replay']+=record['replay_prediction_tokens']
   done=tokens_before(begin+accumulation);log.write(json.dumps({'optimizer_update':update+1,'micro_sequences':begin+accumulation,'cpt_tokens':done,'lr':lr,'micro_records':records})+'\n');log.flush()
   state={'arm':a.arm,'micro_sequences':begin+accumulation,'cpt_tokens':done,'prediction_tokens':totals,'semantic_contract':contract,'plan':str(plan_path.resolve()),'gradient_accumulation':accumulation,'asset_identity_policy':'user_attested_clone/no_sha_validation'}
   if (update+1)%plan['resume_every_updates']==0:save_bundle(out/'resume',wrapper,opt,state)
   if done in checkpoints:save_bundle(out/f'checkpoint-{done}',wrapper,opt,state)
 completion={'status':'TRAINING_ENDPOINT_COMPLETE_NEEDS_EVALUATION','arm':a.arm,'cpt_tokens':tokens_before(micro_stop),'micro_sequences':micro_stop,'prediction_tokens':totals,'elapsed_seconds':__import__('time').monotonic()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'gpu':gpu,'gradient_accumulation':accumulation,'asset_identity_policy':'user_attested_clone/no_sha_validation'}
 (out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
 (out/'status.json').write_text(json.dumps(completion,indent=2)+'\n')
if __name__=='__main__':main()
