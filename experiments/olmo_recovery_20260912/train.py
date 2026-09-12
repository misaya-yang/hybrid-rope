#!/usr/bin/env python3
"""Resumable matched Llama recovery training; default prints the locked work only."""
import argparse,hashlib,json,math,os,random,shutil
from pathlib import Path
import numpy as np,torch
from experiments.evq_recovery.data import JsonlIndex
from scripts.experiments.cross_audit.tables import check_table,tensor_sha
from .runtime import load_model,step,trainable_sha,validate_cuda

def lr_for(update,total,peak):
 warm=max(1,math.ceil(total*.05))
 if update<warm:return peak*(update+1)/warm
 return peak*(.1+.45*(1+math.cos(math.pi*(update-warm)/max(1,total-warm))))
def order(i,n,seed):
 epoch,pos=divmod(i,n);x=list(range(n));random.Random(seed+epoch).shuffle(x);return x[pos]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def verify_bound(plan,manifest_path,tables_path):
 if sha(manifest_path)!=plan['data_manifest_sha256']:raise ValueError('data manifest identity drift')
 model=Path(plan['model_path']);identity=plan['model_identity']
 if identity['weights']=='PENDING_HASH_AT_ACQUISITION':raise ValueError('model weights remain unbound; regenerate plan after acquisition')
 for group in ('metadata','weights'):
  for name,expected in identity[group].items():
   if sha(model/name)!=expected:raise ValueError('model identity drift: '+name)
 root=Path(__file__).resolve().parents[2]
 for name,expected in plan['code_identity'].items():
  if sha(root/name)!=expected:raise ValueError('training code identity drift: '+name)
 manifest=json.loads(Path(manifest_path).read_text())
 for key in ('cpt_train','sft_train','replay_train'):
  entry=manifest[key];path=Path(entry['path'])
  if not path.is_file() or sha(path)!=entry['sha256']:raise ValueError('training data identity drift: '+key)
 tables=json.loads(Path(tables_path).read_text())
 if set(tables)!=set(plan['arms']):raise ValueError('table arm set drift')
 for name,entry in tables.items():
  values=np.asarray(entry['values_float32'],dtype=np.float32);check_table(values,128)
  if tensor_sha(values)!=entry['tensor_sha256']:raise ValueError('table tensor drift: '+name)
 cpt=np.load(manifest['cpt_train']['path'],mmap_mode='r')
 if cpt.ndim!=2 or cpt.shape[1]!=plan['train_length']+1:raise ValueError('CPT must contain real 16K next-token windows')
 return manifest,tables
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
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=['Native','Cosh','OfficialYaRN'],required=True);p.add_argument('--until-cpt-tokens',type=int,default=33554432);p.add_argument('--gradient-accumulation',type=int);p.add_argument('--resume',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if a.resume:a.resume=recoverable_checkpoint(a.resume)
 plan_path=a.root/'plan.json';tables_path=a.root/'tables.json';plan=json.loads(plan_path.read_text());manifest_path=Path(plan['data_manifest']);manifest=json.loads(manifest_path.read_text());contract_hash=sha(plan_path)+sha(tables_path)+sha(manifest_path);length=plan['train_length'];micro_stop=a.until_cpt_tokens//length;total_micro=plan['schedule_cpt_tokens']//length
 accumulation=plan['gradient_accumulation']
 if a.gradient_accumulation is not None and a.gradient_accumulation!=accumulation:raise ValueError('gradient accumulation is locked by plan')
 if a.until_cpt_tokens%length or a.until_cpt_tokens>plan['schedule_cpt_tokens'] or micro_stop%accumulation:raise ValueError('endpoint must align with 16K and accumulation')
 manifest,tables=verify_bound(plan,manifest_path,tables_path)
 print(json.dumps({'status':'DRY_RUN_VALIDATED' if not a.execute else 'STARTING','arm':a.arm,'cpt_tokens':a.until_cpt_tokens,'micro_sequences':micro_stop,'gradient_accumulation':accumulation}))
 if not a.execute:return
 gpu=validate_cuda()
 cpt=np.load(manifest['cpt_train']['path'],mmap_mode='r');sft=JsonlIndex(manifest['sft_train']['path']);replay=JsonlIndex(manifest['replay_train']['path']);model,wrapper=load_model(plan,tables[a.arm],checkpoint=a.resume);initial_trainable_sha=trainable_sha(model);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['learning_rate'],betas=tuple(plan['betas']),weight_decay=0.)
 start=0;out=a.root/'runs'/a.arm
 if a.resume:
  state=json.loads((a.resume/'state.json').read_text())
  if state['contract_hash']!=contract_hash or state['arm']!=a.arm or state['gradient_accumulation']!=accumulation:raise ValueError('resume contract/table/data/arm/accumulation mismatch')
  start=state['micro_sequences'];training=torch.load(a.resume/'training.pt',map_location='cpu',weights_only=False);opt.load_state_dict(training['optimizer']);restore_rng(training['rng'])
 elif out.exists():raise FileExistsError(out)
 out.mkdir(parents=True,exist_ok=bool(a.resume));native=np.asarray(tables['Native']['values_float32'],dtype=np.float32)
 (out/'status.json').write_text(json.dumps({'status':'RUNNING','arm':a.arm,'start_micro_sequence':start,'target_cpt_tokens':a.until_cpt_tokens,'initial_trainable_sha256':initial_trainable_sha,'gpu':gpu},indent=2)+'\n')
 checkpoints=set(plan['checkpoint_cpt_tokens'])|{a.until_cpt_tokens}
 with (out/'steps.jsonl').open('a') as log:
  started=__import__('time').monotonic();totals={'cpt':0,'sft':0,'replay':0}
  for begin in range(start,micro_stop,accumulation):
   records=[];update=begin//accumulation;lr=lr_for(update,total_micro//accumulation,plan['learning_rate'])
   for group in opt.param_groups:group['lr']=lr
   for offset in range(accumulation):
    i=begin+offset;record=step(model,wrapper,opt,cpt[order(i,len(cpt),plan['seed'])],sft[order(i,len(sft),plan['seed']+100)],replay[order(i,len(replay),plan['seed']+200)],native_table=native,weights=plan['loss_weights'],chunk_size=128,amp=True,zero_grad=offset==0,do_step=offset==accumulation-1,gradient_scale=1/accumulation);records.append(record);totals['cpt']+=record['cpt_prediction_tokens'];totals['sft']+=record['sft_prediction_tokens'];totals['replay']+=record['replay_prediction_tokens']
   done=(begin+accumulation)*length;log.write(json.dumps({'optimizer_update':update+1,'micro_sequences':begin+accumulation,'cpt_tokens':done,'lr':lr,'micro_records':records})+'\n');log.flush()
   state={'arm':a.arm,'micro_sequences':begin+accumulation,'cpt_tokens':done,'contract_hash':contract_hash,'plan':str(plan_path.resolve()),'gradient_accumulation':accumulation}
   if (update+1)%plan['resume_every_updates']==0:save_bundle(out/'resume',wrapper,opt,state)
   if done in checkpoints:save_bundle(out/f'checkpoint-{done}',wrapper,opt,state)
 completion={'status':'TRAINING_ENDPOINT_COMPLETE_NEEDS_EVALUATION','arm':a.arm,'cpt_tokens':micro_stop*length,'prediction_tokens':totals,'elapsed_seconds':__import__('time').monotonic()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'gpu':gpu,'initial_trainable_sha256':initial_trainable_sha,'gradient_accumulation':accumulation}
 (out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
 (out/'status.json').write_text(json.dumps(completion,indent=2)+'\n')
if __name__=='__main__':main()
