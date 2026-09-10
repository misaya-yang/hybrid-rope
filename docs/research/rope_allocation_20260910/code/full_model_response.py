"""Whole-model signed frequency response; weights frozen, exact target answer CE.
This measures a proposal direction. It is not a claim of successful extension.
"""
import argparse,json,time,types,contextlib
from pathlib import Path
import torch
from experiments.nongeometric_screen.worker import Worker,save

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--history',default='/root/autodl-tmp/bm_transfer_20260908');p.add_argument('--length',type=int,default=32768);p.add_argument('--offload',action='store_true');p.add_argument('--split',type=int);a=p.parse_args()
root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
w=Worker(root,a.history);w.apply({'table':w.tables['MrPro']})
model=w.model
base=torch.tensor(w.tables['MrPro']['values_float32'],device='cuda',dtype=torch.float32)
delta=torch.nn.Parameter(torch.zeros(64,device='cuda',dtype=torch.float32))
gain=w.tables['MrPro']['gain']
rot=model.model.rotary_emb
old_forward=rot.forward

def forward(self,x,position_ids):
    freq=base*torch.exp(-delta)
    inv=freq[None,:,None].expand(position_ids.shape[0],-1,1)
    pos=position_ids[:,None,:].float()
    with torch.autocast(device_type='cuda',enabled=False):
        phase=(inv.float()@pos.float()).transpose(1,2)
        emb=torch.cat((phase,phase),dim=-1)
        co,si=emb.cos()*gain,emb.sin()*gain
    return co.to(x.dtype),si.to(x.dtype)
rot.forward=types.MethodType(forward,rot)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
model.train()
if any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules()):
    raise ValueError('Nonzero dropout would change frozen response conditions')
rows=[r for r in w.screen if r['length_cap']==a.length and r['task'] in ('niah_multikey_2','niah_multiquery') and r['row_id'].endswith(('_0','_1'))]
if a.split is not None:rows=[r for r in rows if r['row_id'].endswith('_'+str(a.split))]
meta={'status':'RUNNING','weights_updated':False,'frequency_updated':False,'gain':gain,'rows':[r['row_id'] for r in rows],'loss':'Teacher-forced full correct-answer cross entropy, true whole-prefix gradient','length':a.length,'offload':a.offload,'scope':'Historical development examples; measurement, not independent holdout or success of a new allocation'}
save(root/'response_manifest.json',meta)
print(json.dumps({'phase':'FULL_MODEL_RESPONSE_STARTED','rows':len(rows)}),flush=True)
for row in rows:
    ans=w.tokenizer.encode(' '+', '.join(row['references']),add_special_tokens=False)
    ids=row['prompt_ids']+ans[:-1]
    tokens=torch.tensor([ids],device='cuda')
    target=torch.tensor(ans,device='cuda')
    start=time.monotonic();delta.grad=None
    hooks=torch.autograd.graph.save_on_cpu(pin_memory=True) if a.offload else contextlib.nullcontext()
    with hooks:
        out=model(tokens,use_cache=False,logits_to_keep=len(ans)).logits[0].float()
        losses=torch.nn.functional.cross_entropy(out,target,reduction='none');loss=losses.mean()
        loss.backward()
    g=delta.grad.detach().float().cpu()
    if not torch.isfinite(g).all():raise ValueError('Nonfinite full-model frequency gradient')
    rec={'row_id':row['row_id'],'prompt_sha256':row['prompt_sha256'],'answer_ids':ans,'loss':float(loss.detach()),'per_token_loss':losses.detach().cpu().tolist(),'gradient_log_period':g.tolist(),'seconds':time.monotonic()-start,'peak_allocated':torch.cuda.max_memory_allocated()}
    with (root/'full_model_response.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    print(json.dumps({'row':row['row_id'],'loss':rec['loss'],'seconds':rec['seconds'],'gradient_norm':float(g.norm()),'peak_GB':rec['peak_allocated']/1e9}),flush=True)
    del tokens,target,out,losses,loss
meta['status']='COMPLETE';save(root/'response_manifest.json',meta)
rot.forward=old_forward
