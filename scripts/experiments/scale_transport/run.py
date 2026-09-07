"""Two-hour frozen-weight pilot: response collection, guards, then own-method QA.

No optimizer, no training, no automatic retries. Run only with an authorized
manifest and external process-group timeout. Source checkpoints remain immutable.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from scripts.experiments.cross_audit.runtime import cuda_runtime
from scripts.experiments.cross_audit.tables import native_table,transform,tensor_sha
from scripts.experiments.scale_transport.math import replay,background,estimate_beta,quantile_moments,bounded_isotonic
from scripts.eval.longbench_metrics import qa_f1_score

LAYERS=(5,11,17,23,29,35)
LENGTHS=(8192,16384,32768)
GAIN=1+.1*math.log(4)


def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)


def install(model,arr,gain):
    arr=np.asarray(arr,dtype=np.float32)
    if arr.shape!=(64,) or not np.isfinite(arr).all() or np.any(arr<0):raise ValueError('invalid array')
    r=model.model.rotary_emb
    if r.rope_type!='default':raise ValueError('unexpected dynamic source RoPE')
    r.inv_freq=torch.tensor(arr,device='cuda');r.original_inv_freq=r.inv_freq.clone();r.attention_scaling=float(gain)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--prepared',required=True);p.add_argument('--out',required=True)
    p.add_argument('--budget-seconds',type=int,required=True);a=p.parse_args()
    if not 0<a.budget_seconds<=7200:raise ValueError('authorized ceiling is 7200 seconds')
    root=Path(a.root);prepared=Path(a.prepared);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+a.budget_seconds
    def tick(stage,**kw):
        if time.monotonic()>deadline:raise TimeoutError('pilot global budget')
        event=dict(stage=stage,elapsed_seconds=time.monotonic()-started,**kw)
        write(out/'progress.json',event);print(json.dumps(event),flush=True)
    hardware=cuda_runtime();tick('load_model',hardware=hardware)
    ready=json.loads((root/'model_ready.json').read_text());manifest=json.loads((prepared/'manifest.json').read_text())
    for rel,h in manifest['files'].items():
        if hashlib.sha256((prepared/rel).read_bytes()).hexdigest()!=h:raise ValueError('prepared data drift')
    model=AutoModelForCausalLM.from_pretrained(root/'model',local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
    tok=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
    c=model.config
    if (c.model_type,c.num_hidden_layers,c.num_attention_heads,c.num_key_value_heads)!=("qwen2",36,16,2):raise ValueError('pilot only qualified for exact Qwen 3B architecture')
    if c.use_sliding_window or c.hidden_size//c.num_attention_heads!=128:raise ValueError('unexpected attention layout')
    native=model.model.rotary_emb.inv_freq.float().cpu().numpy().copy()
    expected=native_table(128,1e6)
    if not np.allclose(native,expected,rtol=2e-6,atol=0):raise ValueError('Native frequency formula mismatch')
    write(out/'source_identity.json',dict(actual_native_sha256=tensor_sha(native),max_cpu_formula_relative_error=float(np.max(np.abs(native/expected-1))),revision=ready['revision']))
    mr,_,meta=transform(native,dim=128,base=1e6,reference_length=32768,scale=4,method='mrpro')
    if (meta['low'],meta['high'])!=(23,40):raise ValueError('Mr reference identity')
    arrays={'Native':native,'MrPro':mr}
    cache_dir=out/'cache';cache_dir.mkdir();profiles={};native_nll={};parities=[]
    state={}
    handles=[]
    for layer in LAYERS:
        def hook(module,args,kwargs,result,layer=layer):
            hidden=kwargs.get('hidden_states',args[0] if args else None)
            if hidden is None or hidden.shape[0]!=1:raise ValueError('hook layout')
            L=hidden.shape[1];pos=torch.arange(L-57,L,8,device='cuda')
            q=module.q_proj(hidden[:,pos]).reshape(8,16,128).transpose(0,1).detach()
            k=module.k_proj(hidden).reshape(L,2,128).transpose(0,1).detach()
            v=module.v_proj(hidden).reshape(L,2,128).transpose(0,1).detach()
            y,H=replay(q,k,v,module.o_proj.weight,torch.tensor(native,device='cuda'),1.,pos,response=True)
            actual=result[0][0,pos].float();rel=float((y-actual).norm()/actual.norm().clamp_min(1e-8))
            if rel>.05:raise RuntimeError(f'hook/Flash parity failure {layer}: {rel}')
            if not math.isfinite(rel) or not torch.isfinite(H).all():raise RuntimeError('nonfinite response')
            parities.append(rel);state['H']+=H.cpu().numpy()
            if L==32768:
                cache=dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),pos=pos.cpu(),y_native=y.cpu())
                torch.save(cache,cache_dir/f'{state["doc"]}_{layer}.pt')
                inds=torch.linspace(0,L-1,128,device=k.device).long()
                torch.save(k[:,inds].cpu(),cache_dir/f'{state["doc"]}_{layer}_keys.pt')
                for cut in (8192,16384):
                    _,cropped=replay(q,k[:,-cut:],v[:,-cut:],module.o_proj.weight,torch.tensor(native,device='cuda'),1.,pos-(L-cut),response=True)
                    state['crop'][cut]+=cropped.cpu().numpy()
        handles.append(model.model.layers[layer].self_attn.register_forward_hook(hook,with_kwargs=True))
    with torch.inference_mode():
        for di,doc in enumerate(manifest['docs']):
            docid=Path(doc['file']).stem;tokens=np.load(prepared/doc['file']);profiles[docid]={}
            for L in LENGTHS:
                tick('collect',doc=di,length=L)
                state.update(doc=docid,H=np.zeros((64,L)),crop={n:np.zeros((64,n)) for n in (8192,16384)})
                ids=torch.tensor(tokens[-L-1:-1].astype(np.int64),device='cuda')[None,:]
                # Last-logit retention avoids allocating the full sequence x vocabulary output.
                result=model(ids,use_cache=False,logits_to_keep=256)
                # ids end at tokens[-2], target labels must end at tokens[-1].
                labels=torch.tensor(tokens[-256:].astype(np.int64),device='cuda')
                native_nll[f'{docid}:{L}']=float(torch.nn.functional.cross_entropy(result.logits[0].float(),labels))
                profiles[docid][L]=state['H'];np.save(out/f'{docid}_{L}_profile.npy',state['H'])
                if L==32768:
                    for cut,H in state['crop'].items():np.save(out/f'{docid}_crop_{cut}.npy',H)
            tick('collected',doc=di,max_hook_relative_error=max(parities))
    for handle in handles:handle.remove()
    write(out/'collection.json',dict(status='COMPLETE',native_nll=native_nll,max_hook_relative_error=max(parities),layers=LAYERS,queries=8))
    C=[Path(d['file']).stem for d in manifest['docs'] if d['split']=='C'];V=[Path(d['file']).stem for d in manifest['docs'] if d['split']=='V']
    sums={L:sum(profiles[d][L] for d in C) for L in LENGTHS}
    beta=np.zeros(64);beta_early=np.zeros(64);inactive=[];details=[];weights=np.ones(64)
    for j in range(64):
        if min(sums[L][j].sum() for L in LENGTHS)<=0:
            inactive.append(j);details.append({'slot':j,'inactive':True});continue
        late=estimate_beta(sums[16384][j],sums[32768][j]);early=estimate_beta(sums[8192][j],sums[16384][j])
        beta[j]=late['beta'];beta_early[j]=early['beta'];weights[j]=sums[32768][j].sum()*4**(2*beta[j])*late['second_moment']
        details.append(dict(slot=j,late=late,early=early))
    # Freeze Mr's high region. Inactive slots fall back to Mr, not an arbitrary utility cutoff.
    lo=native.astype(float)/4;hi=native.astype(float).copy();proposal=native/4**beta
    for j in list(range(24))+inactive:lo[j]=hi[j]=proposal[j]=float(mr[j]);weights[j]=max(weights[j],1.)
    weights=weights/max(weights.max(),1e-300)
    proposal=bounded_isotonic(proposal,weights,lo,hi).astype(np.float32)
    prediction=[]
    m_mr=-np.log(mr/native)/math.log(4)
    for doc in V:
        scores={name:[0.,0.] for name in ['transport','fixed','full','Mr_interpretation']}
        for j in range(24,64):
            A=profiles[doc][16384][j];B=profiles[doc][32768][j]
            if min(A.sum(),B.sum())<=0:continue
            x,cross,y=quantile_moments(A,B)
            for name,b in [('transport',beta_early[j]),('fixed',0),('full',1),('Mr_interpretation',m_mr[j])]:
                scores[name][0]+=A.sum()*(2**(2*b)*x-2*2**b*cross+y);scores[name][1]+=A.sum()*x
        prediction.append(dict(doc=doc,errors={n:v[0]/max(v[1],1e-300) for n,v in scores.items()}))
    crop_sums={L:sum(np.load(out/f'{d}_crop_{L}.npy') for d in C) for L in (8192,16384)}
    crop_diagnostics=[]
    for j in range(24,64):
        if min(crop_sums[L][j].sum() for L in (8192,16384))>0:
            crop_diagnostics.append(dict(slot=j,real_early_beta=float(beta_early[j]),fixed_content_crop=estimate_beta(crop_sums[8192][j],crop_sums[16384][j])))
    write(out/'visibility_control.json',crop_diagnostics)
    write(out/'proposal.json',dict(status='UNGUARDED_PROPOSAL',slots=details,proposal=proposal.tolist(),inactive=inactive,prediction=prediction,scope='six fixed layers; visibility controls are saved separately; not causal proof'))
    candidates=[((1-l)*mr+l*proposal).astype(np.float32) for l in [0,.25,.5,.75,1.]]
    metrics=[]
    with torch.inference_mode():
        for ci,arr in enumerate(candidates):
            DN=BG=0.;num=0;freq=torch.tensor(arr,device='cuda')
            for di,doc in enumerate(C):
                for layer in LAYERS:
                    tick('guard_replay',candidate=ci,doc=di,layer=layer)
                    cache=torch.load(cache_dir/f'{doc}_{layer}.pt',weights_only=True);otherk=torch.load(cache_dir/f'{C[(di+1)%len(C)]}_{layer}_keys.pt',weights_only=True).to('cuda')
                    q,k,v,pos=[cache[n].to('cuda') for n in ['q','k','v','pos']];wo=model.model.layers[layer].self_attn.o_proj.weight
                    yn=cache['y_native'].to('cuda')
                    yc,_=replay(q,k,v,wo,freq,GAIN,pos);DN+=float((yc-yn).square().sum(-1).mean())
                    gen=torch.Generator().manual_seed(9300+di*100+layer);dist=torch.randint(1,131073,(8,128),generator=gen).to('cuda')
                    BG+=float(background(q,otherk,freq,GAIN,dist));num+=1
            if not all(math.isfinite(x) for x in (DN,BG)):raise RuntimeError('nonfinite replay guard')
            metrics.append(dict(lambda_value=[0,.25,.5,.75,1.][ci],DN=DN/num,background_logmeanexp=BG/num))
            write(out/'guards.json',metrics)
    selected=0
    for i,row in enumerate(metrics):
        if row['DN']<=metrics[0]['DN']+1e-6*max(1.,metrics[0]['DN']) and row['background_logmeanexp']<=metrics[0]['background_logmeanexp']+1e-6*max(1.,abs(metrics[0]['background_logmeanexp'])):selected=i
    final=candidates[selected];np.save(out/'final_inv_freq.npy',final)
    deployment=dict(status='REFERENCE_ONLY' if selected==0 else 'CANDIDATE_READY',selected_lambda=metrics[selected]['lambda_value'],gain=GAIN,table_sha256=tensor_sha(final),actual_m=(-np.log(final/native)/math.log(4)).tolist(),source_revision=ready['revision'])
    write(out/'deployment.json',deployment)
    if selected==0:
        write(out/'manifest.json',dict(status='REFERENCE_ONLY',elapsed_seconds=time.monotonic()-started,hardware=hardware,reason='No new table under frozen guards; no duplicate MrPro evaluation'));return
    rows=[json.loads(l) for l in (prepared/'eval.jsonl').read_text().splitlines()]
    eos=tok.eos_token_id;generation=GenerationConfig(do_sample=False,num_beams=1,use_cache=True,eos_token_id=eos,pad_token_id=tok.pad_token_id or eos)
    records=[]
    with (out/'examples.jsonl').open('x') as f,torch.inference_mode():
        # Reuse Native forward requirements: short generated QA anchor only, never a full opponent matrix.
        jobs=[('Native',native,1.,r) for r in rows if r['bucket']=='native']+[('ScaleTransport',final,GAIN,r) for r in rows]
        for arm,arr,gain,row in jobs:
            tick('generation',arm=arm,row=row['row_id']);install(model,arr,gain)
            ids=torch.tensor(row['ids'],device='cuda')[None,:];before=time.monotonic();torch.cuda.reset_peak_memory_stats()
            output=model.generate(ids,generation_config=generation,max_new_tokens=row['budget'],logits_to_keep=1)
            new=output[0,ids.shape[1]:].tolist();text=tok.decode(new,skip_special_tokens=True);ended=bool(new and new[-1]==eos)
            if tensor_sha(model.model.rotary_emb.inv_freq.cpu().numpy())!=tensor_sha(arr):raise RuntimeError('deployment table drift')
            rec={k:row[k] for k in ['row_id','task','bucket','references','input_tokens','context_sha256']}
            rec.update(arm=arm,output_text=text,generated_ids=new,eos=ended,qa_f1=qa_f1_score(text,row['references']),seconds=time.monotonic()-before,peak_bytes=torch.cuda.max_memory_allocated(),table_sha256=tensor_sha(arr))
            f.write(json.dumps(rec)+'\n');f.flush();records.append(rec)
    write(out/'manifest.json',dict(status='COMPLETE',scope='own-method pilot, natural QA subset; not a matched published leaderboard comparison',hardware=hardware,source_revision=ready['revision'],deployment=deployment,rows=len(records),elapsed_seconds=time.monotonic()-started,examples_sha256=hashlib.sha256((out/'examples.jsonl').read_bytes()).hexdigest()))

if __name__=='__main__':main()
