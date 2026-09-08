"""Real Q/K diagnostic, fixed native reader and static equal-count summary controls.

This is dense-trajectory operator evidence, not generated-answer or speedup evidence.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForImageTextToText,AutoModelForCausalLM,AutoConfig,AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
from groups import phase_groups,controls


def summary_scores(q,k,labels):
    # q [H,Q,D] already scaled; k [H,blocks,B,D].
    means=torch.stack([k[:,:,np.flatnonzero(labels==g),:].mean(2) for g in range(int(labels.max())+1)],dim=2)
    counts=torch.as_tensor(np.bincount(labels),device=q.device,dtype=q.dtype)
    scores=torch.einsum('hqd,hbrd->hqbr',q,means)
    return torch.logsumexp(scores+counts.log(),dim=-1)

def audit(q,kr,v,qn,kn,positions,omega,block=64,local=2048,topk=32):
    H,Q,D=q.shape;N=kr.shape[1];blocks=N//block;cut=blocks*block
    if blocks<=topk+local//block+2:raise ValueError('Not enough competing remote blocks')
    keys=kr[:,:cut].reshape(H,blocks,block,D)
    keysn=kn[:,:cut].reshape(H,blocks,block,D)
    starts=torch.arange(blocks,device=q.device)*block
    eligible=(starts[None,:]>0)&((starts[None,:]+block-1)<(positions[:,None]-local+1))
    all_scores=torch.einsum('hqd,hnd->hqn',q,kr)
    causal=torch.arange(N,device=q.device)[None,:]<=positions[:,None]
    all_scores.masked_fill_(~causal[None],-torch.inf)
    fullp=all_scores.softmax(-1);dense=torch.einsum('hqn,hnd->hqd',fullp,v)
    mass=fullp[:,:,:cut].reshape(H,Q,blocks,block).sum(-1)
    f=torch.logsumexp(all_scores[:,:,:cut].reshape(H,Q,blocks,block),-1)
    fn=torch.logsumexp(torch.einsum('hqd,hbtd->hqbt',qn,keysn),-1)
    means=summary_scores(q,keys,np.zeros(block,dtype=int))
    meansn=summary_scores(qn,keysn,np.zeros(block,dtype=int))
    grouped=controls(phase_groups(omega,block,4))
    estimated={'RoPEMean':means,'NoPEMean':meansn,**{name:summary_scores(q,keys,labels) for name,labels in grouped.items()}}
    records=[]
    eligible_h=eligible[None].expand(H,-1,-1)
    remote_total=(mass*eligible_h).sum(-1)
    for name,score in estimated.items():
        indices=score.masked_fill(~eligible_h,-torch.inf).topk(topk,-1).indices
        selected=torch.zeros_like(score,dtype=torch.bool).scatter_(-1,indices,True)&eligible_h
        retained=(mass*selected).sum(-1)/remote_total.clamp_min(1e-30)
        token_selected=torch.zeros(H,Q,N,device=q.device,dtype=torch.bool)
        token_selected[:,:,:cut]=selected.repeat_interleave(block,-1)
        mandatory=causal & ((torch.arange(N,device=q.device)[None,:]<block)|(torch.arange(N,device=q.device)[None,:]>=(positions[:,None]-local+1)))
        token_selected |= mandatory[None]
        p=all_scores.masked_fill(~token_selected,-torch.inf).softmax(-1)
        output=torch.einsum('hqn,hnd->hqd',p,v)
        rel=(output-dense).norm(dim=-1)/dense.norm(dim=-1).clamp_min(1e-12)
        gap=(f-score)[eligible_h]
        records.append({'method':name,'remote_retained_mass':float(retained.mean()),
            'headwise_value_relative_l2':float(rel.mean()),'mean_signed_logmass_error':float(gap.mean()),
            'mean_absolute_logmass_error':float(gap.abs().mean()),
            'per_head_retained_mass':retained.mean(-1).cpu().tolist(),
            'per_head_value_relative_l2':rel.mean(-1).cpu().tolist()})
    return {'records':records,'group_labels':{k:v.tolist() for k,v in grouped.items()},
        'counts':np.bincount(grouped['PSR']).tolist(),'mean_remote_probability':float(remote_total.mean()),
        'mean_abs_position_target_change':float((f-fn)[eligible_h].abs().mean()),
        'mean_abs_nope_summary_error':float((fn-meansn)[eligible_h].abs().mean()),
        'decomposition_max_residual':float(((f-meansn)-((f-fn)+(fn-meansn)))[eligible_h].abs().max()),
        'heads':H,'queries':Q,'blocks':blocks,'remote_topk':topk,'local_tokens':local,'block_size':block}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    config=AutoConfig.from_pretrained(a.model)
    loader=AutoModelForImageTextToText if config.model_type=='qwen3_5' else AutoModelForCausalLM
    if config.model_type=='qwen2':
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as rotate
    else:rotate=apply_rotary_pos_emb
    model=loader.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
    rotary=[m for name,m in model.named_modules() if name.endswith('rotary_emb') and hasattr(m,'inv_freq')]
    if len(rotary)!=1 or float(rotary[0].attention_scaling)!=1.:raise ValueError('This prototype requires native unit amplitude')
    omega=rotary[0].inv_freq.detach().float().cpu().numpy()
    layers={m.layer_idx:m for m in model.modules() if m.__class__.__name__ in ('Qwen3_5Attention','Qwen2Attention')}
    indices=sorted(layers);chosen=[indices[0],indices[len(indices)//2],indices[-1]]
    raw={i:{} for i in chosen};captured={};hooks=[]
    for i in chosen:
        m=layers[i]
        def pre(mod,args,kwargs,i=i):raw[i]['cos_sin']=kwargs['position_embeddings']
        def qhook(mod,args,out,i=i):
            if out.ndim==3:out=out.view(*out.shape[:2],-1,layers[i].head_dim)
            raw[i]['q']=out.transpose(1,2).detach()
        def khook(mod,args,out,i=i):
            if out.ndim==3:out=out.view(*out.shape[:2],-1,layers[i].head_dim)
            raw[i]['k']=out.transpose(1,2).detach()
        q_source=m.q_norm if hasattr(m,'q_norm') else m.q_proj
        k_source=m.k_norm if hasattr(m,'k_norm') else m.k_proj
        hooks.extend([m.register_forward_pre_hook(pre,with_kwargs=True),q_source.register_forward_hook(qhook),k_source.register_forward_hook(khook)])
    old=ALL_ATTENTION_FUNCTIONS.get_interface('sdpa',None)
    def interface(module,q,k,v,mask,**kwargs):
        if module in [layers[i] for i in chosen]:
            i=module.layer_idx;qn,kn=raw[i]['q'],raw[i]['k'];cos,sin=raw[i]['cos_sin']
            qcheck,kcheck=rotate(qn,kn,cos,sin)
            if not torch.equal(qcheck,q) or not torch.equal(kcheck,k):raise RuntimeError('Raw capture / rotary reconstruction parity failed')
            repeats=q.shape[1]//k.shape[1];tail=8
            data={'q':q[0,:,-tail:].float()*float(module.scaling),
                'kr':k[0].repeat_interleave(repeats,dim=0).float(),'v':v[0].repeat_interleave(repeats,dim=0).float(),
                'qn':qn[0,:,-tail:].float()*float(module.scaling),'kn':kn[0].repeat_interleave(repeats,dim=0).float(),
                'positions':torch.arange(k.shape[-2]-tail,k.shape[-2],device=q.device)}
            captured[i]=audit(**data,omega=omega)
            captured[i]['raw_reconstruction_bitwise']=True
            # Save bounded true activations once per sampled layer, for independent estimators.
            block=64;nb=k.shape[-2]//block
            for group in range(k.shape[1]):
                head_slice=slice(group*repeats,(group+1)*repeats)
                query_positions=data['positions'].repeat(repeats)
                starts=torch.arange(nb,device=q.device)*block
                eligible=(starts[None]>0)&((starts[None]+block-1)<(query_positions[:,None]-2048+1))
                np.savez_compressed(a.output/f'{current_id}_layer{i}_kv{group}.npz',
                  queries_scaled=data['q'][head_slice].reshape(-1,q.shape[-1]).cpu().numpy(),
                  keys_rotated=k[0,group,:nb*block].float().reshape(nb,block,-1).cpu().numpy(),
                  queries_nope_scaled=data['qn'][head_slice].reshape(-1,q.shape[-1]).cpu().numpy(),
                  keys_nope=kn[0,group,:nb*block].float().reshape(nb,block,-1).cpu().numpy(),
                  values=v[0,group,:nb*block].float().reshape(nb,block,-1).cpu().numpy(),
                  query_positions=query_positions.cpu().numpy(),block_starts=starts.cpu().numpy(),
                  eligible=eligible.cpu().numpy(),omega=omega,nope_is_amplitude_matched=np.array(True))

        return old(module,q,k,v,mask,**kwargs)
    ALL_ATTENTION_FUNCTIONS.register('sdpa',interface)
    rows=[json.loads(x) for x in a.inputs.read_text().splitlines()]
    selected=[x for x in rows if x.get('length_budget')==16384 and x['family'] in (0,1) and not x['reverse'] and x['task']=='marker_M7']
    if len(selected)!=2:raise ValueError('Expected two predetermined development prompts')
    (a.output/'status.json').write_text(json.dumps({'status':'RUNNING','pid':os.getpid(),'rows':[x['row_id'] for x in selected]}))
    records=[];started=time.monotonic()
    try:
        with torch.inference_mode():
            for row in selected:
                current_id=row['row_id'];captured.clear()
                prompt=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                inputs=tokenizer(prompt,return_tensors='pt').to('cuda')
                model(**inputs,use_cache=False,logits_to_keep=1)
                for i in chosen:
                    result={'row_id':current_id,'layer':i,'input_tokens':inputs['input_ids'].shape[-1],**captured[i]}
                    records.append(result);print(json.dumps({k:result[k] for k in ('row_id','layer','counts','records')}),flush=True)
    finally:
        ALL_ATTENTION_FUNCTIONS.register('sdpa',old)
        for h in hooks:h.remove()
    result={'status':'COMPLETE','evidence':'Dense-trajectory operator diagnosis; no answer improvement or speedup claim',
      'model_type':config.model_type,'model':str(a.model),'record_count':len(records),'records':records,'seconds':time.monotonic()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'selection_granularity':'per_query_head_diagnostic_only; not a deployed shared_GQA_selector'}
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    (a.output/'status.json').write_text(json.dumps({'status':'COMPLETE','records':len(records)}))
if __name__=='__main__':main()
