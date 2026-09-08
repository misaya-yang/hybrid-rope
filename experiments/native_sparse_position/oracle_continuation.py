"""Query-blind prefix + complete generation under exact R/N block-ranking oracles.
All score every source key: causal diagnostic, explicitly NOT an efficient method.
"""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

class Oracle:
    def __init__(self,model,block=64,local=2048,topk=32):
        self.block,self.local,self.topk=block,local,topk
        self.layers={m.layer_idx:m for m in model.modules() if m.__class__.__name__=='Qwen2Attention'}
        self.q={};self.prefix_raw={};self.raw={};self.pos=0;self.mode='prefix';self.cos={};self.hooks=[];self.calls=0
        for i,m in self.layers.items():
            def pre(mod,args,kwargs,i=i):self.cos[i]=kwargs['position_embeddings']
            def qhook(mod,args,out,i=i):self.q[i]=out.view(*out.shape[:2],-1,self.layers[i].head_dim).transpose(1,2)
            def khook(mod,args,out,i=i):
                k=out.view(*out.shape[:2],-1,self.layers[i].head_dim).transpose(1,2)
                if self.mode=='prefix':self.prefix_raw[i]=k.detach().clone()
                elif self.mode=='NoPEOracle':self.raw[i][...,self.pos:self.pos+1,:].copy_(k)
            self.hooks.extend([m.register_forward_pre_hook(pre,with_kwargs=True),m.q_proj.register_forward_hook(qhook),m.k_proj.register_forward_hook(khook)])
        self.original=ALL_ATTENTION_FUNCTIONS.get_interface('sdpa',None)
        ALL_ATTENTION_FUNCTIONS.register('sdpa',self.interface)
    def reset(self,mode,prefix_length):
        self.mode=mode;self.pos=prefix_length;self.q.clear();self.raw.clear();self.calls=0
        if mode=='NoPEOracle':
            for i,k in self.prefix_raw.items():
                buf=torch.empty(*k.shape[:2],prefix_length+1024,k.shape[-1],dtype=k.dtype,device=k.device)
                buf[...,:prefix_length,:].copy_(k);self.raw[i]=buf
    def interface(self,module,q,k,v,mask,**kw):
        if module.layer_idx not in self.layers:return self.original(module,q,k,v,mask,**kw)
        i=module.layer_idx
        if self.mode=='prefix':
            qr,kr=apply_rotary_pos_emb(self.q[i],self.prefix_raw[i],*self.cos[i])
            if not torch.equal(qr,q) or not torch.equal(kr,k):raise RuntimeError('Raw/RoPE prefix capture parity failed')
            return self.original(module,q,k,v,mask,**kw)
        if self.mode=='Dense':return self.original(module,q,k,v,mask,**kw)
        if q.shape[2]!=1 or k.shape[2]!=self.pos+1:raise RuntimeError('Only one-token causal continuation qualified')
        self.calls+=1;N=k.shape[2];H=q.shape[1];KV=k.shape[1];D=q.shape[-1]
        qs=q[0,:,0].float() if self.mode=='RoPEOracle' else self.q[i][0,:,0].float()
        ks=k[0].float() if self.mode=='RoPEOracle' else self.raw[i][0,:,:N].float()
        score=torch.einsum('kgd,knd->kgn',qs.reshape(KV,H//KV,D)*float(module.scaling),ks).reshape(H,N)
        nb=N//self.block;starts=torch.arange(nb,device=q.device)*self.block
        eligible=(starts>0)&(starts+self.block<=N-self.local)
        count=min(self.topk,int(eligible.sum()))
        if count:
            fm=score[:,:nb*self.block].reshape(H,nb,self.block).logsumexp(-1)
            chosen=fm.masked_fill(~eligible[None],-torch.inf).topk(count,-1).indices
            remote=(chosen[:,:,None]*self.block+torch.arange(self.block,device=q.device)).reshape(H,-1)
        else:remote=torch.empty(H,0,device=q.device,dtype=torch.long)
        mandatory=torch.cat([torch.arange(min(self.block,N),device=q.device),torch.arange(max(self.block,N-self.local),N,device=q.device)])
        ids=torch.cat([remote,mandatory[None].expand(H,-1)],-1).sort(-1).values
        if self.calls<=len(self.layers):
            if not bool((ids[:,1:]>ids[:,:-1]).all()) or int(ids.max())>=N:raise RuntimeError('Duplicate/future selected index')
        heads=torch.arange(H,device=q.device)//(H//KV)
        key=k[0,heads[:,None],ids][None];value=v[0,heads[:,None],ids][None]
        out=F.scaled_dot_product_attention(q,key,value,is_causal=False,dropout_p=0.,scale=float(module.scaling))
        return out.transpose(1,2).contiguous(),None
    def close(self):
        ALL_ATTENTION_FUNCTIONS.register('sdpa',self.original)
        for h in self.hooks:h.remove()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    tok=AutoTokenizer.from_pretrained(a.model);model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    if model.config.model_type!='qwen2':raise ValueError('This first adapter targets Qwen2 only')
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
    oracle=Oracle(model);eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    rows=[json.loads(x) for x in a.inputs.read_text().splitlines()]
    rows=[r for r in rows if r.get('length_budget')==16384 and r['family']<4 and r['task'] in ('current','marker_P2')]
    assert len(rows)==16
    # All-keys gather qualification on real prefix tokens: exact Dense parity.
    with torch.inference_mode():
        text=tok.apply_chat_template([{'role':'user','content':rows[0]['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        qual_ids=tok(text,return_tensors='pt')['input_ids'].cuda()[:,:129]
        oracle.mode='prefix'
        qual_base=model(input_ids=qual_ids[:,:128],use_cache=True,logits_to_keep=1).past_key_values
        qual_outputs={}
        for method in ('Dense','RoPEOracle','NoPEOracle'):
            oracle.reset(method,128)
            o=model(input_ids=qual_ids[:,128:],past_key_values=copy.deepcopy(qual_base),position_ids=torch.tensor([[128]],device='cuda'),use_cache=True,logits_to_keep=1)
            qual_outputs[method]=o.logits.detach().clone()
        parity={m:bool(torch.equal(qual_outputs[m],qual_outputs['Dense'])) for m in ('RoPEOracle','NoPEOracle')}
        if not all(parity.values()):raise RuntimeError(f'All-key gather logit parity failed: {parity}')
        del qual_outputs,qual_base,o
    print(json.dumps({'all_key_gather_bitwise_logit_parity':parity}),flush=True)
    frozen_code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__)]}
    meta={'status':'RUNNING','pid':os.getpid(),'questions':len(rows),'methods':['Dense','RoPEOracle','NoPEOracle'],
          'all_key_gather_bitwise_logit_parity':parity,'block':64,'local':2048,'remote_topk_per_query_head':32,'max_answer_tokens':48,'code_sha256':frozen_code,
          'boundary':'All original full-attention layers sparse from first question token; native RoPE reader unchanged',
          'cost_boundary':'Oracles scan all keys; NOT a runtime/efficiency method; per-query-head diagnostic budget'}
    (a.output/'status.json').write_text(json.dumps(meta,indent=2));start=time.monotonic();current_group=None;records=[]
    try:
        with torch.inference_mode(),open(a.output/'outputs.jsonl','x') as stream:
            for row in rows:
                rendered=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                encoded=tok(rendered,return_offsets_mapping=True)
                ids=torch.tensor([encoded['input_ids']],device='cuda');cut_char=rendered.rfind('\n\nQuestion:')
                if cut_char<0:raise ValueError('Question boundary absent')
                cut=next(i for i,(_,end) in enumerate(encoded['offset_mapping']) if end>cut_char)
                prefix=ids[:,:cut];tail=ids[:,cut:];group=(row['family'],row['reverse'])
                if current_group!=group:
                    oracle.mode='prefix';oracle.prefix_raw.clear();oracle.q.clear()
                    tick=time.monotonic();base=model(input_ids=prefix,use_cache=True,logits_to_keep=1).past_key_values
                    prefix_seconds=time.monotonic()-tick;base_ids=prefix.clone();current_group=group
                elif not torch.equal(prefix,base_ids):raise ValueError('Questions do not share identical token prefix')
                if tail.shape[1]+48>=1024:raise ValueError('Continuation buffer exceeds qualification')
                for method in meta['methods']:
                    oracle.reset(method,cut);cache=copy.deepcopy(base);tick=time.monotonic();last=None
                    def step(token,cache):
                        oracle.pos=cache.get_seq_length()
                        return model(input_ids=token,past_key_values=cache,position_ids=torch.tensor([[oracle.pos]],device='cuda'),
                          attention_mask=torch.ones((1,oracle.pos+1),device='cuda',dtype=torch.long),use_cache=True,logits_to_keep=1)
                    for j in range(tail.shape[1]):
                        last=step(tail[:,j:j+1],cache);cache=last.past_key_values
                    generated=[]
                    for j in range(48):
                        token=int(last.logits[0,-1].argmax());generated.append(token)
                        if token in eos:break
                        last=step(torch.tensor([[token]],device='cuda'),cache);cache=last.past_key_values
                    answer=tok.decode(generated,skip_special_tokens=True)
                    record={**row,'prompt':None,'method':method,'input_tokens':ids.shape[1],'prefix_tokens':cut,'question_ingest_tokens':tail.shape[1],
                      'prefix_sha256':hashlib.sha256(prefix.cpu().numpy().astype('<i8').tobytes()).hexdigest(),
                      'generated_ids':generated,'output_text':answer,'text_exact':answer.strip()==row['expected'],
                      'ended_eos':generated[-1] in eos,'continuation_seconds':time.monotonic()-tick,
                      'prefix_seconds':prefix_seconds,'selector_calls':oracle.calls}
                    record['full_exact_and_eos']=record['text_exact'] and record['ended_eos'];records.append(record)
                    stream.write(json.dumps(record)+'\n');stream.flush();print(json.dumps({k:record[k] for k in ('row_id','method','output_text','text_exact','ended_eos','continuation_seconds')}),flush=True)
    finally:oracle.close()
    meta.update(status='COMPLETE',seconds=time.monotonic()-start,outputs=len(records),peak_cuda_bytes=torch.cuda.max_memory_allocated())
    (a.output/'status.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
