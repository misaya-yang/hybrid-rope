"""Block mean selector reference.
Rebuilds FP32 summaries per token; validates quality, not deployment performance.
"""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from pair_envelope import build_quest,score_quest
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as rotate_qwen2
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb as rotate_qwen35

class Oracle:
    def __init__(self,model,block=64,local=2048,topk=32):
        self.block,self.local,self.topk=block,local,topk
        self.layers={m.layer_idx:m for m in model.modules() if m.__class__.__name__ in ('Qwen2Attention','Qwen3_5Attention')}
        self.rotate=rotate_qwen35 if model.config.model_type=='qwen3_5' else rotate_qwen2
        self.prefix_post={};self.summary={};self.ranges={};self.q={};self.prefix_raw={};self.raw={};self.pos=0;self.mode='prefix';self.cos={};self.hooks=[];self.calls=0
        for i,m in self.layers.items():
            def pre(mod,args,kwargs,i=i):self.cos[i]=kwargs['position_embeddings']
            def qhook(mod,args,out,i=i):self.q[i]=out.view(*out.shape[:2],-1,self.layers[i].head_dim).transpose(1,2)
            def khook(mod,args,out,i=i):
                k=out.view(*out.shape[:2],-1,self.layers[i].head_dim).transpose(1,2)
                if self.mode=='prefix':self.prefix_raw[i]=k.detach().clone()
                elif self.mode in ('NoPEOracle','NoPEMean'):self.raw[i][...,self.pos:self.pos+1,:].copy_(k)
            self.hooks.extend([m.register_forward_pre_hook(pre,with_kwargs=True),(m.q_norm if hasattr(m,'q_norm') else m.q_proj).register_forward_hook(qhook),(m.k_norm if hasattr(m,'k_norm') else m.k_proj).register_forward_hook(khook)])
        self.original=ALL_ATTENTION_FUNCTIONS.get_interface('sdpa',None)
        ALL_ATTENTION_FUNCTIONS.register('sdpa',self.interface)
    def reset(self,mode,prefix_length):
        self.mode=mode;self.pos=prefix_length;self.q.clear();self.raw.clear();self.calls=0
        if mode in ('NoPEOracle','NoPEMean'):
            for i,k in self.prefix_raw.items():
                buf=torch.empty(*k.shape[:2],prefix_length+1024,k.shape[-1],dtype=k.dtype,device=k.device)
                buf[...,:prefix_length,:].copy_(k);self.raw[i]=buf
    def interface(self,module,q,k,v,mask,**kw):
        if module.layer_idx not in self.layers:return self.original(module,q,k,v,mask,**kw)
        i=module.layer_idx
        if self.mode=='prefix':
            qr,kr=self.rotate(self.q[i],self.prefix_raw[i],*self.cos[i])
            if not torch.equal(qr,q) or not torch.equal(kr,k):raise RuntimeError('Raw/RoPE prefix capture parity failed')
            self.prefix_post[i]=k.detach()
            return self.original(module,q,k,v,mask,**kw)
        if self.mode=='Dense':return self.original(module,q,k,v,mask,**kw)
        if q.shape[2]!=1 or k.shape[2]!=self.pos+1:raise RuntimeError('Only one-token causal continuation qualified')
        self.calls+=1;N=k.shape[2];H=q.shape[1];KV=k.shape[1];D=q.shape[-1]
        qs=self.q[i][0,:,0].float() if self.mode=='NoPEMean' else q[0,:,0].float()
        nb=self.summary[self.mode][i][0].shape[1] if self.mode in self.summary else self.ranges[self.mode][i].minimum.shape[1]//(2 if self.mode=='QuestSplit32' else 1) if self.mode in self.ranges else N//self.block
        starts=torch.arange(nb,device=q.device)*self.block
        eligible=(starts>0)&(starts+self.block<=N-self.local)
        count=min(self.topk,int(eligible.sum()))
        if count:
            if self.mode in self.ranges:
                fm=score_quest(qs*float(module.scaling),self.ranges[self.mode][i])[0]
                if self.mode=='QuestSplit32':fm=fm.reshape(H,nb,2).amax(-1)
            elif self.mode in self.summary:
                means,lc=self.summary[self.mode][i]
                dots=torch.einsum('kgd,kbrd->kgbr',qs.reshape(KV,H//KV,D)*float(module.scaling),means.float())
                fm=(dots+lc[:,None]).logsumexp(-1).reshape(H,nb)
            else:
                ks=self.raw[i][0,:,:N].float() if self.mode=='NoPEMean' else k[0].float()
                means=ks[:,:nb*self.block].reshape(KV,nb,self.block,D).mean(2)
                fm=torch.einsum('kgd,kbd->kgb',qs.reshape(KV,H//KV,D)*float(module.scaling),means).reshape(H,nb)
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


def labels_farthest(keys,R=4):
    # [batch_blocks,B,D], deterministic tie order, repeated centers make empty padding.
    n,B,D=keys.shape;center=(keys-keys.mean(1,keepdim=True)).square().sum(-1).argmax(-1)
    rows=torch.arange(n,device=keys.device);distances=[];nearest=None
    for r in range(R):
        d=(keys-keys[rows,center][:,None]).square().sum(-1);distances.append(d)
        nearest=d if nearest is None else torch.minimum(nearest,d)
        center=nearest.argmax(-1)
    return torch.stack(distances,-1).argmin(-1)

def summarize(keys,labels,R=4):
    n,B,D=keys.shape;means=[];counts=[]
    for r in range(R):
        mask=labels==r;count=mask.sum(-1);counts.append(count)
        means.append((keys*mask[:,:,None]).sum(1)/count[:,None].clamp_min(1))
    return torch.stack(means,1),torch.stack(counts,1)

def build(oracle):
    oracle.summary={m:{} for m in ('PostMetric4','PreMetric4','MatchedContiguous4')}
    oracle.ranges={m:{} for m in ('Quest','QuestSplit32')}
    oracle.build_info={}
    for i,k in oracle.prefix_post.items():
        KV,N,D=k.shape[1:];nb=N//oracle.block;B=oracle.block
        kr=k[0,:,:nb*B].reshape(KV*nb,B,D).float()
        kn=oracle.prefix_raw[i][0,:,:nb*B].reshape(KV*nb,B,D).float()
        torch.cuda.synchronize();started=time.monotonic()
        oracle.ranges['Quest'][i]=build_quest(kr.reshape(KV,nb,B,D))
        oracle.ranges['QuestSplit32'][i]=build_quest(kr.reshape(KV,nb*2,B//2,D))
        post=labels_farthest(kr);pre=labels_farthest(kn)
        pm,counts=summarize(kr,post)
        contiguous=(torch.arange(B,device=k.device)[None,:,None]>=counts.cumsum(-1)[:,None,:-1]).sum(-1)
        for method,pair in [('PostMetric4',(pm,counts)),('PreMetric4',summarize(kr,pre)),('MatchedContiguous4',summarize(kr,contiguous))]:
            means,ct=pair
            if not bool((ct.sum(-1)==B).all()) or not bool(torch.isfinite(means).all()):raise ValueError('Invalid cache summary')
            oracle.summary[method][i]=(means.reshape(KV,nb,4,D),ct.float().log().reshape(KV,nb,4))
        torch.cuda.synchronize()
        oracle.build_info[i]={'seconds_all_methods':time.monotonic()-started,'methods':{m:{'descriptor_bytes':sum(t.numel()*t.element_size() for t in layers[i]),'dtype':'float32','representatives':4} for m,layers in oracle.summary.items()}}
        oracle.build_info[i]['methods'].update({m:layers[i].byte_info() for m,layers in oracle.ranges.items()})
