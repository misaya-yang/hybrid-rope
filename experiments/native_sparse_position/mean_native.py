"""Block mean selector reference.
Rebuilds FP32 summaries per token; validates quality, not deployment performance.
"""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as rotate_qwen2
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb as rotate_qwen35

class Oracle:
    def __init__(self,model,block=64,local=2048,topk=32):
        self.block,self.local,self.topk=block,local,topk
        self.layers={m.layer_idx:m for m in model.modules() if m.__class__.__name__ in ('Qwen2Attention','Qwen3_5Attention')}
        self.rotate=rotate_qwen35 if model.config.model_type=='qwen3_5' else rotate_qwen2
        self.q={};self.prefix_raw={};self.raw={};self.pos=0;self.mode='prefix';self.cos={};self.hooks=[];self.calls=0
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
            return self.original(module,q,k,v,mask,**kw)
        if self.mode=='Dense':return self.original(module,q,k,v,mask,**kw)
        if q.shape[2]!=1 or k.shape[2]!=self.pos+1:raise RuntimeError('Only one-token causal continuation qualified')
        self.calls+=1;N=k.shape[2];H=q.shape[1];KV=k.shape[1];D=q.shape[-1]
        qs=q[0,:,0].float() if self.mode in ('RoPEOracle','RoPEMean') else self.q[i][0,:,0].float()
        ks=k[0].float() if self.mode in ('RoPEOracle','RoPEMean') else self.raw[i][0,:,:N].float()
        nb=N//self.block;starts=torch.arange(nb,device=q.device)*self.block
        eligible=(starts>0)&(starts+self.block<=N-self.local)
        count=min(self.topk,int(eligible.sum()))
        if count:
            if not self.mode.endswith('Mean'):raise ValueError('Mean-only reference')
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
