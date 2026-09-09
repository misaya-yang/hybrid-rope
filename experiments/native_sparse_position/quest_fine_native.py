"""Quest with actual 32-token retrieval pages at the same 1024 remote-token budget.

This frontier baseline changes retrieval granularity; it is distinct from the
fixed-64-token-page summary attribution. Sink64 and local2048 stay unchanged.
"""
import torch
import torch.nn.functional as F
from mean_native import Oracle as MeanOracle
from pair_envelope import QuestCache,build_quest

def quest_score(query,cache):
    kv,nb,dim=cache.minimum.shape
    q=query.float().reshape(kv,-1,1,dim)
    # Literal Quest min/max score, with FP32 products and accumulation.
    return torch.maximum(q*cache.minimum[:,None],q*cache.maximum[:,None]).sum(-1).reshape(-1,nb)

class Oracle(MeanOracle):
    def __init__(self,model,block=64,local=2048,topk=16):
        self.prefix_post={};self.caches={};self.page=32
        super().__init__(model,block,local,topk)
    def interface(self,module,q,k,v,mask,**kw):
        if module.layer_idx not in self.layers or self.mode=='Dense':return super().interface(module,q,k,v,mask,**kw)
        i=module.layer_idx
        if self.mode=='prefix':
            self.prefix_post[i]=k.detach()
            return super().interface(module,q,k,v,mask,**kw)
        if self.mode!='QuestFine32':raise ValueError('Unexpected method')
        if q.shape[2]!=1 or k.shape[2]!=self.pos+1:raise ValueError('One-token causal continuation only')
        self.calls+=1;N=k.shape[2];H=q.shape[1];KV=k.shape[1];D=q.shape[-1];B=self.page
        if N>self.local+self.block+B:
            cache=self.caches[i];nb=cache.minimum.shape[1]
            if (N-self.local)//B>nb:raise ValueError('New remote keys outside frozen prefix index')
            starts=torch.arange(nb,device=q.device)*B
            eligible=(starts>=self.block)&(starts+B<=N-self.local)
            count=min(self.topk*self.block//B,int(eligible.sum()))
            score=quest_score(q[0,:,0].float()*float(module.scaling),cache)
            chosen=score.masked_fill(~eligible[None],-torch.inf).topk(count,-1).indices
            remote=(chosen[...,None]*B+torch.arange(B,device=q.device)).reshape(H,-1)
        else:remote=torch.empty(H,0,dtype=torch.long,device=q.device)
        mandatory=torch.cat((torch.arange(min(self.block,N),device=q.device),torch.arange(max(self.block,N-self.local),N,device=q.device)))
        ids=torch.cat((remote,mandatory[None].expand(H,-1)),-1).sort(-1).values
        if self.calls<=len(self.layers):
            if not bool((ids[:,1:]>ids[:,:-1]).all()) or int(ids.max())>=N:raise ValueError('Duplicate/future KV index')
        heads=torch.arange(H,device=q.device)//(H//KV)
        key=k[0,heads[:,None],ids][None];value=v[0,heads[:,None],ids][None]
        result=F.scaled_dot_product_attention(q,key,value,is_causal=False,dropout_p=0.,scale=float(module.scaling))
        return result.transpose(1,2).contiguous(),None

def build(oracle):
    oracle.caches={};oracle.build_info={}
    for i,k in oracle.prefix_post.items():
        KV,N,D=k.shape[1:];B=oracle.page
        reference=build_quest(k[0,:,:N//B*B].reshape(KV,N//B,B,D))
        lo,hi=reference.minimum.bfloat16(),reference.maximum.bfloat16()
        if not torch.equal(lo.float(),reference.minimum) or not torch.equal(hi.float(),reference.maximum):
            raise ValueError('Quest endpoints are not losslessly BF16-representable')
        cache=QuestCache(lo,hi,B,reference.source_key_bytes)
        query=oracle.q[i][0,:,-1].float()*float(oracle.layers[i].scaling)
        if not torch.equal(quest_score(query,cache),quest_score(query,reference)):
            raise ValueError('BF16 endpoint storage changed FP32 Quest scores')
        oracle.caches[i]=cache
        oracle.build_info[i]={**cache.byte_info(),'cache_dtype':'bfloat16',
            'lossless_bf16_endpoint_roundtrip':True,'fp32_score_bitwise_parity':True,
            'parity_query':'last pre-RoPE prefix query; storage/arithmetic equivalence only'}
