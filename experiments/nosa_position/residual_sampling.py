"""E06 fresh current-query CIS sampling of the PC2 exponential residual."""
import math
import os

import torch

from .exact_probe import ExactBlockSelector
from .projected_distribution import exact_current_blocks
from .runtime import select_with_scores
from .selector_controls import build_summary, summary_logmass, _concat_summary


class ResidualSamplingSelector(ExactBlockSelector):
    def __init__(self,mode='e06_residual_sampling',**kwargs):
        super().__init__('exact_mass',**kwargs)
        self.summaries={};self.weights={};self.generator=None
        self.seed=int(os.environ.get('PC2_SAMPLE_SEED','20260910'))
        self.metrics['algorithm_seed']=self.seed

    def _build(self,context):
        h,length,d=context.k.shape;size=context.settings.block_size;full=length//size
        if int(context.query_positions[0])==0:
            self.summaries.pop(context.layer_idx,None);self.weights.pop(context.layer_idx,None)
        old=self.summaries.get(context.layer_idx);done=old.blocks if old else 0
        if full>done:
            x=context.k[:,done*size:full*size].reshape(h,full-done,size,d)
            bias=context.cis[:,done*size:full*size].float().reshape(h,full-done,size)
            old=_concat_summary(old,build_summary(x,bias,'pc2'))
            self.summaries[context.layer_idx]=old
            w=bias.softmax(-1)
            self.weights[context.layer_idx]=torch.cat((self.weights[context.layer_idx],w),1) if done else w
            self.metrics['e06_build_raw_k_elements']=self.metrics.get('e06_build_raw_k_elements',0)+x.numel()
        self.metrics['max_metadata_bytes']=sum(s.nbytes() for s in self.summaries.values())+sum(w.nbytes for w in self.weights.values())
        return old,full

    @torch.no_grad()
    def logmass(self,context):
        h,length,d=context.k.shape;size=context.settings.block_size
        q=context.q.float().reshape(h,-1,context.q.shape[1],d)/math.sqrt(d)
        summary,full=self._build(context)
        out=torch.full((*q.shape[:3],math.ceil(length/size)),-torch.inf,device=q.device)
        if self.generator is None:
            self.generator=torch.Generator(device=q.device).manual_seed(self.seed)
        if full:
            weights=self.weights[context.layer_idx]
            blocks=torch.arange(full,device=q.device)
            head=torch.arange(h,device=q.device)[:,None,None,None]
            for start in range(0,q.shape[2],16):
                stop=min(start+16,q.shape[2]);count=stop-start;a=q[:,:,start:stop]
                # One independent sample set per live query/block, shared by all
                # query heads in a KV group because their CIS weights are equal.
                draws=torch.multinomial(weights.reshape(h*full,size),count*8,replacement=True,generator=self.generator)
                draws=draws.reshape(h,full,count,8).permute(0,2,1,3)
                positions=blocks[None,None,:,None]*size+draws
                raw=context.k[head,positions].float()
                centered=raw-summary.mean[:,None,:,None]
                u,v=a.chunk(2,-1);ku,kv=centered.chunk(2,-1)
                pairs=u[:,:,:,None,None]*ku[:,None]+v[:,:,:,None,None]*kv[:,None]
                x=pairs.sum(-1)
                polynomial=1+x+.5*pairs.square().sum(-1)
                linear=torch.einsum('hgqd,hbd->hgqb',a,summary.mean)
                second=summary_logmass(a,summary,'pc2')-summary.log_weight[:,None,None]-linear
                estimate=1+second.double()+(x.double().exp()-polynomial.double()).mean(-1)
                bad=(estimate<=0)|~torch.isfinite(estimate)
                mass=summary.log_weight[:,None,None]+linear+estimate.clamp_min(1e-300).log()
                out[:,:,start:stop,:full]=mass.float()
                # Exact fallback touches only affected query/block pairs, shared
                # over the GQA heads, not every raw block on that query.
                for hh in range(h):
                    qi,bi=torch.where(bad[hh].any(0))
                    if qi.numel()==0:continue
                    key_positions=bi[:,None]*size+torch.arange(size,device=q.device)
                    keys=context.k[hh,key_positions].float()
                    bias=context.cis[hh,key_positions].float()
                    logits=torch.einsum('gnd,ntd->gnt',a[hh,:,qi],keys)+bias[None]
                    exact=logits.logsumexp(-1)
                    out[hh,:,start:stop,:full].permute(1,2,0)[qi,bi]=exact.T
                    self.metrics['e06_exact_fallback_pairs']=self.metrics.get('e06_exact_fallback_pairs',0)+qi.numel()
                    self.metrics['e06_fallback_raw_k_elements']=self.metrics.get('e06_fallback_raw_k_elements',0)+keys.numel()
                self.metrics['e06_sampled_raw_k_elements']=self.metrics.get('e06_sampled_raw_k_elements',0)+raw.numel()
                # Logical unique token fetches; repeated draws still incur gather
                # operations, whose full element count is reported above.
                ordered=draws.sort(-1).values
                unique=1+(ordered[...,1:]!=ordered[...,:-1]).sum(-1)
                self.metrics['e06_unique_sampled_tokens']=self.metrics.get('e06_unique_sampled_tokens',0)+int(unique.sum())
                self.metrics['e06_query_block_pairs']=self.metrics.get('e06_query_block_pairs',0)+h*count*full
        return exact_current_blocks(context,out,self.metrics)

    @torch.no_grad()
    def __call__(self,context):
        self.metrics['calls']+=1
        if math.ceil(context.k.shape[1]/context.settings.block_size)<=context.settings.topk:
            return select_with_scores(context,context.q.new_empty(0))
        return select_with_scores(context,self.logmass(context).softmax(-1).sum(1))
