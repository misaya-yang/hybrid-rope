"""E05 two deterministic key-space components and matched descriptor controls."""
import math

import torch

from .exact_probe import ExactBlockSelector
from .projected_distribution import exact_current_blocks
from .runtime import select_with_scores


class TwoComponentSelector(ExactBlockSelector):
    modes=('e05_two_component','e05_contiguous','e05_rank4','e09_independent')

    def __init__(self, mode='e05_two_component', **kwargs):
        self.variant=mode
        super().__init__('exact_mass',**kwargs)
        self.descriptors={}

    def _build(self,context):
        h,length,d=context.k.shape
        size=context.settings.block_size
        full=length//size
        if int(context.query_positions[0])==0:self.descriptors.pop(context.layer_idx,None)
        old=self.descriptors.get(context.layer_idx)
        done=old['mean'].shape[1] if old else 0
        if full>done:
            x=context.k[:,done*size:full*size].float().reshape(h,full-done,size,d)
            bias=context.cis[:,done*size:full*size].float().reshape(h,full-done,size)
            if self.variant in ('e05_rank4','e09_independent'):
                w=bias.softmax(-1);mean=(w[...,None]*x).sum(-2)
                weighted=(x-mean[:,:,None])*w.sqrt()[...,None]
                gram=[REDACTED_EMAIL](-1,-2)
                _,vectors=torch.linalg.eigh(gram)
                rank=4 if self.variant=='e05_rank4' else 32
                factors=vectors[...,-rank:].transpose(-1,-2)@weighted
                fresh={'mean':mean,'logz':bias.logsumexp(-1),'factors':factors}
            else:
                if self.variant=='e05_contiguous':
                    labels=(torch.arange(size,device=x.device)>=size//2).expand(h,full-done,size)
                else:
                    first=(x-x[:,:,:1]).square().sum(-1).argmax(-1)
                    center1=x.gather(2,first[:,:,None,None].expand(-1,-1,1,d))
                    second=(x-center1).square().sum(-1).argmax(-1)
                    center2=x.gather(2,second[:,:,None,None].expand(-1,-1,1,d))
                    labels=(x-center2).square().sum(-1)<(x-center1).square().sum(-1)
                means=[];logz=[];vx=[];vy=[];xy=[]
                for component in (False,True):
                    part=bias.masked_fill(labels!=component,-torch.inf)
                    z=part.logsumexp(-1)
                    w=torch.nan_to_num(part.softmax(-1),nan=0.)
                    mean=(w[...,None]*x).sum(-2)
                    centered=x-mean[:,:,None];u,v=centered.chunk(2,-1)
                    means.append(mean);logz.append(z)
                    vx.append((w[...,None]*u.square()).sum(-2))
                    vy.append((w[...,None]*v.square()).sum(-2))
                    xy.append((w[...,None]*u*v).sum(-2))
                fresh={'mean':torch.stack(means,2),'logz':torch.stack(logz,2),
                       'vx':torch.stack(vx,2),'vy':torch.stack(vy,2),'xy':torch.stack(xy,2)}
            old={name:torch.cat((old[name],value),1) for name,value in fresh.items()} if old else fresh
            self.descriptors[context.layer_idx]=old
            self.metrics['e05_build_raw_k_elements']=self.metrics.get('e05_build_raw_k_elements',0)+x.numel()
        self.metrics['max_metadata_bytes']=sum(t.nbytes for v in self.descriptors.values() for t in v.values())
        return old,full

    @torch.no_grad()
    def logmass(self,context):
        h,length,d=context.k.shape
        q=context.q.float().reshape(h,-1,context.q.shape[1],d)/math.sqrt(d)
        cache,full=self._build(context)
        out=torch.full((*q.shape[:3],math.ceil(length/context.settings.block_size)),-torch.inf,device=q.device)
        if full:
            if self.variant in ('e05_rank4','e09_independent'):
                linear=torch.einsum('hgqd,hbd->hgqb',q,cache['mean'])+cache['logz'][:,None,None]
                projection=torch.einsum('hgqd,hbrd->hgqbr',q,cache['factors'])
                out[...,:full]=linear+.5*projection.square().sum(-1)
            else:
                u,v=q.chunk(2,-1)
                linear=torch.einsum('hgqd,hbcd->hgqbc',q,cache['mean'])+cache['logz'][:,None,None]
                variance=(torch.einsum('hgqd,hbcd->hgqbc',u.square(),cache['vx'])
                          +2*torch.einsum('hgqd,hbcd->hgqbc',u*v,cache['xy'])
                          +torch.einsum('hgqd,hbcd->hgqbc',v.square(),cache['vy']))
                out[...,:full]=(linear+.5*variance.clamp_min(0)).logsumexp(-1)
        return exact_current_blocks(context,out,self.metrics)

    @torch.no_grad()
    def __call__(self,context):
        self.metrics['calls']+=1
        if math.ceil(context.k.shape[1]/context.settings.block_size)<=context.settings.topk:
            return select_with_scores(context,context.q.new_empty(0))
        return select_with_scores(context,self.logmass(context).softmax(-1).sum(1))
