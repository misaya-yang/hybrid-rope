"""Finite distributions, slot-preserving projection, and full rotary replay."""
import math
import numpy as np
import torch


def quantile_moments(a, b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if np.any(a<0) or np.any(b<0) or min(a.sum(),b.sum())<=0:
        raise ValueError('nonpositive profile mass')
    ca=np.cumsum(a/a.sum());cb=np.cumsum(b/b.sum());ca[-1]=cb[-1]=1.
    edges=np.unique(np.r_[0.,ca,cb]);mass=np.diff(edges);u=(edges[:-1]+edges[1:])/2
    qa=np.searchsorted(ca,u);qb=np.searchsorted(cb,u)
    return float(mass@(qa*qa)),float(mass@(qa*qb)),float(mass@(qb*qb))


def estimate_beta(a,b,ratio=2.):
    x,c,y=quantile_moments(a,b)
    if min(x,y)<=0:raise ValueError('only zero-distance response')
    raw=c/y;r=np.clip(raw,1/ratio,1.)
    return dict(beta=float(-np.log(r)/np.log(ratio)),raw_r=raw,
                residual=float(max(0.,(r*r*y-2*r*c+x)/x)),second_moment=y)


def bounded_isotonic(values,weights,lower,upper):
    """Decreasing weighted PAVA with per-slot bounds; never reorders slots."""
    if len({len(x) for x in [values,weights,lower,upper]})!=1 or not all(np.isfinite(x).all() for x in [values,weights,lower,upper]):
        raise ValueError("nonfinite or inconsistent projection inputs")
    blocks=[]
    for i,(v,w,lo,hi) in enumerate(zip(values,weights,lower,upper)):
        if not w>0 or lo>hi:raise ValueError('invalid isotonic input')
        block=[i,i+1,w,w*v,lo,hi,float(np.clip(v,lo,hi))];blocks.append(block)
        while len(blocks)>1 and blocks[-2][6]<blocks[-1][6]:
            b=blocks.pop();a=blocks.pop();lo=max(a[4],b[4]);hi=min(a[5],b[5])
            if lo>hi:raise ValueError('infeasible pooled bounds')
            w=a[2]+b[2];total=a[3]+b[3]
            blocks.append([a[0],b[1],w,total,lo,hi,float(np.clip(total/w,lo,hi))])
    out=np.zeros(len(values))
    for a,b,_,_,_,_,v in blocks:out[a:b]=v
    if np.any(np.diff(out)>1e-12) or np.any(out<np.array(lower)-1e-12) or np.any(out>np.array(upper)+1e-12):
        raise ValueError('projection contract')
    return out


def replay(q,k,v,wo,frequencies,gain,positions,*,response=False):
    """q=[H,Q,D], k/v=[KV,L,D], wo=[hidden,H*D]; exact split-half layout.

    Full visible-key normalization. Only Q selected queries materialized, never LxL.
    Returns projected attention output and optionally independent-phase response.
    """
    H,Q,D=q.shape;L=k.shape[1];K=D//2
    if H%k.shape[0] or D%2:raise ValueError('head layout')
    q=q.float();k=k.float();v=v.float();wo=wo.float()
    delta=positions[:,None]-torch.arange(L,device=q.device)[None,:]
    phase=delta[:,:,None]*frequencies.float()[None,None,:]
    co,si=phase.cos(),phase.sin();mask=delta<0;factor=gain*gain/math.sqrt(D)
    output=torch.zeros(Q,wo.shape[0],device=q.device)
    profile=torch.zeros(K,L,device=q.device,dtype=torch.float64) if response else None
    for h in range(H):
        kh=k[h//(H//k.shape[0])];vh=v[h//(H//v.shape[0])];qh=q[h];wh=wo[:,h*D:(h+1)*D]
        C=qh[:,:K,None].transpose(1,2)*kh[None,:,:K]+qh[:,K:,None].transpose(1,2)*kh[None,:,K:]
        S=qh[:,:K,None].transpose(1,2)*kh[None,:,K:]-qh[:,K:,None].transpose(1,2)*kh[None,:,:K]
        logits=factor*(C*co+S*si).sum(-1);logits.masked_fill_(mask,-torch.inf)
        p=logits.softmax(-1);o=p@vh;output+=o@wh.T
        if response:
            derivative=factor*(-C*si+S*co)
            gram=wh.T@wh
            energy=((vh@gram)*vh).sum(-1)[None,:]+((o@gram)*o).sum(-1)[:,None]-2*(o@gram)@vh.T
            chi=derivative.square()*(p.square()*energy.clamp_min(0))[:,:,None]
            for t in range(Q):profile.index_add_(1,delta[t].clamp_min(0),chi[t].T.double())
    if response:profile[:,0]=0
    return output,profile


def background(q,k,frequencies,gain,distances):
    """Cross-document keys, full table logits, fixed sampled target distances."""
    H,Q,D=q.shape;K=D//2;out=[];co=(distances[:,:,None]*frequencies).cos();si=(distances[:,:,None]*frequencies).sin()
    for h in range(H):
        qh=q[h].float();kh=k[h//(H//k.shape[0])].float()
        C=qh[:,None,:K]*kh[None,:,:K]+qh[:,None,K:]*kh[None,:,K:]
        S=qh[:,None,:K]*kh[None,:,K:]-qh[:,None,K:]*kh[None,:,:K]
        z=(C*co+S*si).sum(-1)*(gain*gain/math.sqrt(D))
        out.append(torch.logsumexp(z,dim=-1)-math.log(kh.shape[0]))
    return torch.stack(out).mean()
