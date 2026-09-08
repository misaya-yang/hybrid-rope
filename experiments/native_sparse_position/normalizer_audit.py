"""Head-weight distortion from coarse LSE on saved native Q/K, no model forward.

This mathematical probe uses fully observed windows and exact FP64 softmax. It
does not claim bitwise equivalence to any fused InfLLM kernel or final task effect.
"""
import argparse,json
from pathlib import Path
import numpy as np

def lse(s):
    mx=s.max(-1,keepdims=True)
    return (mx+np.log(np.exp(s-mx).sum(-1,keepdims=True))).squeeze(-1)

def analyze(q,k,positions):
    # Last eight query tokens for each query head were saved in head-major order.
    G=len(q)//8;q=q.reshape(G,8,-1);positions=positions.reshape(G,8)
    assert np.all(positions==positions[:1])
    k=k.reshape(-1,k.shape[-1]);N=len(k)
    means={};scores={};valid={}
    for name,width,stride in [('fine',32,16),('coarse',128,64)]:
        starts=np.arange(0,N-width+1,stride)
        means[name]=np.stack([k[i:i+width].mean(0) for i in starts])
        sc=np.einsum('gqd,cd->gqc',q,means[name],optimize=False)
        valid[name]=starts[None,:]+width-1<=positions[0,:,None]
        scores[name]=np.where(valid[name][None],sc,-np.inf)
    lf,lc=lse(scores['fine']),lse(scores['coarse'])
    delta=lf-lc;weights=np.exp(delta-lse(delta.T)[None,:])
    pf=np.exp(scores['fine']-lf[:,:,None]);exact=pf.mean(0)
    approximate=(weights[:,:,None]*pf).sum(0)
    C=exact.shape[-1];topk=min(32,C)
    exact_idx=np.argpartition(exact,-topk,axis=-1)[:,-topk:]
    approximate_idx=np.argpartition(approximate,-topk,axis=-1)[:,-topk:]
    overlap=np.mean([len(set(x)&set(y))/topk for x,y in zip(exact_idx,approximate_idx)])
    return {'query_heads':G,'mean_effective_heads':float(np.mean(1/np.sum(weights**2,axis=0))),
      'mean_max_head_weight':float(weights.max(0).mean()),'uniform_head_weight':1/G,
      'mean_centered_log_weight_std':float(delta.std(0).mean()),
      'mean_shared_fine_distribution_tv':float(np.abs(exact-approximate).sum(-1).mean()/2),
      'mean_top32_fine_window_overlap':float(overlap),'head_weights':weights.tolist(),
      'mean_fine_minus_coarse_logZ':float(delta.mean())}

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    records=[]
    for path in sorted(a.run.glob('*.npz')):
        with np.load(path) as z:
            assert np.array_equal(z['block_starts'],np.arange(len(z['block_starts']))*64)
            for mode,qkey,kkey in [('RoPE','queries_scaled','keys_rotated'),('NoPE','queries_nope_scaled','keys_nope')]:
                records.append({'file':path.name,'mode':mode,**analyze(z[qkey].astype(np.float64),z[kkey].astype(np.float64),z['query_positions'])})
    metrics=('query_heads','mean_effective_heads','mean_max_head_weight','uniform_head_weight','mean_centered_log_weight_std','mean_shared_fine_distribution_tv','mean_top32_fine_window_overlap')
    means={m:{k:float(np.mean([r[k] for r in records if r['mode']==m])) for k in metrics} for m in ('RoPE','NoPE')}
    result={'evidence':'Frozen dense-trajectory mathematical diagnostic, fully causal fine32/16 and coarse128/64 means; no native fused-kernel or answer claim','records':records,'equal_file_means':means}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(means,indent=2))
if __name__=='__main__':main()
