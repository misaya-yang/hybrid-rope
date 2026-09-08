"""Decompose score variation on saved activations; no new model forwards.

For each block, phase-only keys rotate that block's mean pre-RoPE key. The
residual includes actual content deviations and BF16 rotation rounding. Their
sum is the saved native key exactly, so the score-variance identity is exact.
"""
import argparse,json
from pathlib import Path
import numpy as np
from groups import phase_groups,controls

def rotate(k,positions,omega):
    width=2*len(omega);h=len(omega)
    angles=positions[...,None]*omega
    c=np.concatenate([np.cos(angles),np.cos(angles)],-1)
    s=np.concatenate([np.sin(angles),np.sin(angles)],-1)
    rotary=k[...,:width]
    return np.concatenate([rotary*c+np.concatenate([-rotary[...,h:],rotary[...,:h]],-1)*s,k[...,width:]],-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    records=[]
    for path in sorted(a.run.glob('*.npz')):
        with np.load(path) as z:
            q=z['queries_scaled'].astype(np.float64);k=z['keys_rotated'].astype(np.float64);u=z['keys_nope'].astype(np.float64)
            omega=z['omega'].astype(np.float64);eligible=z['eligible'];B=k.shape[1]
            positions=z['block_starts'][:,None]+np.arange(B)[None,:]
            same=np.broadcast_to(u.mean(1,keepdims=True),u.shape)
            phase=rotate(same,positions,omega)
            s=np.einsum('qd,btd->qbt',q,k,optimize=False)
            sp=np.einsum('qd,btd->qbt',q,phase,optimize=False);sc=s-sp
            reconstruction=rotate(u,positions,omega)
            round_rel=float(np.linalg.norm(reconstruction-k)/np.linalg.norm(k))
            for name,labels in {'Mean':np.zeros(B,dtype=int),**controls(phase_groups(omega,B,4))}.items():
                totals={x:np.zeros(s.shape[:2]) for x in ('native','phase','content_residual','twice_covariance')}
                for g in np.unique(labels):
                    ix=np.flatnonzero(labels==g);weight=len(ix)/B
                    ap=sp[:,:,ix]-sp[:,:,ix].mean(-1,keepdims=True)
                    ac=sc[:,:,ix]-sc[:,:,ix].mean(-1,keepdims=True)
                    aa=s[:,:,ix]-s[:,:,ix].mean(-1,keepdims=True)
                    for key,value in [('native',aa*aa),('phase',ap*ap),('content_residual',ac*ac),('twice_covariance',2*ap*ac)]:
                        totals[key]+=weight*value.mean(-1)
                residual=totals['native']-totals['phase']-totals['content_residual']-totals['twice_covariance']
                if abs(residual).max()>1e-8:raise ValueError('Variance identity failed')
                records.append({'file':path.name,'method':name,'eligible_query_blocks':int(eligible.sum()),
                    'bf16_rotation_reconstruction_relative_error':round_rel,'identity_max_residual':float(abs(residual).max()),
                    **{key:float(value[eligible].mean()) for key,value in totals.items()}})
    if not records:raise ValueError('Missing saved activations')
    means={m:{key:float(np.mean([r[key] for r in records if r['method']==m])) for key in ('native','phase','content_residual','twice_covariance')} for m in sorted({r['method'] for r in records})}
    result={'evidence':'Uniform eligible query/block score-variance decomposition on prior dense activations; not a task or selector ranking guarantee',
      'records':records,'equal_file_means':means}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(means,indent=2))
if __name__=='__main__':main()
