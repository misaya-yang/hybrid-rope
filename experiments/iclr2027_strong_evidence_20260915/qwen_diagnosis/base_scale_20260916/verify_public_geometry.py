import numpy as np,json,math
from pathlib import Path

def check(b,K,L,s,head=128):
 k=np.arange(K);w=b**(-k/K);turns=L*w/(2*np.pi)
 l=int(np.flatnonzero(turns>32)[-1]);h=int(np.flatnonzero(turns<1)[0]);n=h-l
 q=np.clip(k-l,0,n);u=q/n;c=np.log(b)/K
 P=q*(q+1)/(n*(n+1));T=q*(3*n*n+3*n+1-q*q)/(n*(n+1)*(2*n+1))
 vp=w*s**(-P);vt=w*s**(-T)
 ep=np.log(L*w[l]/(64*np.pi));em=np.log(2*np.pi/(L*w[h]))
 z=np.arange(n+1)/n
 expected=64*np.pi*32**(-z)*np.exp((1-z)*ep-z*em)
 residual=float(np.max(np.abs(expected/(L*w[l:h+1])-1)))
 assert residual<1e-12
 assert abs(n*c-(np.log(32)+ep+em))<1e-12
 D=int(s*L);d=np.arange(D,dtype=float);weight=D-d;weight/=weight.sum()
 def rms(va,vb,H):
  dist=np.arange(H,dtype=float);weights=H-dist;weights/=weights.sum()
  return float(np.sqrt(sum(float(np.dot(weights,4*np.sin(dist*delta/2)**2)) for delta in va-vb)*2/head))
 diff=vp-vt
 return {'base':b,'K':K,'L':L,'s':s,'head':head,'band':[l,h],'n':n,'c':float(c),'eta_plus':float(ep),'eta_minus':float(em),'turn_identity_relative_error':residual,'max_wavelength_ratio':float(np.max(vp/vt)),'max_target_phase_difference':float(np.max(D*abs(diff))),'causal_rms_head':rms(vt,vp,D),'causal_rms_fp32_head':rms(vt.astype(np.float32).astype(float),vp.astype(np.float32).astype(float),D),'rms_at_128k':rms(vt,vp,131072) if b==1e6 else None,'example_k31':{'P':float(P[31]),'T':float(T[31]),'T_wavelength_growth':float(s**T[31]),'P_wavelength_growth':float(s**P[31]),'T_target_phase':float(D*vt[31]),'P_target_phase':float(D*vp[31])} if b==1e6 else None}
res={name:check(*args) for name,args in {'llama_s4':(500000.,64,8192,4),'olmo_s4':(500000.,64,4096,4),'qwen_s4':(1e6,64,32768,4),'qwen_s8':(1e6,64,32768,8),'glm_s4':(1e4,32,32768,4)}.items()}
res['qwen_ABF_phase_ratios']={k:8*100**(-k/64) for k in [23,29,32,40]}
res['conditions']='Public geometric grids; boundary crossings present; no model loading. RMS uses integer causal weights, full head normalization, unscaled rotations.'
Path(__file__).with_name('results.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
