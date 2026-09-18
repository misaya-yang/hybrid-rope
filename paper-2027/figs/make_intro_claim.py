"""Introduction concept figure: three public RoPE allocations and distance stretch.

Exact mathematical profiles on the same Llama-3-8B native grid, evaluated in
float64. YaRN uses its official frequency blend and floor/ceil boundaries.
These are analytic frequency tables, not model scores or FP32 execution receipts.
This figure illustrates allocation and its phase-equivalent distance response.
"""
from pathlib import Path
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
K,base,L,s=64,500000.,8192,4.
k=np.arange(K);native=1/np.power(base,k/K)
turns=L*native/(2*np.pi)
l,h=int(np.flatnonzero(turns>32)[-1]),int(np.flatnonzero(turns<1)[0]);n=h-l
q=np.clip(k-l,0,n)
p=q*(q+1)/(n*(n+1))
t=q*(3*n*n+3*n+1-q*q)/(n*(n+1)*(2*n+1))
yarn_l=max(math.floor(K*math.log(L/(32*2*math.pi))/math.log(base)),0)
yarn_h=min(math.ceil(K*math.log(L/(2*math.pi))/math.log(base)),2*K-1)
ramp=np.clip((k-yarn_l)/(yarn_h-yarn_l),0,1)
tables={'YaRN':native*(1-ramp+ramp/s),'MrRoPE-Pro':native*np.power(s,-p),'TailSpline':native*np.power(s,-t)}
assert (l,h)==(18,35) and (yarn_l,yarn_h)==(18,35)
assert all(np.all(np.diff(v)<0) for v in tables.values())
for v in tables.values():
    assert np.array_equal(v[:l+1],native[:l+1])
    assert np.array_equal(v[h:],native[h:]/s)
stretch={name:native/v for name,v in tables.items()}
assert np.all(stretch['TailSpline']>=stretch['MrRoPE-Pro']-1e-12)
styles={'YaRN':('#009E73','--'),'MrRoPE-Pro':('#0072B2','-.'),'TailSpline':('#D55E00','-')}
INK,GRAY='#20262B','#B7BEC4'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'mathtext.fontset':'stix','font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7})
fig=plt.figure(figsize=(7.2,2.2),facecolor='white')
fig.text(.025,.95,'(a) Same range, different allocation',fontsize=10.5,fontweight='bold',color=INK)
fig.text(.555,.95,'(b) Different distance responses',fontsize=10.5,fontweight='bold',color=INK)
ax=fig.add_axes([.035,.20,.405,.66])
for y,(name,values) in zip([2,1,0],tables.items()):
    x=-np.log10(values);color=styles[name][0]
    ax.hlines(y,0,x[-1],color='#D1D6DA',lw=.8)
    ax.scatter(x,np.full(K,y),s=8,color=GRAY,zorder=2)
    ax.scatter(x[l+1:h],np.full(n-1,y),s=17,color=color,zorder=3)
    ax.scatter(x[[0,-1]],[y,y],s=28,edgecolor=INK,facecolor='white',zorder=4)
    ax.text(0,y+.25,name,fontsize=10,color=color,fontweight='bold')
for x in [0,-np.log10(native[-1]/s)]:ax.vlines(x,-.12,2.15,color='#8D969E',ls=':',lw=.8)
ax.set(ylim=(-.25,2.75),xlim=(-.12,-np.log10(native[-1]/s)+.12),yticks=[],xticks=[0,2,4,6],
       xticklabels=[r'$10^0$',r'$10^{-2}$',r'$10^{-4}$',r'$10^{-6}$'])
ax.set_xlabel('Rotary frequency (fast to slow; log scale)',fontsize=9.5,labelpad=5)
for side in ['left','top','right']:ax.spines[side].set_visible(False)
ax.tick_params(axis='x',labelsize=10,length=3)
ax=fig.add_axes([.615,.20,.36,.66])
for name in tables:
    color,style=styles[name]
    ax.plot(k[l:h+1],stretch[name][l:h+1],color=color,ls=style,lw=1.8,label=name)
ax.scatter([l,h],[1,s],s=24,facecolor='white',edgecolor=INK,zorder=5)
ax.axhline(1,color='#B9C1C8',lw=.65,ls=':',zorder=0);ax.axhline(s,color='#B9C1C8',lw=.65,ls=':',zorder=0)
ax.set(xlim=(l-.5,h+.5),ylim=(.85,s+.2),xticks=[l,(l+h)/2,h],xticklabels=['Fast end','Middle','Slow end'],
       yticks=[1,2,3,4],yticklabels=['1x','2x','3x','4x'])
ax.set_xlabel('Channels in the transition band',fontsize=9.5,labelpad=5)
ax.set_ylabel(r'Distance stretch $\omega_k^N/\omega_k^{\prime}$',fontsize=10,labelpad=5)
ax.legend(frameon=False,loc='upper left',fontsize=8.5,handlelength=2)
ax.tick_params(axis='both',labelsize=9,length=3)
for side in ['top','right']:ax.spines[side].set_visible(False)
for ext in ['pdf','png','svg']:
    fig.savefig(HERE/f'fig_intro_claim.{ext}',dpi=240,bbox_inches='tight',pad_inches=.04)
plt.close(fig)
receipt={'status':'manuscript_concept_figure','scope':'Analytic frequency allocation and phase-equivalent distance; no task scores, no model evaluation, no claim of FP32 receipt identity.',
 'model_geometry':{'model':'Llama-3-8B','base':base,'pairs':K,'native_length':L,'scale':s},'mrpro_tailspline_band':[l,h],'official_yarn_band':[yarn_l,yarn_h],
 'yarn_source':'scripts/lib/rope/official_yarn.py: official 32/1 floor-ceil correction and frequency blend',
 'same_endpoints':True,'same_outer_bands':True,'common_gain':1+.1*math.log(s),
 'frequency_tables':{name:v.tolist() for name,v in tables.items()},'native_frequency_table':native.tolist(),
 'distance_stretch':{name:v.tolist() for name,v in stretch.items()},'model_execution':False}
(HERE/'intro_claim_inputs.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({key:receipt[key] for key in ['status','model_geometry','mrpro_tailspline_band','official_yarn_band','same_endpoints','same_outer_bands','common_gain','model_execution']},indent=2))
