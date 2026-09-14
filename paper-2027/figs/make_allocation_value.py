"""Portable main-paper figures from recorded observations and analytic tables."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from make_story_figures import white
from make_exponent_revision_figures import axis_style, finish, BLUE, ORANGE, INK, GRID
HERE=Path(__file__).resolve().parent
D=json.loads((HERE/'allocation_value_inputs.json').read_text())['data']

def verify():
    n=17;q=np.arange(1,n+1,dtype=float)
    eps=3*(n+q)*(n-q+1)/(n*(n+1)*(2*n+1))
    diff=np.eye(n-1,n,k=1)-np.eye(n-1,n)
    h=diff.T@diff;h[-1,-1]+=1
    v=np.linalg.solve(h,np.ones(n));v/=v.sum()
    assert np.allclose(eps,v,atol=1e-14) and np.all(eps>0)
    m=q*(3*n*n+3*n+1-q*q)/(n*(n+1)*(2*n+1))
    assert np.allclose(np.cumsum(eps),m)
    h[0,0]+=1;v=np.linalg.solve(h,np.ones(n));v/=v.sum()
    assert np.allclose(v,6*q*(n-q+1)/(n*(n+1)*(n+2)))
    for width in [1,2,17,18,64]:
        k=np.arange(1,width+1,dtype=float)
        ee=3*(width+k)*(width-k+1)/(width*(width+1)*(2*width+1))
        dd=np.eye(width-1,width,k=1)-np.eye(width-1,width)
        hh=dd.T@dd;hh[-1,-1]+=1
        vv=np.linalg.solve(hh,np.ones(width));vv/=vv.sum()
        assert np.allclose(ee,vv,atol=1e-13) and np.all(ee>0)
    t=D['tailspline'];w=np.array([.25,.5,.25])
    for key in ['full13','ppl']:
        delta=w@(np.array(t[key]['TailSpline'])-t[key]['MrPro'])
        assert abs(delta-t['auc_delta'][key])<1e-6
    assert abs(w@t['niah_delta']-.0375)<1e-10

    # Exact stiffness versus independent Gauss-Legendre integration.
    nodes,weights=np.polynomial.legendre.leggauss(100)
    phi=(nodes+1)/2
    for tau in [np.sqrt(2),2.,4.]:
        rho=tau*np.cosh(tau*(1-phi))/np.sinh(tau)
        numerical=np.dot(weights/2,(1-rho)**2/rho)
        exact=np.sinh(tau)*np.arctan(np.sinh(tau))/tau**2-1
        assert abs(numerical-exact)<1e-12
    from fractions import Fraction as F
    for n in [2,17,18,64]:
        bm=sum(F(q*(q+1)*(3*n+2-2*q),n*(n+1)*(n+2))for q in range(1,n))
        uni=sum(F(q,n)for q in range(1,n))
        pro=sum(F(q*(q+1),n*(n+1))for q in range(1,n))
        assert bm==uni==F(n-1,2) and bm-pro==F(n-1,6)
    for values in D['llama_three_arm'].values():
        for v in values.values():assert abs(np.exp(v['nll'])-v['ppl'])<1e-8

def method_overview():
    """Analytic method schematic: exact quantiles and finite-grid profiles.

    K=8 and tau=2 make the allocation illustration legible; n=17 is the
    Llama transition width. No model observations or generated artwork enter.
    """
    tau=2.;k=8;u=(np.arange(k)+.5)/k
    quant=1-np.arcsinh((1-u)*np.sinh(tau))/tau
    z=(quant-quant[0])/(quant[-1]-quant[0]);geo=np.linspace(0,1,k)
    assert len(z)==len(geo)==k and z[0]==0 and z[-1]==1
    assert np.all(np.diff(z)>0) and np.all(z<=geo+1e-14)
    n=17;indices=np.arange(-4,n+5);q=np.clip(indices,0,n)
    ts=q*(3*n*n+3*n+1-q*q)/(n*(n+1)*(2*n+1))
    bm=q*(q+1)*(3*n+2-2*q)/(n*(n+1)*(n+2))
    for m in [ts,bm]:
        assert np.all(np.diff(m)>=0) and np.all(m[indices<=0]==0)
        assert np.all(m[indices>=n]==1)
    fig=plt.figure(figsize=(7.25,3.5),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    teal='#008577';gray='#929BA4'
    def label(x,y,s,**kw):
        return ax.text(x,y,s,**({'color':INK,'fontsize':9,'va':'center'}|kw))
    def arrow(start,end,color=INK,lw=1.,scale=10):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=scale,
                                    linewidth=lw,color=color,shrinkA=0,shrinkB=0))
    label(.025,.955,'(a) Allocation beyond the base',fontsize=10.5,fontweight='bold')
    label(.255,.825,r'$x_k=a+Rz_k$',fontsize=18,ha='center')
    left,right=.095,.425
    for values,y,color,name in [(geo,.635,BLUE,'Geometric'),(z,.425,ORANGE,'Reallocated')]:
        ax.plot([left,right],[y,y],color=INK,lw=1.0)
        xx=left+(right-left)*values
        ax.vlines(xx[1:-1],y-.027,y+.027,color=color,lw=2.1)
        ax.scatter(xx[[0,-1]],[y,y],s=32,facecolor='white',edgecolor=INK,lw=1.2,zorder=4)
        ax.text(left,y+.068,name,color=color,fontsize=10,fontweight='bold',va='center')
    for a,b in zip(geo[1:-1],z[1:-1]):
        ax.plot([left+(right-left)*a,left+(right-left)*b],[.602,.458],
                color=gray,lw=.7,ls=(0,(3,3)),zorder=0)
    ax.vlines([left,right],.265,.685,color=gray,lw=.8,ls=(0,(3,3)),zorder=0)
    arrow((left,.335),(right,.335),lw=.7,scale=8)
    label(left,.30,'Fast',ha='center',fontsize=8)
    label(right,.30,'Slow',ha='center',fontsize=8)
    label(.26,.29,r'Normalized exponent $z$',ha='center',fontsize=8)
    ax.plot([left,left,right,right],[.225,.245,.245,.225],color=INK,lw=.9)
    label(.26,.195,'Same frequency range',ha='center',fontsize=9)
    label(.255,.075,'Fixed endpoints, different spacing',ha='center',
          fontsize=10,fontweight='bold')

    label(.478,.53,'Design\nspacing',ha='center',fontsize=9.5,fontweight='bold',linespacing=1.4)
    arrow((.468,.625),(.523,.775),lw=1.1,scale=11)
    arrow((.468,.43),(.523,.285),lw=1.1,scale=11)

    label(.54,.955,'(b) Learning: Cosh',fontsize=10.5,fontweight='bold')
    ax.plot([.54,.98],[.915,.915],color=ORANGE,lw=1.1)
    density=fig.add_axes([.55,.655,.145,.195])
    phi=np.linspace(0,1,161);rho=tau*np.cosh(tau*(1-phi))/np.sinh(tau)
    density.fill_between(phi,0,rho,color=ORANGE,alpha=.09)
    density.plot(phi,rho,color=ORANGE,lw=1.6)
    density.set(xlim=(0,1.04),ylim=(0,rho[0]*1.08));density.axis('off')
    density.annotate('',xy=(1.04,0),xytext=(0,0),arrowprops=dict(arrowstyle='->',lw=.7,color=INK))
    density.annotate('',xy=(0,rho[0]*1.08),xytext=(0,0),arrowprops=dict(arrowstyle='->',lw=.7,color=INK))
    label(.552,.882,r'Density $\rho(\phi)$',fontsize=8.5)
    label(.625,.609,'Density prior',ha='center',fontsize=8.5)
    arrow((.705,.745),(.745,.745),lw=1.0)
    xx=.765+.21*z
    ax.plot([xx[0],xx[-1]],[.745,.745],color=INK,lw=.9)
    ax.vlines(xx[1:-1],.718,.772,color=ORANGE,lw=1.8)
    ax.scatter(xx[[0,-1]],[.745,.745],s=23,facecolor='white',edgecolor=INK,lw=1.,zorder=4)
    label(.872,.84,'Analytic quantiles',fontsize=8.5,ha='center')
    label(.872,.658,'Train with the table',fontsize=8.5,ha='center',fontweight='bold')

    label(.54,.505,'(c) Zero training: TailSpline / BM',fontsize=10.5,fontweight='bold')
    ax.plot([.54,.98],[.464,.464],color=teal,lw=1.1)
    curve=fig.add_axes([.558,.145,.42,.215])
    curve.axvspan(-4,0,color='#F0F2F4',zorder=0)
    curve.axvspan(n,n+4,color='#F0F2F4',zorder=0)
    curve.plot(indices,ts,color=ORANGE,lw=1.6,marker='o',ms=2.2,label='TailSpline')
    curve.plot(indices,bm,color=teal,lw=1.4,ls='--',marker='s',ms=1.8,label='BM')
    curve.axvline(0,color=gray,lw=.65,ls=':');curve.axvline(n,color=gray,lw=.65,ls=':')
    curve.set(xlim=(-4,n+4),ylim=(-.06,1.09),xticks=[0,n],xticklabels=[r'$l$',r'$h$'],yticks=[0,1])
    curve.tick_params(labelsize=7.5,length=2,pad=2)
    curve.spines[['top','right']].set_visible(False)
    curve.set_ylabel(r'$m_k$',rotation=0,labelpad=7,fontsize=9)
    curve.yaxis.set_label_coords(-.052,.77)
    curve.text(-2,1.20,'Keep',ha='center',fontsize=8,clip_on=False)
    curve.text(n/2,1.20,'Redistribute',ha='center',fontsize=8,clip_on=False)
    curve.text(n+2,1.20,r'Interpolate $/s$',ha='center',fontsize=8,clip_on=False)
    curve.text(4.7,.78,'TailSpline',color=ORANGE,fontsize=8)
    curve.text(10.4,.25,'BM',color=teal,fontsize=8)
    label(.768,.069,r'Public grid, $L$, $s$ $\longrightarrow$ one static table',ha='center',fontsize=8.5)
    label(.768,.016,'No weight updates or activation fitting',ha='center',fontsize=8)
    finish(fig,'fig_method_overview')
    # An editable text-preserving SVG accompanies the print PDF.
    with plt.rc_context({'svg.fonttype':'none'}):
        fig.savefig(HERE/'fig_method_overview.svg',bbox_inches='tight',pad_inches=.035)

def overview():
    fig,axes=plt.subplots(1,3,figsize=(7.25,2.7))
    ax=axes[0];u=(np.arange(32)+.5)/32;quant=1-np.arcsinh((1-u)*np.sinh(4))/4;z=(quant-quant[0])/(quant[-1]-quant[0])
    for xs,y,col,label in [(np.linspace(0,1,32),.7,BLUE,'Uniform'),(z,.25,ORANGE,'Cosh')]:
        ax.scatter(xs,np.full(32,y),s=8,color=col);ax.scatter([0,1],[y,y],s=25,facecolor='white',edgecolor=INK);ax.text(.03,y+.12,label,color=col,fontsize=8)
    ax.set(xlim=(-.05,1.05),ylim=(0,1),xlabel='Normalized exponent',yticks=[]);ax.spines[['left','right','top']].set_visible(False);ax.set_title('(a) Same endpoints',loc='left')
    ax=axes[1];ls=[256,512,1024,2048];block=D['range']['fixed_training_range'];vs=np.array([[block[str(l)]['seed_values'][str(s)] for l in ls] for s in [42,137,256]])
    for v in vs:ax.plot(range(4),v,color='#AFB7BD',lw=.8,marker='o',ms=2)
    ax.plot(range(4),vs.mean(0),color=ORANGE,lw=1.8,marker='D',ms=3);ax.axhline(0,color=INK,lw=.7)
    ax.set(xticks=range(4),xticklabels=['1x','2x','4x','8x'],xlabel='Eval. / train length',ylabel='Cosh - Geo tail NLL');ax.set_title('(b) Three-seed training',loc='left');axis_style(ax)
    ax=axes[2];keys=['same_support_geometric','nearest_movement_profile_ramp','derived']
    for i,(k,col,lab) in enumerate(zip(keys,[BLUE,'#009E73',ORANGE],['Uniform','Ramp','Derived'])):
        vals=[D['frozen'][m]['macro'][k]*100 for m in ['olmo_16k_unseen9','qwen_64k_core4']]
        ax.bar(np.arange(2)+(i-1)*.24,vals,width=.23,color=col,label=lab)
    ax.set(xticks=[0,1],xticklabels=['OLMo\n16K','Qwen\n64K'],ylabel='Task macro (%)',ylim=(0,86));ax.set_title('(c) Frozen models',loc='left');ax.legend(frameon=False,fontsize=6.8,ncol=3,loc='upper center',columnspacing=.5,handlelength=.8);axis_style(ax)
    fig.subplots_adjust(left=.035,right=.99,bottom=.25,top=.85,wspace=.68);finish(fig,'fig_evidence_overview')

def learning():
    fig,axes=plt.subplots(1,2,figsize=(7.25,3.0),gridspec_kw={'wspace':.45})
    ws=256*np.exp(-np.log(256)*np.arange(32)/32);g=white(ws)
    a=np.array([[.5*np.sum(g[2*i:2*i+2,2*j:2*j+2]**2) for j in range(32)]for i in range(32)])
    ax=axes[0];im=ax.imshow(a,origin='lower',vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.add_patch(Rectangle((23.5,23.5),8,8,fill=False,edgecolor=ORANGE,lw=1.2));ax.set(xlabel='Frequency-pair index',ylabel='Frequency-pair index');ax.set_title('(a) Which rotary pairs overlap?',loc='left',fontsize=8.5);fig.colorbar(im,ax=ax,fraction=.045,pad=.03)
    ax=axes[1];mla=D['mla'];ls=mla['eval_lengths']
    for key,col,style,label in [('GEO',BLUE,'-','Geo'),('EVQ',ORANGE,'-','Cosh'),('GEO+YaRN(s=4)',BLUE,'--','Geo + blend'),('EVQ+YaRN(s=4)',ORANGE,'--','Cosh + blend')]:
        sample=np.array([[mla['extended'][key][str(seed)][str(l)] for l in ls]for seed in mla['seeds']]);vals=sample.mean(0)
        if key in ['GEO','EVQ']:ax.fill_between(np.array(ls)/1024,sample.min(0),sample.max(0),color=col,alpha=.10,linewidth=0)
        ax.plot(np.array(ls)/1024,vals,style,color=col,marker='o',ms=3,label=label)
    ax.set(xlabel='Evaluation length (K tokens)',ylabel='PPL',xticks=[8,16,24,32]);ax.set_title('(b) Does allocation help after learning?',loc='left',fontsize=8.5);ax.legend(frameon=False,fontsize=7.5);axis_style(ax)
    fig.subplots_adjust(left=.08,right=.99,bottom=.22,top=.87);finish(fig,'fig_allocation_learning')

def tailspline():
    fig=plt.figure(figsize=(7.25,6.2))
    gs=fig.add_gridspec(3,3,height_ratios=[.72,1,1])
    top=fig.add_subplot(gs[0,:]);n=17;q=np.arange(1,n+1)
    for vals,col,label in [(2*q/(n*(n+1)),BLUE,'MrPro'),(6*q*(n-q+1)/(n*(n+1)*(n+2)),'#009E73','BM'),(3*(n+q)*(n-q+1)/(n*(n+1)*(2*n+1)),ORANGE,'TailSpline')]:
        top.plot(np.arange(n+2),np.r_[0,vals,0],color=col,lw=1.6,marker="o",ms=2.5,label=label)
    top.set(xlabel='Gap index (0 and 18: unchanged outer gaps)',ylabel='Extra log gap / log s',xticks=[0,1,5,9,13,17,18])
    top.set_title('(a) Extra log gaps: the boundary trade-off',loc='left',fontsize=9)
    top.set_ylim(-.004,.15)
    top.legend(frameon=False,ncol=3,loc='upper center',fontsize=8)
    top.annotate('Larger entry jump',xy=(1,3/(2*n+1)),xytext=(2.6,.112),fontsize=7.5,color=ORANGE,
                 arrowprops=dict(arrowstyle='->',color=ORANGE,lw=.8))
    top.annotate('Smaller tail jump',xy=(n,6/((n+1)*(2*n+1))),xytext=(11.1,.025),fontsize=7.5,color=ORANGE,
                 arrowprops=dict(arrowstyle='->',color=ORANGE,lw=.8))
    top.axvspan(-.2,.5,color='#F1F3F5',zorder=0);top.axvspan(17.5,18.2,color='#F1F3F5',zorder=0);axis_style(top)
    axes=np.array([[fig.add_subplot(gs[r+1,c]) for c in range(3)] for r in range(2)])
    for row,(key,name) in enumerate([('tailspline','Llama'),('tailspline_olmo','OLMo')]):
        t=D[key];x=np.array(t['lengths'])/1024
        w=np.array([.25,.5,.25])
        for col,metric in [(0,'full13'),(2,'ppl')]:
            ax=axes[row,col]
            for method,color in [('MrPro',BLUE),('TailSpline',ORANGE)]:
                ax.plot(x,np.array(t[metric][method])*(100 if metric=='full13' else 1),color=color,marker='o',ms=3,label=method)
            assert abs(w@(np.array(t[metric]['TailSpline'])-t[metric]['MrPro'])-t['auc_delta'][metric])<2e-6
            ax.set(ylabel='Task macro (%)' if col==0 else 'PPL')
            if row==0 and col==0:ax.legend(frameon=False,fontsize=7)
        ax=axes[row,1];ax.axhline(0,color=INK,lw=.7);ax.plot(x,np.array(t['niah_delta'])*100,color=ORANGE,marker='o',ms=3);ax.set(ylabel='Difference (pp)')
        for col,title in enumerate(['RULER-13','NIAH difference','Natural-text PPL']):
            ax=axes[row,col];ax.set(xticks=x,xlabel='Length (K tokens)');ax.set_title(f'({chr(98+row*3+col)}) {name}: {title}',loc='left',fontsize=8.5);axis_style(ax)
    fig.subplots_adjust(left=.075,right=.99,bottom=.10,top=.95,wspace=.60,hspace=.90);finish(fig,'fig_tailspline_main')

def table():
    rows=[]
    for stage in ['50%','75%','100%']:
        vals=[np.mean(list(v['16384']for v in D['mla']['progression'][a][stage].values()))for a in ['GEO','EVQ']]
        rows.append(f"MLA, {stage.replace('%',r'\%')} budget & PPL at $16$K & {vals[0]:.1f} & {vals[1]:.1f} "+r'\\')
    text=r'''\begin{table}[t]
\centering\small
\caption{\textbf{Allocation benefits across learning protocols.} Each block compares matched arms, with its own metric. MLA uses three seeds, midpoint Cosh and $8$K training. The $750$M pair uses fixed endpoints and one seed. OLMo uses one matched adaptation pair with physical inputs $\le4$K and explicit long-phase exposure; its score requires the full answer and EOS. These blocks are not pooled.}
\label{tab:training-models}
\begin{tabular}{@{}llrr@{}}
\toprule
Protocol & Endpoint & Geo / Native & Cosh \\
\midrule
'''+ '\n'.join(rows)+r'''
\midrule
$750$M continuation & PPL at $4$K & 22.0 & 22.3 \\
 & PPL at $16$K & 45.1 & 24.4 \\
 & $8$K answer-token exact & $0/40$ & $31/40$ \\
\midrule
OLMo matched adaptation & $4$K complete + EOS & $95/100$ & $100/100$ \\
 & $8$K complete + EOS & $18/100$ & $98/100$ \\
 & $16$K complete + EOS & $0/100$ & $60/100$ \\
\bottomrule
\end{tabular}
\end{table}
'''
    (HERE.parent/'tables/table_learning_main.tex').write_text(text)

if __name__=='__main__':
    verify();method_overview();overview();learning();tailspline();table();print('Method overview, three evidence figures, learning table and TailSpline algebra verified.')
