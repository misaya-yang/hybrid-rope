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
    w=np.array([.25,.5,.25])
    for model in ['tailspline','tailspline_olmo']:
        block=D[model]
        for metric in ['full13','ppl']:
            delta=w@(np.array(block[metric]['TailSpline'])-np.array(block[metric]['MrPro']))
            assert abs(delta-block['auc_delta'][metric])<2e-6
        lo,hi=block['auc_delta_ci95']['full13']
        assert lo<block['auc_delta']['full13']<hi
    assert abs(w@D['tailspline']['niah_delta']-.0375)<1e-10

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
        ax.text(left,y+.068,name,color=color,fontsize=10,fontweight='bold',va='center',
                bbox=dict(facecolor='white',edgecolor='none',pad=1.2),zorder=5)
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
    curve.yaxis.set_label_coords(-.055,.53)
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
    svg=HERE/'fig_method_overview.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')

def overview():
    fig,axes=plt.subplots(1,3,figsize=(7.25,2.45))
    ax=axes[0];u=(np.arange(32)+.5)/32
    quant=1-np.arcsinh((1-u)*np.sinh(4))/4
    z=(quant-quant[0])/(quant[-1]-quant[0])
    for values,y,color,label in [(np.linspace(0,1,32),.7,BLUE,'Geo'),(z,.25,ORANGE,'Cosh')]:
        ax.scatter(values,np.full(32,y),s=8,color=color)
        ax.scatter([0,1],[y,y],s=25,facecolor='white',edgecolor=INK)
        ax.text(.03,y+.12,label,color=color,fontsize=9)
    ax.set(xlim=(-.05,1.05),ylim=(0,1),xlabel='Normalized exponent',yticks=[])
    ax.spines[['left','right','top']].set_visible(False)
    ax.set_title('(a) Matched endpoints',loc='left',fontsize=9)
    ax=axes[1];lengths=[256,512,1024,2048];block=D['range']['fixed_training_range']
    values=np.array([[block[str(length)]['seed_values'][str(seed)]
                      for length in lengths] for seed in [42,137,256]])
    assert np.all(values[:,1:]<0)
    for row in values:
        ax.plot(range(4),row,color='#AFB7BD',lw=.85,marker='o',ms=2)
    ax.plot(range(4),values.mean(0),color=ORANGE,lw=1.8,marker='D',ms=3)
    ax.axhline(0,color=INK,lw=.7)
    ax.set(xticks=range(4),xticklabels=['1x','2x','4x','8x'],
           xlabel='Eval. / train length',ylabel='Cosh - Geo tail NLL')
    ax.set_title('(b) Three-seed fixed support',loc='left',fontsize=9)
    axis_style(ax)
    ax=axes[2];keys=['same_support_geometric','nearest_movement_profile_ramp','derived']
    for i,(key,color,label) in enumerate(zip(keys,[BLUE,'#009E73',ORANGE],['Uniform','Ramp','Derived'])):
        values=[D['frozen'][model]['macro'][key]*100
                for model in ['olmo_16k_unseen9','qwen_64k_core4']]
        ax.bar(np.arange(2)+(i-1)*.24,values,width=.23,color=color,label=label)
    ax.set(xticks=[0,1],xticklabels=['OLMo 16K\nHeld-out (9)','Qwen 64K\nDev. (4)'],
           ylabel='Task macro (%)',ylim=(0,86))
    ax.set_title('(c) Frozen matched support',loc='left',fontsize=9)
    ax.legend(frameon=False,fontsize=7,ncol=3,loc='upper center',columnspacing=.5,handlelength=.8)
    axis_style(ax)
    fig.subplots_adjust(left=.075,right=.99,bottom=.26,top=.85,wspace=.69)
    finish(fig,'fig_evidence_overview')


def geometry():
    frequencies=256*np.exp(-np.log(256)*np.arange(32)/32)
    gram=white(frequencies)
    overlap=np.array([[.5*np.sum(gram[2*i:2*i+2,2*j:2*j+2]**2)
                       for j in range(32)] for i in range(32)])
    slow=gram[48:,48:]
    rank=np.trace(slow)**2/np.sum(slow**2)
    assert abs(rank-2.11474)<1e-5
    assert np.allclose(np.diag(overlap),1) and np.allclose(overlap,overlap.T)
    fig=plt.figure(figsize=(4.7,2.15))
    ax=fig.add_axes([.14,.24,.40,.72])
    im=ax.imshow(overlap,origin='lower',vmin=0,vmax=1,cmap='Blues',aspect='equal')
    ax.add_patch(Rectangle((23.5,23.5),8,8,fill=False,edgecolor=ORANGE,lw=1.3))
    ax.set(xlabel='Frequency-pair index',ylabel='Frequency-pair index',xticks=[0,8,16,24,31],yticks=[0,8,16,24,31])
    color_ax=fig.add_axes([.58,.24,.025,.72]);fig.colorbar(im,cax=color_ax,ticks=[0,.5,1])
    fig.text(.71,.81,'Full-pair overlap',fontsize=10,fontweight='bold')
    fig.text(.71,.62,'Slowest eight pairs',fontsize=9,color=ORANGE)
    fig.text(.71,.44,r'$r_2 = 2.11$',fontsize=15,color=INK)
    fig.text(.71,.26,r'$\omega L\in[1.19,4]$',fontsize=10)
    finish(fig,'fig_allocation_geometry')


def learning():
    fig,ax=plt.subplots(figsize=(5.6,2.05))
    mla=D['mla'];lengths=mla['eval_lengths']
    for key,color,style,label in [('GEO',BLUE,'-','Geo'),('EVQ',ORANGE,'-','Cosh'),
                                  ('GEO+YaRN(s=4)',BLUE,'--','Geo + blend'),
                                  ('EVQ+YaRN(s=4)',ORANGE,'--','Cosh + blend')]:
        sample=np.array([[mla['extended'][key][str(seed)][str(length)]
                          for length in lengths] for seed in mla['seeds']])
        if key in ['GEO','EVQ']:
            ax.fill_between(np.array(lengths)/1024,sample.min(0),sample.max(0),color=color,alpha=.12,linewidth=0)
        ax.plot(np.array(lengths)/1024,sample.mean(0),style,color=color,marker='o',ms=3,label=label)
    ax.set(xlabel='Evaluation length (K tokens)',ylabel='PPL',xticks=[8,16,24,32])
    ax.legend(frameon=False,fontsize=9,ncol=2,loc='upper left')
    axis_style(ax)
    fig.subplots_adjust(left=.12,right=.99,bottom=.25,top=.96)
    finish(fig,'fig_allocation_learning')


def tailspline():
    fig=plt.figure(figsize=(7.25,3.7))
    gs=fig.add_gridspec(2,2,height_ratios=[.65,1])
    top=fig.add_subplot(gs[0,:]);n=17;q=np.arange(1,n+1)
    for values,color,label in [(2*q/(n*(n+1)),BLUE,'MrPro'),
                               (6*q*(n-q+1)/(n*(n+1)*(n+2)),'#009E73','BM'),
                               (3*(n+q)*(n-q+1)/(n*(n+1)*(2*n+1)),ORANGE,'TailSpline')]:
        top.plot(np.arange(n+2),np.r_[0,values,0],color=color,lw=1.6,marker='o',ms=2.5,label=label)
    top.set(xlabel='Gap index (0 and 18: unchanged outer gaps)',ylabel=r'Extra gap / $\log s$',
            xticks=[0,1,5,9,13,17,18],ylim=(-.004,.16))
    top.set_title('(a) Boundary increments',loc='left',fontsize=10)
    top.legend(frameon=False,ncol=3,loc='lower right',bbox_to_anchor=(1,1.01),fontsize=9)
    top.annotate('Larger entry jump',xy=(1,3/(2*n+1)),xytext=(2.6,.112),fontsize=8,color=ORANGE,
                 arrowprops=dict(arrowstyle='->',color=ORANGE,lw=.8))
    top.annotate('Smaller tail jump',xy=(n,6/((n+1)*(2*n+1))),xytext=(11.1,.025),fontsize=8,color=ORANGE,
                 arrowprops=dict(arrowstyle='->',color=ORANGE,lw=.8))
    top.axvspan(-.2,.5,color='#F1F3F5',zorder=0);top.axvspan(17.5,18.2,color='#F1F3F5',zorder=0)
    axis_style(top)
    for col,(key,name) in enumerate([('tailspline','Llama'),('tailspline_olmo','OLMo')]):
        ax=fig.add_subplot(gs[1,col]);block=D[key];x=np.array(block['lengths'])/1024
        for method,color,marker in [('MrPro',BLUE,'o'),('TailSpline',ORANGE,'D')]:
            ax.plot(x,np.array(block['full13'][method])*100,color=color,marker=marker,ms=3,label=method)
        delta=block['auc_delta']['full13']*100
        lo,hi=np.array(block['auc_delta_ci95']['full13'])*100
        ax.set_title(f'({chr(98+col)}) {name}: RULER-13',loc='left',fontsize=10)
        ax.text(.02,.07 if col==0 else .43,f'AUC difference {delta:+.2f}pp\nPaired 95% CI [{lo:+.2f}, {hi:+.2f}]',
                transform=ax.transAxes,fontsize=8.5,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1.5))
        ax.set(xticks=x,xlabel='Length (K tokens)',ylabel='Task macro (%)',ylim=(0,103))
        ax.legend(frameon=False,fontsize=8,ncol=2,loc='upper right')
        axis_style(ax)
    fig.subplots_adjust(left=.095,right=.99,bottom=.13,top=.94,wspace=.30,hspace=.95)
    finish(fig,'fig_tailspline_main')


def deployment_details():
    fig,axes=plt.subplots(2,2,figsize=(7.25,4.3))
    for row,(key,name) in enumerate([('tailspline','Llama'),('tailspline_olmo','OLMo')]):
        block=D[key];x=np.array(block['lengths'])/1024
        values=[np.array(block['niah_delta'])*100,
                100*(np.array(block['ppl']['TailSpline'])/np.array(block['ppl']['MrPro'])-1)]
        for col,(value,metric,label) in enumerate(zip(values,['NIAH subset','PPL vs MrPro'],['Difference (pp)','PPL change (%)'])):
            ax=axes[row,col];ax.axhline(0,color=INK,lw=.7)
            ax.plot(x,value,color=ORANGE,marker='o',ms=3)
            ax.set(xticks=x,xlabel='Length (K tokens)',ylabel=label)
            ax.set_title(f'({chr(97+2*row+col)}) {name}: {metric}',loc='left',fontsize=10)
            ax.margins(x=.12,y=.3);axis_style(ax)
    fig.subplots_adjust(left=.10,right=.99,bottom=.13,top=.94,wspace=.35,hspace=.65)
    finish(fig,'fig_tailspline_details')


def table():
    rows=[]
    for stage in ['50%','75%','100%']:
        vals=[np.mean(list(v['16384']for v in D['mla']['progression'][a][stage].values()))for a in ['GEO','EVQ']]
        escaped_stage=stage.replace('%',r'\%')
        rows.append(f"{escaped_stage} budget & PPL at $16$K & {vals[0]:.1f} & {vals[1]:.1f} "+r'\\')
    text=r'''\begin{table}[t]
\centering\small
\caption{\textbf{Allocation benefits across learning protocols.} Each block names its reference. MLA uses unanchored midpoint Cosh and $8$K training; the $750$M pair fixes endpoints. OLMo adaptation uses physical inputs $\le4$K with explicit long-phase exposure and requires the full answer plus EOS.}
\label{tab:training-models}
\begin{tabular}{@{}llrr@{}}
\toprule
Protocol & Metric & Reference & Cosh \\
\midrule
\multicolumn{4}{@{}l@{}}{\textit{MLA learning: Geo reference, three seeds}} \\
'''+ '\n'.join(rows)+r'''
\midrule
\multicolumn{4}{@{}l@{}}{\textit{$750$M continuation: Geo reference, one seed}} \\
After continuation & PPL at $4$K & 22.0 & 22.3 \\
 & PPL at $16$K & 45.1 & 24.4 \\
 & $8$K answer-token exact & $0/40$ & $31/40$ \\
\midrule
\multicolumn{4}{@{}l@{}}{\textit{OLMo adaptation: Native reference, one matched pair}} \\
After phase exposure & $4$K complete + EOS & $95/100$ & $100/100$ \\
 & $8$K complete + EOS & $18/100$ & $98/100$ \\
 & $16$K complete + EOS & $0/100$ & $60/100$ \\
\bottomrule
\end{tabular}
\end{table}
'''
    (HERE.parent/'tables/table_learning_main.tex').write_text(text)

if __name__=='__main__':
    verify();method_overview();overview();geometry();learning();tailspline();deployment_details();table();print('Five main figures, deployment supplement, learning table and CPU algebra verified.')
