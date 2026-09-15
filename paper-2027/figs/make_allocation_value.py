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
    """Lead with distinct interventions, using existing recorded observations."""
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.05))
    u=(np.arange(8)+.5)/8
    q=1-np.arcsinh((1-u)*np.sinh(2))/2
    z=(q-q[0])/(q[-1]-q[0])
    ax=axes[0]
    for y, values, color, label in [(1,np.linspace(0,1,8),BLUE,'Geometric'),(0,z,ORANGE,'Cosh')]:
        ax.hlines(y,0,1,color=INK,lw=.7)
        ax.scatter(values[1:-1],np.full(6,y),s=20,color=color)
        ax.scatter([0,1],[y,y],s=23,facecolor='white',edgecolor=INK,zorder=3)
        ax.text(.05,y+.17,label,fontsize=8,color=color)
    ax.set(xlim=(-.05,1.05),ylim=(-.25,1.48),yticks=[],xticks=[0,1],xlabel='Normalized exponent z')
    ax.set_title('(a) Fixed support',loc='left',fontsize=9)
    axis_style(ax)
    ax=axes[1];lengths=[256,512,1024,2048]
    block=D['range']['fixed_training_range']
    values=np.array([[block[str(length)]['seed_values'][str(seed)] for length in lengths] for seed in [42,137,256]])
    for row in values: ax.plot(range(4),row,color='#B4BBC2',lw=.8,marker='o',ms=2)
    ax.plot(range(4),values.mean(0),color=ORANGE,lw=1.5,marker='D',ms=3)
    ax.axhline(0,color=INK,lw=.7)
    ax.set(xticks=range(4),xticklabels=['1x','2x','4x','8x'],xlabel='Eval. / train length',ylabel='Cosh - Geo NLL')
    ax.set_title('(b) Three paired seeds',loc='left',fontsize=9);axis_style(ax)
    ax=axes[2]
    crossing=D['controlled_crossing']['mean_tail_nll_two_seed_length1024']
    matrix=np.array([[crossing[w][t] for t in ['fmrope_derived','cosh_derived']]
                     for w in ['fmrope_weights','anchored_cosh_weights']])
    assert matrix[0,0]<matrix[0,1] and matrix[1,1]<matrix[1,0]
    ax.imshow(matrix,cmap='Blues',vmin=3,vmax=6,aspect='auto')
    for row in range(2):
        for col in range(2):
            ax.text(col,row,f'{matrix[row,col]:.3f}',ha='center',va='center',fontsize=9,
                    color='white' if matrix[row,col]>4.5 else INK,
                    fontweight='bold' if row==col else 'normal')
    ax.set(xticks=[0,1],xticklabels=['Geo table','Cosh table'],yticks=[0,1],yticklabels=['Geo W','Cosh W'])
    ax.tick_params(length=0,labelsize=7.5)
    ax.set_title('(c) Learned compatibility',loc='left',fontsize=9)
    fig.subplots_adjust(left=.035,right=.99,bottom=.26,top=.86,wspace=.65)
    finish(fig,'fig_method_overview')
    with plt.rc_context({'svg.fonttype':'none'}):
        fig.savefig(HERE/'fig_method_overview.svg',bbox_inches='tight',pad_inches=.035)
    svg=HERE/'fig_method_overview.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def overview():
    fig,axes=plt.subplots(1,3,figsize=(7.25,2.45))
    lengths=[256,512,1024,2048]
    for ax,key,title,color in zip(axes[:2],['target_matched_range','fixed_training_range'],
                                  ['(a) Retargeted support','(b) Retained training support'],[BLUE,ORANGE]):
        block=D['range'][key]
        values=np.array([[block[str(length)]['seed_values'][str(seed)]
                          for length in lengths] for seed in [42,137,256]])
        assert np.all(values[:,1:]>0) if key=='target_matched_range' else np.all(values[:,1:]<0)
        for row in values:
            ax.plot(range(4),row,color='#AFB7BD',lw=.85,marker='o',ms=2)
        ax.plot(range(4),values.mean(0),color=color,lw=1.8,marker='D',ms=3)
        ax.axhline(0,color=INK,lw=.7)
        ax.set(xticks=range(4),xticklabels=['1x','2x','4x','8x'],
               xlabel='Eval. / train length',ylabel='Cosh - Geo tail NLL',ylim=(-.52,.76))
        ax.set_title(title,loc='left',fontsize=9)
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
    data=json.loads((HERE/'field_gap_inputs.json').read_text())
    clean=data['clean'];mid=data['clean16k']
    fig,axes=plt.subplots(1,2,figsize=(7.25,2.7),gridspec_kw={'width_ratios':[1,1.55]})
    ax=axes[0]
    points=[mid['contrasts']['mrpro']['delta_by_length']['16384'],clean['point']['tailspline']-clean['point']['mrpro']]
    intervals=[mid['contrasts']['mrpro']['bootstrap']['delta_by_length_interval95']['16384'],clean['contrast']['bootstrap']['delta_log_auc']['interval95']]
    for y,(point,interval,color) in enumerate(zip(points,intervals,[BLUE,ORANGE])):
        lo,hi=np.array(interval)*100;point*=100
        ax.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',color=color,capsize=3,ms=5)
        ax.text(point,y+.19,f'{point:+.2f} pp',ha='center',va='top',fontsize=9,color=color)
    ax.axvline(0,color=INK,lw=.7);ax.set(yticks=[0,1],yticklabels=['16K (2L)','32K (4L)'],ylim=(-.45,1.5),xlim=(-.5,14.5),xticks=[0,5,10],xlabel='Full-13 gain (pp)')
    ax.invert_yaxis();ax.set_title('(a) Quality at both lengths',loc='left',fontsize=9);axis_style(ax)
    ax=axes[1];keys=list(clean['task_deltas']);ys=np.arange(len(keys))
    v32=np.array([clean['task_deltas'][t]for t in keys])*100
    v16=np.array([mid['summaries']['tailspline']['by_length']['16384']['tasks'][t]['official']-mid['summaries']['mrpro']['by_length']['16384']['tasks'][t]['official']for t in keys])*100
    names=[k.replace('niah_','').replace('multikey','MK').replace('single','S').replace('multiquery','MQ').replace('multivalue','MV')for k in keys]
    ax.barh(ys-.18,v16,height=.34,color=BLUE,label='16K: 50/task')
    ax.barh(ys+.18,v32,height=.34,color=ORANGE,label='32K: 200/task')
    ax.axvline(0,color=INK,lw=.7);ax.invert_yaxis();ax.set(yticks=ys,yticklabels=names,xlim=(-11,42),xticks=[0,20,40],xlabel='Task gain over MrPro (pp)')
    ax.tick_params(axis='y',labelsize=7.5);ax.set_title('(b) All task means',loc='left',fontsize=9)
    ax.legend(frameon=False,fontsize=7,loc='lower right');axis_style(ax)
    fig.subplots_adjust(left=.12,right=.985,bottom=.20,top=.86,wspace=.58)
    finish(fig,'fig_tailspline_main')


def tailspline_classic():
    fig,axes=plt.subplots(1,2,figsize=(6.2,2.5))
    for ax,(key,name) in zip(axes,[('tailspline','Llama'),('tailspline_olmo','OLMo')]):
        block=D[key];x=np.array(block['lengths'])/1024
        for method,color,marker in [('MrPro',BLUE,'o'),('TailSpline',ORANGE,'D')]:
            ax.plot(x,np.array(block['full13'][method])*100,color=color,marker=marker,ms=3,label=method)
        delta=block['auc_delta']['full13']*100;lo,hi=np.array(block['auc_delta_ci95']['full13'])*100
        ax.set_title(f'{name}: classic padded panel',loc='left',fontsize=9)
        ax.set(xticks=x,xlabel='Length (K tokens)',ylabel='Task macro (%)',ylim=(0,103))
        ax.text(.02,-.36,f'AUC {delta:+.2f}pp, 95% CI [{lo:.2f}, {hi:.2f}]',transform=ax.transAxes,fontsize=8)
        ax.legend(frameon=False,fontsize=7,loc='lower left');axis_style(ax)
    fig.subplots_adjust(left=.10,right=.985,bottom=.31,top=.88,wspace=.42)
    finish(fig,'fig_tailspline_classic')


def construction_contrast():
    fig,ax=plt.subplots(figsize=(4.5,1.8))
    n=17;q=np.arange(1,n+1)
    for values,color,name in [(2*q/(n*(n+1)),BLUE,'MrPro'),(3*(n+q)*(n-q+1)/(n*(n+1)*(2*n+1)),ORANGE,'TailSpline')]:
        ax.plot(np.arange(n+2),np.r_[0,values,0],color=color,label=name,lw=1.5)
    ax.set(xlabel='Gap index',ylabel='Extra gap / log s',xticks=[0,9,18])
    ax.legend(frameon=False,fontsize=8,loc='upper center',ncol=2);axis_style(ax)
    fig.subplots_adjust(left=.15,right=.985,bottom=.29,top=.90)
    finish(fig,'fig_construction_contrast')


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
    verify();method_overview();overview();geometry();learning();tailspline();tailspline_classic();construction_contrast();deployment_details();table();print('Five main figures, deployment supplement, learning table and CPU algebra verified.')
