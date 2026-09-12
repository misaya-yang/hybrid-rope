"""Rebuild the integrated manuscript figures from recorded results and exact geometry.
No model execution. Portable inputs are exported for the manuscript source archive.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCES = {
    'frozen': 'data/curated/frozen_fixed_support_mature_20260823.json',
    'qa': 'docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json',
    'mla': 'data/curated/table18_mla_3seed_aggregate.json',
    'continuation': 'data/curated/phase15_750m_continue_result_20260306.json',
}
BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'mathtext.fontset':'stix','pdf.fonttype':42,'font.size':9,'axes.labelsize':9,
 'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
 'axes.spines.top':False,'axes.spines.right':False})

def data():
    portable = OUT/'story_figure_inputs.json'
    if all((ROOT / f).is_file() for f in SOURCES.values()):
        values = {k:json.loads((ROOT/f).read_text()) for k,f in SOURCES.items()}
        records = {k:{'file':f,'sha256':hashlib.sha256((ROOT/f).read_bytes()).hexdigest()} for k,f in SOURCES.items()}
        portable.write_text(json.dumps({'sources':records,'data':values},indent=2)+'\n')
    else:
        values = json.loads(portable.read_text())['data']
    return values

def gram(w,v):
    a = lambda x: np.sinc(x/np.pi)
    b = lambda x: 0. if x == 0 else 2*np.sin(x/2)**2/x
    d,s = w-v,w+v
    return .5*np.array([[a(d)+a(s), b(s)-b(d)], [b(s)+b(d),a(d)-a(s)]])

def white(ws,quad=False):
    if quad:
        t,weights = np.polynomial.legendre.leggauss(256)
        t,weights = (t+1)/2,weights/2
        basis = [np.column_stack((np.cos(w*t),np.sin(w*t))) for w in ws]
        g = lambda i,j: basis[i].T@(weights[:,None]*basis[j])
    else:
        g = lambda i,j: gram(ws[i],ws[j])
    factors=[]
    for i in range(len(ws)):
        ev,v=np.linalg.eigh(g(i,i))
        assert np.min(ev)>0
        factors.append((v / np.sqrt(ev))@v.T)
    return np.block([[factors[i]@g(i,j)@factors[j] for j in range(len(ws))] for i in range(len(ws))])

def rank(g):
    assert np.all(np.isfinite(g))
    return np.trace(g)**2/np.sum(g*g.T)

def finish(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.03)
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)

def geometry():
    ws=256*np.exp(-np.log(256)*np.arange(32)/32)
    g=white(ws)
    affinity=np.array([[.5*np.sum(g[2*i:2*i+2,2*j:2*j+2]**2) for j in range(32)] for i in range(32)])
    counts=np.arange(2,17)
    ranks=np.array([rank(white(ws[-n:])) for n in counts])
    n8=float(rank(white(ws[-8:])))
    error=abs(n8-rank(white(ws[-8:],True)))
    assert error<1e-10 and abs(n8-2.11474)<1e-5
    assert abs(rank(white(ws[-4:]))-2.00361)<1e-5
    anchor=4096*np.exp(-np.log(500000)*np.arange(41,64)/64)
    assert abs(rank(white(anchor))-2.00013)<1e-4
    ev=np.linalg.eigvalsh(white(ws[-8:]))[::-1]
    result={'measure':'Uniform[0,L]','base':256,'K':32,'L':256,
       'slow_subset_counts':counts.tolist(),'slow_subset_renyi2_rank':ranks.tolist(),
       'slow8_rank':n8,'slow8_leading_eigenvalues':ev[:3].tolist(),
       'slow8_quadrature_error':error,'anchor_500k_rank':float(rank(white(anchor)))}
    (OUT/'finite_window_geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(7,2.55),gridspec_kw={'width_ratios':[1.03,1], 'wspace':.48})
    cmap=LinearSegmentedColormap.from_list('overlap',['#FFFFFF','#CCE3EF',BLUE])
    im=axes[0].imshow(affinity,origin='lower',vmin=0,vmax=1,cmap=cmap,aspect='auto')
    axes[0].add_patch(Rectangle((23.5,23.5),8,8,fill=False,edgecolor=ORANGE,lw=1.5))
    axes[0].set(xlabel='Frequency-pair index',ylabel='Frequency-pair index',xticks=[0,8,16,24,31],yticks=[0,8,16,24,31])
    axes[0].set_title('(a) Full-pair positional overlap',loc='left')
    cb=fig.colorbar(im,ax=axes[0],fraction=.045,pad=.03,ticks=[0,.5,1]);cb.ax.tick_params(labelsize=7)
    ax=axes[1]
    ax.plot(counts,ranks,color=BLUE,marker='o',ms=3,lw=1.6)
    ax.axhline(2,color='#777777',lw=.7,ls='--')
    ax.scatter([8],[n8],color=ORANGE,s=30,zorder=3)
    ax.annotate('8 pairs: $r_2=2.11$',(8,n8),xytext=(5,3.3),color=ORANGE,
                arrowprops={'arrowstyle':'->','color':ORANGE,'lw':.9},fontsize=9)
    ax.set(xlabel='Number of slowest pairs included',ylabel='Block-whitened effective rank',xlim=(1.5,16.5),ylim=(1.8,4.7),xticks=[2,4,8,12,16])
    ax.set_title('(b) Finite-window concentration',loc='left')
    ax.grid(axis='y',color='#E2E6E9',lw=.5)
    finish(fig,'fig_allocation_basis')
    return result

def frozen(values):
    result=values['frozen']['olmo_16k_unseen9']['macro']
    frozen_scores=100*np.array([result[k] for k in ['same_support_geometric','nearest_movement_profile_ramp','derived']])
    names={'hotpotqa':'HotpotQA','2wikimqa':'2WikiMQA','qasper':'Qasper','narrativeqa':'NarrativeQA','multifieldqa_en':'MultiFieldQA'}
    qa=values['qa']; tasks=list(names)
    a=[];b=[]
    for task in tasks:
        rows=[r for r in qa['rows'] if r['row_id'].startswith(task+'_') and r['input_tokens']>4096]
        a.append(np.mean([r['baseline'] for r in rows])*100)
        b.append(np.mean([r['candidate'] for r in rows])*100)
        assert len(rows)==qa['by_stratum']['extended'][task]['n']
    assert sum(qa['by_stratum']['extended'][t]['n'] for t in tasks)==631
    a.append(np.mean(a));b.append(np.mean(b))
    assert abs(a[-1]-21.62460611603836)<.01 and abs(b[-1]-25.44)<.01
    fig,axes=plt.subplots(1,2,figsize=(7,2.65),gridspec_kw={'width_ratios':[1,1.25],'wspace':.58})
    ax=axes[0]; y=np.arange(3)
    ax.barh(y,frozen_scores,color=[BLUE,GREEN,ORANGE],height=.53)
    ax.set(yticks=y,yticklabels=['Uniform','Coarse ramp','Derived'],xlabel='RULER task macro (%)',xlim=(0,74))
    ax.invert_yaxis()
    for yy,v in zip(y,frozen_scores): ax.text(v+1,yy,f'{v:.2f}',va='center',fontsize=8)
    ax.set_title('(a) Fixed-support intervention',loc='left',pad=12)
    ax.grid(axis='x',color='#E2E6E9',lw=.5);ax.set_axisbelow(True)
    ax=axes[1]; y=np.arange(6)
    for yy,aa,bb in zip(y,a,b): ax.plot([aa,bb],[yy,yy],color='#ABB3BA',lw=1.6,zorder=1)
    ax.scatter(a,y,color=BLUE,marker='o',s=21,label='MrRoPE-Pro(4)',zorder=2)
    ax.scatter(b,y,color=ORANGE,marker='D',s=21,label='BM(4)',zorder=2)
    ax.set(yticks=y,yticklabels=[names[t] for t in tasks]+['Task mean'],xlabel='Whole-response token F1 (%)',xlim=(0,57))
    ax.invert_yaxis();ax.grid(axis='x',color='#E2E6E9',lw=.5)
    ax.set_title('(b) Natural-QA application',loc='left',pad=12)
    ax.legend(loc='upper right',frameon=False,fontsize=7,handletextpad=.3)
    for value,offset in [(a[-1],-8),(b[-1],8)]:
        ax.annotate(f'{value:.2f}',(value,5),xytext=(offset,-12),textcoords='offset points',ha='center',fontsize=8)
    ax.set_ylim(5.9,-.6)
    finish(fig,'fig_frozen_application')

if __name__=='__main__':
    values=data(); receipt=geometry(); frozen(values)
    for label in ['GEO','EVQ','GEO+YaRN(s=4)','EVQ+YaRN(s=4)']:
        for length in ['8192','16384']:
            actual=np.mean([values['mla']['extended'][label][str(seed)][length] for seed in values['mla']['seeds']])
            assert abs(actual-values['mla']['summary'][label][length]['mean'])<.011
    print(f'Built two story figures; finite-window r2={receipt["slow8_rank"]:.5f}, quadrature error={receipt["slow8_quadrature_error"]:.3g}; MLA seed aggregates and QA rows verified.')
