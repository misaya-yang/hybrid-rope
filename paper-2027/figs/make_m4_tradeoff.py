#!/usr/bin/env python3
"""Plot all prespecified M4 configurations; no models or fitted Pareto frontier."""
from pathlib import Path
import json
import math
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ARMS = ['anchored_cosh_m075', 'anchored_cosh_rule', 'anchored_cosh_m125', 'anchored_exp_rule']
NAMES = ['Cosh 0.75x', 'Reference Cosh', 'Cosh 1.25x', 'Matched exponential']

def main():
    inp = json.loads((HERE / 'm4_tradeoff_inputs.json').read_text())
    runs = inp['runs']
    lookup = {(r['rope_base'], r['seq_len'], r['head_dim'], r['seed'], r['arm']): r for r in runs}
    assert len(lookup) == len(runs) == 180
    configs = sorted({key[:3] for key in lookup})
    assert len(configs) == 12
    weights = np.log2(np.array([2, 4, 8]) + 1)
    points = []
    for base, length, width in configs:
        for arm in ARMS:
            delta_in, delta_ood = [], []
            for seed in [42, 137, 256]:
                new = lookup[(base, length, width, seed, arm)]
                ref = lookup[(base, length, width, seed, 'native_geo')]
                for run in [new, ref]:
                    approximate = np.average([math.log(run['ppl'][str(length * r)]) for r in [2,4,8]], weights=weights)
                    assert abs(approximate - run['weighted_extrapolation_nll']) < 1e-6
                delta_in.append(math.log(new['ppl'][str(length)]) - math.log(ref['ppl'][str(length)]))
                delta_ood.append(new['weighted_extrapolation_nll'] - ref['weighted_extrapolation_nll'])
            x, y = float(np.mean(delta_in)), float(np.mean(delta_ood))
            aggregate = next(c for c in inp['by_structural_config'] if (c['rope_base'],c['seq_len'],c['head_dim']) == (base,length,width))
            assert abs(y - (aggregate['seed_mean_weighted_nll'][arm] - aggregate['seed_mean_weighted_nll']['native_geo'])) < 1e-12
            points.append(dict(rope_base=base, train_length=length, head_dim=width, arm=arm, delta_in=x, delta_ood=y))
    with (HERE / 'm4_tradeoff_points.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0])); writer.writeheader(); writer.writerows(points)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(2,2,figsize=(7.1,4.8),sharex=True,sharey=True)
    low = min(0, min(p[k] for p in points for k in ['delta_in','delta_ood']))
    high = max(0, max(p[k] for p in points for k in ['delta_in','delta_ood']))
    padding = (high-low)*.08
    limits=(low-padding,high+padding)
    for ax,arm,name in zip(axes.flat,ARMS,NAMES):
        ax.axhline(0,color='#6B7280',lw=.7);ax.axvline(0,color='#6B7280',lw=.7)
        for p in points:
            if p['arm'] != arm: continue
            ax.scatter(p['delta_in'],p['delta_ood'],c='#286A9D' if p['rope_base']==500000 else '#C66F29',
                       marker='o' if p['train_length']==256 else '^',s=35,edgecolors='white',linewidth=.4)
        ax.set_title(name,loc='left');ax.set_xlim(limits);ax.set_ylim(limits)
        ax.spines[['right','top']].set_visible(False);ax.grid(alpha=.12)
    for ax in axes[1]:ax.set_xlabel('Training-window NLL difference')
    for ax in axes[:,0]:ax.set_ylabel('Weighted OOD NLL difference')
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color='#286A9D',marker='o',ls='',label='Base 500K'),Line2D([],[],color='#C66F29',marker='o',ls='',label='Base 1M'),Line2D([],[],color='#707070',marker='o',ls='',label='Train 256'),Line2D([],[],color='#707070',marker='^',ls='',label='Train 1024')]
    fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False)
    fig.subplots_adjust(left=.12,right=.985,bottom=.12,top=.87,hspace=.38,wspace=.17)
    assert all(limits[0] <= p[k] <= limits[1] for p in points for k in ['delta_in','delta_ood']), 'Point outside fixed axes'
    fig.savefig(HERE/'fig_m4_tradeoffs.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(HERE/'fig_m4_tradeoffs.png',dpi=200)
    summary = {arm:{'ood_improves':sum(p['delta_ood']<0 for p in points if p['arm']==arm),'both_improve':sum(p['delta_in']<0 and p['delta_ood']<0 for p in points if p['arm']==arm)} for arm in ARMS}
    print(json.dumps({'configurations':12,'points':48,'paired_seeds_per_point':3,'summary':summary}))

if __name__ == '__main__': main()
