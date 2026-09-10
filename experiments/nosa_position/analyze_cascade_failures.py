"""Offline decomposition from saved SAME-STATE matrices, never independent Q/K."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch


def main():
    root=Path('results/pc2_cascade_same_state_v2')
    torch.set_num_threads(4)
    matrices=torch.load(root/'same_state_scores.pt',map_location='cpu',weights_only=True)
    real={(r['row_id'],r['layer'],r['position'],r['factor']):r for r in map(json.loads,(root/'states.jsonl').read_text().splitlines())}
    rows=[]
    for r in matrices:
        ex=r['exact_logmass'][:,:,0];ap=r['e04_logmass'][:,:,0];m=r['mandatory'][0];h,g,b=ex.shape
        p=ex.softmax(-1);pa=ap.softmax(-1);slots=min(33,b);free=slots-int(m.sum())
        truth=torch.argsort(p.sum(1).masked_fill(m,torch.inf),descending=True,stable=True)[:,:slots]
        order=torch.argsort(pa.sum(1).masked_fill(m,-torch.inf),descending=True,stable=True)
        score=p.sum(1);sortedscore=score.masked_fill(m,-torch.inf).sort(descending=True).values
        for factor in (2,4):
            c=m[None].expand(h,-1).clone();c.scatter_(1,order[:,:min(free*factor,b-int(m.sum()))],True)
            mix=torch.where(c[:,None],ex,ap);mixedscore=mix.softmax(-1).sum(1)
            target=c.gather(1,truth);free_target=(~m[truth]);coveredfree=float((target&free_target).sum()/free_target.sum())
            eqk=torch.argsort(score.masked_fill(~c,-torch.inf).masked_fill(m,torch.inf),descending=True,stable=True)[:,:slots]
            mqk=torch.argsort(mixedscore.masked_fill(~c,-torch.inf).masked_fill(m,torch.inf),descending=True,stable=True)[:,:slots]
            exactqk=bool((eqk.sort().values==truth.sort().values).all());mixedqk=bool((mqk.sort().values==truth.sort().values).all())
            den=(mix.logsumexp(-1)-ex.logsumexp(-1)).expm1()
            outside_mass=(p*(~c[:,None])).sum(-1)
            original=real[(r['row_id'],r['layer'],r['position'],factor)]
            family='multiquery' if 'multiquery' in r['row_id'] else 'multikey' if 'multikey' in r['row_id'] else 'occurrence' if 'occurrence' in r['row_id'] else 'qasper'
            row={'row_id':r['row_id'],'family':family,'layer':r['layer'],'position':r['position'],'factor':factor,'blocks':b,'free_qk_coverage':coveredfree,'all_free_qk_covered':bool(target.all()),'matrix_oracle_qk_equal':exactqk,'matrix_mixed_qk_equal':mixedqk,'candidate_fraction':float(c.float().mean()),'outside_exact_mass_max':float(outside_mass.max()),'relative_denominator_error_max':float(den.abs().max()),'relative_denominator_error_head_spread':float((den.max(1).values-den.min(1).values).max()),'cutoff_margin_min':float((sortedscore[:,free-1]-sortedscore[:,free]).min()) if 0<free<b-int(m.sum()) else None,**{k:original[k] for k in ['oracle_final_equal','mixed_final_equal','candidate_only_final_equal','warm_coarse_seconds','candidate_exact_seconds','candidate_route_seconds','mix_and_final_route_seconds','exact_seconds']}}
            rows.append(row)
    (root/'failure_decomposition.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    groups=defaultdict(list)
    for r in rows:groups[(r['family'],r['factor'])].append(r)
    summary=[]
    for (family,factor),rr in sorted(groups.items()):
        covered=[r for r in rr if r['all_free_qk_covered']]
        entry={'family':family,'factor':factor,'states':len(rr),'sources':len({r['row_id'] for r in rr})}
        for k in ['free_qk_coverage','all_free_qk_covered','candidate_fraction','oracle_final_equal','mixed_final_equal','candidate_only_final_equal']:entry[k]=float(np.mean([r[k] for r in rr]))
        entry['mixed_final_failure_given_all_qk_covered']=float(np.mean([not r['mixed_final_equal'] for r in covered])) if covered else None
        for k in ['relative_denominator_error_max','relative_denominator_error_head_spread','outside_exact_mass_max','cutoff_margin_min']:
            v=[r[k] for r in rr if r[k] is not None];entry[k+'_quantiles']=np.quantile(v,[0,.5,.9,1]).tolist() if v else None
        entry['coarse_to_exact_score_ratio']=sum(r['warm_coarse_seconds'] for r in rr)/sum(r['exact_seconds'] for r in rr)
        summary.append(entry)
    (root/'failure_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
