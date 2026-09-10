"""Export actual paired rows and a coverage-aware shared-DEV result table."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--results-root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.results_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    panel={r['row_id']:r for r in read_rows(Path(args.data))}
    folders=[root/'pc2_exact_mass_full_dev_v1']+sorted(root.glob('pc2_ten_v1_*'))
    records={};conflicts=[]
    for folder in folders:
        # Failed execution configurations remain archived, but do not own the
        # canonical row when a complete replacement run uses a smaller chunk.
        superseded = {
            'pc2_ten_v1_E01_retrieval_dev_v1',
            'pc2_ten_v1_common48_e07_group_budget_v1',
            'pc2_ten_v1_common48_e07_group_budget_e07retry1024_v1',
        }
        if folder.name in superseded or not (folder/'generations.jsonl').exists():continue
        for r in read_rows(folder/'generations.jsonl'):
            rid=r['row_id']
            if rid not in panel:continue
            mode=r['selector'];seed=r.get('selector_metrics',{}).get('algorithm_seed')
            if seed is not None and seed!=20260910:mode+='__seed'+str(seed)
            if folder.name=='pc2_exact_mass_full_dev_v1' and mode!='exact_mass':continue
            key=(rid,mode)
            if key in records:
                if records[key]['generated_token_ids']!=r['generated_token_ids']:
                    conflicts.append({'row_id':rid,'mode':mode,'first_run':records[key]['source_run'],'other_run':str(folder)})
                continue
            records[key]={**r,'ten_family':panel[rid]['ten_family'],'source_unit':panel[rid]['source_unit'],
                          'source_run':str(folder),'reported_mode':mode}
    exported=[];groups=defaultdict(list)
    for (rid,mode),r in sorted(records.items()):
        metric='official_recall' if 'official_recall' in r else 'qa_f1' if 'qa_f1' in r else 'exact_plus_eos'
        baseline=records.get((rid,'exact_mass'))
        paired=baseline is not None
        row={**r,'primary_metric':metric,'B0_score':float(baseline[metric]) if paired else None,
             'paired_difference':float(r[metric])-float(baseline[metric]) if paired else None,
             'raw_k_read_counters':{k:v for k,v in r.get('selector_metrics',{}).items() if 'raw_k' in k or 'raw_key' in k},
             'raw_v_read_counters':{k:v for k,v in r.get('selector_metrics',{}).items() if 'raw_v' in k},
             'scoring_seconds':None,'routing_seconds':None,'reader_seconds':None,
             'cost_caveat':'wall times include the actual execution/concurrency; unmeasured breakdowns are null, not zero'}
        exported.append(row)
        if paired:groups[(mode,r['ten_family'],metric)].append(row)
    (out/'per_example.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in exported))
    table=[];rng=np.random.default_rng(20260910)
    for (mode,family,metric),rows in sorted(groups.items()):
        differences=np.array([r['paired_difference'] for r in rows]);n=len(rows)
        bootstrap=differences[rng.integers(0,n,(5000,n))].mean(1)
        lo,hi=np.quantile(bootstrap,[.025,.975])
        table.append({'mode':mode,'family':family,'metric':metric,'paired_n':n,'planned_n':12,
                      'status':'COMPLETE_DEV_CELL' if n==12 else 'PARTIAL_DEV_CELL',
                      'candidate_mean':float(np.mean([float(r[metric]) for r in rows])),
                      'B0_mean':float(np.mean([r['B0_score'] for r in rows])),
                      'paired_difference':float(differences.mean()),'wins':int((differences>0).sum()),
                      'losses':int((differences<0).sum()),'ties':int((differences==0).sum()),
                      'dev_bootstrap_low':float(lo),'dev_bootstrap_high':float(hi),
                      'uncertainty':'descriptive DEV bootstrap conditional on observed sources; not selection-adjusted confirmation or noninferiority',
                      'metadata_bytes_max':max(r.get('selector_metrics',{}).get('max_metadata_bytes',0) for r in rows),
                      'mean_total_seconds':float(np.mean([r['total_seconds'] for r in rows])),
                      'eos_rate':float(np.mean([r['ended_with_eos'] for r in rows]))})
    primary=('exact_mass','e01_int8','e02_nonlinear','e03_temporal','e04_empirical','e05_two_component',
             'e06_residual_sampling','e07_group_budget','e08_tail_value','e09_local','e10_cutoff')
    for mode in primary:
        for family in ('retrieval','occurrence','multi_hop','natural_qa'):
            if any(r['mode']==mode and r['family']==family for r in table):continue
            metric='official_recall' if family=='retrieval' else 'exact_plus_eos' if family=='occurrence' else 'qa_f1'
            table.append({'mode':mode,'family':family,'metric':metric,'paired_n':0,'planned_n':12,
                          'status':'NO_LOCAL_RESULT_YET','candidate_mean':None,'B0_mean':None,
                          'paired_difference':None,'wins':0,'losses':0,'ties':0,'dev_bootstrap_low':None,
                          'dev_bootstrap_high':None,'uncertainty':'no paired local result; inspect live queue for running/pending status',
                          'metadata_bytes_max':None,'mean_total_seconds':None,'eos_rate':None})
    if table:
        with (out/'summary.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (out/'export_status.json').write_text(json.dumps({'exported_rows':len(exported),'conflicting_replicas':conflicts,
                                                    'scope':'available common48 DEV only; missing rows remain incomplete'},indent=2)+'\n')
    print(json.dumps({'exported_rows':len(exported),'table_rows':len(table),'conflicts':len(conflicts)}))


if __name__=='__main__':main()
