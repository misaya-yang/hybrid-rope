"""Paired raw-generation and fixed-query selector diagnostics, no answer extraction."""
import argparse,hashlib,json,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--reference',type=Path);a=p.parse_args()
 out=a.directory;rows=[json.loads(x) for x in (out/'outputs.jsonl').read_text().splitlines()];summary={'output_sha256':hashlib.sha256((out/'outputs.jsonl').read_bytes()).hexdigest(),'methods':{}}
 for method in sorted(set(r['method'] for r in rows)):
  part=[r for r in rows if r['method']==method]
  summary['methods'][method]={'n':len(part),'raw_full_exact_and_eos':sum(r['output_text']==r['expected'] and r['ended_eos'] for r in part),'trimmed_full_exact_and_eos':sum(r['output_text'].strip()==r['expected'] and r['ended_eos'] for r in part),'eos':sum(r['ended_eos'] for r in part),'continuation_seconds':sum(r['continuation_seconds'] for r in part),'diagnostic_seconds':sum(r.get('diagnostic_cost',{}).get('seconds',0) for r in part)}
 if a.reference:
  refs={(r['row_id'],r['method']):r for r in map(json.loads,(a.reference/'outputs.jsonl').read_text().splitlines())}
  summary['reference_replays']=[{'row_id':r['row_id'],'method':r['method'],'same_prefix':r['prefix_sha256']==refs[(r['row_id'],r['method'])]['prefix_sha256'],'raw_generated_ids_equal':r['generated_ids']==refs[(r['row_id'],r['method'])]['generated_ids']} for r in rows if (r['row_id'],r['method']) in refs]
 diag=[]
 for path in out.glob('*_diagnostics.json'):diag.extend(json.loads(path.read_text()))
 if diag:
  flat={};joined={}
  for record in diag:
   for method,heads in record['comparisons'].items():
    d=flat.setdefault(method,{'source_selected':0,'eligible_source_pairs':0,'source_max_rank_top16':0,'inflations':[],'competitor_inflations':[]})
    for head in heads:
     for s in head['sources']:
      if not s.get('eligible'):continue
      d['eligible_source_pairs']+=1;d['source_selected']+=s['selected_remote'];d['source_max_rank_top16']+=s['true_max_rank_min']<=16
      if s['upper_bound_inflation'] is not None:d['inflations'].append(s['upper_bound_inflation'])
      key=(record['layer'],record['position'],head['head'],s['block']);joined.setdefault(key,{})[method]=s
     d['competitor_inflations'].extend(s['upper_bound_inflation'] for s in head['selected_competitors'] if s.get('eligible') and s['upper_bound_inflation'] is not None)
  for d in flat.values():
   for k in ('inflations','competitor_inflations'):
    vals=d.pop(k);d[k+'_mean']=statistics.mean(vals) if vals else None
  summary['diagnostics']={'boundary':'Same failed RoPEMean query trajectory, 5 positions x first/middle/last layers, per-head source cells; these are correlated diagnostics, not independent task wins','methods':flat,'Pair_over_Quest_source_admissions':sum(x['PairEnvelope']['selected_remote'] and not x['Quest']['selected_remote'] for x in joined.values()),'Quest_over_Pair_source_admissions':sum(x['Quest']['selected_remote'] and not x['PairEnvelope']['selected_remote'] for x in joined.values())}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
