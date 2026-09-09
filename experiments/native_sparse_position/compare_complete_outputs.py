"""Audit exact paired generation contracts without selective row filtering."""
import argparse,hashlib,json,math
from collections import defaultdict
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
 status=json.loads((root/'status.json').read_text());raw=(root/'outputs.jsonl').read_bytes();records=[json.loads(x) for x in raw.splitlines()];groups=defaultdict(dict)
 for r in records:
  if r['method'] in groups[r['row_id']]:raise ValueError('Duplicate row/method')
  groups[r['row_id']][r['method']]=r
 methods=status['methods'];complete=[(rid,g) for rid,g in groups.items() if set(g)==set(methods)]
 if status['status']=='COMPLETE' and len(complete)!=status['questions']:raise ValueError('Missing paired outputs')
 for rid,g in complete:
  for field in ('prefix_sha256','input_tokens','prefix_tokens','question_ingest_tokens','expected'):
   if len({str(r[field]) for r in g.values()})!=1:raise ValueError(f'{rid} differs in {field}')
 ok=lambda r: r['output_text']==r['expected'] and r['ended_eos']
 n=len(complete);summary={'status':status['status'],'complete_paired_rows':n,'expected_rows':status['questions'],'outputs_sha256':hashlib.sha256(raw).hexdigest(),'metric':'raw entire body exactly equals expected string, terminal EOS required; no whitespace trimming','counts':{},'paired':{},'rows':[]}
 for m in methods:
  rows=[g[m] for _,g in complete];summary['counts'][m]={'correct':sum(ok(r) for r in rows),'eos':sum(r['ended_eos'] for r in rows),'n':n,'continuation_seconds':sum(r['continuation_seconds'] for r in rows)}
 for m in ('PostMetric4','PreMetric4'):
  if m not in methods:continue
  for baseline in methods:
   if baseline==m:continue
   wins=sum(ok(g[m]) and not ok(g[baseline]) for _,g in complete);losses=sum(ok(g[baseline]) and not ok(g[m]) for _,g in complete);d=wins+losses
   prob=min(1.,2*sum(math.comb(d,i) for i in range(min(wins,losses)+1))/2**d) if d else 1.
   summary['paired'][m+'_vs_'+baseline]={'wins':wins,'losses':losses,'ties':n-d,'raw_id_equal':sum(g[m]['generated_ids']==g[baseline]['generated_ids'] for _,g in complete),'two_sided_sign_test_p_unadjusted':prob,'boundary':'conditional on this synthetic key/placement assay, unadjusted multiple comparisons; not independent natural documents'}
 for rid,g in complete:summary['rows'].append({'row_id':rid,'expected':next(iter(g.values()))['expected'],'correct':{m:ok(g[m]) for m in methods},'outputs':{m:g[m]['output_text'] for m in methods}})
 (root/'paired_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
