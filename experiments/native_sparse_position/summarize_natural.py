"""Recompute whole-completion endpoints and paired changes from retained outputs."""
import argparse,collections,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.eval.longbench_metrics import qa_f1_score,normalize_text

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    status=json.loads((a.run/'status.json').read_text())
    if status['status']!='COMPLETE':raise ValueError('Run is incomplete')
    rows=[json.loads(s) for s in (a.run/'outputs.jsonl').read_text().splitlines()]
    if len(rows)!=status['outputs']:raise ValueError('Output count drift')
    grouped=collections.defaultdict(dict)
    for r in rows:
        if r['method'] in grouped[r['row_id']]:raise ValueError('Duplicate output')
        grouped[r['row_id']][r['method']]=r
        if abs(qa_f1_score(r['output_text'],r['references'])-r['qa_f1'])>1e-12:raise ValueError('F1 drift')
        if any(r['output_text'].strip()==x for x in r['references'])!=r['text_exact']:raise ValueError('Exact drift')
        if any(normalize_text(r['output_text'])==normalize_text(x) for x in r['references'])!=r['normalized_exact']:raise ValueError('Normalized exact drift')
    if len(grouped)!=status['questions']:raise ValueError('Question count drift')
    for v in grouped.values():
        if set(v)!=set(status['methods']):raise ValueError('Missing method')
        for field in ('prefix_sha256','references','input_tokens','question_ingest_tokens'):
            if any(r[field]!=v['Dense'][field] for r in v.values()):raise ValueError(f'Paired contract drift: {field}')
    totals={}
    for task in ['ALL']+sorted({r['task'] for r in rows}):
        totals[task]={}
        for m in status['methods']:
            rs=[r for r in rows if r['method']==m and (task=='ALL' or r['task']==task)]
            totals[task][m]={'n':len(rs),'whole_answer_f1':sum(r['qa_f1'] for r in rs)/len(rs),
              'trimmed_exact_and_eos':sum(r['text_exact'] and r['ended_eos'] for r in rs),
              'normalized_exact_and_eos':sum(r['normalized_exact'] and r['ended_eos'] for r in rs),
              'eos':sum(r['ended_eos'] for r in rs)}
    pairs=[]
    for row_id,v in grouped.items():
        pairs.append({'row_id':row_id,'task':v['Dense']['task'],'identical_R_N_tokens':v['RoPEOracle']['generated_ids']==v['NoPEOracle']['generated_ids'],
          'N_minus_R_f1':v['NoPEOracle']['qa_f1']-v['RoPEOracle']['qa_f1'],
          'R_minus_dense_f1':v['RoPEOracle']['qa_f1']-v['Dense']['qa_f1'],
          'N_minus_dense_f1':v['NoPEOracle']['qa_f1']-v['Dense']['qa_f1']})
    summary={'evidence':'24-case development QA, custom prompt, entire completion scored; not official full benchmark or efficiency evidence',
      'status_sha256':hashlib.sha256((a.run/'status.json').read_bytes()).hexdigest(),'totals':totals,'pairs':pairs,
      'R_N_identical_tokens':sum(p['identical_R_N_tokens'] for p in pairs),
      'N_vs_R_F1':{k:sum((p['N_minus_R_f1']>1e-12 if k=='wins' else p['N_minus_R_f1']< -1e-12 if k=='losses' else abs(p['N_minus_R_f1'])<=1e-12) for p in pairs) for k in ('wins','losses','ties')}}
    (a.run/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='pairs'},indent=2))
if __name__=='__main__':main()
