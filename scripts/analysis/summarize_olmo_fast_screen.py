"""Rescore complete OLMo screen artifacts and preserve paired evidence."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.experiments.olmo_fast_screen.ruler_bench import score, summarize, verdict


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arm(root, run, name):
    path=root/run/(name+'.jsonl')
    receipt=json.loads((root/run/(name+'.json')).read_text())
    if receipt['status']!='COMPLETE' or sha(path)!=receipt['raw_sha256']:
        raise ValueError('incomplete or changed raw result: '+str(path))
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if receipt['row_ids']!=[r['row_id'] for r in rows]:raise ValueError('row sequence drift')
    for row in rows:
        if abs(score(row,row['output_text'])-row['correct'])>1e-12:
            raise ValueError('saved score differs from official rescore')
    return rows,dict(raw_path=str(path),raw_sha256=sha(path),receipt_sha256=sha(root/run/(name+'.json')),
        table_identity={k:v for k,v in receipt['table'].items() if k in ('tensor_sha256','gain','gain_by_slot')},
        elapsed_seconds=receipt['elapsed_seconds'],summary=summarize(rows))


def compare(c,b):
    result=verdict(c,b)
    by_id={r['row_id']:r for r in b}
    result['paired_by_length']={}
    for cap in sorted({r['length_cap'] for r in c}):
        changes=[r['correct']-by_id[r['row_id']]['correct'] for r in c if r['length_cap']==cap]
        result['paired_by_length'][str(cap)]=dict(wins=sum(x>0 for x in changes),
            ties=sum(x==0 for x in changes),losses=sum(x<0 for x in changes))
    result['loss_rows']=[dict(row_id=r['row_id'],references=r['references'],
        baseline_score=by_id[r['row_id']]['correct'],candidate_score=r['correct'],
        baseline_text=by_id[r['row_id']]['output_text'],candidate_text=r['output_text'])
        for r in c if r['correct']<by_id[r['row_id']]['correct']]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.root
    specs=[('development','run_ruler_01',['MrPro','MrProBM']),
           ('seed_replication','run_holdout_01',['MrPro','MrProBM']),
           ('gain_followup','run_gain_01',['MrProBM','BMSelectiveGain','BMUniformMatchedGain']),
           ('scale8','run_s8_01',['MrPro','MrProBM']),
           ('existing_controls','run_controls_01',['MrUni','OfficialYaRN']),
           ('scale_cap','run_cap_01',['BMCappedS4','BMFreq8Gain4'])]
    report=dict(status='RESCORED_COMPLETED_ARMS',experiments={},
        scope='Small six-task RULER panels. Scores use official substring matching; no full RULER or cross-model claim.',
        inference_limits=['Prompts, not individual reference strings, are the units.',
            'QA source questions can recur across lengths; lengths are reported separately.',
            'Development-selected improvements require independent input confirmation.',
            'Aggregate gain is not per-task or per-row native retention.',
            'No geometric or gain mechanism is established by score differences alone.'])
    raw={}
    for label,run,names in specs:
        if not all((root/run/(n+'.json')).exists() for n in names):continue
        records={};arms={}
        for name in names:records[name],arms[name]=load_arm(root,run,name)
        raw[label]=records
        report['experiments'][label]=dict(arms=arms,reference=names[0],
            comparisons={name:compare(records[name],records[names[0]]) for name in names[1:]})
    if 'existing_controls' in raw and 'seed_replication' in raw:
        bm=raw['seed_replication']['MrProBM']
        report['bm_vs_existing_controls']={name:compare(bm,rows) for name,rows in raw['existing_controls'].items()}
    if 'scale_cap' in raw and 'scale8' in raw:
        report['scale_cap_vs_original_s8']={name:compare(rows,raw['scale8']['MrProBM'])
                                          for name,rows in raw['scale_cap'].items()}
    if (root/'run_native_01/Native.json').exists():
        rows,receipt=load_arm(root,'run_native_01','Native')
        report['native_short_reference']=receipt
    if 'development' in raw and 'gain_followup' in raw:
        a,b=raw['development']['MrProBM'],raw['gain_followup']['MrProBM']
        report['scalar_runtime_replay_exact']=len(a)==len(b) and all(
            x['row_id']==y['row_id'] and x['generated_ids']==y['generated_ids'] for x,y in zip(a,b))
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({label:{n:a['summary']['by_length'] for n,a in e['arms'].items()}
                      for label,e in report['experiments'].items()},indent=2))


if __name__=='__main__':main()
