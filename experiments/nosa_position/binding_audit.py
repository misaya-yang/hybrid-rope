"""Audit explicit key/value assertions separately from official set recall.

Only outputs that name a queried key or explicitly say 'respectively' are
interpreted as binding claims. Unordered or unparsed answers are left unknown,
not relabelled wrong. Source needles, not another model's answer, own the gold.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def source_bindings(row):
    question=re.findall(r'What are all the special magic numbers for (.*?) mentioned in the provided text\?',row['prompt'],re.I)
    if len(question)!=1:raise ValueError('ambiguous query list')
    keys=[re.sub(r'^and\s+','',x.strip().lower()) for x in question[0].split(',')]
    if len(set(keys))!=len(keys):raise ValueError('duplicate query keys')
    needles=re.findall(r'special magic numbers? for ([^\n]+?) is:\s*(\d+)\.',row['prompt'],re.I)
    gold={key:set() for key in keys}
    for key,value in needles:
        key=key.strip().lower()
        if key in gold:gold[key].add(value)
    if not all(gold.values()) or set().union(*gold.values())!=set(row['references']):
        raise ValueError('source needle values do not reproduce frozen references')
    return keys,gold


def inspect_answer(row,result):
    keys,gold=source_bindings(row);text=result['output_text']
    claims=[]
    for key in keys:
        pattern=r'(?:^|\n)\s*(?:[-*]\s*)?'+re.escape(key)+r'\s*:\s*(\d+)'
        claims.extend((key,value) for value in re.findall(pattern,text,re.I))
    mode='explicit_labels' if claims else 'not_interpreted'
    cardinality_error=None
    if not claims and re.search(r'\brespectively\b',text,re.I):
        values=[v.replace(',','') for v in re.findall(r'(?<!\d)(?:\d{1,3}(?:,\d{3})+|\d+)(?!\d)',text)]
        if len(values)==len(keys):
            claims=list(zip(keys,values));mode='explicit_respectively'
        else:
            mode='explicit_cardinality_error'
            cardinality_error={'queried_objects':len(keys),'asserted_values':len(values),'values':values}
    wrong=[{'key':key,'asserted':value,'source_values':sorted(gold[key])} for key,value in claims if value not in gold[key]]
    correct_keys={key for key,value in claims if value in gold[key]}
    complete=False if cardinality_error else ((len(correct_keys)==len(keys) and not wrong) if claims else None)
    return {'row_id':row['row_id'],'selector':result['selector'],'official_recall':result['official_recall'],
        'ended_with_eos':result['ended_with_eos'],'output_text':text,'query_order':keys,
        'source_bindings':{k:sorted(v) for k,v in gold.items()},'interpretation':mode,
        'binding_cardinality_error':cardinality_error,
        'claims':[{'key':key,'value':value} for key,value in claims],
        'correctly_asserted_key_count':len(correct_keys) if claims else None,
        'queried_key_count':len(keys),'wrong_explicit_claims':wrong,
        'all_bindings_correct':complete,'all_bindings_correct_plus_eos':complete and result['ended_with_eos'] if complete is not None else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(); data=Path(args.data);sha=hashlib.sha256(data.read_bytes()).hexdigest()
    rows={r['row_id']:r for r in map(json.loads,data.read_text().splitlines()) if r['task']=='niah_multiquery' and r['split']=='dev'}
    found={};source_runs={}
    for run in map(Path,args.runs):
        contract=json.loads((run/'contract.json').read_text())
        if contract['data_sha256']!=sha:raise ValueError('source data identity mismatch')
        for result in map(json.loads,(run/'generations.jsonl').read_text().splitlines()):
            rid=result['row_id']
            if rid not in rows:continue
            key=(rid,result['selector'])
            if key in found and found[key]['generated_token_ids']!=result['generated_token_ids']:
                raise ValueError('conflicting repeated output')
            found[key]=result;source_runs.setdefault(key,[]).append(str(run))
    audited=[{**inspect_answer(rows[rid],result),'source_runs':source_runs[rid,arm]} for (rid,arm),result in sorted(found.items())]
    summary={}
    for arm in sorted({r['selector'] for r in audited}):
        selected=[r for r in audited if r['selector']==arm]
        summary[arm]={'input_count':len(selected),'official_recall_mean':sum(r['official_recall'] for r in selected)/len(selected),
            'explicitly_interpretable_outputs':sum(r['interpretation']!='not_interpreted' for r in selected),
            'outputs_with_wrong_explicit_binding':sum(bool(r['wrong_explicit_claims']) for r in selected),
            'outputs_with_invalid_binding_cardinality':sum(r['binding_cardinality_error'] is not None for r in selected),
            'outputs_with_all_bindings_correct_plus_eos':sum(r['all_bindings_correct_plus_eos'] is True for r in selected)}
    report={'scope':'post-hoc explicit-assertion audit, not a replacement official metric; unparsed answers unknown; differing sample counts not pooled',
        'data_sha256':sha,'summary':summary,'rows':audited}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
