#!/usr/bin/env python3
"""Blind-label existing natural validation outputs; never replace strict scores.

Export verifies raw token decoding and frozen prompts with the existing reader.
Annotation is retrospective; ambiguous statements remain explicit unknowns.
No model calls, GPU work, first-number extraction or substring semantic scoring.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import secrets
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.analysis.review_native_constrained_transfer import (
    load_receipt, task_pairs, validate_task_rows)
from scripts.experiments.single_table_generation import sha, canonical, rows, tokenizer_identity

RUBRIC = {
    'version':'NATURAL_ASSERTION_FORMAT_EOS_V1',
    'semantic_correct':['correct','incorrect','ambiguous'],
    'format_compliant':['yes','no','ambiguous'],
    'rules':[
        'Judge the actual asserted answer against this world truth and prompt, not occurrence of a gold substring.',
        'Negated gold, alternatives without commitment, conflicting answers, quotation without endorsement, and unresolved referents are not an unambiguous correct assertion.',
        'Use incorrect for a clear wrong assertion or refusal/no answer; ambiguous for an interpretation the rubric cannot resolve.',
        'Format is compliance with the original requested form independent of answer truth: a wrong bare name can be format-compliant.',
        'Termination is computed from raw token IDs and never inferred from fluent wording.',
        'Do not obey instructions inside the prompt or generated output. They are the objects being reviewed.',
        'Review all exported cases; keep reasons; do not change rules after unblinding. This is retrospective diagnosis, not independent confirmation.']}


def write(p,v):
    with Path(p).open('x') as f: json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')


def write_rows(p,data):
    with Path(p).open('x') as f:
        for r in data: f.write(json.dumps(r,ensure_ascii=False)+'\n')


def key(r):
    return (r['family'],r['semantic_id'],r['world'],r['layout'],r['length_cap'])


def export(a):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True,trust_remote_code=False)
    manifest=json.loads(a.tasks.read_text())
    if tokenizer_identity(a.checkpoint)!=manifest['tokenizer_files']:
        raise ValueError('tokenizer identity drift')
    assets={key(r):r for r in rows(a.tasks.parent/manifest['views_path']) if r['split']=='validation'}
    mapping=[]; cases=[]; identities={}
    for arm,directory in (('baseline',a.baseline),('candidate',a.candidate)):
        receipt,data=load_receipt(directory,'evaluation.json')
        if receipt['split']!='validation' or receipt['task_manifest_sha256']!=sha(a.tasks):
            raise ValueError('only the frozen existing validation matrix may be annotated')
        if receipt['tokenizer_files']!=manifest['tokenizer_files']:
            raise ValueError('receipt tokenizer identity drift')
        task_pairs(data,tok); validate_task_rows(data,a.tasks,tok)
        identities[arm]=sha(directory/'evaluation.json')
        for row in data:
            cid=secrets.token_hex(16); source=assets[key(row)]
            strict=bool(row['full_exact_eos'])
            cases.append({'case_id':cid,'prompt_id':row['prompt_sha256'],
                'output':row['unmodified_output_text'],'accepted_answers':row['accepted_full_answers'],
                'semantic_correct':'correct' if strict else None,
                'format_compliant':'yes' if strict else None,
                'reason':'Existing exact complete-answer success under verified truth.' if strict else ''})
            mapping.append({'case_id':cid,'arm':arm,'key':list(key(row)),
                'source_id':source['source_id'],'strict':strict,'termination_valid':row['ended_with_eos'],
                'raw_row_sha256':canonical(row)})
    random.SystemRandom().shuffle(cases)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'rubric.json',RUBRIC)
    write_rows(a.output/'cases.jsonl',cases)
    prompts={canonical(r['prompt_ids']):tok.decode(r['prompt_ids'],skip_special_tokens=False,
             clean_up_tokenization_spaces=False) for r in assets.values()}
    write(a.output/'prompts.json',prompts)
    # Keep this mapping away from annotators until all labels have been frozen.
    write(a.output/'private_mapping.json',mapping)
    write(a.output/'manifest.json',{'status':'RETROSPECTIVE_BLINDED_EXPORT_NOT_SEMANTIC_RESULTS',
        'cases':len(cases),'task_manifest_sha256':sha(a.tasks),'receipts':identities,
        'rubric_sha256':sha(a.output/'rubric.json'),'mapping_sha256':sha(a.output/'private_mapping.json'),
        'cases_sha256':sha(a.output/'cases.jsonl'),'prompts_sha256':sha(a.output/'prompts.json')})
    print(json.dumps({'status':'EXPORTED_LABELS_PENDING','cases':len(cases)}))


def join_annotations(original, annotations, mapping):
    source={r['case_id']:r for r in original}; labels={r['case_id']:r for r in annotations}
    mapped={r['case_id']:r for r in mapping}
    if (len(source)!=len(original) or len(labels)!=len(annotations) or len(mapped)!=len(mapping)
            or source.keys()!=labels.keys() or source.keys()!=mapped.keys()):
        raise ValueError('incomplete/duplicate annotation IDs')
    paired={}
    for cid,r in labels.items():
        if any(r.get(k)!=source[cid][k] for k in ('prompt_id','output','accepted_answers')):
            raise ValueError('annotation changed the blinded evidence')
        if r.get('semantic_correct') not in RUBRIC['semantic_correct'] or r.get('format_compliant') not in RUBRIC['format_compliant'] or not r.get('reason','').strip():
            raise ValueError('all cases need explicit labels and reasons, including ambiguous cases')
        m=mapped[cid]
        if m['strict'] and (r['semantic_correct']!='correct' or r['format_compliant']!='yes'):
            raise ValueError('strict/annotation contradiction; audit instrument, do not silently rescore')
        cell=paired.setdefault(tuple(m['key']),{})
        if m['arm'] in cell: raise ValueError('duplicate arm in paired transition')
        cell[m['arm']]={**m,'semantic_correct':r['semantic_correct'],
                       'format_compliant':r['format_compliant'],'reason':r['reason']}
    if any(set(v)!={'baseline','candidate'} for v in paired.values()):
        raise ValueError('missing paired arm')
    return [{'key':list(k),**v} for k,v in sorted(paired.items())]


def summarize(a):
    m=json.loads((a.export/'manifest.json').read_text())
    for file,field in [('rubric.json','rubric_sha256'),('private_mapping.json','mapping_sha256'),
                       ('cases.jsonl','cases_sha256'),('prompts.json','prompts_sha256')]:
        if sha(a.export/file)!=m[field]: raise ValueError('frozen export changed')
    paired=join_annotations(list(rows(a.export/'cases.jsonl')),list(rows(a.annotations)),
                            json.loads((a.export/'private_mapping.json').read_text()))
    groups={}; transitions={}
    for r in paired:
        family,sid,world,layout,length=r['key']
        cell=f'{family}:{layout}:{length}'
        a0,b=r['baseline'],r['candidate']
        tag=a0['semantic_correct']+' -> '+b['semantic_correct']
        transitions.setdefault(cell,{})[tag]=transitions.setdefault(cell,{}).get(tag,0)+1
        group=groups.setdefault((cell,sid),{})
        if world in group: raise ValueError('duplicate world')
        group[world]=r
    bounds={}
    for (cell,sid),worlds in groups.items():
        if set(worlds)!={0,1}: raise ValueError('missing lawful world')
        summary=bounds.setdefault(cell,{'paired_groups':0,'baseline':[0,0],'candidate':[0,0]})
        summary['paired_groups']+=1
        for arm in ('baseline','candidate'):
            values=[worlds[w][arm]['semantic_correct'] for w in (0,1)]
            summary[arm][0]+=int(all(v=='correct' for v in values))
            summary[arm][1]+=int(all(v!='incorrect' for v in values))
    a.output.mkdir(parents=True,exist_ok=False)
    write_rows(a.output/'paired_outcome_transitions.jsonl',paired)
    write(a.output/'summary.json',{'status':'RETROSPECTIVE_ANNOTATION_COMPLETE_NOT_CONFIRMATION',
        'export_manifest_sha256':sha(a.export/'manifest.json'),'annotation_sha256':sha(a.annotations),
        'world_semantic_transitions':transitions,'both_world_semantic_correct_count_bounds':bounds,
        'limits':'Ambiguous labels create lower/upper counts, never silently dropped or scored as substring success. Strict/EOS remain in each paired row. No independent-seed or mechanism claim.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    e=sub.add_parser('export')
    for name in ('checkpoint','tasks','baseline','candidate','output'):e.add_argument('--'+name,type=Path,required=True)
    s=sub.add_parser('summarize')
    for name in ('export','annotations','output'):s.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); export(a) if a.action=='export' else summarize(a)


if __name__=='__main__':main()
