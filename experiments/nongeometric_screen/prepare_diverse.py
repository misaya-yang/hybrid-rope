"""Stratify only the new QA sample selection; reuse official generation/scoring.

No candidate is evaluated during preparation. Other task inputs are copied
verbatim from the new-seed cohort. First sixteen QA rows span sixteen articles.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

import numpy as np
import yaml
from transformers import AutoTokenizer


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def main(source,out,upstream,old):
    source,out,upstream,old=map(Path,(source,out,upstream,old))
    manifest=json.loads((source/'manifest.json').read_text())
    model=manifest['model_path'];tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    old_questions={r['prompt_text'].rsplit('Question:',1)[-1].split('<|im_end|>')[0].strip()
                   for r in rows(old/'prompts.jsonl') if r['row_id'].startswith('qa_1_')}
    squad=json.loads((upstream/'scripts/data/synthetic/json/squad.json').read_text())
    metadata=[];by_article={};excluded=set()
    for article in squad['data']:
        candidates=[]
        for paragraph_index,para in enumerate(article['paragraphs']):
            for q in para['qas']:
                if q['is_impossible']:continue
                meta=dict(index=len(metadata),title=article['title'],paragraph=paragraph_index,question=q['question'],id=q['id'])
                metadata.append(meta);candidates.append(meta)
                if q['question'] in old_questions:excluded.add(article['title'])
        by_article[article['title']]=candidates
    titles=sorted((t for t,rs in by_article.items() if t not in excluded and len({r['paragraph'] for r in rs})>=2),
        key=lambda t:digest([20260910,t]))[:16]
    if len(titles)!=16:raise ValueError('need sixteen eligible source articles')
    selected=[[],[]]
    for title in titles:
        candidates=sorted(by_article[title],key=lambda r:digest([20260910,r['id']]))
        first=candidates[0];second=next(r for r in candidates if r['paragraph']!=first['paragraph'])
        selected[0].append(first);selected[1].append(second)
    selected=selected[0]+selected[1]
    assert not any(r['question'] in old_questions for r in selected)
    out.mkdir(parents=True,exist_ok=False)
    all_rows=[r for r in rows(source/'screen.jsonl') if r['task']!='qa_1']
    prompt_rows=[r for r in rows(source/'prompts.jsonl') if not r['row_id'].startswith('qa_1_')]
    config=yaml.safe_load((upstream/'scripts/synthetic.yaml').read_text())['qa_1']
    # Execute the unchanged module without invoking its main; only reorder QAS
    # according to the outcome-blind, article-stratified selection above.
    old_argv,old_path,old_cwd=sys.argv,sys.path[:],os.getcwd()
    try:
        sys.path.insert(0,str(upstream/'scripts/data/synthetic'));os.chdir(upstream)
        const=runpy.run_path(str(upstream/'scripts/data/synthetic/constants.py'))['TASKS']['qa']
        template=tok.apply_chat_template([{'role':'user','content':const['template']}],tokenize=False,add_generation_prompt=True)+const.get('answer_prefix','')
        budget=const['tokens_to_generate']
        for cap in (32768,131072):
            sys.argv=[str(upstream/'scripts/data/synthetic/qa.py'),'--save_dir',str(out/'source'/str(cap)),
                '--save_name','qa_1','--tokenizer_path',model,'--tokenizer_type','hf','--max_seq_length',str(cap),
                '--tokens_to_generate',str(budget),'--num_samples','32','--pre_samples','0','--random_seed','20260910',
                '--template',template,'--dataset',config['args']['dataset']]
            with (out/f'qa_{cap}.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                ns=runpy.run_path(sys.argv[0],run_name='ruler_qa_stratified')
                fn=ns['generate_samples'];g=fn.__globals__;original_qas=g['QAS']
                for m in selected:
                    if original_qas[m['index']]['query']!=m['question']:raise ValueError('official question order mismatch')
                g['QAS']=[original_qas[m['index']] for m in selected]
                generated=fn(32,cap,str(out/'source'/str(cap)))
            for i,(raw,meta) in enumerate(zip(generated,selected)):
                text=raw['input']+raw.get('answer_prefix','');ids=tok.encode(text,add_special_tokens=False)
                if len(ids)+budget>cap:raise ValueError('stratified input exceeds unchanged budget')
                row=dict(row_id=f'qa_1_{cap}_{i}',task='qa_1',family='qa',upstream_index=meta['index'],
                    length_cap=cap,prompt_ids=ids,prompt_sha256=digest(ids),input_tokens=len(ids),references=raw['outputs'],
                    max_new_tokens=budget,qa_source=meta)
                all_rows.append(row);prompt_rows.append(dict(row_id=row['row_id'],prompt_text=text,references=raw['outputs'],upstream_index=meta['index'],qa_source=meta))
            print(json.dumps(dict(qa_cap=cap,rows=len(generated),articles=len(titles))),flush=True)
    finally:sys.argv,sys.path=old_argv,old_path;os.chdir(old_cwd)
    tasks=manifest['tasks'];order=lambda r:(r.get('length_cap',int(r['row_id'].split('_')[-2])),tasks.index(r.get('task','qa_1') if 'task' in r else r['row_id'].rsplit('_',2)[0]),int(r['row_id'].rsplit('_',1)[1]))
    all_rows.sort(key=order)
    index={r['row_id']:i for i,r in enumerate(all_rows)};prompt_rows.sort(key=lambda r:index[r['row_id']])
    for name,rs in [('screen.jsonl',all_rows),('prompts.jsonl',prompt_rows)]:
        with (out/name).open('w') as f:
            for r in rs:f.write(json.dumps(r)+'\n')
    for name in ('tables.json','queue.json','generation_config.json','qualification.jsonl'):shutil.copyfile(source/name,out/name)
    manifest.update(status='PREPARED_STRATIFIED_HOLDOUT_GPU_NOT_RUN',screen_rows=len(all_rows),
        screen_input_tokens=sum(r['input_tokens'] for r in all_rows),screen_min_max_tokens=[min(r['input_tokens'] for r in all_rows),max(r['input_tokens'] for r in all_rows)],
        row_order=[r['row_id'] for r in all_rows],prompt_collection_sha256=digest([r['prompt_ids'] for r in all_rows]),
        qa_sampling=dict(articles=titles,excluded_development_articles=sorted(excluded),selected=selected,
            first_16='one question from each of sixteen articles',next_16='second question from a different paragraph of each article'),
        reused_nonqa_manifest_sha256=sha(source/'manifest.json'),selection_code_sha256=sha(__file__),
        scope='New-seed six-task RULER subset; only QA sampling is article-stratified; official generation/scoring and model/decoder unchanged')
    manifest['prepared_files']={name:sha(out/name) for name in ('screen.jsonl','prompts.jsonl','tables.json','queue.json','generation_config.json','qualification.jsonl')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status=manifest['status'],rows=len(all_rows),articles=titles)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--upstream',required=True);p.add_argument('--old',required=True)
    a=p.parse_args();main(a.source,a.out,a.upstream,a.old)
