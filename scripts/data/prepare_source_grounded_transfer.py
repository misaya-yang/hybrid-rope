#!/usr/bin/env python3
"""Source-grounded counterfactual QA and independent Native replay assets.

Controlled transformations of official annotated facts, NOT untouched 2Wiki or
SQuAD benchmark scores. No model outcome chooses a table, gain or loss setting.
"""
from __future__ import annotations
import argparse
import copy
import json
import random
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.single_table_generation import (
    canonical,sha,write_json,rows,tokenizer_identity,load_runtime,guard_resources,greedy,gold_prefix_trace,WEIGHT_SHA)

FAMILIES=('single_evidence','double_evidence','binding')
QUOTAS={'single_evidence':64,'double_evidence':32,'binding':32}
RELATIONS={'place of birth','country of citizenship','director','author','composer','spouse','father','mother','country','country of origin'}


def fold(source):
    number=int(canonical(source)[:8],16)%100
    return 'train' if number<60 else 'calibration' if number<70 else 'validation' if number<85 else 'test'


def task_fold(source):
    # Task data has no calibration split: its unused bucket is held out as test.
    value=fold(source)
    return 'test' if value=='calibration' else value


def occurrences(text,entity):
    return list(re.finditer(r'(?<!\w)'+re.escape(entity)+r'(?!\w)',text,flags=re.I))


def relabel(text,old,new):
    if not old or not new or old.casefold()==new.casefold() or not occurrences(text,old):
        raise ValueError('relabeling requires a present, distinct entity')
    return re.sub(r'(?<!\w)'+re.escape(old)+r'(?!\w)',lambda _:new,text,flags=re.I)


def truth_answer(proof,world):
    facts=proof['world_facts'][world]; query=proof['query']
    current=query['subject']
    for relation in query['relations']:
        matches={obj for sub,rel,obj in facts if sub.casefold()==current.casefold() and rel==relation}
        if len(matches)!=1: raise ValueError('truth graph is ambiguous or disconnected')
        current=next(iter(matches))
    return current


def aliases(answer):
    return [answer] if answer.endswith('.') else [answer,answer+'.']


def question(subject,relation,split):
    forms={'train':'According to the passages, what is the {r} of {s}?',
           'validation':'Using only the passages, identify {s}\'s {r}.',
           'test':'What do the passages name as the {r} for {s}?'}
    return forms[split].format(r=relation,s=subject)


def prompt(tokenizer,contexts,q):
    body='Use only the supplied passages, even if they describe a counterfactual world. Give only the answer, without explanation.\n\n'
    body+='\n\n'.join(contexts)+'\n\nQuestion: '+q
    return tokenizer.apply_chat_template([{'role':'user','content':body}],tokenize=True,add_generation_prompt=True,return_dict=False)


def candidate(tokenizer,family,split,source_ids,q,contexts0,contexts1,proof,lineage):
    answers=[truth_answer(proof,w) for w in (0,1)]
    if answers[0].casefold()==answers[1].casefold(): raise ValueError('identical world answers')
    if any(occurrences(q,answer) for answer in answers): raise ValueError('answer entity occurs in fixed question')
    result={'family':family,'split':split,'source_ids':sorted(source_ids),'question':q,
            'proof':proof,'template_lineage':lineage,'worlds':[]}
    for w,contexts in enumerate((contexts0,contexts1)):
        if not occurrences('\n'.join(contexts),answers[w]): raise ValueError('answer absent from source text')
        ids=prompt(tokenizer,contexts,q)
        target=tokenizer.encode(answers[w],add_special_tokens=False)+[tokenizer.eos_token_id]
        if len(ids)+max(64,len(target))>2048: raise ValueError('compact too long')
        result['worlds'].append({'contexts':contexts,'compact_prompt_ids':ids,'answer':answers[w],
                                'target_ids':target,'accepted_full_answers':aliases(answers[w])})
    result['semantic_id']=canonical({k:result[k] for k in ('family','source_ids','question','proof')})
    return result


def build_candidates(args):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    archive=zipfile.ZipFile(args.two_wiki_zip)
    # Official train is for adaptation; official dev supplies held-out task rows.
    datasets={name:json.load(archive.open(name+'.json')) for name in ('train','dev')}
    triples=[]; doubles=[]
    for partition,data in datasets.items():
        for row in data:
            paragraphs={title:title+'\n'+' '.join(sentences) for title,sentences in row['context']}
            titles=list(dict.fromkeys(title for title,_ in row['supporting_facts']))
            if not titles or any(title not in paragraphs for title in titles): continue
            facts=row['evidences']
            for subject,relation,obj in facts:
                if relation not in RELATIONS or not 3<=len(obj)<=80 or occurrences(row['question'],obj): continue
                matching=[title for title in titles if occurrences(paragraphs[title],subject) and occurrences(paragraphs[title],obj)]
                if len(matching)!=1: continue
                title=matching[0]
                triples.append({'partition':partition,'source':'2wiki:'+title,'raw_id':row['_id'],
                                'fact':[subject,relation,obj],'context':paragraphs[title]})
            if (row['type']=='compositional' and len(facts)==2 and facts[0][2].casefold()==facts[1][0].casefold()
                    and facts[1][2].casefold()==row['answer'].casefold() and 3<=len(row['answer'])<=80
                    and not occurrences(row['question'],row['answer'])):
                text='\n'.join(paragraphs[t] for t in titles)
                if occurrences(text,row['answer']):
                    doubles.append({'partition':partition,'sources':['2wiki:'+t for t in titles],
                                    'raw_id':row['_id'],'facts':facts,'contexts':[paragraphs[t] for t in titles],
                                    'question':row['question']})
    donors={}
    for r in triples:
        if r['partition']=='train': donors.setdefault(r['fact'][1],[]).append(r['fact'][2])
    donors={r:sorted(set(values)) for r,values in donors.items()}
    def replacement(relation,old):
        values=donors.get(relation,[])
        if len(values)<2: return None
        index=int(canonical(old)[:8],16)%len(values)
        for offset in range(len(values)):
            value=values[(index+offset)%len(values)]
            if value.casefold()!=old.casefold(): return value
        return None
    counts=Counter(); used=set(); candidates=[]; pending={}; rejects=Counter()
    limits={('train',f):512 if f=='single_evidence' else 384 for f in FAMILIES}
    limits.update({('validation',f):32 if f=='single_evidence' else 16 for f in FAMILIES})
    limits.update({('test',f):64 for f in FAMILIES})
    def keep(c):
        key=(c['split'],c['family'])
        if counts[key]>=limits[key] or used.intersection(c['source_ids']): return False
        candidates.append(c); used.update(c['source_ids']); counts[key]+=1; return True
    # Reserve the scarcest valid relation-chain sources before easier one-hop pools.
    for row in sorted(doubles,key=lambda r:canonical(r['raw_id'])):
        splits={task_fold(source) for source in row['sources']}
        if len(splits)!=1: continue
        split=next(iter(splits))
        if split=='calibration': continue
        relations=tuple(f[1] for f in row['facts'])
        template_split=('train','validation','test')[int(canonical(relations)[:8],16)%3]
        if split!=template_split: continue
        if counts[(split,'double_evidence')]>=limits[(split,'double_evidence')]: continue
        facts=row['facts']; old=facts[1][2]; new=replacement(facts[1][1],old)
        if not new or old.casefold() in row['question'].casefold(): continue
        try:
            edited=[relabel(c,old,new) if occurrences(c,old) else c for c in row['contexts']]
            changed=copy.deepcopy(facts);changed[1][2]=new
            proof={'rule':'uniform_terminal_object_relabel_in_annotated_chain','raw_ids':[row['raw_id']],
                   'query':{'subject':facts[0][0],'relations':[facts[0][1],facts[1][1]]},'world_facts':[facts,changed]}
            keep(candidate(tok,'double_evidence',split,row['sources'],row['question'],row['contexts'],edited,proof,
                           'annotated_relation_chain:'+canonical(relations)))
        except ValueError as error: rejects[str(error)]+=1
    for row in sorted(triples,key=lambda r:canonical((r['raw_id'],r['fact']))):
        split=task_fold(row['source'])
        if split=='calibration' or (split=='train')!=(row['partition']=='train'): continue
        if all(counts[(split,f)]>=limits[(split,f)] for f in ('single_evidence','binding')): continue
        sub,rel,old=row['fact']; new=replacement(rel,old)
        if not new or occurrences(row['context'],new): continue
        try:
            # Alternate source-disjoint single facts and two-subject binding cases.
            prior=pending.get((split,rel))
            if prior and prior['source']!=row['source'] and counts[(split,'binding')]<limits[(split,'binding')]:
                sub0,_,answer0=prior['fact']; answer1=old
                if (answer0.casefold()!=answer1.casefold() and not occurrences(prior['context'],answer1)
                        and not occurrences(row['context'],answer0) and not occurrences(sub0,answer0)):
                    proof={'rule':'swap_objects_of_two_annotated_facts','raw_ids':[prior['raw_id'],row['raw_id']],
                           'query':{'subject':sub0,'relations':[rel]},
                           'world_facts':[[prior['fact'],row['fact']],[[sub0,rel,answer1],[sub,rel,answer0]]]}
                    c=candidate(tok,'binding',split,[prior['source'],row['source']],question(sub0,rel,split),
                                [prior['context'],row['context']],
                                [relabel(prior['context'],answer0,answer1),relabel(row['context'],answer1,answer0)],
                                proof,'binding_'+split)
                    if keep(c): pending.pop((split,rel),None); continue
            if counts[(split,'single_evidence')]<limits[(split,'single_evidence')]:
                proof={'rule':'uniform_object_relabel_in_annotated_source','raw_ids':[row['raw_id']],
                       'query':{'subject':sub,'relations':[rel]},'world_facts':[[row['fact']],[[sub,rel,new]]]}
                c=candidate(tok,'single_evidence',split,[row['source']],question(sub,rel,split),
                            [row['context']],[relabel(row['context'],old,new)],proof,'single_'+split)
                keep(c)
            else: pending.setdefault((split,rel),row)
        except ValueError as error: rejects[str(error)]+=1
    path=args.output/'candidates.jsonl'
    with path.open('w') as handle:
        for c in candidates: handle.write(json.dumps(c,ensure_ascii=False)+'\n')
    complete=all(counts[(s,f)]==limits[(s,f)] for s in ('validation','test') for f in FAMILIES) and all(counts[('train',f)]>=QUOTAS[f] for f in FAMILIES)
    report={'status':'SOURCE_GROUNDED_COUNTERFACTUAL_CANDIDATES_V1' if complete else 'BLOCKED_INCOMPLETE_CANDIDATE_POOLS','two_wiki_zip_sha256':sha(args.two_wiki_zip),
            'candidate_pool_path':path.name,'candidate_pool_sha256':sha(path),'tokenizer_files':tokenizer_identity(args.checkpoint),
            'counts':{':'.join(k):v for k,v in sorted(counts.items())},'rejections':dict(rejects),
            'scope':'Controlled source-grounded QA; original annotations plus explicit substitution proof. Not official benchmark scores.',
            'chain_holdout':'Whole annotated relation chains and source articles are split; official train/dev may supply disjoint controlled holdouts.',
            'script_sha256':sha(__file__)}
    write_json(args.output/'manifest.json',report);print(json.dumps(report,indent=2))


def qualify(args):
    if json.loads((args.candidates.parent/'manifest.json').read_text())['status']!='SOURCE_GROUNDED_COUNTERFACTUAL_CANDIDATES_V1':
        raise ValueError('complete train/validation/test source pools required before GPU qualification')
    pool=list(rows(args.candidates)); train=[r for r in pool if r['split']=='train']
    # Fixed interleaving prevents the first abundant family consuming all screening.
    buckets={f:[r for r in train if r['family']==f] for f in FAMILIES}
    ordered=[bucket[i] for i in range(max(map(len,buckets.values()))) for bucket in buckets.values() if i<len(bucket)]
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model,tok,identity=load_runtime(args)
    counts=Counter();accepted=[];qrows=[];rejects=[];screened=0;start=time.monotonic()
    with (args.output/'raw_qualification.jsonl').open('w',buffering=1) as handle:
        for c in ordered:
            if counts[c['family']]>=QUOTAS[c['family']]: continue
            if screened>=2000 or all(counts[f]>=QUOTAS[f] for f in FAMILIES): break
            guard_resources(model,identity,start,args);screened+=1;worlds=[];ok=True
            for w,view in enumerate(c['worlds']):
                generated=greedy(model,view['compact_prompt_ids'],tok.eos_token_id,64)
                ended=bool(generated and generated[-1]==tok.eos_token_id)
                text=tok.decode(generated[:-1] if ended else generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                correct=ended and text in view['accepted_full_answers']
                trace=gold_prefix_trace(model,view['compact_prompt_ids'],generated) if correct else None
                correct=correct and all(m>0 for m in trace['margins'])
                record={'semantic_id':c['semantic_id'],'family':c['family'],'world':w,'generated_ids':generated,
                        'raw_text':text,'correct':correct,'proof_sha256':canonical(c['proof'])}
                handle.write(json.dumps(record)+'\n');ok=ok and correct
                if correct:
                    worlds.append({'semantic_id':c['semantic_id'],'world':w,'truth_verified':True,
                                   'generated_ids':generated,'target_ids':generated,'compact_prompt_ids':view['compact_prompt_ids'],
                                   'gold_margins':trace['margins'],'proof_sha256':canonical(c['proof'])})
            if ok: accepted.append(c['semantic_id']);qrows.extend(worlds);counts[c['family']]+=1
            else: rejects.append({'semantic_id':c['semantic_id'],'reason':'Native complete compact output/EOS/margins did not pass both worlds'})
            if screened%8==0: print(json.dumps({'screened':screened,'accepted':dict(counts),'seconds':time.monotonic()-start}),flush=True)
    result={'status':'QUALIFIED_128_GROUPS' if dict(counts)==QUOTAS else 'BLOCKED_INSUFFICIENT_NATIVE_COMPACT_QUALIFICATION',
            'teacher':identity,'teacher_weight_sha256':identity['checkpoint_sha256'],'teacher_table':'native','teacher_gain':1.,
            'selection_rule':'fixed_order_native_compact_both_worlds','screened_candidates':screened,
            'candidate_pool_path':str(args.candidates.resolve()),'candidate_pool_sha256':sha(args.candidates),
            'rejections':rejects,'rows':qrows,'accepted_semantic_ids':accepted,'counts':dict(counts),
            'raw_sha256':sha(args.output/'raw_qualification.jsonl'),'seconds':time.monotonic()-start}
    write_json(args.output/'qualification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','rejections','accepted_semantic_ids','teacher')},indent=2))


def build_native(args):
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    tok=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    limits={'train':128,'calibration':32,'validation':64,'test':500}
    counts=Counter();bank={};result=[];seen_ids=set()
    pilot_sources={r['source'] for r in json.loads(args.pilot_cases.read_text())['rows']} if args.pilot_cases else set()
    task_titles={source.split(':',1)[1].casefold() for c in rows(args.candidates) for source in c['source_ids']
                 if source.startswith('2wiki:')} if args.candidates else set()
    def qa(group,source,ident,q,answer,proof,accepted=None):
        split=fold(source)
        ids=tok.apply_chat_template([{'role':'user','content':q}],tokenize=True,add_generation_prompt=True,return_dict=False)
        gold=tok.encode(answer,add_special_tokens=False)
        if not gold or len(ids)+max(64,len(gold)+1)>4096: return
        row={'id':canonical((group,ident)),'source_id':source,'split':split,'group':group,
             'input_ids':ids+gold,'prediction_positions':list(range(len(ids)-1,len(ids)+len(gold))),
             'prompt_ids':ids,'generation_budget':64,'accepted_full_answers':accepted or aliases(answer),
             'truth_verified':True,'truth_proof':proof,
             'prefix_scope':'source-truth answer prefixes; not a claim that original Native greedily produced them'}
        if row['id'] in seen_ids: return
        seen_ids.add(row['id'])
        bank.setdefault((group,source),[]).append(row)
    text_bank={'wikipedia':{},'fineweb_edu':{}}
    def text(domain,source,ident,value):
        ids=tok.encode(value,add_special_tokens=False)[:4096]
        if len(ids)<128: return
        positions=sorted({round(i*(len(ids)-2)/31) for i in range(32)})
        row={'id':canonical(('text',ident)),'source_id':source,'split':fold(source),'group':'text',
             'text_domain':domain,'input_ids':ids,'prediction_positions':positions,
             'source_text_sha256':canonical(value)}
        text_bank[domain].setdefault(source,[]).append(row)
    data=json.loads(args.squad.read_text())['data']
    for article in data:
        if article['title'] in pilot_sources or article['title'].casefold() in task_titles: continue
        source='squad:'+article['title'];role=int(canonical(source)[:8],16)%3
        for pi,para in enumerate(article['paragraphs']):
            if role==0 and len(text_bank['wikipedia'].get(source,[]))<16: text('wikipedia',source,(source,pi),para['context'])
            if role!=1: continue  # role2 is reserved for independent task development/holdout
            for item in para['qas']:
                if len(bank.get(('instruction',source),[]))>=64: break
                valid=[a['text'] for a in item['answers'] if para['context'][a['answer_start']:a['answer_start']+len(a['text'])]==a['text']]
                if not valid: continue
                accepted=sorted(set(v for answer in valid for v in aliases(answer)))
                qa('instruction',source,item['id'],
                   'Answer using only the paragraph. Give only the answer text, without explanation.\n\n'+para['context']+'\n\nQuestion: '+item['question'],
                   valid[0],{'dataset':'SQuAD1.1','qa_id':item['id'],'answer_spans_verified':True},accepted)
    for item in rows(args.gsm):
        answer=item['answer'].split('####')[-1].strip().replace(',','')
        if not re.fullmatch(r'-?[0-9]+(?:\.[0-9]+)?',answer): continue
        source='gsm8k:'+canonical(item['question'])
        qa('reasoning',source,source,item['question']+'\nGive only the final numeric answer, without explanation or units.',answer,
           {'dataset':'GSM8K','official_solution_sha256':canonical(item['answer']),'answer_rule':'official #### suffix'})
    rng=random.Random(20260904)
    for i in range(6000):
        numbers=rng.sample(range(100,1000),3);task=i%2
        if task==0:
            q='Sort these integers in ascending order and output only a JSON array: '+', '.join(map(str,numbers))
            answer=json.dumps(sorted(numbers));accepted=[answer,json.dumps(sorted(numbers),separators=(',',':'))]
        else:
            words=rng.sample(['cedar','maple','birch','elm','pine','ash','willow','oak'],4);position=rng.randrange(1,5)
            q='Read this sequence: '+' '.join(words)+f'. Output only the word in position {position}, counting from 1.'
            answer=words[position-1];accepted=[answer]
        source='format:'+canonical(q)
        qa('position_format',source,source,q,answer,{'dataset':'declared_synthetic_format_regression','independent_rule':'sort integers or one-based indexing'},accepted)
    # Round-robin across source articles, never take an entire quota from one article.
    for group in ('instruction','reasoning','position_format'):
        sources=sorted((s for g,s in bank if g==group),key=canonical)
        for index in range(max(len(bank[(group,s)]) for s in sources)):
            for source in sources:
                options=bank[(group,source)]
                if index>=len(options): continue
                row=options[index];key=(group,row['split'])
                if counts[key]<limits[row['split']]: result.append(row);counts[key]+=1
    text_limits={'train':64,'calibration':16,'validation':32,'test':128}
    for path in args.fineweb:
        parquet=pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256,columns=['text','url']):
            for item in batch.to_pylist():
                source='fineweb:'+canonical(item['url'])
                split=fold(source)
                if len([s for s in text_bank['fineweb_edu'] if fold(s)==split])>=text_limits[split]: continue
                text('fineweb_edu',source,source,item['text'])
            if all(sum(fold(s)==split for s in text_bank['fineweb_edu'])>=number for split,number in text_limits.items()): break
        if all(sum(fold(s)==split for s in text_bank['fineweb_edu'])>=number for split,number in text_limits.items()): break
    for domain,sources in text_bank.items():
        ordered=sorted(sources,key=canonical);domain_counts=Counter()
        for index in range(max(map(len,sources.values()))):
            for source in ordered:
                if index>=len(sources[source]): continue
                row=sources[source][index];split=row['split']
                if domain_counts[split]<text_limits[split]:
                    result.append(row);counts[('text',split)]+=1;domain_counts[split]+=1
        if dict(domain_counts)!=text_limits: raise ValueError(f'insufficient {domain} Native text sources: {domain_counts}')
    expected={(group,split):(256 if group=='text' and split=='test' else number)
              for group in ('text','instruction','reasoning','position_format') for split,number in limits.items()}
    if dict(counts)!=expected: raise ValueError(f'incomplete Native strata: {counts}')
    path=args.output/'native_rows.jsonl'
    with path.open('w') as handle:
        for row in result: handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest={'status':'NATIVE_REPLAY_POOL_V1','rows_path':path.name,'rows_sha256':sha(path),
              'tokenizer_files':tokenizer_identity(args.checkpoint),'counts':{':'.join(k):v for k,v in counts.items()},
              'sources':{'squad_sha256':sha(args.squad),'gsm_sha256':sha(args.gsm),'fineweb_files':{p.name:sha(p) for p in args.fineweb}},
              'scope':'Fresh adaptation-disjoint source groups; public data may be in pretraining. Format stratum is synthetic and named separately.',
              'distillation_prefixes':'fixed source-truth answer prefixes, not asserted original greedy trajectories',
              'excluded_model_pilot_articles':sorted(pilot_sources),
              'script_sha256':sha(__file__)}
    write_json(args.output/'manifest.json',manifest);print(json.dumps(manifest,indent=2))


def materialize(args):
    """Freeze physical near/far views after source-truth and Native qualification."""
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    tok=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    q=json.loads(args.qualification.read_text())
    if q['status']!='QUALIFIED_128_GROUPS' or q['candidate_pool_sha256']!=sha(args.candidates):
        raise ValueError('complete original-Native qualification required')
    pool=list(rows(args.candidates));accepted=set(q['accepted_semantic_ids'])
    selected=[c for c in pool if c['split']!='train' or c['semantic_id'] in accepted]
    native_manifest=json.loads(args.native_pool.read_text())
    native=list(rows(args.native_pool.parent/native_manifest['rows_path']))
    excluded_native={r['source_id'] for r in native}
    exposed=set()
    if args.pilot_cases: exposed={r['source'] for r in json.loads(args.pilot_cases.read_text())['rows']}
    # Explicitly held-out DATASET QA, not a claim of an unseen reasoning primitive.
    extras=[]
    for article in sorted(json.loads(args.squad.read_text())['data'],key=lambda r:canonical(r['title'])):
        source='squad:'+article['title']
        if int(canonical(source)[:8],16)%3!=2 or article['title'] in exposed or source in excluded_native: continue
        found=None
        for para in article['paragraphs']:
            for item in para['qas']:
                if not re.match(r'^(Who|Where)\b',item['question']) or not item['answers']: continue
                old=item['answers'][0]['text'];offset=item['answers'][0]['answer_start']
                if (len(old)<3 or para['context'][offset:offset+len(old)]!=old or occurrences(item['question'],old)
                        or re.search(r'\b(first|last|oldest|youngest|most|least|before|after)\b',item['question'],re.I)): continue
                new='Avery Rowan' if item['question'].startswith('Who') else 'Cedar Bay'
                if occurrences(para['context'],new): continue
                proof={'rule':'uniform_answer_entity_relabel_in_official_span_QA','raw_ids':[item['id']],
                       'official_span':[offset,old],'query':{'subject':item['id'],'relations':['annotated answer']},
                       'world_facts':[[[item['id'],'annotated answer',old]],[[item['id'],'annotated answer',new]]]}
                try:
                    found=candidate(tok,'heldout_source_qa','test',[source],item['question'],[para['context']],
                                    [relabel(para['context'],old,new)],proof,'heldout_original_squad_who_where')
                except ValueError: continue
                break
            if found: break
        if found: extras.append(found)
        if len(extras)==64: break
    if len(extras)!=64: raise ValueError('insufficient source-disjoint held-out QA articles')
    selected+=extras
    # Reject cross-dataset aliases of the same Wikipedia source, not just ids.
    def wiki_title(source):
        return source.split(':',1)[1].casefold().replace('_',' ') if source.startswith(('squad:','2wiki:')) else None
    native_titles={wiki_title(s) for s in excluded_native}-{None}
    overlap={s for c in selected for s in c['source_ids'] if wiki_title(s) in native_titles}
    if overlap: raise ValueError('Native/task Wikipedia source overlap; repair the frozen source partition before qualification: '+str(sorted(overlap)))
    backgrounds={s:[] for s in ('train','validation','test')}
    # Different shard from Native replay; exact URL/source exclusion is retained.
    for path in args.fineweb:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['text','url']):
            for item in batch.to_pylist():
                source='fineweb:'+canonical(item['url']);split=fold(source)
                if split not in backgrounds or source in excluded_native or len(backgrounds[split])>=128: continue
                ids=tok.encode('\n\n'+item['text'],add_special_tokens=False)[:4096]
                if len(ids)>=256: backgrounds[split].append({'source':source,'text':item['text'],'ids':ids})
            if all(len(v)>=128 for v in backgrounds.values()): break
        if all(len(v)>=128 for v in backgrounds.values()): break
    if any(len(v)<128 for v in backgrounds.values()): raise ValueError('insufficient source-disjoint background banks')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    qmap={(r['semantic_id'],r['world']):r for r in q['rows']}
    eos=tok.eos_token_id;view_path=args.output/'transport_views.jsonl';counts=Counter()
    with view_path.open('w') as handle:
        for c in selected:
            sid=c['semantic_id'];split=c['split']
            instruction='Use only the supplied passages, even if they describe a counterfactual world. Give only the answer, without explanation.\n\n'
            marked=tok.apply_chat_template([{'role':'user','content':instruction+'SOURCE_BANK_MARKER'+'\n\nQuestion: '+c['question']}],tokenize=False,add_generation_prompt=True)
            left,right=marked.split('SOURCE_BANK_MARKER');prefix=tok.encode(left,add_special_tokens=False);suffix=tok.encode(right,add_special_tokens=False)
            banks=[b for b in backgrounds[split] if all(not occurrences(b['text'],w['answer']) for w in c['worlds'])]
            if len(banks)<64: raise ValueError('background exclusions exhausted bank')
            random.Random(int(sid[:16],16)).shuffle(banks)
            for world,view in enumerate(c['worlds']):
                gold=qmap[(sid,world)]['target_ids'] if split=='train' else view['target_ids']
                cells=[('compact',2048),('near',8192),('far',16384)] if split=='train' else [('compact',2048)]+[(layout,length) for length in ([16384] if split=='validation' else [16384,32768,65536]) for layout in ('near','far')]
                block=tok.encode('\n\n'.join(view['contexts'])+'\n',add_special_tokens=False)
                for layout,length in cells:
                    placement=None;background_sources=[]
                    if layout=='compact': ids=view['compact_prompt_ids']
                    else:
                        size=length-64-len(prefix)-len(suffix);body=[]
                        for b in banks:
                            body.extend(b['ids']);background_sources.append(b['source'])
                            if len(body)>=size: break
                        if len(body)<size: raise ValueError('background bank cannot supply physical length without repetition')
                        body=body[:size];far_start=int(.1*size);near_start=size-len(block)
                        if far_start+len(block)>near_start: raise ValueError('source/oracle swap overlaps')
                        body[far_start:far_start+len(block)]=block
                        if layout=='near':
                            tail=body[near_start:near_start+len(block)]
                            body[near_start:near_start+len(block)]=block;body[far_start:far_start+len(block)]=tail
                        placement=(len(prefix)+(near_start if layout=='near' else far_start),len(block))
                        ids=prefix+body+suffix
                        if len(ids)+64!=length: raise ValueError('physical context length drift')
                    row={'semantic_id':sid,'source_id':canonical(c['source_ids']),'source_lineages':c['source_ids'],
                         'template_lineage':c['template_lineage'],'split':split,'world':world,'family':c['family'],
                         'layout':layout,'length_cap':length,'prompt_ids':ids,'target_ids':gold,'generation_budget':64,
                         'accepted_full_answers':view['accepted_full_answers'],'truth_proof_sha256':canonical(c['proof']),
                         'source_block':placement,'background_sources':background_sources}
                    handle.write(json.dumps(row,separators=(',',':'))+'\n');counts[(split,layout,length)]+=1
    proof_path=args.output/'source_proofs.jsonl'
    with proof_path.open('w') as handle:
        for c in selected: handle.write(json.dumps(c,ensure_ascii=False)+'\n')
    # Absolute private candidate path is intentional in the private data manifest.
    q['candidate_pool_path']=str(args.candidates.resolve())
    write_json(args.output/'native_compact_qualification.json',q)
    manifest={'status':'QUALIFIED_NATURAL_TRANSPORT_V1','tokenizer_files':tokenizer_identity(args.checkpoint),
              'eos_token_id':eos,'views_path':view_path.name,'views_sha256':sha(view_path),
              'qualification_path':'native_compact_qualification.json','qualification_sha256':sha(args.output/'native_compact_qualification.json'),
              'source_proofs_sha256':sha(proof_path),'native_pool_sha256':sha(args.native_pool),
              'evaluation_splits':{'validation':{'groups':64,'lengths':[2048,16384]},'test':{'groups':256,'lengths':[2048,16384,32768,65536]}},
              'scope':'Controlled source-grounded counterfactuals. Held-out SQuAD is cross-dataset QA, not a claim of unseen reasoning primitives.',
              'counts':{':'.join(map(str,k)):v for k,v in counts.items()},'script_sha256':sha(__file__)}
    write_json(args.output/'manifest.json',manifest);print(json.dumps(manifest,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('candidates','qualify','native','materialize'))
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint-contract',type=Path)
    p.add_argument('--two-wiki-zip',type=Path);p.add_argument('--candidates',type=Path)
    p.add_argument('--squad',type=Path);p.add_argument('--gsm',type=Path);p.add_argument('--fineweb',type=Path,nargs='+')
    p.add_argument('--qualification',type=Path);p.add_argument('--native-pool',type=Path);p.add_argument('--pilot-cases',type=Path)
    p.add_argument('--authorized',action='store_true');p.add_argument('--max-seconds',type=float,default=1800)
    args=p.parse_args();args.data=None;args.table=None;args.adapter=None;args.gain=1.;args.seed=42;args.min_headroom_gib=1.
    if args.action=='candidates': build_candidates(args)
    elif args.action=='native': build_native(args)
    elif args.action=='materialize': materialize(args)
    else: qualify(args)

if __name__=='__main__': main()
