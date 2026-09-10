"""Prepare actual contiguous text, native-template SFT, and source-held-out QA."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np

from .acquire import file_hash
from .data import chat_ids, sha_text, supervised_chat, write_json


def tar_json(path):
    # Read only JSON members, never extract arbitrary archive paths.
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith('.json'):
                with archive.extractfile(member) as stream:
                    yield member.name, json.load(stream)


def qasper_context(paper):
    sections = [paper['title'], paper.get('abstract', '')]
    for section in paper['full_text']:
        sections += [section['section_name']] + section['paragraphs']
    return '\n\n'.join(x for x in sections if x)


def shingles(text, width=32, stride=16):
    words = re.findall(r'\w+', text.lower())
    return {hashlib.blake2b(' '.join(words[i:i+width]).encode(), digest_size=8).digest()
            for i in range(0, max(0, len(words)-width+1), stride)}


def qasper_references(question):
    references = []
    for record in question['answers']:
        answer = record['answer']
        if answer['unanswerable']:
            text = 'unanswerable'
        elif answer.get('yes_no') is not None:
            text = 'yes' if answer['yes_no'] else 'no'
        elif answer.get('free_form_answer'):
            text = answer['free_form_answer']
        else:
            text = ', '.join(answer.get('extractive_spans', []))
        if text.strip():
            references.append(text.strip())
    return list(dict.fromkeys(references))


def prepare_qasper(sources, out, tokenizer):
    denied = set()
    candidates = defaultdict(list)
    counts = Counter()
    docs = defaultdict(set)
    for archive in ('qasper-train-dev.tgz', 'qasper-test.tgz'):
        for name, papers in tar_json(sources/archive):
            if 'train' in Path(name).name:
                continue  # QASPER is evaluation only in this experiment.
            split = 'test' if 'test' in Path(name).name else 'dev'
            for paper_id, paper in sorted(papers.items()):
                if not isinstance(paper, dict) or 'full_text' not in paper:
                    continue
                context = qasper_context(paper)
                denied.update(shingles(context))
                docs[split].add(paper_id)
                for question in sorted(paper['qas'], key=lambda q: sha_text(q['question']))[:2]:
                    refs = qasper_references(question)
                    if not refs:
                        continue
                    content = ('Read the complete paper and answer the question using only the paper. '
                               'Give a concise answer. If the paper does not answer it, say unanswerable.\n\n'
                               + context + '\n\nQuestion: ' + question['question'])
                    ids = chat_ids(tokenizer, [{'role': 'user', 'content': content}], generation=True)
                    cap = next((x for x in (4096, 8192, 16384, 32768) if len(ids)+256 <= x), None)
                    if cap is None:
                        counts['too_long_intact'] += 1
                        continue
                    row = dict(id=sha_text(paper_id+question['question']), source_id='qasper:'+paper_id,
                               split=split, task='qasper', prompt_ids=ids, input_tokens=len(ids),
                               references=refs, generation_budget=256, length_bucket=cap,
                               prompt_sha256=sha_text(content), metric='full-response token F1; normalized exact secondary')
                    candidates[split, cap].append(row)
    if docs['dev'] & docs['test']:
        raise ValueError('QASPER official dev/test document overlap')
    for split in ('dev', 'test'):
        with (out/f'qa_{split}.jsonl').open('w') as f:
            for cap in (4096, 8192, 16384, 32768):
                rows = sorted(candidates[split, cap], key=lambda r:r['id'])[:64]
                for row in rows:
                    f.write(json.dumps(row)+'\n')
                counts[f'{split}_{cap}'] = len(rows)
    return denied, dict(counts)


def prepare_pg19(sources, out, tokenizer):
    books = json.loads((sources/'pg19_books.json').read_text())
    train, heldout = [], defaultdict(list)
    rows = defaultdict(list)
    seen = {}
    for i, book in enumerate(books):
        path = sources/'pg19'/book['key']
        digest = file_hash(path)
        split = book['split']
        if digest in seen and seen[digest] != split:
            raise ValueError('PG19 identical text across splits')
        seen[digest] = split
        tokens = tokenizer.encode(path.read_text(), add_special_tokens=False)
        if split == 'train':
            # Non-overlapping prediction spans. Only the boundary token is shared.
            for offset in range(0, len(tokens)-16384, 16384):
                array = np.asarray(tokens[offset:offset+16385], dtype='<i4')
                train.append(array)
                rows['train'].append(dict(source_id=book['key'], source_sha256=digest, offset=offset,
                                          array_sha256=hashlib.sha256(array.tobytes()).hexdigest()))
        elif len(tokens) >= 32769:
            offset = int(sha_text(book['key'])[:12],16) % (len(tokens)-32768)
            array = np.asarray(tokens[offset:offset+32769], dtype='<i4')
            heldout[split].append(array)
            rows[split].append(dict(source_id=book['key'], source_sha256=digest, offset=offset,
                                   array_sha256=hashlib.sha256(array.tobytes()).hexdigest()))
        if (i+1) % 16 == 0:
            print(json.dumps({'pg19_books':i+1,'train_windows':len(train)}),flush=True)
    if len(train)<256 or min(len(heldout[s]) for s in ('validation','test'))<8:
        raise ValueError('insufficient distinct physical long-text material')
    np.save(out/'cpt_train.npy', np.stack(train), allow_pickle=False)
    for split, values in heldout.items():
        np.save(out/f'lm_{split}.npy', np.stack(values), allow_pickle=False)
    write_json(out/'pg19_manifest.json',dict(rows=rows, training_prediction_tokens=len(train)*16384,
        train_window_length=16384, eval_window_length=32768,
        policy='Contiguous real book text; official split isolation; no repeated filler or synthetic position gaps.'))
    return dict(train_windows=len(train), train_prediction_tokens=len(train)*16384,
                heldout_books={s:len(v) for s,v in heldout.items()})


def prepare_sft(sources, out, tokenizer, denied):
    counts = Counter()
    candidates = defaultdict(list)
    groups = defaultdict(set)
    with (sources/'longalign.jsonl').open() as stream:
        for i,line in enumerate(stream):
            record = json.loads(line)
            messages = record['messages']
            if len(messages)!=2 or [m['role'] for m in messages]!=['user','assistant']:
                counts['unsupported_conversation'] += 1
                continue
            question, answer = (m['content'] for m in messages)
            if len(re.findall('[\u4e00-\u9fff]',question)) > .01*len(question):
                counts['non_english'] += 1
                continue
            # Character prefilter only avoids tokenizing clearly unsuitable extremes.
            if not 12000 <= len(question) <= 120000 or not answer.strip():
                counts['character_prefilter'] += 1
                continue
            if len(shingles(question,stride=1) & denied) >= 3:
                counts['evaluation_document_overlap'] += 1
                continue
            content_group = sha_text(' '.join(question.split())[:8192])
            number = int(content_group[:8],16)%10
            split = 'dev' if number==0 else 'test' if number==1 else 'train'
            item = supervised_chat(tokenizer,question,answer)
            if not 4096 < item['prompt_tokens'] or len(item['input_ids'])>16384:
                counts['token_length_excluded'] += 1
                continue
            bucket = 8192 if len(item['input_ids'])<=8192 else 16384
            item.update(id=record['id'],source_id='longalign:'+content_group,split=split,
                        length_bucket=bucket, prompt_sha256=sha_text(question),
                        answer_sha256=sha_text(answer), provenance='public LongAlign synthetic assistant response')
            candidates[split,bucket].append(item)
            groups[split].add(content_group)
            if (i+1)%500==0:
                print(json.dumps({'longalign_rows_read':i+1,'eligible':sum(map(len,candidates.values()))}),flush=True)
    if any(groups[a]&groups[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):
        raise ValueError('LongAlign document-group split overlap')
    for split in ('train','dev','test'):
        with (out/f'sft_{split}.jsonl').open('w') as f:
            for bucket in (8192,16384):
                limit = 768 if split=='train' else 96
                rows = sorted(candidates[split,bucket],key=lambda r:sha_text(r['id']))[:limit]
                for row in rows:f.write(json.dumps(row)+'\n')
                counts[f'{split}_{bucket}']=len(rows)
    if counts['train_16384']<64 or counts['dev_16384']<8:
        raise ValueError('not enough intact real OOD long-instruction rows')
    return dict(counts)


def screen_sft_files(out,tokenizer,denied):
    """Check selected rows with all query offsets, including shifted copied passages."""
    excluded=set();counts=Counter()
    for split in ('train','dev','test'):
        with (out/f'sft_{split}.jsonl').open() as f:
            for line in f:
                row=json.loads(line)
                text=tokenizer.decode(row['input_ids'][:row['target_start']],skip_special_tokens=True)
                if len(shingles(text,stride=1)&denied)>=3:excluded.add(row['source_id'])
    for split in ('train','dev','test'):
        path=out/f'sft_{split}.jsonl';temporary=path.with_suffix('.checked')
        with path.open() as source,temporary.open('w') as target:
            for line in source:
                row=json.loads(line)
                if row['source_id'] in excluded:
                    counts['additional_qasper_overlap_rows_removed']+=1
                    continue
                target.write(line);counts[f'{split}_{row["length_bucket"]}']+=1
        temporary.replace(path)
    if counts['train_16384']<64 or counts['dev_16384']<8:
        raise ValueError('source-screened long instruction pool too small')
    return dict(counts)


def prepare_native(sources,out,tokenizer):
    counts=Counter();groups=defaultdict(set)
    handles={s:(out/f'native_{s}.jsonl').open('w') for s in ('train','dev','test')}
    try:
        for line in (sources/'native_rows.jsonl').open():
            row=json.loads(line)
            split={'train':'train','validation':'dev','test':'test'}.get(row['split'])
            if split is None:continue
            groups[split].add(row['source_id'])
            if row['group']=='text':
                ids=row['input_ids']
                item=dict(input_ids=ids,target_start=1,prompt_tokens=0,answer_tokens=len(ids)-1)
                prompt_ids=[]
            else:
                text=tokenizer.decode(row['prompt_ids'],skip_special_tokens=False)
                # The historical OLMo pool used a leading BOS. Apply current native template once.
                if text.startswith(tokenizer.bos_token or '\0'):
                    text=text[len(tokenizer.bos_token):]
                prefix,suffix='<|user|>\n','\n<|assistant|>\n'
                if not text.startswith(prefix) or not text.endswith(suffix):
                    raise ValueError('unknown historical Native template')
                text=text[len(prefix):-len(suffix)]
                references=row['accepted_full_answers']
                if not references:raise ValueError('Native replay needs verified answer')
                item=supervised_chat(tokenizer,text,references[0])
                prompt_ids=item['input_ids'][:item['target_start']]
            if len(item['input_ids'])>4096:continue
            item.update(id=row['id'],source_id=row['source_id'],split=split,task=row['group'],
                        prompt_ids=prompt_ids,references=row.get('accepted_full_answers',[]),
                        generation_budget=row.get('generation_budget',0),
                        provenance='historical source-separated Native pool; not newly blind confirmation')
            handles[split].write(json.dumps(item)+'\n');counts[f'{split}_{row["group"]}']+=1
    finally:
        for f in handles.values():f.close()
    if any(groups[a]&groups[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):
        raise ValueError('Native replay source split overlap')
    return dict(counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stages',nargs='+',default=['qasper','pg19','sft','native'])
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    manifest_path=args.out/'data_manifest.json'
    info=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    denied,qa_counts=prepare_qasper(args.sources,args.out,tokenizer)
    info['qasper']=qa_counts
    for stage,fn in [('pg19',prepare_pg19),('sft',prepare_sft),('native',prepare_native)]:
        if stage in args.stages:
            info[stage]=fn(args.sources,args.out,tokenizer,denied) if stage=='sft' else fn(args.sources,args.out,tokenizer)
            write_json(manifest_path,info)
    if 'audit' in args.stages:
        info['sft'].update(screen_sft_files(args.out,tokenizer,denied))
        info['sft_source_screen']=dict(reference_words=32,reference_stride=16,query_stride=1,
            minimum_matches=3,scope='Exact normalized passage overlap against all official QASPER dev/test papers; not semantic paraphrase detection.')
    info.update(status='CPU_DATA_READY' if all(x in info for x in ('pg19','sft','native')) else 'PARTIAL',
                tokenizer_chat_template_sha256=sha_text(tokenizer.chat_template),
                eos_token_id=tokenizer.eos_token_id, no_model_forward=True,
                files={p.name:dict(bytes=p.stat().st_size,sha256=file_hash(p)) for p in args.out.iterdir()
                       if p.is_file() and p.name!='data_manifest.json'})
    write_json(manifest_path,info)
    print(json.dumps({k:v for k,v in info.items() if k!='files'}),flush=True)


if __name__=='__main__':main()
