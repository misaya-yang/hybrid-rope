"""Freeze source-backed oracle layouts and one position-preserving distractor edit."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import json
from pathlib import Path
import re

from transformers import AutoTokenizer
from .bench import digest
from .prepare import sha_file, write

CASES = ('niah_multikey_2_131072_1','vt_131072_1','fwe_131072_3','qa_1_131072_3','vt_131072_0')


def token_positions(offsets, spans):
    starts,ends=zip(*offsets);keep=set()
    for a,b in spans:
        keep.update(range(bisect_right(ends,a),bisect_left(starts,b)))
    return sorted(keep)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True,type=Path)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--squad',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--cases',nargs='+',default=list(CASES))
    args=p.parse_args();out=args.out.resolve();prepared=args.prepared.resolve()
    manifest=json.loads((prepared/'manifest.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(manifest['model_path'],local_files_only=True)
    rows={r['row_id']:r for r in map(json.loads,(prepared/'screen.jsonl').read_text().splitlines())}
    prompts={r['row_id']:r for r in map(json.loads,(prepared/'prompts.jsonl').read_text().splitlines())}
    bm={r['row_id']:r for r in map(json.loads,(args.run/'MrProBM.jsonl').read_text().splitlines())}
    squad=json.loads(args.squad.read_text());layouts=[];edits=[]
    out.mkdir(parents=True,exist_ok=False)
    for key in args.cases:
        row=rows[key];text=prompts[key]['prompt_text']
        encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        if encoded['input_ids']!=row['prompt_ids']:raise ValueError('source token alignment differs')
        offsets=encoded['offset_mapping'];details={};evidence=[]
        if row['task']=='niah_multikey_2':
            records=list(re.finditer(r'(?:^|\n)One of the special magic numbers for ([a-z-]+) is: (\d+)\.',text))
            target=[m for m in records if m[2] in row['references']]
            wrong_value=re.search(r'\d+',bm[key]['output_text'])[0]
            wrong=[m for m in records if m[2]==wrong_value]
            if len(target)!=1 or len(wrong)!=1:raise ValueError('ambiguous source records')
            evidence=[target[0].span(),wrong[0].span()]
            spans=[(0,records[0].start()),*evidence,(text.rfind('\nWhat is'),len(text))]
            details.update(target_key=target[0][1],wrong_key=wrong[0][1],wrong_value=wrong_value)
            # Change only the confusable key, retaining the same token count and all positions.
            match=wrong[0];a,b=match.span(1);allowed=set(token_positions(offsets,[(a,b)]))
            for new_key in ('neutral-slime','ordinary-slime','gentle-slime','distant-slime','pleasant-slime'):
                if new_key in text:continue
                changed=text[:a]+new_key+text[b:]
                ids=tokenizer.encode(changed,add_special_tokens=False)
                if len(ids)!=len(row['prompt_ids']):continue
                delta=[j for j,(x,y) in enumerate(zip(ids,row['prompt_ids'])) if x!=y]
                if delta and set(delta)<=allowed:
                    edits.append(dict(row_id=key,variant='neutralize_confusable_key',prompt_ids=ids,
                        prompt_sha256=digest(ids),changed_positions=delta,old_key=match[1],new_key=new_key,
                        unchanged_length=len(ids),references=row['references']))
                    break
            else:raise ValueError('no fixed-token-count distractor replacement')
        elif row['task']=='vt':
            instruction='Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n'
            first=text.index(instruction);second=text.index(instruction,first+len(instruction))
            prefix_end=second+len(instruction)
            assignments=list(re.finditer(r'VAR\s+([A-Z]+)\s*=\s*(?:VAR\s+[A-Z]+|\d+)',text))
            target=[m for m in assignments if m[1] in row['references']]
            if len(target)!=len(row['references']):raise ValueError('missing chain assignment')
            evidence=[(text.rfind('\n',0,m.start()),text.find('\n',m.end())) for m in target]
            spans=[(0,prefix_end),*evidence,(text.rfind('\nQuestion:'),len(text))]
            details.update(preserved_entire_icl=True,assignments=[m[0] for m in target])
        elif row['task']=='qa_1':
            query=text.rsplit('Question: ',1)[1].split('<|im_end|>')[0]
            contexts=[paragraph['context'] for article in squad['data'] for paragraph in article['paragraphs']
                      for qa in paragraph['qas'] if qa['question']==query and not qa['is_impossible']]
            if len(contexts)!=1 or text.count(contexts[0])!=1:raise ValueError('ambiguous QA gold document')
            start=text.index(contexts[0]);headers=list(re.finditer(r'Document \d+:\n',text))
            header=max(m.start() for m in headers if m.start()<start)
            evidence=[(header,start+len(contexts[0]))]
            suffix=text.rfind('\n\nAnswer the question based on the given documents.')
            spans=[(0,headers[0].start()),*evidence,(suffix,len(text))]
            details.update(query=query,gold_document_characters=len(contexts[0]))
        elif row['task']=='fwe':
            prefix='Find the three most frequently appeared coded words. '
            start=text.index(prefix)+len(prefix);end=text.rfind('\nQuestion:')
            words=list(re.finditer(r'\b[a-z]{6}\b',text[start:end]))
            counts=Counter(m[0] for m in words)
            selected=set(row['references'])|{w for w in re.findall(r'\b[a-z]{6}\b',bm[key]['output_text']) if w in counts}
            evidence=[(start+m.start(),start+m.end()) for m in words if m[0] in selected]
            spans=[(0,start),*evidence,(end,len(text))]
            details.update(preserved_counts={w:counts[w] for w in sorted(selected)},
                           scope='Keep every occurrence of gold and model-chosen competing coded words; their ranking is unchanged.')
        else:raise ValueError('unhandled task')
        if any(a<0 or b<=a for a,b in spans):raise ValueError('invalid source spans')
        keep=token_positions(offsets,spans)
        if keep[0]!=0 or keep[-1]!=len(row['prompt_ids'])-1:raise ValueError('lost prompt framing')
        selected_ids=[row['prompt_ids'][j] for j in keep]
        compact=tokenizer.decode(selected_ids,skip_special_tokens=False)
        if row['task']=='fwe':
            counts=Counter(re.findall(r'\b[a-z]{6}\b',compact))
            if any(counts[w]!=n for w,n in details['preserved_counts'].items()):
                raise ValueError('token-subsequence changed selected word counts')
        elif any(ref not in compact for ref in row['references']):
            # QA aliases may not be literal substrings (the official dataset can contain them).
            if row['task']!='qa_1':raise ValueError('lost evidence content')
        starts,ends=zip(*offsets)
        details['evidence_token_spans']=[[bisect_right(ends,a),bisect_left(starts,b)] for a,b in evidence]
        record=dict(row_id=key,prompt_sha256=row['prompt_sha256'],keep_positions=keep,
                    retained_tokens=len(keep),original_tokens=len(row['prompt_ids']),details=details)
        layouts.append(record)
        (out/(key+'_selected.txt')).write_text(compact)
        print(json.dumps(dict(row_id=key,retained=len(keep),original=len(row['prompt_ids']))),flush=True)
    write(out/'layouts.json',layouts);write(out/'edits.json',edits)
    write(out/'manifest.json',dict(status='SOURCE_LAYOUTS_FROZEN_BEFORE_DIAGNOSTIC_OUTPUTS',
        prepared=str(prepared),parent_manifest_sha256=sha_file(prepared/'manifest.json'),
        baseline_run=str(args.run.resolve()),squad_sha256=sha_file(args.squad),
        files={name:sha_file(out/name) for name in ('layouts.json','edits.json')},
        case_ids=args.cases,
        scope='Selected source-backed cases; oracle interventions, not a deployment method.'))


if __name__=='__main__':main()
