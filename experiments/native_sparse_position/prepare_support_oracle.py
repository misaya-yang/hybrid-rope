"""Reuse native-window RULER inputs; locate evidence from query keys, never answers."""
import argparse,hashlib,json,re
from pathlib import Path
from transformers import AutoTokenizer

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);t=AutoTokenizer.from_pretrained(a.model)
    rows=[json.loads(s) for s in a.source.read_text().splitlines()];prepared=[]
    for r in rows:
        if r['length_cap']!=32768 or r['task'] not in ('niah_multikey_2','niah_multiquery'):continue
        text=t.decode(r['prompt_ids'],skip_special_tokens=False);encoded=t(text,add_special_tokens=False,return_offsets_mapping=True)
        if encoded['input_ids']!=r['prompt_ids']:raise ValueError('Source token roundtrip differs')
        cutchar=max(text.rfind('\nWhat is the special magic number for '),text.rfind('\nWhat are all the special magic numbers for '))
        if cutchar<0:raise ValueError('Question boundary absent')
        cut=next(i for i,(_,end) in enumerate(encoded['offset_mapping']) if end>cutchar)
        question=text[cutchar:text.find('<|im_end|>',cutchar)]
        m=re.search(r'for (.*?) mentioned in the provided text',question)
        if m is None:raise ValueError('Query identifiers absent')
        keys=[x.strip() for x in re.split(r',\s*(?:and\s+)?|\s+and\s+',m.group(1))]
        spans=[];support=set();offsets=encoded['offset_mapping']
        for key in keys:
            hits=list(re.finditer(re.escape(key),text[:cutchar]))
            if len(hits)!=1:raise ValueError(f'Query key is not unique: {key}, {len(hits)}')
            lo=max(text.rfind('One of the special magic numbers for ',0,hits[0].start()),text.rfind('The special magic number for ',0,hits[0].start()))
            hi=text.find('.',hits[0].end())+1
            if lo<0 or hits[0].start()-lo>100:raise ValueError('Record prefix not found near queried key')
            if hi<=0 or hi>cutchar:raise ValueError('Record boundary missing')
            inds=[i for i,(start,end) in enumerate(offsets) if end>lo and start<hi]
            span=(min(inds),max(inds)+1);spans.append({'query_key':key,'token_span':span})
            support.update(i//64 for i in inds)
        eligible=set(range(1,(cut-2048)//64));support=sorted(support&eligible)
        if not support:continue
        wrong=[]
        for b in support:
            candidates=[j for j in eligible if j not in support and j not in wrong and j//8==b//8]
            if not candidates:raise ValueError('No matched distance-bin control')
            wrong.append(min(candidates,key=lambda j:(abs(j-b),j)))
        assert len(wrong)==len(support)<32
        prepared.append({**r,'prefix_tokens':cut,'source_record_spans':spans,'support_blocks':support,'wrong_blocks':sorted(wrong),
          'locator_contract':'Input-only oracle: parse queried key identifiers and locate their source record lines. Gold answer values are not used to choose blocks.',
          'control_contract':'Equal number of disjoint blocks in the same 512-token position bins; closest eligible block, deterministic ties.',
          'source_prompt_sha256':hashlib.sha256(text.encode()).hexdigest()})
    if not prepared:raise ValueError('No eligible source records')
    (a.output/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prepared))
    (a.output/'manifest.json').write_text(json.dumps({'rows':len(prepared),'source':str(a.source),'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
      'selection':'All native32K multikey2 and multiquery examples with remote input-located support; no outcome selection; no prompt changes','block':64,'local':2048,'remote_topk_per_query_head':32},indent=2))
    print(json.dumps([{k:r[k] for k in ('row_id','prefix_tokens','support_blocks','wrong_blocks')} for r in prepared]),flush=True)
if __name__=='__main__':main()
