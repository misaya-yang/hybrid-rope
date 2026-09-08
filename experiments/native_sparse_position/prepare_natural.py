"""Freeze untruncated natural QA by source hash, jointly fitting both tokenizers."""
import argparse, hashlib, json, zipfile
from pathlib import Path
from transformers import AutoTokenizer

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--models',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    tokenizers=[AutoTokenizer.from_pretrained(m) for m in a.models];selected=[];counts={}
    with zipfile.ZipFile(a.archive) as z:
        for task in ('hotpotqa','qasper','multifieldqa_en'):
            raw=z.read('data/'+task+'.jsonl');candidates=[]
            for i,line in enumerate(raw.decode().splitlines()):
                r=json.loads(line)
                prompt=('Read the following document and answer the question using only information in it. '
                  'Give a concise answer and then stop.\n\nDocument:\n'+r['context']+'\n\nQuestion: '+r['input']+'\nAnswer:')
                lengths=[len(t.encode(t.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False))) for t in tokenizers]
                if min(lengths)<8192 or max(lengths)+128>16384:continue
                h=hashlib.sha256(line.encode()).hexdigest()
                candidates.append(dict(row_id=f'{task}_{i}',task=task,prompt=prompt,references=r['answers'],
                  source_row_index=i,source_row_sha256=h,context_sha256=hashlib.sha256(r['context'].encode()).hexdigest(),
                  joint_token_lengths=lengths,max_new_tokens=128))
            candidates.sort(key=lambda r:r['source_row_sha256']);selected+=candidates[:8]
            counts[task]={'eligible':len(candidates),'selected':min(8,len(candidates)),'source_member_sha256':hashlib.sha256(raw).hexdigest()}
    if len(selected)<12:raise ValueError(f'Insufficient natural QA cases: {counts}')
    (a.output/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    (a.output/'manifest.json').write_text(json.dumps({'status':'PREPARED','selection':'First eight per task by source-row SHA, 8192..16256 tokens in every native tokenizer; no truncation or output selection',
      'archive':str(a.archive),'archive_sha256':hashlib.sha256(a.archive.read_bytes()).hexdigest(),'models':a.models,'counts':counts,'rows':len(selected),
      'prompt_scope':'Custom document-before-question template; development subset, not official full LongBench'},indent=2))
    print(json.dumps(counts),flush=True)
if __name__=='__main__':main()
