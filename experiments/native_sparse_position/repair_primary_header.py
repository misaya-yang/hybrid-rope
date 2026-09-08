"""Restore the native chat delimiter without regenerating any task content."""
import argparse,hashlib,json
from pathlib import Path
from transformers import AutoTokenizer

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True);tok=AutoTokenizer.from_pretrained(a.model);receipts=[]
    for split in ('compact_dev','frozen_long'):
        source=a.source/(split+'.jsonl');rows=[]
        for row in map(json.loads,source.read_text().splitlines()):
            old_ids=row['prompt_ids'];old=tok.decode(old_ids,skip_special_tokens=False)
            assert old.endswith('<|im_start|>assistant')
            assert tok(old,add_special_tokens=False)['input_ids']==old_ids
            fixed=old+'\n';ids=tok(fixed,add_special_tokens=False)['input_ids']
            assert ids[:len(old_ids)]==old_ids and len(ids)==len(old_ids)+1
            assert len(ids)+row['max_new_tokens']<=row['length_cap']
            row.update(prompt_ids=ids,input_tokens=len(ids),prompt_sha256=hashlib.sha256(fixed.encode()).hexdigest(),parent_prompt_sha256=row['prompt_sha256'],header_repair='Restore exactly one native assistant-header newline removed by upstream .strip()')
            rows.append(row)
        target=a.output/(split+'.jsonl');target.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        receipts.append({'split':split,'rows':len(rows),'parent_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    (a.output/'manifest.json').write_text(json.dumps({'parent':str(a.source),'operation':'Only restore one assistant-header token; data, order, task and expected answer unchanged','primary_metric':'Entire raw decoded answer body, removing only terminal EOS, exactly equals expected string; EOS required','receipts':receipts},indent=2))
    print(json.dumps(receipts))
if __name__=='__main__':main()
