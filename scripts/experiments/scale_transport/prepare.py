"""CPU-only selection of disjoint long calibration documents and natural QA."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--parquet',required=True)
    p.add_argument('--longbench',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=False);tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    docs=[];row=0
    for batch in pq.ParquetFile(a.parquet).iter_batches(batch_size=128,columns=['text']):
        for text in batch.column(0).to_pylist():
            row+=1
            if len(text)<100000:continue
            ids=tok.encode(text,add_special_tokens=False)
            if len(ids)<32769:continue
            sha=hashlib.sha256(text.encode()).hexdigest()
            if any(d['sha256']==sha for d in docs):continue
            name=f'doc_{len(docs):02d}.npy';np.save(root/name,np.asarray(ids[:32769],dtype=np.int32))
            docs.append(dict(file=name,sha256=sha,source_row=row-1,source=a.parquet,split='C' if len(docs)<8 else 'V',tokens=32769))
            print(json.dumps({'selected_calibration_docs':len(docs),'source_row':row}),flush=True)
            if len(docs)==16:break
        if len(docs)==16:break
    if len(docs)!=16:raise RuntimeError(f'need 16 long real documents, found {len(docs)}')
    rows=[];counts={'native':0,'long':0}
    template='You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:'
    with zipfile.ZipFile(a.longbench) as z:
        for i,line in enumerate(z.open('data/narrativeqa.jsonl')):
            r=json.loads(line);content=template.format(**r)
            ids=tok.apply_chat_template([{'role':'user','content':content}],tokenize=True,add_generation_prompt=True,return_dict=False)
            if not isinstance(ids,list) or not all(isinstance(t,int) for t in ids):raise TypeError('expected flat token ID list')
            bucket='native' if 4096<=len(ids)<=16384 else 'long' if 32768<len(ids)<=65408 else None
            if bucket is None or counts[bucket]>=8:continue
            sha=hashlib.sha256(r['context'].encode()).hexdigest()
            if sha in {d['sha256'] for d in docs}:raise RuntimeError('calibration/task overlap')
            record={'row_id':f'narrativeqa_{i}','task':'narrativeqa','bucket':bucket,'ids':ids,'references':r['answers'],'budget':128,'source_index':i,'context_sha256':sha,'input_tokens':len(ids)}
            rows.append(record);counts[bucket]+=1
            if min(counts.values())==8:break
    if min(counts.values())<8:raise RuntimeError(f'QA buckets insufficient: {counts}')
    with (root/'eval.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.iterdir() if f.is_file()}
    (root/'manifest.json').write_text(json.dumps({'status':'COMPLETE','docs':docs,'eval_counts':counts,'files':files,'scope':'pilot; first eligible rows, natural task subset, not full LongBench'},indent=2))
if __name__=='__main__':main()
