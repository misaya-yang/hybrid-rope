"""Held-out complete event families embedded in independent natural documents."""
import random
from smoke import COLORS

def prepare(tokenizer,shard,context_tokens):
    import pyarrow.parquet as pq
    rng=random.Random(20260909);backgrounds=[];source_rows=[];row_id=0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=256,columns=['text']):
        for text in batch.column(0).to_pylist():
            ids=tokenizer.encode(text,add_special_tokens=False)
            if len(ids)>=context_tokens and '[M7]' not in text and '[P2]' not in text:
                backgrounds.append(ids);source_rows.append(row_id)
            row_id+=1
            if len(backgrounds)==8:break
        if len(backgrounds)==8:break
    if len(backgrounds)!=8:raise ValueError('Need 8 independent long documents')
    rows=[]
    for family in range(8):
        c1,c2=rng.sample(COLORS,2);unit=f'ZX{family+71}'
        events=[f'[M7] Unit {unit} was assigned color {c1}.',f'[P2] Unit {unit} was assigned color {c2}.']
        for length in (512,context_tokens):
            budget=max(0,length-280)
            bg=backgrounds[family][:budget]
            cuts=[0,budget//4,3*budget//4,budget]
            chunks=[tokenizer.decode(bg[cuts[i]:cuts[i+1]]) for i in range(3)]
            for reverse in (False,True):
                records=events[::-1] if reverse else events;colors=[c2,c1] if reverse else [c1,c2]
                body=chunks[0]+'\n'+records[0]+'\n'+chunks[1]+'\n'+records[1]+'\n'+chunks[2]
                qs=[('marker_M7','What color is assigned in record [M7]?',c1),('marker_P2','What color is assigned in record [P2]?',c2),
                    ('first',f'What was the first color assigned to unit {unit}?',colors[0]),
                    ('current',f'What color is currently assigned to unit {unit}?',colors[1]),
                    ('history',f'List both colors assigned to unit {unit} in chronological order.',', '.join(colors))]
                for task,q,answer in qs:
                    prompt=('Within the archive below, only the two labeled records [M7] and [P2] assign colors. '
                      'They appear in chronological order, from earliest to latest. A new assignment replaces the previous color of that unit. '
                      'Record labels are identifiers, not timestamps. Other archive text is unrelated. '
                      'Answer using only color names. For a list, separate names with a comma and a space. Then stop.\n\n'
                      +body+'\n\nQuestion: '+q+'\nAnswer:')
                    rows.append(dict(row_id=f'heldout_f{family}_L{length}_r{int(reverse)}_{task}',family=family,reverse=reverse,
                     task=task,prompt=prompt,expected=answer,length_budget=length,source_shard='004',source_document_row=source_rows[family]))
    return rows
