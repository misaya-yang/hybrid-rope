"""DEV transfer: insert known synthetic records into complete natural documents.

This is synthetic retrieval with natural distractors, not LongBench QA or an
exact length/depth-matched causal comparison with the repeated-background data.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from .prepare import EXACT, render_split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    source = Path(args.source)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    keys = sorted((r for r in rows if r['split'] == 'dev' and r['task'] == 'single_kv'), key=lambda r:r['row_id'])
    natural = {task: sorted((r for r in rows if r['split'] == 'dev' and r['task'] == task), key=lambda r:r['row_id'])
               for task in ['hotpotqa', '2wikimqa']}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    result = []
    for i, original in enumerate(keys):
        document = natural[['hotpotqa', '2wikimqa'][i % 2]][i // 2]
        background = document['raw_context']
        depth = original['answer_depth_fraction'][0]
        cut = background.rfind('\n', 0, int(len(background) * depth))
        if cut < 0:
            cut = max(0, background.rfind(' ', 0, int(len(background) * depth)))
        record = original['records'][0]
        line = f"Record key={record['key']}; value={record['value']}."
        insertion = '\n\n' + line + '\n\n'
        context = background[:cut] + insertion + background[cut:]
        assert context[:cut] + context[cut+len(insertion):] == background
        row = render_split(tokenizer, context, original['raw_question'],
                           'Output exactly the value string and then stop. No explanation, punctuation, or extra spaces.')
        start = row['full_prompt'].index(line)
        offsets = tokenizer(row['full_prompt'], add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        record_ids = [j for j,(a,b) in enumerate(offsets[:row['prefix_length']]) if a >= start and b <= start+len(line) and b>a]
        counts = Counter(row['prefix_ids'][4:])
        row.update(row_id=f'single_kv_natural_bg_dev_{i:03d}', task='single_kv_natural_bg', split='dev',
                   doc_id=document['doc_id'], material_cluster_id=document['context_sha256'],
                   expected=original['expected'], references=[original['expected']], score_contract=EXACT,
                   max_new_tokens=64, parent_row_id=original['row_id'], background_doc_id=document['doc_id'],
                   background_task=document['task'], complete_background_preserved=True, records=[record],
                   record_token_indices=record_ids, record_token_depth=record_ids[0]/row['prefix_length'],
                   record_uniform_probability=len(record_ids)/(row['prefix_length']-4),
                   record_type_balanced_probability=sum(1/counts[row['prefix_ids'][j]] for j in record_ids)/len(counts),
                   source=dict(kind='synthetic_retrieval_natural_distractors', original_qa_evaluation=False,
                               generator='experiments/pm_keep/prepare_natural_background.py',
                               parent_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        assert row['prefix_length'] + 128 < 32768
        result.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in result)
    if out.exists() and out.read_text() != content:
        raise ValueError('Do not overwrite a different frozen dataset')
    out.write_text(content)
    print(json.dumps(dict(rows=len(result), prefix_lengths=[r['prefix_length'] for r in result],
                          first_record_uniform=result[0]['record_uniform_probability'],
                          first_record_type_balanced=result[0]['record_type_balanced_probability'])))


if __name__ == '__main__':
    main()
