"""Retokenize the SAME frozen families, without new outcome/length selection."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from .prepare_mrcr_pairs import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=[json.loads(x) for x in args.inputs.read_text().splitlines()]
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    config=AutoConfig.from_pretrained(args.model,local_files_only=True)
    config=getattr(config,'text_config',config)
    caps={}
    for row in rows:
        ids=tok.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        prefix=tok.apply_chat_template(row['messages'][:-1],tokenize=True,add_generation_prompt=False,enable_thinking=False)
        if ids[:len(prefix)]!=prefix:raise ValueError('chat prefix is not identical')
        answer_tokens=len(tok.encode(row['references'][0],add_special_tokens=False))
        row.update(input_ids=ids,input_tokens=len(ids),prefix_tokens=len(prefix),
            input_ids_sha256=digest(ids),expected_answer_tokens=answer_tokens)
        caps[row['family_id']]=max(caps.get(row['family_id'],0),answer_tokens+64)
    for row in rows:
        row['max_new_tokens']=caps[row['family_id']]
        if row['input_tokens']+row['max_new_tokens']>config.max_position_embeddings:
            raise ValueError('a frozen family does not fit; do not silently drop it')
    args.output.mkdir(parents=True)
    payload=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
    (args.output/'inputs.jsonl').write_text(payload)
    meta=dict(status='PREPARED_SAME_FAMILIES',rows=len(rows),families=len(caps),
        source_inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        inputs_sha256=hashlib.sha256(payload.encode()).hexdigest(),model=str(args.model),
        max_position_embeddings=config.max_position_embeddings,enable_thinking=False,
        prompt_tokens_range=[min(r['input_tokens'] for r in rows),max(r['input_tokens'] for r in rows)],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))


if __name__=='__main__':main()
