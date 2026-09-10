"""Prepare one fixed official RULER task shard on CPU; preserve every source row."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer


TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
         'niah_multikey_2', 'niah_multikey_3', 'niah_multivalue', 'niah_multiquery',
         'vt', 'cwe', 'fwe', 'qa_1', 'qa_2')
UPSTREAM_COMMIT = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--length', type=int, default=131072)
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--seed', type=int, default=137)
    args = parser.parse_args()
    if args.count <= 0 or args.length <= 0:
        raise ValueError('positive fixed count and length required')
    args.out.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    constants = args.upstream/'scripts/data/synthetic/constants.py'
    spec = importlib.util.spec_from_file_location('ruler_data_constants', constants)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configs = yaml.safe_load((args.upstream/'scripts/synthetic.yaml').read_text())
    config = configs[args.task]
    base = module.TASKS[config['task']]
    template = tok.apply_chat_template([{'role': 'user', 'content': base['template']}],
        tokenize=False, add_generation_prompt=True)+base.get('answer_prefix', '')
    generator = args.upstream/f"scripts/data/synthetic/{config['task']}.py"
    argv = [sys.executable, str(generator), '--save_dir', str(args.out/'source'),
        '--save_name', args.task, '--subset', 'validation', '--tokenizer_path', str(args.model),
        '--tokenizer_type', 'hf', '--max_seq_length', str(args.length),
        '--tokens_to_generate', str(base['tokens_to_generate']), '--num_samples', str(args.count),
        '--random_seed', str(args.seed), '--template', template]
    for key, value in config['args'].items():
        argv.extend(['--'+key, str(value)])
    with (args.out/'generator.log').open('x') as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'TOKENIZERS_PARALLELISM': 'false',
                 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                 'NLTK_DATA': os.environ.get('NLTK_DATA', '/root/autodl-tmp/nltk_data')})
    source_path = args.out/'source'/args.task/'validation.jsonl'
    source = [json.loads(line) for line in source_path.read_text().splitlines()]
    if len(source) != args.count:
        raise ValueError('official generator sample count')
    rows = []
    with (args.out/'rows.jsonl').open('x') as output:
        for index, row in enumerate(source):
            text = row['input']+row.get('answer_prefix', '')
            ids = tok.encode(text, add_special_tokens=False)
            if not 0 < len(ids)+base['tokens_to_generate'] <= args.length:
                raise ValueError('physical prompt plus original answer reserve exceeds bucket')
            if not row['outputs'] or any(not isinstance(ref, str) or not ref for ref in row['outputs']):
                raise ValueError('official references must be nonempty strings')
            record = dict(row_id=f'{args.task}_{args.length}_{args.seed}_{index}', task=args.task,
                upstream_index=row['index'], ids=ids, references=row['outputs'],
                length_cap=args.length, input_tokens=len(ids), budget=base['tokens_to_generate'],
                prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                ids_sha256=hashlib.sha256(np.asarray(ids, dtype='<i4').tobytes()).hexdigest())
            output.write(json.dumps(record)+'\n')
            rows.append({key: value for key, value in record.items() if key != 'ids'})
    source_files = [constants, generator, args.upstream/'scripts/synthetic.yaml',
        args.upstream/'scripts/data/tokenizer.py', args.upstream/'scripts/data/manifest_utils.py',
        args.upstream/'scripts/eval/evaluate.py', args.upstream/'scripts/eval/synthetic/constants.py']
    asset_dir = args.upstream/'scripts/data/synthetic/json'
    assets = []
    if config['args'].get('type_haystack') == 'essay':
        assets.append(asset_dir/'PaulGrahamEssays.json')
    if config['task'] == 'qa':
        assets.append(asset_dir/(config['args']['dataset']+'.json'))
    manifest = dict(status='COMPLETE', upstream_commit=UPSTREAM_COMMIT,
        task=args.task, length_cap=args.length, rows=len(rows), seed=args.seed,
        generation_budget=base['tokens_to_generate'], source_jsonl_sha256=sha(source_path),
        rows_sha256=sha(args.out/'rows.jsonl'), code_sha256=sha(__file__),
        source_files={str(path.relative_to(args.upstream)): sha(path) for path in source_files+assets},
        min_max_input_tokens=[min(r['input_tokens'] for r in rows), max(r['input_tokens'] for r in rows)],
        template=template, generator_argv=argv,
        limitations='Official task generator and original token budgets; sample count and pinned revision may differ from published MrRoPE. Matched comparisons apply only to the frozen inputs and tasks; full RULER macro requires all 13 tasks.')
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({key: manifest[key] for key in ('status', 'task', 'rows', 'rows_sha256', 'min_max_input_tokens')}), flush=True)


if __name__ == '__main__':
    main()
