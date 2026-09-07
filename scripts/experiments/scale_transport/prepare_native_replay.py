"""Reuse source-separated Native train/validation rows with matching token IDs.

CPU curation only. Teacher probabilities must later come from this original
Qwen3B checkpoint; historical Qwen1.5B outputs are never reused as its teacher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GROUPS = ('instruction', 'reasoning', 'position_format', 'text')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260907)
    args = parser.parse_args()
    manifest = json.loads((args.pool/'manifest.json').read_text())
    source = args.pool/manifest['rows_path']
    if manifest['status'] != 'NATIVE_REPLAY_POOL_V1' or sha(source) != manifest['rows_sha256']:
        raise ValueError('source Native pool identity')
    if sha(args.model/'tokenizer.json') != manifest['tokenizer_files']['tokenizer.json']:
        raise ValueError('token IDs must have identical tokenizer semantics')
    rows = []
    with source.open() as f:
        for line in f:
            row = json.loads(line)
            if row['split'] in ('train', 'validation'):
                rows.append(row)
    selected = {}
    for split in ('train', 'validation'):
        chosen = []
        for group in GROUPS:
            candidates = [r for r in rows if r['split'] == split and r['group'] == group]
            candidates.sort(key=lambda r: hashlib.sha256(f'{args.seed}:{split}:{r["id"]}'.encode()).hexdigest())
            if len(candidates) < 32:
                raise ValueError('insufficient predeclared Native stratum')
            chosen.extend(candidates[:32])
        for row in chosen:
            if row.get('position_ids') is not None or not 0 < len(row['input_ids']) <= 32768:
                raise ValueError('physical Native inputs only')
            if row['group'] != 'text':
                if not row.get('truth_verified') or not row.get('accepted_full_answers') or not row.get('prompt_ids'):
                    raise ValueError('generated retention requires existing verified full answers')
                if len(row['prompt_ids'])+row['generation_budget'] > 32768:
                    raise ValueError('full generation reserve must stay Native')
        selected[split] = chosen
    if {r['source_id'] for r in selected['train']} & {r['source_id'] for r in selected['validation']}:
        raise ValueError('Native replay/validation source overlap')
    args.out.mkdir(parents=True, exist_ok=False)
    for split, chosen in selected.items():
        with (args.out/f'{split}.jsonl').open('x') as f:
            for row in chosen:
                f.write(json.dumps(row)+'\n')
    result = {'status': 'CPU_NATIVE_ROWS_READY_NO_TEACHER_OR_TRAINING', 'seed': args.seed,
        'groups': GROUPS, 'rows_per_group_per_split': 32, 'source_manifest_sha256': sha(args.pool/'manifest.json'),
        'source_rows_sha256': sha(source), 'tokenizer_json_sha256': sha(args.model/'tokenizer.json'),
        'train_rows_sha256': sha(args.out/'train.jsonl'), 'validation_rows_sha256': sha(args.out/'validation.jsonl'),
        'code_sha256': sha(__file__), 'selected_source_groups': {
            split: {group: len({r['source_id'] for r in chosen if r['group'] == group}) for group in GROUPS}
            for split, chosen in selected.items()},
        'limits': 'Historical source pool, not a newly blind benchmark. New original-Qwen3B teacher required; no old model logits. Full-output/EOS and text likelihood remain separate endpoints.'}
    (args.out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
