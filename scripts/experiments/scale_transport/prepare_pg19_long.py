"""CPU-only, source-verified contiguous PG19 train windows for a later LoRA phase.

No model forward or optimizer. One window per distinct book; never pack short
documents or stretch positions and call them a physical long-context example.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--length', type=int, default=65536)
    parser.add_argument('--seed', type=int, default=20260907)
    args = parser.parse_args()
    selection = json.loads((args.sources/'selection.json').read_text())
    if selection['split'] != 'train' or len(selection['books']) != 128 or args.length != 65536:
        raise ValueError('declared 128 distinct train books and physical 64K required')
    if len({row['key'] for row in selection['books']}) != 128:
        raise ValueError('duplicate training source')
    for row in selection['books']:
        if not row['key'].startswith('train/') or Path(row['key']).name != row['key'].split('/')[1]:
            raise ValueError('source must belong to the declared train split')
        raw = (args.sources/Path(row['key']).name).read_bytes()
        if len(raw) != row['bytes'] or hashlib.md5(raw).hexdigest() != row['etag']:
            raise ValueError('frozen source size or cloud MD5 mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    arrays, records = [], []
    for row in selection['books']:
        path = args.sources/Path(row['key']).name
        text = path.read_text(encoding='utf-8')
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < args.length+1:
            raise ValueError(f'book is too short for one true 64K window: {row["key"]}')
        offset_seed = hashlib.sha256(f'{args.seed}:{row["key"]}'.encode()).digest()
        offset = int.from_bytes(offset_seed[:8], 'little') % (len(tokens)-args.length)
        window = np.asarray(tokens[offset:offset+args.length+1], dtype=np.int32)
        if window.shape != (65537,) or (window < 0).any():
            raise ValueError('physical 64K input plus next-token label')
        arrays.append(window)
        records.append({'source': row['key'], 'source_sha256': digest(path),
            'source_bytes': path.stat().st_size, 'source_tokens': len(tokens), 'offset_tokens': offset,
            'window_token_sha256': hashlib.sha256(window.astype('<i4').tobytes()).hexdigest()})
        if len(records) % 16 == 0:
            print(json.dumps({'prepared_books': len(records), 'physical_tokens_each': args.length}), flush=True)
    data = np.stack(arrays)
    np.save(args.out/'train64k.npy', data, allow_pickle=False)
    tokenizer_files = {p.name: digest(p) for p in args.model.iterdir()
        if p.is_file() and (p.name.startswith('tokenizer') or p.name in ('vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json'))}
    result = {'status': 'CPU_DATA_READY_NO_TRAINING', 'split': 'train', 'books': records,
        'shape': list(data.shape), 'dtype': str(data.dtype), 'physical_input_length': args.length,
        'prediction_tokens_one_pass': len(records)*args.length,
        'distinct_books': len(records), 'seed': args.seed,
        'window_rule': 'One SHA-seeded contiguous token window per frozen book; no document packing, synthetic positions, answer selection or repeated-book fill.',
        'special_tokens': 'Plain source tokenization without invented BOS or terminal EOS; final target is the next observed source token.',
        'selection_sha256': digest(args.sources/'selection.json'),
        'tokenizer_files': tokenizer_files, 'code_sha256': digest(__file__),
        'train_npy_sha256': digest(args.out/'train64k.npy'),
        'limits': 'Preparation only. 8.39M prediction tokens per pass are much less than published YaRN training. This does not qualify memory, gradients, forgetting or generated ability.'}
    (args.out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'shape', 'prediction_tokens_one_pass', 'train_npy_sha256')}), flush=True)


if __name__ == '__main__':
    main()
