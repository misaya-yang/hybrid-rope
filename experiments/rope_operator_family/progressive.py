"""Output distillation on actual student-prefix states, one layer at a time.

Only calibration tokens enter this fit. Native Q/K/V targets remain frozen.
The original MLPs, residual paths, and layer norms propagate each fitted layer's
states before fitting the next layer. No language-model weights are optimized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .model import CompactAttention
from .run import fingerprint, load_model, read_factors, runtime_record, save_factors
from .study import FitConfig, digest, fit_layer, load_record, load_rows, records_for, write_json


@torch.no_grad()
def propagate(layer, hidden, device):
    x = hidden.to(device)
    positions = torch.arange(x.shape[1], device=device).unsqueeze(0)
    value = layer(x, position_ids=positions, attention_mask=None, use_cache=False)
    return (value[0] if isinstance(value, tuple) else value).detach().cpu()


def fit_progressive(model, rows, capture_path, initialization, output, config, base_model, runtime):
    source, initial, out = Path(capture_path), Path(initialization), Path(output)
    manifest = json.loads((source / 'manifest.json').read_text())
    init_manifest = json.loads((initial / 'manifest.json').read_text())
    if init_manifest.get('checkpoint_role') != 'unoptimized_initialization' or init_manifest['status'] != 'complete':
        raise ValueError('need complete common unoptimized initialization')
    if digest(source / 'manifest.json') != init_manifest['specification']['capture_sha256']:
        raise ValueError('initialization capture differs')
    if base_model != manifest.get('base_model'):
        raise ValueError('native teacher weights differ')
    rows = [row for row in rows if row['split'] == 'calibration']
    expected = [r['id'] for r in manifest['records'] if r['split'] == 'calibration']
    if [r['id'] for r in rows] != expected:
        raise ValueError('calibration rows/order differ from native capture')
    specification = dict(capture_sha256=digest(source / 'manifest.json'),
                         shape=init_manifest['specification']['shape'],
                         fold=init_manifest['specification']['fold'], fit=asdict(config),
                         input_ids_sha256=hashlib.sha256(json.dumps([r['input_ids'] for r in rows]).encode()).hexdigest(),
                         initialization_sha256={f'layer_{i:03d}.pt': digest(initial / f'layer_{i:03d}.pt')
                                                for i in range(manifest['layers'])},
                         training_inputs='student prefix states; native attention output/value targets')
    spec_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        if json.loads((out / 'manifest.json').read_text())['specification_sha256'] != spec_hash:
            raise ValueError('output belongs to another progressive configuration')
    device = model.model.embed_tokens.weight.device
    model.eval().requires_grad_(False)
    with torch.no_grad():
        hidden = [model.model.embed_tokens(torch.tensor(r['input_ids'], device=device)[None]).cpu() for r in rows]
    receipt = dict(status='running', specification=specification, specification_sha256=spec_hash,
                   base_model=base_model, layers=manifest['layers'], completed_layers=[], runtime=runtime,
                   checkpoint_role='fitted_operator', fit_mode='progressive_student_inputs')
    start = time.monotonic()
    for index, layer in enumerate(model.model.layers):
        target = out / f'layer_{index:03d}.pt'
        native_paths = records_for(source, index, 'calibration')
        if target.exists():
            saved = torch.load(target, map_location='cpu', weights_only=True)
            if saved['specification_sha256'] != spec_hash:
                raise ValueError('incompatible saved progressive layer')
            factors = read_factors(target, device)
        else:
            student_paths = []
            for row_index, (row, state, native_path) in enumerate(zip(rows, hidden, native_paths)):
                original = load_record(native_path)
                if row['id'] != original['id'] or len(row['input_ids']) != len(original['key_positions']):
                    raise ValueError('native teacher record does not match calibration input')
                with torch.no_grad():
                    normed = layer.input_layernorm(state.to(device))
                    attention = layer.self_attn
                    q = attention.q_proj(normed)[0][original['query_positions'].to(device)]
                    q = q.reshape(-1, manifest['model_shape']['heads'], manifest['model_shape']['head_dim'])
                    record = dict(q=q.cpu(), k=attention.k_proj(normed)[0].cpu(), v=attention.v_proj(normed)[0].cpu(),
                                  query_positions=original['query_positions'], key_positions=original['key_positions'],
                                  id=row['id'], source_id=row['source_id'], split='calibration')
                if index == 0:
                    for name in ('q', 'k', 'v'):
                        torch.testing.assert_close(record[name].float(), original[name], rtol=0, atol=0)
                path = out / 'student_inputs' / f'layer_{index:03d}' / f'record_{row_index:05d}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(record, path)
                student_paths.append(path)
            factors = read_factors(initial / f'layer_{index:03d}.pt', device)
            try:
                result = fit_layer(factors, student_paths, config, out / f'layer_{index:03d}', native_paths)
            except Exception as error:
                receipt.update(status='failed', failed_layer=index, error=f'{type(error).__name__}: {error}')
                write_json(out / 'manifest.json', receipt)
                raise
            save_factors(target, factors, specification_sha256=spec_hash)
            write_json(out / f'layer_{index:03d}' / 'result.json', result)
        layer.self_attn = CompactAttention(layer.self_attn, factors, index)
        hidden = [propagate(layer, value, device) for value in hidden]
        receipt['completed_layers'].append(index)
        receipt['seconds_this_invocation'] = time.monotonic() - start
        write_json(out / 'manifest.json', receipt)
        print(json.dumps(dict(stage='progressive', layer=index, seconds=receipt['seconds_this_invocation'])), flush=True)
    receipt['status'] = 'complete'
    write_json(out / 'manifest.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'data', 'capture', 'init-from', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    data = Path(args.data)
    if digest(data / 'capture.jsonl') != json.loads((data / 'manifest.json').read_text())['files']['capture.jsonl']:
        raise ValueError('frozen calibration tokens changed')
    model = load_model(args.model, args.device, args.dtype)
    fit_progressive(model, load_rows(data / 'capture.jsonl'), args.capture, args.init_from, args.out,
                    FitConfig(steps=args.steps, score_weight=0, output_weight=1),
                    fingerprint(args.model), runtime_record(args))


if __name__ == '__main__':
    main()
