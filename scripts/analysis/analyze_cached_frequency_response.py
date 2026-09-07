"""Read existing Qwen pilot caches on CPU; compare fixed historical tables.

No model load/forward, autograd, optimizer, candidate search, or GPU access.
The output describes frozen layer-input replay, not downstream utility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from shared_frequency_response import response, self_check


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(array, dtype='<f4'):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def fixed_tables(root, recovered):
    old = json.loads(recovered.read_text())
    identity = old['source_identity']
    native = np.asarray(old['native_float32'], dtype=np.float32)
    movement = np.asarray(old['legacy_m_float64'], dtype=np.float64)
    p2 = np.asarray(old['log_s4_float32'], dtype=np.float32)
    for array, dtype, key in (
        (native, '<f4', 'native_omega_sha256_float32'),
        (movement, '<f8', 'movement_sha256_float64'),
        (p2, '<f4', 'log_s4_tensor_sha256_float32'),
    ):
        if tensor_sha(array, dtype) != identity[key]:
            raise ValueError(f'recovered identity mismatch: {key}')
    runtime = json.loads((root/'runs/pilot_01/source_identity.json').read_text())
    if tensor_sha(native) != runtime['actual_native_sha256']:
        raise ValueError('historical/current Native geometry mismatch')
    t = np.clip(np.arange(64) - 23, 0, 17)
    mr = (native.astype(float) / 4**(t*(t+1)/306)).astype(np.float32)
    deployment = json.loads((root/'runs/pilot_01/deployment.json').read_text())
    if tensor_sha(mr) != deployment['table_sha256'] or deployment['selected_lambda'] != 0:
        raise ValueError('MrPro replay must match the actual reference-only deployment')
    proposal = np.asarray(json.loads((root/'runs/pilot_01/proposal.json').read_text())['proposal'], dtype=np.float32)
    return native, {'MrPro': mr, 'HistoricalP2': p2, 'ExecutedProposal': proposal}, runtime


def energy(x):
    return float(np.square(x).sum(axis=-1).mean())


def run(args):
    # Torch is used only for weights_only CPU deserialization; safetensors reads
    # the six W_O tensors without instantiating a Transformers model.
    import torch
    from safetensors import safe_open

    torch.set_num_threads(1)
    root = args.root
    args.out.mkdir(parents=True, exist_ok=False)
    native, tables, runtime = fixed_tables(root, args.recovered)
    manifest_path = root/'prepared_v2/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    weight_index = json.loads((root/'model/model.safetensors.index.json').read_text())['weight_map']
    gain = 1 + .1 * math.log(4)
    identity = {
        'status': 'EXISTING_CACHE_CPU_REPLAY_NO_MODEL_FORWARD',
        'runtime_source': runtime,
        'prepared_manifest_sha256': sha(manifest_path),
        'recovered_json_sha256': sha(args.recovered),
        'analysis_code_sha256': sha(Path(__file__)),
        'response_code_sha256': sha(Path(__file__).with_name('shared_frequency_response.py')),
        'algebra_check': self_check(),
        'tables': {name: {'float32_sha256': tensor_sha(a), 'values': a.tolist()} for name, a in tables.items()},
        'gain': gain,
        'gain_scope': 'Matched c=.1 for all compared frequency tables; historical performance used c=.074.',
        'jacobian_reference': 'MrPro, nu = nu_ref + native_omega * delta',
        'layers': args.layers,
        'docs': args.docs,
        'limits': 'Native hidden states, selected queries, six layers, 32K only; C/V are previously exposed. No causal capability attribution, new deployment table or optimizer.',
    }
    (args.out/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    started = time.monotonic()
    with (args.out/'rows.jsonl').open('x') as rows:
        for doc in manifest['docs']:
            docid = Path(doc['file']).stem
            if args.docs and docid not in args.docs:
                continue
            for layer in identity['layers']:
                before = time.monotonic()
                cache_path = root/f'runs/pilot_01/cache/{docid}_{layer}.pt'
                cached = torch.load(cache_path, map_location='cpu', weights_only=True)
                q, k, v = [cached[n].float().numpy() for n in ('q', 'k', 'v')]
                pos = cached['pos'].numpy()
                key = f'model.layers.{layer}.self_attn.o_proj.weight'
                with safe_open(root/'model'/weight_index[key], framework='pt', device='cpu') as weights:
                    projection = weights.get_tensor(key).float().numpy()
                common = (q, k, v, projection)
                yn = response(*common, native, pos, compute_jacobian=False)['output']
                saved = cached['y_native'].numpy()
                parity = float(np.linalg.norm(yn-saved)/np.linalg.norm(saved))
                if not np.isfinite(parity) or parity > .005:
                    raise ValueError(f'cached FP32 replay parity mismatch {docid}/{layer}: {parity}')
                ref = response(*common, tables['MrPro'], pos, gain=gain, parameter_scale=native)
                record = {
                    'doc': docid, 'split': doc['split'], 'layer': layer,
                    'cache_sha256': sha(cache_path),
                    'projection_float32_sha256': tensor_sha(projection),
                    'query_positions': pos.tolist(),
                    'native_saved_replay_relative_error': parity,
                    'native_output_energy': energy(yn),
                    'mr_output_energy': energy(ref['output']),
                    'mr_minus_native_energy': energy(ref['output']-yn),
                    'shared_diagonal': np.diag(ref['shared_gram']).tolist(),
                    'within_head_diagonal': ref['within_head_diagonal'].tolist(),
                    'independent_noise_diagonal': ref['independent_noise_diagonal'].tolist(),
                    'finite_changes': {},
                }
                for name in ('HistoricalP2', 'ExecutedProposal'):
                    target = tables[name]
                    y = response(*common, target, pos, gain=gain, compute_jacobian=False)['output']
                    change = y-ref['output']
                    direction = (target.astype(float)-tables['MrPro'])/native
                    linear = ref['jacobian'] @ direction
                    finite_energy, linear_energy = energy(change), energy(linear)
                    record['finite_changes'][name] = {
                        'output_minus_native_energy': energy(y-yn),
                        'output_minus_mr_energy': finite_energy,
                        'linear_predicted_energy': linear_energy,
                        'linear_relative_error': math.sqrt(energy(linear-change)/finite_energy) if finite_energy else None,
                        'linear_cosine': float(np.sum(linear*change)/np.sqrt(np.sum(linear**2)*np.sum(change**2))) if min(finite_energy, linear_energy)>0 else None,
                        'max_phase_change_over_cached_window': float(pos.max()*np.max(np.abs(target.astype(float)-tables['MrPro']))),
                    }
                # Save the signed Jacobian and full Gram, not merely their
                # diagonals, so later analysis does not discard coherence.
                np.savez_compressed(args.out/f'{docid}_{layer}.npz',
                                    jacobian=ref['jacobian'], shared_gram=ref['shared_gram'],
                                    native_output=yn, mr_output=ref['output'])
                record['response_npz_sha256'] = sha(args.out/f'{docid}_{layer}.npz')
                record['seconds'] = time.monotonic()-before
                rows.write(json.dumps(record)+'\n'); rows.flush()
                print(json.dumps({'doc': docid, 'layer': layer, 'seconds': record['seconds'],
                                  'native_parity': parity}), flush=True)
    (args.out/'completion.json').write_text(json.dumps({'status': 'COMPLETE_CACHE_ANALYSIS_ONLY',
        'seconds': time.monotonic()-started, 'rows_sha256': sha(args.out/'rows.jsonl')}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--recovered', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--docs', nargs='*', default=[])
    parser.add_argument('--layers', type=int, nargs='+', choices=[5, 11, 17, 23, 29, 35],
                        default=[5, 11, 17, 23, 29, 35])
    run(parser.parse_args())
