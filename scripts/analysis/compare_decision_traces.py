"""CPU-only, single-decision comparison of two actual greedy generation traces.

Consumes carrier_ruler_run.attach_decision_trace/save_decision_trace artifacts.
Example: --run-a RUN_A --run-b RUN_B --candidate candidate.json --table-a MrPro
--table-b Carrier --rows rows.jsonl --row-id ROW --out comparison.json.

Optional --spans JSON must specify row_id, prompt_sha256, source, and target /
distractor lists of [start, end) PROMPT TOKEN intervals. No semantic masks are
inferred, including ICL examples. Cross replays are layer-local numerical
counterfactuals, not model forwards or attribution of final answer correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<f4').tobytes()).hexdigest()


def one_row(path, row_id):
    matches = []
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('row_id') == row_id:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(f'exactly one {row_id!r} required in {path}')
    return matches[0]


def select_decision(a, b):
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    if common < min(len(a), len(b)):
        return common, common, 'FIRST_ACTUAL_TOKEN_DIVERGENCE'
    status = 'IDENTICAL_OUTPUT_FIRST_DECISION_ONLY' if a == b else 'LENGTH_ONLY_DIFFERENCE_FIRST_DECISION_ONLY'
    return (0 if a and b else None), common, status


def checked_tensor(folder, manifest, name):
    if Path(name).name != name or digest(folder / name) != manifest['files_sha256'][name]:
        raise ValueError(f'trace tensor identity: {name}')
    return torch.load(folder / name, map_location='cpu', weights_only=True)


def load_run(run, row, table, revision):
    folder = run / 'decision_trace' / row['row_id']
    manifest = json.loads((folder / 'manifest.json').read_text())
    deployment = json.loads((run / 'deployment.json').read_text())
    example = one_row(run / 'examples.jsonl', row['row_id'])
    if manifest['status'] != 'ACTUAL_GENERATION_DECISIONS_CAPTURED':
        raise ValueError('actual generation trace required')
    for record in (manifest, example):
        for key in ('row_id', 'input_tokens', 'prompt_sha256'):
            if record[key] != row[key]:
                raise ValueError(f'input identity mismatch: {key}')
    if example['references'] != row['references'] or manifest['generated_ids'] != example['generated_ids']:
        raise ValueError('generated sequence/reference identity mismatch')
    if (deployment['model_revision'] != revision or deployment['table'] != table
            or example['tensor_sha256'] != table['tensor_sha256'] or example['gain'] != table['gain']):
        raise ValueError('deployment must match the supplied frozen table and model')
    layers = sorted(int(m[1]) for name in manifest['files_sha256']
                    if (m := re.fullmatch(r'layer_(\d+)\.pt', name)))
    if not layers or layers != list(range(len(layers))):
        raise ValueError('contiguous complete layer trace required')
    return folder, manifest, deployment, example, layers


def span_masks(path, row, visible):
    if path is None:
        return {}, None
    specification = json.loads(path.read_text())
    if (specification['row_id'] != row['row_id'] or specification['prompt_sha256'] != row['prompt_sha256']
            or not isinstance(specification.get('source'), str) or not specification['source'].strip()):
        raise ValueError('explicit span identity and source required')
    masks = {}
    for name in ('target', 'distractor'):
        mask = torch.zeros(visible, dtype=torch.bool)
        for interval in specification[name]:
            if (len(interval) != 2 or any(type(x) is not int for x in interval)
                    or not 0 <= interval[0] < interval[1] <= row['input_tokens']):
                raise ValueError('spans must be half-open prompt token intervals')
            mask[interval[0]:interval[1]] = True
        if not mask.any():
            raise ValueError(f'empty explicit {name} mask')
        masks[name] = mask
    if (masks['target'] & masks['distractor']).any():
        raise ValueError('target and distractor intervals overlap')
    return masks, {'specification': specification, 'sha256': digest(path),
                   'visible_token_counts': {k: int(v.sum()) for k, v in masks.items()}}


def rotate_absolute_bf16(values, positions, frequency, gain):
    """Independent absolute-position implementation of split-half HF RoPE."""
    phase = positions.float()[:, None] * frequency[None, :]
    cos = torch.cat((phase.cos(), phase.cos()), dim=-1).mul(gain).to(torch.bfloat16)
    sin = torch.cat((phase.sin(), phase.sin()), dim=-1).mul(gain).to(torch.bfloat16)
    half = values.shape[-1] // 2
    rotated_half = torch.cat((-values[..., half:], values[..., :half]), dim=-1)
    return values * cos + rotated_half * sin


def validate_cache(cache, steps, prompt, step, pairs):
    q, k, v = (cache[x] for x in ('q', 'k', 'v'))
    expected_positions = torch.arange(prompt - 1, prompt + steps - 1)
    if (q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[1] != steps
            or k.shape[1] != prompt + steps - 1 or q.shape[2] != 2 * pairs or k.shape[2] != 2 * pairs
            or q.shape[0] % k.shape[0] or not torch.equal(cache['pos'].cpu(), expected_positions)
            or cache['y_actual'].shape != (steps, q.shape[0] * q.shape[2])
            or cache['output_projection'].shape != (q.shape[0] * q.shape[2], q.shape[0] * q.shape[2])):
        raise ValueError('trace layout, GQA mapping, positions, or visible history mismatch')
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError('this numerical replay requires the actual BF16 projected Q/K/V')
    if any(not torch.isfinite(t).all() for t in (q, k, v, cache['y_actual'], cache['output_projection'])):
        raise ValueError('nonfinite captured state')
    return int(cache['pos'][step])


def replay(cache, step, frequency, gain, masks):
    pos = int(cache['pos'][step])
    q = cache['q'][:, step:step + 1]
    k, v = (cache[name][:, :pos + 1] for name in ('k', 'v'))
    qr = rotate_absolute_bf16(q, torch.tensor([pos]), frequency, gain).float()[:, 0]
    kr = rotate_absolute_bf16(k, torch.arange(pos + 1), frequency, gain).float()
    kv_heads, length, dim = kr.shape
    groups = qr.shape[0] // kv_heads
    # Full frequency table and every observed visible key, never L by L attention.
    logits = torch.einsum('hgd,hnd->hgn', qr.reshape(kv_heads, groups, dim), kr) / math.sqrt(dim)
    probabilities = logits.softmax(dim=-1)
    head_output = torch.einsum('hgn,hnd->hgd', probabilities, v.float())
    # FP32 accumulation with BF16 boundaries corresponding to Flash output and
    # o_proj output. CPU accumulation is not claimed identical to CUDA kernels.
    flat = head_output.to(torch.bfloat16).float().reshape(-1)
    projected = torch.mv(cache['output_projection'].float(), flat)
    output = projected.to(torch.bfloat16).double()
    stats = {}
    if masks:
        z = logits.reshape(qr.shape[0], length)
        p = probabilities.reshape(qr.shape[0], length)
        stats = {'target_mass_by_head': p[:, masks['target']].sum(-1).tolist(),
                 'distractor_mass_by_head': p[:, masks['distractor']].sum(-1).tolist(),
                 'target_vs_distractor_log_odds_by_head':
                     (torch.logsumexp(z[:, masks['target']], -1)
                      - torch.logsumexp(z[:, masks['distractor']], -1)).tolist()}
    return output, stats


def lm_decision(folder, manifest, step, token_a, token_b):
    scores = checked_tensor(folder, manifest, 'generation_scores.pt')
    if scores.ndim != 2 or scores.shape[0] != len(manifest['generated_ids']):
        raise ValueError('one full processed LM score row per actual generated token required')
    values = scores[step].float()
    chosen = manifest['generated_ids'][step]
    if (torch.isnan(values).any() or torch.isposinf(values).any() or not torch.isfinite(values[chosen])
            or int(values.argmax()) != chosen):
        raise ValueError('trace scores do not support the recorded greedy decision')
    logp = values.log_softmax(-1)
    top_values, top_ids = values.topk(min(10, values.numel()))
    finite = lambda x: float(x) if torch.isfinite(x) else None
    return {'actual_token': chosen, 'processed_score_actual': float(values[chosen]),
            'log_probability_actual': float(logp[chosen]),
            'score_token_A': finite(values[token_a]), 'score_token_B': finite(values[token_b]),
            'score_token_A_minus_B': finite(values[token_a] - values[token_b]),
            'log_probability_token_A': finite(logp[token_a]), 'log_probability_token_B': finite(logp[token_b]),
            'top_token_ids': top_ids.tolist(), 'top_processed_scores': [finite(x) for x in top_values],
            'scores_sha256': manifest['files_sha256']['generation_scores.pt'],
            'null_score_meaning': 'a nonfinite cross-token score or margin, usually a processed -infinity score',
            'score_kind': 'actual generate() processed LM scores, not raw unprocessed LM logits'}


def compare(args):
    if not 1 <= args.threads <= 4:
        raise ValueError('CPU threads must be between one and four')
    torch.set_num_threads(args.threads)
    if Path(args.row_id).name != args.row_id:
        raise ValueError('row id must be a single directory component')
    row = one_row(args.rows, args.row_id)
    if len(row['ids']) != row['input_tokens']:
        raise ValueError('provided prompt token count mismatch')
    candidate = json.loads(args.candidate.read_text())
    tables = [candidate['tables'][key] for key in (args.table_a, args.table_b)]
    for table in tables:
        if (np.asarray(table['values_float32']).ndim != 1
                or not np.isfinite(table['values_float32']).all() or not table['values_float32']
                or tensor_sha(table['values_float32']) != table['tensor_sha256']
                or not math.isfinite(table['gain']) or table['gain'] <= 0):
            raise ValueError('finite frozen frequency and gain identity required')
    if tables[0]['gain'] != tables[1]['gain'] or len(tables[0]['values_float32']) != len(tables[1]['values_float32']):
        raise ValueError('comparison requires the same gain and number of rotary pairs')
    runs = [load_run(run, row, table, candidate['model_revision'])
            for run, table in zip((args.run_a, args.run_b), tables)]
    if runs[0][4] != runs[1][4] or runs[0][2].get('adapter') != runs[1][2].get('adapter'):
        raise ValueError('layer/model adapter identities differ')
    sequences = [run[1]['generated_ids'] for run in runs]
    step, common, status = select_decision(*sequences)
    report = {'status': status, 'row_id': args.row_id, 'prompt_sha256': row['prompt_sha256'],
        'provided_rows_sha256': digest(args.rows), 'input_tokens': row['input_tokens'],
        'input_ids_sha256_int32': hashlib.sha256(np.asarray(row['ids'], dtype='<i4').tobytes()).hexdigest(),
        'candidate_sha256': digest(args.candidate), 'table_keys': [args.table_a, args.table_b],
        'table_tensor_sha256': [t['tensor_sha256'] for t in tables], 'gain': tables[0]['gain'],
        'model_revision': candidate['model_revision'], 'analysis_code_sha256': digest(__file__),
        'cpu_threads': args.threads, 'new_model_forwards': 0, 'shared_generated_prefix_length': common,
        'full_shared_generated_prefix_ids': sequences[0][:common], 'decision_step': step,
        'actual_outputs': [{'arm': label, 'generated_ids': run[1]['generated_ids'],
            'output_text': run[3]['output_text'], 'eos': run[3]['eos'],
            'official_score': run[3].get('official_score'),
            'trace_manifest_sha256': digest(run[0] / 'manifest.json')}
            for label, run in zip(('A', 'B'), runs)], 'layers': [],
        'limits': 'One actual common-prefix decision only. State includes all preceding layer and visible-prefix changes. Local frequency/state decomposition is not final LM-score or answer-correctness attribution. Diagonal parity calibrates an empirical error reference; off-diagonal replay error is unobserved. Headwise attention-odds error cannot be bounded from projected output parity. Explicit supplied spans only; ICL is never auto-labelled as a distractor.'}
    if step is None:
        return report
    report['decision_shared_generated_prefix_ids'] = sequences[0][:step]
    report['decision_query_position'] = row['input_tokens'] + step - 1
    masks, spans = span_masks(args.spans, row, row['input_tokens'] + step)
    report['explicit_spans'] = spans
    token_a, token_b = (seq[step] for seq in sequences)
    report['LM_decision'] = {label: lm_decision(run[0], run[1], step, token_a, token_b)
                             for label, run in zip(('A', 'B'), runs)}
    frequencies = [torch.tensor(t['values_float32'], dtype=torch.float32) for t in tables]
    started = time.monotonic()
    with torch.inference_mode():
        for layer in runs[0][4]:
            if time.monotonic() - started > 600:
                raise TimeoutError('ten-minute bounded CPU analysis expired')
            name = f'layer_{layer}.pt'
            caches = [checked_tensor(run[0], run[1], name) for run in runs]
            for cache, seq in zip(caches, sequences):
                validate_cache(cache, len(seq), row['input_tokens'], step, len(frequencies[0]))
            if not torch.equal(caches[0]['output_projection'], caches[1]['output_projection']):
                raise ValueError('output projection weights differ; this is not a frequency/state comparison')
            outputs, attention = {}, {}
            for i, cache in enumerate(caches):
                for j, frequency in enumerate(frequencies):
                    key = f'{"AB"[i]}_state__{"AB"[j]}_table'
                    outputs[key], attention[key] = replay(cache, step, frequency, tables[0]['gain'], masks)
            aa, ab, ba, bb = (outputs[key] for key in ('A_state__A_table', 'A_state__B_table', 'B_state__A_table', 'B_state__B_table'))
            actual_a, actual_b = (cache['y_actual'][step].double() for cache in caches)
            error_a, error_b = (float(torch.linalg.vector_norm(x - y)) for x, y in ((aa, actual_a), (bb, actual_b)))
            reference = error_a + error_b
            direct = ((ab - aa) + (bb - ba)) / 2
            state = ((ba - aa) + (bb - ab)) / 2
            total, observed = bb - aa, actual_b - actual_a
            if not torch.allclose(direct + state, total, atol=1e-12, rtol=1e-12):
                raise ValueError('finite symmetric decomposition identity failed')
            norm = lambda x: float(torch.linalg.vector_norm(x))
            component = lambda x: {'norm': norm(x), 'squared_norm': float(x @ x),
                'below_diagonal_parity_reference': norm(x) <= reference,
                'direction_evidence': 'NONE_BELOW_OBSERVED_PARITY_REFERENCE' if norm(x) <= reference
                    else 'EXCEEDS_DIAGONAL_REFERENCE_ONLY_OFF_DIAGONAL_ERROR_UNOBSERVED'}
            record = {'layer': layer, 'query_position': report['decision_query_position'],
                'visible_keys': row['input_tokens'] + step,
                'source_tensor_sha256': {label: run[1]['files_sha256'][name] for label, run in zip(('A', 'B'), runs)},
                'actual_B_minus_A_output_norm': norm(observed), 'replayed_B_minus_A_output_norm': norm(total),
                'actual_vs_replayed_total_difference_norm': norm(total - observed),
                'diagonal_parity': {'A_absolute': error_a, 'B_absolute': error_b,
                    'A_relative': error_a / max(norm(actual_a), 1e-12), 'B_relative': error_b / max(norm(actual_b), 1e-12)},
                'observed_total_error_bound': reference,
                'frequency_direct_component': component(direct), 'prior_state_component': component(state),
                'component_cross_term': float(2 * (direct @ state)),
                'within_A_state_frequency_change_norm': norm(ab - aa),
                'within_B_state_frequency_change_norm': norm(bb - ba),
                'attention_from_explicit_spans': attention if masks else None}
            report['layers'].append(record)
            del cache, caches, outputs, attention, aa, ab, ba, bb, direct, state, total, observed, actual_a, actual_b
    report['elapsed_replay_seconds'] = time.monotonic() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('run-a', 'run-b', 'candidate', 'rows', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('table-a', 'table-b', 'row-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--spans', type=Path)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve existing analysis output; choose a new path')
    report = compare(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'decision_step': report['decision_step'],
                      'layers': len(report['layers']), 'output': str(args.out)}))


if __name__ == '__main__':
    main()
