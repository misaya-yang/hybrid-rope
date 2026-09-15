#!/usr/bin/env python3
"""Benchmark generation and LM prefill strategies without mixing their gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from experiments.olmo_recovery_20260912.recovery_v2_eval import greedy_tokens, lm_loss_rows


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(values: list[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def parse_chunks(value: str) -> list[int]:
    """Zero means direct; positive values are chunk sizes."""
    chunks = [int(item.strip()) for item in value.split(',') if item.strip()]
    if not chunks or len(set(chunks)) != len(chunks) or any(chunk < 0 for chunk in chunks):
        raise ValueError('chunks must be unique nonnegative integers')
    return chunks


def memory_snapshot(torch) -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        'free_bytes': int(free), 'total_bytes': int(total),
        'allocated_bytes': int(torch.cuda.memory_allocated()),
        'reserved_bytes': int(torch.cuda.memory_reserved()),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.incomplete')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--lm-array', type=Path, required=True)
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--chunks', default='0,8192,16384,32768')
    parser.add_argument('--max-new-tokens', type=int, default=0,
                        help='0 uses the frozen row budget; otherwise an engineering override')
    parser.add_argument('--whole-nll-tolerance', type=float, default=5e-4)
    parser.add_argument('--tail-nll-tolerance', type=float, default=1e-3)
    parser.add_argument('--minimum-free-fraction', type=float, default=0.10)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    chunks = parse_chunks(args.chunks)
    if args.length <= 0 or args.max_new_tokens < 0:
        raise ValueError('length must be positive and max-new-tokens nonnegative')
    if not 0.0 <= args.minimum_free_fraction < 1.0:
        raise ValueError('minimum-free-fraction must be in [0,1)')

    import torch
    from transformers import AutoTokenizer
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model
    from experiments.olmo_recovery_20260912.runtime import validate_cuda
    from scripts.experiments.cross_audit.tables import install_static
    from scripts.experiments.olmo_fast_screen.ruler_bench import score

    environment = validate_cuda()
    rows = [row for row in read_jsonl(args.panel) if int(row['length_cap']) == args.length]
    if not rows:
        raise ValueError('panel has no row at the requested length')
    row = rows[0]
    max_new_tokens = args.max_new_tokens or int(row['max_new_tokens'])
    lm = np.load(args.lm_array, mmap_mode='r', allow_pickle=False)
    if lm.ndim != 2 or lm.shape[1] < args.length + 1:
        raise ValueError('LM array is shorter than the benchmark length')
    payload = json.loads(args.table.read_text())
    table = payload.get('table', payload)
    values = np.asarray(table['values_float32'], dtype=np.float32)
    gain = float(table['gain'])

    model, _, _ = load_model(args.model, 'Native', training=False)
    install_static(model, values, gain)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = model.generation_config.pad_token_id
    if pad is None:
        pad = min(eos)
    prompt = torch.tensor([row['prompt_ids']], dtype=torch.long, device='cuda')
    lm_ids = torch.tensor(lm[0, :args.length + 1].copy(), dtype=torch.long, device='cuda').unsqueeze(0)
    baseline = memory_snapshot(torch)
    results = []

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for chunk in chunks:
            entry = {'chunk': chunk}
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            before = memory_snapshot(torch); started = time.perf_counter()
            try:
                generated = greedy_tokens(
                    model, prompt, max_new_tokens=max_new_tokens,
                    eos_ids=eos, pad_token_id=pad, prefill_chunk_size=chunk,
                )
                torch.cuda.synchronize()
                generated_text = tokenizer.decode(
                    generated[:-1] if generated and generated[-1] in eos else generated,
                    skip_special_tokens=False, clean_up_tokenization_spaces=False,
                )
                entry['generation'] = {
                    'status': 'ok', 'seconds': time.perf_counter() - started,
                    'generated_ids': generated, 'generated_ids_sha256': digest(generated),
                    'ruler_score': float(score(row, generated_text)),
                    'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                    'peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                    'before': before, 'after': memory_snapshot(torch),
                }
            except torch.OutOfMemoryError as error:
                entry['generation'] = {'status': 'oom', 'error': str(error)}
            torch.cuda.empty_cache()

            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            before = memory_snapshot(torch); started = time.perf_counter()
            try:
                losses = lm_loss_rows(model, lm_ids, prefill_chunk_size=chunk)
                torch.cuda.synchronize()
                entry['lm'] = {
                    'status': 'ok', 'seconds': time.perf_counter() - started,
                    'whole_nll': losses['whole_loss_sum'] / losses['whole_target_count'],
                    'tail128_nll': losses['tail128_loss_sum'] / losses['tail128_target_count'],
                    'actual_strategy': ('direct_no_cache_v1' if losses['lm_prefill_chunk_size'] == 0
                                        else 'dynamic_cache_exact_nll_v1'),
                    'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                    'peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                    'before': before, 'after': memory_snapshot(torch),
                }
            except torch.OutOfMemoryError as error:
                entry['lm'] = {'status': 'oom', 'error': str(error)}
            torch.cuda.empty_cache()
            results.append(entry)
            write_json(args.out.with_name(args.out.name + '.progress'), {
                'status': 'IN_PROGRESS', 'length': args.length,
                'completed_chunks': len(results), 'results': results,
            })

    generation_reference = next((item for item in results if item['generation']['status'] == 'ok'), None)
    lm_reference = next((item for item in results if item['lm']['status'] == 'ok'), None)
    for item in results:
        generation = item['generation']
        if generation['status'] == 'ok' and generation_reference is not None:
            reference = generation_reference['generation']
            generation['generated_ids_equal_reference'] = generation['generated_ids_sha256'] == reference['generated_ids_sha256']
            generation['score_equal_reference'] = generation['ruler_score'] == reference['ruler_score']
            generation['free_fraction_at_peak'] = (
                generation['before']['total_bytes'] - generation['peak_reserved_bytes']
            ) / generation['before']['total_bytes']
            generation['stable'] = bool(
                generation['generated_ids_equal_reference']
                and generation['score_equal_reference']
                and generation['free_fraction_at_peak'] >= args.minimum_free_fraction
            )
        else:
            generation['stable'] = False

        lm_result = item['lm']
        if lm_result['status'] == 'ok' and lm_reference is not None:
            reference = lm_reference['lm']
            lm_result['whole_nll_abs_delta_reference'] = abs(lm_result['whole_nll'] - reference['whole_nll'])
            lm_result['tail128_nll_abs_delta_reference'] = abs(lm_result['tail128_nll'] - reference['tail128_nll'])
            lm_result['finite'] = bool(math.isfinite(lm_result['whole_nll']) and math.isfinite(lm_result['tail128_nll']))
            lm_result['free_fraction_at_peak'] = (
                lm_result['before']['total_bytes'] - lm_result['peak_reserved_bytes']
            ) / lm_result['before']['total_bytes']
            lm_result['stable'] = bool(
                lm_result['finite']
                and lm_result['whole_nll_abs_delta_reference'] <= args.whole_nll_tolerance
                and lm_result['tail128_nll_abs_delta_reference'] <= args.tail_nll_tolerance
                and lm_result['free_fraction_at_peak'] >= args.minimum_free_fraction
            )
        else:
            lm_result['stable'] = False

    stable_generation = [item for item in results if item['generation'].get('stable')]
    stable_lm = [item for item in results if item['lm'].get('stable')]
    recommended_generation = min(stable_generation, key=lambda item: item['generation']['seconds'])['chunk'] if stable_generation else None
    recommended_lm = min(stable_lm, key=lambda item: item['lm']['seconds'])['chunk'] if stable_lm else None
    report = {
        'status': 'PREFILL_CHUNK_BENCHMARK_COMPLETE_V2',
        'environment': environment, 'model': str(args.model.resolve()),
        'table': str(args.table.resolve()), 'length': args.length,
        'generation_row_id': row['row_id'], 'lm_document': 0,
        'max_new_tokens': max_new_tokens,
        'whole_nll_tolerance': args.whole_nll_tolerance,
        'tail_nll_tolerance': args.tail_nll_tolerance,
        'minimum_free_fraction': args.minimum_free_fraction,
        'baseline_memory': baseline, 'results': results,
        'recommended_generation_chunk': recommended_generation,
        'recommended_lm_chunk': recommended_lm,
        'scope': (
            'one complete-budget generation row plus one LM document; runtime engineering only; '
            'generation and LM recommendations are deliberately independent'
        ),
    }
    write_json(args.out, report)
    args.out.with_name(args.out.name + '.progress').unlink(missing_ok=True)
    print(json.dumps({
        'status': report['status'],
        'recommended_generation_chunk': recommended_generation,
        'recommended_lm_chunk': recommended_lm,
    }))


if __name__ == '__main__':
    main()
