#!/usr/bin/env python3
"""Benchmark exact-context generation and LM scoring across prefill chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.olmo_recovery_20260912.recovery_v2_eval import (
    greedy_tokens,
    lm_loss_rows,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(values: list[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def parse_chunks(value: str) -> list[int]:
    chunks = [int(item.strip()) for item in value.split(',') if item.strip()]
    if not chunks or len(set(chunks)) != len(chunks) or any(chunk <= 0 for chunk in chunks):
        raise ValueError('chunks must be unique positive integers')
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--lm-array', type=Path, required=True)
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--chunks', default='8192,16384,32768')
    parser.add_argument('--max-new-tokens', type=int, default=4)
    parser.add_argument('--nll-tolerance', type=float, default=5e-4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    chunks = parse_chunks(args.chunks)
    if args.max_new_tokens <= 0 or args.length <= 0:
        raise ValueError('length and max-new-tokens must be positive')

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
    lm = np.load(args.lm_array, mmap_mode='r', allow_pickle=False)
    if lm.ndim != 2 or lm.shape[1] < args.length + 1:
        raise ValueError('LM array is shorter than the benchmark length')
    table_payload = json.loads(args.table.read_text())
    table = table_payload.get('table', table_payload)
    values = np.asarray(table['values_float32'], dtype=np.float32)
    gain = float(table['gain'])

    model, _, _ = load_model(args.model, 'Native', training=False)
    install_static(model, values, gain)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = tokenizer.pad_token_id or model.generation_config.pad_token_id or min(eos)
    prompt = torch.tensor([row['prompt_ids']], dtype=torch.long, device='cuda')
    lm_ids = torch.tensor(
        lm[0, :args.length + 1].copy(), dtype=torch.long, device='cuda'
    ).unsqueeze(0)
    baseline_allocated = int(torch.cuda.memory_allocated())
    results = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for chunk in chunks:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                generated = greedy_tokens(
                    model, prompt, max_new_tokens=args.max_new_tokens,
                    eos_ids=eos, pad_token_id=pad, prefill_chunk_size=chunk,
                )
                torch.cuda.synchronize()
                generation_seconds = time.perf_counter() - started
                generated_text = tokenizer.decode(
                    generated[:-1] if generated and generated[-1] in eos else generated,
                    skip_special_tokens=False, clean_up_tokenization_spaces=False,
                )
                lm_started = time.perf_counter()
                losses = lm_loss_rows(
                    model, lm_ids, prefill_chunk_size=chunk
                )
                torch.cuda.synchronize()
                lm_seconds = time.perf_counter() - lm_started
                results.append({
                    'chunk': chunk, 'status': 'ok',
                    'generation_seconds': generation_seconds,
                    'lm_seconds': lm_seconds,
                    'total_seconds': generation_seconds + lm_seconds,
                    'generated_ids': generated,
                    'generated_ids_sha256': digest(generated),
                    'ruler_score': float(score(row, generated_text)),
                    'whole_nll': losses['whole_loss_sum'] / losses['whole_target_count'],
                    'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                    'incremental_peak_bytes': int(torch.cuda.max_memory_allocated()) - baseline_allocated,
                    'peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                })
            except torch.OutOfMemoryError as error:
                results.append({'chunk': chunk, 'status': 'oom', 'error': str(error)})
                torch.cuda.empty_cache()

    reference = next((row for row in results if row['status'] == 'ok'), None)
    for row_result in results:
        if row_result['status'] != 'ok' or reference is None:
            row_result['stable_vs_first'] = False
            continue
        row_result['generated_ids_equal_first'] = (
            row_result['generated_ids_sha256'] == reference['generated_ids_sha256']
        )
        row_result['score_equal_first'] = row_result['ruler_score'] == reference['ruler_score']
        row_result['nll_abs_delta_vs_first'] = abs(row_result['whole_nll'] - reference['whole_nll'])
        row_result['stable_vs_first'] = bool(
            row_result['score_equal_first']
            and row_result['nll_abs_delta_vs_first'] <= args.nll_tolerance
        )
    stable = [row for row in results if row.get('stable_vs_first')]
    recommended = min(stable, key=lambda row: row['total_seconds'])['chunk'] if stable else None
    report = {
        'status': 'PREFILL_CHUNK_BENCHMARK_COMPLETE_V1',
        'environment': environment,
        'model': str(args.model.resolve()),
        'table': str(args.table.resolve()),
        'length': args.length,
        'generation_row_id': row['row_id'],
        'lm_document': 0,
        'max_new_tokens': args.max_new_tokens,
        'nll_tolerance': args.nll_tolerance,
        'results': results,
        'recommended_chunk': recommended,
        'scope': 'one generation row plus one LM document; runtime engineering only, not task evidence',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'recommended_chunk': recommended}))


if __name__ == '__main__':
    main()
