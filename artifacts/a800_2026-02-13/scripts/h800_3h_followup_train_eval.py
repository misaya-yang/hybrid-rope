#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

FS_SCRIPT = Path('/opt/dfrope/from_scratch_dfrope_train_eval.py')
CACHE_DIR = Path('/opt/dfrope/results/h800_parallel/cache')
OUT_DIR = Path('/opt/dfrope/results/h800_3h_followup')
VAR_DIR = OUT_DIR / 'variants'

DATASET = 'roneneldan/TinyStories'
TOKENIZER = 'EleutherAI/pythia-70m'
TRAIN_TOKENS = 50_000_000
VAL_TOKENS = 2_500_000
SEED = 42

LENGTHS = [2048, 4096, 8192, 12288, 14336, 16384]
N_CHUNKS = 8
RANDOM_SEEDS = [42, 123, 777]
AMP_DTYPE = torch.bfloat16


def load_module(py_file: Path):
    spec = importlib.util.spec_from_file_location('dfrope_followup_fs', str(py_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {py_file}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def geometric_freq(K: int, theta: float) -> np.ndarray:
    idx = np.arange(K, dtype=np.float64)
    return 1.0 / np.power(theta, idx / K)


def anchored_polynomial_freq(K: int, p: float, omega_max: float, omega_min: float) -> np.ndarray:
    t = np.arange(K, dtype=np.float64) / (K - 1)
    log_omega = np.log(omega_max) + np.power(t, p) * (np.log(omega_min) - np.log(omega_max))
    return np.exp(log_omega)


def sigmoid_freq(K: int, omega_max: float, omega_min: float, steepness: float, midpoint: float) -> np.ndarray:
    t = np.arange(K, dtype=np.float64) / (K - 1)
    s = 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))
    log_omega = np.log(omega_max) + s * (np.log(omega_min) - np.log(omega_max))
    return np.exp(log_omega)


def hybrid_geo_poly(omega_geo: np.ndarray, omega_poly: np.ndarray, alpha_poly: float) -> np.ndarray:
    return (1.0 - alpha_poly) * omega_geo + alpha_poly * omega_poly


def build_variants(fs) -> List[Dict[str, Any]]:
    K = 32
    p = 3.9
    omf = 0.3

    geo10k = geometric_freq(K, 10000.0)
    geo500k = geometric_freq(K, 500000.0)
    geo100k = geometric_freq(K, 100000.0)

    omega_min_10k = float(geo10k[-1]) * omf
    omega_min_100k = float(geo100k[-1]) * omf
    omega_min_500k = float(geo500k[-1]) * omf

    sig8_10k = sigmoid_freq(K, omega_max=float(geo10k[0]), omega_min=omega_min_10k, steepness=8.0, midpoint=0.5)
    sig8_100k = sigmoid_freq(K, omega_max=float(geo100k[0]), omega_min=omega_min_100k, steepness=8.0, midpoint=0.5)
    sig8_500k = sigmoid_freq(K, omega_max=float(geo500k[0]), omega_min=omega_min_500k, steepness=8.0, midpoint=0.5)

    poly_500k = anchored_polynomial_freq(K, p=p, omega_max=float(geo500k[0]), omega_min=omega_min_500k)
    hyb_500k = hybrid_geo_poly(geo500k, poly_500k, alpha_poly=0.2)

    return [
        {
            'name': 'geo_10k_baseline',
            'group': 'baseline',
            'rope_cfg': fs.RopeConfig(kind='standard', theta=10000.0),
            'recipe': {'kind': 'standard', 'theta': 10000.0},
        },
        {
            'name': 'geo_500k',
            'group': 'theta_scaling',
            'rope_cfg': fs.RopeConfig(kind='standard', theta=500000.0),
            'recipe': {'kind': 'standard', 'theta': 500000.0},
        },
        {
            'name': 'sigmoid_steep8_mid0.5_omf0.3',
            'group': 'sigmoid',
            'rope_cfg': fs.RopeConfig(kind='custom', theta=1000.0, custom_omega=[float(x) for x in sig8_10k.tolist()]),
            'recipe': {'kind': 'sigmoid', 'theta_base': 10000.0, 'steepness': 8.0, 'midpoint': 0.5, 'omf': omf},
        },
        {
            'name': 'sigmoid_th100k_steep8_mid0.5_omf0.3',
            'group': 'sigmoid_high_theta',
            'rope_cfg': fs.RopeConfig(kind='custom', theta=1000.0, custom_omega=[float(x) for x in sig8_100k.tolist()]),
            'recipe': {'kind': 'sigmoid', 'theta_base': 100000.0, 'steepness': 8.0, 'midpoint': 0.5, 'omf': omf},
        },
        {
            'name': 'sigmoid_th500k_steep8_mid0.5_omf0.3',
            'group': 'sigmoid_high_theta',
            'rope_cfg': fs.RopeConfig(kind='custom', theta=1000.0, custom_omega=[float(x) for x in sig8_500k.tolist()]),
            'recipe': {'kind': 'sigmoid', 'theta_base': 500000.0, 'steepness': 8.0, 'midpoint': 0.5, 'omf': omf},
        },
        {
            'name': 'hybrid_basegeo500k_alpha0.2',
            'group': 'hybrid_high_theta',
            'rope_cfg': fs.RopeConfig(kind='custom', theta=1000.0, custom_omega=[float(x) for x in hyb_500k.tolist()]),
            'recipe': {'kind': 'hybrid_geo_poly', 'theta_base': 500000.0, 'p': p, 'alpha_poly': 0.2, 'omf': omf},
        },
    ]


def starts_for(total: int, L: int, slicing: str, n_chunks: int, seed: int) -> List[int]:
    needed = L + 1
    max_start = total - needed
    if max_start <= 0:
        raise ValueError(f'val tokens too short for L={L}')

    if slicing == 'sequential':
        if n_chunks == 1:
            return [0]
        arr = np.linspace(0, max_start, n_chunks, dtype=np.int64)
        return [int(x) for x in arr.tolist()]

    if slicing == 'random_start':
        rng = random.Random(seed + L * 1009)
        return [rng.randint(0, max_start) for _ in range(n_chunks)]

    raise ValueError(f'bad slicing: {slicing}')


@torch.inference_mode()
def eval_slice(model, val_tokens: torch.Tensor, L: int, slicing: str, seed: int, n_chunks: int) -> Dict[str, Any]:
    starts = starts_for(int(val_tokens.numel()), L, slicing, n_chunks, seed)
    losses: List[float] = []

    for s in starts:
        x = val_tokens[s:s + L + 1].unsqueeze(0).cuda(non_blocking=True)
        with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=True):
            loss = model.loss_on_batch(x, logits_chunk=256)
        losses.append(float(loss.item()))

    ppls = [math.exp(v) for v in losses]
    return {
        'mean': float(np.mean(ppls)),
        'std': float(np.std(ppls)),
        'n_chunks': n_chunks,
    }


@torch.inference_mode()
def eval_variant(model, val_tokens: torch.Tensor) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for L in LENGTHS:
        seq = eval_slice(model, val_tokens, L, 'sequential', seed=SEED, n_chunks=N_CHUNKS)

        per_seed: Dict[str, Any] = {}
        seed_means: List[float] = []
        for rs in RANDOM_SEEDS:
            rr = eval_slice(model, val_tokens, L, 'random_start', seed=rs, n_chunks=N_CHUNKS)
            per_seed[str(rs)] = rr
            seed_means.append(rr['mean'])

        out[str(L)] = {
            'sequential': seq,
            'random_start': {
                'per_seed': per_seed,
                'mean_over_seeds': float(np.mean(seed_means)),
                'std_over_seeds': float(np.std(seed_means)),
                'n_seeds': len(RANDOM_SEEDS),
                'n_chunks_per_seed': N_CHUNKS,
            },
        }

    return out


def write_summary(results: Dict[str, Any], path: Path) -> None:
    items = []
    for name, x in results['variants'].items():
        if 'error' in x:
            continue
        p16 = x['eval']['16384']['random_start']['mean_over_seeds']
        p2 = x['eval']['2048']['random_start']['mean_over_seeds']
        items.append((name, p16, p16 / max(p2, 1e-9), x['group']))
    items.sort(key=lambda t: t[1])

    baseline = None
    for n, p16, _, _ in items:
        if n == 'geo_10k_baseline':
            baseline = p16

    lines: List[str] = []
    lines.append('H800 3h Follow-up (Train+Eval) Summary')
    lines.append('')
    lines.append(f"- ts: {results['ts']}")
    lines.append(f"- variants: {len(results['variants'])}")
    lines.append(f"- lengths: {results['lengths']}")
    lines.append(f"- random_start seeds: {results['random_seeds']}")
    lines.append('')
    lines.append('## Leaderboard (random_start @16K)')
    lines.append('')
    lines.append('| rank | variant | group | ppl@16k | collapse_ratio(16k/2k) | delta_vs_geo10k |')
    lines.append('|---:|---|---|---:|---:|---:|')

    for i, (n, p16, cr, g) in enumerate(items, 1):
        delta = (p16 - baseline) if baseline is not None else float('nan')
        lines.append(f'| {i} | {n} | {g} | {p16:.3f} | {cr:.3f} | {delta:+.3f} |')

    lines.append('')
    lines.append('## Multi-Length (random_start mean over seeds)')
    lines.append('')
    lines.append('| variant | 2k | 4k | 8k | 12k | 14k | 16k |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')

    for n, _, _, _ in items:
        e = results['variants'][n]['eval']
        vals = [e[str(L)]['random_start']['mean_over_seeds'] for L in LENGTHS]
        lines.append('| ' + n + ' | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')

    lines.append('')
    lines.append('## Notes')
    lines.append('')
    lines.append('- All variants are retrained under identical data/order/seed, then re-evaluated with stricter slicing+seed settings.')
    lines.append('- Lower ppl@16k and lower collapse ratio indicate better extrapolation stability.')

    path.write_text('\n'.join(lines))


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VAR_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    fs = load_module(FS_SCRIPT)

    fs.set_seed(SEED)
    train_tokens, val_tokens, data_meta = fs.build_or_load_token_cache(
        out_dir=CACHE_DIR,
        dataset_name=DATASET,
        tokenizer_name=TOKENIZER,
        target_train_tokens=TRAIN_TOKENS,
        target_val_tokens=VAL_TOKENS,
        seed=SEED,
    )

    n_seq = (train_tokens.numel() - 1) // 2048
    order = fs.build_train_order(n_seq, seed=SEED, out_file=CACHE_DIR / f'train_order_seed{SEED}.pt')

    cfg = fs.TrainConfig(
        seq_len=2048,
        dim=512,
        n_layers=6,
        n_heads=8,
        intermediate=2048,
        vocab_size=50304,
        lr=6e-4,
        warmup_ratio=0.02,
        target_effective_batch=32,
    )

    variants = build_variants(fs)

    results: Dict[str, Any] = {
        'ts': time.strftime('%Y-%m-%d_%H%M%S'),
        'source': 'h800_3h_followup_train_eval',
        'data': asdict(data_meta),
        'train_config': asdict(cfg),
        'lengths': LENGTHS,
        'n_chunks': N_CHUNKS,
        'random_seeds': RANDOM_SEEDS,
        'amp_dtype': 'bf16',
        'variants': {},
    }

    for v in variants:
        name = v['name']
        print(f'\\n[run] {name}', flush=True)
        one_out = VAR_DIR / name
        one_out.mkdir(parents=True, exist_ok=True)

        spec = fs.VariantSpec(name, v['rope_cfg'])

        try:
            train_res = fs.train_one_variant(
                spec=spec,
                cfg=cfg,
                train_tokens=train_tokens,
                order=order,
                out_dir=one_out,
                seed=SEED,
                use_amp=True,
                smoke_steps=0,
                amp_dtype=AMP_DTYPE,
                save_checkpoint=True,
            )

            ckpt = Path(train_res['train']['checkpoint'])
            model = fs.TinyGPT(
                vocab_size=cfg.vocab_size,
                dim=cfg.dim,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                intermediate=cfg.intermediate,
                rope_cfg=v['rope_cfg'],
            ).cuda()
            sd = torch.load(ckpt, map_location='cpu')
            model.load_state_dict(sd, strict=True)
            model.eval()

            eval_res = eval_variant(model, val_tokens.to(torch.long))
            rand16 = eval_res['16384']['random_start']['mean_over_seeds']
            print(f"[done] {name} rand@16k={rand16:.3f}", flush=True)

            results['variants'][name] = {
                'name': name,
                'group': v['group'],
                'recipe': v['recipe'],
                'train': train_res['train'],
                'checkpoint': str(ckpt),
                'eval': eval_res,
                'derived': {
                    'rand_collapse_ratio_16k_over_2k': float(
                        eval_res['16384']['random_start']['mean_over_seeds'] / max(eval_res['2048']['random_start']['mean_over_seeds'], 1e-9)
                    ),
                    'seq_collapse_ratio_16k_over_2k': float(
                        eval_res['16384']['sequential']['mean'] / max(eval_res['2048']['sequential']['mean'], 1e-9)
                    ),
                },
            }

            with open(one_out / 'result.json', 'w') as f:
                json.dump(results['variants'][name], f, indent=2)

            del model
            torch.cuda.empty_cache()

        except Exception as ex:
            print(f'[error] {name}: {ex}', flush=True)
            results['variants'][name] = {
                'name': name,
                'group': v['group'],
                'recipe': v['recipe'],
                'error': str(ex),
            }

    elapsed = time.time() - t0
    results['elapsed_sec'] = elapsed

    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    write_summary(results, OUT_DIR / 'summary.md')

    with open(OUT_DIR / 'run.log', 'w') as f:
        f.write(f'elapsed_sec={elapsed:.2f}\\n')

    print('\\n[done] wrote', OUT_DIR / 'results.json', flush=True)
    print('[done] wrote', OUT_DIR / 'summary.md', flush=True)


if __name__ == '__main__':
    main()
