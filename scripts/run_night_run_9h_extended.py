#!/usr/bin/env python3
"""
9小时夜间自动化实验 - 扩展版 - 2026-02-14
充分利用9小时，最大化论文数据产出
"""
import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ 全局配置 ============
MODEL_PATH = '/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct'
OUTPUT_BASE = Path('/root/autodl-tmp/dfrope/hybrid-rope/results/night_run_9h_extended')
DATA_NAME = 'wikitext'
DATA_CONFIG = 'wikitext-103-raw-v1'
DATA_SPLIT = 'validation'
MAX_TOKENS = 250000
DEFAULT_SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def log(stage, msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}][{stage}] {msg}")
    sys.stdout.flush()

def save_results(output_dir, results, name='results'):
    with open(output_dir / f'{name}.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_data(tokenizer, max_tokens=None):
    from datasets import load_dataset
    ds = load_dataset(DATA_NAME, DATA_CONFIG, split=DATA_SPLIT, trust_remote_code=True)
    text = '\n\n'.join(ds['text'])
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[:max_tokens] if max_tokens else tokens[:MAX_TOKENS]
    return torch.tensor(tokens, dtype=torch.long)

def load_wikitext_split(tokenizer, split, max_tokens=None):
    """Load and tokenize a specific WikiText split into a 1D token tensor."""
    from datasets import load_dataset
    ds = load_dataset(DATA_NAME, DATA_CONFIG, split=split, trust_remote_code=True)
    text = '\n\n'.join(ds['text'])
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[:max_tokens] if max_tokens else tokens[:MAX_TOKENS]
    return torch.tensor(tokens, dtype=torch.long)

# ============ RoPE 实现 ============
def compute_inv_freq_geometric(head_dim, theta):
    """标准几何RoPE"""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

def compute_inv_freq_anchored(head_dim, theta, anchor_factor, anchor_dim, slope):
    """Anchored sigmoid RoPE"""
    d_half = head_dim // 2
    positions = torch.arange(0, d_half).float()
    j0 = anchor_dim
    k = slope / 100.0
    sigmoid_weight = torch.sigmoid(k * (positions - j0))
    theta_eff = theta * (1.0 + (anchor_factor - 1.0) * sigmoid_weight)
    return 1.0 / (theta_eff ** (2 * positions / head_dim))

def apply_rope_patch(model, inv_freq):
    for name, module in model.named_modules():
        if 'rotary_emb' in name.lower():
            module.inv_freq = inv_freq.to(model.device)
            break
    return model

def load_model():
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True
    )
    # Avoid storing KV cache for full-sequence loss eval; saves memory at long L.
    try:
        model.config.use_cache = False
    except Exception:
        pass
    mem_used = torch.cuda.max_memory_allocated() / 1024**3
    return model, mem_used

@torch.no_grad()
def eval_ppl(model, tokens, length, seed=42, random_start=True):
    model.eval()
    set_seed(seed)
    
    max_start = len(tokens) - length
    if max_start > 0 and random_start:
        start = np.random.randint(0, max_start)
    else:
        start = 0
    
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    
    window = tokens[start:start+length].unsqueeze(0).cuda()
    outputs = model(window, labels=window)
    loss = outputs.loss.item()
    ppl = np.exp(loss)
    
    elapsed = time.time() - t0
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    throughput = length / elapsed
    
    return {
        'ppl': round(ppl, 3),
        'loss': round(loss, 4),
        'elapsed_sec': round(elapsed, 2),
        'throughput_tps': round(throughput, 1),
        'mem_peak_gb': round(mem_peak, 2),
        'status': 'ok'
    }

def run_config(name, tokens, lengths, seed, inv_func, **kwargs):
    """运行单个配置，OOM时停止更长长度"""
    results = {'config': name, 'params': kwargs, 'data': {}}
    
    log("CFG", f"Loading {name}")
    model, mem = load_model()
    log("CFG", f"Model loaded, mem={mem:.2f}GB")
    
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    inv_freq = inv_func(head_dim, **kwargs)
    model = apply_rope_patch(model, inv_freq)
    
    for L in lengths:
        log("EVL", f"  L={L}")
        try:
            r = eval_ppl(model, tokens, L, seed)
            log("EVL", f"  L={L}: PPL={r['ppl']:.3f}")
        except Exception as e:
            r = {'status': 'error', 'error': str(e)[:200]}
            log("EVL", f"  L={L}: ERROR - {str(e)[:100]}")
        
        results['data'][str(L)] = r
        
        if r.get('status') != 'ok':
            log("EVL", f"  Stop {name} at L={L}")
            break
    
    del model
    torch.cuda.empty_cache()
    return results

# ============ 扩展实验设计 ============

def run_theta_sweep(tokens):
    """实验1: Theta细粒度扫描 (1.5h)"""
    log("THETA", "="*60)
    log("THETA", "Experiment 1: Theta Fine-Grained Sweep")
    log("THETA", "="*60)
    
    output_dir = OUTPUT_BASE / '1_theta_sweep'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    thetas = [500000, 800000, 1000000, 1500000, 2000000, 3000000, 5000000]
    lengths = [2048, 8192, 16384, 24576, 32768]
    
    results = {'experiment': 'theta_sweep', 'timestamp': datetime.now().isoformat(), 'configs': {}}
    
    for theta in thetas:
        name = f'geo_{theta//1000}k'
        log("THETA", f"Running {name}")
        r = run_config(name, tokens, lengths, DEFAULT_SEED, compute_inv_freq_geometric, theta=theta)
        results['configs'][name] = r
        save_results(output_dir, results)
    
    return results

def run_anchor_factor_ablation(tokens):
    """实验2: anchor_factor消融 (1h)"""
    log("FACTOR", "="*60)
    log("FACTOR", "Experiment 2: Anchor Factor Ablation")
    log("FACTOR", "="*60)
    
    output_dir = OUTPUT_BASE / '2_anchor_factor_ablation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    factors = [5, 10, 15, 20, 30, 50]
    lengths = [2048, 8192, 16384, 24576, 32768]
    
    results = {'experiment': 'anchor_factor_ablation', 'timestamp': datetime.now().isoformat(), 'configs': {}}
    
    for factor in factors:
        name = f'anchored_x{factor}'
        log("FACTOR", f"Running {name}")
        r = run_config(name, tokens, lengths, DEFAULT_SEED, compute_inv_freq_anchored,
                       theta=100000, anchor_factor=factor, anchor_dim=16, slope=0.5)
        results['configs'][name] = r
        save_results(output_dir, results)
    
    return results

def run_anchor_dim_ablation(tokens):
    """实验3: anchor_dim消融 (45min)"""
    log("DIM", "="*60)
    log("DIM", "Experiment 3: Anchor Dim Ablation")
    log("DIM", "="*60)
    
    output_dir = OUTPUT_BASE / '3_anchor_dim_ablation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dims = [8, 12, 16, 20, 24, 32]
    lengths = [2048, 8192, 16384, 24576]
    
    results = {'experiment': 'anchor_dim_ablation', 'timestamp': datetime.now().isoformat(), 'configs': {}}
    
    for dim in dims:
        name = f'anchored_dim{dim}'
        log("DIM", f"Running {name}")
        r = run_config(name, tokens, lengths, DEFAULT_SEED, compute_inv_freq_anchored,
                       theta=100000, anchor_factor=20, anchor_dim=dim, slope=0.5)
        results['configs'][name] = r
        save_results(output_dir, results)
    
    return results

def run_slope_ablation(tokens):
    """实验4: slope消融 (30min)"""
    log("SLOPE", "="*60)
    log("SLOPE", "Experiment 4: Slope Ablation")
    log("SLOPE", "="*60)
    
    output_dir = OUTPUT_BASE / '4_slope_ablation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    slopes = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
    lengths = [2048, 8192, 16384, 24576]
    
    results = {'experiment': 'slope_ablation', 'timestamp': datetime.now().isoformat(), 'configs': {}}
    
    for slope in slopes:
        name = f'anchored_slope{slope}'
        log("SLOPE", f"Running {name}")
        r = run_config(name, tokens, lengths, DEFAULT_SEED, compute_inv_freq_anchored,
                       theta=100000, anchor_factor=20, anchor_dim=16, slope=slope)
        results['configs'][name] = r
        save_results(output_dir, results)
    
    return results

def run_boundary_dense_scan(tokens):
    """实验5: 边界密集扫描 (1h)"""
    log("BOUNDARY", "="*60)
    log("BOUNDARY", "Experiment 5: Dense Boundary Scan")
    log("BOUNDARY", "="*60)
    
    output_dir = OUTPUT_BASE / '5_boundary_dense_scan'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 密集长度：每4k一个点
    lengths = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 
               22528, 24576, 26624, 28672, 30720, 32768]
    
    configs = [
        ('geo_500k', compute_inv_freq_geometric, {'theta': 500000}),
        ('geo_1M', compute_inv_freq_geometric, {'theta': 1000000}),
        ('anchored_x20', compute_inv_freq_anchored, {'theta': 100000, 'anchor_factor': 20, 'anchor_dim': 16, 'slope': 0.5}),
    ]
    
    results = {'experiment': 'boundary_dense_scan', 'timestamp': datetime.now().isoformat(), 'configs': {}}
    
    for name, func, kwargs in configs:
        log("BOUNDARY", f"Running {name}")
        r = run_config(name, tokens, lengths, DEFAULT_SEED, func, **kwargs)
        results['configs'][name] = r
        save_results(output_dir, results)
    
    return results

def run_multi_seed_robustness(tokens):
    """实验6: 多种子稳健性 (1.5h)"""
    log("ROBUST", "="*60)
    log("ROBUST", "Experiment 6: Multi-Seed Robustness")
    log("ROBUST", "="*60)
    
    output_dir = OUTPUT_BASE / '6_multi_seed_robustness'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seeds = [42, 123, 456, 777, 999, 1234, 5678]
    lengths = [8192, 16384, 24576, 32768]
    
    configs = [
        ('geo_500k', compute_inv_freq_geometric, {'theta': 500000}),
        ('anchored_x20', compute_inv_freq_anchored, {'theta': 100000, 'anchor_factor': 20, 'anchor_dim': 16, 'slope': 0.5}),
    ]
    
    results = {'experiment': 'multi_seed_robustness', 'timestamp': datetime.now().isoformat(), 'configs': {}}
    
    for name, func, kwargs in configs:
        log("ROBUST", f"Running {name}")
        config_results = {'config': name, 'params': kwargs, 'data': {}}
        
        model, _ = load_model()
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        inv_freq = func(head_dim, **kwargs)
        model = apply_rope_patch(model, inv_freq)
        
        for L in lengths:
            ppls = []
            for seed in seeds:
                log("ROBUST", f"  L={L}, seed={seed}")
                try:
                    r = eval_ppl(model, tokens, L, seed=seed)
                    ppls.append(r['ppl'])
                except Exception as e:
                    ppls.append(None)
                    log("ROBUST", f"  ERROR: {str(e)[:50]}")
            
            valid = [p for p in ppls if p is not None]
            if valid:
                config_results['data'][str(L)] = {
                    'seeds': seeds,
                    'ppls': ppls,
                    'mean': round(np.mean(valid), 3),
                    'std': round(np.std(valid), 3),
                    'min': round(np.min(valid), 3),
                    'max': round(np.max(valid), 3)
                }
            else:
                config_results['data'][str(L)] = {'error': 'all seeds failed'}
        
        results['configs'][name] = config_results
        del model
        torch.cuda.empty_cache()
        save_results(output_dir, results)
    
    return results

def run_cross_dataset(tokens):
    """实验7: 跨域验证 (1h)"""
    log("CROSS", "="*60)
    log("CROSS", "Experiment 7: Cross-Dataset Validation")
    log("CROSS", "="*60)
    
    output_dir = OUTPUT_BASE / '7_cross_dataset'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {'experiment': 'cross_dataset', 'timestamp': datetime.now().isoformat(), 'datasets': {}}
    
    # 使用wikitext不同split
    from datasets import load_dataset
    tokenizer = load_tokenizer()
    
    datasets_to_try = [
        ('wikitext_val', lambda: load_data(tokenizer)),  # 复用 validation
        ('wikitext_test', lambda: load_wikitext_split(tokenizer, 'test')),
    ]
    
    # 尝试TinyStories
    try:
        ts = load_dataset('roneneldan/TinyStories', split='validation', trust_remote_code=True)
        datasets_to_try.append(('TinyStories', lambda ts=ts: torch.tensor(
            tokenizer.encode('\n\n'.join(ts['text'][:5000]), add_special_tokens=False)[:MAX_TOKENS], dtype=torch.long)))
        log("CROSS", "TinyStories available")
    except Exception as e:
        log("CROSS", f"TinyStories not available: {e}")
    
    configs = [
        ('geo_500k', compute_inv_freq_geometric, {'theta': 500000}),
        ('anchored_x20', compute_inv_freq_anchored, {'theta': 100000, 'anchor_factor': 20, 'anchor_dim': 16, 'slope': 0.5}),
    ]
    lengths = [2048, 8192, 16384]
    
    for ds_name, ds_loader in datasets_to_try:
        log("CROSS", f"Dataset: {ds_name}")
        try:
            ds_tokens = ds_loader() if callable(ds_loader) else ds_loader
            log("CROSS", f"Loaded {len(ds_tokens)} tokens")
            
            ds_results = {'dataset': ds_name, 'configs': {}}
            for name, func, kwargs in configs:
                log("CROSS", f"  Config: {name}")
                r = run_config(name, ds_tokens, lengths, DEFAULT_SEED, func, **kwargs)
                ds_results['configs'][name] = r
            
            results['datasets'][ds_name] = ds_results
        except Exception as e:
            log("CROSS", f"Failed to load {ds_name}: {e}")
            results['datasets'][ds_name] = {'error': str(e)}
        
        save_results(output_dir, results)
    
    return results

def run_ppl_ratio_analysis(tokens, prev_results):
    """实验8: PPL比率深度分析 (45min)"""
    log("RATIO", "="*60)
    log("RATIO", "Experiment 8: PPL Ratio Analysis")
    log("RATIO", "="*60)
    
    output_dir = OUTPUT_BASE / '8_ppl_ratio_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 在关键边界点做精细评估
    critical_lengths = [14336, 16384, 18432, 20480, 22528, 24576]
    seeds = [42, 123, 777]
    
    configs = [
        ('geo_500k', compute_inv_freq_geometric, {'theta': 500000}),
        ('geo_1M', compute_inv_freq_geometric, {'theta': 1000000}),
        ('anchored_x20', compute_inv_freq_anchored, {'theta': 100000, 'anchor_factor': 20, 'anchor_dim': 16, 'slope': 0.5}),
    ]
    
    results = {'experiment': 'ppl_ratio_analysis', 'timestamp': datetime.now().isoformat(), 'configs': {}}
    
    for name, func, kwargs in configs:
        log("RATIO", f"Running {name}")
        config_results = {'config': name, 'params': kwargs, 'data': {}}
        
        model, _ = load_model()
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        inv_freq = func(head_dim, **kwargs)
        model = apply_rope_patch(model, inv_freq)
        
        for L in critical_lengths:
            ppls = []
            for seed in seeds:
                log("RATIO", f"  L={L}, seed={seed}")
                try:
                    r = eval_ppl(model, tokens, L, seed=seed)
                    ppls.append(r['ppl'])
                except:
                    ppls.append(None)
            
            valid = [p for p in ppls if p is not None]
            if valid:
                config_results['data'][str(L)] = {
                    'ppls': ppls,
                    'mean': round(np.mean(valid), 3),
                    'std': round(np.std(valid), 3)
                }
        
        results['configs'][name] = config_results
        del model
        torch.cuda.empty_cache()
        save_results(output_dir, results)
    
    return results

def generate_final_summary(all_results):
    """生成最终汇总"""
    log("SUMMARY", "Generating final summary...")
    
    def _ppl_at(cfg_result, L):
        try:
            return cfg_result.get("data", {}).get(str(L), {}).get("ppl")
        except Exception:
            return None

    def _fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "-"

    summary = [
        "# 9小时扩展实验汇总",
        f"\n**时间**: {datetime.now().isoformat()}",
        f"**模型**: LLaMA-3-8B-Instruct",
        "",
        "## 实验概览",
        "",
        "1. **Theta扫描**: geo_500k → geo_5M",
        "2. **Anchor Factor消融**: x5 → x50",
        "3. **Anchor Dim消融**: dim8 → dim32",
        "4. **Slope消融**: 0.1 → 2.0",
        "5. **边界密集扫描**: 每4k一个点",
        "6. **多种子稳健性**: 7个seeds",
        "7. **跨域验证**: wikitext + TinyStories",
        "8. **PPL比率分析**: 关键边界点",
        "",
    ]

    # === Quick leaderboards at L=16384 ===
    leader_specs = [
        ("Theta 扫描", "1_theta_sweep", "configs"),
        ("Anchor Factor 消融", "2_factor_ablation", "configs"),
        ("Anchor Dim 消融", "3_dim_ablation", "configs"),
        ("Slope 消融", "4_slope_ablation", "configs"),
    ]

    for title, key, cfg_key in leader_specs:
        block = all_results.get(key, {}) or {}
        cfgs = block.get(cfg_key, {}) if isinstance(block, dict) else {}
        rows = []
        for name, cfg_res in (cfgs or {}).items():
            ppl_2k = _ppl_at(cfg_res, 2048)
            ppl_16k = _ppl_at(cfg_res, 16384)
            collapse = (ppl_16k / ppl_2k) if isinstance(ppl_2k, (int, float)) and isinstance(ppl_16k, (int, float)) else None
            rows.append((name, ppl_2k, ppl_16k, collapse))

        rows = sorted(rows, key=lambda r: (r[2] is None, r[2] if r[2] is not None else 1e9))
        if not rows:
            continue

        summary.extend([
            "",
            f"## {title}（按 PPL@16K 升序）",
            "| Config | PPL@2K | PPL@16K | Collapse(16K/2K) |",
            "|---|---:|---:|---:|",
        ])
        for name, p2, p16, c in rows:
            summary.append(f"| {name} | {_fmt(p2)} | {_fmt(p16)} | {_fmt(c)} |")

    # === Multi-seed robustness (mean±std) ===
    ms = all_results.get("6_multi_seed", {}) or {}
    ms_cfgs = ms.get("configs", {}) if isinstance(ms, dict) else {}
    if ms_cfgs:
        summary.extend([
            "",
            "## 多种子稳健性（mean±std）",
        ])
        # Pick a stable set of lengths for table readability.
        for cfg_name, cfg_res in ms_cfgs.items():
            data = (cfg_res or {}).get("data", {})
            if not isinstance(data, dict) or not data:
                continue
            summary.append(f"\n### {cfg_name}")
            summary.extend([
                "| Length | mean±std | min | max | n_seeds |",
                "|---:|---:|---:|---:|---:|",
            ])
            for L in [8192, 16384, 24576, 32768]:
                d = data.get(str(L), {})
                if not isinstance(d, dict) or "mean" not in d:
                    continue
                summary.append(
                    f"| {L} | {d.get('mean', '-'):.3f}±{d.get('std', 0.0):.3f} | "
                    f"{d.get('min', '-'):.3f} | {d.get('max', '-'):.3f} | {d.get('n_seeds', len(d.get('seeds', [])))} |"
                )

    # === Cross-dataset snapshot at 16K ===
    cd = all_results.get("7_cross_dataset", {}) or {}
    datasets = cd.get("datasets", {}) if isinstance(cd, dict) else {}
    if datasets:
        summary.extend([
            "",
            "## 跨域验证（PPL@16K）",
            "| Dataset | geo_500k | anchored_x20 |",
            "|---|---:|---:|",
        ])
        for ds_name, ds_res in datasets.items():
            if not isinstance(ds_res, dict) or "configs" not in ds_res:
                continue
            cfgs = ds_res.get("configs", {})
            geo = _ppl_at(cfgs.get("geo_500k", {}), 16384)
            anc = _ppl_at(cfgs.get("anchored_x20", {}), 16384)
            summary.append(f"| {ds_name} | {_fmt(geo)} | {_fmt(anc)} |")

    summary.extend([
        "",
        "---",
        "*Generated by run_night_run_9h_extended.py*",
    ])
    
    summary_text = "\n".join(summary)
    summary_path = OUTPUT_BASE / 'summary.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    return summary_text

def main():
    log("MAIN", "="*60)
    log("MAIN", "9小时扩展实验开始")
    log("MAIN", "="*60)
    
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    tokenizer = load_tokenizer()
    tokens = load_data(tokenizer)
    log("MAIN", f"Loaded {len(tokens)} tokens")
    
    all_results = {}
    
    # 依次执行8个实验
    all_results['1_theta_sweep'] = run_theta_sweep(tokens)
    all_results['2_factor_ablation'] = run_anchor_factor_ablation(tokens)
    all_results['3_dim_ablation'] = run_anchor_dim_ablation(tokens)
    all_results['4_slope_ablation'] = run_slope_ablation(tokens)
    all_results['5_boundary_scan'] = run_boundary_dense_scan(tokens)
    all_results['6_multi_seed'] = run_multi_seed_robustness(tokens)
    all_results['7_cross_dataset'] = run_cross_dataset(tokens)
    all_results['8_ppl_ratio'] = run_ppl_ratio_analysis(tokens, all_results)
    
    # 汇总
    summary = generate_final_summary(all_results)
    
    log("MAIN", "="*60)
    log("MAIN", "全部9小时实验完成！")
    log("MAIN", "="*60)
    print(summary)

if __name__ == '__main__':
    main()
