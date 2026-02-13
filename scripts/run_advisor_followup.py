#!/usr/bin/env python3
"""
导师建议的后续实验 - 2026-02-14
A. θ上限对照: geo_1M/geo_2M vs anchored_x20
B. 边界扫描: 2k/8k/16k/24k/32k 曲线
C. 短长度检查: anchor_dim 在短序列上的效应
"""
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置
MODEL_PATH = '/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct'
DATA_NAME = 'wikitext-103-raw-v1'
DATA_SPLIT = 'validation'
MAX_TOKENS = 250000
SEED = 42
OUTPUT_BASE = Path('/root/autodl-tmp/dfrope/hybrid-rope/results/advisor_followup_2026-02-14')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_data(tokenizer):
    from datasets import load_from_disk, load_dataset
    # 尝试直接从缓存加载
    cache_path = Path.home() / '.cache' / 'huggingface' / 'datasets' / 'wikitext'
    try:
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=DATA_SPLIT, trust_remote_code=True)
    except:
        # 备选：直接加载缓存
        ds = load_from_disk(str(cache_path / 'wikitext-103-raw-v1'))
    text = '\n\n'.join(ds['text'])
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[:MAX_TOKENS]
    return torch.tensor(tokens, dtype=torch.long)

def compute_inv_freq_geometric(head_dim, theta):
    """标准几何RoPE"""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

def compute_inv_freq_anchored(head_dim, theta, anchor_factor, anchor_dim, slope):
    """Anchored sigmoid RoPE
    
    θ_eff(j) = θ · [1 + (α−1)·σ(k·(j−j0))]
    """
    d_half = head_dim // 2
    positions = torch.arange(0, d_half).float()
    
    # sigmoid transition centered at anchor_dim
    j0 = anchor_dim
    k = slope / 100.0  # slope scaling
    sigmoid_weight = torch.sigmoid(k * (positions - j0))
    
    # effective theta per dimension
    theta_eff = theta * (1.0 + (anchor_factor - 1.0) * sigmoid_weight)
    
    # inverse frequency
    return 1.0 / (theta_eff ** (2 * positions / head_dim))

def apply_rope_patch(model, inv_freq):
    for name, module in model.named_modules():
        if 'rotary_emb' in name.lower():
            module.inv_freq = inv_freq.to(model.device)
            break
    return model

@torch.no_grad()
def eval_ppl(model, tokens, length, seed=42):
    """评估单个长度点的PPL"""
    model.eval()
    set_seed(seed)
    
    max_start = len(tokens) - length
    if max_start > 0:
        start = np.random.randint(0, max_start)
    else:
        start = 0
    
    window = tokens[start:start+length].unsqueeze(0).cuda()
    outputs = model(window, labels=window)
    loss = outputs.loss.item()
    return np.exp(loss)

def run_config(model, tokens, config_name, lengths, seed, inv_freq_func, **kwargs):
    """运行单个配置在多个长度上"""
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    inv_freq = inv_freq_func(head_dim, **kwargs)
    model = apply_rope_patch(model, inv_freq)
    
    results = {}
    for length in lengths:
        ppl = eval_ppl(model, tokens, length, seed)
        results[length] = {
            'ppl': round(ppl, 3),
            'status': 'ok'
        }
        print(f"    L={length}: PPL={ppl:.3f}")
    
    return results

def task_a_theta_ceiling(tokens, tokenizer):
    """
    任务A: θ上限对照
    比较 geo_1M, geo_2M vs anchored_x20
    """
    print("\n" + "="*60)
    print("任务A: θ上限对照 (geo_1M, geo_2M vs anchored_x20)")
    print("="*60)
    
    lengths = [2048, 16384]
    output_dir = OUTPUT_BASE / 'task_a_theta_ceiling'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = {
        'geo_1M': {'func': compute_inv_freq_geometric, 'kwargs': {'theta': 1000000}},
        'geo_2M': {'func': compute_inv_freq_geometric, 'kwargs': {'theta': 2000000}},
        'anchored_x20': {'func': compute_inv_freq_anchored, 'kwargs': {
            'theta': 100000, 'anchor_factor': 20, 'anchor_dim': 16, 'slope': 0.5
        }},
    }
    
    results = {'meta': {'task': 'theta_ceiling', 'lengths': lengths, 'seed': SEED}, 'data': {}}
    
    for config_name, config in configs.items():
        print(f"\n  Config: {config_name}")
        print(f"    kwargs: {config['kwargs']}")
        
        # 每个配置加载新模型
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True
        )
        
        data = run_config(
            model, tokens, config_name, lengths, SEED,
            config['func'], **config['kwargs']
        )
        results['data'][config_name] = data
        
        del model
        torch.cuda.empty_cache()
    
    # 保存结果
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印摘要
    print("\n  摘要:")
    print(f"  {'Config':<15} {'PPL@2k':<10} {'PPL@16k':<10} {'Collapse':<10}")
    print(f"  {'-'*45}")
    for config_name, data in results['data'].items():
        ppl_2k = data[2048]['ppl']
        ppl_16k = data[16384]['ppl']
        collapse = ppl_16k / ppl_2k
        print(f"  {config_name:<15} {ppl_2k:<10.2f} {ppl_16k:<10.2f} {collapse:<10.2f}x")
    
    return results

def task_b_boundary_sweep(tokens, tokenizer):
    """
    任务B: 边界扫描
    geo_500k vs anchored_x20 在 2k/8k/16k/24k/32k 上
    """
    print("\n" + "="*60)
    print("任务B: 边界扫描 (2k/8k/16k/24k/32k)")
    print("="*60)
    
    lengths = [2048, 8192, 16384, 24576, 32768]
    output_dir = OUTPUT_BASE / 'task_b_boundary_sweep'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = {
        'geo_500k': {'func': compute_inv_freq_geometric, 'kwargs': {'theta': 500000}},
        'anchored_x20': {'func': compute_inv_freq_anchored, 'kwargs': {
            'theta': 100000, 'anchor_factor': 20, 'anchor_dim': 16, 'slope': 0.5
        }},
    }
    
    results = {'meta': {'task': 'boundary_sweep', 'lengths': lengths, 'seed': SEED}, 'data': {}}
    
    for config_name, config in configs.items():
        print(f"\n  Config: {config_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True
        )
        
        data = run_config(
            model, tokens, config_name, lengths, SEED,
            config['func'], **config['kwargs']
        )
        results['data'][config_name] = data
        
        del model
        torch.cuda.empty_cache()
    
    # 保存结果
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印摘要
    print("\n  摘要:")
    print(f"  {'Config':<15} " + " ".join([f"L={l:<8}" for l in lengths]))
    print(f"  {'-'*60}")
    for config_name, data in results['data'].items():
        ppls = [str(data[l]['ppl']) for l in lengths]
        print(f"  {config_name:<15} " + " ".join([f"{p:<8}" for p in ppls]))
    
    return results

def task_c_short_length_check(tokens, tokenizer):
    """
    任务C: 短长度检查 anchor_dim 效应
    anchored_x20_anchor0 vs anchored_x20_anchor16 在 128/256/512/1024/2048
    """
    print("\n" + "="*60)
    print("任务C: 短长度检查 anchor_dim 效应")
    print("="*60)
    
    lengths = [128, 256, 512, 1024, 2048]
    output_dir = OUTPUT_BASE / 'task_c_short_length'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = {
        'anchored_x20_anchor0': {'func': compute_inv_freq_anchored, 'kwargs': {
            'theta': 100000, 'anchor_factor': 20, 'anchor_dim': 0, 'slope': 0.5
        }},
        'anchored_x20_anchor16': {'func': compute_inv_freq_anchored, 'kwargs': {
            'theta': 100000, 'anchor_factor': 20, 'anchor_dim': 16, 'slope': 0.5
        }},
    }
    
    results = {'meta': {'task': 'short_length_anchor_dim', 'lengths': lengths, 'seed': SEED}, 'data': {}}
    
    for config_name, config in configs.items():
        print(f"\n  Config: {config_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True
        )
        
        data = run_config(
            model, tokens, config_name, lengths, SEED,
            config['func'], **config['kwargs']
        )
        results['data'][config_name] = data
        
        del model
        torch.cuda.empty_cache()
    
    # 保存结果
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印摘要
    print("\n  摘要:")
    print(f"  {'Config':<25} " + " ".join([f"L={l:<6}" for l in lengths]))
    print(f"  {'-'*60}")
    for config_name, data in results['data'].items():
        ppls = [str(data[l]['ppl']) for l in lengths]
        print(f"  {config_name:<25} " + " ".join([f"{p:<6}" for p in ppls]))
    
    return results

def main():
    print("="*60)
    print("导师建议的后续实验 - 2026-02-14")
    print("="*60)
    print(f"模型: {MODEL_PATH}")
    print(f"数据: {DATA_NAME}/{DATA_SPLIT}")
    print(f"Seed: {SEED}")
    print(f"输出目录: {OUTPUT_BASE}")
    
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\n加载 tokenizer 和数据...")
    tokenizer = load_tokenizer()
    tokens = load_data(tokenizer)
    print(f"加载了 {len(tokens)} tokens")
    
    # 执行三个任务
    results_a = task_a_theta_ceiling(tokens, tokenizer)
    results_b = task_b_boundary_sweep(tokens, tokenizer)
    results_c = task_c_short_length_check(tokens, tokenizer)
    
    # 生成总摘要
    summary = f"""# 导师建议的后续实验结果

## 任务A: θ上限对照
| Config | PPL@2k | PPL@16k | Collapse |
|--------|--------|---------|----------|
| geo_1M | {results_a['data']['geo_1M'][2048]['ppl']:.2f} | {results_a['data']['geo_1M'][16384]['ppl']:.2f} | {results_a['data']['geo_1M'][16384]['ppl']/results_a['data']['geo_1M'][2048]['ppl']:.2f}x |
| geo_2M | {results_a['data']['geo_2M'][2048]['ppl']:.2f} | {results_a['data']['geo_2M'][16384]['ppl']:.2f} | {results_a['data']['geo_2M'][16384]['ppl']/results_a['data']['geo_2M'][2048]['ppl']:.2f}x |
| anchored_x20 | {results_a['data']['anchored_x20'][2048]['ppl']:.2f} | {results_a['data']['anchored_x20'][16384]['ppl']:.2f} | {results_a['data']['anchored_x20'][16384]['ppl']/results_a['data']['anchored_x20'][2048]['ppl']:.2f}x |

## 任务B: 边界扫描
(见 task_b_boundary_sweep/results.json)

## 任务C: 短长度 anchor_dim 检查
(见 task_c_short_length/results.json)

## 结论
(待分析)
"""
    
    with open(OUTPUT_BASE / 'summary.md', 'w') as f:
        f.write(summary)
    
    print("\n" + "="*60)
    print("全部实验完成！")
    print("="*60)

if __name__ == '__main__':
    main()