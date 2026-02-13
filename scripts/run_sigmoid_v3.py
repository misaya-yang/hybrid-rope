#!/usr/bin/env python3
"""
Sigmoid RoPE v3 - 重新设计的参数化方案

核心思路：
1. 在低维度匹配geometric的快速衰减（保持短序列性能）
2. 在高维度使用更平缓的衰减（改善长序列外推）

设计方案：
- sigmoid_modulated: 用sigmoid调制geometric，在高维度减缓衰减
- hybrid_geometric_sigmoid: 分段设计，低维度geometric + 高维度sigmoid
- anchored_sigmoid: 锚定关键频率点，确保性能边界
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH = '/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct'
OUTPUT_DIR = Path('/root/autodl-tmp/dfrope/hybrid-rope/results/sigmoid_v3')

def compute_inv_freq_geometric(head_dim, theta=100000):
    """标准geometric RoPE"""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

def compute_inv_freq_sigmoid_v3(head_dim, theta=100000, alpha=0.3):
    """
    Sigmoid v3: 调制geometric曲线
    
    思路：inv_freq = geo * sigmoid_modulation
    - 低维度：sigmoid ≈ 1，保持geometric特性
    - 高维度：sigmoid < 1，减缓衰减
    
    参数：
    - theta: 基础theta值
    - alpha: 调制强度 (0-1)，越大则高维度衰减越慢
    """
    geo = compute_inv_freq_geometric(head_dim, theta)
    dim = head_dim // 2
    positions = torch.arange(0, dim).float()
    
    # Sigmoid调制：在维度空间中从1渐变到(1-alpha)
    # 使用dim/2作为中点，使得高维度区域得到调制
    mid = dim / 2
    steepness = 4.0 / dim  # 陡峭度，使过渡平滑
    modulation = 1.0 - alpha * torch.sigmoid(steepness * (positions - mid))
    
    return geo * modulation

def compute_inv_freq_hybrid(head_dim, theta_geo=100000, theta_sig=500000, split_ratio=0.5):
    """
    Hybrid: 分段设计
    - 低维度 (0 ~ split_ratio*dim): 使用geometric with theta_geo
    - 高维度 (split_ratio*dim ~ dim): 使用geometric with theta_sig (更大theta=更慢衰减)
    
    这样低维度保持短序列性能，高维度改善长序列外推
    """
    dim = head_dim // 2
    split_point = int(dim * split_ratio)
    
    geo_low = compute_inv_freq_geometric(head_dim, theta_geo)
    geo_high = compute_inv_freq_geometric(head_dim, theta_sig)
    
    inv_freq = geo_low.clone()
    inv_freq[split_point:] = geo_high[split_point:]
    
    return inv_freq

def compute_inv_freq_anchored_sigmoid(head_dim, theta=100000, anchor_dim=16, anchor_factor=10.0):
    """
    Anchored Sigmoid: 锚定关键频率
    
    - 在低维度（< anchor_dim）保持geometric
    - 在高维度用sigmoid过渡到更大的有效theta
    - anchor_factor控制高维度theta的放大倍数
    
    物理直觉：低维度频率对短序列重要，高维度频率对长序列外推重要
    """
    dim = head_dim // 2
    geo_base = compute_inv_freq_geometric(head_dim, theta)
    geo_extended = compute_inv_freq_geometric(head_dim, theta * anchor_factor)
    
    # Sigmoid过渡
    positions = torch.arange(0, dim).float()
    transition_center = anchor_dim
    steepness = 0.5  # 平滑过渡
    weight = torch.sigmoid(steepness * (positions - transition_center))
    
    # 低维度用base，高维度用extended
    inv_freq = geo_base * (1 - weight) + geo_extended * weight
    
    return inv_freq

def apply_rope_patch(model, inv_freq):
    """应用RoPE补丁"""
    for name, module in model.named_modules():
        if 'rotary_emb' in name.lower():
            module.inv_freq = inv_freq.to(model.device)
            return True
    return False

@torch.no_grad()
def eval_ppl(model, tokens, length):
    """计算perplexity"""
    model.eval()
    window = tokens[:length].unsqueeze(0).cuda()
    outputs = model(window, labels=window)
    return np.exp(outputs.loss.item())

def run_config(config_name, inv_freq_fn, tokens, lengths, seed=42, windows=5):
    """运行单个配置"""
    print(f"\n{'='*60}\n  {config_name}\n{'='*60}")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True
    )
    
    # 获取head_dim并计算inv_freq
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    inv_freq = inv_freq_fn(head_dim)
    
    # 应用补丁
    if not apply_rope_patch(model, inv_freq):
        print("ERROR: Failed to apply RoPE patch!")
        return None
    
    print(f"  inv_freq range: [{inv_freq.min():.6e}, {inv_freq.max():.6e}]")
    
    results = {}
    for length in lengths:
        print(f"  Length {length}...")
        ppl_values = []
        for i in range(windows):
            torch.manual_seed(seed + i)
            np.random.seed(seed + i)
            max_start = len(tokens) - length
            if max_start > 0:
                start = np.random.randint(0, max_start)
                window = tokens[start:start+length]
            else:
                window = tokens[:length]
            ppl = eval_ppl(model, window, length)
            ppl_values.append(ppl)
            print(f"    Window {i+1}/{windows}: PPL={ppl:.3f}")
        results[length] = {
            'ppl_mean': float(np.mean(ppl_values)),
            'ppl_std': float(np.std(ppl_values))
        }
        print(f"  PPL@{length}: {results[length]['ppl_mean']:.3f} ± {results[length]['ppl_std']:.3f}")
    
    del model
    torch.cuda.empty_cache()
    return results

def main():
    print("="*60)
    print("Sigmoid RoPE v3 Experiment")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    text = '\n\n'.join(ds['text'])
    tokens = torch.tensor(tokenizer.encode(text, add_special_tokens=False)[:250000])
    print(f"Loaded {len(tokens)} tokens")
    
    # 测试配置
    lengths = [2048, 4096, 8192, 16384]
    
    configs = {
        'geo_100k': lambda hd: compute_inv_freq_geometric(hd, theta=100000),
        'geo_500k': lambda hd: compute_inv_freq_geometric(hd, theta=500000),
        'sigmoid_v3_alpha0.3': lambda hd: compute_inv_freq_sigmoid_v3(hd, theta=100000, alpha=0.3),
        'sigmoid_v3_alpha0.5': lambda hd: compute_inv_freq_sigmoid_v3(hd, theta=100000, alpha=0.5),
        'hybrid_50_500': lambda hd: compute_inv_freq_hybrid(hd, theta_geo=100000, theta_sig=500000, split_ratio=0.5),
        'anchored_x10': lambda hd: compute_inv_freq_anchored_sigmoid(hd, theta=100000, anchor_factor=10),
    }
    
    all_results = {}
    for config_name, inv_freq_fn in configs.items():
        results = run_config(config_name, inv_freq_fn, tokens, lengths)
        if results:
            all_results[config_name] = results
    
    # 计算collapse ratio
    output = {}
    for config_name, results in all_results.items():
        ppl_2k = results[2048]['ppl_mean']
        ppl_16k = results[16384]['ppl_mean']
        output[config_name] = {
            'ppl_2k': round(ppl_2k, 3),
            'ppl_4k': round(results[4096]['ppl_mean'], 3),
            'ppl_8k': round(results[8192]['ppl_mean'], 3),
            'ppl_16k': round(ppl_16k, 3),
            'collapse_ratio': round(ppl_16k/ppl_2k, 3)
        }
    
    # 保存结果
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # 生成报告
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Config':<25} {'PPL@2k':>10} {'PPL@16k':>12} {'Collapse':>10}")
    print("-"*60)
    for config_name, data in sorted(output.items(), key=lambda x: x[1]['collapse_ratio']):
        print(f"{config_name:<25} {data['ppl_2k']:>10.3f} {data['ppl_16k']:>12.3f} {data['collapse_ratio']:>10.3f}x")
    
    # 假设检验
    geo_100k_collapse = output.get('geo_100k', {}).get('collapse_ratio', float('inf'))
    best_sigmoid = min(
        [(k, v['collapse_ratio']) for k, v in output.items() if 'sigmoid' in k or 'hybrid' in k or 'anchored' in k],
        key=lambda x: x[1],
        default=(None, float('inf'))
    )
    
    print("\n" + "="*60)
    print("HYPOTHESIS TEST")
    print("="*60)
    print(f"geo_100k collapse: {geo_100k_collapse}x")
    if best_sigmoid[0]:
        print(f"Best new design ({best_sigmoid[0]}): {best_sigmoid[1]}x")
        if best_sigmoid[1] < geo_100k_collapse:
            print(f"✅ PASSED: New design is {geo_100k_collapse/best_sigmoid[1]:.2f}x more stable!")
        else:
            print(f"❌ FAILED: New design did not improve over geo_100k")

if __name__ == '__main__':
    main()