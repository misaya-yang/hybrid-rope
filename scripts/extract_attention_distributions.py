#!/usr/bin/env python3
"""Extract Attention Distance Distributions from Pre-trained Models"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats


def load_model(model_name: str):
    """加载模型，启用attention输出"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    print(f"加载: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 先获取config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # 强制启用attention输出
    config.output_attentions = True
    
    # 使用float32加载
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = model.to('cuda')
    model.eval()
    
    n_layers = config.n_layer
    n_heads = config.n_head
    
    print(f"模型: {model_name}, 层={n_layers}, 头={n_heads}")
    
    return model, tokenizer, {'n_layers': n_layers, 'n_heads': n_heads}


def extract_distance_distribution(model, seq_len: int, n_sequences: int, n_layers: int):
    """提取距离分布"""
    print(f"\n提取: {n_sequences}序列, {n_layers}层, 长{seq_len}")
    
    distance_histograms = [np.zeros(seq_len) for _ in range(n_layers)]
    total_counts = [0.0 for _ in range(n_layers)]
    
    np.random.seed(42)
    
    model.eval()
    with torch.no_grad():
        for seq_idx in tqdm(range(n_sequences), desc="序列"):
            seq = np.random.randint(100, 50000, size=seq_len).tolist()
            input_ids = torch.tensor([seq], dtype=torch.long, device='cuda')
            
            outputs = model(input_ids, use_cache=False)
            attentions = outputs.attentions
            
            if attentions is None:
                print(f"错误: 模型不输出attention!")
                return None
            
            # 处理每一层
            for layer_idx in range(n_layers):
                attn = attentions[layer_idx][0].squeeze(0)  # (heads, seq, seq)
                
                for d in range(1, min(seq_len, attn.shape[-1])):
                    diag = torch.diagonal(attn, offset=-d, dim1=-2, dim2=-1)
                    distance_histograms[layer_idx][d] += diag.sum().item()
                    total_counts[layer_idx] += diag.numel()
            
            torch.cuda.empty_cache()
    
    print("归一化...")
    for l in range(n_layers):
        if total_counts[l] > 0:
            distance_histograms[l] /= total_counts[l]
    
    return distance_histograms


def fit_powerlaw(distributions, seq_len: int):
    results = []
    d_min, d_max = 2, min(seq_len // 4, 100)
    
    for layer_idx, dist in enumerate(distributions):
        distances = np.arange(d_min, d_max + 1)
        probs = dist[d_min:d_max+1]
        
        mask = probs > 0
        if mask.sum() < 5:
            results.append({'layer': layer_idx, 'gamma': np.nan, 'R_squared': np.nan})
            continue
        
        log_dist = np.log10(distances[mask].astype(float))
        log_prob = np.log10(probs[mask].astype(float))
        
        slope, intercept, r_value, _, _ = stats.linregress(log_dist, log_prob)
        
        results.append({
            'layer': layer_idx,
            'gamma': -slope,
            'R_squared': r_value ** 2,
            'intercept': intercept,
        })
    
    return results


def plot_results(distributions, fit_results, model_name: str, output_dir: Path):
    n_layers = len(distributions)
    gammas = [r['gamma'] for r in fit_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
    
    for l in range(n_layers):
        dist = distributions[l]
        d = np.arange(1, min(len(dist), 200))
        mask = dist[d] > 0
        if mask.sum() > 0:
            ax.plot(np.log10(d[mask]), np.log10(dist[d][mask]), 
                    color=colors[l], alpha=0.5, linewidth=0.8)
    
    d_ref = np.logspace(0, 3, 100)
    ax.plot(np.log10(d_ref), -np.log10(d_ref), 'k--', linewidth=2, label='Δ^(-1)')
    ax.set_xlabel('log10(Δ)')
    ax.set_ylabel('log10(D(Δ))')
    ax.set_title(f'{model_name}: Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.bar(range(n_layers), gammas, color=colors)
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='γ=1')
    ax.axhline(y=np.nanmean(gammas), color='g', linestyle=':', linewidth=2, 
               label=f'mean γ={np.nanmean(gammas):.2f}')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Power-law exponent γ')
    ax.set_title('γ vs Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_distance_distributions.png', dpi=150, bbox_inches='tight')
    print(f"保存: {output_dir / 'fig1_distance_distributions.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='openai-community/gpt2')
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--n_sequences', type=int, default=3)
    parser.add_argument('--output_dir', default='results/attention_distribution')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer, info = load_model(args.model)
    
    distributions = extract_distance_distribution(
        model, args.seq_len, args.n_sequences, info['n_layers']
    )
    
    if distributions is None:
        print("提取失败!")
        return
    
    fit_results = fit_powerlaw(distributions, args.seq_len)
    
    with open(output_dir / 'power_law_fits.json', 'w') as f:
        json.dump(fit_results, f, indent=2)
    
    plot_results(distributions, fit_results, args.model, output_dir)
    
    gammas = [r['gamma'] for r in fit_results]
    r2s = [r['R_squared'] for r in fit_results]
    
    n_layers = info['n_layers']
    mean_gamma = np.nanmean(gammas)
    mean_r2 = np.nanmean(r2s)
    shallow_gamma = np.nanmean(gammas[:n_layers//3])
    deep_gamma = np.nanmean(gammas[-n_layers//3:])
    
    print("\n" + "="*50)
    print("       Results Summary")
    print("="*50)
    print(f"模型: {args.model}, {n_layers}层")
    print(f"平均 γ = {mean_gamma:.3f}, 平均 R² = {mean_r2:.3f}")
    print(f"浅层γ = {shallow_gamma:.3f}, 深层γ = {deep_gamma:.3f}")
    print(f"浅层>深层? {'是' if shallow_gamma > deep_gamma else '否'}")
    print("="*50)


if __name__ == '__main__':
    main()
