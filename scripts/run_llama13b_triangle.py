#!/usr/bin/env python3
"""
LLaMA-13B 三角对照 + 边界曲线实验
目标：验证 anchored_x20 在更大模型上的效果

三角配置：
1) geo_500k
2) geo_2M  
3) anchored_x20: theta=100k, factor=20, slope=0.5, dim=16

输出：results/llama13b_triangle_boundary/
"""
import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ 配置 ============
# 根据服务器选择模型路径
MODEL_CANDIDATES = [
    '/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct',  # 8B fallback
    '/root/autodl-tmp/models/llama-13b',
    '/root/autodl-tmp/models/Llama-2-13b-hf',
    '/root/models/llama-13b',
]

OUTPUT_BASE = Path('/root/autodl-tmp/dfrope/hybrid-rope/results/llama13b_triangle_boundary')
DATA_NAME = 'wikitext'
DATA_CONFIG = 'wikitext-103-raw-v1'
DATA_SPLIT = 'validation'
MAX_TOKENS = 300000
DEFAULT_SEED = 42

# 固定长度序列
LENGTHS = [2048, 8192, 16384, 24576, 32768, 49152]

# 三角对照配置
CONFIGS = {
    'geo_500k': {
        'type': 'geometric',
        'theta': 500000,
    },
    'geo_2M': {
        'type': 'geometric', 
        'theta': 2000000,
    },
    'anchored_x20': {
        'type': 'anchored',
        'theta': 100000,
        'anchor_factor': 20,
        'anchor_dim': 16,
        'slope': 0.5,
    },
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def log(stage, msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}][{stage}] {msg}")
    sys.stdout.flush()

def find_model():
    """查找可用的模型"""
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            log("MODEL", f"Found model at {path}")
            return path
    raise FileNotFoundError("No LLaMA model found!")

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_data(tokenizer, max_tokens=None):
    from datasets import load_dataset
    ds = load_dataset(DATA_NAME, DATA_CONFIG, split=DATA_SPLIT, trust_remote_code=True)
    text = '\n\n'.join(ds['text'])
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[:max_tokens] if max_tokens else tokens[:MAX_TOKENS]
    log("DATA", f"Loaded {len(tokens)} tokens from {DATA_NAME}")
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
    """应用RoPE patch"""
    for name, module in model.named_modules():
        if 'rotary_emb' in name.lower():
            module.inv_freq = inv_freq.to(model.device)
            break
    return model

def load_model(model_path):
    """加载模型"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map='auto', 
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    mem_used = torch.cuda.max_memory_allocated() / 1024**3
    log("MODEL", f"Model loaded, GPU memory: {mem_used:.2f}GB")
    return model, mem_used

@torch.no_grad()
def eval_ppl(model, tokens, length, seed=42):
    """评估PPL"""
    model.eval()
    set_seed(seed)
    
    max_start = len(tokens) - length
    if max_start > 0:
        start = np.random.randint(0, max_start)
    else:
        start = 0
    
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    
    window = tokens[start:start+length].unsqueeze(0).cuda()
    
    try:
        outputs = model(window, labels=window)
        loss = outputs.loss.item()
        ppl = np.exp(loss)
        
        elapsed = time.time() - t0
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3
        
        return {
            'ppl': round(ppl, 3),
            'loss': round(loss, 4),
            'elapsed_sec': round(elapsed, 2),
            'mem_peak_gb': round(mem_peak, 2),
            'status': 'ok'
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {
            'status': 'oom',
            'error': str(e)[:200]
        }
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)[:200]
        }

def run_config(name, config, tokens, model_path, lengths):
    """运行单个配置"""
    log("CONFIG", f"\n{'='*60}")
    log("CONFIG", f"Running: {name}")
    log("CONFIG", f"Params: {config}")
    log("CONFIG", f"{'='*60}")
    
    results = {
        'name': name,
        'config': config,
        'data': {},
        'max_length': 0,
        'oom_at': None
    }
    
    # 加载模型
    try:
        model, mem = load_model(model_path)
    except Exception as e:
        log("ERROR", f"Failed to load model: {e}")
        results['error'] = str(e)
        return results
    
    # 计算inv_freq
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    if config['type'] == 'geometric':
        inv_freq = compute_inv_freq_geometric(head_dim, config['theta'])
    else:  # anchored
        inv_freq = compute_inv_freq_anchored(
            head_dim, 
            config['theta'],
            config['anchor_factor'],
            config['anchor_dim'],
            config['slope']
        )
    
    model = apply_rope_patch(model, inv_freq)
    
    # 依次评估各长度
    for L in lengths:
        log("EVAL", f"  L={L}...")
        r = eval_ppl(model, tokens, L, seed=DEFAULT_SEED)
        
        if r['status'] == 'ok':
            log("EVAL", f"  L={L}: PPL={r['ppl']:.3f}, mem={r['mem_peak_gb']:.2f}GB")
            results['data'][str(L)] = r
            results['max_length'] = L
        elif r['status'] == 'oom':
            log("EVAL", f"  L={L}: OOM! Stopping longer lengths for this config.")
            results['data'][str(L)] = r
            results['oom_at'] = L
            break
        else:
            log("EVAL", f"  L={L}: FAILED - {r.get('error', 'unknown')[:50]}")
            results['data'][str(L)] = r
            break
    
    del model
    torch.cuda.empty_cache()
    return results

def compute_collapse_ratio(results):
    """计算collapse ratio"""
    # 找到2048的PPL作为baseline
    baseline_ppl = None
    for name, r in results.items():
        if '2048' in r.get('data', {}):
            baseline_ppl = r['data']['2048'].get('ppl')
            break
    
    if baseline_ppl is None:
        return
    
    for name, r in results.items():
        r['collapse_ratio'] = {}
        for L_str, data in r.get('data', {}).items():
            if data.get('status') == 'ok' and 'ppl' in data:
                L = int(L_str)
                r['collapse_ratio'][L_str] = round(data['ppl'] / baseline_ppl, 3)

def estimate_boundary(results):
    """估计边界 L* (首次 ppl(L) > ppl(2k)*5 或 ppl(L) > 100)"""
    boundaries = {}
    
    for name, r in results.items():
        ppl_2k = r.get('data', {}).get('2048', {}).get('ppl', 11.0)
        threshold1 = ppl_2k * 5
        threshold2 = 100
        threshold = min(threshold1, threshold2)
        
        boundary = None
        for L_str in sorted([int(x) for x in r.get('data', {}).keys()]):
            L = int(L_str)
            data = r['data'].get(str(L), {})
            if data.get('status') == 'ok':
                ppl = data.get('ppl', 0)
                if ppl > threshold:
                    boundary = L
                    break
        
        boundaries[name] = {
            'threshold_used': threshold,
            'boundary_L': boundary
        }
        r['boundary'] = boundary
    
    return boundaries

def plot_results(results, output_dir):
    """绘制图表"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'geo_500k': 'red', 'geo_2M': 'blue', 'anchored_x20': 'green'}
    markers = {'geo_500k': 'o', 'geo_2M': 's', 'anchored_x20': '^'}
    
    # 图1: PPL vs Length (log scale)
    ax1 = axes[0]
    for name, r in results.items():
        lengths = []
        ppls = []
        for L_str in sorted([int(x) for x in r.get('data', {}).keys()]):
            data = r['data'].get(str(L_str), {})
            if data.get('status') == 'ok' and 'ppl' in data:
                lengths.append(int(L_str))
                ppls.append(data['ppl'])
        
        if lengths:
            ax1.semilogy(lengths, ppls, 
                        marker=markers.get(name, 'o'),
                        color=colors.get(name, 'black'),
                        label=name,
                        linewidth=2,
                        markersize=8)
    
    ax1.set_xlabel('Context Length', fontsize=12)
    ax1.set_ylabel('Perplexity (log scale)', fontsize=12)
    ax1.set_title('LLaMA-13B: PPL vs Context Length', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: Normalized Collapse Ratio
    ax2 = axes[1]
    for name, r in results.items():
        if 'collapse_ratio' not in r:
            continue
        lengths = []
        ratios = []
        for L_str in sorted([int(x) for x in r['collapse_ratio'].keys()]):
            lengths.append(int(L_str))
            ratios.append(r['collapse_ratio'][str(L_str)])
        
        if lengths:
            ax2.plot(lengths, ratios,
                    marker=markers.get(name, 'o'),
                    color=colors.get(name, 'black'),
                    label=name,
                    linewidth=2,
                    markersize=8)
    
    ax2.axhline(y=5, color='gray', linestyle='--', label='Collapse threshold (5x)')
    ax2.set_xlabel('Context Length', fontsize=12)
    ax2.set_ylabel('PPL(L) / PPL(2k)', fontsize=12)
    ax2.set_title('Normalized Collapse Ratio', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    fig_path = output_dir / 'figures' / 'ppl_vs_length.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'figures' / 'normalized_collapse_vs_length.png', dpi=150, bbox_inches='tight')
    log("PLOT", f"Saved figures to {fig_path.parent}")

def generate_summary(results, model_path, output_dir):
    """生成summary.md"""
    lines = [
        "# LLaMA-13B 三角对照 + 边界曲线实验",
        "",
        f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**模型**: {model_path}",
        f"**数据**: {DATA_NAME}/{DATA_CONFIG} ({DATA_SPLIT})",
        f"**Seed**: {DEFAULT_SEED}",
        "",
        "## 资源设置",
        "- batch_size: 1",
        "- dtype: float16",
        f"- 测试长度: {LENGTHS}",
        "",
        "## 三角对照配置",
        "",
        "| 配置 | 类型 | 参数 |",
        "|------|------|------|",
    ]
    
    for name, config in CONFIGS.items():
        if config['type'] == 'geometric':
            params = f"theta={config['theta']}"
        else:
            params = f"theta={config['theta']}, factor={config['anchor_factor']}, dim={config['anchor_dim']}, slope={config['slope']}"
        lines.append(f"| {name} | {config['type']} | {params} |")
    
    lines.extend([
        "",
        "## 关键结果",
        "",
        "### PPL @ 各长度",
        "",
        "| 配置 | 2k | 8k | 16k | 24k | 32k | 49k |",
        "|------|-----|-----|------|------|------|------|",
    ])
    
    for name in ['geo_500k', 'geo_2M', 'anchored_x20']:
        r = results.get(name, {})
        row = [name]
        for L in [2048, 8192, 16384, 24576, 32768, 49152]:
            data = r.get('data', {}).get(str(L), {})
            if data.get('status') == 'ok':
                row.append(f"{data['ppl']:.2f}")
            elif data.get('status') == 'oom':
                row.append('OOM')
            else:
                row.append('-')
        lines.append("| " + " | ".join(row) + " |")
    
    lines.extend([
        "",
        "### Collapse Ratio (PPL(L)/PPL(2k))",
        "",
        "| 配置 | 8k | 16k | 24k | 32k |",
        "|------|-----|------|------|------|",
    ])
    
    for name in ['geo_500k', 'geo_2M', 'anchored_x20']:
        r = results.get(name, {})
        row = [name]
        for L in [8192, 16384, 24576, 32768]:
            ratio = r.get('collapse_ratio', {}).get(str(L))
            if ratio:
                row.append(f"{ratio:.2f}x")
            else:
                row.append('-')
        lines.append("| " + " | ".join(row) + " |")
    
    # 边界分析
    lines.extend([
        "",
        "### 边界估计 (首次 PPL > 5x baseline 或 PPL > 100)",
        "",
        "| 配置 | 边界长度 |",
        "|------|----------|",
    ])
    
    for name in ['geo_500k', 'geo_2M', 'anchored_x20']:
        boundary = results.get(name, {}).get('boundary')
        lines.append(f"| {name} | {boundary if boundary else '未达到'} |")
    
    # 结论
    lines.extend([
        "",
        "## 结论",
        "",
        "### (i) anchored_x20 是否在 13B 上压平退化/后移边界？",
        "",
    ])
    
    # 自动生成结论
    geo_500k_boundary = results.get('geo_500k', {}).get('boundary')
    anchored_boundary = results.get('anchored_x20', {}).get('boundary')
    
    if geo_500k_boundary and anchored_boundary:
        shift = (anchored_boundary - geo_500k_boundary) / geo_500k_boundary * 100
        lines.append(f"- geo_500k 边界: {geo_500k_boundary}")
        lines.append(f"- anchored_x20 边界: {anchored_boundary}")
        if shift > 0:
            lines.append(f"- **anchored_x20 边界后移 {shift:.0f}%** ✅")
        else:
            lines.append(f"- anchored_x20 边界无改善")
    
    lines.extend([
        "",
        "### (ii) geo_2M 能否完全替代 anchored？",
        "",
    ])
    
    geo_2m_16k = results.get('geo_2M', {}).get('data', {}).get('16384', {}).get('ppl')
    anchored_16k = results.get('anchored_x20', {}).get('data', {}).get('16384', {}).get('ppl')
    
    if geo_2m_16k and anchored_16k:
        diff = geo_2m_16k - anchored_16k
        if diff > 5:
            lines.append(f"- 16k处: geo_2M PPL={geo_2m_16k:.2f}, anchored_x20 PPL={anchored_16k:.2f}")
            lines.append(f"- **anchored_x20 优于 geo_2M {diff:.2f} PPL点**")
            lines.append("- geo_2M 不能完全替代 anchored")
        else:
            lines.append(f"- 16k处差异较小 ({diff:.2f})")
    
    lines.extend([
        "",
        "---",
        f"*Generated by run_llama13b_triangle.py at {datetime.now().isoformat()}*"
    ])
    
    summary_text = "\n".join(lines)
    summary_path = output_dir / 'summary.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    log("SUMMARY", f"Summary saved to {summary_path}")
    return summary_text

def main():
    log("MAIN", "="*60)
    log("MAIN", "LLaMA-13B 三角对照 + 边界曲线实验")
    log("MAIN", "="*60)
    
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / 'figures').mkdir(parents=True, exist_ok=True)
    
    # 查找模型
    model_path = find_model()
    log("MAIN", f"Using model: {model_path}")
    
    # 加载数据
    tokenizer = load_tokenizer(model_path)
    tokens = load_data(tokenizer)
    
    # 运行三组配置
    all_results = {}
    
    for name, config in CONFIGS.items():
        r = run_config(name, config, tokens, model_path, LENGTHS)
        all_results[name] = r
        
        # 保存中间结果
        with open(OUTPUT_BASE / 'results.json', 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 计算collapse ratio
    compute_collapse_ratio(all_results)
    
    # 估计边界
    boundaries = estimate_boundary(all_results)
    log("MAIN", f"Boundaries: {boundaries}")
    
    # 保存最终结果
    with open(OUTPUT_BASE / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 绘图
    plot_results(all_results, OUTPUT_BASE)
    
    # 生成summary
    summary = generate_summary(all_results, model_path, OUTPUT_BASE)
    
    log("MAIN", "="*60)
    log("MAIN", "实验完成！")
    log("MAIN", "="*60)
    print(summary)

if __name__ == '__main__':
    main()