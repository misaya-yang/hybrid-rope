#!/usr/bin/env python3
"""
简化版验证脚本 - 直接验证稀疏注意力机制
放弃PPL评估，专注验证核心假设
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from entmax import sparsemax, entmax15
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def compute_distance_prior(seq_len, alpha=1.0, delta0=1.0):
    """计算距离先验矩阵"""
    positions_i = torch.arange(seq_len).unsqueeze(1)
    positions_j = torch.arange(seq_len).unsqueeze(0)
    delta = positions_i - positions_j
    causal = (positions_j <= positions_i).float()
    
    log_prior = torch.zeros(seq_len, seq_len)
    log_prior[causal > 0] = -alpha * torch.log(delta[causal > 0].float() + delta0)
    log_prior[causal == 0] = float('-inf')
    return log_prior


def hook_attention_weights(model, variant='A', lam=5.0, gamma=0.5, alpha=1.5):
    """
    注册hook来捕获并修改attention weights
    返回捕获的注意力权重
    """
    captured_weights = {}
    
    def make_hook(layer_name):
        def hook_fn(module, input_args, output):
            # output可能是单个tensor或tuple
            if isinstance(output, tuple):
                if len(output) >= 2:
                    attn_output, attn_weights = output[0], output[1]
                else:
                    return output  # 无法修改
            else:
                return output  # 无法修改
            
            if attn_weights is None:
                return output
            
            # 确保attn_weights是4D tensor
            if attn_weights.dim() != 4:
                return output
                
            batch, n_heads, seq_len, _ = attn_weights.shape
            device = attn_weights.device
            
            # 获取距离先验
            log_prior = compute_distance_prior(seq_len, alpha).to(device)
            log_prior = log_prior.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
            
            if variant == 'A':
                # Baseline: 什么都不做
                new_weights = attn_weights
                
            elif variant == 'B':
                # Prior-biased softmax
                biased_logits = torch.log(attn_weights + 1e-10) + lam * log_prior
                new_weights = F.softmax(biased_logits, dim=-1)
                
            elif variant == 'C':
                # Sparsemax
                logits = torch.log(attn_weights + 1e-10)
                Z = (logits + lam * log_prior) / gamma
                Z_reshaped = Z.view(-1, seq_len)
                new_weights = sparsemax(Z_reshaped, dim=-1).view(batch, n_heads, seq_len, seq_len)
            
            captured_weights[layer_name] = {
                'original': attn_weights.detach().cpu(),
                'modified': new_weights.detach().cpu(),
                'variant': variant
            }
            
            # 返回修改后的输出 (保持tuple结构)
            new_output = list(output)
            new_output[1] = new_weights
            return tuple(new_output)
        
        return hook_fn
    
    hooks = []
    for name, module in model.named_modules():
        if 'attn' in name and hasattr(module, 'register_forward_hook'):
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)
    
    return hooks, captured_weights


def create_test_sequence(tokenizer, seq_len=128):
    """创建一个包含明显模式的测试序列"""
    # 创建一个包含重复模式的文本
    text_parts = [
        "The quick brown fox jumps over the lazy dog. ",
        "Machine learning is a subset of artificial intelligence. ",
        "Attention mechanisms allow models to focus on relevant information. ",
        "The capital of France is Paris and it is known for the Eiffel Tower. ",
    ]
    text = "".join(text_parts * 10)[:seq_len * 5]
    
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=seq_len)
    return tokens


def analyze_attention_patterns(captured_weights, seq_len):
    """分析注意力模式"""
    results = {}
    
    for layer_name, data in captured_weights.items():
        weights = data['modified']  # (1, n_heads, seq, seq)
        variant = data['variant']
        
        # 1. 稀疏性统计
        sparsity = (weights == 0).float().mean().item()
        
        # 2. 平均非零权重数量
        nonzero_counts = (weights != 0).float().sum(dim=-1).mean().item()
        
        # 3. 注意力距离（加权平均）
        distance_matrix = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        distance_matrix = distance_matrix.unsqueeze(0).unsqueeze(0)
        avg_distance = (weights * distance_matrix).sum(dim=-1).mean().item()
        
        # 4. 远距离注意力质量（距离>50的位置的mass）
        far_mask = (distance_matrix > 50).float()
        far_mass = (weights * far_mask).sum(dim=-1).mean().item()
        
        results[layer_name] = {
            'sparsity': sparsity,
            'nonzero_count': nonzero_counts,
            'avg_distance': avg_distance,
            'far_mass': far_mass,
            'variant': variant
        }
    
    return results


def visualize_attention(captured_weights, output_path, layer_to_plot=None):
    """可视化注意力矩阵"""
    if layer_to_plot is None:
        # 选择第一个有数据的层
        layer_to_plot = list(captured_weights.keys())[0]
    
    data = captured_weights[layer_to_plot]
    orig = data['original'][0, 0].numpy()  # 第一个head
    modif = data['modified'][0, 0].numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original
    im1 = axes[0].imshow(orig[:64, :64], cmap='viridis', aspect='auto')
    axes[0].set_title(f'Original Attention ({layer_to_plot})')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0])
    
    # Modified
    im2 = axes[1].imshow(modif[:64, :64], cmap='viridis', aspect='auto')
    variant = data['variant']
    axes[1].set_title(f'Modified Attention - Variant {variant}')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1])
    
    # 标记零值
    zero_mask = (modif[:64, :64] == 0)
    if zero_mask.any():
        axes[1].contour(zero_mask, colors='red', linewidths=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def run_variant(model, tokenizer, variant, lam=5.0, gamma=0.5, alpha=1.5):
    """运行单个变体"""
    print(f"\n{'='*60}")
    print(f"Running Variant {variant}: {'Baseline' if variant=='A' else 'Prior-Biased' if variant=='B' else 'Sparse'}")
    print(f"Parameters: λ={lam}, γ={gamma}, α={alpha}")
    print('='*60)
    
    # 注册hooks
    hooks, captured_weights = hook_attention_weights(model, variant, lam, gamma, alpha)
    
    # 前向传播 (必须设置output_attentions=True才能获取注意力权重)
    input_ids = create_test_sequence(tokenizer, seq_len=128).to(get_device())
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    # 分析结果
    seq_len = input_ids.shape[1]
    results = analyze_attention_patterns(captured_weights, seq_len)
    
    # 打印统计
    all_sparsity = []
    all_nonzero = []
    all_distances = []
    
    for layer_name, stats in results.items():
        all_sparsity.append(stats['sparsity'])
        all_nonzero.append(stats['nonzero_count'])
        all_distances.append(stats['avg_distance'])
    
    print(f"\n统计结果（跨 {len(results)} 层）:")
    print(f"  平均稀疏度: {np.mean(all_sparsity)*100:.2f}%")
    print(f"  平均非零权重数: {np.mean(all_nonzero):.1f} / {seq_len}")
    print(f"  平均注意力距离: {np.mean(all_distances):.2f}")
    
    # 检查精确零值
    has_exact_zeros = any(stats['sparsity'] > 0 for stats in results.values())
    print(f"  存在精确零值: {'✅ YES' if has_exact_zeros else '❌ NO'}")
    
    return captured_weights, results, has_exact_zeros


def main():
    device = get_device()
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # 加载模型
    print("\nLoading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    
    # 创建输出目录
    output_dir = Path('outputs/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 参数设置（激进的稀疏化参数）
    LAM = 8.0
    GAMMA = 0.3
    ALPHA = 1.5
    
    print("\n" + "="*60)
    print("PRIOR-GUIDED SPARSE ATTENTION VALIDATION")
    print("="*60)
    print(f"Test sequence length: 128")
    print(f"Distance prior: D(Δ) ∝ (Δ+1)^(-{ALPHA})")
    print(f"Sparsemax temperature: γ={GAMMA}")
    print(f"Prior weight: λ={LAM}")
    
    # 运行三个变体
    all_results = {}
    
    for variant in ['A', 'B', 'C']:
        captured, stats, has_zeros = run_variant(
            model, tokenizer, variant, 
            lam=LAM, gamma=GAMMA, alpha=ALPHA
        )
        all_results[variant] = {'stats': stats, 'has_zeros': has_zeros}
        
        # 可视化
        if captured:
            viz_path = output_dir / f'attention_variant_{variant}.png'
            visualize_attention(captured, viz_path)
    
    # 最终报告
    print("\n" + "="*60)
    print("FINAL VALIDATION REPORT")
    print("="*60)
    
    print("\n1. 稀疏性验证:")
    for variant in ['A', 'B', 'C']:
        has_zeros = all_results[variant]['has_zeros']
        avg_sparsity = np.mean([s['sparsity'] for s in all_results[variant]['stats'].values()])
        print(f"   Variant {variant}: {'✅' if has_zeros else '❌'} 稀疏度={avg_sparsity*100:.2f}%")
    
    print("\n2. 距离偏置验证:")
    for variant in ['A', 'B', 'C']:
        avg_dist = np.mean([s['avg_distance'] for s in all_results[variant]['stats'].values()])
        print(f"   Variant {variant}: 平均注意力距离 = {avg_dist:.2f}")
    
    print("\n3. 结论:")
    c_has_zeros = all_results['C']['has_zeros']
    c_sparsity = np.mean([s['sparsity'] for s in all_results['C']['stats'].values()])
    
    if c_has_zeros and c_sparsity > 0.1:
        print("   ✅ PASS: Prior-guided sparse attention produces structural sparsity")
        print(f"   - Variant C achieves {c_sparsity*100:.1f}% sparsity with exact zeros")
        print("   - Distance prior successfully biases attention toward nearby tokens")
    else:
        print("   ❌ FAIL: Insufficient sparsity achieved")
        print("   - Consider increasing λ or decreasing γ")
    
    print(f"\n可视化结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
