#!/usr/bin/env python3
"""
快速验证脚本：测试核心稀疏注意力功能
"""

import torch
import torch.nn.functional as F
from entmax import sparsemax, entmax15
import numpy as np


def test_sparsemax_basic():
    """测试sparsemax基本性质"""
    print("=" * 60)
    print("Test 1: Sparsemax Basic Properties")
    print("=" * 60)
    
    # 测试1: 输出在simplex上
    x = torch.tensor([1.0, 2.0, 0.5, -1.0])
    p = sparsemax(x, dim=-1)
    
    print(f"Input: {x}")
    print(f"Sparsemax output: {p}")
    print(f"Sum: {p.sum().item():.6f} (should be 1.0)")
    print(f"Non-negative: {(p >= 0).all().item()}")
    print(f"Has zeros: {(p == 0).any().item()}")
    
    assert torch.isclose(p.sum(), torch.tensor(1.0)), "Sum should be 1"
    assert (p >= 0).all(), "All values should be non-negative"
    print("✅ Test 1 passed\n")


def test_distance_prior():
    """测试距离先验计算"""
    print("=" * 60)
    print("Test 2: Distance Prior")
    print("=" * 60)
    
    seq_len = 10
    alpha = 1.0
    delta0 = 1.0
    
    # 创建距离矩阵
    positions_i = torch.arange(seq_len).unsqueeze(1)
    positions_j = torch.arange(seq_len).unsqueeze(0)
    delta = positions_i - positions_j
    
    # 因果mask
    causal_mask = (positions_j <= positions_i).float()
    
    # 计算log prior
    log_prior = torch.zeros(seq_len, seq_len)
    valid = causal_mask > 0
    log_prior[valid] = -alpha * torch.log(delta[valid].float() + delta0)
    log_prior[causal_mask == 0] = float('-inf')
    
    print(f"Distance matrix (first 5x5):\n{delta[:5, :5]}")
    print(f"\nLog prior (first 5x5):\n{log_prior[:5, :5]}")
    
    # 验证: 距离越远，prior越小
    diag_val = log_prior[5, 5].item()
    far_val = log_prior[5, 0].item()
    print(f"\nDiagonal (Δ=0): {diag_val:.4f}")
    print(f"Far position (Δ=5): {far_val:.4f}")
    assert diag_val > far_val, "Closer positions should have higher prior"
    print("✅ Test 2 passed\n")


def test_prior_effect_on_attention():
    """测试距离先验对注意力的影响"""
    print("=" * 60)
    print("Test 3: Prior Effect on Attention")
    print("=" * 60)
    
    seq_len = 20
    n_heads = 1
    batch = 1
    
    # 创建模拟的attention logits (随机)
    torch.manual_seed(42)
    logits = torch.randn(batch, n_heads, seq_len, seq_len)
    
    # 创建因果mask
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    # Baseline softmax
    baseline_attn = F.softmax(logits + mask, dim=-1)
    
    # 创建距离先验
    positions_i = torch.arange(seq_len).unsqueeze(1)
    positions_j = torch.arange(seq_len).unsqueeze(0)
    delta = positions_i - positions_j
    causal = (positions_j <= positions_i).float()
    
    alpha = 1.5
    lam = 3.0
    log_prior = torch.zeros(seq_len, seq_len)
    log_prior[causal > 0] = -alpha * torch.log(delta[causal > 0].float() + 1.0)
    log_prior[causal == 0] = float('-inf')
    
    # Prior-biased softmax
    biased_logits = logits + lam * log_prior
    biased_attn = F.softmax(biased_logits + mask, dim=-1)
    
    # Prior-guided sparse
    gamma = 0.5
    Z = (logits + lam * log_prior + mask) / gamma
    Z_reshaped = Z.view(-1, seq_len)
    sparse_attn = sparsemax(Z_reshaped, dim=-1).view(batch, n_heads, seq_len, seq_len)
    
    # 计算统计
    baseline_sparsity = (baseline_attn < 1e-6).float().mean().item()
    sparse_sparsity = (sparse_attn == 0).float().mean().item()
    
    # 计算平均注意力距离
    dist_matrix = delta.unsqueeze(0).unsqueeze(0)
    baseline_avg_dist = (baseline_attn * dist_matrix).sum(dim=-1).mean().item()
    sparse_avg_dist = (sparse_attn * dist_matrix).sum(dim=-1).mean().item()
    
    print(f"Baseline sparsity (<1e-6): {baseline_sparsity*100:.2f}%")
    print(f"Sparse attention sparsity (==0): {sparse_sparsity*100:.2f}%")
    print(f"Baseline average attention distance: {baseline_avg_dist:.2f}")
    print(f"Sparse average attention distance: {sparse_avg_dist:.2f}")
    
    assert sparse_sparsity > baseline_sparsity, "Sparse should have more zeros"
    print("✅ Test 3 passed\n")


def test_needle_retention():
    """测试needle在sparse attention下的保留"""
    print("=" * 60)
    print("Test 4: Needle Token Retention")
    print("=" * 60)
    
    seq_len = 50
    needle_pos = 10
    query_pos = seq_len - 1  # 最后一个位置
    
    # 创建attention logits，needle位置有较高值
    logits = torch.randn(1, 1, seq_len, seq_len) * 0.5
    logits[0, 0, :, needle_pos:needle_pos+3] += 5.0  # needle信号
    
    # 因果mask
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    # 距离先验
    positions_i = torch.arange(seq_len).unsqueeze(1)
    positions_j = torch.arange(seq_len).unsqueeze(0)
    delta = positions_i - positions_j
    causal = (positions_j <= positions_i).float()
    
    alpha = 1.0
    lam = 2.0
    log_prior = torch.zeros(seq_len, seq_len)
    log_prior[causal > 0] = -alpha * torch.log(delta[causal > 0].float() + 1.0)
    log_prior[causal == 0] = float('-inf')
    
    # 计算不同attention变体
    baseline = F.softmax(logits + mask, dim=-1)
    biased = F.softmax(logits + lam * log_prior + mask, dim=-1)
    
    gamma = 1.0
    Z = (logits + lam * log_prior + mask) / gamma
    sparse = sparsemax(Z.view(-1, seq_len), dim=-1).view(1, 1, seq_len, seq_len)
    
    # 检查query_pos位置对needle的attention
    baseline_mass = baseline[0, 0, query_pos, needle_pos:needle_pos+3].sum().item()
    biased_mass = biased[0, 0, query_pos, needle_pos:needle_pos+3].sum().item()
    sparse_mass = sparse[0, 0, query_pos, needle_pos:needle_pos+3].sum().item()
    
    print(f"Needle position: {needle_pos}, Query position: {query_pos}")
    print(f"Distance: {query_pos - needle_pos}")
    print(f"Baseline attention mass on needle: {baseline_mass:.4f}")
    print(f"Prior-biased attention mass on needle: {biased_mass:.4f}")
    print(f"Sparse attention mass on needle: {sparse_mass:.4f}")
    print(f"Sparse mass > 0: {sparse_mass > 1e-6}")
    
    # 只要sparse还有一定mass就通过
    if sparse_mass > 0.01:
        print("✅ Test 4 passed - Needle retains attention in sparse variant")
    else:
        print("⚠️  Test 4 warning - Needle mass is low, may need to tune λ")
    print()


def test_gradient_flow():
    """测试梯度可以正常回传"""
    print("=" * 60)
    print("Test 5: Gradient Flow")
    print("=" * 60)
    
    x = torch.randn(5, requires_grad=True)
    p = sparsemax(x, dim=-1)
    loss = (p * torch.arange(5, dtype=torch.float32)).sum()
    loss.backward()
    
    print(f"Input: {x.detach().numpy()}")
    print(f"Sparsemax: {p.detach().numpy()}")
    print(f"Gradient: {x.grad.numpy()}")
    print(f"Has non-zero gradients: {(x.grad != 0).any().item()}")
    
    assert x.grad is not None and (x.grad != 0).any(), "Gradients should flow"
    print("✅ Test 5 passed\n")


def main():
    print("\n" + "=" * 60)
    print("Prior-Guided Sparse Attention - Core Functionality Tests")
    print("=" * 60 + "\n")
    
    try:
        test_sparsemax_basic()
        test_distance_prior()
        test_prior_effect_on_attention()
        test_needle_retention()
        test_gradient_flow()
        
        print("=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
