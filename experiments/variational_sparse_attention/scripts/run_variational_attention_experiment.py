#!/usr/bin/env python3
"""
Prior-guided Variational Sparse Attention Experiment
验证在 M4 Max 上的可行性与收益

三种注意力变体:
A) Baseline: 原始 softmax attention
B) Prior-Biased Softmax: logits += λ * log D(Δ) 后 softmax
C) Prior-Guided Sparse Attention: sparsemax(Z/γ), Z = logits + λ log D(Δ)
"""

import os
import sys
import json
import csv
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from datasets import load_dataset
from entmax import sparsemax, entmax15

# 设置matplotlib后端以支持无头环境
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# 配置与常量
# ==============================================================================

SEED = 42
MAX_SEQ_LEN = 1024  # GPT2默认
DEFAULT_N_TOKENS = 200000  # 评估用的token数


def set_seed(seed: int = SEED):
    """设置所有随机种子以保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        # MPS没有明确的种子设置，但CPU种子已设置
        pass


def get_device():
    """获取可用设备 (优先 MPS for M4 Max)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==============================================================================
# 距离先验计算
# ==============================================================================

class DistancePrior:
    """
    距离先验 D(Δ) ∝ (Δ + δ0)^(-α)
    log_prior = -α * log(Δ + δ0)
    """
    
    def __init__(self, alpha: float = 1.0, delta0: float = 1.0, max_seq_len: int = MAX_SEQ_LEN):
        self.alpha = alpha
        self.delta0 = delta0
        self.max_seq_len = max_seq_len
        
        # 预计算距离矩阵和log prior (因果mask)
        # 位置i只能关注到位置j <= i
        # Δ = i - j (当前位置到被关注位置的距离)
        positions_i = torch.arange(max_seq_len).unsqueeze(1)  # (seq_len, 1)
        positions_j = torch.arange(max_seq_len).unsqueeze(0)  # (1, seq_len)
        
        # 距离矩阵: Δ[i,j] = i - j (如果j <= i), 否则inf
        self.delta_matrix = positions_i - positions_j  # (seq_len, seq_len)
        
        # 因果mask: 未来位置设为很大的负数
        self.causal_mask = (positions_j <= positions_i).float()
        
        # 计算 log prior
        # 对于有效位置 (j <= i): log D(Δ) = -α * log(Δ + δ0)
        # 对于未来位置: 设为 -inf
        self.log_prior = torch.zeros(max_seq_len, max_seq_len)
        valid_positions = self.causal_mask > 0
        self.log_prior[valid_positions] = -alpha * torch.log(
            self.delta_matrix[valid_positions].float() + delta0
        )
        # 未来位置设为非常小的值
        self.log_prior[self.causal_mask == 0] = float('-inf')
        
    def get_log_prior(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """获取指定长度的log prior矩阵"""
        return self.log_prior[:seq_len, :seq_len].to(device)
    
    def get_delta_matrix(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """获取指定长度的距离矩阵"""
        return self.delta_matrix[:seq_len, :seq_len].to(device)


# ==============================================================================
# 注意力变体实现
# ==============================================================================

def baseline_softmax_attention(
    attn_weights: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    A组: 原始softmax attention (baseline)
    attn_weights: (batch, n_heads, seq_len, seq_len)
    """
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    # 沿最后一个维度softmax
    attn_probs = F.softmax(attn_weights, dim=-1)
    return attn_probs


def prior_biased_softmax_attention(
    attn_weights: torch.Tensor,
    distance_prior: DistancePrior,
    lam: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    B组: Prior-Biased Softmax
    logits += λ * log D(Δ) 后 softmax
    """
    batch, n_heads, seq_len, _ = attn_weights.shape
    device = attn_weights.device
    
    # 获取log prior并扩展到batch和head维度
    log_prior = distance_prior.get_log_prior(seq_len, device)  # (seq_len, seq_len)
    log_prior = log_prior.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    # 应用prior bias
    biased_logits = attn_weights + lam * log_prior
    
    if attention_mask is not None:
        biased_logits = biased_logits + attention_mask
    
    attn_probs = F.softmax(biased_logits, dim=-1)
    return attn_probs


def prior_guided_sparse_attention(
    attn_weights: torch.Tensor,
    distance_prior: DistancePrior,
    lam: float = 1.0,
    gamma: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
    use_entmax15: bool = False
) -> torch.Tensor:
    """
    C组: Prior-Guided Sparse Attention
    sparsemax(Z/γ), Z = logits + λ log D(Δ)
    
    如果使用entmax15，则是1.5-entmax，介于softmax和sparsemax之间
    """
    batch, n_heads, seq_len, _ = attn_weights.shape
    device = attn_weights.device
    
    # 获取log prior
    log_prior = distance_prior.get_log_prior(seq_len, device)
    log_prior = log_prior.unsqueeze(0).unsqueeze(0)
    
    # 计算 Z
    Z = attn_weights + lam * log_prior
    
    if attention_mask is not None:
        Z = Z + attention_mask
    
    # 应用sparsemax或entmax
    # 需要reshape: (batch * n_heads * seq_len, seq_len)
    Z_reshaped = Z.view(-1, seq_len)
    
    if use_entmax15:
        probs_reshaped = entmax15(Z_reshaped / gamma, dim=-1)
    else:
        probs_reshaped = sparsemax(Z_reshaped / gamma, dim=-1)
    
    attn_probs = probs_reshaped.view(batch, n_heads, seq_len, seq_len)
    return attn_probs


# ==============================================================================
# GPT2 Attention Patch
# ==============================================================================

class PatchedGPT2Attention:
    """用于monkey patch GPT2 attention的类"""
    
    def __init__(
        self,
        model: GPT2LMHeadModel,
        variant: str,  # 'A', 'B', 'C'
        distance_prior: Optional[DistancePrior] = None,
        lam: float = 1.0,
        gamma: float = 1.0,
        use_entmax15: bool = False,
        save_stats: bool = False
    ):
        self.model = model
        self.variant = variant
        self.distance_prior = distance_prior
        self.lam = lam
        self.gamma = gamma
        self.use_entmax15 = use_entmax15
        self.save_stats = save_stats
        self.stats = defaultdict(list)
        self.original_forwards = {}
        
    def _create_attention_hook(self):
        """创建attention替换函数"""
        def hook_fn(module, input_args, output):
            # GPT2Attention forward输出 (attn_output, attn_weights, past_key_value)
            # 我们需要在中间修改attn_weights的计算
            # 但由于transformers内部已经计算好了，我们需要重新计算
            pass
        return hook_fn
    
    def patch_model(self):
        """
        Monkey patch GPT2 model的attention模块
        我们直接替换整个GPT2Attention._attn方法
        """
        for name, module in self.model.named_modules():
            if hasattr(module, '_attn'):
                # 保存原始方法
                self.original_forwards[name] = module._attn
                # 替换为新方法
                module._attn = self._create_patched_attn(module, name)
                
    def unpatch_model(self):
        """恢复原始方法"""
        for name, module in self.model.named_modules():
            if hasattr(module, '_attn') and name in self.original_forwards:
                module._attn = self.original_forwards[name]
    
    def _create_patched_attn(self, attn_module, module_name):
        """创建替换的_attn方法"""
        original_attn = attn_module._attn
        variant = self.variant
        prior = self.distance_prior
        lam = self.lam
        gamma = self.gamma
        use_entmax15 = self.use_entmax15
        save_stats = self.save_stats
        stats = self.stats
        
        def patched_attn(query, key, value, attention_mask=None, head_mask=None):
            """
            替换的attention计算
            query, key, value: (batch, n_heads, seq_len, head_dim)
            """
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            
            if attn_weights.size() != (query.size(0), query.size(1), query.size(2), key.size(2)):
                raise ValueError(f"Attention weights shape mismatch")
            
            # 缩放
            attn_weights = attn_weights / (value.size(-1) ** 0.5)
            
            # 应用不同的注意力变体
            if variant == 'A':
                # Baseline softmax
                attn_probs = baseline_softmax_attention(attn_weights, attention_mask)
                
            elif variant == 'B':
                # Prior-biased softmax
                attn_probs = prior_biased_softmax_attention(
                    attn_weights, prior, lam, attention_mask
                )
                
            elif variant == 'C':
                # Prior-guided sparse attention
                attn_probs = prior_guided_sparse_attention(
                    attn_weights, prior, lam, gamma, attention_mask, use_entmax15
                )
            else:
                raise ValueError(f"Unknown variant: {variant}")
            
            # 保存统计信息
            if save_stats:
                with torch.no_grad():
                    # 稀疏性统计
                    sparsity = (attn_probs == 0).float().mean().item()
                    stats[f'{module_name}_sparsity'].append(sparsity)
                    
                    # 非零数量
                    nonzero_count = (attn_probs != 0).float().sum(dim=-1).mean().item()
                    stats[f'{module_name}_nonzero'].append(nonzero_count)
                    
                    # 熵
                    entropy = -(attn_probs * (attn_probs + 1e-10).log()).sum(dim=-1).mean().item()
                    stats[f'{module_name}_entropy'].append(entropy)
            
            # 应用head mask
            if head_mask is not None:
                attn_probs = attn_probs * head_mask
            
            # 计算attention输出
            attn_output = torch.matmul(attn_probs, value)
            
            return attn_output, attn_probs
        
        return patched_attn


# ==============================================================================
# 数据加载
# ==============================================================================

class TokenizedTextDataset(Dataset):
    """简单的tokenized文本数据集"""
    
    def __init__(self, tokens: torch.Tensor, seq_len: int = MAX_SEQ_LEN):
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        return self.tokens[idx:idx + self.seq_len + 1]


def load_wikitext_data(
    tokenizer: GPT2Tokenizer,
    n_tokens: int = DEFAULT_N_TOKENS,
    split: str = "validation",
    cache_dir: Optional[str] = None
) -> torch.Tensor:
    """
    加载WikiText-2数据并tokenize
    返回: token ids tensor
    """
    print(f"Loading WikiText-2 {split} split...")
    
    # 加载数据集
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir)
    
    # 合并所有文本
    all_text = "\n\n".join([item["text"] for item in dataset if item["text"].strip()])
    
    # Tokenize
    print(f"Tokenizing text (first {n_tokens} tokens)...")
    tokens = tokenizer.encode(all_text, add_special_tokens=False)
    tokens = tokens[:n_tokens]
    
    print(f"Total tokens: {len(tokens)}")
    return torch.tensor(tokens, dtype=torch.long)


# ==============================================================================
# PPL 评估
# ==============================================================================

def evaluate_ppl(
    model: GPT2LMHeadModel,
    dataset: TokenizedTextDataset,
    device: torch.device,
    batch_size: int = 4,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    评估语言建模困惑度
    返回: {'ppl': float, 'loss': float, 'nll': float, 'n_tokens': int}
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch_tokens in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            batch_tokens = batch_tokens.to(device)
            
            # 输入是前seq_len个token，目标是后seq_len个token
            input_ids = batch_tokens[:, :-1]
            target_ids = batch_tokens[:, 1:]
            
            # Forward
            outputs = model(input_ids, labels=target_ids)
            
            # 使用logits手动计算loss（更可靠）
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 累计NLL
            n_tokens = shift_labels.numel()
            total_nll += loss.item()
            total_tokens += n_tokens
    
    avg_loss = total_nll / total_tokens
    ppl = math.exp(min(avg_loss, 20))  # 限制exp防止溢出
    
    return {
        'ppl': ppl,
        'loss': avg_loss,
        'nll': total_nll,
        'n_tokens': total_tokens
    }


# ==============================================================================
# 稀疏性与距离分析
# ==============================================================================

def analyze_sparsity_and_distance(
    model: GPT2LMHeadModel,
    dataset: TokenizedTextDataset,
    distance_prior: DistancePrior,
    device: torch.device,
    variant: str,
    lam: float,
    gamma: float,
    use_entmax15: bool = False,
    n_samples: int = 10,
    batch_size: int = 1
) -> Dict[str, np.ndarray]:
    """
    分析稀疏性和距离结构
    返回各类统计指标
    """
    model.eval()
    
    # 创建patch
    patcher = PatchedGPT2Attention(
        model, variant, distance_prior, lam, gamma, use_entmax15, save_stats=True
    )
    patcher.patch_model()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 收集统计信息
    all_layer_sparsity = []
    all_layer_entropy = []
    all_layer_nonzero = []
    
    with torch.no_grad():
        for batch_idx, batch_tokens in enumerate(dataloader):
            if batch_idx >= n_samples:
                break
            
            batch_tokens = batch_tokens.to(device)
            input_ids = batch_tokens[:, :-1]
            
            # Forward (会触发统计收集)
            _ = model(input_ids)
    
    # 恢复原始方法
    patcher.unpatch_model()
    
    # 整理统计信息 (按层聚合)
    stats = patcher.stats
    layer_stats = defaultdict(lambda: defaultdict(list))
    
    for key, values in stats.items():
        # key格式: "transformer.h.X.attn_sparsity"
        parts = key.split('.')
        if len(parts) >= 3 and parts[-2] == 'attn':
            layer_idx = parts[-3] if parts[-3].isdigit() else '0'
            metric = parts[-1]
            layer_stats[layer_idx][metric].extend(values)
    
    # 计算每层平均值
    results = {}
    for layer_idx in sorted(layer_stats.keys(), key=int):
        for metric, values in layer_stats[layer_idx].items():
            key = f"layer_{layer_idx}_{metric}"
            results[key] = np.mean(values)
    
    # 整体统计
    all_sparsity = [v for k, v in results.items() if 'sparsity' in k]
    all_entropy = [v for k, v in results.items() if 'entropy' in k]
    all_nonzero = [v for k, v in results.items() if 'nonzero' in k]
    
    results['avg_sparsity'] = np.mean(all_sparsity) if all_sparsity else 0.0
    results['avg_entropy'] = np.mean(all_entropy) if all_entropy else 0.0
    results['avg_nonzero'] = np.mean(all_nonzero) if all_nonzero else float(MAX_SEQ_LEN)
    
    return results


# ==============================================================================
# Needle-in-a-Haystack 测试
# ==============================================================================

class NeedleDataset:
    """合成Needle-in-a-Haystack数据"""
    
    NEEDLE_TOKEN = "MAGIC_NEEDLE_12345"
    
    def __init__(self, tokenizer: GPT2Tokenizer, seq_len: int = 4096):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.needle_ids = tokenizer.encode(self.NEEDLE_TOKEN, add_special_tokens=False)
        
    def create_sequence(self, haystack_text: str, needle_position: int) -> Tuple[torch.Tensor, int]:
        """
        创建包含needle的序列
        needle_position: needle在序列中的位置 (0到seq_len之间)
        返回: (token_ids, needle_actual_position)
        """
        # Tokenize haystack
        haystack_ids = self.tokenizer.encode(haystack_text, add_special_tokens=False)
        
        # 确保有足够长度
        needle_len = len(self.needle_ids)
        available_space = self.seq_len - needle_len
        
        # 截断或填充haystack
        if len(haystack_ids) > available_space:
            haystack_ids = haystack_ids[:available_space]
        
        # 插入needle
        insert_pos = min(needle_position, len(haystack_ids))
        sequence_ids = haystack_ids[:insert_pos] + self.needle_ids + haystack_ids[insert_pos:]
        
        # 截断到seq_len
        sequence_ids = sequence_ids[:self.seq_len]
        
        # 如果不够，用随机token填充
        while len(sequence_ids) < self.seq_len:
            sequence_ids.append(self.tokenizer.pad_token_id or 0)
        
        # 计算needle实际token位置范围
        needle_start = insert_pos
        needle_end = insert_pos + needle_len
        
        return torch.tensor(sequence_ids), (needle_start, needle_end)


def evaluate_needle_attention(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    distance_prior: DistancePrior,
    device: torch.device,
    variant: str,
    lam: float,
    gamma: float,
    use_entmax15: bool = False,
    seq_len: int = 4096,
    needle_positions: Optional[List[int]] = None
) -> Dict:
    """
    评估needle在attention中的权重
    返回needle在不同位置时的attention mass
    """
    model.eval()
    
    if needle_positions is None:
        # 测试不同距离的needle位置
        needle_positions = [100, 500, 1000, 2000, 3000, 3500]
    
    # 创建patch来捕获attention权重
    attention_weights_cache = {}
    
    def make_hook(layer_idx):
        def hook(module, input_args, output):
            # output: (attn_output, attn_weights, past_key_value)
            if len(output) > 1 and output[1] is not None:
                attention_weights_cache[layer_idx] = output[1].detach().cpu()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, '_attn'):
            layer_idx = name.split('.')[-2] if 'h.' in name else '0'
            hook = module.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)
    
    # 创建patch用于sparse attention
    patcher = PatchedGPT2Attention(
        model, variant, distance_prior, lam, gamma, use_entmax15, save_stats=False
    )
    patcher.patch_model()
    
    # 准备haystack文本
    haystack = "The quick brown fox jumps over the lazy dog. " * 500
    needle_dataset = NeedleDataset(tokenizer, seq_len)
    
    results = []
    
    for pos in needle_positions:
        if pos >= seq_len - 50:
            continue
            
        # 创建序列
        input_ids, needle_range = needle_dataset.create_sequence(haystack, pos)
        input_ids = input_ids.unsqueeze(0).to(device)
        
        attention_weights_cache.clear()
        
        with torch.no_grad():
            _ = model(input_ids)
        
        # 分析attention
        needle_start, needle_end = needle_range
        query_pos = input_ids.shape[1] - 1  # 最后一个位置作为query
        
        total_needle_mass = 0.0
        n_layers = len(attention_weights_cache)
        
        for layer_idx, attn_weights in attention_weights_cache.items():
            # attn_weights: (1, n_heads, seq_len, seq_len)
            # 取query_pos位置的attention分布
            attn_dist = attn_weights[0, :, query_pos, :]  # (n_heads, seq_len)
            
            # 计算needle位置的attention mass
            needle_mass = attn_dist[:, needle_start:needle_end].sum(dim=-1).mean().item()
            total_needle_mass += needle_mass
        
        avg_needle_mass = total_needle_mass / n_layers if n_layers > 0 else 0.0
        distance = query_pos - needle_end
        
        results.append({
            'needle_position': pos,
            'needle_distance': distance,
            'avg_attention_mass': avg_needle_mass,
            'is_nonzero': avg_needle_mass > 1e-6
        })
    
    # 清理
    for hook in hooks:
        hook.remove()
    patcher.unpatch_model()
    
    return {
        'results': results,
        'seq_len': seq_len,
        'variant': variant,
        'lam': lam,
        'gamma': gamma
    }


# ==============================================================================
# 可视化
# ==============================================================================

def plot_sparsity_comparison(
    stats_dict: Dict[str, Dict],
    output_path: Path
):
    """绘制稀疏性对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 按层绘制稀疏性
    ax1 = axes[0]
    for variant, stats in stats_dict.items():
        layer_sparsity = {k: v for k, v in stats.items() if 'layer_' in k and 'sparsity' in k}
        if layer_sparsity:
            layers = sorted(layer_sparsity.keys(), key=lambda x: int(x.split('_')[1]))
            values = [layer_sparsity[l] for l in layers]
            layer_nums = [int(l.split('_')[1]) for l in layers]
            ax1.plot(layer_nums, values, marker='o', label=f'Variant {variant}')
    
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Sparsity')
    ax1.set_title('Sparsity by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 平均稀疏性条形图
    ax2 = axes[1]
    variants = list(stats_dict.keys())
    avg_sparsity = [stats_dict[v].get('avg_sparsity', 0) for v in variants]
    colors = ['blue', 'orange', 'green']
    bars = ax2.bar(variants, avg_sparsity, color=colors[:len(variants)])
    ax2.set_ylabel('Average Sparsity')
    ax2.set_title('Average Attention Sparsity')
    ax2.set_ylim(0, 1)
    
    for bar, val in zip(bars, avg_sparsity):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sparsity plot to {output_path}")


def plot_needle_results(
    needle_results: Dict[str, Dict],
    output_path: Path
):
    """绘制needle测试结果"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'A': 'blue', 'B': 'orange', 'C': 'green'}
    markers = {'A': 'o', 'B': 's', 'C': '^'}
    
    for variant, data in needle_results.items():
        results = data['results']
        distances = [r['needle_distance'] for r in results]
        masses = [r['avg_attention_mass'] for r in results]
        
        ax.plot(distances, masses, marker=markers.get(variant, 'o'),
                label=f'Variant {variant}', color=colors.get(variant, None),
                linewidth=2, markersize=8)
    
    ax.set_xlabel('Needle Distance (tokens)')
    ax.set_ylabel('Average Attention Mass on Needle')
    ax.set_title('Needle-in-a-Haystack: Attention Mass vs Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved needle plot to {output_path}")


def plot_ppl_comparison(
    ppl_results: Dict[str, float],
    output_path: Path
):
    """绘制PPL对比图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    variants = list(ppl_results.keys())
    ppls = list(ppl_results.values())
    colors = ['blue', 'orange', 'green']
    
    bars = ax.bar(variants, ppls, color=colors[:len(variants)])
    ax.set_ylabel('Perplexity')
    ax.set_title('Language Modeling Perplexity Comparison')
    ax.set_xlabel('Attention Variant')
    
    # 添加数值标签
    for bar, val in zip(bars, ppls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    
    # 添加baseline参考线
    if 'A' in ppl_results:
        baseline = ppl_results['A']
        ax.axhline(y=baseline * 1.05, color='red', linestyle='--', alpha=0.5,
                   label='5% degradation threshold')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PPL plot to {output_path}")


# ==============================================================================
# 主实验流程
# ==============================================================================

def run_experiment(args):
    """运行完整实验"""
    
    # 设置
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存环境信息
    env_info = {
        'timestamp': timestamp,
        'device': str(device),
        'pytorch_version': torch.__version__,
        'seed': args.seed,
        'model_name': args.model_name,
        'alpha': args.alpha,
        'lam': args.lam,
        'gamma': args.gamma,
        'use_entmax15': args.use_entmax15
    }
    with open(output_dir / 'env.json', 'w') as f:
        json.dump(env_info, f, indent=2)
    
    # 加载模型和tokenizer
    print(f"\nLoading {args.model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    
    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建距离先验
    distance_prior = DistancePrior(
        alpha=args.alpha,
        delta0=args.delta0,
        max_seq_len=MAX_SEQ_LEN
    )
    
    # 加载数据
    print(f"\nLoading WikiText-2 data (first {args.n_tokens} tokens)...")
    tokens = load_wikitext_data(tokenizer, n_tokens=args.n_tokens, split="validation")
    dataset = TokenizedTextDataset(tokens, seq_len=args.seq_len)
    print(f"Dataset size: {len(dataset)} sequences")
    
    # ============= PPL 评估 =============
    print("\n" + "="*60)
    print("PHASE 1: Perplexity Evaluation")
    print("="*60)
    
    ppl_results = {}
    all_stats = {}
    
    for variant in ['A', 'B', 'C']:
        print(f"\n--- Variant {variant}: {get_variant_name(variant)} ---")
        
        # 创建patch
        use_entmax = args.use_entmax15 if variant == 'C' else False
        patcher = PatchedGPT2Attention(
            model, variant, distance_prior, args.lam, args.gamma, use_entmax
        )
        patcher.patch_model()
        
        # 评估PPL
        metrics = evaluate_ppl(
            model, dataset, device,
            batch_size=args.batch_size,
            max_batches=args.max_batches
        )
        
        ppl_results[variant] = metrics['ppl']
        
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  PPL:  {metrics['ppl']:.2f}")
        print(f"  Tokens evaluated: {metrics['n_tokens']}")
        
        # 稀疏性分析 (只针对B和C)
        if variant in ['B', 'C'] or args.analyze_all:
            print(f"  Analyzing sparsity...")
            stats = analyze_sparsity_and_distance(
                model, dataset, distance_prior, device,
                variant, args.lam, args.gamma, use_entmax,
                n_samples=args.n_sparsity_samples,
                batch_size=1
            )
            all_stats[variant] = stats
            print(f"  Average sparsity: {stats.get('avg_sparsity', 0):.4f}")
            print(f"  Average nonzero:  {stats.get('avg_nonzero', 0):.2f}")
        
        # 恢复原始模型
        patcher.unpatch_model()
        
        # 验证一致性
        if variant == 'A':
            print("  Verifying consistency (forward twice)...")
            patcher.patch_model()
            metrics2 = evaluate_ppl(
                model, dataset, device,
                batch_size=args.batch_size,
                max_batches=min(10, args.max_batches) if args.max_batches else 10
            )
            patcher.unpatch_model()
            
            diff = abs(metrics['loss'] - metrics2['loss'])
            print(f"  Loss difference between runs: {diff:.2e}")
            if diff > 1e-5:
                print("  WARNING: Inconsistency detected!")
    
    # 保存PPL结果
    with open(output_dir / 'ppl_results.json', 'w') as f:
        json.dump(ppl_results, f, indent=2)
    
    # 绘制PPL对比
    plot_ppl_comparison(ppl_results, output_dir / 'ppl_comparison.png')
    
    # 绘制稀疏性对比
    if all_stats:
        plot_sparsity_comparison(all_stats, output_dir / 'sparsity_comparison.png')
    
    # ============= Needle 测试 =============
    print("\n" + "="*60)
    print("PHASE 2: Needle-in-a-Haystack Test")
    print("="*60)
    
    needle_results = {}
    
    for variant in ['A', 'B', 'C']:
        print(f"\n--- Variant {variant} ---")
        
        use_entmax = args.use_entmax15 if variant == 'C' else False
        result = evaluate_needle_attention(
            model, tokenizer, distance_prior, device,
            variant, args.lam, args.gamma, use_entmax,
            seq_len=min(args.needle_seq_len, args.seq_len),
            needle_positions=args.needle_positions
        )
        
        needle_results[variant] = result
        
        print(f"  Sequence length: {result['seq_len']}")
        for r in result['results']:
            status = "✓" if r['is_nonzero'] else "✗"
            print(f"  Distance {r['needle_distance']:4d}: mass={r['avg_attention_mass']:.6f} {status}")
    
    # 保存needle结果
    with open(output_dir / 'needle_results.json', 'w') as f:
        json.dump(needle_results, f, indent=2)
    
    # 绘制needle结果
    plot_needle_results(needle_results, output_dir / 'needle_attention_mass.png')
    
    # ============= 最终报告 =============
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    
    baseline_ppl = ppl_results['A']
    
    print("\n1. Perplexity Results:")
    for variant, ppl in ppl_results.items():
        rel_change = (ppl - baseline_ppl) / baseline_ppl * 100
        print(f"   Variant {variant}: PPL = {ppl:.2f} ({rel_change:+.2f}%)")
    
    print("\n2. Sparsity Analysis:")
    for variant in ['B', 'C']:
        if variant in all_stats:
            stats = all_stats[variant]
            print(f"   Variant {variant}:")
            print(f"     - Average sparsity: {stats.get('avg_sparsity', 0)*100:.2f}%")
            print(f"     - Average nonzero weights: {stats.get('avg_nonzero', 0):.1f}")
    
    print("\n3. Needle Test:")
    for variant in ['A', 'B', 'C']:
        results = needle_results[variant]['results']
        nonzero_count = sum(1 for r in results if r['is_nonzero'])
        total = len(results)
        print(f"   Variant {variant}: {nonzero_count}/{total} positions retain needle")
    
    # 结论
    print("\n4. Conclusion:")
    c_ppl = ppl_results.get('C', baseline_ppl)
    c_degradation = (c_ppl - baseline_ppl) / baseline_ppl
    c_sparsity = all_stats.get('C', {}).get('avg_sparsity', 0)
    
    if c_degradation <= 0.05 and c_sparsity > 0.1:
        conclusion = (
            "✅ PASS: Prior-guided sparse attention is VIABLE.\n"
            f"   - PPL degradation: {c_degradation*100:.2f}% (threshold: 5%)\n"
            f"   - Achieved sparsity: {c_sparsity*100:.2f}%\n"
            "   - Structural sparsity with distance prior works without breaking LM performance."
        )
    elif c_degradation > 0.05:
        conclusion = (
            "❌ FAIL: PPL degradation exceeds threshold.\n"
            f"   - PPL degradation: {c_degradation*100:.2f}% (threshold: 5%)\n"
            "   - Need to tune λ, γ, or α hyperparameters."
        )
    else:
        conclusion = (
            "⚠️  PARTIAL: Low sparsity achieved.\n"
            f"   - Sparsity only: {c_sparsity*100:.2f}%\n"
            "   - Need to increase λ or decrease γ for more sparsity."
        )
    
    print(f"   {conclusion}")
    
    # 保存最终报告
    report = {
        'ppl_results': ppl_results,
        'sparsity_results': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv 
                                 for kk, vv in v.items()} 
                           for k, v in all_stats.items()},
        'needle_results': needle_results,
        'conclusion': conclusion,
        'parameters': vars(args)
    }
    
    with open(output_dir / 'final_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nAll results saved to: {output_dir}")
    
    return report


def get_variant_name(variant: str) -> str:
    """获取变体名称"""
    names = {
        'A': 'Baseline Softmax',
        'B': 'Prior-Biased Softmax',
        'C': 'Prior-Guided Sparse Attention'
    }
    return names.get(variant, variant)


# ==============================================================================
# 命令行入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prior-guided Variational Sparse Attention Experiment'
    )
    
    # 模型与数据
    parser.add_argument('--model-name', type=str, default='gpt2',
                        help='Model name (gpt2, gpt2-medium)')
    parser.add_argument('--n-tokens', type=int, default=DEFAULT_N_TOKENS,
                        help='Number of tokens to evaluate')
    parser.add_argument('--seq-len', type=int, default=MAX_SEQ_LEN,
                        help='Sequence length')
    
    # 距离先验参数
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Power-law exponent for distance prior')
    parser.add_argument('--delta0', type=float, default=1.0,
                        help='Offset for distance prior')
    parser.add_argument('--lam', type=float, default=5.0,
                        help='Weight for prior bias (λ)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Temperature for sparsemax (γ)')
    parser.add_argument('--use-entmax15', action='store_true',
                        help='Use entmax15 instead of sparsemax for variant C')
    
    # 实验控制
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Max batches to evaluate (None for all)')
    parser.add_argument('--n-sparsity-samples', type=int, default=10,
                        help='Number of samples for sparsity analysis')
    parser.add_argument('--analyze-all', action='store_true',
                        help='Analyze sparsity for all variants including A')
    
    # Needle测试参数
    parser.add_argument('--needle-seq-len', type=int, default=4096,
                        help='Sequence length for needle test')
    parser.add_argument('--needle-positions', type=int, nargs='+',
                        default=[100, 500, 1000, 2000, 3000],
                        help='Needle positions to test')
    
    # 输出与复现
    parser.add_argument('--output-dir', type=str, default='outputs/var_attn',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # 运行实验
    run_experiment(args)


if __name__ == '__main__':
    main()
