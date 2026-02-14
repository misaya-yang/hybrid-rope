#!/usr/bin/env python3
"""
评估训练好的700M模型
====================
使用wikitext等真实语料测试不同序列长度和RoPE配置
"""

import os
import sys
import json
import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def compute_freq_curve(freq_type, head_dim, theta_base=10000, **kwargs):
    """计算不同频率类型的inv_freq曲线"""
    dim = head_dim // 2
    
    if freq_type == 'orig':
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, dim, dtype=torch.float32) * 2 / head_dim))
        
    elif freq_type == 'geometric':
        scale = kwargs.get('scale', 1.0)
        inv_freq = 1.0 / ((theta_base * scale) ** (torch.arange(0, dim, dtype=torch.float32) * 2 / head_dim))
        
    elif freq_type == 'hybrid':
        alpha = kwargs.get('alpha', 0.2)
        p = kwargs.get('p', 3.9)
        omf = kwargs.get('omf', 0.3)
        t = torch.arange(0, dim, dtype=torch.float32) / dim
        poly = torch.pow(t + 1e-8, p)
        mixed = (1 - alpha) * t + alpha * poly
        inv_freq = omf / (theta_base ** mixed)
        
    elif freq_type == 'sigmoid':
        steepness = kwargs.get('steepness', 8.0)
        midpoint = kwargs.get('midpoint', 0.5)
        omf = kwargs.get('omf', 0.3)
        t = torch.arange(0, dim, dtype=torch.float32) / dim
        sigmoid_t = torch.sigmoid(steepness * (t - midpoint))
        inv_freq = omf / (theta_base ** sigmoid_t)
        
    else:
        raise ValueError(f"Unknown freq_type: {freq_type}")
    
    return inv_freq


def patch_rope_freq(model, freq_type, **kwargs):
    """修改模型的RoPE频率"""
    patched = []
    for name, module in model.named_modules():
        if 'rotary_emb' in name or name.endswith('.rotary_emb'):
            if hasattr(module, 'inv_freq'):
                head_dim = module.inv_freq.shape[0] * 2
                new_inv_freq = compute_freq_curve(freq_type, head_dim, **kwargs)
                module.inv_freq = nn.Parameter(new_inv_freq, requires_grad=False)
                patched.append(name)
                print(f"Patched {name}")
    return patched


def load_wikitext_data(tokenizer, seq_length=2048, max_samples=None):
    """加载wikitext数据"""
    from datasets import load_dataset
    
    print(f"Loading wikitext-2-raw-v1, seq_length={seq_length}")
    
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    except:
        # 如果网络不可用，使用本地缓存或生成简单数据
        print("Warning: Could not load wikitext, using generated data")
        texts = ["The quick brown fox jumps over the lazy dog. " * 1000 for _ in range(100)]
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": texts})
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_length,
            padding="max_length",
            return_tensors=None,
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    
    if max_samples:
        tokenized = tokenized.select(range(min(max_samples, len(tokenized))))
    
    return tokenized


def evaluate_ppl(model, dataset, device, batch_size=1):
    """计算困惑度"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    from torch.utils.data import DataLoader
    from transformers import default_data_collator
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_data_collator)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # 计算有效token数
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(min(avg_loss, 10))
    
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description="评估训练好的700M模型")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--freq_type", type=str, default="orig", choices=["orig", "geometric", "hybrid", "sigmoid"])
    parser.add_argument("--seq_lengths", type=str, default="2048,4096,8192", help="逗号分隔的序列长度")
    parser.add_argument("--max_samples", type=int, default=None, help="每个长度最大样本数")
    parser.add_argument("--output_dir", type=str, default="./results/eval_700m")
    
    # 频率参数
    parser.add_argument("--theta_base", type=float, default=100000)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--p", type=float, default=3.9)
    parser.add_argument("--omf", type=float, default=0.3)
    parser.add_argument("--steepness", type=float, default=8.0)
    parser.add_argument("--midpoint", type=float, default=0.5)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"eval_{args.freq_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"\nLoading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    
    # 应用频率修改
    freq_kwargs = {
        'theta_base': args.theta_base,
        'alpha': args.alpha,
        'p': args.p,
        'omf': args.omf,
        'steepness': args.steepness,
        'midpoint': args.midpoint,
    }
    
    if args.freq_type != 'orig':
        print(f"\nApplying {args.freq_type} frequency modification...")
        patched = patch_rope_freq(model, args.freq_type, **freq_kwargs)
        print(f"Patched {len(patched)} layers")
    
    # 评估
    results = {
        "model_path": args.model_path,
        "freq_type": args.freq_type,
        "freq_kwargs": freq_kwargs,
        "seq_lengths": {},
    }
    
    print(f"\n=== Evaluating with {args.freq_type} frequency ===")
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence length: {seq_len} ---")
        
        try:
            dataset = load_wikitext_data(tokenizer, seq_length=seq_len, max_samples=args.max_samples)
            print(f"Loaded {len(dataset)} samples")
            
            loss, ppl = evaluate_ppl(model, dataset, device)
            
            results["seq_lengths"][str(seq_len)] = {
                "loss": loss,
                "ppl": ppl,
                "num_samples": len(dataset),
            }
            print(f"Loss: {loss:.4f}, PPL: {ppl:.2f}")
            
        except Exception as e:
            print(f"Error at length {seq_len}: {e}")
            results["seq_lengths"][str(seq_len)] = {"error": str(e)}
    
    # 保存结果
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Results saved to {output_dir}")
    print(f"Freq type: {args.freq_type}")
    print("PPL by length:")
    for length, data in results["seq_lengths"].items():
        if "ppl" in data:
            print(f"  {length}: {data['ppl']:.2f}")


if __name__ == "__main__":
    main()