#!/usr/bin/env python3
"""
训练阶段频率对比实验
====================
关键问题: 训练时确定频率 vs 推理时即插即用，哪个更好？

实验设计:
- 模型: 350M 或 500M
- 配置: baseline, hybrid, geometric
- 训练: 长文本数据，序列长度8k
- 评估: 2k-64k PPL

资源需求:
- 350M @ seq8k: ~20GB
- 500M @ seq8k: ~30GB
- 96GB显存充足
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
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType


def compute_freq_curve(freq_type, head_dim, theta_base=10000, **kwargs):
    """
    计算不同频率类型的inv_freq曲线
    
    Args:
        freq_type: 'orig', 'geometric', 'hybrid', 'sigmoid'
        head_dim: 注意力头维度
        theta_base: 基础theta值
        **kwargs: 额外参数 (alpha, steepness, midpoint, omf, p)
    """
    dim = head_dim // 2
    
    if freq_type == 'orig':
        # 原始RoPE频率
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, dim, dtype=torch.float32) * 2 / head_dim))
        
    elif freq_type == 'geometric':
        # 几何缩放
        scale = kwargs.get('scale', 1.0)
        inv_freq = 1.0 / ((theta_base * scale) ** (torch.arange(0, dim, dtype=torch.float32) * 2 / head_dim))
        
    elif freq_type == 'hybrid':
        # 混合频率: 多项式 + 可调参数
        alpha = kwargs.get('alpha', 0.2)
        p = kwargs.get('p', 3.9)
        omf = kwargs.get('omf', 0.3)  # outer multiplier factor
        
        t = torch.arange(0, dim, dtype=torch.float32) / dim
        # 多项式部分
        poly = torch.pow(t + 1e-8, p)
        # 混合
        mixed = (1 - alpha) * t + alpha * poly
        # 转换为inv_freq
        inv_freq = omf / (theta_base ** mixed)
        
    elif freq_type == 'sigmoid':
        # Sigmoid频率
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
    """
    修改模型的RoPE频率
    """
    patched = []
    
    for name, module in model.named_modules():
        if 'rotary_emb' in name or name.endswith('.rotary_emb'):
            if hasattr(module, 'inv_freq'):
                # 获取head_dim
                head_dim = module.inv_freq.shape[0] * 2
                
                # 计算新的频率
                new_inv_freq = compute_freq_curve(freq_type, head_dim, **kwargs)
                
                # 替换
                module.inv_freq = nn.Parameter(new_inv_freq, requires_grad=False)
                patched.append(name)
                
                print(f"Patched {name}: inv_freq shape {new_inv_freq.shape}")
                print(f"  inv_min={new_inv_freq.min().item():.6e}, inv_max={new_inv_freq.max().item():.6e}")
    
    return patched


def create_model(model_size='350m', freq_type='orig', **freq_kwargs):
    """
    创建模型并应用频率修改
    """
    # 模型配置映射
    configs = {
        '125m': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
        '350m': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},
        '500m': {'n_layer': 24, 'n_head': 16, 'n_embd': 1280},
        '700m': {'n_layer': 32, 'n_head': 16, 'n_embd': 1536},
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model_size: {model_size}")
    
    # 从头创建模型（不是加载预训练）
    config = AutoConfig.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        **configs[model_size],
        vocab_size=32000,
        max_position_embeddings=8192,
        use_cache=True,
    )
    
    model = AutoModelForCausalLM.from_config(config)
    
    # 应用频率修改
    if freq_type != 'orig':
        patched = patch_rope_freq(model, freq_type, **freq_kwargs)
        print(f"Patched {len(patched)} rotary embedding layers")
    
    return model


def load_training_data(seq_length=8192, max_samples=None):
    """
    加载训练数据
    """
    print(f"Loading training data with seq_length={seq_length}")
    
    # 使用TinyStories或Pile数据
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="train")
    except:
        dataset = load_dataset("the_pile", split="train", streaming=True)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(examples):
        output = tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_length,
            padding="max_length",
            return_tensors=None,
        )
        output["labels"] = output["input_ids"].copy()
        return output
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    return tokenized, tokenizer


def evaluate_model(model, tokenizer, device, lengths=[2048, 4096, 8192, 16384, 32768]):
    """
    在不同序列长度评估PPL
    """
    model.eval()
    results = {}
    
    # 加载评估数据
    eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join(eval_dataset["text"])
    
    for length in lengths:
        try:
            # 截取评估文本
            eval_tokens = tokenizer(
                eval_text[:length * 10],  # 粗略估计
                return_tensors="pt",
                truncation=True,
                max_length=length,
            )
            
            input_ids = eval_tokens["input_ids"].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
                ppl = math.exp(loss)
            
            results[length] = {"loss": loss, "ppl": ppl}
            print(f"  Length {length}: PPL={ppl:.3f}, Loss={loss:.4f}")
            
        except Exception as e:
            print(f"  Length {length}: Error - {e}")
            results[length] = {"error": str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="训练阶段频率对比实验")
    parser.add_argument("--model_size", type=str, default="350m", choices=["125m", "350m", "500m", "700m"])
    parser.add_argument("--freq_type", type=str, default="orig", choices=["orig", "geometric", "hybrid", "sigmoid"])
    parser.add_argument("--seq_length", type=int, default=8192)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="./results/train_freq_comparison")
    parser.add_argument("--max_samples", type=int, default=None, help="限制训练样本数")
    
    # 频率参数
    parser.add_argument("--theta_base", type=float, default=100000)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--p", type=float, default=3.9)
    parser.add_argument("--omf", type=float, default=0.3)
    parser.add_argument("--steepness", type=float, default=8.0)
    parser.add_argument("--midpoint", type=float, default=0.5)
    
    args = parser.parse_args()
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.model_size}_{args.freq_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Model size: {args.model_size}, Freq type: {args.freq_type}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 创建模型
    freq_kwargs = {
        'theta_base': args.theta_base,
        'alpha': args.alpha,
        'p': args.p,
        'omf': args.omf,
        'steepness': args.steepness,
        'midpoint': args.midpoint,
    }
    
    model = create_model(args.model_size, args.freq_type, **freq_kwargs)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    
    # 加载数据
    train_dataset, tokenizer = load_training_data(args.seq_length, args.max_samples)
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        fp16=True,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        report_to="none",
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # 训练前评估
    print("\n=== Pre-training Evaluation ===")
    pre_eval = evaluate_model(model, tokenizer, device)
    
    # 训练
    print("\n=== Training ===")
    trainer.train()
    
    # 训练后评估
    print("\n=== Post-training Evaluation ===")
    post_eval = evaluate_model(model, tokenizer, device)
    
    # 保存结果
    results = {
        "timestamp": timestamp,
        "model_size": args.model_size,
        "freq_type": args.freq_type,
        "freq_kwargs": freq_kwargs,
        "total_params": total_params,
        "seq_length": args.seq_length,
        "max_steps": args.max_steps,
        "pre_training_eval": pre_eval,
        "post_training_eval": post_eval,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 保存模型
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")
    
    print(f"\nResults saved to {output_dir}")
    
    # 打印摘要
    print("\n=== Summary ===")
    print(f"Config: {args.model_size} with {args.freq_type} frequency")
    print("PPL comparison (pre -> post):")
    for length in sorted(post_eval.keys()):
        if "error" not in post_eval[length]:
            pre_ppl = pre_eval.get(length, {}).get("ppl", "N/A")
            post_ppl = post_eval[length]["ppl"]
            print(f"  {length}: {pre_ppl:.2f} -> {post_ppl:.2f}")


if __name__ == "__main__":
    main()