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
    使用Qwen2.5配置作为基础（服务器本地已有）
    """
    # 模型配置映射 - 基于Qwen2.5架构
    configs = {
        '125m': {'num_hidden_layers': 12, 'num_attention_heads': 12, 'hidden_size': 768, 'intermediate_size': 2048, 'num_key_value_heads': 4},
        '350m': {'num_hidden_layers': 24, 'num_attention_heads': 16, 'hidden_size': 1024, 'intermediate_size': 2816, 'num_key_value_heads': 4},
        '500m': {'num_hidden_layers': 24, 'num_attention_heads': 16, 'hidden_size': 1280, 'intermediate_size': 3520, 'num_key_value_heads': 4},
        '700m': {'num_hidden_layers': 28, 'num_attention_heads': 16, 'hidden_size': 1536, 'intermediate_size': 4096, 'num_key_value_heads': 4},
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model_size: {model_size}")
    
    # 使用本地Qwen模型配置
    # 优先使用本地缓存，避免网络访问
    local_qwen_path = "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct"
    
    config = AutoConfig.from_pretrained(
        local_qwen_path,
        **configs[model_size],
        vocab_size=151936,  # Qwen vocab size
        max_position_embeddings=8192,
        use_cache=True,
        tie_word_embeddings=False,
        local_files_only=True,
    )
    
    model = AutoModelForCausalLM.from_config(config)
    
    # 应用频率修改
    if freq_type != 'orig':
        patched = patch_rope_freq(model, freq_type, **freq_kwargs)
        print(f"Patched {len(patched)} rotary embedding layers")
    
    return model


def load_training_data(seq_length=8192, max_samples=None, tokenizer=None):
    """
    加载训练数据 - 使用本地tokenizer和简单生成的数据
    """
    print(f"Loading training data with seq_length={seq_length}")
    
    # 使用本地Qwen tokenizer
    local_qwen_path = "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct"
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(local_qwen_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 生成简单训练数据 - 避免网络依赖
    import random
    random.seed(42)
    
    # 使用一些简单的文本模板
    templates = [
        "The study of language models has advanced significantly in recent years. ",
        "Machine learning algorithms can learn patterns from large amounts of data. ",
        "Natural language processing enables computers to understand human language. ",
        "Deep learning has revolutionized artificial intelligence research. ",
        "Transformers have become the dominant architecture for NLP tasks. ",
    ]
    
    # 生成足够长的文本
    texts = []
    n_samples = max_samples if max_samples else 1000
    for i in range(n_samples):
        # 随机组合模板创建长文本
        text = ""
        while len(text) < seq_length:
            text += random.choice(templates)
        texts.append(text[:seq_length * 4])  # 大约seq_length个字符
    
    print(f"Generated {len(texts)} training samples")
    
    # 手动创建dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": texts})
    
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
    
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    return tokenized, tokenizer


def evaluate_model(model, tokenizer, device, lengths=[2048, 4096, 8192]):
    """
    在不同序列长度评估PPL - 使用生成的简单文本
    """
    model.eval()
    results = {}
    
    # 生成评估文本 - 避免网络依赖
    eval_templates = [
        "The development of artificial intelligence has transformed many industries. ",
        "Language models are becoming increasingly powerful and useful. ",
        "Research in machine learning continues to advance rapidly. ",
        "Natural language understanding is a key challenge in AI. ",
        "Deep learning has enabled many breakthrough applications. ",
    ]
    
    # 生成足够长的评估文本
    import random
    random.seed(123)
    eval_text = ""
    while len(eval_text) < 100000:  # 足够长的文本
        eval_text += random.choice(eval_templates)
    
    for length in lengths:
        try:
            # 截取评估文本
            eval_tokens = tokenizer(
                eval_text[:length * 4],
                return_tensors="pt",
                truncation=True,
                max_length=length,
            )
            
            input_ids = eval_tokens["input_ids"].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
                ppl = math.exp(min(loss, 10))  # 防止溢出
            
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
    
    # 训练配置 - 96GB显存充分利用
    # 700M@8192: 模型~2.2GB + 激活~20GB/batch (gradient_checkpointing)
    # 有效batch=16 (4*4), max_steps=800
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=4,  # 96GB显存支持700M+batch=4
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_steps=1000,
        eval_steps=1000,
        bf16=True,  # 使用bf16替代fp16（与gradient_checkpointing兼容）
        gradient_accumulation_steps=4,  # 有效batch=16
        warmup_steps=100,
        report_to="none",
        gradient_checkpointing=True,  # 启用梯度检查点节省显存
    )
    
    # 训练器 - 不传tokenizer参数，使用data_collator处理
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
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