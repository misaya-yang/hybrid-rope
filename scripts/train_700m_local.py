#!/usr/bin/env python3
"""
使用本地模型和数据训练700M - 离线版本
无需访问HuggingFace Hub
"""

import os
import sys
import json
import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

# 下载wikitext数据
def download_wikitext_raw():
    """手动下载wikitext-2原始数据"""
    import urllib.request
    
    data_dir = Path("/root/autodl-tmp/wikitext_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    urls = {
        "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "valid": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
        "test": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
    }
    
    files = {}
    for split, url in urls.items():
        filepath = data_dir / f"{split}.txt"
        if not filepath.exists():
            print(f"Downloading {split}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"  Downloaded {filepath}")
            except Exception as e:
                print(f"  Failed to download {split}: {e}")
                # 创建一些示例数据
                with open(filepath, 'w') as f:
                    f.write("=" * 50 + "\n")
                    f.write("Sample text for training\n")
                    for i in range(100):
                        f.write(f"This is sample sentence number {i}. The quick brown fox jumps over the lazy dog.\n")
                print(f"  Created sample data: {filepath}")
        files[split] = str(filepath)
    
    return files


class LocalTextDataset(Dataset):
    """本地文本数据集"""
    def __init__(self, filepath, tokenizer, seq_length=2048, max_samples=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        print(f"Loading text from {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Total text length: {len(text)} chars")
        
        # Tokenize
        print("Tokenizing...")
        self.tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"Total tokens: {len(self.tokens)}")
        
        # Create samples
        self.samples = []
        for i in range(0, len(self.tokens) - seq_length, seq_length // 2):
            sample = self.tokens[i:i + seq_length + 1]
            if len(sample) >= seq_length // 2:
                self.samples.append(sample[:seq_length + 1])
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Created {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # Pad if needed
        if len(input_ids) < self.seq_length:
            pad_len = self.seq_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class LossPrintingCallback(TrainerCallback):
    """每步打印loss"""
    def __init__(self, log_file=None):
        self.log_file = log_file
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            step = state.global_step
            loss = logs['loss']
            lr = logs.get('learning_rate', 0)
            msg = f"[Step {step}] loss={loss:.6f}, lr={lr:.2e}"
            print(msg, flush=True)
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(msg + '\n')


def compute_ppl(model, dataloader, device):
    """计算困惑度"""
    print("\n=== Computing Perplexity ===")
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            if (i + 1) % 10 == 0:
                print(f"  Eval batch {i+1}/{len(dataloader)}, current avg loss: {total_loss/total_tokens:.4f}")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(min(avg_loss, 10))  # Clamp to avoid overflow
    
    print(f"Final: avg_loss={avg_loss:.4f}, PPL={ppl:.2f}")
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description="使用本地数据训练700M模型")
    parser.add_argument("--model_path", type=str, 
                       default="./results/train_freq_comparison/700m_orig_20260214_140024/model",
                       help="本地模型路径")
    parser.add_argument("--output_dir", type=str, default="./results/train_700m_local")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=50)
    
    # 训练参数
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    
    args = parser.parse_args()
    
    # 设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model path: {args.model_path}")
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "training.log"
    
    # 加载模型和tokenizer
    print("\n=== Loading Model ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.1f}M")
    
    # 下载/准备数据
    print("\n=== Preparing Data ===")
    files = download_wikitext_raw()
    
    # 创建数据集
    print("\n=== Loading Datasets ===")
    train_dataset = LocalTextDataset(
        files["train"], tokenizer, 
        seq_length=args.seq_length,
        max_samples=args.max_train_samples
    )
    eval_dataset = LocalTextDataset(
        files["valid"], tokenizer,
        seq_length=args.seq_length,
        max_samples=args.max_eval_samples
    )
    
    # 预训练评估
    print("\n=== Pre-training Evaluation ===")
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    pre_loss, pre_ppl = compute_ppl(model, eval_loader, device)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=1,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[LossPrintingCallback(log_file)],
    )
    
    # 开始训练
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    with open(log_file, 'a') as f:
        f.write(f"\n=== Training Started at {datetime.now()} ===\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Total params: {total_params/1e6:.1f}M\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Max steps: {args.max_steps}\n")
        f.write(f"Pre-training PPL: {pre_ppl:.2f}\n\n")
    
    trainer.train()
    
    # 训练后评估
    print("\n=== Post-training Evaluation ===")
    post_loss, post_ppl = compute_ppl(model, eval_loader, device)
    
    # 保存结果
    results = {
        "model_path": args.model_path,
        "total_params": total_params,
        "train_samples": len(train_dataset),
        "max_steps": args.max_steps,
        "pre_training": {"loss": pre_loss, "ppl": pre_ppl},
        "post_training": {"loss": post_loss, "ppl": post_ppl},
        "improvement": {
            "loss_drop": pre_loss - post_loss,
            "ppl_ratio": pre_ppl / post_ppl if post_ppl > 0 else 0,
        }
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 保存模型
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Pre-training:  Loss={pre_loss:.4f}, PPL={pre_ppl:.2f}")
    print(f"Post-training: Loss={post_loss:.4f}, PPL={post_ppl:.2f}")
    print(f"Loss drop: {pre_loss - post_loss:.4f}")
    print(f"\nModel saved to: {output_dir / 'final_model'}")


if __name__ == "__main__":
    main()