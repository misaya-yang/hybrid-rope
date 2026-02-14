#!/usr/bin/env python3
"""
使用Wikitext真实语料训练700M模型
===================================
确保每步loss都打印，使用真实数据进行训练和评估
"""

import os
import sys
import json
import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_dataset

# 打印每步loss的回调
class LossPrintingCallback(TrainerCallback):
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


def load_wikitext_dataset(tokenizer, seq_length=2048, max_samples=None):
    """加载wikitext-2数据集"""
    print(f"\n=== Loading Wikitext-2 Dataset ===")
    print(f"Sequence length: {seq_length}")
    print(f"Max samples: {max_samples}")
    
    # 加载数据集
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
        print(f"Loaded wikitext-2-raw-v1 successfully")
    except Exception as e:
        print(f"Error loading wikitext: {e}")
        print("Trying to use cached version or alternative...")
        raise
    
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_length,
            padding=False,
            return_tensors=None,
        )
        return outputs
    
    # Tokenize
    print("Tokenizing train dataset...")
    train_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train",
    )
    
    print("Tokenizing validation dataset...")
    eval_dataset = dataset["validation"].map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing validation",
    )
    
    # 过滤空样本
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 10)
    eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) > 10)
    
    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(max_samples // 5, len(eval_dataset))))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # 打印前几个样本检查
    print("\n=== Sample Data Check ===")
    for i in range(min(3, len(train_dataset))):
        sample_ids = train_dataset[i]["input_ids"]
        sample_text = tokenizer.decode(sample_ids[:100])
        print(f"Sample {i}: {sample_text[:200]}...")
    
    return train_dataset, eval_dataset


def compute_ppl(model, eval_dataset, tokenizer, device, batch_size=4, max_samples=100):
    """计算困惑度"""
    print("\n=== Computing Perplexity ===")
    model.eval()
    
    # 限制样本数
    if len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            if (i + 1) % 10 == 0:
                print(f"  Eval batch {i+1}/{len(dataloader)}, current avg loss: {total_loss/total_tokens:.4f}")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_loss)
    
    print(f"Final: avg_loss={avg_loss:.4f}, PPL={ppl:.2f}")
    
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description="使用Wikitext训练700M模型")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="基础模型")
    parser.add_argument("--output_dir", type=str, default="./results/train_700m_wikitext")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_train_samples", type=int, default=None, help="最大训练样本数，None=全部")
    parser.add_argument("--max_eval_samples", type=int, default=100)
    
    # 训练参数
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    
    args = parser.parse_args()
    
    # 设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "training.log"
    
    # 加载模型和tokenizer
    print("\n=== Loading Model ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.1f}M")
    print(f"Model dtype: {model.dtype}")
    
    # 加载数据
    train_dataset, eval_dataset = load_wikitext_dataset(
        tokenizer, 
        seq_length=args.seq_length,
        max_samples=args.max_train_samples
    )
    
    # 训练前评估
    print("\n=== Pre-training Evaluation ===")
    pre_loss, pre_ppl = compute_ppl(
        model, eval_dataset, tokenizer, device, 
        batch_size=1, max_samples=args.max_eval_samples
    )
    print(f"Pre-training: Loss={pre_loss:.4f}, PPL={pre_ppl:.2f}")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LossPrintingCallback(log_file)],
    )
    
    # 开始训练
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    with open(log_file, 'a') as f:
        f.write(f"\n=== Training Started at {datetime.now()} ===\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Total params: {total_params/1e6:.1f}M\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Max steps: {args.max_steps}\n")
        f.write(f"Pre-training PPL: {pre_ppl:.2f}\n\n")
    
    trainer.train()
    
    # 训练后评估
    print("\n=== Post-training Evaluation ===")
    post_loss, post_ppl = compute_ppl(
        model, eval_dataset, tokenizer, device,
        batch_size=1, max_samples=args.max_eval_samples
    )
    print(f"Post-training: Loss={post_loss:.4f}, PPL={post_ppl:.2f}")
    
    # 保存结果
    results = {
        "model_name": args.model_name,
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
    print(f"PPL ratio: {pre_ppl/post_ppl:.2f}x" if post_ppl > 0 else "N/A")
    print(f"\nModel saved to: {output_dir / 'final_model'}")
    
    with open(log_file, 'a') as f:
        f.write(f"\n=== Training Complete ===\n")
        f.write(f"Post-training PPL: {post_ppl:.2f}\n")
        f.write(f"Results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()