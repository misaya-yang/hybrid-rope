#!/usr/bin/env python3
"""
全面的Theta搜索实验 - 使用LLaMA原生rope_scaling
测试多种scaling配置，长时间运行
"""

import os
import sys
import json
import time
import math
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 结果目录
RESULT_DIR = Path("/root/autodl-tmp/dfrope/hybrid-rope/results/comprehensive_theta")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULT_DIR / "run.log"

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def evaluate_model(model, tokenizer, device, seq_len, num_windows=10):
    """评估模型perplexity"""
    # 加载数据
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    
    # 确保足够长
    if len(tokens) < seq_len * 4:
        tokens = tokens.repeat(math.ceil(seq_len * 4 / len(tokens)) + 1)
    tokens = tokens.to(device)
    
    # 评估
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(num_windows):
            start = torch.randint(0, len(tokens) - seq_len - 1, (1,)).item()
            input_chunk = tokens[start:start+seq_len].unsqueeze(0)
            
            try:
                outputs = model(input_chunk, labels=input_chunk)
                loss = outputs.loss
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item() * seq_len
                    total_tokens += seq_len
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return None, None, "OOM"
                raise
    
    if total_tokens == 0:
        return None, None, "ERROR"
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    return ppl, avg_loss, mem

def run_experiment_with_scaling(model_path, tokenizer, rope_scaling_config, device, test_lengths):
    """使用指定rope_scaling配置运行实验"""
    results = {"config": rope_scaling_config, "data": {}, "boundary": None}
    
    # 重新加载模型以应用新的rope_scaling
    log(f"[LOAD] Loading model with rope_scaling={rope_scaling_config}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            rope_scaling=rope_scaling_config,
            trust_remote_code=True
        )
    except Exception as e:
        log(f"[ERROR] Failed to load model: {e}")
        return results
    
    model.eval()
    baseline_ppl = None
    
    for length in test_lengths:
        log(f"[TEST] L={length}")
        start_time = time.time()
        
        try:
            torch.cuda.empty_cache()
            ppl, loss, mem = evaluate_model(model, tokenizer, device, length)
            elapsed = time.time() - start_time
            
            if ppl is None:
                log(f"[RESULT] L={length}: {mem}")
                results["data"][str(length)] = {"status": mem}
                if mem == "OOM":
                    results["oom_at"] = length
                    break
                continue
            
            results["data"][str(length)] = {
                "ppl": round(ppl, 3),
                "loss": round(loss, 4),
                "elapsed_sec": round(elapsed, 2),
                "mem_gb": round(mem, 2),
                "status": "ok"
            }
            
            log(f"[RESULT] L={length}: PPL={ppl:.3f}, time={elapsed:.1f}s, mem={mem:.2f}GB")
            
            if baseline_ppl is None:
                baseline_ppl = ppl
            
            if results["boundary"] is None and (ppl > 5 * baseline_ppl or ppl > 100):
                results["boundary"] = length
                log(f"[BOUNDARY] Found at L={length}")
            
            if ppl > 1000:
                log(f"[SKIP] PPL too high")
                break
                
        except Exception as e:
            log(f"[ERROR] L={length}: {e}")
            results["data"][str(length)] = {"status": "error", "message": str(e)}
            break
    
    # 释放模型
    del model
    torch.cuda.empty_cache()
    
    return results

def main():
    log("=" * 60)
    log("全面Theta搜索实验 - LLaMA原生rope_scaling")
    log("=" * 60)
    
    device = torch.device("cuda:0")
    model_path = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
    
    # 测试长度
    test_lengths = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                    18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768,
                    36864, 40960, 45056, 49152, 53248, 57344, 61440, 65536]
    
    # 加载tokenizer
    log("[TOKENIZER] Loading...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 实验配置
    experiments = []
    
    # 1. Baseline (无scaling)
    experiments.append(("baseline", None))
    
    # 2. Linear scaling - 不同factor
    for factor in [2, 4, 8, 16]:
        experiments.append((f"linear_{factor}x", {"type": "linear", "factor": factor}))
    
    # 3. Dynamic NTK
    for factor in [2, 4, 8, 16]:
        experiments.append((f"dynamic_{factor}x", {"type": "dynamic", "factor": factor}))
    
    # 4. YaRN - 不同配置
    for factor in [2, 4, 8, 16]:
        experiments.append((f"yarn_{factor}x", {"type": "yarn", "factor": factor}))
    
    # 5. Llama3 (如果支持)
    for factor in [2, 4, 8, 16]:
        experiments.append((f"llama3_{factor}x", {"type": "llama3", "factor": factor}))
    
    total = len(experiments)
    log(f"[PLAN] Total experiments: {total}")
    log(f"[PLAN] Test lengths: {len(test_lengths)} points")
    log(f"[PLAN] Estimated time: {total * 2}+ minutes")
    
    all_results = {}
    start_time = time.time()
    
    for i, (name, scaling_config) in enumerate(experiments):
        log(f"\n{'#'*60}")
        log(f"[PROGRESS] {i+1}/{total}: {name}")
        if i > 0:
            elapsed = (time.time() - start_time) / 60
            remaining = (total - i) * (elapsed / i)
            log(f"[TIME] Elapsed: {elapsed:.1f}min, ETA: {remaining:.1f}min")
        log(f"{'#'*60}")
        
        try:
            result = run_experiment_with_scaling(
                model_path, tokenizer, scaling_config, device, test_lengths
            )
            all_results[name] = result
            
            # 保存中间结果
            with open(RESULT_DIR / "results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)
                
        except Exception as e:
            log(f"[ERROR] {name} failed: {e}")
            import traceback
            log(traceback.format_exc())
            all_results[name] = {"error": str(e)}
    
    # 生成报告
    log("\n" + "=" * 60)
    log("生成报告...")
    
    total_time = (time.time() - start_time) / 60
    log(f"[TOTAL] {total_time:.1f} minutes")
    
    # Markdown报告
    lines = ["# 全面Theta搜索报告", "", f"时间: {datetime.now()}", "", "## 结果汇总", ""]
    lines.append("| 配置 | 边界 | 8k PPL | 16k PPL | 32k PPL | 48k PPL | 64k PPL |")
    lines.append("|------|------|--------|---------|---------|---------|---------|")
    
    for name, data in all_results.items():
        if "error" in data:
            continue
        d = data.get("data", {})
        boundary = data.get("boundary", "N/A")
        
        def get_ppl(l):
            v = d.get(str(l), {}).get("ppl")
            return f"{v:.2f}" if isinstance(v, float) else "N/A"
        
        lines.append(f"| {name} | {boundary} | {get_ppl(8192)} | {get_ppl(16384)} | {get_ppl(32768)} | {get_ppl(49152)} | {get_ppl(65536)} |")
    
    with open(RESULT_DIR / "report.md", "w") as f:
        f.write("\n".join(lines))
    
    log(f"[DONE] 结果: {RESULT_DIR}")

if __name__ == "__main__":
    main()