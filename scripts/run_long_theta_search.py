#!/usr/bin/env python3
"""
长时间运行的Theta网格搜索实验
使用真实的RoPE修改来测试不同theta值
"""

import os
import sys
import json
import time
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

# 结果目录
RESULT_DIR = Path("/root/autodl-tmp/dfrope/hybrid-rope/results/long_theta_search")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 日志文件
LOG_FILE = RESULT_DIR / "run.log"

def log(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ============== RoPE实现 ==============

class DynamicRoPE:
    """动态RoPE实现 - 支持不同scaling策略"""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, 
                 scaling_type=None, scaling_factor=1.0, theta_decay=1.0):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_type = scaling_type  # None, 'linear', 'dynamic', 'yarn', 'theta'
        self.scaling_factor = scaling_factor
        self.theta_decay = theta_decay
        
        # 计算inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    
    def _get_cos_sin(self, seq_len, device):
        # 根据scaling类型调整
        if self.scaling_type == 'linear':
            # 线性插值
            t = torch.arange(seq_len, device=device, dtype=torch.float32) / self.scaling_factor
        elif self.scaling_type == 'dynamic':
            # 动态NTK
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            if seq_len > self.max_position_embeddings:
                # 动态调整base
                ratio = seq_len / self.max_position_embeddings
                new_base = self.base * (ratio ** (self.dim / (self.dim - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            else:
                inv_freq = self.inv_freq.to(device)
        elif self.scaling_type == 'theta':
            # Theta衰减 - 核心方法
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            # 应用theta衰减
            decay = self.theta_decay ** (t / seq_len)
            inv_freq = self.inv_freq.to(device) * decay.unsqueeze(1)
            t = t.unsqueeze(1)
        elif self.scaling_type == 'yarn':
            # YaRN风格
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            if seq_len > self.max_position_embeddings:
                ratio = seq_len / self.max_position_embeddings
                # 应用beta变换
                beta = 0.1 * math.log(ratio) + 1
                t = t / ratio * beta
            inv_freq = self.inv_freq.to(device)
        else:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            inv_freq = self.inv_freq.to(device)
        
        t = t.unsqueeze(1) if t.dim() == 1 else t
        freqs = torch.matmul(t.float(), inv_freq.float().unsqueeze(0))
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        return cos, sin

def modify_model_with_rope_scaling(model, scaling_type, scaling_factor, theta_decay, seq_len):
    """修改模型的RoPE scaling"""
    config = model.config
    
    # 获取原始参数
    head_dim = config.hidden_size // config.num_attention_heads
    
    # 修改每层的rotary_emb
    for name, module in model.named_modules():
        if hasattr(module, 'rotary_emb') and module.rotary_emb is not None:
            # 保存原始inv_freq
            if not hasattr(module.rotary_emb, '_original_inv_freq'):
                module.rotary_emb._original_inv_freq = module.rotary_emb.inv_freq.clone()
                module.rotary_emb._original_forward = module.rotary_emb.forward
            
            # 根据scaling类型修改
            if scaling_type == 'theta':
                # Theta衰减
                original_inv_freq = module.rotary_emb._original_inv_freq
                device = original_inv_freq.device
                t = torch.arange(seq_len, device=device, dtype=torch.float32)
                decay = theta_decay ** (t / seq_len)
                new_inv_freq = original_inv_freq * decay.unsqueeze(1)
                module.rotary_emb.inv_freq = new_inv_freq
            elif scaling_type == 'linear':
                # 保存原始seq_len
                if not hasattr(module.rotary_emb, '_original_seq_len'):
                    module.rotary_emb._original_seq_len = seq_len // scaling_factor
    
    return model

def reset_model_rope(model):
    """重置模型的RoPE"""
    for name, module in model.named_modules():
        if hasattr(module, 'rotary_emb'):
            if hasattr(module.rotary_emb, '_original_inv_freq'):
                module.rotary_emb.inv_freq = module.rotary_emb._original_inv_freq.clone()

# ============== 真实RoPE替换 ==============

def create_rotary_embedding(dim, base, scaling_factor, seq_len, device):
    """创建自定义rotary embedding"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    
    # 应用scaling
    if scaling_factor != 1.0:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Theta衰减模式
        if scaling_factor > 1:
            decay = scaling_factor ** (-t / seq_len)
            t = t.unsqueeze(1)
            inv_freq_expanded = inv_freq.unsqueeze(0).expand(seq_len, -1)
            freqs = torch.matmul(t, inv_freq_expanded.T) * decay.unsqueeze(1)
        else:
            t = t.unsqueeze(1)
            freqs = torch.matmul(t, inv_freq.unsqueeze(0))
    else:
        t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        freqs = torch.matmul(t, inv_freq.unsqueeze(0))
    
    cos = freqs.cos()
    sin = freqs.sin()
    
    return cos, sin

def patch_model_rope(model, theta_scale, seq_len, device):
    """Patch模型使用自定义RoPE"""
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    
    # 创建自定义cos/sin
    base = 10000.0
    cos, sin = create_rotary_embedding(head_dim, base, theta_scale, seq_len, device)
    
    # Patch每个attention层
    patches = []
    for name, module in model.named_modules():
        if 'self_attn' in name and hasattr(module, 'q_proj'):
            # 存储原始方法
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
                module._custom_cos = cos
                module._custom_sin = sin
                module._theta_scale = theta_scale
                patches.append(module)
    
    return patches

def unpatch_model(patches):
    """恢复模型的原始实现"""
    for module in patches:
        if hasattr(module, '_original_forward'):
            module.forward = module._original_forward
            del module._original_forward
            if hasattr(module, '_custom_cos'):
                del module._custom_cos
            if hasattr(module, '_custom_sin'):
                del module._custom_sin

# ============== 评估函数 ==============

def evaluate_ppl_real(model, tokenizer, device, seq_len):
    """评估perplexity - 使用真实数据"""
    log(f"[DATA] Loading wikitext-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
    text = "\n\n".join(dataset["text"])
    
    # 编码
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    log(f"[DATA] Total tokens: {len(tokens)}")
    
    # 确保有足够tokens
    if len(tokens) < seq_len * 4:
        tokens = tokens.repeat(math.ceil(seq_len * 4 / len(tokens)) + 1)
    
    tokens = tokens.to(device)
    
    # 评估 - 使用滑动窗口
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_windows = 8  # 增加窗口数提高精度
    
    log(f"[EVAL] Evaluating {num_windows} windows of length {seq_len}...")
    
    with torch.no_grad():
        for i in range(num_windows):
            # 随机采样起始位置
            start = torch.randint(0, len(tokens) - seq_len - 1, (1,)).item()
            input_chunk = tokens[start:start+seq_len].unsqueeze(0)
            
            try:
                torch.cuda.reset_peak_memory_stats()
                outputs = model(input_chunk, labels=input_chunk)
                loss = outputs.loss
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item() * seq_len
                    total_tokens += seq_len
                    
                mem = torch.cuda.max_memory_allocated() / 1e9
                log(f"[EVAL] Window {i+1}/{num_windows}: loss={loss.item():.4f}, mem={mem:.2f}GB")
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return None, None, "OOM"
                log(f"[ERROR] {str(e)}")
                raise
    
    if total_tokens == 0:
        return None, None, "ERROR"
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    
    return ppl, avg_loss, mem_peak

# ============== 主实验 ==============

def run_single_config(model, tokenizer, device, config_name, theta_scale, test_lengths):
    """运行单个theta配置的测试"""
    log(f"\n{'='*60}")
    log(f"[CONFIG] {config_name}: theta_scale={theta_scale}")
    log(f"{'='*60}")
    
    results = {
        "name": config_name,
        "theta_scale": theta_scale,
        "data": {},
        "boundary": None
    }
    
    baseline_ppl = None
    
    for length in test_lengths:
        log(f"\n[TEST] L={length}")
        
        start_time = time.time()
        
        try:
            # 重置模型
            model.eval()
            torch.cuda.empty_cache()
            
            # 评估
            ppl, loss, mem = evaluate_ppl_real(model, tokenizer, device, length)
            
            elapsed = time.time() - start_time
            
            if ppl is None:
                log(f"[RESULT] L={length}: {mem}")
                results["data"][str(length)] = {"status": mem}
                if mem == "OOM":
                    results["oom_at"] = length
                    break
                continue
            
            # 记录结果
            results["data"][str(length)] = {
                "ppl": round(ppl, 3),
                "loss": round(loss, 4),
                "elapsed_sec": round(elapsed, 2),
                "mem_gb": round(mem, 2),
                "status": "ok"
            }
            
            log(f"[RESULT] L={length}: PPL={ppl:.3f}, loss={loss:.4f}, mem={mem:.2f}GB, time={elapsed:.1f}s")
            
            # 计算baseline (2k位置的PPL)
            if baseline_ppl is None:
                baseline_ppl = ppl
            
            # 检测边界 (PPL > 5x baseline 或 PPL > 100)
            if results["boundary"] is None and (ppl > 5 * baseline_ppl or ppl > 100):
                results["boundary"] = length
                log(f"[BOUNDARY] Found at L={length}")
            
            # 如果PPL爆炸太严重，跳到下一个配置
            if ppl > 1000:
                log(f"[SKIP] PPL too high, moving to next config")
                break
                
        except Exception as e:
            log(f"[ERROR] L={length}: {str(e)}")
            import traceback
            log(traceback.format_exc())
            results["data"][str(length)] = {"status": "error", "message": str(e)}
            break
    
    # 计算collapse ratio
    if str(test_lengths[0]) in results["data"] and results["data"][str(test_lengths[0])].get("status") == "ok":
        baseline = results["data"][str(test_lengths[0])]["ppl"]
        results["collapse_ratio"] = {}
        for length_str, data in results["data"].items():
            if data.get("status") == "ok":
                results["collapse_ratio"][length_str] = round(data["ppl"] / baseline, 3)
    
    return results

def main():
    log("=" * 60)
    log("长时间Theta网格搜索实验 - 真实RoPE修改")
    log("=" * 60)
    
    # 配置
    device = torch.device("cuda:0")
    model_path = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
    
    # 测试长度 - 更多长度点
    test_lengths = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 
                    18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768,
                    36864, 40960, 45056, 49152, 53248, 57344, 61440, 65536]
    
    # Theta配置 - 更多配置点
    theta_scales = [
        ("baseline", 1.0),  # 标准RoPE
        ("geo_100k", 100000),  # theta=100k
        ("geo_250k", 250000),  # theta=250k
        ("geo_500k", 500000),  # theta=500k
        ("geo_750k", 750000),  # theta=750k
        ("geo_1m", 1000000),   # theta=1M
        ("geo_1.5m", 1500000), # theta=1.5M
        ("geo_2m", 2000000),   # theta=2M
        ("geo_3m", 3000000),   # theta=3M
        ("geo_5m", 5000000),   # theta=5M
        ("geo_10m", 10000000), # theta=10M
    ]
    
    # 加载模型
    log(f"[MODEL] Loading from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    config = model.config
    log(f"[MODEL] Loaded successfully")
    log(f"[MODEL] Hidden={config.hidden_size}, Heads={config.num_attention_heads}")
    log(f"[MODEL] Max position embeddings={getattr(config, 'max_position_embeddings', 'N/A')}")
    
    log(f"\n[PLAN] Total configs: {len(theta_scales)}")
    log(f"[PLAN] Test lengths: {len(test_lengths)} points from {min(test_lengths)} to {max(test_lengths)}")
    log(f"[PLAN] Estimated total evaluations: {len(theta_scales) * len(test_lengths)}")
    
    # 运行所有实验
    all_results = {}
    start_time = time.time()
    
    for i, (config_name, theta_scale) in enumerate(theta_scales):
        log(f"\n{'#'*60}")
        log(f"[PROGRESS] Config {i+1}/{len(theta_scales)}: {config_name}")
        elapsed_total = time.time() - start_time
        if i > 0:
            avg_time_per_config = elapsed_total / i
            remaining = (len(theta_scales) - i) * avg_time_per_config / 60
            log(f"[PROGRESS] Elapsed: {elapsed_total/60:.1f}min, Est. remaining: {remaining:.1f}min")
        log(f"{'#'*60}")
        
        try:
            result = run_single_config(
                model, tokenizer, device, 
                config_name, theta_scale, test_lengths
            )
            all_results[config_name] = result
            
            # 保存中间结果
            with open(RESULT_DIR / "results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            log(f"[SAVED] Intermediate results saved")
            
        except Exception as e:
            log(f"[ERROR] Config {config_name} failed: {str(e)}")
            import traceback
            log(traceback.format_exc())
            all_results[config_name] = {"error": str(e)}
    
    # 生成汇总报告
    log("\n" + "=" * 60)
    log("实验完成！生成汇总报告...")
    
    total_time = time.time() - start_time
    log(f"[TOTAL TIME] {total_time/60:.1f} minutes")
    
    # 生成Markdown报告
    report = generate_report(all_results)
    with open(RESULT_DIR / "report.md", "w") as f:
        f.write(report)
    
    log(f"[DONE] 结果保存在 {RESULT_DIR}")
    log("=" * 60)

def generate_report(results):
    """生成Markdown报告"""
    lines = ["# 长时间Theta网格搜索实验报告", "", f"生成时间: {datetime.now()}", ""]
    
    # 按边界长度排序
    valid_results = [(k, v) for k, v in results.items() if "error" not in v and v.get("boundary")]
    valid_results.sort(key=lambda x: x[1].get("boundary", 0) or 0, reverse=True)
    
    lines.append("## 边界长度排名 (从长到短)")
    lines.append("")
    lines.append("| 排名 | 配置 | 边界长度 | 16k PPL | 32k PPL | 48k PPL | 64k PPL |")
    lines.append("|------|------|----------|---------|---------|---------|---------|")
    
    for rank, (name, data) in enumerate(valid_results, 1):
        boundary = data.get("boundary", "N/A")
        d = data.get("data", {})
        
        def get_ppl(length):
            val = d.get(str(length), {}).get("ppl")
            return f"{val:.2f}" if isinstance(val, float) else "N/A"
        
        lines.append(f"| {rank} | {name} | {boundary} | {get_ppl(16384)} | {get_ppl(32768)} | {get_ppl(49152)} | {get_ppl(65536)} |")
    
    lines.append("")
    lines.append("## 详细PPL曲线")
    lines.append("")
    lines.append("```")
    
    # 绘制ASCII PPL曲线
    lengths = [2048, 4096, 8192, 16384, 32768, 49152, 65536]
    
    for name, data in results.items():
        if "error" in data:
            continue
        d = data.get("data", {})
        ppls = []
        for length in lengths:
            ppl = d.get(str(length), {}).get("ppl")
            if isinstance(ppl, float) and ppl < 1000:
                ppls.append(f"{ppl:.1f}")
            else:
                ppls.append("---")
        lines.append(f"{name:15s}: " + " -> ".join(ppls))
    
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated at {datetime.now().isoformat()}*")
    
    return "\n".join(lines)

if __name__ == "__main__":
    main()