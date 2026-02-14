#!/usr/bin/env python3
"""
Qwen Hybrid-RoPE位置压缩对比实验
使用Qwen2.5-7B-Instruct模型测试
"""
import subprocess
import json
import os

# SSH配置
SSH_CMD = r'C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0'

def run_ssh(cmd, timeout=30):
    """执行SSH命令"""
    full_cmd = f'{SSH_CMD} "{cmd}"'
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip()

# Qwen实验脚本
QWEN_SCRIPT = '''
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

MODEL_NAME = "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct"
OUTPUT_DIR = "/root/autodl-tmp/dfrope/hybrid-rope/results/qwen_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("Qwen2.5-7B Hybrid-RoPE位置压缩实验")
print("="*60)

# 加载模型
print("加载Qwen模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print(f"模型加载完成")

# 测试长度和压缩因子
TEST_LENGTHS = [4096, 8192, 16384, 24576, 32768, 40960]
COMPRESSION_FACTORS = [None, 1.5, 2.0]  # None = baseline

def compress_positions(position_ids, original_max=8192, compression_factor=1.5):
    """应用位置压缩 - 使用torch.where避免dtype问题"""
    pos = position_ids.float()
    compressed = torch.where(
        pos > original_max,
        original_max + (pos - original_max) / compression_factor,
        pos
    )
    return compressed.round().long()

def generate_passkey_text(length):
    """生成passkey测试文本"""
    import random
    random.seed(42)
    
    # 简单填充
    filler = "The quick brown fox jumps over the lazy dog. " * 10
    
    # Passkey
    passkey = random.randint(10000, 99999)
    
    # 构建文本
    text = ""
    while len(text) < length - 200:
        text += filler
    
    # 插入passkey
    insert_pos = len(text) // 2
    text = text[:insert_pos] + f" The secret passkey is {passkey}. Remember this number. " + text[insert_pos:]
    text += f"\\n\\nQuestion: What is the passkey mentioned in the text above? Answer with only the number.\\nAnswer:"
    
    return text, passkey

def calculate_ppl(model, tokenizer, text):
    """计算困惑度"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=65536)
    input_ids = inputs["input_ids"].cuda()
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        ppl = torch.exp(outputs.loss).item()
    
    return ppl, input_ids.shape[1]

results = {}

for cf in COMPRESSION_FACTORS:
    method_name = "baseline" if cf is None else f"cf{cf}"
    results[method_name] = {}
    print(f"\\n{'='*40}\\n测试方法: {method_name}\\n{'='*40}")
    
    for length in TEST_LENGTHS:
        print(f"\\n测试长度: {length}")
        
        try:
            # 生成测试文本
            text, passkey = generate_passkey_text(length)
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=65536)
            input_ids = inputs["input_ids"].cuda()
            actual_len = input_ids.shape[1]
            
            # 应用位置压缩 - 使用整数索引
            if cf is not None:
                position_ids = torch.arange(actual_len).unsqueeze(0).cuda()
                compressed_pos = compress_positions(position_ids, compression_factor=cf)
                # Qwen需要Long类型的position_ids
                inputs["position_ids"] = compressed_pos.long()
            
            # 计算PPL
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                ppl = torch.exp(outputs.loss).item()
            
            # 显存使用
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            
            results[method_name][length] = {
                "ppl": round(ppl, 3),
                "actual_len": actual_len,
                "mem_gb": round(mem_gb, 2),
                "status": "ok"
            }
            print(f"PPL: {ppl:.3f}, 长度: {actual_len}, 显存: {mem_gb:.2f}GB")
            
            # 清理
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            results[method_name][length] = {"status": str(e)}
            print(f"错误: {e}")
            torch.cuda.empty_cache()
            gc.collect()

# 保存结果
output_file = os.path.join(OUTPUT_DIR, "results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\\n结果已保存到: {output_file}")
print("="*60)
print("实验完成!")
print("="*60)
'''

# 脚本路径
REMOTE_SCRIPT = "/root/autodl-tmp/dfrope/hybrid-rope/scripts/run_qwen_comparison.py"

def upload_script_via_ssh(script_content, remote_path):
    """通过SSH上传脚本"""
    # 使用base64编码传输
    import base64
    encoded = base64.b64encode(script_content.encode()).decode()
    
    # 分块传输
    chunk_size = 4000
    for i in range(0, len(encoded), chunk_size):
        chunk = encoded[i:i+chunk_size]
        cmd = f'echo "{chunk}" >> /tmp/script_b64.txt'
        run_ssh(cmd, timeout=30)
    
    # 解码并保存
    run_ssh(f'base64 -d /tmp/script_b64.txt > {remote_path} && rm /tmp/script_b64.txt', timeout=30)

def main():
    print("=" * 60)
    print("Qwen2.5-7B Hybrid-RoPE对比实验")
    print("=" * 60)
    
    # 1. 保存本地脚本
    print("\n1. 创建实验脚本...")
    script_file = "scripts/run_qwen_comparison.py"
    os.makedirs("scripts", exist_ok=True)
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(QWEN_SCRIPT)
    print(f"本地脚本已保存: {script_file}")
    
    # 2. 上传脚本到远程
    print("\n2. 上传脚本到远程服务器...")
    try:
        upload_script_via_ssh(QWEN_SCRIPT, REMOTE_SCRIPT)
        print("脚本上传完成")
    except Exception as e:
        print(f"上传失败: {e}，尝试直接运行...")
    
    # 3. 创建输出目录
    run_ssh("mkdir -p /root/autodl-tmp/dfrope/hybrid-rope/results/qwen_comparison", timeout=30)
    
    # 4. 运行实验
    print("\n3. 启动Qwen实验...")
    run_cmd = f'cd /root/autodl-tmp/dfrope/hybrid-rope && nohup /root/miniconda3/bin/python {REMOTE_SCRIPT} > results/qwen_comparison/run.log 2>&1 &'
    try:
        run_ssh(run_cmd, timeout=10)
    except subprocess.TimeoutExpired:
        pass  # 后台运行，超时是正常的
    
    print("Qwen实验已启动！")
    print("\n查看进度命令:")
    print(f'  {SSH_CMD} "tail -30 /root/autodl-tmp/dfrope/hybrid-rope/results/qwen_comparison/run.log"')

if __name__ == "__main__":
    main()