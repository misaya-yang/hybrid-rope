#!/usr/bin/env python3
"""
上传并运行Qwen即插即用PPL实验到远程服务器
"""
import subprocess
import os
import time
import base64

# SSH配置
SSH_CMD = r'C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0'

# 路径配置
LOCAL_SCRIPT = "scripts/run_qwen_plugandplay_wikitext_eval.py"
REMOTE_SCRIPT = "/root/autodl-tmp/dfrope/hybrid-rope/scripts/run_qwen_plugandplay_wikitext_eval.py"
OUT_DIR = "/opt/dfrope/results/qwen_plugandplay_wikitext_v1"
MODEL_PATH = "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct"


def run_ssh(cmd, timeout=60):
    """执行SSH命令"""
    full_cmd = f'{SSH_CMD} "{cmd}"'
    try:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", -1


def upload_script_base64(local_path, remote_path):
    """通过base64编码上传脚本"""
    print(f"[Upload] {local_path} -> {remote_path}")
    
    with open(local_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    encoded = base64.b64encode(content.encode()).decode()
    
    # 清理旧文件
    run_ssh("rm -f /tmp/script_b64.txt")
    
    # 分块传输
    chunk_size = 4000
    for i in range(0, len(encoded), chunk_size):
        chunk = encoded[i:i+chunk_size]
        cmd = f'echo "{chunk}" >> /tmp/script_b64.txt'
        stdout, stderr, rc = run_ssh(cmd, timeout=30)
        if rc != 0:
            print(f"[Error] Chunk {i//chunk_size}: {stderr}")
            return False
    
    # 解码并保存
    decode_cmd = f"mkdir -p {os.path.dirname(remote_path)} && base64 -d /tmp/script_b64.txt > {remote_path} && rm /tmp/script_b64.txt"
    stdout, stderr, rc = run_ssh(decode_cmd, timeout=30)
    if rc != 0:
        print(f"[Error] Decode failed: {stderr}")
        return False
    
    print(f"[Upload] Done")
    return True


def main():
    print("=" * 60)
    print("Qwen即插即用PPL实验 - 远程执行")
    print("=" * 60)
    
    # 1. 上传脚本
    print("\n[Step 1] 上传脚本...")
    if not upload_script_base64(LOCAL_SCRIPT, REMOTE_SCRIPT):
        print("[Error] 上传失败")
        return
    print("[OK] 脚本上传成功")
    
    # 2. 创建输出目录
    print("\n[Step 2] 创建输出目录...")
    stdout, stderr, rc = run_ssh(f"mkdir -p {OUT_DIR}", timeout=30)
    if rc != 0:
        print(f"[Error] 创建目录失败: {stderr}")
        return
    print(f"[OK] 输出目录: {OUT_DIR}")
    
    # 3. 检查模型是否存在
    print("\n[Step 3] 检查模型...")
    stdout, stderr, rc = run_ssh(f"ls -la {MODEL_PATH}/config.json", timeout=30)
    if rc != 0:
        print(f"[Warning] 模型路径不存在: {MODEL_PATH}")
        print(f"[Info] 将自动从ModelScope下载")
    else:
        print(f"[OK] 模型已存在: {MODEL_PATH}")
    
    # 4. 启动实验（后台运行）
    print("\n[Step 4] 启动实验...")
    
    run_cmd = f"""cd /root/autodl-tmp/dfrope/hybrid-rope && \
OUT_DIR={OUT_DIR} \
QWEN_MODEL_PATH={MODEL_PATH} \
WINDOWS_PER_SEED=10 \
MAX_EVAL_TOKENS=400000 \
LOAD_IN_4BIT=1 \
nohup /root/miniconda3/bin/python {REMOTE_SCRIPT} > {OUT_DIR}/run.log 2>&1 &"""
    
    # 使用nohup后台运行
    stdout, stderr, rc = run_ssh(run_cmd, timeout=10)
    print("[OK] 实验已启动（后台运行）")
    
    # 5. 等待几秒后检查进度
    print("\n[Step 5] 检查初始进度...")
    time.sleep(5)
    
    check_cmd = f"tail -50 {OUT_DIR}/run.log 2>/dev/null || echo 'Log not ready yet'"
    stdout, stderr, rc = run_ssh(check_cmd, timeout=30)
    print(stdout)
    
    print("\n" + "=" * 60)
    print("实验已启动！")
    print("=" * 60)
    print(f"\n查看日志命令:")
    print(f'  {SSH_CMD} "tail -50 {OUT_DIR}/run.log"')
    print(f"\n查看结果命令:")
    print(f'  {SSH_CMD} "cat {OUT_DIR}/results.json"')
    print(f"\n检查进程命令:")
    print(f'  {SSH_CMD} "ps aux | grep run_qwen_plugandplay"')
    
    # 6. 持续监控进度
    print("\n" + "=" * 60)
    print("持续监控进度 (每60秒刷新一次，按Ctrl+C停止)...")
    print("=" * 60)
    
    try:
        while True:
            time.sleep(60)
            print(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            # 检查日志
            stdout, stderr, rc = run_ssh(f"tail -20 {OUT_DIR}/run.log 2>/dev/null", timeout=30)
            print(stdout)
            
            # 检查是否完成
            stdout, stderr, rc = run_ssh(f"test -f {OUT_DIR}/results.json && grep -q 'elapsed_minutes' {OUT_DIR}/results.json && echo 'COMPLETED'", timeout=30)
            if "COMPLETED" in stdout:
                print("\n" + "=" * 60)
                print("实验完成！正在获取结果...")
                print("=" * 60)
                
                # 获取最终结果
                stdout, stderr, rc = run_ssh(f"cat {OUT_DIR}/results.json", timeout=60)
                print("\n[最终结果 results.json]:")
                print(stdout)
                
                # 获取完整日志
                stdout, stderr, rc = run_ssh(f"tail -100 {OUT_DIR}/run.log", timeout=60)
                print("\n[运行日志末尾]:")
                print(stdout)
                break
    except KeyboardInterrupt:
        print("\n[Info] 监控已停止，实验仍在后台运行")
        print(f"使用以下命令手动检查: {SSH_CMD} \"tail -50 {OUT_DIR}/run.log\"")


if __name__ == "__main__":
    main()