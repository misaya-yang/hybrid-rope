# 5090 Passkey 混合训练实验

## 服务器
- SSH: `ssh -p 48142 root@connect.westc.gpuhub.com`
- 密码: `3wog+1mHWO4C`
- GPU: 5090 32GB

## Claude Code 提示词

见下方"提示词"章节，直接复制给 Claude Code。

## 实验目的
验证：当模型学会了检索能力后，EVQ 的频率分配是否能让检索更好地外推到训练长度以外。

## 实验逻辑
- 训练：L=2048，90% FineWeb-Edu + 10% passkey → 模型在 ≤2K 学会检索
- 评估：测 PPL + passkey @{2K, 4K, 8K, 16K}
- 关键对比：Geo vs EVQ 在 4K-16K（外推区间）的 passkey retrieval 差异
- 分离"会不会检索"和"PE 能不能帮检索外推"两个变量

---

## 提示词（复制给 Claude Code / Codex）

```
你是实验执行助手。在5090服务器上执行 passkey 混合训练实验。

【服务器】
- SSH: ssh -p 48142 root@connect.westc.gpuhub.com
- 密码: 3wog+1mHWO4C
- GPU: 5090 32GB

【任务】
运行 350M 模型的 passkey 混合训练实验：90% FineWeb-Edu + 10% passkey 训练数据。
目的：验证模型学会检索后，EVQ 的频率分配能否让检索更好地外推到训练长度以外。

【步骤】
1. 连接服务器，确认 GPU 可用（nvidia-smi）
2. 进入代码目录：cd /root/autodl-tmp/dfrope/hybrid-rope/scripts/m4_evq_sweep
3. 修改 run_evq_sweep.py 中 TIER_CONFIGS 的 "350m" 条目，把 eval_lengths 改为：
   "eval_lengths": [2048, 4096, 8192, 10240, 12288, 16384],
   （代码已有 OOM 保护，eval 时某长度崩了会自动跳过继续，所以放心加）
4. 创建输出目录并执行实验：
   mkdir -p /root/autodl-tmp/evq_passkey_mix_10pct
   nohup python -u run_evq_sweep.py \
       --tier 350m \
       --taus 0.0,1.5 \
       --seeds 42 \
       --base 500000 \
       --passkey_mix_ratio 0.10 \
       --work_dir /root/autodl-tmp/evq_passkey_mix_10pct \
       > /root/autodl-tmp/evq_passkey_mix_10pct/run.log 2>&1 &
5. 监控日志直到完成：tail -f /root/autodl-tmp/evq_passkey_mix_10pct/run.log
   - 确认看到 "[passkey-train] mix target=10.00%" 说明混合数据生效
   - 预计 1-2 小时（350M × 100M tokens × 2 configs）

【完成后】
6. 读取结果目录下每个 run 的 result.json
7. 输出对比表格：
   - PPL@{2K, 4K, 8K, 10K, 12K, 16K}: Geo vs EVQ（delta 和百分比）
   - Passkey retrieval@{2K, 4K, 8K, 10K, 12K, 16K}: Geo vs EVQ
   - 重点关注：2K passkey 两边是否都高（验证学会了检索）？4K以上 EVQ 是否优于 Geo？
8. 保存完整报告到 /root/autodl-tmp/evq_passkey_mix_10pct/analysis_report.md
```

---

## 技术细节
- 代码已完全支持：`run_evq_sweep.py --passkey_mix_ratio 0.10`
- MixedDataset 和 make_passkey_training_sample 在 eval_passkey_scratch.py 中
- 350m tier 默认 100M tokens，5090 <40GB 会自动 batch=2, eval 仅 [2K,4K,8K]
- 需要 `--eval_16k` 加上 16K 评估（可能 OOM，eval 时 batch=1 应该能挤进去）

## 成功标准
- EVQ 在 8K/16K passkey 明显优于 Geo（>10pp）→ 强信号
- 两边都高或都低 → passkey 不区分 PE，指标无用
- 两边 2K 都能但 16K 都不行 → 模型太小

## 后续
- 10% 有效 → 加 seed（137, 256）
- 10% 太多影响 PPL → 试 5%
- 成功 → R6000 也加 passkey mix
