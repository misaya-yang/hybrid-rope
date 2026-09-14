# Llama最终评测执行交接（2026-09-14）

## 冻结目标

当前只比较 `Meta-Llama-3-8B-Instruct` 上的 exact TailSpline 与 exact
MrRoPE-Pro。两臂使用同一 S=4、canonical band `[18,35]`、gain
`1.138629436111989`、checkpoint、decoder 与 scorer；不训练、不加 YaRN/BM、
不搜索新曲线。

最终组合不是 `500/task`：

1. 32K Full-13 RULER，`200 examples/task`，每臂2600条；
2. 五任务 LongBench Natural-QA，复用此前冻结的631个源行，重新从原始
   LongBench 行恢复官方 prompt 内容并用 Llama chat template/tokenizer 构造；
3. 既有 Llama 8K Native PPL 控制继续复用，不重跑。

RULER 的每任务200条由既有10条加新增190条组成。两臂严格使用相同
prompts。Natural-QA 的631条是历史 OLMo tokenizer 下 `>4096` 的冻结源池；
新 manifest 另外记录它们相对 Llama 原生8K的真实长度分层。因此最终报告
不能把631条全部表述为“超过 Llama 原生长度”。

## 服务器与当前无人值守链

```bash
ssh -p 37849 root@connect.westc.seetacloud.com
```

- 代码：`/root/autodl-tmp/hybrid-rope`
- 实验根：`/root/autodl-tmp/today_rope_plan_20260914`
- RULER根：`tailspline_llama_s4_32k_ruler200`
- Natural-QA根：`tailspline_llama_s4_naturalqa631`
- RULER supervisor PID文件：
  `tailspline_llama_s4_32k_ruler200/supervisor.pid`
- 全链 PID文件：`tailspline_llama_final_benchmark_chain.pid`
- 全链日志：`tailspline_llama_final_benchmark_chain.log`

全链顺序如下：

```text
RULER任务逐格运行：TailSpline → MrPro
    × 13 tasks × 190新增行（已有10行/任务复用）
    ↓ 验证2600行/臂、200行/任务、prompt完全配对
RULER paired report
    ↓
Natural-QA：TailSpline 631 → MrPro 631
    ↓ 验证原始F1、EOS/cap、paired row/source identity
Natural-QA paired cluster-bootstrap report
```

RULER按任务分目录运行，任一已完成分片会被跳过，未完成分片从自己的raw
前缀恢复。这样既不等所有CPU资产，也不会因单个任务中断而重跑其他任务。
13个任务的正式输入现均已完成。QA1最初使用了越过SQuAD可用题数的
`pre_samples=6000`，而上游生成器会吞掉该越界并无限重试；该无效进程已停止，
QA1改用与base10不重叠的合法区间 `5200..5389` 并完成190条。其余任务复用
已生成资产的前190条。旧 `tailspline_llama_s4_32k_full500` 的执行 supervisor
已停止；不要重新启动旧 full500 runner。

首个190条TailSpline分片实际约26分钟完成；随后已自动切到同任务MrPro，
切换后GPU重新达到100%。按这一实测速率，RULER两臂约需12小时，随后631×2
自然QA因输入更短但无法普遍组成等长batch，粗估另需1--3小时。以日志和
`live.json` 为准，不把该粗估当完成回执。

## 家中监控命令

先确认进程身份，再看GPU：

```bash
root=/root/autodl-tmp/today_rope_plan_20260914
ps -p "$(cat "$root/tailspline_llama_s4_32k_ruler200/supervisor.pid")" -o pid,stat,etime,args=
ps -p "$(cat "$root/tailspline_llama_final_benchmark_chain.pid")" -o pid,stat,etime,args=
nvidia-smi
tail -50 "$root/tailspline_llama_s4_32k_ruler200/logs/ruler200_supervisor.log"
tail -50 "$root/tailspline_llama_final_benchmark_chain.log"
find "$root/tailspline_llama_s4_32k_ruler200/runs" -name status.json -print -exec cat {} \;
```

当前分片的进度写在对应 run 目录的 `live.json`。短暂模型换表/重载会有数秒
空档；若GPU持续空闲两分钟以上，先检查最近日志和两个 PID 的实际命令，不要
删除任何raw。

只有在对应 PID 已消失且最终报告不存在时，才恢复：

```bash
cd /root/autodl-tmp/hybrid-rope
root=/root/autodl-tmp/today_rope_plan_20260914
nohup experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_s4_32k_ruler200_staged.sh \
  > "$root/tailspline_llama_s4_32k_ruler200/logs/ruler200_supervisor.log" 2>&1 < /dev/null &
echo $! > "$root/tailspline_llama_s4_32k_ruler200/supervisor.pid"
```

RULER完成但全链未进入Natural-QA时，可按同样的“PID已消失”条件恢复全链：

```bash
cd /root/autodl-tmp/hybrid-rope
root=/root/autodl-tmp/today_rope_plan_20260914
nohup experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_final_benchmark_chain.sh \
  > "$root/tailspline_llama_final_benchmark_chain.log" 2>&1 < /dev/null &
echo $! > "$root/tailspline_llama_final_benchmark_chain.pid"
```

## 完成与失败判定

完整RULER报告：

`tailspline_llama_s4_32k_ruler200/reports/tailspline_vs_mrpro_full13_32k_200_per_task.json`

完整Natural-QA报告：

`tailspline_llama_s4_naturalqa631/reports/tailspline_vs_mrpro_naturalqa631.json`

RULER报告含 `"status": "MATCHED_GENERATION_RANGE_REPORT_V1"`、Natural-QA
报告含 `"status": "COMPLETE"` 后，全链写入：

- `tailspline_llama_final_benchmark_sha256.txt`
- `tailspline_llama_final_benchmark_complete.txt`

若 supervisor 消失但相应报告/完成标记不存在，就是失败；查看最后一个启动
分片的日志。不要只凭进程结束或文件名判断实验有效。不要在两臂完整前读取
单臂partial分数。

当前未设置Codex定时任务，服务器也不会自动关机；由作者在家监控并在确认
两个最终报告、raw和SHA文件均落数据盘后决定关机。
