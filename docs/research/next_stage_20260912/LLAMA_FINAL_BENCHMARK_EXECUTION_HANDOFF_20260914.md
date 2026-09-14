# Llama最终评测执行交接（2026-09-14）

## 冻结目标

当前主比较是 `Meta-Llama-3-8B-Instruct` 上的 exact TailSpline 与 exact
MrRoPE-Pro；同剂量归因只补固定C，classic强基线只补YaRN/BM。扩展臂使用
同一 S=4、canonical band `[18,35]`、gain `1.138629436111989`、checkpoint、
decoder 与 scorer；不训练、不搜索新曲线。

最终组合不是 `500/task`：

1. classic 390+PPL138上只新增同剂量C单臂；
2. 32K Full-13 RULER，`200 examples/task`，T/P每臂2600条；
3. 五任务 LongBench Natural-QA，复用此前冻结的631个源行，重新从原始
   LongBench 行恢复官方 prompt 内容并用 Llama chat template/tokenizer 构造；
4. classic 390+PPL138上只新增YaRN/BM，T/P复用；
5. 既有 Llama 8K Native PPL 控制继续复用，不重跑。

RULER每任务直接保留上游生成流的前200条，`source-order`，不做depth-balanced
oversampling、multi-evidence profile选择或内容padding；两臂严格使用相同
prompts。推理时为了batch=2只使用attention mask屏蔽的左侧pad token，pad不进入
有效上下文，position从0开始，评分输入不包含pad。Natural-QA 的631条是历史 OLMo tokenizer 下 `>4096` 的冻结源池；
新 manifest 另外记录它们相对 Llama 原生8K的真实长度分层。因此最终报告
不能把631条全部表述为“超过 Llama 原生长度”。

## 服务器与当前无人值守链

```bash
ssh -p 37849 root@connect.westc.seetacloud.com
```

- 代码：`/root/autodl-tmp/hybrid-rope`
- 实验根：`/root/autodl-tmp/today_rope_plan_20260914`
- RULER正式根：`tailspline_llama_s4_32k_ruler200_clean`
- 已退出的padding诊断根：`tailspline_llama_s4_32k_ruler200`
- Natural-QA根：`tailspline_llama_s4_naturalqa631`
- RULER supervisor PID文件（正式RULER启动后出现）：
  `tailspline_llama_s4_32k_ruler200_clean/supervisor.pid`
- 全链 PID文件：`tailspline_llama_final_benchmark_chain.pid`
- 全链日志：`tailspline_llama_final_benchmark_chain.log`

全链顺序如下：

```text
完成当前已开始的单任务padding两臂，仅留作诊断、不进最终表
    ↓
同剂量C单臂：复用classic T raw，生成C的390 generation＋138 LM
    ↓
干净RULER：TailSpline 2600 → MrPro 2600
    × 13 tasks × 200 source-order rows，无内容padding
    ↓ 验证2600行/臂、200行/任务、prompt完全配对
RULER paired report
    ↓
Natural-QA：TailSpline 631 → MrPro 631
    ↓ 验证原始F1、EOS/cap、paired row/source identity
Natural-QA paired cluster-bootstrap report
    ↓
classic新增YaRN 390＋138 LM → BM 390＋138 LM；T/P raw复用
```

正式RULER每臂常驻加载一次模型，raw逐行写入；中断后按相同冻结顺序从raw
前缀恢复。13个任务的正式源输入现均已完成。QA1最初使用了越过SQuAD可用题数的
`pre_samples=6000`，而上游生成器会吞掉该越界并无限重试；该无效进程已停止，
QA1前190条改用合法区间 `5200..5389`，后10条使用 `5390..5399`。其余任务
直接取已生成上游source的前200条。旧 `tailspline_llama_s4_32k_full500` 与
padding版 `tailspline_llama_s4_32k_ruler200` 的执行 supervisor 均已停止；
不要重新启动这两个旧runner。

退出前的padding诊断首个190条TailSpline约26分钟；同任务MrPro结束后主链
自动切到同剂量C，再进入无padding正式RULER。正式RULER和Natural-QA都用
masked left-pad batch=2；连同C和YaRN/BM，当前ready队列粗估约15--20小时。
以日志和 `live.json` 为准，不把粗估当完成回执。

## 家中监控命令

先确认进程身份，再看GPU：

```bash
root=/root/autodl-tmp/today_rope_plan_20260914
test ! -f "$root/tailspline_llama_s4_32k_ruler200_clean/supervisor.pid" || \
  ps -p "$(cat "$root/tailspline_llama_s4_32k_ruler200_clean/supervisor.pid")" -o pid,stat,etime,args=
ps -p "$(cat "$root/tailspline_llama_final_benchmark_chain.pid")" -o pid,stat,etime,args=
nvidia-smi
tail -50 "$root/tailspline_llama_s4_32k_ruler200_clean/logs/clean_supervisor.log"
tail -50 "$root/tailspline_llama_final_benchmark_chain.log"
find "$root/tailspline_llama_s4_32k_ruler200_clean/runs" -name status.json -print -exec cat {} \;
```

当前分片的进度写在对应 run 目录的 `live.json`。短暂模型换表/重载会有数秒
空档；若GPU持续空闲两分钟以上，先检查最近日志和两个 PID 的实际命令，不要
删除任何raw。

只有在对应 PID 已消失且最终报告不存在时，才恢复：

```bash
cd /root/autodl-tmp/hybrid-rope
root=/root/autodl-tmp/today_rope_plan_20260914
nohup experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_s4_32k_ruler200_clean.sh \
  > "$root/tailspline_llama_s4_32k_ruler200_clean/logs/clean_supervisor.log" 2>&1 < /dev/null &
echo $! > "$root/tailspline_llama_s4_32k_ruler200_clean/supervisor.pid"
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

完整同剂量C报告：

`tailspline_llama_s4_matched_dose_c/reports/tailspline_vs_dose_control_c_classic.json`

完整RULER报告：

`tailspline_llama_s4_32k_ruler200_clean/reports/tailspline_vs_mrpro_full13_32k_200_per_task_clean.json`

完整Natural-QA报告：

`tailspline_llama_s4_naturalqa631/reports/tailspline_vs_mrpro_naturalqa631.json`

完整classic强基线报告：

`tailspline_llama_s4_classic_strong_baselines/reports/tailspline_vs_mrpro_yarn_bm_classic.json`

四份报告完成后，全链写入：

- `stable_accept_ready_queue_sha256.txt`
- `stable_accept_ready_queue_complete.txt`

若 supervisor 消失但相应报告/完成标记不存在，就是失败；查看最后一个启动
分片的日志。不要只凭进程结束或文件名判断实验有效。不要在两臂完整前读取
单臂partial分数。

当前未设置Codex定时任务，服务器也不会自动关机；由作者在家监控并在确认
两个最终报告、raw和SHA文件均落数据盘后决定关机。
