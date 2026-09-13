# Checkpoint attention replay（2026-09-13）

本目录实现固定 RoPE 表三接口主线的CPU replay与局部求解基础。当前状态是
**合成数据合同通过，真实checkpoint capture/运行等价尚未执行**；不能据此声称
band、tail depth或transition已由模型求出。

- `core.py`：有限log-length `rho` 网格、单调累计增量到频率表、精确circular
  attention-KL、Native均值与worst-group/CVaR约束。
- `capture_io.py`：流式pre-RoPE Q/K小型receipt，显式记录causal lag符号、
  split-half布局与GQA重复。
- `capture_checkpoint.py`：在Native输入上从真实q_proj/k_proj捕获任务均衡的少量query
  与完整key序列；只允许Native长度输入，不与64K生成同时挂hook。
- `rank_tables.py`：在冻结finite-rho网格上回放多张实际table receipt；输出只称
  retrospective proxy rank，禁止据此过滤真实任务候选。
- `solver.py`：允许exact-zero增量的closed-simplex局部PSD QP；输入是当前点附近
  的step模型，输出绝对增量、active support和KKT residual。

它复用`../../scripts/analysis/native_attention_kl.py`，该工具现支持独立Native/
candidate gain和position dilation。合成测试验证GQA、gain平方、符号、KL零点、
有限差分及active zeros。下一步先在OLMo 4K做runtime parity，再做Llama已知band的
retrospective排序；真实capture前不得直接进入C1/C2/C3任务评测。

理论与完整实验边界见
[固定表三接口理论审计](../../docs/research/next_stage_20260912/THREE_INTERFACE_REPLAY_THEORY_AUDIT_20260913.md)。
