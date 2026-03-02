你现在是实验结果回收与统计分析助手。请按下面步骤执行，直到输出最终报告：

【服务器与路径】
- 服务器：ssh -p 48142 root@connect.westc.gpuhub.com
- 密码：3wog+1mHWO4C
- 工作目录：/root/autodl-tmp/evq_phase9_350m_L2048_50M_tau0_tau1.5
- 主结果文件：results_final.json
- 中间结果：results_checkpoint.json
- 运行日志：run.log

【实验设定】
- 模型：350M
- 配置：L_train=2048, base=500K, train_tokens=50M
- 比较组：Geo(tau=0.0) vs EVQ(tau=1.5)
- seeds：42,137,256,314
- 目标指标：PPL@2048/4096/8192/16384，passkey retrieval，mean_nll_gap

【执行要求】
1. 先检查进程是否结束（run.pid + ps）。
2. 若未结束，每 120 秒轮询一次；结束后立即继续。
3. 读取 results_final.json，提取以下 run：
   - 350m_tau0.00_seed42/137/256/314
   - 350m_tau1.50_seed42/137/256/314
4. 生成表格：
   - 每个 seed 的 Geo vs EVQ：PPL 四档、retrieval、mean_nll_gap
   - seed 维度差值：EVQ-Geo
   - 均值±标准差（n=4）
5. 做统计检验（必须写清方法与 p 值）：
   - seed 级配对检验：retrieval、PPL@8192、PPL@16384、mean_nll_gap
   - passkey trial 级配对检验：从 passkey.details 做 EVQ vs Geo 配对 sign test / McNemar
   - 重点判断“全局只多3个命中是否噪音”
6. 额外输出：
   - 按长度的 passkey 对比（2048/4096/8192）
   - 按网格 (L, depth) 的 retrieval 差值热表（0.1/0.2/0.5/0.8/0.9）
7. 最后给结论分级：
   - Strong / Weak / Inconclusive
   - 是否建议继续扩大 seed 或改超参
8. 输出为 Markdown 报告，保存到：
   - /root/autodl-tmp/evq_phase9_350m_L2048_50M_tau0_tau1.5/final_multiseed_analysis.md

注意：
- 不要重跑训练。
- 如果缺失某个 seed 的 run，明确列出缺失项并继续输出“当前可得结论”。
- 结论里必须单独回答：“+3 命中是不是噪音？”