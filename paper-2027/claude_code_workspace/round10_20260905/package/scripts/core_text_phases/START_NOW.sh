#!/bin/bash
# 开机后唯一需要运行的命令
# Phase 21a (1B) + Phase 21b (续训+500M) 顺序全自动执行
# 总ETA: ~45min(1B) + ~25min(续训) = ~70min

set -e
LOGDIR=/root/autodl-tmp/125m_mla32_4k_logs
mkdir -p $LOGDIR

echo "=== START: $(date) ===" | tee ${LOGDIR}/master.log

# Phase 21a: 125M on v1, 1B tokens
bash /root/autodl-tmp/scripts/core_text_phases/run_125m_4k_1b_v1.sh 2>&1 | tee -a ${LOGDIR}/master.log

# Phase 21b: continue from 1B -> 1.5B on v2
bash /root/autodl-tmp/scripts/core_text_phases/run_125m_4k_continue_v2.sh 2>&1 | tee -a ${LOGDIR}/master.log

echo "=== ALL DONE: $(date) ===" | tee -a ${LOGDIR}/master.log
echo "结果对比: 500M / 750M / 1B / 1.5B GEO vs EVQ"
