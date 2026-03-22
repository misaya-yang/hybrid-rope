#!/bin/bash
# compile改回default + 清除之前失败的runs
sed -i 's/mode="reduce-overhead"/mode="default"/' /root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
grep 'torch.compile' /root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py | grep mode
rm -rf /root/autodl-tmp/50m_mla32_4k_tau_sweep/50m_*
echo "done: compile=default, old runs cleared"
