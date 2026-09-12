# llama3_60dir_20260911：分类索引

本地实现和记录入口。论文结论只从登记结果owner读取；此index不提升证据强度、不重述实时GPU状态。

总入口：[index.md](../../index.md)

- [PAPER_FAITHFUL_REROUTE_20260911.md](PAPER_FAITHFUL_REROUTE_20260911.md) — Paper-faithful reroute: progressive radix conversion before broad search
- [README.md](README.md) — Llama-3-8B：20 方向 × 60 配置 —— 试验规范与 CPU 参考实现
- [SCORING_CONTRACT.md](SCORING_CONTRACT.md) — Llama Plan B frozen scoring contract
- [anytime_scale_supervisor.py](anytime_scale_supervisor.py) — anytime_scale_supervisor.py
- [arms_llama3.csv](arms_llama3.csv) — arms_llama3.csv
- [budget.py](budget.py) — budget.py
- [budget_example.json](budget_example.json) — budget_example.json
- [candidates.csv](candidates.csv) — candidates.csv
- [candidates_corrected.csv](candidates_corrected.csv) — candidates_corrected.csv
- [coverage_report.py](coverage_report.py) — coverage_report.py
- [cpu_test.json](cpu_test.json) — cpu_test.json
- [deployed_review_subset.manifest.json](deployed_review_subset.manifest.json) — deployed_review_subset.manifest.json
- [directions.json](directions.json) — directions.json
- [engineering_pilot_bridge.py](engineering_pilot_bridge.py) — engineering_pilot_bridge.py
- [execution_order.json](execution_order.json) — execution_order.json
- [gpu_core_parity.py](gpu_core_parity.py) — gpu_core_parity.py
- [hash_checkpoint.py](hash_checkpoint.py) — hash_checkpoint.py
- [integration_canary_bridge.py](integration_canary_bridge.py) — integration_canary_bridge.py
- [llama_planb_queue.py](llama_planb_queue.py) — llama_planb_queue.py
- [llama_runner.py](llama_runner.py) — llama_runner.py
- [llama_s_supervisor.py](llama_s_supervisor.py) — llama_s_supervisor.py
- [llama_svh_queue.py](llama_svh_queue.py) — llama_svh_queue.py
- [operators.py](operators.py) — operators.py
- [p_readiness_report.py](p_readiness_report.py) — p_readiness_report.py
- [paired_report.py](paired_report.py) — paired_report.py
- [paper_s16_64k_supervisor.py](paper_s16_64k_supervisor.py) — paper_s16_64k_supervisor.py
- [parse_plan.py](parse_plan.py) — parse_plan.py
- [phase1.py](phase1.py) — phase1.py
- [plan_arms.py](plan_arms.py) — plan_arms.py
- [planb_matched_controls.py](planb_matched_controls.py) — planb_matched_controls.py
- [post_factor_candidate_supervisor.py](post_factor_candidate_supervisor.py) — post_factor_candidate_supervisor.py
- [prepare_planb_panel.py](prepare_planb_panel.py) — prepare_planb_panel.py
- [prepare_qwen_ribb_bridge.py](prepare_qwen_ribb_bridge.py) — prepare_qwen_ribb_bridge.py
- [priority_correction_20260911.json](priority_correction_20260911.json) — priority_correction_20260911.json
- [reference/](reference)
- [repair_panel_source_ids.py](repair_panel_source_ids.py) — repair_panel_source_ids.py
- [runtime_authorization.json](runtime_authorization.json) — runtime_authorization.json
- [scale_gain_supervisor.py](scale_gain_supervisor.py) — scale_gain_supervisor.py
- [split_panel.py](split_panel.py) — split_panel.py
- [static_identity_audit.json](static_identity_audit.json) — static_identity_audit.json
- [static_identity_audit.py](static_identity_audit.py) — static_identity_audit.py
- [test_data_identity_repair.py](test_data_identity_repair.py) — test_data_identity_repair.py
- [test_llama_svh_queue.py](test_llama_svh_queue.py) — test_llama_svh_queue.py
- [test_p_readiness_report.py](test_p_readiness_report.py) — test_p_readiness_report.py
- [test_planb_matched_controls.py](test_planb_matched_controls.py) — test_planb_matched_controls.py
- [test_stats_contract.py](test_stats_contract.py) — test_stats_contract.py
- [test_tools.py](test_tools.py) — test_tools.py
- [tool_tests.json](tool_tests.json) — tool_tests.json
- [upgrade_report.py](upgrade_report.py) — upgrade_report.py
- [validate_planb_data.py](validate_planb_data.py) — validate_planb_data.py
