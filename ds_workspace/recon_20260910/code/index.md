# 保留的校验与构造依赖

- [coverage_theory_20260911.py](coverage_theory_20260911.py)：A16构造来源，论文`verify_recovered_assets.py`读取其中的C42函数。
- [pro_tables_20260911.py](pro_tables_20260911.py)：`experiments/twotrack_20260911/theory_checks.py`直接导入。
- [midband_n.py](midband_n.py)：旧profile的构造与计数计算。
- [verify_pro_a_20260911.py](verify_pro_a_20260911.py)：保留结果的数值对账。

`tables/`和`coverage_theory_results.json`保留原有表／回执身份。旧patch、扫描、远程启动和审计副本已删除；它们不是当前运行入口。恢复路径见[清理清单](../../../docs/maintenance/ds_workspace_cleanup_20260918.json)。
