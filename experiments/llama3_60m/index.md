# llama3_60m：分类索引

本地实现和记录入口。论文结论只从登记结果owner读取；此index不提升证据强度、不重述实时GPU状态。

总入口：[index.md](../../index.md)

- [README.md](README.md) — llama3_60m — Llama-3-8B 的 20 方向 / 60 规则实现
- [adapter.py](adapter.py) — adapter.py
- [build_ec_tables.py](build_ec_tables.py) — build_ec_tables.py
- [c_collect.py](c_collect.py) — c_collect.py
- [cfree.py](cfree.py) — cfree.py
- [constructions.py](constructions.py) — constructions.py
- [core.py](core.py) — core.py
- [directions.py](directions.py) — directions.py
- [history_registry.jsonl](history_registry.jsonl) — history_registry.jsonl
- [panel.py](panel.py) — panel.py
- [patch_mgain.py](patch_mgain.py) — patch_mgain.py
- [selftest.py](selftest.py) — selftest.py
- [test_adapter_integration.py](test_adapter_integration.py) — test_adapter_integration.py
