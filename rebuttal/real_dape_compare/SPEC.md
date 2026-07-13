# SPEC：诚实 DAPE 对照实验（最低成本 / 5090）

## 科学目标（允许的问题）

1. 在 **与 Table 4 相同的 PE-dominant 协议**（125M, L=128, 15M tok）上，把对照拆成：
   - `geo` — geometric / midpoint-geo RoPE
   - `evq` — EVQ-Cosh fixed τ（默认 5.0，与 Table 4 打印行一致）
   - `free_inv_freq` — 32 可学 inv_freq（**旧论文行「DAPE」的真实身份**）
   - `kerple` — 仅 Kerple bias（中间基线）
   - `dape_kerple_mlp` — Kerple + attention-score MLP（**Zheng-inspired DAPE-ish**）
2. 报告 **绝对 PPL** 与 **零额外参数** 两个读数，禁止把 free_inv_freq 叫 DAPE。

## 明确不做的声明

- 不声称官方 GPT-NeoX DAPE 复现。
- 不声称 EVQ 替代 DAPE。
- 不把本实验升级为 Primary 除非 3-seed 完成且 claim 重写经作者批准。
- 默认 **不** 为「翻盘赢 DAPE」设计；预期见 `FINDINGS.md`。

## 协议 P1 — PE-dominant（最低成本主跑）

| 字段 | 值 |
|------|-----|
| model | 125M (h=768, L=12, heads=12, d=64) |
| L_train | 128 |
| tokens | 15_000_000 |
| base | 500_000 |
| lr | 6e-4 |
| global batch | 64 sequences（micro 自适应） |
| eval lengths | 128, 256, 512, 1024, 2048, 4096, 8192 |
| eval chunks | 8，固定 seed 9999 |
| methods | `geo,evq,free_inv_freq,kerple,dape_kerple_mlp` |
| seeds | pilot: `42`；完整: `42,137,256` |
| τ(EVQ) | 5.0（Table 4 打印行；公式 \(64/\sqrt{128}\approx5.66\) 可另开 `--tau`） |

## 协议 P2 — 不重跑

Phase11B L=256 100M 的 Geo/EVQ ± DAPE 已有 3-seed JSON。
仅当 P1 与 phase11b 结论冲突时才考虑复现 P2。

## 成功 / 失败判定（科学）

| 结局 | 含义 | 可用 claim |
|------|------|------------|
| EVQ@8K < free_inv_freq@8K | 复现 Table 4 诚实版 | 闭式分配优于无约束可学频率（本协议） |
| dape_kerple_mlp@8K ≪ plain EVQ@8K | 与 phase11b 一致 | 算子容量主导；EVQ 是零参数轴 |
| EVQ+dape 相对 geo+dape 无优势 | 与 phase11b 一致 | 不宣称互补赢 |
| EVQ@8K < dape_kerple_mlp@8K 且显著 | **意外**；需多 seed + 审查实现 | 仅在严格验证后可讨论 |

## 实现身份标签（代码强制写入 result.json）

```json
{
  "method_id": "dape_kerple_mlp",
  "identity": "zheng_inspired_kerple_plus_attn_mlp",
  "not": ["zheng2024_official_gpt_neox", "free_inv_freq_32", "paper_table4_row_dape"]
}
```

## 数据

- 优先：本地 FineWeb-Edu token cache（`--train-cache` / `--val-cache`）
- 否则：streaming FineWeb-Edu（需网络；**GPU 计时前应先缓存好**）
- 禁止：在 GPU 开机后现下大数据。

## 输出

```
$WORK/
  manifests/run_manifest.json
  runs/<run_id>/result.json
  runs/<run_id>/config.json
  runs/<run_id>/inv_freq.pt          # 若可序列化
  runs/<run_id>/learned_inv_freq.pt  # free_inv_freq only
  aggregate/summary.json
  aggregate/summary.md
```

每个 `result.json` 必须含：seed, method_id, identity, ppl dict, train_time_sec, inv_freq_hash, git_sha（若可得）, torch/cuda 版本。
