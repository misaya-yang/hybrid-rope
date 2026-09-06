# 2026-04 实验脚本

## 实验A: LoRA PE Comparison (GEO vs EVQ at 7B)

### 流水线

```
00_preflight.sh              # 环境检查 (0min, 无GPU)
    ↓
01a_lora_train_geo_s42.sh    # GEO seed42 (~1.5h)
01b_lora_train_geo_s43.sh    # GEO seed43 (~1.5h)
01c_lora_train_geo_s44.sh    # GEO seed44 (~1.5h)
01d_lora_train_evq_s43.sh    # EVQ seed43 (~1.5h)  ← seed42已有
01e_lora_train_evq_s44.sh    # EVQ seed44 (~1.5h)
    ↓ (5 runs × ~1.5h = ~7.5h)
02_lora_gen_retrieval_data.sh # 生成检索数据 (几秒, 无GPU)
    ↓
03_lora_stage2_retrieval.sh   # 6个checkpoint各续训50步检索 (~30min)
    ↓
04_lora_eval_ruler.sh         # RULER评测所有 (~14h)
04_lora_eval_ruler.sh quick   # 快速验证 (~4h, 5 trials)
```

### 执行方式 (每步单独nohup)

```bash
cd /root/autodl-tmp/hybrid-rope

# Step 1: 逐个训练 (可以一个完了手动启动下一个)
nohup bash scripts/2026-04/01a_lora_train_geo_s42.sh > logs/01a.log 2>&1 &
# 完成后:
nohup bash scripts/2026-04/01b_lora_train_geo_s43.sh > logs/01b.log 2>&1 &
# ... 以此类推

# Step 2: 生成检索数据 (无需GPU)
bash scripts/2026-04/02_lora_gen_retrieval_data.sh

# Step 3: Stage2检索微调
nohup bash scripts/2026-04/03_lora_stage2_retrieval.sh > logs/03.log 2>&1 &

# Step 4: RULER评测 (先quick验证)
nohup bash scripts/2026-04/04_lora_eval_ruler.sh quick > logs/04_quick.log 2>&1 &
# 确认方向后跑full:
nohup bash scripts/2026-04/04_lora_eval_ruler.sh > logs/04_full.log 2>&1 &
```

### AIHANDOFF合规检查

- [x] `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- [x] 每个seed单独Python进程 (单独.sh文件)
- [x] work_dir在 `/root/autodl-tmp/`
- [x] 用 `/root/miniconda3/bin/python` 绝对路径
- [x] 已有checkpoint自动跳过
- [x] nohup后台运行

### 预算

| 步骤 | 时间 | 费用 |
|------|------|------|
| Stage1训练 (5 runs) | ~7.5h | ~45元 |
| Stage2检索 (6 runs) | ~0.5h | ~3元 |
| RULER quick | ~4h | ~24元 |
| RULER full | ~14h | ~84元 |
| **总计** | **~26h** | **~156元** |
