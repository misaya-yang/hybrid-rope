# Phase 14 快速验证: 750M Geo+YaRN vs Hybrid+YaRN

## 目标

在 750M 的 Geo 和 Hybrid 最终 checkpoint 上，推理时施加 YaRN scaling，对比 passkey retrieval@8K。

**只测两组**: Geo+YaRN 和 Hybrid+YaRN。不测原始 Geo/Hybrid（Phase 9F 已有数据）。

## 做法

修改 `eval_passkey_teacher_forcing.py`，加一个 `--yarn_scale_factor` 参数。

YaRN 推理时缩放逻辑：在模型加载后、eval 前，对 `model` 中存储的 `inv_freq` 施加 YaRN progressive scaling：

```python
def apply_yarn_scaling(inv_freq, scale_factor, original_max_len=2048):
    """YaRN progressive frequency scaling (inference-time only)."""
    import math
    wavelength = 2 * math.pi / inv_freq

    beta_fast = 32
    beta_slow = 1
    low_freq_wavelen = original_max_len / 1.0    # low_freq_factor=1
    high_freq_wavelen = original_max_len / 4.0    # high_freq_factor=4

    smooth = (wavelength - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
    smooth = smooth.clamp(0, 1)

    scaled_freq = inv_freq / scale_factor
    new_freq = (1 - smooth) * scaled_freq + smooth * inv_freq
    return new_freq
```

关键: 找到模型里存 `inv_freq` 的地方（可能是 `model.inv_freq` 或在各 attention layer 的 buffer 里），用上面的函数替换。

## 执行

```bash
# Geo + YaRN@4x (eval 8K on 2K-trained model)
python scripts/m4_evq_sweep/eval_passkey_teacher_forcing.py \
    --work_dir /root/autodl-tmp/evq_phase9 \
    --run_id <geo_run_name> \
    --tier 750m \
    --yarn_scale_factor 4 \
    --eval_lengths 2048 4096 8192 \
    --trials 40

# Hybrid + YaRN@4x
python scripts/m4_evq_sweep/eval_passkey_teacher_forcing.py \
    --work_dir /root/autodl-tmp/evq_phase9 \
    --run_id <hybrid_run_name> \
    --tier 750m \
    --rope_type hybrid \
    --yarn_scale_factor 4 \
    --eval_lengths 2048 4096 8192 \
    --trials 40
```

注意: run_id 和参数名要看 `eval_passkey_teacher_forcing.py` 里的实际接口对齐，上面只是示意。

## 预期

- Geo+YaRN@8K: ~65% retrieval
- Hybrid+YaRN@8K: ~100% retrieval

如果差距 >30pp 就是超线性协同的铁证。

## 耗时

每组 ~15min，两组 ~30min。

## Claude Code 提示词

```
读 docs/prompts/PROMPT_PHASE14_QUICK_YARN_TEST.md。
任务: 修改 eval_passkey_teacher_forcing.py 加入 --yarn_scale_factor，
然后对 750M 的 Geo 和 Hybrid checkpoint 各跑一次 YaRN@4x 的 passkey eval。
```
