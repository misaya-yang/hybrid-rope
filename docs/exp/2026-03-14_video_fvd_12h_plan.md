# 12h R6000 Blackwell 96GB: FVD 实验计划

> **目标**: 在 12 小时内拿到 FVD(EVQ) < FVD(Geo) 的干净证据
> **硬件**: R6000 Blackwell 96GB
> **关键约束**: FVD 需要 (1) 视频生成 (2) I3D 特征提取 (3) Fréchet distance

---

## 核心分析：为什么不直接用现有 10h 脚本

现有 `run_phase23_blackwell_10h.sh` 方案有两个问题：

1. **只测 PPL，没有 FVD** — 审稿人会说 "PPL 不等于生成质量"
2. **Moving MNIST 太简单** — PPL 差异可能不大，因为数据本身低熵

FVD 的价值在于：它直接衡量 **模型在时间维度外推时生成视频的质量**。如果 EVQ 在 128 帧 (4× 训练长度) 的 FVD 显著优于 Geo，这就是论文 video section 的核心证据。

---

## 时间预算分配

| Phase | 任务 | 时间 | 说明 |
|-------|------|------|------|
| A | 环境+数据准备 | 0.5h | 安装 I3D, 准备数据 |
| B | 训练 (3 arms × 1 seed) | 5h | geo_k8, geo_k16, evq_k16 |
| C | 视频生成 (每个 arm 2048 videos × 4 帧数) | 2h | autoregressive generation |
| D | FVD 计算 | 1h | I3D feature extraction + Fréchet |
| E | 复现验证 seed 137 (geo_k16 vs evq_k16) | 3h | 关键对比的第二个 seed |
| F | 整理结果 | 0.5h | |
| **合计** | | **12h** | |

---

## Phase A: 环境准备 (0.5h)

### 安装 FVD 依赖

```bash
pip install torch-fidelity scipy
# I3D TorchScript model from StyleGAN-V (最可靠的 FVD 实现)
pip install einops
```

### 下载 I3D 预训练权重

```python
# StyleGAN-V 的 I3D TorchScript 模型
# 会自动下载到 ~/.cache/
import urllib.request
I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
I3D_PATH = "data/video_temporal/external/i3d_torchscript.pt"
urllib.request.urlretrieve(I3D_URL, I3D_PATH)
```

### 确认数据

```bash
python3 scripts/data_prep/prepare_moving_mnist_video.py  # 如果还没有
ls data/video_temporal/generated/moving_mnist_medium/manifest.json
```

---

## Phase B: 训练 (5h)

### 精简到 3 个 arm

在 12h 约束下，砍掉 geo_k12 和 evq_k12（中间值），只留最有区分度的对比：

| Variant | K_t | τ | 用途 |
|---------|-----|---|------|
| **geo_k8** | 8 | 0 | 时间低频不够的 baseline |
| **geo_k16** | 16 | 0 | 时间低频充足的 Geo baseline |
| **evq_k16** | 16 | auto | EVQ 重分配同样的时间预算 |

```bash
python3 scripts/video_temporal/run_video_temporal_allocation_sweep.py \
  --profile blackwell96 \
  --variants "geo_k8,geo_k16,evq_k16" \
  --seeds 42 \
  --epochs 16 \
  --eval-chunks 16 \
  --work-dir results/supporting_video/phase23_fvd/pass1_seed42
```

**为什么 16 epochs 够了**：Moving MNIST 16000 videos × 16 epochs = 256K updates，对 16-layer transformer 足够收敛。

---

## Phase C: 视频生成 (2h)

### 关键设计：Teacher-Forced Start + Autoregressive Continuation

```python
def generate_videos(model, real_videos, train_frames, target_frames, n_generate=2048):
    """
    1. 取 real video 的前 train_frames 帧作为 context
    2. Autoregressive 生成后续帧直到 target_frames
    3. 返回完整视频 (context + generated)
    """
    generated = []
    for i in range(n_generate):
        context = real_videos[i, :train_frames]  # (32, H, W) 的 token 序列
        # Autoregressive generation: predict one frame (8×8=64 tokens) at a time
        current = context.clone()
        for t in range(train_frames, target_frames):
            # Forward pass, take last 64 logits, sample next frame tokens
            logits = model(current)
            next_frame_tokens = sample_frame(logits, temperature=0.9)
            current = torch.cat([current, next_frame_tokens], dim=0)
        generated.append(current)
    return torch.stack(generated)
```

### 生成矩阵

| 帧数 | 外推倍数 | 说明 |
|------|----------|------|
| 32f | 1× (in-dist) | FVD baseline, 应该差不多 |
| 64f | 2× | 中等外推 |
| 96f | 3× | 较大外推 |
| 128f | 4× | 极端外推, 区分度最大 |

每个 arm × 每个帧数生成 2048 videos。

### Token → Pixel 解码

Moving MNIST 用的是 **patch mean quantization**：
- 每个 token 是 8×8 patch 的平均灰度值 (0-255)
- 解码 = 用 token value 填充整个 8×8 patch
- 最终帧 = 64×64 灰度图

```python
def decode_tokens_to_frames(tokens, patch_size=8, grid_h=8, grid_w=8):
    """tokens: (frames * grid_h * grid_w,) → frames: (frames, 64, 64)"""
    frames = tokens.reshape(-1, grid_h, grid_w).float() / 255.0
    frames = frames.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, patch_size, patch_size)
    frames = frames.reshape(-1, grid_h * patch_size, grid_w * patch_size)
    return frames
```

---

## Phase D: FVD 计算 (1h)

### I3D FVD Pipeline

```python
import torch
from scipy.linalg import sqrtm

def load_i3d(path="data/video_temporal/external/i3d_torchscript.pt"):
    model = torch.jit.load(path).eval().cuda()
    return model

def extract_i3d_features(videos, i3d_model, batch_size=16):
    """
    videos: (N, T, H, W) grayscale float32 [0,1]
    I3D expects: (N, T, 3, 224, 224) float32

    Steps:
    1. Grayscale → RGB (repeat channel)
    2. Resize 64→224
    3. Sample/pad to 16 frames per clip
    4. Extract 400-dim logits or 1024-dim pre-logits
    """
    features = []
    for i in range(0, len(videos), batch_size):
        batch = videos[i:i+batch_size]
        # (B, T, H, W) → (B, T, 3, H, W)
        batch = batch.unsqueeze(2).expand(-1, -1, 3, -1, -1)
        # Resize to 224×224
        B, T, C, H, W = batch.shape
        batch = batch.reshape(B*T, C, H, W)
        batch = F.interpolate(batch, size=(224, 224), mode='bilinear')
        batch = batch.reshape(B, T, C, 224, 224)
        # I3D expects (B, C, T, H, W)
        batch = batch.permute(0, 2, 1, 3, 4).cuda()
        # Extract features (before final FC)
        with torch.no_grad():
            feat = i3d_model(batch)  # depends on I3D variant
        features.append(feat.cpu())
    return torch.cat(features, dim=0)

def compute_fvd(real_features, gen_features):
    """Fréchet distance between two multivariate Gaussians."""
    mu_r, sigma_r = real_features.mean(0).numpy(), np.cov(real_features.numpy(), rowvar=False)
    mu_g, sigma_g = gen_features.mean(0).numpy(), np.cov(gen_features.numpy(), rowvar=False)
    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fvd = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fvd)
```

### 计算矩阵

对每个 (arm, frame_count) 组合：
1. 取 2048 real videos 作为参考分布
2. 取 2048 generated videos 作为生成分布
3. 两组都过 I3D → 1024 维特征
4. 计算 FVD

---

## Phase E: 复现 (3h)

只跑关键对比 geo_k16 vs evq_k16 × seed 137：

```bash
python3 scripts/video_temporal/run_video_temporal_allocation_sweep.py \
  --profile blackwell96 \
  --variants "geo_k16,evq_k16" \
  --seeds 137 \
  --epochs 16 \
  --eval-chunks 16 \
  --work-dir results/supporting_video/phase23_fvd/pass2_seed137
```

然后用同样的 FVD pipeline 评估。

---

## 预期结果表

```
| Variant  | K_t | τ    | FVD@32f | FVD@64f | FVD@96f | FVD@128f | PPL@128f |
|----------|-----|------|---------|---------|---------|----------|----------|
| geo_k8   | 8   | 0    | ~X      | ~X      | ~X      | ~X       | ~X       |
| geo_k16  | 16  | 0    | ~X      | ~X      | ~X      | ~X       | ~X       |
| evq_k16  | 16  | auto | ~X      | ~X      | ~X      | ~X       | ~X       |
```

### 预期模式

1. **geo_k8 → geo_k16**: FVD@64f+ 下降 → 证明时间低频预算重要
2. **geo_k16 → evq_k16**: FVD@64f+ 进一步下降 → EVQ 重分配优于 uniform
3. **In-dist (32f)**: FVD 差异很小 → waterbed pattern 在视频中复现
4. **PPL 和 FVD 方向一致** → 两个指标互相印证

---

## 备选方案：如果 I3D FVD 不靠谱

Moving MNIST 64×64 灰度图在 I3D (Kinetics 预训练) 上可能不是最佳特征空间。

### Plan B: Frame-level FID + Temporal Coherence

1. **Per-frame FID** (用 InceptionV3，更成熟): 衡量单帧质量
2. **Temporal coherence score**: 相邻帧差分的 L2 norm → 衡量运动平滑度
3. **两者联合** → 替代 FVD 的更鲁棒度量

### Plan C: 直接生成 + MSE + SSIM

如果时间不够搞 FVD：
- 对 2048 个 test videos, teacher-force 前 32 帧, autoregressive 后 32-128 帧
- 和 ground truth 未来帧计算 MSE 和 SSIM
- 这是最简单的度量, 但 "预测质量" 也是有效证据

---

## 论文更新目标

如果实验成功，更新:

1. **Appendix C.8** (或新增): Video temporal FVD table
   ```
   "EVQ reduces FVD by X% at 4× temporal extrapolation
   on Moving MNIST, with the same waterbed pattern as text."
   ```

2. **§4 最后一段**: 加一句 prediction
   ```
   "The same mechanism predicts that temporal channels in 3D
   video RoPE should also benefit from EVQ-style reshaping."
   ```

3. **§6 Limitations**: 升级 video evidence 状态
   ```
   从 "supportive" → "supporting with FVD confirmation"
   ```

---

## 最终建议：12h 怎么用最值

**如果只能选一件事**: 跑 geo_k16 vs evq_k16 × 2 seeds + FVD，砍掉 geo_k8。因为:
- geo_k8 只是证明 "更多时间通道有用"，VideoRoPE 已经证明了这一点
- **EVQ > Geo 在相同 K_t 预算下** 才是论文独有的贡献
- 两个 seed 足够声称方向性结论

**时间分配**: 训练 4 runs (2 variants × 2 seeds) ~6h + 生成 ~2h + FVD ~1h + buffer 3h

**总结**: 12h 足够拿到 "EVQ 在时间外推时 FVD 优于 Geo" 的双 seed 证据。
