# AI Handoff: Video DiT 实验失败记录

## 犯了什么错

### 1. 模型过大（最严重的错误）
- 选了300M参数的模型去训练64×64的MNIST级数据
- 原因：盲目追求"匹配VideoGPT的268.7M参数量"
- 正确做法：geo vs evq是相对比较，50M模型就够了。大模型只会让训练变慢，不会让对比更有意义

### 2. 在GPU上反复OOM调参（浪费最多钱的错误）
- 尝试顺序：bs=128→OOM, bs=112→OOM, bs=96→OOM, bs=80→OOM
- 每次OOM都浪费2-5分钟GPU时间
- 正确做法：先用小batch跑3步benchmark，测出每sample的显存占用，然后直接算出最大batch

### 3. 梯度检查点的误用
- 以为大batch+梯度检查点=更快，实际上梯度检查点增加33%计算量
- bs=448+ckpt每步17秒，samples/sec反而很低
- 正确做法：不用ckpt，用能装下的最大batch，追求samples/sec而非显存利用率

### 4. torch.compile的误判
- torch.compile额外占用~20GB显存，导致batch必须更小
- 在已经很大的模型上用compile得不偿失
- 正确做法：小模型+compile效果好，大模型+compile可能OOM

### 5. 环境变量设置错误
- PYTORCH_CUDA_ALLOC_CONF放在torch import之后，CUDA已初始化，设置无效
- 浪费了一轮完整的OOM调试

### 6. 日志间隔太长
- 200步打一次日志，用户等10分钟看不到任何进度
- 正确做法：50步甚至20步打一次

### 7. 没有先benchmark就开始长时间训练
- 每次都是"先跑起来看看"，结果OOM/太慢才发现问题
- 正确做法：用3-5步benchmark测量真实step time和显存，确认后再正式跑

## 浪费了多少
- R6000 96GB，10 RMB/小时
- 浪费时间：~1.5小时（全部在调参和OOM）
- 浪费金额：~15 RMB（这台机器）
- 完成的有效训练：0步

## 正确方案（应该从一开始就这么做）

```
模型：~50M（hidden=512, 8层, 8头, head_dim=64, patch_size=8）
Batch：32-64（不需要梯度检查点）
Epochs：30
预计时间：10-15分钟/method，总计<1小时
```

## 文件位置
- 模型：scripts/video_temporal/video_dit.py
- 实验脚本：scripts/video_temporal/run_dit_temporal.py
- 之前的VideoGPT结果：results/supporting_video/
