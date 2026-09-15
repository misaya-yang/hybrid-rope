# Hybrid-RoPE理论深化CPU核验结果

更新：2026-09-15。状态：**外部理论深化提案对应的15类代数／数值检查全部通过；
该结果只验证算子关系与条件命题，不验证模型性能、任务中介或通用最优表。**

## 1. 判决

`Hybrid_RoPE_Theory_Deepening_20260914.md`提出的主要数学链条在其声明条件下成立：

\[
z\rightarrow\text{相位logit}\rightarrow\text{key相对竞争}
\rightarrow\text{value聚合}\rightarrow\text{局部读出}.
\]

原提案声称附带`verify_theory.py/checks.json`，但交付目录中未找到这两个文件。
本仓库因此独立重建验证器，而不是引用无法复算的外部摘要。固定seed为`20260914`，
没有加载checkpoint、调用CUDA或读取任务分数。

## 2. 核验身份

- 外部输入SHA256：
  `6b5d72e9e3f741ce638f32caea8004418199778e94f3ae009867595a436aa1a3`；
- 独立脚本：[verify_theory_deepening.py](../../../experiments/iclr2027_three_track_sprint_20260915/verify_theory_deepening.py)；
- 脚本SHA256：
  `59214c2669265836f302445980d640f198f1e8c6ce2128a31b9d76adf450a8a1`；
- 固定结果：[theory_deepening_checks.json](../../../experiments/iclr2027_three_track_sprint_20260915/theory_deepening_checks.json)；
- 结果SHA256：
  `1d623100da8e6e8fd11a85177f3771ab61be4b7b78de1213da5be0a9b2baf6df`。

复算命令：

```bash
python3 -m experiments.iclr2027_three_track_sprint_20260915.verify_theory_deepening \
  --out experiments/iclr2027_three_track_sprint_20260915/theory_deepening_checks.json
```

## 3. 十五类检查

| 编号 | 核验对象 | 样例数 | 最大残差或关键证书 |
|---:|---|---:|---:|
| 1 | 原生log-frequency的归一化`z`与span | 12 | `1.78e-15` |
| 2 | 内容相位logit对`z`的解析梯度 | 200 | `2.39e-10` |
| 3 | 有限softmax指数重加权 | 200 | `2.22e-16` |
| 4 | key间log-odds变化 | 200 | `1.11e-15` |
| 5 | 有限value协方差恒等式 | 200 | `5.55e-16` |
| 6 | attention—value局部损失链式法则 | 100 | `1.66e-10` |
| 7 | 冻结／适配曲率的Schur补 | 100 | `8.88e-16`；适配块最小特征值`1.21` |
| 8 | gain平方进入logit及竞争者补偿 | 100 | `1.78e-15` |
| 9 | gain仿射可恢复条件 | 100 | `4.44e-16`；非仿射最小残差`0.218` |
| 10 | `/S`保相位、符号安全终点和中频二次最优解 | 200 | `1.13e-13` |
| 11 | YaRN指数ramp递增、凸性与尺度饱和 | 4组倍率 | 全部通过 |
| 12 | 部署`m→z`及T−C精确恒等式 | `n=1…64` | `2.15e-16` |
| 13 | T−C条件边际配对 | 200 | `5.00e-15` |
| 14 | roughness超额与有限旋转算子界 | 263 | `1.10e-13` |
| 15 | 稀疏访问、top-k、压缩旋转、旋转value、MoE、mHC | 500 | 旋转value导数`1.35e-10` |

第15类中，移除attention质量的`2Vr`界、top-k／MoE margin证书、含内容一阶项的
压缩旋转界和双随机矩阵范数界均通过。完整旋转value导数与有限差分一致；若省略
value直接旋转项，100组中的最小误差仍为`0.107`，因此该通路在相应算子中不能被
attention-map变化替代。

## 4. 可以支持的理论陈述

以下内容可以作为带条件的算子关系写入论文：

1. 改频不改变单层RoPE正交变换的Q/K范数，但通过内容相位改变key间相对logit；
2. 有限频率干预对attention与value输出的影响可分别写成精确softmax重加权和
   value协方差；
3. 单一gain只能精确吸收逐query的正仿射分数变化，一般shape或排序变化不可由
   温度恢复；跨query共享gain还要求相同比例；
4. T−C在固定总log位移下的反对称差分，可精确写成前后配对的平均边际价值差；
5. 稀疏访问集合变化、集合内reader变化和旋转value直接运输是不同计算通路。

Schur补、softmax恒等式、矩阵范数界和margin证书本身是标准数学。论文增量应表述为：
这些关系如何组织固定支持频率干预、gain控制、T−C实验和架构边界；不能把初等公式
重新命名为独立理论贡献。

## 5. CPU不能支持的陈述

本结果不支持：

- TailSpline、C、MrPro或YaRN的真实任务排序；
- one-sided boundary是任意checkpoint的任务最优条件；
- key竞争变化已经实证中介了完整答案收益；
- 某个gain规律能够跨head、layer、checkpoint或架构迁移；
- 稀疏路由、压缩summary、旋转value、MoE或mHC已经改善模型输出；
- post-hoc Native-Z5一定增强成熟checkpoint。

这些问题分别需要已冻结的完整生成、同输入干预或真实模型前向。尤其是论文中心句
中的`mediated by`只有在路径干预成立后才可使用；当前安全写法是`acts through the
computational pathways of key competition and value readout`。

## 6. 论文放置

主文只保留最短计算链：有限softmax/value关系、gain仿射可恢复条件以及T−C条件边际
配对。它们直接服务“剂量、温度可吸收变化和残余shape是不同干预方向”。

Schur补、旋转稳定界、稀疏访问／压缩界、旋转value导数、MoE margin与mHC传播分解
进入证明附录或讨论。没有对应行为结果前，不将稀疏、MoE或mHC升级为本文第三条实证
贡献，也不为它们启动新的曲线或模型搜索。

## 7. 与现有证据的关系

本核验扩展A35/A37的坐标、条件构造与TailSpline数学身份，但不替代A39/A40的真实
模型结果。T−C任务归因仍由E1结果及其运行条件承担；Native窗口增强仍由Native-Z5
实验承担。该CPU结果不改变当前TailSpline、Natural-QA、YaRN和Native参照的GPU队列。
