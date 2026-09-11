# RUNBOOK：在公司执行的命令与判读

**2026-09-11** · 服务器 `ssh -p 27741 root@connect.westc.seetacloud.com`，工作目录 `/root/autodl-tmp/phase1_20260910`
所有命令都可直接复制。**先读 §0 的当前状态，再决定跑什么。**

---

## §0 当前状态（离开时）

| 链 | 状态 | 内容 |
|---|---|---|
| `chain_serial` | 跑 `evq` | EVQ τ=0.5/1/2 首次长程 → 之后 `gain`（BM@g1, BM@gY） |
| `chain_holdout` | 跑 `hold_a1b64` | 之后 `hold_b4wide`（最终测试的剩下两臂） |
| `chain_pro` | **等 chain_serial** | `--steps 22`（Pro §9.3）→ `--gain-tables` 2×2（Pro §5） |
| `qwen4x` | 6/10 臂 | 跨模型长程仪器 |

**查看进度（一条命令）**：
```bash
ssh -p 27741 root@connect.westc.seetacloud.com 'cd /root/autodl-tmp/phase1_20260910 && \
  for f in chain_serial chain_holdout chain_pro; do echo -n "$f: "; tail -1 $f.log 2>/dev/null; done; \
  echo "--- qwen4x $(grep -c "\"arm\"" qwen4x.log)/10 ---"; \
  ps -eo args|grep -c "[o]lmo_beta\|[q]wen_longnll"'
```

---

## §1 ★ 最重要的一件事：held-out 结果推翻了一个头条

**必须读**：`HOLDOUT_VERDICT_20260911.md`

72 行 held-out 面板（6 个从未用于选择的任务）上：

| | acc |
|---|---|
| 部署 BM | **0.5350** |
| `b3_lo14`（选择面板上的冠军） | **0.5350** |
| **Δ** | **−0.00pp, t=−0.00**（14胜/11负/47平） |

**而在用于选择的 350 行面板上它是 +14.20pp（t=+6.64）。**

> **⟹ "+12～14pp" 是任务选择的产物，集中在 `niah_single_3` 一个任务。**
> **"某个表超过部署 BM" 在本面板上不被支持。**
> **仍然成立（跨仪器复现、效应巨大）：MrRoPE 显著差于部署 BM（0.0709 vs 0.4167）。**
> 第一期目标（**零训练超过 MrRoPE**）达成；"超过部署 BM"没有被稳健证明。

`chain_holdout` 会再给两个独立读数（`hold_a1b64`、`hold_b4wide`），跑完请一并读。

---

## §2 读结果的两条命令

**A. RULER 350 行（选择面板）配对**——所有臂放一起：
```bash
ssh -p 27741 root@connect.westc.seetacloud.com \
  'cd /root/autodl-tmp/phase1_20260910 && /root/miniconda3/bin/python ruler_paired.py'
```

**B. held-out 72 行配对**：
```bash
ssh -p 27741 root@connect.westc.seetacloud.com 'cd /root/autodl-tmp/phase1_20260910 && \
 /root/miniconda3/bin/python -c "
import json,math,numpy as np
def R(p):
    o={}
    for l in open(p):
        try: d=json.loads(l)
        except: continue
        if \"row_id\" in d: o[d[\"row_id\"]]=d[\"correct\"]
    return o
import glob,os
A={os.path.basename(f)[:-6]:R(f) for f in glob.glob(\"/holdout/*.jsonl\")}
base=A.get(\"beta_b1p0\")
for k,v in sorted(A.items()):
    c=sorted(set(v)&set(base)); d=np.array([v[x]-base[x] for x in c],float)
    se=d.std(ddof=1)/math.sqrt(len(d)) if len(d)>1 else float(\"nan\")
    print(\"%-22s n=%3d acc=%.4f  vsBM %+.2fpp t=%+.2f W/L/T=%d/%d/%d\"%(
      k,len(c),np.mean([v[x] for x in c]),d.mean()*100,(d.mean()/se if se>0 else 0),
      int((d>0).sum()),int((d<0).sum()),int((d==0).sum())))
"'
```

**C. 连续长程仪器**（cheap screen，每臂 ~25 s）：结果在 `<dir>/rows.jsonl`。

---

## §3 Pro 模型的新发现：已写成代码

**表构造器与自检**（纯 numpy，不花 GPU）：
```bash
ssh -p 27741 root@connect.westc.seetacloud.com \
  'cd /root/autodl-tmp/phase1_20260910 && /root/miniconda3/bin/python pro_tables_20260911.py'
```
**本机也能跑**（同一份代码在仓库里）：
```bash
cd <repo> && python ds_workspace/recon_20260910/code/pro_tables_20260911.py
```

**它构造并自检三张表，全部断言通过**：

| 表 | 来源 | 已核验 |
|---|---|---|
| `step42` | Pro §9.3 | `m_j = 1[j≥22]`，**S = 42.000000 精确**（四个平台成员同预算；此前 `step_hi25` 的 S=39，从未预算匹配过） |
| `condEVQ` | Pro §8 | **τ = 2.0301373113**、**S = 42.000000**、**增量方差 = 18.78373**、阈值中心 22、ν 严格递减、两端与平台与赢家一致 —— **逐条与 Pro 给的值吻合** |
| `transport_a1b64` | Pro §7 | 传输常数 `a·C = 9.632959861`、`a·D = 0.949828334`；**传输后 S = 33.47081679** —— 与 Pro 给的 `33.4708167895` 逐位吻合 |

**跑这两张表**（runner 已接好，`--dry-run` 已验证）：
```bash
# 350 行 RULER（仲裁者）
ssh -p 27741 root@connect.westc.seetacloud.com 'cd /root/autodl-tmp/phase1_20260910 && \
export PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910/repoharness:/root/autodl-tmp/phase1_20260910 && \
setsid nohup /root/miniconda3/bin/python olmo_beta.py \
  --root /root/autodl-tmp/phase1_20260910/olmo_pro \
  --model /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct \
  --panel /root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl \
  --archive /root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01 \
  --betas "" --turns "" --pro-tables condEVQ,step42 \
  > olmo_pro.log 2>&1 < /dev/null & disown'
```

```bash
# cheap 连续仪器先筛（~1 分钟）
ssh -p 27741 root@connect.westc.seetacloud.com 'cd /root/autodl-tmp/phase1_20260910 && \
export PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910/repoharness:/root/autodl-tmp/phase1_20260910 && \
setsid nohup /root/miniconda3/bin/python olmo_longnll.py \
  --root /root/autodl-tmp/phase1_20260910/contpro \
  --model /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct \
  --nll-dir /root/autodl-tmp/olmo_fast_screen_20260908/prepared_nll_02 \
  --only pro_condEVQ,pro_step42,beta_b1_BM > contpro.log 2>&1 < /dev/null & disown'
```

**预注册预测（读数前写下）**：

| 表 | 预测 | 依据 |
|---|---|---|
| `step42` | **应当仍差**（若阶跃族在该预算下也输，则阶跃族在此预算出局；Pro 明确说这**不能**推广到其它预算，也不能定位是哪个任务项造成的） | 此前 `step_hi25`（S=39）RULER 0.1121 |
| `condEVQ` | **未知**。Pro 明确说它**不是**性能最优表，f 只是"已验证表内部的一个零预算残差方向"。终点本身是一次便宜的初看 | — |

**Pro §8 的零预算残差方向**（等 B/D 结果后再用）：
`m(a) = m_win + a·(m_condEVQ − m_win)`，`0 ≤ a ≤ 1`。任意 a 保持 S=42、两端、排序与单调性。
**这是一个连续方向，不是一次扫描**——a 由真实任务响应决定。

---

## §4 Pro 模型给出的、**与我们结论冲突**的三处（已写进文档）

| Pro 的说法 | 我们的原说法 | 处置 |
|---|---|---|
| **角点定理从未适用于我们测的东西**——它是为另一个成本 `C_j = 4t_j(1−4^(−m))`（凹）推的，而 `derive_tstar` 代入的是 `0.5(ln4)²ΣF_jj m_j²`（凸、在 native 处斜率 0）。代换改变了曲率 | 我们写"Fisher 路线死于符号错" | **两者都对且互补**。Pro 的更精确：**曲率变了**，所以角点定理不适用；我们的符号发现是**为什么**凸替换是错的 |
| **不能称水填充/KKT 被实验否定**。Fisher 成本在 m=0 处导数为 0，而假定收益 U'(0)>0 ⟹ 这个可分离模型**本身就预测小幅渗漏** | 我们写"taper/leak 判决了水填充" | **接受更正**。taper/leak 是**具体干预的成本**，不是对水填充的普遍否定 |
| **gain 是效应修饰因子，不是混淆项**；且不能说"文献从未联合处理 gain"（AdaRoPE 就是） | 我们写"整个文献可能在次优 gain 上比较" | **接受更正**，措辞已弱化 |

**Pro 还指出一条我们没做的**（`NO_STATIC_FUNCTIONAL` 的补充）：
64 个对角**不足以**代表 Fisher——`F_plus=[[1,.99],[.99,1]]` 与 `F_minus=[[1,-.99],[-.99,1]]`
对角相同、沿 (1,1) 差 **199 倍**。子代理 A 补：`ANALYTIC_BUDGET §2.1` 把**损失对角性**标成 `[证]`，
但 `∂log ν_j/∂m_j = −ln4` 是**频率**的导数、不是损失的导数——**对角性是假设，不是定理**。
我们自己的非可加性证据（单槽 ≤0.05 vs 整表 0.03–0.86）独立地杀掉可分离泛函。

> ### ⚠ 但 (f) 作为 SNR 修复**被否证**
> 子代理 A 定量算过：**方向探针无论移动多少槽，只给【一个】配对损失数 ⟹ 没有跨槽平均**。
> 增益最多 `√N_eff = √18 = 4.24×`：
> - 窗内：单槽 SNR 0.02–0.10 ⟹ 方向 SNR **0.08–0.42，仍 < 1**（差约 30 倍）
> - 长程：单槽 SNR 0.03–0.49 ⟹ 方向 SNR **0.12–2.1，勉强**
> - 而且 `√18` 也能靠平均 18 次单槽测量拿到（同成本）⟹ **(f) 的真正内容是【成本】（7 vs 65 次前向 = 9.3×），不是新的统计功效。**
>   **在它真正好用的步长（`‖Δm‖ ≳ 0.25`）上，它已经不是梯度探针了。**
> **结论：(f) 值得做，但理由是省算力；它不是 SNR<1 的解。** 子代理 A 另建议**改靶测 R 的 Hessian**（Fisher 已被证伪）。

---

## §5 还没跑的（按 ROI 排序）

1. **`--pro-tables condEVQ,step42`**（§3）——命令已给，runner 已验证
2. **投影 Fisher**（§4 末）——需要新脚本。**注意：作为 SNR 修复已被否证**（§4），它的价值是**成本 9.3×**；子代理 A 建议改靶测 R 的 Hessian
3. **Pro §7 的 Qwen 传输表**——`transport_a1b64` 已构造（S=33.47081679），可直接加进 `qwen_longnll.py` 的臂表；它测的是**物理谱形状迁移**，不是"Q/K 用途相同"
4. **`--gain-tables` 2×2**——已排在 `chain_pro`，命令：`--gains 1.0,1.138629436111989 --gain-tables a1_b64,mrpro,b3`
5. **Pro §6 的 `D_pattern`**——**零 GPU，从现有 jsonl 直接算**：`Δ_mean = mean(A−B)`、`D_pattern = mean(|A−B|)`。**均值相同但 D_pattern 大 ⟹ 它们修的是不同样本。** 子代理 `pro-verify-B` 在做

---

## §6 服务器上新增/改动的文件（都在 `phase1_20260910/`）

| 文件 | 作用 |
|---|---|
| `pro_tables_20260911.py` | Pro 的三张表 + 自检 |
| `patch_pro_arms.py` / `repair_pro_arms.py` | 把 pro 表接进两个 runner（第二个是修复第一个的 bug） |
| `patch_gain2x2.py` | 给 `olmo_beta.py` 加 `--gain-tables`（否则 2×2 不可达） |
| `chain_pro_items.py` | 排 `chain_pro.sh`：`--steps 22` → gain 2×2 |
| `patch_release.py` / `patch_agent_tables.py` | release 轴与两个子代理的表（**已测，全部差于 BM**） |
| `ruler_paired.py` | 跨臂配对检验（服务端跑） |
| `fisher_stability.py` / `marginal_damage.py` / `kkt_gradient.py` / `kkt_step.py` | 本轮四个诊断仪器 |
| `qwen_longnll.py` | Qwen 4× 长程仪器 |
| `holdout_final.py` | held-out 最终测试链 |
