# -*- coding: utf-8 -*-
"""T1 候选频率表生成器（纯 CPU，幂等）。
用法: python3 analysis/unify_20260910/tables/generate_candidate_tables.py
产出: 同目录 CANDIDATE_TABLES.md / .csv / .json
数据源: analysis/unify_20260910/tables/ground_truth_tables.json (G1, 只读校验用)
规则族: D4_transport_rule.md §2/§4/§5；BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md §3（队列 0446/0448/0449 冻结定义）；GROUND_README.md §6 修正表。
"""
import json, math, os, csv

HERE = os.path.dirname(os.path.abspath(__file__))
G1 = json.load(open(os.path.join(HERE, "ground_truth_tables.json")))
C = G1["meta"]["constants"]
W, S, L, Dr = 32768, 4, 131072, 64
NB = C["native_log_gap"]          # ln b / 64 = 0.21586735246819178
LN4 = C["lnS"]                    # 1.3862943611198906
RHO_NAT = C["native_period_ratio"]# 1.2409377607517196
OFFICIAL_GAIN = C["official_gain"]# 1.138629436111989

def omega(j):          # 原生频率 b^{-j/64}
    return math.exp(-NB * j)

def ramp_m(N):         # 径向族（MrRoPE Eq.14）: m_q = q(q+1)/(N(N+1)), q=clip(j-23,0,N)
    def f(j):
        q = max(0, min(N, j - 23))
        return q * (q + 1) / (N * (N + 1))
    return f

def uni_m(N):          # 均分桥（MrRoPE Eq.13, 盲解）: m_j = min(1, (j-23)/N)
    def f(j):
        return min(1.0, max(0.0, (j - 23) / N))
    return f

NOTCH = 10.0 / 306.0   # s28_less 相对 MrPro 的 m28 回拨量（m 单位）= 0.0326797
NOTCH_NATS = NOTCH * LN4  # 0.045304 nats

def with_notch(base_m):
    mm = list(base_m)
    mm[28] = mm[28] - NOTCH
    return mm

def smooth27(base_m):   # 变体 b：完全抹平 δ27（m28 := m27）
    mm = list(base_m)
    mm[28] = mm[27]
    return mm

def arr(fn):
    return [fn(j) for j in range(64)]

# ---------- 表定义 ----------
M = G1["methods"]
stack_m = list(M["StackFrontBack"]["m_j"])   # G1 公式重建数组（0446）
# G1 反演噪声更正：slot27 与 slot28 相差 3.8e-9（m28<m27，假单调违例）。
# s28 手术构造即 m28:=m27（E1 部署两槽同 fp32 值）；交付表取 m28:=m27（G1 值），其余逐位采 G1。
STACK27_28_NOISE = abs(stack_m[28] - stack_m[27])
stack_m[28] = stack_m[27]

TABLES = []
def add(id_, name, m, family, Np, **kw):
    TABLES.append(dict(id=id_, name=name, m=[float(x) for x in m], family=family, Np=Np, **kw))

for N in (17, 16, 15, 14, 13):
    add(f"ramp_N{N}", f"径向族 Eq.14 N′={N}", arr(ramp_m(N)), "ramp", N)
add("stack_0446", "0446 StackFrontBack（s28_less ⊕ LBS 双手术拼接）", stack_m, "composite", None)
r16 = arr(ramp_m(16)); r15 = arr(ramp_m(15))
add("ramp16_notch", "ramp16 ⊕ s28缺口（D4 §5 条件主选，0446 判可加后）", with_notch(r16), "ramp+notch", 16)
add("ramp15_notch", "ramp15 ⊕ s28缺口（次选形）", with_notch(r15), "ramp+notch", 15)
for N in (17, 16, 15, 14, 13):
    add(f"uniform_N{N}", f"均分桥 Eq.13（盲解族）N′={N}", arr(uni_m(N)), "uniform", N)

# ---------- 逐槽派生量 ----------
def slots(m):
    out = []
    for j in range(64):
        mj = m[j]
        nu = omega(j) * (4 ** (-mj))
        T = 2 * math.pi / nu
        D = W * (4 ** mj)
        r = omega(j) * W / (2 * math.pi)          # 原生窗内圈数（m 不变量，=G1 r_j_native）
        eps = m[j + 1] - mj if j < 63 else None   # ε_{j→j+1}（m 单位）
        lam = (4 ** eps) if eps is not None else None
        rho = RHO_NAT * lam if lam is not None else None  # = T_{j+1}/T_j（D4 §0 gap 编号：本行 ρ 即 gap j 的洞比）
        out.append(dict(j=j, m=mj, nu=nu, T=T, D=D, r=r, eps=eps, lam=lam, rho=rho))
    return out

def metrics(m):
    sl = slots(m)
    holes = [(j, sl[j]["rho"]) for j in range(63)]
    gmax_j, gmax = max(holes, key=lambda t: t[1])
    band = [(j, sl[j]["rho"]) for j in range(23, 40)]
    gb_j, gb = max(band, key=lambda t: t[1])
    js = min(j for j in range(24, 64) if m[j] >= 1 - 1e-12)
    cons = sum(NB + LN4 * (m[g + 1] - m[g]) - NB for g in range(23, 40))  # Σ桥超原生预算 = ln4·(m40−m23)
    return dict(
        max_hole=gmax, max_hole_gap=gmax_j,
        band_hole=gb, band_hole_gap=gb_j,
        j_star=js,
        F_arc=sum(max(0.0, 1 - m[j]) for j in (36, 37, 38, 39)),
        BL=sum(m[j] for j in range(24, 29)),
        sum_m=sum(m), B_minus24=sum(m) - 24.0,
        D36_39=[W * 4 ** m[j] for j in (36, 37, 38, 39)],
        m36_39=[m[j] for j in (36, 37, 38, 39)],
        m28=m[28], m39=m[39],
        conservation_ln4=cons, conservation_resid=abs(cons - LN4),
        mono_ok=all(m[j + 1] >= m[j] - 1e-12 for j in range(63)),
        box_ok=all(-1e-12 <= x <= 1 + 1e-12 for x in m),
        I1_ok=all(abs(m[j]) < 1e-12 for j in range(24)),
        I2_ok=all(abs(m[j] - 1) < 1e-12 for j in range(40, 64)),
    )

for t in TABLES:
    t["slots"] = slots(t["m"])
    t["met"] = metrics(t["m"])

# ---------- 校验（vs G1 与文档锚点） ----------
def mdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))

checks = []
def chk(name, got, exp, tol=0.0, note=""):
    ok = abs(got - exp) <= tol
    checks.append(dict(check=name, got=got, expected=exp, tol=tol, verdict=("MATCH" if ok else "MISMATCH"), note=note))
    return ok

chk("ramp17_vs_G1_MrPro_maxdm", mdiff(arr(ramp_m(17)), M["MrPro"]["m_j"]), 0.0, 1e-6, "G1 声明 ν 张量 bit-exact（sha 33cbe3a4…）")
chk("ramp16_vs_G1_MrProN16_maxdm", mdiff(arr(ramp_m(16)), M["MrProN16"]["m_j"]), 0.0, 1e-6, "0448 冻结公式 q(q+1)/272")
chk("ramp15_vs_G1_MrProN15_maxdm", mdiff(arr(ramp_m(15)), M["MrProN15"]["m_j"]), 0.0, 1e-6, "0449 冻结公式 q(q+1)/240")
chk("uniform17_vs_G1_MrUni_maxdm", mdiff(arr(uni_m(17)), M["MrUni"]["m_j"]), 0.0, 1e-6, "G1 声明 ν bit-exact（sha 4f35b71e…）；实测 64.5833/73.3333")
comp = arr(ramp_m(17)); comp[28] = M["E1_s28_less"]["m_j"][28]
for j in (36, 37, 38, 39): comp[j] = M["LongBridgeSlower"]["m_j"][j]
chk("stack0446_vs_双手术拼接_maxdm", mdiff(stack_m, comp), 0.0, 1e-6, "G1 Stack 数组 = MrPro⊕s28(28)⊕LBS(36–39) 逐槽差≤4e-9；交付表槽28已取 m28:=m27 抹平 G1 反演噪声（见下条）")
chk("stack_G1_noise_27_28", STACK27_28_NOISE, 0.0, 1e-7, "G1 原数组 m28 比 m27 低 3.8e-9：s28 构造两槽应相等（E1 部署同 fp32 值），反演噪声所致；交付取 m28:=m27")
tstack = next(t for t in TABLES if t["id"] == "stack_0446")
chk("Stack_max_hole_value", tstack["met"]["max_hole"], 1.4930, 5e-4, "GR §6 修正值复算吻合")
chk("Stack_max_hole_pos", tstack["met"]["max_hole_gap"], 38, 0, "GR §6/任务口径标 @g35 —— 位置标签笔误（值 1.4930 在 g38）")
chk("Stack_m28_vs_doc", tstack["met"]["m28"], 0.065, 5e-4, "BUDGET:48 文档值 .065 ✓（精确 0.065359；行号引用于 2026-09-10 由 45→48 更新）")
tr16 = next(t for t in TABLES if t["id"] == "ramp_N16")
chk("N16_max_hole", tr16["met"]["max_hole"], 1.4608, 5e-4, "GR §6 1.461@g38")
chk("N16_max_hole_pos", tr16["met"]["max_hole_gap"], 38, 0)
tr15 = next(t for t in TABLES if t["id"] == "ramp_N15")
chk("N15_max_hole", tr15["met"]["max_hole"], 1.4757, 5e-4, "GR §6 1.476@g37")
chk("N15_max_hole_pos", tr15["met"]["max_hole_gap"], 37, 0)
tr13 = next(t for t in TABLES if t["id"] == "ramp_N13")
tr14 = next(t for t in TABLES if t["id"] == "ramp_N14")
chk("N13_max_hole", tr13["met"]["max_hole"], 1.513, 5e-3, "D4 §4 表 1.513@g35")
chk("N13_max_hole_pos", tr13["met"]["max_hole_gap"], 35, 0)
chk("N14_max_hole", tr14["met"]["max_hole"], 1.493, 5e-3, "D4 §4 表 1.493@g36")
chk("N14_max_hole_pos", tr14["met"]["max_hole_gap"], 36, 0)
chk("MrPro_D39", M["MrPro"]["D_j"][39], 112361, 1.0, "GR §5-2 复算值")
chk("N16_notch_m28", next(t for t in TABLES if t["id"] == "ramp16_notch")["met"]["m28"], 30/272 - NOTCH, 1e-12, "= 30/272 − 10/306（固定量回拨，D4 §5 口径 0.0453 nats）")
for t in TABLES:
    chk(f"conservation_{t['id']}", t["met"]["conservation_resid"], 0.0, 1e-6)
    assert t["met"]["mono_ok"] and t["met"]["box_ok"] and t["met"]["I1_ok"] and t["met"]["I2_ok"], t["id"]

# ---------- 搬运量（vs MrPro） ----------
mrpro = arr(ramp_m(17))
def transport(m):
    d = [m[j] - mrpro[j] for j in range(64)]
    return dict(sum_abs_dm=sum(abs(x) for x in d), max_slot_shift=abs(max(d, key=abs)),
                sum_pos=sum(x for x in d if x > 0))

# ---------- 序列化 ----------
def fmt(x, nd): return f"{x:.{nd}f}"

HEADER_COLS = ["j", "m_j", "ν_j", "T_j", "D_j", "r_j", "λ_j=4^ε_{j→j+1}", "ρ_j=T_{j+1}/T_j"]
def md_table(t, prec_m=6):
    lines = ["| " + " | ".join(HEADER_COLS) + " |", "|" + "---|" * 8]
    for s in t["slots"]:
        lam = "—" if s["lam"] is None else fmt(s["lam"], 6)
        rho = "—" if s["rho"] is None else fmt(s["rho"], 4)
        lines.append(f"| {s['j']} | {fmt(s['m'], prec_m)} | {s['nu']:.6e} | {s['T']:,.1f} | {s['D']:,.0f} | {fmt(s['r'],3)} | {lam} | {rho} |")
    return "\n".join(lines)

IDENTITY = {
 "ramp_N17": ("**≡ 已部署 MrPro 基线**（MrRoPE-Pro，Eq.14 在 (dl,dh)=(23,40)）。G1 公式重建与部署 fp32 ν 张量 **bit-exact**（sha 33cbe3a4…，[已验证]，GR §3）。36 行面板 **87.2222 / 78.1250**（[已验证]，digest_panel §5.1 A1）。恒等检查 = 本表生成器对 G1 `m_j` 数组 max|Δm| < 1e-6（实测 ~3.7e-8）。", "已执行（面板基线）"),
 "ramp_N16": ("**= 队列 0448 MrProN16**（BUDGET:49 冻结定义 q(q+1)/272；GR §6 修正值 max洞 1.461@g38、B=6.000）。G1 公式重建吻合（max|Δm|≈3.3e-8）。**未执行，无分数**。预注册预测：D2 泛函 R=51.15 > MrPro 47.44 ⇒ 预测劣于 MrPro [假设-预注册]；D4 平衡权重 s=.5 下 argmin N′*=16 ⇒ 预测全场最优 [部分证据-边界内决策]——两判据相反，0448 正是判决实验。", "队列 0448（GPU 回来即跑）"),
 "ramp_N15": ("**= 队列 0449 MrProN15**（BUDGET:50，q(q+1)/240；GR §6 max洞 1.476@g37、B=6.667；D4 表中 F_arc=.3667）。G1 重建吻合（≈3.6e-8）。**未执行，无分数**。D2 预测 R=51.51 劣于 MrPro；D4 次选 N′*=15（s∈[.2425,.4029]〔更正 2026-09-10，V_math §4：原 .24/.41〕）[假设-预注册]。", "队列 0449"),
 "ramp_N14": ("**新构造，未入队、未测**。最近已测参照：完成位置槽 37 介于 P2（槽 32 完成，72.92/81.67 [已验证]）与 LBS（m39=.9799 未完成，80.56/80.07 [已验证]）之间，但形状为纯径向族、与二者均不同。D4 §4 [已验证比较]：mid 带洞 δ=.1238 越过 Smooth 级天花板 .1130 ⇒ 盲形不可行，须带 mid 缺口才有资格入队。", "未入队新点"),
 "ramp_N13": ("**新构造，未入队、未测**。max 洞 1.513@g35 为族内最宽洞（D4 §4 表 [已验证]）；mid δ=.1429 超 Smooth 级天花板，violation 最深。仅作族梯度端点呈现，不构成候选。", "未入队新点"),
 "stack_0446": ("**= 队列 0446 StackFrontBack**。构造 = MrPro ⊕ s28_less(槽28) ⊕ LBS(槽36–39)（BUDGET:48）。本表直接采用 G1 `StackFrontBack.m_j` 数组，并独立复算其 = 两个**已验证**手术数组的拼接（逐槽零差，校验项 stack0446_vs_双手术拼接）。m28=0.065359（doc .065 ✓）、m36–39=.6252/.7295/.8465/.9799（doc .625/.729/.847/.980 ✓）。分量身份：s28_less 87.2222/83.3333、LBS 80.5556/80.0694（均 [已验证]）。**拼接体未执行，无分数**。D2 预测 R=46.55 全场最优 [假设-预注册]；D1 §6 预测长端修复被洞部分回收 [部分证据]——分歧本身即 0446 的检验内容。", "队列 0446"),
 "ramp16_notch": ("**新构造（D4 §5/§6.3 条件主选）**：若 0446 判双机制可加，主交付改为 ramp16⊕notch。notch = m28 回拨 10/306=0.032680 m 单位（=0.045304 nats，s28_less 对 MrPro 实测同量 [已验证]），下游 m_29+ 不动（守恒自动保持）。未入队、未测，无分数。变体（δ27 完全抹平 m28:=m27=0.073529，回拨 0.050970 nats）见 §5.1 注。", "未入队（条件候选）"),
 "ramp15_notch": ("**新构造（次选形）**：ramp15 同 notch。若 0449 判 N15>N16（族继续下探），D4 §6.3 要求下探必须带缺口——本表即该分支的预构造。未入队、未测。变体 m28:=m27=0.083333（回拨 0.057762 nats）见注。", "未入队（条件候选）"),
 "uniform_N17": ("**≡ 已部署 MrUni**（G1 声明 ν bit-exact，sha 4f35b71e…，[已验证]）。36 行 **64.5833 / 73.3333**（[已验证]）。D4 §2 关键否定结果：字面盲目标（位置无关洞项）的解族成员，被面板否证——bank 负载 BL=0.8824 灾难（D1 A3：m28=.294 → 32K −22.6pp）。列此作身份检查与反例锚。", "已执行（已测败点）"),
 "uniform_N16": ("盲解族参照点（未执行、未入队）。洞平坦 ρ=1.3533 全绿，但 BL=0.75 仍在 MrUni 型 bank 灾难区。", "未入队（参照）"),
 "uniform_N15": ("盲解族参照点（未执行）。BL=0.625。", "未入队（参照）"),
 "uniform_N14": ("盲 min-max 打平点（洞 .0714 = 弧 .0714 精确打平，D4 §2 [已验证=代数]），但 mid δ=.1238 违 Smooth 级天花板 ⇒ 盲解连同 argmin 作废为目标函数错设证据 [已验证比较]。未执行。", "未入队（参照）"),
 "uniform_N13": ("盲解族端点参照（未执行）。BL=0.375，洞 .0769；F_arc=0。", "未入队（参照）"),
}

PRED = {  # 只引已预注册的结构预测，不虚构分数
 "ramp_N16": ["D2 泛函 R=51.15（预测劣于 MrPro 78.13）", "D4 argmin N′*=16（预测全场最优）", "预注册判据：长端 78.13<N16≤N15 单调 ⇒ ramp 单参数族成立；N16>max(N15,78.13) ⇒ D4 命中锁定 N′*=16；N16≤78.13 非单调 ⇒ 天花板项活跃，回摆 N′=17 或 16+notch。**重叠象限裁决 [补充 2026-09-10，V_predictions b4/修复表 3：$N16\\le78.13\\wedge N15>N16$（V 形）同时触发本行\"回摆\"与 2.3 节\"下探\"且动作相反——优先序约定：V 形 ⇒ N16 为局部极小，先下探（带 mid 缺口的 14/13）；下探仍 ≤78.13 且相对 N17 无改善才回摆 17/16+notch。T2 K4 只覆盖另一象限，本条补齐；详见 D4 §6.3]**"],
 "ramp_N15": ["D2 泛函 R=51.51（预测劣于 MrPro）", "D4 次选（$s\\in[.2425,.4029]$〔更正 2026-09-10，V_math §4：原 .24/.41，复算 .2425/.4029〕）", "判据：N15>N16 ⇒ 沿族下探 14/13 但必须带 mid 缺口（若同时 $N16\\le78.13$，V 形象限按 2.2 节优先序执行 [V_predictions b4]）"],
 "stack_0446": ["D2 泛函 R=46.55（预测优于全部 5 张已测表，含 s28 46.67）", "D4 判据：≥83%@128K 且 ≥80%@32K ⇒ 双机制可加 ⇒ 主交付改 ramp16⊕notch；若不敌 max(s28,LBS)=83.33 ⇒ notch 与完成不可同预算〔补充 2026-09-10，V_predictions b3/b5/修复表 2+6：(i) 区间 $78.125\\le\\text{Stack}\\le83.3333$（赢 MrPro、输 s28）按 T2 **K1b 中间判据**处理——D2\"优于全部 5 表\"序陈述死、双机制不杀、主交付不动；原判据在此区间无预注册动作的缺口已闭合。(ii) \"≥80%@32K\" 分量**执行条件化**：协议 NONGEOMETRIC（\"No new 32K experiments\"）未解除前不可评估，队列若 128K-only 则记 N/A 不记判负（T2 (e) R8）〕"],
}

# ---------- 汇总指标表 ----------
def met_row(t):
    mt = t["met"]
    return (f"| `{t['id']}` | {t['name']} | {mt['j_star']} | {fmt(mt['m28'],6)} | "
            f"{fmt(mt['band_hole'],4)}@g{mt['band_hole_gap']} | {fmt(mt['F_arc'],4)} | {fmt(mt['BL'],4)} | "
            f"{fmt(mt['sum_m'],3)} | {fmt(mt['B_minus24'],3)} | {mt['D36_39'][3]:,.0f} | "
            f"{IDENTITY[t['id']][1]} |")

SUMMARY_HDR = ("| 表 | 构造 | j\\* | m₂₈ | max洞(带) | F_arc | BL | Σm | B=Σm−24 | D₃₉ | 状态 |\n|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|")

MD = []
A = MD.append
A("# T1 候选频率表（零训练 RoPE 上下文扩展 · Qwen2.5-3B · 64 槽全表）")
A("")
A("日期 2026-09-10。GPU 关闭，纯 CPU 交付。生成器：`analysis/unify_20260910/tables/generate_candidate_tables.py`（幂等，直接 `python3` 运行可复现本文全部数字）。机读版：`CANDIDATE_TABLES.json`（64 槽全数组）与 `CANDIDATE_TABLES.csv`（长表）。")
A("")
A("规则出处：D4 `analysis/unify_20260910/answers/D4_transport_rule.md` §2/§4/§5/§6；队列冻结定义 `docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md` §3–4；修正值表 `analysis/unify_20260910/tables/GROUND_README.md` §6。地面真值：`ground_truth_tables.json`（G1，只读）。")
A("")
A("## 0. 记号、常数与约定")
A("")
A("$$\\nu_j=\\omega_j S^{-m_j},\\quad \\omega_j=b^{-j/64},\\quad b=10^6,\\ S=4,\\ W=32768,\\ L=131072,\\ j=0..63$$")
A("")
A(f"- $\\ln S=\\ln4={LN4:.6f}$；原生 log-gap $\\mathrm{NB}=(\\ln b)/64={NB:.7f}$；原生周期比 $\\rho_{{\\rm nat}}=b^{{1/64}}={RHO_NAT:.7f}$；官方 gain $a=1+0.1\\ln4={OFFICIAL_GAIN:.6f}$（三张队列表与全部对照共用官方 gain [BUD §3]）。[已验证：G1 `meta.constants` 逐位]")
A("- 逐槽列定义（与 G1 口径一致 [GR §2]）：$T_j=2\\pi/\\nu_j$（部署周期）；$D_j=W\\cdot4^{m_j}$（识别地平线）；$r_j=W/T_j^{\\rm nat}=\\omega_j W/2\\pi$（**原生窗内圈数**，对全族为 $m$ 不变量；部署后 128K 内圈数 $=r_j\\cdot4^{1-m_j}$）；$\\varepsilon_{j\\to j+1}=m_{j+1}-m_j$；$\\lambda_j=S^{\\varepsilon_{j\\to j+1}}=4^{\\varepsilon}$（本步频率缩放因子）；洞比 $\\rho_j=T_{j+1}/T_j=\\rho_{\\rm nat}\\cdot\\lambda_j$（**gap 编号约定：第 j 行的 $\\rho_j$ 是槽 $j\\to j+1$ 那个 gap 的洞比**，与 [D4 §0][G1 `gap_j`] 同；末行 j=63 无 next，记 —）。")
A("- 桥带 $g\\in[23,39]$（17 个 gap）；端点不变量 I1（$m_j=0,\\ j\\le23$）、I2（$m_j=1,\\ j\\ge40$）；守恒 $\\sum_{g=23}^{39}(\\mathrm{gap}_g-\\mathrm{NB})=\\ln S$（每张交付表机器核验，校验节列出，残差 <1e-6 [已验证]）。")
A("- 证据级标注：[已验证]=对 G1/部署张量/文档锚点复算一致；[已验证=代数]闭式；[假设-预注册]=冻结队列判决实验的预注册预测，**不是分数**；未执行臂一律无分数，不虚构（G1 evidence_policy、SYNTHESIS §7）。")
A("")
A("### 0.1 术语勘定（任务口径 vs 地面真值）")
A("")
A("任务 (1) 称\"均分桥 $N'\\in\\{13..17\\}$、$N'=17\\approx$ MrPro 恒等检查\"。地面真值的两个锚点唯一钉死其指**径向族**（MrRoPE Eq.14 形，$m_q=q(q+1)/(N'(N'+1))$）：MrPro 部署 = $N'=17$ 径向族 bit-exact [已验证]；0448/0449 冻结定义 = \"MrPro **径向族** $N'=16/15$\" [BUDGET:49-50]。字面均分桥（Eq.13 形，$m_j=\\min(1,(j-23)/N')$）的 $N'=17$ 成员是**已测败点 MrUni**（64.5833/73.3333 [已验证]），非 MrPro。故本文**主交付 = 径向族五表**；均分族五表全部附带（§4 参考节 + CSV/JSON 全量），其 $N'=17$ 表即 MrUni 身份检查。")
A("")
A("### 0.2 勘误与新发现（本轮复算）")
A("")
A(f"1. **0446 Stack 最大洞位置标签更正**：GR §6/D1 §5/D2 c3 记 \"1.493@**g35**\"。按 G1 Stack 数组直算，$\\rho_{{35}}=1.4562$，最大值 $1.4930$ 在 **g38**（$m_{{39}}-m_{{38}}=0.13338$）[已验证]。值 1.4930 无误（GR §6 采用修正值本已正确），**位置标签为笔误**；N16@38、N15@37 标签无误 [已验证]。结构读法不变：Stack 洞源在 LBS 完成手术本身（与 LBS 单臂同洞），与 s28 分量无关——GR §6\"洞源在 gap35→36 边缘\"的归因句随位置标签一并更正为 gap38→39。")
A("2. BUDGET 旧版（行号 45–47，现 48–50，行号引用 2026-09-10 随 BUDGET 验证修订下移更新）的 max 洞 1.86/1.76/1.80× 维持作废，以 GR §6 重算值为准（本生成器逐表复算 MATCH，见 §5.2）。")
A("3. UNIFIED:41 \"m36–39=0.51–0.89\" 槽位错位（0.5098=m35）维持 [GR §5-2 / D3 §1]，本文件全部按 0.5948–0.8889 口径。")
A(f"4. G1 `StackFrontBack.m_j` 数组槽 27 与 28 相差 {STACK27_28_NOISE:.2e}（$m_{{28}}<m_{{27}}$，假单调违例）：s28 手术构造即 $m_{{28}}:=m_{{27}}$（E1 部署两槽同 fp32 值），差异为 fp32 反演噪声 [已验证]。交付表取 $m_{{28}}:=m_{{27}}$（G1 值 0.06535945），其余 63 槽逐位采 G1。")
A("")
A("## 1. 全部候选表汇总指标")
A("")
A("(D₃₉ 列为危险区最右槽地平线；MrPro 对照 112,361 < L=131,072 是其欠完成的定量。j\\* = 严格完成阈 $\\min\\{j\\ge24:m_j=1\\}$。F_arc $=\\sum_{36..39}(1-m_j)^+$，BL $=\\sum_{24..28}m_j$——均为 D4 §3 坐标。)")
A("")
A(SUMMARY_HDR)
for t in TABLES:
    A(met_row(t))
A("")
A("径向族族梯度（D4 §5 主搬运）复核 [已验证=代数]：$17\\to16$ 从末 gap g39 抽 $0.111111$ m 单位（$0.154033$ nats（$=\\ln4/9$），$\\ln4$ 的 11.11%）〔更正 2026-09-10，V_math §5/错误目录 7：原 $0.15399$ 第 4 位错〕摊回 g23–38（各 $\\times1.125$），槽 39 完成（$D_{39}\\to131072$）、$F_{\\rm arc}\\ 1.046\\to0.6765$，代价最大洞 $1.448@g39\\to1.461@g38$ 左移；$16\\to15$ 再抽 g38 的 $0.117647$（$0.163093$ nats〔更正 V_math §5：原 $0.16306$〕）×$1.1333$ 摊回，槽 38 完成，$F_{\\rm arc}\\to0.3667$。搬运全程 bank/尾带零触碰（I1/I2 保持，校验节逐表 MATCH）。")
A("")

# ---------- 分表 ----------
A("## 2. 径向族（Eq.14 形，主交付）")
A("")
A("$$m_j=\\frac{q(q+1)}{N'(N'+1)},\\qquad q=\\mathrm{clip}(j-23,\\,0,\\,N'),\\qquad j^*=23+N'$$")
A("")
_sec = 0
for t in TABLES:
    if t["family"] not in ("ramp", "ramp+notch"):
        continue
    _sec += 1
    mt = t["met"]
    A(f"### 2.{_sec} `{t['id']}` — {t['name']}")
    A("")
    A("**身份关系**：" + IDENTITY[t["id"]][0])
    A("")
    tp = PRED.get(t["id"])
    if tp:
        A("**预注册预测（非分数）**：" + "；".join(tp) + "。")
        A("")
    if t["family"].startswith("ramp"):
        A(f"构造参数：$N'={t['Np']}$，$j^*={mt['j_star']}$，分母 $N'(N'+1)={t['Np']*(t['Np']+1) if t['Np'] else '—'}$，每步行程 $\\varepsilon_{{22+q}}=q/\\tfrac{{N'(N'+1)}}{{2}}$；末步 $\\varepsilon=\\frac{{2}}{{N'+1}}$，末步 $\\lambda=4^{{2/(N'+1)}}$。" if t["family"] == "ramp" else f"构造参数：ramp $N'={t['Np']}$ + notch $m_{{28}}:-10/306$（0.032680 m 单位 = 0.045304 nats）。")
        A("")
    tr = transport(t["m"])
    A(f"vs MrPro 位移：$\\sum|\\Delta m|={tr['sum_abs_dm']:.4f}$，最大单槽 $|\\Delta m|={tr['max_slot_shift']:.4f}$。关键值：$m_{{28}}={fmt(mt['m28'],6)}$，$m_{{36..39}}$=" + "/".join(fmt(x, 4) for x in mt["m36_39"]) + f"，$D_{{39}}$={mt['D36_39'][3]:,.0f}，max 洞 $\\rho={fmt(mt['band_hole'],4)}$@g{mt['band_hole_gap']}，$F_{{\\rm arc}}={fmt(mt['F_arc'],4)}$，$BL={fmt(mt['BL'],4)}$，$\\Sigma m={fmt(mt['sum_m'],3)}$，$B={fmt(mt['B_minus24'],3)}$。守恒残差 {mt['conservation_resid']:.2e}。[已验证=CPU 复算]")
    A("")
    A(md_table(t))
    A("")

A("## 3. 队列 0446 StackFrontBack（双手术拼接，G1 数组直采）")
A("")
ts = next(t for t in TABLES if t["id"] == "stack_0446")
mt = ts["met"]
A("**身份关系**：" + IDENTITY["stack_0446"][0])
A("")
A("**预注册预测（非分数）**：" + "；".join(PRED["stack_0446"]) + "。")
A("")
A(f"逐槽手术清单：$m_{{28}}$={fmt(mt['m28'],6)}（s28_less 值，=m27，$\\delta_{{27}}=0$）；$m_{{36..39}}$=" + "/".join(fmt(x,4) for x in mt["m36_39"]) + f"（LBS 值）；其余 59 槽 = MrPro 逐位（$m_{28}$ 取 $m_{27}$ 值以抹平 G1 反演噪声，§0.2-4）。max 洞 $\\rho={fmt(mt['band_hole'],4)}$@g{mt['band_hole_gap']}（与 LBS 单臂同值同位，s28 分量不新增洞），$F_{{\\rm arc}}={fmt(mt['F_arc'],4)}$（=LBS），$BL={fmt(mt['BL'],4)}$（=s28），$\\Sigma m={fmt(mt['sum_m'],3)}$，$B={fmt(mt['B_minus24'],3)}$，$D_{{39}}$={mt['D36_39'][3]:,.0f}（doc \"127.5K\" ✓）。[已验证=对 G1 数组直采+拼接复算零差]")
A("")
A(md_table(ts))
A("")

A("## 4. 均分桥族（Eq.13 形，盲解参考 + MrUni 身份检查）")
A("")
A("$$m_j=\\min\\!\\big(1,\\tfrac{{j-23}}{{N'}}\\big)\\qquad(\\lambda_j\\equiv4^{{1/N'}}\\ \\text{桥内平坦})$$")
A("")
A("注：均分族桥内洞为**平坦平台**（各 $\\varepsilon\\equiv1/N'$），汇总表 max洞位置列在平台内由 fp 舍入决定（$\\pm$最后一 ulp），无位置含义。")
A("")
A("D4 §2 [已验证=代数]：位置无关洞目标下此族为盲 min-max 解；其 $N'=17$ 成员即已测败点 MrUni，$N'=14$ 为盲打平点但违 Smooth 级 mid 天花板——该族在此列出是为**否证字面目标函数**并给出与径向族的对照（$N'=17$：均分 1.3464 平坦洞 vs 径向 1.448 末洞；$N'=14$：洞 .0714 平摊 vs mid δ .1238 超限）。")
A("")
for t in [x for x in TABLES if x["family"] == "uniform"]:
    mt = t["met"]
    A(f"### 4.{[x['id'] for x in TABLES if x['family']=='uniform'].index(t['id'])+1} `{t['id']}` — {t['name']}")
    A("")
    A("**身份关系**：" + IDENTITY[t["id"]][0])
    A("")
    A(f"$j^*$={mt['j_star']}，$m_{{28}}$={fmt(mt['m28'],4)}，桥洞平坦 $\\rho={fmt(mt['band_hole'],4)}$，$F_{{\\rm arc}}$={fmt(mt['F_arc'],4)}，$BL$={fmt(mt['BL'],4)}，$\\Sigma m$={fmt(mt['sum_m'],3)}，$B$={fmt(mt['B_minus24'],3)}。[已验证=CPU 复算]")
    A("")
    A(md_table(t))
    A("")

A("## 5. 校验、复算与合规")
A("")
A("### 5.1 notch 变体注")
A("")
A("本交付的 notch 取 **D4 §5 字面量**：$m_{28}$ 固定回拨 $0.032680$ m 单位（=0.0453 nats，s28_less 实测对 MrPro 的同量回拨 [已验证]）。若按 s28_less 的**构造式**字面移植（\"$m_{28}:=m_{27}$，完全抹平 $\\delta_{27}$\"），则 ramp16 的 $m_{28}=0.073529$（回拨 0.050970 nats）、ramp15 的 $m_{28}=0.083333$（回拨 0.057762 nats），其余 63 槽不变。两读数仅在槽 28 一个数上分歧；0446 判决后可由作者定夺（D4 §6.3 未钉死 notch 的读数）。")
A("")
A("### 5.2 自动校验结果（生成器 `checks` 节全量入 JSON）")
A("")
A("| 校验 | 观测 | 期望 | 判 |")
A("|---|---|---|---|")
for c in checks:
    g = c["got"] if isinstance(c["got"], str) else (f"{c['got']:.6g}" if isinstance(c["got"], float) else c["got"])
    e = f"{c['expected']:.6g}" if isinstance(c["expected"], float) else c["expected"]
    A(f"| {c['check']} | {g} | {e} (±{c['tol']}) | {c['verdict']}{'；'+c['note'] if c['note'] else ''} |")
A("")
A("全部 MATCH [已验证]；13 张表的 I1/I2/单调/$m\\in[0,1]$/守恒残差断言在生成器内为 hard assert，任一失败脚本不产出文件。")
A("")
A("### 5.3 否决清单合规（SYNTHESIS §2/§6/§7）")
A("")
A("- 无未执行臂分数：0446/0448/0449 及全部新构造表 score=null；引用数字全部可指到 G1/GR/BUDGET/D 系列文件行号。")
A("- 无\"结构指标⇒能力\"跳接：洞/$B$/F_arc/BL 仅作**被测量**与**已预注册预测的输入**呈现；\"哪张赢\"一律指向冻结队列判决 [GR §2 依据 SYNTHESIS C5]。")
A("- YaRN 身份：本文不涉及 YaRN 表重建；径向族身份按 digest_mrrope-evq §5.1（MrPro ≡ MrRoPE Eq.14+Eq.16 同表 [已验证-独立CPU]）。")
A("- 端点不变量按 I1/I2 硬约束执行（任务描述\"从哪搬到哪\"的合法域 = 17 个桥 gap 内部重排 [D4 §5]）。")
A("")
A("## 6. 部署备注")
A("")
A(f"- 三张已排队表（`stack_0446`/`ramp_N16`/`ramp_N15`）的部署参数即其冻结公式；与本 JSON 的 `m_j` 数组差 <1e-7（fp32 反演噪声），队列回执对账以 sha256(ν fp32) 为准。官方 gain $a={OFFICIAL_GAIN:.9f}$ 全族统一。")
A("- `ramp16_notch`/`ramp15_notch`/`ramp_N13`/`ramp_N14` 若入队，须走 D4 §6.3 预注册判据分支（notch 两表以 0446 可加性判决为前提；13/14 以 0449 族下探信号为前提且建议加装 mid 缺口），未判决前不得冠\"改进方法\"名义。")
A("- 每槽数值列：$r_j$ 为原生圈数（m 不变量）；部署 128K 内圈数请读 $r_j\\cdot4^{1-m_j}$（= $\\nu_jL/2\\pi$，JSON 内 `rL_j` 已给）。末行 $\\lambda,\\rho$ 为 — （gap 需两槽定义；j=63 是梯子顶端）。")
A("")

open(os.path.join(HERE, "CANDIDATE_TABLES.md"), "w").write("\n".join(MD) + "\n")

# ---------- CSV ----------
with open(os.path.join(HERE, "CANDIDATE_TABLES.csv"), "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["table_id", "family", "N_prime", "j", "m_j", "nu_j", "T_j", "D_j", "r_j_native", "rL_j_deployed_128K", "eps_gap_next", "lambda_gap_next", "rho_gap_next"])
    for t in TABLES:
        for s in t["slots"]:
            rL = s["nu"] * L / (2 * math.pi)
            wcsv.writerow([t["id"], t["family"], t["Np"] if t["Np"] is not None else "", s["j"],
                           repr(s["m"]), repr(s["nu"]), repr(s["T"]), repr(s["D"]), repr(s["r"]), repr(rL),
                           repr(s["eps"]) if s["eps"] is not None else "",
                           repr(s["lam"]) if s["lam"] is not None else "",
                           repr(s["rho"]) if s["rho"] is not None else ""])

# ---------- JSON ----------
out = {
  "meta": {
    "task": "T1 候选频率表（径向族 N'13-17 / 0446 Stack / notch 变体 / 均分族参考）",
    "date": "2026-09-10", "generated_by": os.path.basename(__file__),
    "rule_source": ["answers/D4_transport_rule.md §2,§4,§5,§6", "BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md §3-4", "GROUND_README.md §6"],
    "constants": C,
    "column_definitions": {
      "m_j": "log_4(omega_j/nu_j)", "nu_j": "omega_j*4^-m_j", "T_j": "2pi/nu_j",
      "D_j": "W*4^m_j (识别地平线)", "r_j_native": "omega_j*W/2pi (原生窗内圈数, m 不变量)",
      "rL_j_deployed_128K": "nu_j*L/2pi", "eps_gap_next": "m_{j+1}-m_j",
      "lambda_gap_next": "4^eps", "rho_gap_next": "T_{j+1}/T_j = rho_nat*4^eps; 第j行=gap j(0-based, j->j+1)的洞比"
    },
    "conventions": {
      "j_star": "min{j>=24: m_j=1} 严格完成阈", "bridge_gaps": "g=23..39 (17 gap)",
      "I1": "m_j=0, j<=23", "I2": "m_j=1, j>=40",
      "conservation": "sum_{23..39}(gap_g - native) = ln S 每表 hard-assert"
    },
    "terminology_note": "任务'均分桥N'=17≈MrPro'锚点实指径向族(Eq.14)；字面均分族(Eq.13)N'=17=MrUni(64.5833/73.3333已测)。两族全交付。",
    "errata": [
      "0446 Stack max洞位置 GR §6 标 @g35，复算 @g38（值 1.4930 正确；LBS 同洞同位；s28 分量不新增洞结论不变）",
      "BUDGET 旧版 :45–47（现 48–50）max洞 1.86/1.76/1.80× 作废，用 GR §6 重算值 1.4930/1.4608/1.4757",
      "UNIFIED:41 MrPro m36–39=0.51–0.89 为槽位错位；本文件用 0.5948–0.8889",
      "G1 StackFrontBack 数组 m28−m27=−3.8e-9 为 fp32 反演噪声（s28 构造两槽相等）；交付表取 m28:=m27，其余逐位采 G1"
    ],
    "score_policy": "未执行臂一律无分数（G1 evidence_policy / SYNTHESIS §7）；预测字段为冻结队列预注册判据的引用，非结果",
    "official_gain": OFFICIAL_GAIN,
  },
  "tables": [],
  "checks": checks,
  "g1_identity": {
    "MrPro": {"sha_nu_deployed": M["MrPro"]["formula_vs_deployed"]["sha256_deployed"], "panel_scores": M["MrPro"]["panel_scores"], "bit_exact": True},
    "MrUni": {"sha_nu_deployed": M["MrUni"]["formula_vs_deployed"]["sha256_deployed"], "panel_scores": M["MrUni"]["panel_scores"], "bit_exact": True},
    "E1_s28_less_panel": M["E1_s28_less"]["panel_scores"],
    "LongBridgeSlower_panel": M["LongBridgeSlower"]["panel_scores"],
    "FullLagP2_Transfer3B_panel": M["FullLagP2_Transfer3B"]["panel_scores"],
    "queue_status": {"0446": "未执行", "0448": "未执行", "0449": "未执行", "0450": "未执行(E1 holdout 确认)", "0451": "未执行(P2 长端确认)"},
  },
}
for t in TABLES:
    mt = t["met"]; tr = transport(t["m"])
    sl = []
    for s in t["slots"]:
        d = dict(s); d["rL"] = s["nu"] * L / (2 * math.pi)
        sl.append(d)
    out["tables"].append({
      "id": t["id"], "name": t["name"], "family": t["family"], "N_prime": t["Np"],
      "construction": ("m_j = q(q+1)/%s, q=clip(j-23,0,N)" % (t["Np"] * (t["Np"] + 1))) if t["family"] == "ramp"
        else ("m_j = min(1,(j-23)/%s)" % t["Np"]) if t["family"] == "uniform"
        else ("MrPro ⊕ m28:=s28_less(0.065359) ⊕ m36-39:=LBS" if t["id"] == "stack_0446"
              else "ramp%d + m28 -= 10/306 (=0.0453 nats, s28_less 实测回拨量)" % t["Np"]),
      "identity_relation": IDENTITY[t["id"]][0], "queue_status": IDENTITY[t["id"]][1],
      "predictions_preregistered": PRED.get(t["id"], []),
      "metrics": {k: v for k, v in mt.items()},
      "transport_vs_MrPro": tr,
      "panel_scores": (M["MrPro"]["panel_scores"] if t["id"] == "ramp_N17" else
                M["MrUni"]["panel_scores"] if t["id"] == "uniform_N17" else None),
      "m_j": t["m"], "slots": sl,
    })
json.dump(out, open(os.path.join(HERE, "CANDIDATE_TABLES.json"), "w"), ensure_ascii=False, indent=1)

# ---------- 控制台摘要 ----------
print("== 汇总表 ==")
print(SUMMARY_HDR)
for t in TABLES:
    print(met_row(t))
print()
print("== checks ==")
for c in checks:
    print(f"{c['verdict']:8s} {c['check']}: got={c['got']:.6g} exp={c['expected']:g}")
bad = [c for c in checks if c["verdict"] != "MATCH"]
print("MISMATCH:", len(bad))
print("WROTE:", [os.path.join(HERE, x) for x in ("CANDIDATE_TABLES.md", "CANDIDATE_TABLES.csv", "CANDIDATE_TABLES.json")])
