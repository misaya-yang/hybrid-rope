# Runbook — boot day

Operational only. The theory, the protocol and the failure attribution are in
`README.md`; this file is what to type, in what order, and what to expect.

## State as of 2026-09-10 (no-card mode)

Deployed and verified on the rented host (`ssh -p <port> root@connect.<zone>.seetacloud.com`):

> **The endpoint is not stable — never hardcode it in a committed file.** The
> host and port have changed repeatedly across this programme (`weste:36549`,
> `westd:42013/41749/26252/35068`, `westb:43850`, `westc:27741`, `westc:57109`).
> On 2026-09-09 the user's own answer to "which host" was a *different port on
> the same zone* than the one that was live the week before. Take the current
> endpoint from the user's most recent message at boot time, and confirm with
> `nvidia-smi -L` that a card is attached before planning anything around it.

```
/root/autodl-tmp/nongeometric_screen_20260909/code/     <- the working tree; run everything from here
  experiments/curvature_20260910/                       <- this package
  experiments/nongeometric_screen/                      <- the frozen-weight harness
  analysis/unify_20260910/tables/ground_truth_tables.json
```

Stage 0 passes 16/16 on the server. The measured facts it confirmed:

| | |
|---|---|
| deployed-table reconstruction | native 8.2e-8, MrPro 8.4e-8, YaRN 1.1e-7 (max rel. diff — float32 rounding) |
| checkpoint | `Qwen2ForCausalLM`, θ=10⁶, head_dim 128 → 64 slots, native window 32768 |
| `install()` under transformers 5.15.1 | verified end to end on a config-derived tiny model: `Qwen2RotaryEmbedding`, `inv_freq` is a buffer of 64, `attention_scaling`/`original_inv_freq` present, forward finite |
| rope config | θ lives in `rope_parameters` (5.x), with a `rope_scaling` alias; `rope_theta` at top level is **None** — the loader handles it, a hand-read of `config.json` would not |
| probe corpus | `long_inputs/pg19_test_37702.npy`, 131073 tokens, **held out** from the panel's documents |
| panel documents | `proofpile` 001364/001901, `pg19` 28988/30312 |
| env | python 3.12.3 at `/root/miniconda3/bin/python`, transformers 5.15.1, torch 2.8.0+cu128, no flash_attn |
| disk | 20.6 GiB free on `/root/autodl-tmp` |

**Nothing has been run on a GPU.** Every number in `README.md` is a projection
from the host's own archived timings, not a measurement.

## The queue is frozen, and it is not this package's queue

There is a pre-registered queue already committed to the harness, and it runs
first. From `UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md:143`:

```
0446 StackFrontBack -> 0448 MrProN16 -> 0449 MrProN15
  -> 0450 E1 holdout16 -> 0451 P2 long-end confirmation
```

**These five are the Core-A/B/C verification set, and 0450/0451 are the
holdout** — `NEXT_DERIVATION_KKT_PROBLEM.md:113` pins them ("holdout (0450/0451)
前 F 不得声称逐行预测力"). They are roughly 6–7 hours of GPU on their own.

Two consequences for boot day:

1. **`QUEUE_ID` must not be 0450** (it was, in the first version of this
   package — that would have spent the holdout's namespace on a job from the
   development panel). The default is now **`0500`**, which sorts after all
   five, so if the frozen queue is still in `queue/` when the card comes back,
   those five run first and this package follows.
2. **If the session is short, the frozen five win.** They are already
   committed, they are the higher-priority use of the card, and this package is
   not. Run `driver.sh all` only with enough wall clock left for both — or
   accept that the panel job (s5) will sit in `queue/` behind them and be read
   next session. The core chain s0–s4 is still worth its 90 minutes on its own:
   it produces `G`, the residual spread, and the replay gate, and none of those
   depend on the panel.

Check before queueing anything:

```bash
ls $HARNESS/queue/ | sort | tail
ls $HARNESS/done/  | sort | tail
```

## Before the GPU boots

One thing to have decided, because it is pre-registered and must not be chosen
after seeing the curve:

**`EPS`** — the native output-KL budget, nats/token. Default `1e-3`.

`s1b` measures `cost(MrRoPE) = D_N(MrRoPE − native)` and prints it next to
`EPS`. If `EPS` is far below it — and it will be, by a large factor — then `s3`
solves inside a ball that does not contain MrRoPE, and the result is
*"the best direction at this budget"*, **not** *"MrRoPE is optimal"*. That is
fine and is arguably the more useful question, but the receipt must say which
one it is. For the KKT/EXPLAINS reading, re-solve at `--eps cost(MrRoPE)` and
report both. Do not change `EPS` itself to make a branch appear.

## Boot sequence

```bash
ssh -p 27741 root@connect.westc.seetacloud.com
cd /root/autodl-tmp/nongeometric_screen_20260909/code
nvidia-smi -L                       # confirm the card is actually attached
```

Then the pre-flight again, on the GPU now — it re-checks everything and is free:

```bash
bash experiments/curvature_20260910/driver.sh s0
```

Only if that is 16/16, run the chain. `s0`–`s4` is the core: **under 90 minutes**.

```bash
nohup bash experiments/curvature_20260910/driver.sh all \
  > runs_$(date -u +%Y%m%d_%H%M).log 2>&1 &
tail -f runs_*.log
```

Run it under `nohup`: the chain is long enough that an ssh drop should not kill
it, and each stage skips itself if its output already parses, so an interrupted
chain resumes for free.

`driver.sh all` runs `s0 → s2c → s1 → s2 → s3 → s4 → (s5)`. Note the order:
the in-window gradient (`s2c`, 5 min at 32K) goes before the anything-expensive
stages, so a problem with the objective side surfaces while the session is
still cheap.

### What each stage should print, and what to do if it does not

**`s1`** — 64 one-sided + 8 pair forwards at 32K, fp32, no backward. Watch for
negative `fisher_diag` entries. Up to a few are expected and are handled: an
unpriced slot is **pinned, not freed**, and the driver says so. If it is most of
the table, the fp32 default did not take — check `"dtype":"fp32"` in the receipt
and re-run. OOM here is unlikely (4.1 s/forward at 32K, 21–27 GiB peak) but the
fallback is `--dtype bf16` and a note in the receipt.

**`s1b`** — free, no forwards: it reads the Fisher `s1` just measured plus the
panel's own published tables, and prints three things. The **veto test** on
`E1_s28_less` (a measured pure Pareto point: +5.21 pp at 128K for no 32K cost).
If it lands **inside** the budget and `s3` later returns `G ≈ 0`, the linear
model has missed a step the panel can see and the solve is falsified — by a
number measured before this package existed. Then the **rank check** of `D_N`
against the panel's own 32K column across every scored arm, which is the only
free validation the constraint model gets. Then `cost(MrROPE)`, the calibrated
budget (see above). If the rank check comes back at ρ < 0.3, stop and fix the
metric: a price that cannot order these arms cannot carry a solve.

**`s2c` / `s2a` / `s2b`** — forward-only central differences. Each slot prints a
`g` and a `curvature` line. The curvature field is the local model's own error
bar: if `|curvature| * eps` is comparable to `|g|`, the linear term does not
reach across the step and `s4`'s trust ratio will say so. That is information,
not a crash.

**`s3`** — seconds, CPU. It prints the whole answer:

```
G = g^T P_F g = <number>   ->  EXPLAINS (no payable direction)
                            or  CAN BEAT (payable direction found)
```

**This is the fork.** Read it before reading anything else.

**`s4`** — real forwards on the step, three budget-matched controls, and the
gate. `rho_trust` must land in [0.5, 1.5] and the native-KL exponent in
[1.7, 2.3]. On failure the driver prints the three-branch diagnosis and stops.
**Do not proceed past a failed `s4`.** That stopping rule is pre-registered.

## After the chain — the three branches

### Branch A: `G ≈ 0` (EXPLAINS)

`driver.sh all` exits 0 and skips both `s4` and `s5`. There is no table to
queue, by design: the long-range gradient has no component the native metric
can pay for, so MrRoPE is a KKT point of the stated problem.

What to do: **write it up.** The arithmetic in `README.md` (bridge slope
`ln4·(q+1)/153` = 0.0090608; `N=17` is both the YaRN-width match and the
minimum-budget member; `sum m` 29.333 vs YaRN's 30.104) is the explanation, and
`G ≈ 0` is its measured confirmation. Do not force a step to have something to
run. `s6` and `--no-pin-ends` are the natural follow-ups — they test whether the
result is an artifact of the constraint set rather than of the model.

### Branch B: `G > 0` and the gate passes

`s5` queues **two** jobs. Then start the harness worker:

```bash
cd /root/autodl-tmp/nongeometric_screen_20260909/code
export PATH=/root/miniconda3/bin:$PATH      # a bare `python` is NOT on the login PATH
nohup python -m experiments.nongeometric_screen.worker \
  --root /root/autodl-tmp/nongeometric_screen_20260909 \
  --history /root/autodl-tmp/bm_transfer_20260908 \
  > harness_$(date -u +%Y%m%d_%H%M).log 2>&1 &
```

(`driver.sh` exports that PATH itself, which is why the chain works without it;
a hand-typed `python` in a plain ssh shell fails with `command not found`.
Verified: `python` is absent from the login PATH, `/root/miniconda3/bin/python`
is 3.12.3 with numpy 2.3.2.)

The worker runs `sorted(queue/*.json)[0]` one at a time — the registration job
first, then the `long_eval` job (the naming guarantees the order; `panel_jobs.py`
refuses to write a pair that would run the other way). Progress lands in
`$HARNESS/live.json`; per-job output in `$HARNESS/results/<name>/`.

Read, in this order:

1. `results/<name>/summary.json` — RULER wins/losses against the archived MrPro
   rows, and `nll_by_length` deltas at 8192/16384/32768.
2. `$HARNESS/long_nll/<name>_summary.json` — the 64K/131072 cells, paired
   against MrPro measured in the same load. Compare against the archived
   baseline: pg19 131072 = **2.3077**, proofpile 131072 = **1.1030**,
   pg19 65536 = 2.5698, proofpile 65536 = 0.6222.

A candidate that beats MrPro on the long cells while staying inside the
in-window tolerance is the result the whole exercise is for. If it is flat,
read the `--long-loss gold` contingency in `README.md` before concluding
anything: the gradient was measured against tail-token NLL and the panel
measures retrieval, and those are different objectives.

### Branch C: `G > 0` and the gate fails

The chain exits 1. Work the three-branch diagnosis the driver printed, in
order — it distinguishes a constraint-side measurement problem from an
objective-side one from a baseline-identity problem. `README.md`'s attribution
table has the full mapping. Do not re-run the chain hoping for a different
number: change the thing the diagnosis names, and record which branch it was.

## Stopping, resuming, and not wasting the session

```bash
bash experiments/curvature_20260910/driver.sh status   # what has run
touch experiments/curvature_20260910/runs/STOP         # halt at the next stage boundary
```

`STOP` is also honoured by the harness worker (it checks `$HARNESS/STOP` between
rows). Note these are two different files in two different roots.

To force a stage to re-run, delete its output under `.../curvature_20260910/runs/`.
To force a harness job to re-run, delete its entry from `$HARNESS/done/`.

## Shutting the machine down is part of the run, not after it

This is a **standing end-of-run duty**, quoted from the user's own standing
instruction to this programme:

> 按照项目规范写好实验报告和相应文档，提交并推送代码，**同时检查 autoDL 是否成功关机**，
> 先服务器命令关机，不行的话，用 chrome control

and, for the case where a timer is involved:

> 我设置了 50 后关机，你搞一个**定时任务，也就是监控你的东西**，如果又没关机，你就直接
> 操控 chrome 自己进行关机

So "I started the chain under `nohup`" is not the end of the job. The order is:

**1. Confirm the work is actually finished, not merely un-attached.** The two
jobs must be in `$HARNESS/done/`, **not** in `queue/` and not half-written in
`results/`. A job that was mid-flight when the card went away resumes from its
own `.jsonl` rows — the worker skips rows it already has — but only if its queue
file is still there. Deleting `queue/` files to "tidy up" destroys the resume
path and wastes the GPU time already spent on them.

**2. Never power off a machine that is not ours.** The host may be rented per
session and shared; an earlier instance of this programme was told
「你不需要GPU验证，**它正在被其他实验使用**」. Powering off someone else's running
job is destructive and unrecoverable. Check `nvidia-smi` and the worker log
first: if there is work here that this package did not start, stop and ask.

**3. Shut down by command; if that fails, escalate — do not assume.**
A shutdown command returning success is not evidence the instance stopped
billing. Verify the instance actually reaches a stopped state in the AutoDL
console, not just that `ssh` stops answering.

**4. If a timer was set, install a backstop.** A timer the user sets is not a
guarantee; the standing instruction is to run a watchdog that re-checks and, if
the machine is still up, closes it out — by console control if the server-side
path did not work. **Before booting, ask whether a timer is set this session.**
If one is, the backstop is part of the same request, not a follow-up.

**What I have not confirmed** (asked, not yet answered): whether "successfully
shut down" means the AutoDL console shows the instance stopped, or something
weaker. Until that is settled, verify at the console — the conservative reading
costs a minute and the optimistic one costs a night of billing.

**Before that**, per the same standing instruction: write the report and the
docs, and commit and push the code. For this package that means the receipts —
see "What must be recorded" below — and resolving the sanitisation question in
the next section, which blocks any commit of `README.md`/`RUNBOOK.md`.

## GPU discipline — the rules this chain is built to respect

Three standing rules, quoted from the user's own instructions, and what each
one costs this chain:

**"GPU 也绝对不能空转超过 2min"** — the card must never sit idle.
The chain runs back-to-back under one `nohup`, so the only idle is the model
load at each stage boundary: `import transformers` alone is ~13 s and the
weights are ~6 GB from disk. Five GPU stages × one load each is the entire
overhead. The driver now times every boundary and prints a **WARNING if more
than 120 s** elapse between stages, so a gap caused by anything other than a
load is visible in `runs/driver.log` instead of passing silently. What the rule
targets — a card idle while something is being decided — does not happen here,
because the chain never waits on a human or an agent.

**"所用实验配置，要吃满 GPU 计算，最大化显存，最少超过 28gb"** — saturate the card.
This protocol is **inference-only and cannot saturate 28 GB**, and saying so
plainly is better than pretending otherwise. It is in tension with that rule
only if read as universal: the *zero-training* rule in the same corpus is what
makes it inference-only —

> 「你只需要零训练修改 checkpoint 然后按照 MrRoPE 论文中已经报过的测试去对比……这样最节省算力」

The two rules point the same way once the objective is fixed: no training means
no gradient state, no optimizer state, and a peak of 21–27 GiB measured on this
host's own archived rows. **If the requirement is that a run must exceed 28 GB,
this protocol is the wrong shape and the right move is to say so before booting,
not to pad the batch size to hit a number.** The natural 28 GB+ variant is a
fine-tuning arm (MrRoPE's own recipe), which is a different experiment with a
different cost.

**"后续最多 3 个实验，不能隐藏多候选扫描或重置原有预算"** — at most three, no hidden scan.
Accounting, so the count is explicit rather than implied:

| # | experiment | what it is |
|---|---|---|
| 1 | `driver.sh all` | **one** solved table vs MrRoPE. s0–s4 is the measurement chain for that single comparison; it is not three experiments, because s1/s2/s3 produce no result of their own — they are the gradient and the metric that define experiment 1's single step. |
| 2 | the panel run (`s5` + worker) | the downstream test of that one table. |
| 3 | **one** follow-up arm | exactly one of: `--no-pin-ends`, `--free-unpriced`, `--long-loss gold`, or `s6`. |

`s2b` (wide 128K) is a **coverage** decision inside experiment 1, not a fourth
experiment — it widens the slot set the same solve reads. The other arms in the
table below are **not** to be run as well; they are the menu from which
experiment 3 is chosen.

There is no candidate scan here by construction: `eps` is fixed on the command
line before the gradient is measured, `gain` is held at MrRoPE's value and never
co-optimised, and the solve is closed-form — there is no schedule, no seed, no
second hyper-parameter to sweep. The `--eps-sweep` output is labelled
descriptive-only for exactly this reason.

**Counterfactual value is required for each, and is written down.** Before
booting, each of the three has a stated value in both directions — success and
failure — in `README.md`'s gate table and contingencies. The two that look like
non-results are the most valuable ones here:

* **`G ≈ 0` is the explain branch** — MrRoPE is a KKT point of the stated
  problem, which is the unified account nothing in the literature currently has.
  It is a result, not a failed run.
* **the solver losing to the budget-matched controls is a finding** — it would
  mean budget alone decides the outcome and the interior profile is unreadable,
  i.e. every geometric family is a budget choice wearing a shape.

Neither gets re-run hoping to change.

## Server details do not go into committed files

`README.md` and `RUNBOOK.md` contain the ssh port, `/root/autodl-tmp/...` paths
and the harness layout. The repo's own `AGENTS.md` carries the rule —

> 「Never expose or commit author identity, credentials, server details, private
> paths, checkpoints, caches, or ignored raw evidence.」

— so **these two files must not be committed as they stand.** Nothing has been
committed or pushed. Before any commit, either sanitize them (replace the host,
port and absolute paths with placeholders; the code defaults already read from
env vars, so a sanitized doc stays accurate) or move them to an ignored
location. The code itself takes every path from `MODEL`/`NPY`/`BM`/`HARNESS`/
`TABLES` env vars, so nothing in `*.py` needs the real values.

## Optional stages, in order of what they buy

| stage | command | cost | what it adds |
|---|---|---|---|
| `s2b` wide 128K gradient | `driver.sh s2` (already in `all`) | 34 min | widens the slot set from 18 to 30 |
| `s6` MC Fisher | `driver.sh s6` | ~15 min, needs backward at 32K | the exact 64×64 PSD matrix; the only way to measure `step_diag_share` |
| free-ends arm | `solve_kkt --no-pin-ends` | seconds | the PI-like direction the geometric families never explore |
| aggressive arm | `solve_kkt --free-unpriced` | seconds | spends budget on slots this probe could not price — **label it as an arm, not the result** |
| budget curve | `solve_kkt --eps-sweep` | seconds | descriptive only; the operating point stays `--eps` |

## What must be recorded, whatever happens

Every stage writes a JSON receipt with its inputs, its configuration, its
timings and its peak memory. Do not hand-write results into a summary: quote
the receipts. In particular the negative and null outcomes are results here —
`G ≈ 0`, a failed gate with its diagnosis, a flat panel — and each one is
written up rather than re-run until it changes.

Two things must travel with any number reported out of this package:

* **`gain` was held at the harness's MrPro value and was not co-optimised.**
  A candidate that changed it is not comparable to the archived rows.
* **The panel is the historical development set.** Its own `summary.json` says
  so in `scope`, and it is not independent confirmation.
