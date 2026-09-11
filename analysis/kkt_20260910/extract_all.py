#!/usr/bin/env python3
"""Read-only extractor across ALL codex rollouts on disk.

Difference from extract_codex.py: that one hardcodes a single rollout path.
This one walks every rollout under ~/.codex/sessions and never writes there.

Usage:
  python3 extract_all.py user      [--out FILE] [--grep PAT] [--from DATE]
  python3 extract_all.py ops       [--out FILE]          # instruction-shaped user msgs
  python3 extract_all.py assistant [--out FILE] [--grep PAT]
  python3 extract_all.py index                            # rollout inventory
"""
import json, sys, os, re, glob

ROOT = os.path.expanduser("~/.codex/sessions")
theirs = sys.argv[1:]
out_file = None
grep = None
since = None
args = []
i = 0
while i < len(theirs):
    a = theirs[i]
    if a == "--out":
        i += 1; out_file = theirs[i]
    elif a == "--grep":
        i += 1; grep = theirs[i]
    elif a == "--from":
        i += 1; since = theirs[i]
    else:
        args.append(a)
    i += 1
view = args[0] if args else "index"


def rollouts():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*", "*", "*", "rollout-*.jsonl"))):
        out.append(p)
    return out


def records(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield n, json.loads(line)
            except json.JSONDecodeError:
                continue


def text_of(payload):
    parts = []
    for c in payload.get("content") or []:
        if isinstance(c, dict):
            t = c.get("text") or c.get("input_text") or c.get("output_text")
            if t:
                parts.append(t)
    return "\n".join(parts).strip()


def msgs(view_filter=None):
    for p in rollouts():
        rid = os.path.basename(p).replace("rollout-", "").replace(".jsonl", "")
        for n, r in records(p):
            if r.get("type") != "response_item":
                continue
            pl = r.get("payload") or {}
            if pl.get("type") != "message":
                continue
            role = pl.get("role")
            if view_filter and role != view_filter:
                continue
            if role not in ("user", "assistant"):
                continue
            t = text_of(pl)
            if not t:
                continue
            ts = r.get("timestamp", "")
            if since and ts < since:
                continue
            yield rid, n, ts, role, t


OPS = re.compile(
    r"(服务器|ssh|端口|显卡|GPU|A100|H100|5090|Blackwell|AutoDL|westc|跑|实验|训|训练|"
    r"关掉|关机|停|停止|别|不要|不许|禁止|只|先|再|然后|记得|注意|检查|汇报|报告|"
    r"预算|成本|钱|小时|分钟|数据|checkpoint|ckpt|路径|目录|md5|sha|"
    r"bm_transfer|ruler|RULER|passkey|NLL|eval|评测)",
    re.I,
)

buf = []
if view == "index":
    rows = []
    for p in rollouts():
        rid = os.path.basename(p).replace("rollout-", "").replace(".jsonl", "")
        nu = na = ntot = 0
        tmin = tmax = None
        for _, r in records(p):
            ntot += 1
            if r.get("type") == "response_item":
                pl = r.get("payload") or {}
                if pl.get("type") == "message":
                    if pl.get("role") == "user":
                        nu += 1
                    elif pl.get("role") == "assistant":
                        na += 1
            ts = r.get("timestamp")
            if ts:
                tmin = ts if tmin is None or ts < tmin else tmin
                tmax = ts if tmax is None or ts > tmax else tmax
        rows.append((rid, ntot, nu, na, tmin, tmax))
    buf.append(f"{'thread':40s} {'lines':>6} {'user':>5} {'asst':>5}  {'first':20s} {'last':20s}")
    for rid, ntot, nu, na, a, b in rows:
        buf.append(f"{rid:40s} {ntot:6d} {nu:5d} {na:5d}  {(a or '')[:19]:20s} {(b or '')[:19]:20s}")
    tu = sum(r[2] for r in rows)
    buf.append(f"\nTOTAL user messages across all rollouts: {tu}")
elif view in ("user", "assistant", "ops"):
    for rid, n, ts, role, t in msgs(view if view in ("user", "assistant") else "user"):
        if view == "ops" and not OPS.search(t):
            continue
        if grep and not re.search(grep, t, re.I):
            continue
        buf.append(f"\n{'='*100}\n[{rid[:13]}] line {n}  {ts}  ({role})\n{'-'*100}\n{t}")
else:
    sys.exit(f"unknown view {view}")

out = "\n".join(buf)
if out_file:
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(f"wrote {out_file}  ({len(out)} chars, {out.count(chr(10))} lines)")
else:
    print(out)
