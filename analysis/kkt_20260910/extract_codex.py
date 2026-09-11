#!/usr/bin/env python3
"""Read-only extractor for the codex rollout transcript (thread 01a08151).

Usage:
  python3 extract_codex.py <view> [args] [--out FILE] [--max-chars N] [--grep PATTERN]

Views:
  manifest              counts/sizes by record type, and per-view output sizes
  user                  every user message, chronological, with line numbers
  assistant             every visible agent message (AgentMessage), chronological
  reasoning             non-empty reasoning summary/raw text
  tools                 every CommandExecution: command + truncated output
  subagents             SubAgentActivity items (the multi-agent marshalling)
  filechanges           paths touched (FileChange) — what the thread wrote to disk
  compaction <n>        readable render of the n-th context compaction (1..5)
  all                   user + assistant + reasoning + subagents (no tool noise)

NEVER writes to ~/.codex. Output goes to stdout or --out.
"""
import json, sys, re

PATH = "/Users/yang/.codex/sessions/2026/09/08/rollout-2026-09-08T09-59-44-01a08151-4eb4-7572-8865-b321956dda94.jsonl"

theirs = sys.argv[1:]
out_file = None
max_chars = 4000
grep = None
args = []
i = 0
while i < len(theirs):
    a = theirs[i]
    if a == "--out":
        i += 1; out_file = theirs[i]
    elif a == "--max-chars":
        i += 1; max_chars = int(theirs[i])
    elif a == "--grep":
        i += 1; grep = theirs[i]
    else:
        args.append(a)
    i += 1
view = args[0] if args else "manifest"


def text_of(content):
    out = []
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict):
                t = c.get("text") or c.get("summary_text") or ""
                if isinstance(t, list):
                    t = " ".join(x.get("text", "") for x in t if isinstance(x, dict))
                out.append(str(t))
            else:
                out.append(str(c))
    return "\n".join(x for x in out if x)


def clip(s, n=None):
    n = n or max_chars
    s = s if isinstance(s, str) else str(s)
    return s if len(s) <= n else s[:n] + f"\n...[clipped {len(s)-n} chars]"


def records():
    with open(PATH) as f:
        for ln, line in enumerate(f):
            try:
                yield ln, json.loads(line)
            except Exception:
                continue


def line(label, ln, txt):
    return f"\n===== [{label}] line {ln} =====\n{txt}"


def render(view, args):
    buf = []
    if view == "manifest":
        import collections
        size = collections.Counter(); cnt = collections.Counter()
        for ln, d in records():
            t = d.get("type"); pl = d.get("payload")
            pt = pl.get("type") if isinstance(pl, dict) else None
            k = (t, pt) if t in ("response_item", "event_msg") else (t, None)
            size[k] += len(json.dumps(pl, ensure_ascii=False)); cnt[k] += 1
        buf.append("record type | MB | count")
        for k, v in size.most_common(30):
            buf.append(f"{str(k):58s} {v/1e6:7.2f}MB  n={cnt[k]}")
        return "\n".join(buf)

    if view == "compaction":
        n = int(args[1]) if len(args) > 1 else 1
        seen = 0
        for ln, d in records():
            if d.get("type") != "compacted":
                continue
            seen += 1
            if seen != n:
                continue
            pl = d.get("payload") or {}
            buf.append(f"# compaction #{n} @line {ln} window={pl.get('window_number')}")
            for key in ("message", "replacement_history", "guardian_history"):
                v = pl.get(key)
                if not v:
                    continue
                if isinstance(v, list):
                    for m in v:
                        if not isinstance(m, dict):
                            continue
                        role = m.get("role") or m.get("type")
                        txt = text_of(m.get("content") or "")
                        if len(txt) > 6000 and "app-context" in txt[:2000]:
                            continue  # boilerplate
                        if txt.strip():
                            buf.append(line(f"compact#{n}:{key}:{role}", ln, clip(txt, 6000)))
                else:
                    buf.append(line(f"compact#{n}:{key}", ln, clip(v, 8000)))
            return "\n".join(buf)

    for ln, d in records():
        t = d.get("type"); pl = d.get("payload") or {}
        pt = pl.get("type") if isinstance(pl, dict) else None
        if not isinstance(pl, dict):
            continue
        kind = None; txt = None
        if t == "event_msg" and pt == "item_completed":
            it = pl.get("item") or {}
            itt = it.get("type") if isinstance(it, dict) else None
            if itt == "UserMessage" and view in ("user", "all"):
                kind = "USER"; txt = text_of(it.get("content"))
            elif itt == "AgentMessage" and view in ("assistant", "all"):
                kind = "AGENT"; txt = text_of(it.get("content"))
            elif itt == "Reasoning" and view in ("reasoning", "all"):
                s = it.get("summary_text") or it.get("raw_content") or []
                txt = text_of(s) if not isinstance(s, str) else s
                if txt and len(txt.strip()) > 40:
                    kind = "REASONING"
            elif itt == "CommandExecution" and view == "tools":
                cmd = it.get("command")
                cmd = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
                o = it.get("aggregated_output") or it.get("stdout") or ""
                kind = "CMD"; txt = f"$ {cmd}\n[exit {it.get('exit_code')}] {o}"
            elif itt == "SubAgentActivity" and view in ("subagents", "all"):
                kind = "SUBAGENT"; txt = json.dumps(it, ensure_ascii=False)
            elif itt == "FileChange" and view == "filechanges":
                ch = it.get("changes") or {}
                kind = "FILECHANGE"; txt = json.dumps({k: v.get("type") for k, v in ch.items()}, ensure_ascii=False, indent=1)
        elif t == "response_item" and pt == "message" and view in ("user", "assistant"):
            role = pl.get("role")
            if role in ("user", "assistant"):
                txt = text_of(pl.get("content"))
                kind = "USER" if role == "user" else "AGENT"
        elif t == "response_item" and pt == "reasoning" and view in ("reasoning", "all"):
            s = pl.get("summary") or []
            txt = text_of(s)
            if txt and len(txt.strip()) > 40:
                kind = "REASONING"
        if not kind or not txt or not txt.strip():
            continue
        if grep and not re.search(grep, txt):
            continue
        buf.append(line(kind, ln, clip(txt)))
    return "\n".join(buf)


res = render(view, args)
if out_file:
    with open(out_file, "w") as f:
        f.write(res)
    print(f"wrote {out_file}: {len(res)/1e6:.2f}MB  {res.count(chr(10))} lines")
else:
    print(res)
