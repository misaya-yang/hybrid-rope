#!/usr/bin/env python3
"""
Read-only auditor for currently running experiments.

This script never sends signals and never mutates running jobs. It only:
1) reads process metadata,
2) infers likely log/output paths,
3) writes a Markdown snapshot report.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence


KEYWORDS = ("qwen", "longbench", "lora", "train", "eval")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(cmd: Sequence[str]) -> str:
    p = subprocess.run(list(cmd), capture_output=True, text=True, check=False)
    if p.returncode != 0:
        return ""
    return p.stdout


def run_shell(cmd: str, remote: Optional[Dict[str, str]]) -> str:
    if remote is None:
        return run_cmd(["bash", "-lc", cmd])
    password = remote.get("password", "")
    user = remote["user"]
    host = remote["host"]
    port = remote["port"]
    return run_cmd(
        [
            "sshpass",
            "-p",
            password,
            "ssh",
            "-p",
            port,
            "-o",
            "StrictHostKeyChecking=no",
            f"{user}@{host}",
            cmd,
        ]
    )


def parse_proc_lines(text: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        pid, ppid, etimes, cmd = parts
        low = cmd.lower()
        if ".py" not in low:
            continue
        # Skip unrelated Python workers (e.g., inductor compile workers) to keep audit fast.
        if "scripts/" not in low and not any(k in low for k in KEYWORDS):
            continue
        out.append(
            {
                "pid": pid,
                "ppid": ppid,
                "etimes": etimes,
                "cmd": cmd,
                "is_highlight": "yes" if any(k in low for k in KEYWORDS) else "no",
            }
        )
    return out


def extract_arg_value(cmdline: str, key: str) -> str:
    toks = shlex.split(cmdline)
    for idx, tok in enumerate(toks):
        if tok == key and idx + 1 < len(toks):
            return toks[idx + 1]
        if tok.startswith(key + "="):
            return tok.split("=", 1)[1]
    return ""


def infer_logs_and_outputs(proc: Dict[str, str], remote: Optional[Dict[str, str]]) -> Dict[str, str]:
    cmdline = proc["cmd"]
    output_json = extract_arg_value(cmdline, "--output_json")
    output_dir = extract_arg_value(cmdline, "--output_dir")
    data_dir = extract_arg_value(cmdline, "--data_dir")
    longbench_data = extract_arg_value(cmdline, "--longbench_local_data_dir")

    log_candidate = ""
    if output_json:
        p = Path(output_json)
        log_candidate = (p.parent / "eval_longbench.log").as_posix()
    elif output_dir:
        p = Path(output_dir)
        log_candidate = (p / "train.log").as_posix()

    log_tail = ""
    if log_candidate:
        tail = run_shell(f"tail -n 30 {shlex.quote(log_candidate)} 2>/dev/null || true", remote)
        log_tail = tail.strip()

    return {
        "output_json": output_json,
        "output_dir": output_dir,
        "dataset_hint": data_dir or longbench_data,
        "log_path": log_candidate,
        "log_tail": log_tail,
    }


def get_cwd_and_cmdline(pid: str, remote: Optional[Dict[str, str]]) -> Dict[str, str]:
    cwd = run_shell(f"readlink -f /proc/{pid}/cwd 2>/dev/null || true", remote).strip()
    cmdline = run_shell(f"cat /proc/{pid}/cmdline 2>/dev/null | tr '\\0' ' ' || true", remote).strip()
    return {"cwd": cwd, "cmdline": cmdline}


def build_report(
    rows: List[Dict[str, str]],
    remote: Optional[Dict[str, str]],
    report_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Attention Audit Report (Read-Only)")
    lines.append("")
    lines.append(f"- Generated at: `{now()}`")
    if remote is None:
        lines.append("- Target: `local`")
    else:
        lines.append(f"- Target: `{remote['user']}@{remote['host']}:{remote['port']}`")
    lines.append("")
    lines.append("## Running Python Scripts")
    lines.append("")

    if not rows:
        lines.append("- No matching python `.py` process found.")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for row in rows:
        pid = row["pid"]
        meta = get_cwd_and_cmdline(pid, remote)
        inferred = infer_logs_and_outputs({**row, **meta}, remote)
        lines.append(f"### PID `{pid}`")
        lines.append(f"- Highlighted: `{row['is_highlight']}`")
        lines.append(f"- PPID: `{row['ppid']}`")
        lines.append(f"- Elapsed (s): `{row['etimes']}`")
        lines.append(f"- Working dir: `{meta['cwd']}`")
        lines.append(f"- Script cmd: `{meta['cmdline'] or row['cmd']}`")
        lines.append(f"- Output JSON: `{inferred['output_json']}`")
        lines.append(f"- Output dir: `{inferred['output_dir']}`")
        lines.append(f"- Dataset hint: `{inferred['dataset_hint']}`")
        lines.append(f"- Log file: `{inferred['log_path']}`")
        lines.append("- Last 30 log lines:")
        lines.append("```text")
        lines.append(inferred["log_tail"] or "(no readable log tail)")
        lines.append("```")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only process audit.")
    ap.add_argument("--remote_host", type=str, default="")
    ap.add_argument("--remote_port", type=str, default="52592")
    ap.add_argument("--remote_user", type=str, default="root")
    ap.add_argument("--remote_password", type=str, default="")
    ap.add_argument("--report_path", type=Path, default=Path("attn_audit_report.md"))
    args = ap.parse_args()

    remote: Optional[Dict[str, str]] = None
    if args.remote_host.strip():
        if not args.remote_password:
            raise RuntimeError("remote mode requires --remote_password")
        remote = {
            "host": args.remote_host.strip(),
            "port": args.remote_port.strip(),
            "user": args.remote_user.strip(),
            "password": args.remote_password,
        }

    proc_out = run_shell(
        "ps -eo pid,ppid,etimes,args | grep -E 'python .*\\.py' | grep -v grep || true",
        remote,
    )
    rows = parse_proc_lines(proc_out)
    rows = sorted(rows, key=lambda x: int(x["pid"]))
    build_report(rows, remote, args.report_path)

    print(f"Saved report: {args.report_path}")
    for row in rows:
        mark = "*" if row["is_highlight"] == "yes" else "-"
        print(f"{mark} pid={row['pid']} elapsed={row['etimes']} cmd={row['cmd'][:180]}")


if __name__ == "__main__":
    main()
