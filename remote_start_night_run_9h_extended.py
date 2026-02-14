#!/usr/bin/env python3
"""
Start the 9h extended run on the remote R6000 box (non-interactive).

Requires:
  - $env:SEETACLOUD_SSH_PW set
"""

from __future__ import annotations

from seetacloud_plink import run


def main() -> None:
    remote_out = "/root/autodl-tmp/dfrope/hybrid-rope/results/night_run_9h_extended"
    remote_script = "/root/autodl-tmp/dfrope/hybrid-rope/scripts/run_night_run_9h_extended.py"
    remote_log = f"{remote_out}/run.log"

    cmd = (
        f"mkdir -p {remote_out} && "
        f"nohup python3 {remote_script} > {remote_log} 2>&1 & "
        f"echo $! > {remote_out}/pid.txt && "
        f"echo started_pid=$(cat {remote_out}/pid.txt)"
    )
    cp = run(cmd, check=True)
    print((cp.stdout or "").strip())
    print(f"[log]  {remote_log}")
    print(f"[pid]  {remote_out}/pid.txt")


if __name__ == "__main__":
    main()

