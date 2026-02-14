#!/usr/bin/env python3
"""
Tail the remote log for night_run_9h_extended.

Requires:
  - $env:SEETACLOUD_SSH_PW set
"""

from __future__ import annotations

import sys

from seetacloud_plink import run


def main() -> None:
    n = 50
    if len(sys.argv) == 2:
        n = int(sys.argv[1])
    remote_log = "/root/autodl-tmp/dfrope/hybrid-rope/results/night_run_9h_extended/run.log"
    cp = run(f"tail -n {n} {remote_log}", check=False)
    print(cp.stdout or "")


if __name__ == "__main__":
    main()

