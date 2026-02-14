#!/usr/bin/env python3
"""
Minimal PuTTY/plink helpers for this repo.

Security goals:
- Never hardcode passwords in repo files.
- Prefer `-pwfile` over `-pw` to avoid leaking secrets via process command lines.

Usage (PowerShell):
  $env:SEETACLOUD_SSH_PW = "..."
  python upload_extended.py
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional


PLINK = os.environ.get("SEETACLOUD_PLINK", r"C:\Users\Admin\.ssh\plink.exe")
HOST = os.environ.get("SEETACLOUD_HOST", "root@connect.bjb1.seetacloud.com")
PORT = int(os.environ.get("SEETACLOUD_PORT", "42581"))
HOSTKEY = os.environ.get("SEETACLOUD_HOSTKEY")  # e.g. "ssh-ed25519 255 SHA256:..."
PW_ENV = "SEETACLOUD_SSH_PW"


def _require_pw() -> str:
    pw = os.environ.get(PW_ENV)
    if not pw:
        raise RuntimeError(
            f"{PW_ENV} is not set.\n"
            f"- PowerShell: $env:{PW_ENV} = '...'\n"
            f"- CMD: set {PW_ENV}=..."
        )
    return pw


@contextmanager
def _pwfile(pw: str):
    # NOTE: keep this file short-lived and out of the repo tree.
    fd, path = tempfile.mkstemp(prefix="seetacloud_pw_", suffix=".txt")
    try:
        os.write(fd, pw.encode("utf-8"))
        os.close(fd)
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def run(
    remote_cmd: str,
    *,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    pw = _require_pw()
    if not Path(PLINK).exists():
        raise FileNotFoundError(f"plink.exe not found at: {PLINK}")
    with _pwfile(pw) as pwf:
        args = [
            PLINK,
            "-batch",
            "-ssh",
            "-P",
            str(PORT),
        ]
        if HOSTKEY:
            args.extend(["-hostkey", HOSTKEY])
        args.extend(
            [
                HOST,
            "-pwfile",
            pwf,
            remote_cmd,
            ]
        )
        return subprocess.run(
            args,
            check=check,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
        )


def upload_file_base64(local_path: str | Path, remote_path: str, *, chunk_size: int = 4000) -> None:
    lp = Path(local_path)
    data = lp.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")

    # Clear remote file first.
    run(f"rm -f {remote_path}")

    # Append chunks to avoid command-line length limits.
    for i in range(0, len(b64), chunk_size):
        chunk = b64[i : i + chunk_size]
        redir = ">" if i == 0 else ">>"
        # base64 alphabet is safe unquoted here (A-Z a-z 0-9 + / =)
        run(f"printf %s {chunk} | base64 -d {redir} {remote_path}")


def download_file_base64(remote_path: str, local_path: str | Path) -> None:
    cp = run(f"base64 -w0 {remote_path}")
    b64 = (cp.stdout or "").strip()
    if not b64:
        raise RuntimeError(f"Empty download for remote path: {remote_path}")
    data = base64.b64decode(b64)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    Path(local_path).write_bytes(data)


def download_dir_tar_gz_base64(remote_dir: str, local_tar_gz_path: str | Path) -> None:
    # Create a tarball in-stream and base64 encode it, to avoid requiring pscp/scp.
    remote_cmd = f"tar -czf - -C {remote_dir} . | base64 -w0"
    cp = run(remote_cmd)
    b64 = (cp.stdout or "").strip()
    if not b64:
        raise RuntimeError(f"Empty download for remote dir: {remote_dir}")
    data = base64.b64decode(b64)
    Path(local_tar_gz_path).parent.mkdir(parents=True, exist_ok=True)
    Path(local_tar_gz_path).write_bytes(data)
