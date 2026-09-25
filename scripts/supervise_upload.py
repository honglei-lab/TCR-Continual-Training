"""Bounded background upload retries; success requires the full remote hash audit."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, write_json


def active_upload(pid):
    proc = Path(f"/proc/{pid}")
    try:
        command = (proc / "cmdline").read_bytes()
        return b"scripts/modelscope_upload.py" in command and b"Z (zombie)" not in (proc / "status").read_bytes()
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait-pid", type=int)
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()
    if args.wait_pid:
        print(f"Waiting for existing transfer PID {args.wait_pid}; no duplicate transfer", flush=True)
        while active_upload(args.wait_pid):
            time.sleep(30)
    for attempt in range(1, 4):
        if (ROOT / "local/upload_verified.json").exists():
            break
        print(f"Upload/verification attempt {attempt}/3", flush=True)
        result = subprocess.run([sys.executable, "-u", str(ROOT / "scripts/modelscope_upload.py"),
                                 "--execute", "--workers", str(args.workers)])
        if result.returncode == 0:
            break
        if attempt < 3:
            time.sleep(30)
    complete = (ROOT / "local/upload_verified.json").exists()
    write_json(ROOT / "local/upload_supervisor_result.json", {"complete": complete})
    if not complete:
        raise SystemExit("Upload remains incomplete after three retries; see log")
    print("Upload complete and verified.", flush=True)
