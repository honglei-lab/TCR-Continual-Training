"""Start the uploader detached with a private log; no token in argv or the log."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import ROOT, write_json


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--wait-pid", type=int, help="Wait for an existing owned transfer before verification/retry")
    args = p.parse_args()
    pidfile = ROOT / "local/upload_process.json"
    if pidfile.exists() and args.wait_pid is None:
        from common import read_json
        previous = read_json(pidfile)["pid"]
        try:
            os.kill(previous, 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit(f"Previous upload PID {previous} is still alive; inspect it first")
    log = ROOT / "local/modelscope-upload.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "a") as stream:
        command = [sys.executable, "-u", str(ROOT / "scripts/supervise_upload.py"), "--workers", str(args.workers)]
        if args.wait_pid:
            command += ["--wait-pid", str(args.wait_pid)]
        proc = subprocess.Popen(command,
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
    write_json(pidfile, {"pid": proc.pid, "log": str(log), "workers": args.workers})
    print(f"Upload launched: PID {proc.pid}; log: {log}")
