#!/usr/bin/env python3
"""SSH helper for the Shuyi training server.

  python experimental/training/ssh.py run "cmd"        # run a shell command, stream output
  python experimental/training/ssh.py runf script.sh   # run local script content
  python experimental/training/ssh.py put local remote [local remote ...]
  python experimental/training/ssh.py get remote local
"""
import os
import sys
import time

import paramiko

HOST = os.environ.get("SHUYI_HOST", "223.109.239.11")
PORT = int(os.environ.get("SHUYI_PORT", "25924"))
USER = os.environ.get("SHUYI_USER", "root")
PASS = os.environ.get("SHUYI_PASS", "Ewee4ohd")

LDPATH = ":".join(
    "/home/vipuser/miniconda3/lib/python3.12/site-packages/nvidia/" + p + "/lib"
    for p in ("cuda_cupti", "cuda_nvrtc", "cuda_runtime", "cublas", "cudnn", "cufft", "curand", "cusparse", "cusolver", "nvtx", "nvjitlink")
)


def connect(attempts=6):
    last = None
    for i in range(attempts):
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20,
                        banner_timeout=30, auth_timeout=30)
            return cli
        except Exception as exc:  # noqa: BLE001 - retry transient banner failures
            last = exc
            time.sleep(2 + 2 * i)
    raise last


def run(cmd):
    cli = connect()
    chan = cli.get_transport().open_session()
    chan.get_pty()
    chan.exec_command(cmd)
    while True:
        data = chan.recv(65536)
        if not data:
            break
        sys.stdout.write(data.decode("utf-8", "replace"))
        sys.stdout.flush()
    rc = chan.recv_exit_status()
    cli.close()
    return rc


def put(pairs):
    cli = connect()
    sftp = cli.open_sftp()
    for i in range(0, len(pairs), 2):
        local, remote = pairs[i], pairs[i + 1]
        d = os.path.dirname(remote)
        if d:
            try:
                sftp.stat(d)
            except OSError:
                run("mkdir -p " + d)
                sftp = cli.open_sftp()
        sftp.put(local, remote)
        print(f"put {local} -> {remote}")
    sftp.close()
    cli.close()


def get(pairs):
    cli = connect()
    sftp = cli.open_sftp()
    for i in range(0, len(pairs), 2):
        remote, local = pairs[i], pairs[i + 1]
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        sftp.get(remote, local)
        print(f"get {remote} -> {local}")
    sftp.close()
    cli.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    if mode == "run":
        return run("export LD_LIBRARY_PATH=" + LDPATH + "; " + sys.argv[2])
    if mode == "runf":
        return run("export LD_LIBRARY_PATH=" + LDPATH + "; " + open(sys.argv[2], encoding="utf-8").read())
    if mode == "put":
        put(sys.argv[2:])
        return 0
    if mode == "get":
        get(sys.argv[2:])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
