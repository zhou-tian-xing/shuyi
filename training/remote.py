#!/usr/bin/env python3
"""Tiny paramiko SSH helper: run a command on the training server, stream output.

Usage: python training/remote.py "nvidia-smi"
       python training/remote.py -f script.sh   (run local file's content)
"""
import os
import sys

import paramiko

# Same rule as ssh.py: the host and password are per-session facts and live in
# the environment, not in the repository.
HOST = os.environ.get("SHUYI_HOST", "")
PORT = int(os.environ.get("SHUYI_PORT", "22"))
USER = os.environ.get("SHUYI_USER", "root")
PASS = os.environ.get("SHUYI_PASS", "")


def main():
    if not HOST or not PASS:
        print("set SHUYI_HOST and SHUYI_PASS (and SHUYI_PORT if not 22)", file=sys.stderr)
        return 2
    args = sys.argv[1:]
    if not args:
        print("usage: remote.py <cmd> | remote.py -f <file>")
        return 2
    if args[0] == "-f":
        with open(args[1], "r", encoding="utf-8") as f:
            cmd = f.read()
    else:
        cmd = args[0]
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)
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


if __name__ == "__main__":
    sys.exit(main())
