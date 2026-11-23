#!/usr/bin/env python3
"""Upload the repo to an Ubuntu host and optionally start the 15-gen run.

This helper is for the operator's machine, not the training server.  It uses
paramiko (pip install paramiko) and reads the password from an environment
variable so it never appears in the process list:

  export SHUYI_SSH_PASSWORD='...'
  python training/remote_deploy.py --host 1.2.3.4 --user ubuntu \
      --remote-dir /home/ubuntu/shuyi --setup --start --generations 8

The archive excludes build/, runs/, .venv/ and data/*.bin; the server rebuilds
the C++ binaries with training/setup_ubuntu.sh.
"""
import argparse
import fnmatch
import os
import shlex
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root (training/..)


def build_archive(path):
    exclusions = {"build", "runs", ".venv", "__pycache__", ".git"}
    with tarfile.open(path, "w:gz") as archive:
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in exclusions]
            for name in files:
                full = Path(root) / name
                relative = full.relative_to(REPO)
                if any(part in exclusions for part in relative.parts):
                    continue
                if fnmatch.fnmatch(str(relative), "data/*.bin"):
                    continue
                if name.endswith((".pyc", ".o", ".obj", ".exe")):
                    continue
                archive.add(full, arcname=str(Path("shuyi") / relative))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    parser.add_argument("--remote-dir", required=True, help="absolute path on server")
    parser.add_argument("--password-env", default="SHUYI_SSH_PASSWORD")
    parser.add_argument("--key", default="", help="private key path instead of password")
    parser.add_argument("--setup", action="store_true", help="run setup_ubuntu.sh")
    parser.add_argument("--start", action="store_true", help="start 15-gen run with nohup")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args(argv)

    try:
        import paramiko
    except ImportError:
        print("paramiko is required: python -m pip install paramiko", file=sys.stderr)
        return 2

    password = os.environ.get(args.password_env, "")
    if not args.key and not password:
        print(f"missing password env {args.password_env} or --key", file=sys.stderr)
        return 2

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect = {"hostname": args.host, "port": args.port, "username": args.user, "timeout": 20}
    if args.key:
        connect["key_filename"] = args.key
    else:
        connect["password"] = password
    print(f"connecting {args.user}@{args.host}:{args.port}")
    client.connect(**connect)

    def run(command, check=True):
        print(f"$ {command}", flush=True)
        _, stdout, stderr = client.exec_command(command, get_pty=True)
        for line in stdout:
            print(line, end="", flush=True)
        code = stdout.channel.recv_exit_status()
        if check and code != 0:
            raise RuntimeError(f"remote command failed ({code}): {command}")
        return code

    archive = Path(tempfile.gettempdir()) / "shuyi_src.tar.gz"
    remote_archive = "/tmp/shuyi_src.tar.gz"
    try:
        if not args.skip_upload:
            print(f"packing {REPO}")
            build_archive(archive)
            print(f"archive {archive.stat().st_size / (1024 * 1024):.1f} MiB")
            run(f"mkdir -p {shlex.quote(args.remote_dir)}")
            sftp = client.open_sftp()
            sftp.put(str(archive), remote_archive)
            sftp.close()
            run(
                f"tar -xzf {remote_archive} -C {shlex.quote(args.remote_dir)} "
                "--strip-components=1"
            )
        if args.setup:
            run(f"cd {shlex.quote(args.remote_dir)} && bash training/setup_ubuntu.sh")
        if args.start:
            # run_generation.sh reads the run shape from the environment; the
            # 15-gen wrapper it used to call does not exist any more.
            run_name = args.run_name or "run_latest"
            command = (
                f"cd {shlex.quote(args.remote_dir)} && "
                f"RUN_NAME={shlex.quote(run_name)} GENERATIONS={int(args.generations)} "
                f"nohup bash training/run_generation.sh "
                f"> training_full.log 2>&1 & echo $!"
            )
            run(command, check=False)
            print("started in background; monitor with:")
            print(f"  ssh -p {args.port} {args.user}@{args.host} "
                  f"'tail -f {args.remote_dir}/training_full.log'")
        return 0
    finally:
        if archive.exists():
            archive.unlink()
        client.close()


if __name__ == "__main__":
    sys.exit(main())
