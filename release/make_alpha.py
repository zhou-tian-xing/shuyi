#!/usr/bin/env python3
"""Stage and smoke-test a Windows package with CPU and optional CUDA engines.

    python release/make_alpha.py --engine build/shuyi_engine.exe --cuda-engine build-cuda/shuyi_engine.exe \
        --ui build/shuyi.exe --model models/b4c32-s42012-d595499.bin

The result is `dist/shuyi-alpha/` and `dist/shuyi-alpha.zip`, containing only
what a player needs:

    shuyi.exe                 the Win32 front end
    shuyi_engine.exe          the default CPU engine
    shuyi_engine_cuda.exe     optional CUDA engine
    models/<name>.bin         the shipped network (plays under nopass)
    shuyi-ui.ini              the settings the package ships with
    ENGINE_PROTOCOL.md        the contract, for anyone writing another front end
    README.txt                how to start it, and what each number means

Nothing else: no training scripts, no test tools, no source.
"""
import argparse
import shutil
import subprocess
from tempfile import TemporaryDirectory
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "dist"
NAME = "shuyi-alpha"

README = """数弈 alpha
==========

双点 shuyi.exe 即可。界面不含任何搜索代码：引擎是旁边的 shuyi_engine.exe，
由 shuyi-ui.ini 里的 engine= 指定，两个程序只通过文本协议通信。

一分钟上手
----------
1. 启动后默认是“双人对弈 + 无过手规则”。如果引擎加载成功，状态栏会写
   “引擎已启动：shuyi_engine.exe”。
2. 按空格开始分析当前局面；再按空格停止。达到访问上限会自动停下，
   再按一次空格则不计上限、继续同一棵搜索树。
3. 引擎菜单 → 设置… 可以改“分析访问上限 / 落子访问上限 / A值”。
   访问量是唯一与机器无关的预算：512 次模拟在笔记本和 GPU 上含义相同。
4. 引擎菜单 → 引擎信息… 显示当前引擎程序、后端（CUDA/CPU）和模型。

数字怎么看
----------
* 每个点上的数字是这一着在棋盘格上的推荐棋子数字；
* 点上的颜色是“已解算则精确值，否则 Wilson 95% 置信下界”的胜率，
  右上角摘要用的是同一个口径；
* **已解算的点显示目差**（+3 / -12）：胜率只有 0% 和 100%，分不出“输 2 目”
  和“输 18 目”，所以解算出来的点直接给精确差值；
* 没解算的点显示胜率、访问数和均分；均分没有来源时显示 --，
  不会用胜率反推。

换引擎 / 换模型
---------------
* 引擎菜单 → 选择引擎程序… 可以指向任意实现了 ENGINE_PROTOCOL.md 的程序；
* 引擎菜单 → MCTS + 网络 子菜单列出扫描到的全部模型，按规则集标注；
* shuyi-ui.ini 里 engine= 支持相对路径（相对本目录）。

模型在哪
--------
引擎和界面是两个产品，**权重是数据**，不随包固定：把任意 `b<blocks>c<channels>-s<steps>-d<data>.bin`
放进 `models/`，界面菜单（引擎 → MCTS + 网络）里就会出现；没有模型也能启动，
双人对弈照常，只是没有分析。

这个 alpha 包含
---------------
* shuyi.exe            Win32 界面（规则与显示；不含搜索）
* shuyi_engine.exe     默认 CPU 引擎
* shuyi_engine_cuda.exe（可选）通过“选择引擎程序…”启用，需匹配构建时的 GPU 架构，并安装 NVIDIA 驱动和 CUDA 运行库
* DESIGN.md            引擎设计、网络形状、训练思路、现有训练量与已知短板
* RULES.zh-CN.md       规则全文
* ENGINE_PROTOCOL.md   引擎协议（行协议 + JSON 分析流）
* BUILD.txt            这次构建的 commit、时间与三个文件的校验和

这是给开发者之间分享的包：DESIGN.md 里有完整的"为什么这么设计"。
"""

INI = """# shuyi-ui.ini — 界面设置
# engine= 支持相对路径（相对本目录）；留空则自动探测同目录的 shuyi_engine.exe。
engine=shuyi_engine.exe
# 每次分析 / 每次落子的模拟次数。0 表示不限（分析用；落子始终要有上限）。
analysis_visits=512
play_visits=512
# A值：终局分数的平移量（终局 = 棋盘分 + A）。
# 最新 nopass 标定 A* 约 +0.48。界面默认 A=+0.5，可按 0.5 步长调整。
a_value=0.5
# 规则集：nopass / blocking / classic。默认是无过手（阻挡规则 + 一方无子可落立刻终局）。
ruleset=nopass
"""


def buildInfo(args):
    """What exactly was built, so a bug report can name a binary."""
    import datetime
    import hashlib

    def git(*rest):
        try:
            return subprocess.run(["git", *rest], cwd=REPO, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001 - a package must build without git too
            return "unknown"

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    lines = [
        "数弈 alpha build",
        f"commit   {git('rev-parse', '--short', 'HEAD')}",
        f"date     {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"engine   shuyi_engine.exe  sha256:{digest(args.engine)}  (CPU)",
        *([f"cuda     shuyi_engine_cuda.exe  sha256:{digest(args.cuda_engine)}"]
          if args.cuda_engine else []),
        f"ui       {args.ui.name}  sha256:{digest(args.ui)}",
        f"model    {args.model.name}  sha256:{digest(args.model)}"
        if args.model
        else "model    (none: download a .bin and drop it in models/)",
        "",
        "Rebuild: see release/make_alpha.py in the source repository.",
    ]
    return "\n".join(lines) + "\n"


def verify_archive(archive, unpacked, args):
    """Launch the packaged executables using exactly the files in the ZIP."""
    if sys.platform != "win32":
        raise SystemExit("Windows is required to smoke-test the Windows release")
    import ctypes
    import os

    with zipfile.ZipFile(archive) as package:
        package.extractall(unpacked)
    root = unpacked / NAME
    env = os.environ.copy()
    windows = Path(os.environ["SystemRoot"])
    env["PATH"] = os.pathsep.join([str(windows / "System32"), str(windows)])
    env.pop("SHUYI_ENGINE", None)
    env.pop("SHUYI_UI_INI", None)

    def run(command, *, environment, commands=None):
        proc = subprocess.run(command, cwd=root, env=environment, input=commands,
                              capture_output=True, encoding="utf-8", errors="replace", timeout=30)
        if proc.returncode:
            raise SystemExit(f"{command[0].name} smoke test failed: "
                             f"exit=0x{proc.returncode & 0xffffffff:08x} {proc.stderr.strip()}")
        return proc.stdout.splitlines()

    def verify_engine(filename, environment, cuda):
        commands = ["protocol_version", "ruleset", "capabilities"]
        if args.model:
            commands += [f"set_model models/{args.model.name}", "set_engine mcts_nn"]
        commands.append("quit")
        stream = "".join(f"{i} {command}\n" for i, command in enumerate(commands, 1))
        replies = run([root / filename], environment=environment, commands=stream)
        for i, command in enumerate(commands, 1):
            if not any(line.startswith(f"{i} =") for line in replies):
                raise SystemExit(f"{filename}: failed {command}: {replies}")
        if "1 = 2" not in replies or "2 = nopass" not in replies:
            raise SystemExit(f"{filename}: expected protocol v2 and nopass: {replies}")
        if not any(line.startswith("3 =") and f" cuda={int(cuda)} " in line for line in replies):
            raise SystemExit(f"{filename}: unexpected CUDA capability: {replies}")

    old_mode = ctypes.windll.kernel32.SetErrorMode(0x8003)
    try:
        verify_engine("shuyi_engine.exe", env, cuda=False)
        probe = unpacked / "menu.txt"
        run([root / "shuyi.exe", "--dump-models", probe], environment=env)
        if not probe.is_file() or (args.model and args.model.name not in probe.read_text(encoding="utf-8")):
            raise SystemExit("packaged UI did not find its model menu")
        if args.cuda_engine:
            verify_engine("shuyi_engine_cuda.exe", os.environ.copy(), cuda=True)
    finally:
        ctypes.windll.kernel32.SetErrorMode(old_mode)


def stage_package(stage, args):
    stage.mkdir()
    shutil.copy2(args.ui, stage / "shuyi.exe")
    shutil.copy2(args.engine, stage / "shuyi_engine.exe")
    if args.cuda_engine:
        shutil.copy2(args.cuda_engine, stage / "shuyi_engine_cuda.exe")
    if args.model:
        (stage / "models").mkdir()
        shutil.copy2(args.model, stage / "models" / args.model.name)
    shutil.copy2(REPO / "protocol" / "ENGINE_PROTOCOL.md", stage / "ENGINE_PROTOCOL.md")
    shutil.copy2(REPO / "docs" / "RULES.zh-CN.md", stage / "RULES.zh-CN.md")
    shutil.copy2(REPO / "LICENSE", stage / "LICENSE")
    shutil.copy2(REPO / "release" / "ALPHA_NOTES.md", stage / "DESIGN.md")
    (stage / "shuyi-ui.ini").write_text(INI, encoding="utf-8")
    (stage / "README.txt").write_text(README, encoding="utf-8")
    (stage / "BUILD.txt").write_text(buildInfo(args), encoding="utf-8")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--cuda-engine", type=Path)
    parser.add_argument("--ui", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=None,
                        help="network to ship; omit for a package without models.  The "
                             "UI and the engine are the products; weights are a download "
                             "and change far more often than the binaries")
    parser.add_argument("--out", type=Path, default=DIST)
    args = parser.parse_args(argv[1:])

    binaries = [p for p in (args.engine, args.ui, args.cuda_engine, args.model) if p]
    for path in binaries:
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stage = out / NAME
    archive = out / f"{NAME}.zip"
    if stage.resolve() != stage:
        raise SystemExit(f"package directory resolves outside output: {stage}")

    with TemporaryDirectory(prefix=".shuyi-release-", dir=out) as temp:
        temp = Path(temp)
        prepared = temp / NAME
        staged_zip = temp / archive.name
        stage_package(prepared, args)
        with zipfile.ZipFile(staged_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for path in sorted(prepared.rglob("*")):
                if path.is_file():
                    zip_file.write(path, path.relative_to(temp))
        verify_archive(staged_zip, temp / "smoke", args)

        previous = temp / "previous"
        if stage.exists():
            stage.replace(previous)
        try:
            prepared.replace(stage)
            staged_zip.replace(archive)
        except OSError:
            if stage.exists():
                stage.replace(prepared)
            if previous.exists():
                previous.replace(stage)
            raise

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"staged  {stage}")
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            print(f"  {path.stat().st_size:>9}  {path.relative_to(stage)}")
    print(f"archive {archive}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
