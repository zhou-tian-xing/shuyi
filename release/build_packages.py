#!/usr/bin/env python3
"""Stage the four release packages (plus a source archive) and smoke-test each one.

    python release/build_packages.py --version 1.1 --tag v1.1
    python release/build_packages.py --version 1.1 --tag v1.1 --model models/b6c96.bin

Products, because the UI and the engine are separate programs:

    shuyi-<version>-engine-cpu.zip   headless CPU engine
    shuyi-<version>-engine-cuda.zip  headless CUDA engine
    shuyi-<version>-ui-cpu.zip       UI + CPU engine
    shuyi-<version>-ui-cuda.zip      UI + CUDA engine

Weights are *not* shipped by default: they are published separately, and a package
that carries one goes stale the moment a stronger one exists.  `--model` is still
there for a build that should carry one.

Every package names its engine `shuyi_engine.exe`, so unzipping the CUDA engine
package over a UI package swaps the backend without editing a config file.  The
smoke test launches the packaged binaries from the staged directory: an archive
that cannot answer `capabilities` is not a release.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# A release is tested the way a stranger's machine looks: the system directory and
# nothing else.  Inheriting this developer's PATH hid two missing runtimes once.
CLEAN_ENV = {
    "SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"),
    "PATH": os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"),
}

INI = """# shuyi-ui.ini — 界面设置（相对本目录）
engine=shuyi_engine.exe
analysis_visits=512
play_visits=512
# A值：终局分数的平移量（终局 = 棋盘分 + A）。nopass 的标定值约 +0.48~+0.55。
a_value=0.5
ruleset=nopass
"""

COMMON_TAIL = """
一分钟上手
----------
1. 解压到任意目录。
2. 双击 shuyi.exe。左侧「AI评估」那一行显示这次搜索的量与速度，下面一行是状态；
   没有权重时它会写「没有模型：把 .bin 放进 models/」。
3. 点棋盘落子；空格开始 / 暂停分析。开着分析时，落子或翻棋谱都会在新局面接着算。
4. 左侧是分析的结果（胜率、目差、选点、整盘棋的走势），右侧是棋谱树：
   ↑/↓ 回溯（↓ 走主干），点节点跳过去，Delete 删掉一手和它后面的走法。
5. 文件菜单：新局 / 保存棋谱（Ctrl+S）/ 载入棋谱 / 退出。棋谱是文本，分析一起存。

数字怎么看
----------
* 棋盘上每个标记：上排胜率、中排访问数、下排预期终局分；**右上角的上标是要落的那颗子的数字**。
* 标记的颜色是胜率，已解算则精确值、否则 Wilson 95% 置信下界；
* 已解算的显示目差（+3 / -12），因为胜率只有 0% 和 100%，分不出输 2 目和输 18 目；
* 没解算的显示胜率、访问数和均分；均分没有来源时显示 --，不会用胜率反推。
* 棋谱每行左边的小色块：这一手比上一步的评估好了多少——蓝＝明显更好（引擎当时没算过这手），
  绿＝不差，红＝变差。选中那一行时它也在。
* 走势图是整盘棋：实线走主干，只在当前点的分叉处另画一支虚线变着；金色是目差，右侧有刻度。
* 终局不盖任何东西：第一行是灰色的「终局」，下面写谁胜和最终分值，最后一手也有颜色。
* 翻棋谱不会让电脑落子；轮到时面板会写「轮到电脑」，要它走一步用 引擎 → 让电脑走一步。
"""

UI_README = """数弈 {version}
=========

双击 shuyi.exe 即可。界面不含任何搜索代码：引擎是旁边的 shuyi_engine.exe，
由 shuyi-ui.ini 里的 engine= 指定，两个程序只通过文本协议通信。
""" + COMMON_TAIL + """
包里有什么
----------
* shuyi.exe             Win32 界面（管规则、状态、显示）
* shuyi_engine.exe      {engine_note}
* models/               {model_note}
* shuyi-ui.ini          界面设置
* RULES.zh-CN.md       规则全文
* ENGINE_PROTOCOL.md   引擎协议（行协议 + JSON 分析流）
* LICENSE              MIT
* BUILD.txt            这次构建的 commit、时间与每个文件的 sha256

换模型
------
把任意 `b<blocks>c<channels>-s<steps>-d<data>.bin` 放进 models/，
菜单「引擎 → MCTS + 网络」里就会出现，不用重启，也不用改配置。
规则集必须匹配（nopass 的权重不能用在 classic 上），不匹配的模型会被拒绝加载。
"""

ENGINE_README = """数弈 {version} — 无头引擎
=====================

这是引擎，没有界面。它说行协议（见 ENGINE_PROTOCOL.md），由调用方驱动：

    shuyi_engine.exe                 # 从 stdin 读命令，向 stdout 写回复
    1 protocol_version
    2 capabilities
    3 quit

一行一条命令，回复形如 `<序号> = <结果>`。分析结果也可以开成 JSON 流（stream=1）。

它适合
------
* 自己写前端（任意语言）的人：协议是纯文本，不链接任何库；
* 批量跑对局/评测的脚本；
* 想直接拿引擎做研究的人。

{engine_note}

模型
----
引擎启动时没有模型，需要显式加载：

    set_model models/<name>.bin
    set_engine mcts_nn

权重是数据，不随这个包发布。发布页里有单独的最强模型：
https://github.com/zhou-tian-xing/shuyi/releases
"""


def git(*rest: str) -> str:
    try:
        return subprocess.run(["git", *rest], cwd=REPO, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:  # noqa: BLE001 - packaging must work without git too
        return "unknown"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_cuda_engine() -> Path | None:
    """Where the CUDA engine is, or None.

    The CUDA build tree has to live outside the repository: nvcc cannot read the
    project's non-ASCII directory name, so it is built through an ASCII mirror.
    `build-cuda` in the repository is a junction to that tree, which is what the
    first candidate follows; the second is the tree itself, for a checkout whose
    junction was never made.
    """
    for candidate in (REPO / "build-cuda" / "shuyi_engine.exe",
                      REPO.parent / "shuyi_cuda" / "build" / "shuyi_engine.exe"):
        if candidate.is_file():
            return candidate
    return None


def variant_spec() -> dict:
    cpu = REPO / "build" / "shuyi_engine.exe"
    cuda = find_cuda_engine()
    ui = REPO / "build" / "shuyi.exe"
    return {
        "engine-cpu": {"engine": cpu, "ui": None},
        "engine-cuda": {"engine": cuda, "ui": None},
        "ui-cpu": {"engine": cpu, "ui": ui},
        "ui-cuda": {"engine": cuda, "ui": ui},
    }


def stage(stage_dir: Path, engine: Path, ui: Path | None, model: Path | None,
          cuda: bool, runtime: list[Path], version: str) -> None:
    stage_dir.mkdir(parents=True)
    shutil.copy2(engine, stage_dir / "shuyi_engine.exe")
    for dll in runtime:
        shutil.copy2(dll, stage_dir / dll.name)
    if ui is not None:
        shutil.copy2(ui, stage_dir / "shuyi.exe")
        (stage_dir / "shuyi-ui.ini").write_text(INI, encoding="utf-8")
        shutil.copy2(REPO / "docs" / "RULES.zh-CN.md", stage_dir / "RULES.zh-CN.md")
        # The folder exists even when it is empty: the UI scans it, and a README in
        # it says where weights come from -- an absent folder looks like a mistake.
        (stage_dir / "models").mkdir()
        if model is not None:
            shutil.copy2(model, stage_dir / "models" / model.name)
        else:
            (stage_dir / "models" / "README.txt").write_text(
                "把权重放进这个目录：b<blocks>c<channels>-s<steps>-d<data>.bin\n\n"
                "模型不随这个包发布，在发布页单独下载：\n"
                "https://github.com/zhou-tian-xing/shuyi/releases\n\n"
                "规则集必须和棋局匹配（nopass 的权重不能用 classic 局面），页面上标了每个\n"
                "权重对应的规则集；不匹配的模型会被拒绝加载。\n",
                encoding="utf-8")
    shutil.copy2(REPO / "protocol" / "ENGINE_PROTOCOL.md", stage_dir / "ENGINE_PROTOCOL.md")
    shutil.copy2(REPO / "LICENSE", stage_dir / "LICENSE")

    engine_note = ("CUDA 引擎。**前置条件：CUDA 12 运行时**（cudart64_12.dll、"
                   "cublas64_12.dll、cublasLt64_12.dll 必须在 PATH 或本目录），另有 NVIDIA 驱动；"
                   "VC++ 运行库已随包附带" if cuda
                   else "CPU 引擎（任何 x64 Windows 都能跑，无需额外组件）")
    if model is not None and ui is not None:
        model_note = f"{model.name}（随包）"
    else:
        model_note = "不含权重，见项目发布页"
    if ui is not None:
        (stage_dir / "README.txt").write_text(
            UI_README.format(version=version, engine_note=engine_note, model_note=model_note),
            encoding="utf-8")
    else:
        (stage_dir / "README.txt").write_text(
            ENGINE_README.format(version=version, engine_note=engine_note), encoding="utf-8")

    lines = [f"数弈 {version} build",
             f"version  {version}",
             f"commit   {git('rev-parse', '--short', 'HEAD')}",
             f"date     {subprocess.run(['powershell', '-NoProfile', '-Command',
                                        'Get-Date -Format o'], capture_output=True, text=True,
                                       timeout=20).stdout.strip() or 'unknown'}"]
    for path in sorted(stage_dir.rglob("*")):
        if path.is_file() and path.name != "BUILD.txt":
            lines.append(f"sha256  {digest(path)}  {path.relative_to(stage_dir).as_posix()}")
    (stage_dir / "BUILD.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def env_for(cuda_bin: Path | None) -> dict:
    """A stranger's machine: system directory, plus the documented CUDA prerequisite."""
    env = dict(CLEAN_ENV)
    if cuda_bin is not None:
        env["PATH"] = env["PATH"] + os.pathsep + str(cuda_bin)
    return env


def smoke(stage_dir: Path, ui: bool, cuda: bool, model: Path | None,
          cuda_bin: Path | None = None) -> None:
    """Run the packaged binaries exactly as a user would."""
    env = env_for(cuda_bin)
    commands = ["protocol_version", "capabilities"]
    if model is not None and ui:
        commands += [f"set_model models/{model.name}", "set_engine mcts_nn"]
    commands.append("quit")
    stream = "".join(f"{i} {c}\n" for i, c in enumerate(commands, 1))
    proc = subprocess.run([str(stage_dir / "shuyi_engine.exe")], cwd=stage_dir, input=stream,
                          capture_output=True, text=True, timeout=120, env=env,
                          encoding="utf-8", errors="replace")
    if proc.returncode:
        raise SystemExit(f"{stage_dir.name}: engine exit {proc.returncode}: {proc.stderr[:400]}")
    out = proc.stdout
    if "1 = 2" not in out:
        raise SystemExit(f"{stage_dir.name}: engine did not answer protocol v2: {out[:300]}")
    if f"cuda={int(cuda)}" not in out:
        raise SystemExit(f"{stage_dir.name}: expected cuda={int(cuda)}: {out[:300]}")
    if model is not None and ui:
        last = proc.stdout.strip().splitlines()[-2:]
        if any("error" in line.lower() for line in last):
            raise SystemExit(f"{stage_dir.name}: model did not load: {last}")
    if ui:
        menu = stage_dir / "_menu.txt"
        run = subprocess.run([str(stage_dir / "shuyi.exe"), "--dump-models", str(menu)],
                             cwd=stage_dir, capture_output=True, text=True, timeout=120,
                             env=env)
        if run.returncode:
            raise SystemExit(f"{stage_dir.name}: UI exit {run.returncode}")
        text = menu.read_text(encoding="utf-8", errors="replace") if menu.is_file() else ""
        menu.unlink(missing_ok=True)
        if model is not None and model.name not in text:
            raise SystemExit(f"{stage_dir.name}: UI did not list {model.name}: {text[:300]}")


def zip_dir(stage_dir: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(stage_dir.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(stage_dir.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="1.1", help="version string in names and READMEs")
    parser.add_argument("--tag", default="v1.1", help="git tag the source archive comes from")
    parser.add_argument("--model", type=Path, default=None,
                        help="weight to ship inside the UI packages (default: none)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cuda-bin", type=Path, default=Path("E:/CUDA/v12.4/bin"),
                        help="directory holding cudart64_12.dll and cublas64_12.dll")
    args = parser.parse_args()

    version = args.version
    model = None
    if args.model is not None:
        model = (REPO / args.model).resolve() if not args.model.is_absolute() else args.model
        if not model.is_file():
            raise SystemExit(f"model not found: {model}")
    specs = variant_spec()
    if specs["engine-cuda"]["engine"] is None:
        raise SystemExit("CUDA engine not built: run tools\\build_cuda.bat "
                         "(it needs an ASCII mirror of this repository; see the README)")
    for name, spec in specs.items():
        for key in ("engine", "ui"):
            if spec[key] is not None and not spec[key].is_file():
                raise SystemExit(f"{name}: missing {spec[key]}")

    out = (args.out or (REPO / "dist" / version)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = git("status", "--porcelain")
    if sources:
        print("note: the tree has uncommitted changes; BUILD.txt records HEAD, not the files")

    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    # CUDA itself is NOT bundled: cublasLt64_12.dll alone is 473 MB, an absurd
    # download for a 4 MB program.  It is a documented prerequisite instead, and
    # the smoke test runs the CUDA package with that prerequisite on PATH -- which
    # is what a machine that followed the README looks like.
    runtime = []
    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        candidate = system32 / name
        if not candidate.is_file():
            raise SystemExit(f"MSVC runtime not found: {candidate}")
        runtime.append(candidate)

    for name, spec in specs.items():
        cuda = spec["engine"] == specs["engine-cuda"]["engine"]
        stage_dir = out / f"shuyi-{version}-{name}"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage(stage_dir, spec["engine"], spec["ui"], model, cuda, runtime if cuda else [], version)
        smoke(stage_dir, spec["ui"] is not None, cuda, model, args.cuda_bin if cuda else None)
        archive = out / f"shuyi-{version}-{name}.zip"
        archive.unlink(missing_ok=True)
        zip_dir(stage_dir, archive)
        print(f"{archive.name:32s} {archive.stat().st_size / 1024 / 1024:6.1f} MB  smoke ok")

    source_zip = out / f"shuyi-{version}-source.zip"
    subprocess.run(["git", "archive", "--format=zip", f"--prefix=shuyi-{version}/",
                    "-o", str(source_zip), args.tag], cwd=REPO, check=True)
    print(f"{source_zip.name:32s} {source_zip.stat().st_size / 1024 / 1024:6.1f} MB  from {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
