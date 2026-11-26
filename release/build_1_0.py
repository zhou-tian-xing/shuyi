#!/usr/bin/env python3
"""Stage the four 1.0 packages (plus a source archive) and smoke-test each one.

    python release/build_1_0.py --model models/b6c96-s305983-d3226791.bin

Products, because the UI and the engine are separate programs:

    shuyi-1.0-engine-cpu.zip   headless CPU engine
    shuyi-1.0-engine-cuda.zip  headless CUDA engine
    shuyi-1.0-ui-cpu.zip       UI + CPU engine + the strongest model
    shuyi-1.0-ui-cuda.zip      UI + CUDA engine + the strongest model

Every package names its engine `shuyi_engine.exe`, so unzipping the CUDA engine
package over a UI package swaps the backend without editing a config file.  The
smoke test launches the packaged binaries from the staged directory: an archive
that cannot answer `capabilities` is not a release.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

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
1. 解压到任意目录（不要放在中文路径或桌面以外需要权限的地方）。
2. 双击 shuyi.exe。状态栏应显示“引擎已启动：shuyi_engine.exe”。
3. 默认是「双人对弈 + 无过手规则」。点棋盘落子，或按空格让引擎分析当前局面。
4. 引擎菜单 → 设置… 可改「分析访问上限 / 落子访问上限 / A值」。
   访问量是唯一与机器无关的预算：512 次模拟在笔记本和 GPU 上含义相同。
5. 想和引擎下棋：引擎菜单 → MCTS + 网络（模型已随包附带）。

数字怎么看
----------
* 每个点上的数字是这一着在棋盘格上的推荐棋子数字；
* 点的颜色是胜率，已解算则精确值、否则 Wilson 95% 置信下界，右上角摘要同一口径；
* 已解算的点显示目差（+3 / -12），因为胜率只有 0% 和 100%，分不出输 2 目和输 18 目；
* 没解算的点显示胜率、访问数和均分；均分没有来源时显示 --，不会用胜率反推。
"""

UI_README = """数弈 1.0
========

双击 shuyi.exe 即可。界面不含任何搜索代码：引擎是旁边的 shuyi_engine.exe，
由 shuyi-ui.ini 里的 engine= 指定，两个程序只通过文本协议通信。
""" + COMMON_TAIL + """
包里有什么
----------
* shuyi.exe             Win32 界面（管规则、状态、显示）
* shuyi_engine.exe      {engine_note}
* models/               随包权重（{model_note}）
* shuyi-ui.ini          界面设置
* RULES.zh-CN.md       规则全文
* ENGINE_PROTOCOL.md   引擎协议（行协议 + JSON 分析流）
* LICENSE              MIT
* BUILD.txt            这次构建的 commit、时间与每个文件的 sha256

换模型
------
把任意 `b<blocks>c<channels>-s<steps>-d<data>.bin` 放进 models/，
菜单「引擎 → MCTS + 网络」里就会出现，不用重启，也不用改配置。
规则集必须匹配（nopass 的权重不能用在 classic 上）。
"""

ENGINE_README = """数弈 1.0 — 无头引擎
==================

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

权重是数据，不随这个包发布。最强的一份在
https://github.com/zhou-tian-xing/shuyi/releases/tag/v1.0 和仓库的 models/ 目录里。
"""


def git(*rest: str) -> str:
    try:
        return subprocess.run(["git", *rest], cwd=REPO, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:  # noqa: BLE001 - packaging must work without git too
        return "unknown"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant_spec() -> dict:
    cpu = REPO / "build" / "shuyi_engine.exe"
    cuda = REPO / "build-cuda" / "shuyi_engine.exe"
    ui = REPO / "build" / "shuyi.exe"
    return {
        "engine-cpu": {"engine": cpu, "ui": None},
        "engine-cuda": {"engine": cuda, "ui": None},
        "ui-cpu": {"engine": cpu, "ui": ui},
        "ui-cuda": {"engine": cuda, "ui": ui},
    }


def stage(stage_dir: Path, engine: Path, ui: Path | None, model: Path | None,
          cuda: bool) -> None:
    stage_dir.mkdir(parents=True)
    shutil.copy2(engine, stage_dir / "shuyi_engine.exe")
    if ui is not None:
        shutil.copy2(ui, stage_dir / "shuyi.exe")
        (stage_dir / "shuyi-ui.ini").write_text(INI, encoding="utf-8")
        shutil.copy2(REPO / "docs" / "RULES.zh-CN.md", stage_dir / "RULES.zh-CN.md")
        shutil.copy2(REPO / "protocol" / "ENGINE_PROTOCOL.md", stage_dir / "ENGINE_PROTOCOL.md")
    else:
        shutil.copy2(REPO / "protocol" / "ENGINE_PROTOCOL.md", stage_dir / "ENGINE_PROTOCOL.md")
    if model is not None and ui is not None:
        (stage_dir / "models").mkdir()
        shutil.copy2(model, stage_dir / "models" / model.name)
    shutil.copy2(REPO / "LICENSE", stage_dir / "LICENSE")

    engine_note = ("CUDA 引擎（需要 NVIDIA 驱动，构建时匹配的 GPU 架构要能跑）"
                   if cuda else "CPU 引擎（任何 x64 Windows 都能跑）")
    model_note = f"{model.name}，ELO +2708" if model is not None and ui is not None else "无"
    if ui is not None:
        (stage_dir / "README.txt").write_text(
            UI_README.format(engine_note=engine_note, model_note=model_note), encoding="utf-8")
    else:
        (stage_dir / "README.txt").write_text(
            ENGINE_README.format(engine_note=engine_note), encoding="utf-8")

    lines = ["数弈 1.0 build",
             f"version  1.0",
             f"commit   {git('rev-parse', '--short', 'HEAD')}",
             f"date     {subprocess.run(['powershell', '-NoProfile', '-Command',
                                        'Get-Date -Format o'], capture_output=True, text=True,
                                       timeout=20).stdout.strip() or 'unknown'}"]
    for path in sorted(stage_dir.rglob("*")):
        if path.is_file() and path.name != "BUILD.txt":
            lines.append(f"sha256  {digest(path)}  {path.relative_to(stage_dir).as_posix()}")
    (stage_dir / "BUILD.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke(stage_dir: Path, ui: bool, cuda: bool, model: Path | None) -> None:
    """Run the packaged binaries exactly as a user would."""
    commands = ["protocol_version", "capabilities"]
    if model is not None and ui:
        commands += [f"set_model models/{model.name}", "set_engine mcts_nn"]
    commands.append("quit")
    stream = "".join(f"{i} {c}\n" for i, c in enumerate(commands, 1))
    proc = subprocess.run([str(stage_dir / "shuyi_engine.exe")], cwd=stage_dir, input=stream,
                          capture_output=True, text=True, timeout=90,
                          encoding="utf-8", errors="replace")
    if proc.returncode:
        raise SystemExit(f"{stage_dir.name}: engine exit {proc.returncode}: {proc.stderr[:400]}")
    out = proc.stdout
    if "1 = 2" not in out:
        raise SystemExit(f"{stage_dir.name}: engine did not answer protocol v2: {out[:300]}")
    if f"cuda={int(cuda)}" not in out:
        raise SystemExit(f"{stage_dir.name}: expected cuda={int(cuda)}: {out[:300]}")
    if model is not None and ui and "set_engine mcts_nn" in stream:
        last = proc.stdout.strip().splitlines()[-2:]
        if any("error" in line.lower() for line in last):
            raise SystemExit(f"{stage_dir.name}: model did not load: {last}")
    if ui:
        menu = stage_dir / "_menu.txt"
        run = subprocess.run([str(stage_dir / "shuyi.exe"), "--dump-models", str(menu)],
                             cwd=stage_dir, capture_output=True, text=True, timeout=90)
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
    parser.add_argument("--model", type=Path, required=True,
                        help="weight to ship inside the UI packages")
    parser.add_argument("--out", type=Path, default=REPO / "dist" / "1.0")
    parser.add_argument("--tag", default="v1.0")
    args = parser.parse_args()

    model = (REPO / args.model).resolve() if not args.model.is_absolute() else args.model
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    specs = variant_spec()
    for name, spec in specs.items():
        for key in ("engine", "ui"):
            if spec[key] is not None and not spec[key].is_file():
                raise SystemExit(f"{name}: missing {spec[key]}")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = git("status", "--porcelain")
    if sources:
        print("note: the tree has uncommitted changes; BUILD.txt records HEAD, not the files")

    for name, spec in specs.items():
        cuda = spec["engine"] == specs["engine-cuda"]["engine"]
        stage_dir = out / f"shuyi-1.0-{name}"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage(stage_dir, spec["engine"], spec["ui"], model, cuda)
        smoke(stage_dir, spec["ui"] is not None, cuda, model)
        archive = out / f"shuyi-1.0-{name}.zip"
        archive.unlink(missing_ok=True)
        zip_dir(stage_dir, archive)
        print(f"{archive.name:32s} {archive.stat().st_size / 1024 / 1024:6.1f} MB  smoke ok")

    source_zip = out / "shuyi-1.0-source.zip"
    subprocess.run(["git", "archive", "--format=zip", f"--prefix=shuyi-1.0/",
                    "-o", str(source_zip), args.tag], cwd=REPO, check=True)
    print(f"{source_zip.name:32s} {source_zip.stat().st_size / 1024 / 1024:6.1f} MB  from {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
