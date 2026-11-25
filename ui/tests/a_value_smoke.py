"""Drive the Win32 A-value settings, menu and shortcuts against a temporary INI."""
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

user32 = ctypes.windll.user32
EnumCallback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [EnumCallback, wintypes.LPARAM]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.SendMessageTimeoutW.argtypes = [
    wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
    wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t),
]


def wait_window(proc, class_name):
    found = None

    @EnumCallback
    def visit(window, _):
        nonlocal found
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
        if owner.value == proc.pid:
            name = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(window, name, len(name))
            if name.value == class_name:
                found = window
                return False
        return True

    end = time.monotonic() + 10
    while found is None and time.monotonic() < end and proc.poll() is None:
        user32.EnumWindows(visit, 0)
        if found is None:
            time.sleep(0.05)
    if found is None:
        raise RuntimeError(f"could not find {class_name}, process status {proc.poll()}")
    return found


def send(window, message, parameter=0):
    result = ctypes.c_size_t()
    if not user32.SendMessageTimeoutW(window, message, parameter, 0, 2, 5000, ctypes.byref(result)):
        raise RuntimeError(f"message 0x{message:x} timed out")


def a_value(path):
    line = next(line for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith("a_value="))
    return float(line.partition("=")[2])


def main(binary):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        exe = root / "shuyi.exe"
        shutil.copy2(binary, exe)
        ini = root / "shuyi-ui.ini"
        env = os.environ.copy()
        env.pop("SHUYI_ENGINE", None)
        env["SHUYI_UI_INI"] = str(ini)
        with subprocess.Popen([exe], cwd=root, env=env) as proc:
            try:
                window = wait_window(proc, "ShuyiNativeWindow")
                send(window, 0x0111, 5010)  # WM_COMMAND: open Settings
                dialog = wait_window(proc, "ShuyiSettingsWindow")
                send(dialog, 0x0111, 6001)  # Apply the fresh default
                assert a_value(ini) == 0.5
                send(window, 0x0100, 0xDB)  # [
                assert a_value(ini) == 0.0
                send(window, 0x0111, 3002)  # A -0.5 menu
                assert a_value(ini) == -0.5
                send(window, 0x0100, 0xDD)  # ]
                assert a_value(ini) == 0.0
                send(window, 0x0111, 3001)  # A +0.5 menu
                assert a_value(ini) == 0.5
                send(window, 0x0111, 3005)  # A = +2 preset
                assert a_value(ini) == 2.0
                send(window, 0x0111, 3006)  # Restore defaults
                assert a_value(ini) == 0.5
                send(window, 0x0010)  # WM_CLOSE
                assert proc.wait(timeout=5) == 0
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
