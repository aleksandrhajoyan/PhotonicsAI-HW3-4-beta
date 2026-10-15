"""Hardware-free smoke test: can Python locate and load the iRAYPLE MV SDK DLL?

Loads MVSDKmd.dll via ctypes only. Does not call any exported SDK function,
enumerate cameras, or touch hardware.
"""

import ctypes
import os
import platform
import struct
import sys
from pathlib import Path

DLL_NAME = "MVSDKmd.dll"
DEFAULT_RUNTIME_DIR = Path(r"C:\Program Files\HuarayTech\MV Viewer\Runtime\x64")


def locate_dll() -> Path:
    """Prefer MV_GENICAM_64 if it holds the DLL, else the known install path."""
    env_dir = os.environ.get("MV_GENICAM_64")
    if env_dir:
        candidate = Path(env_dir) / DLL_NAME
        if candidate.is_file():
            print(f"Using MV_GENICAM_64: {env_dir}")
            return candidate.resolve()
        print(f"MV_GENICAM_64 set but {DLL_NAME} not found there: {env_dir}")
    else:
        print("MV_GENICAM_64 not set")
    print(f"Falling back to known install path: {DEFAULT_RUNTIME_DIR}")
    return (DEFAULT_RUNTIME_DIR / DLL_NAME).resolve()


def main() -> int:
    if sys.platform != "win32":
        print(f"FAIL: Windows required, got platform {sys.platform!r}")
        return 1

    print(f"Python version: {platform.python_version()}")
    bits = struct.calcsize("P") * 8
    print(f"Interpreter architecture: {bits}-bit ({platform.machine()})")
    if bits != 64:
        print("FAIL: 64-bit Python required for the x64 SDK runtime")
        return 1

    dll_path = locate_dll()
    print(f"DLL path: {dll_path}")
    if not dll_path.is_file():
        print("FAIL: DLL not found")
        return 1

    try:
        with os.add_dll_directory(str(dll_path.parent)):
            ctypes.WinDLL(str(dll_path))
    except OSError as exc:
        print(f"FAIL: ctypes.WinDLL could not load DLL: {exc!r}")
        return 1

    print("SDK_DLL_LOAD_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
