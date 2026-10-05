#!/usr/bin/env python
"""
Windows build script for ArchaeoSight Desktop (PyQt6 + PyInstaller).

Run with the SAME interpreter that has the project dependencies installed
(e.g. the project's conda env), so the bundle matches the runtime:

    python build_windows.py              # one-folder windowed build (recommended)
    python build_windows.py --onefile    # single .exe (large, slower first start)
    python build_windows.py --console    # keep a console window (debugging)
    python build_windows.py --clean      # wipe build/, dist/ and the .spec first
    python build_windows.py --icon app.ico

Output:
    one-folder : dist/ArchaeoSight/ArchaeoSight.exe
    --onefile  : dist/ArchaeoSight.exe

Notes:
- This is a heavy scientific stack (TensorFlow, scikit-learn, hdbscan, pykrige,
  onnx). One-folder builds are recommended; one-file builds with TensorFlow are
  very large and slow to start, and can fail to unpack on some machines.
- The tricky packages below ship compiled extensions and/or data files that
  PyInstaller's stock hooks do not fully resolve, so they are collected in full.
- PyQt6 bundles an old MSVC runtime (msvcp140.dll 14.26) that breaks
  TensorFlow if it gets loaded. After building, every copy of the MSVC runtime
  in the bundle is replaced with the newest one available (see
  _unify_vc_runtime), and the runtime hook preloads that copy at startup.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "ArchaeoSight"
ENTRY = "main.py"
ROOT = Path(__file__).resolve().parent

# Packages collected in full (submodules + data + compiled binaries). These
# either have compiled .pyd extensions (hdbscan, pykrige, scipy via deps) or
# load data files at runtime (tensorflow, onnx, matplotlib).
COLLECT_ALL = [
    "tensorflow",
    "tensorflow_intel",  # Windows TF sometimes splits into this package
    "hdbscan",
    "pykrige",
    "onnx",
    "onnxruntime",
    "skl2onnx",
    "tf2onnx",
    "sklearn",
    "matplotlib",
]

# Cheaper submodule-only sweeps for packages whose stock hooks are reliable but
# which have dynamically imported submodules.
COLLECT_SUBMODULES = [
    "scipy",
    "openpyxl",
]

# Explicit hidden imports for modules pulled in only via pandas' optional engines.
HIDDEN_IMPORTS = [
    "openpyxl",
    "xlrd",
]

# MSVC C++ runtime DLLs. PyInstaller collects several copies: conda's at the
# top of the bundle plus the stale 14.26 ones PyQt6-Qt6 ships in
# PyQt6/Qt6/bin. Only one msvcp140.dll can be loaded per process, and if Qt's
# copy wins, TensorFlow fails to initialize ("DLL load failed while importing
# _pywrap_tensorflow_internal: A dynamic link library (DLL) initialization
# routine failed"). Delvewheel-renamed copies (e.g. ml_dtypes.libs/
# msvcp140-<hash>.dll) have their own module name and are left alone.
VC_RUNTIME_DLLS = (
    "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll",
    "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll", "concrt140.dll",
)
# Where to look for up-to-date copies; on a version tie the earlier one wins.
VC_RUNTIME_SOURCES = [
    Path(sys.prefix),
    Path(sys.prefix) / "Library" / "bin",
    Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
]


def _installed(pkg: str) -> bool:
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError):
        return False


def _filter_installed(pkgs: list[str], kind: str) -> list[str]:
    """Drop packages that are not importable so PyInstaller does not error out."""
    present, missing = [], []
    for p in pkgs:
        (present if _installed(p) else missing).append(p)
    if missing:
        print(f"  note: skipping {kind} for not-installed packages: {', '.join(missing)}")
    return present


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _file_version(path: Path) -> tuple[int, ...]:
    """Windows file version of a DLL, e.g. (14, 51, 36247, 0); (0,) if unknown."""
    import pefile  # PyInstaller dependency on Windows

    pe = pefile.PE(str(path), fast_load=True)
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]])
        ffi = pe.VS_FIXEDFILEINFO[0]
        return (ffi.FileVersionMS >> 16, ffi.FileVersionMS & 0xFFFF,
                ffi.FileVersionLS >> 16, ffi.FileVersionLS & 0xFFFF)
    except (AttributeError, IndexError):
        return (0,)
    finally:
        pe.close()


def _unify_vc_runtime(bundle: Path) -> None:
    """Make every MSVC runtime DLL in the bundle the newest available version.

    `bundle` is the directory that becomes sys._MEIPASS (the one holding
    python3*.dll). Stale copies anywhere in the tree are overwritten, and any
    missing top-level copies (which the runtime hook preloads) are added.
    """
    newest: dict[str, Path] = {}
    for name in VC_RUNTIME_DLLS:
        cands = [d / name for d in VC_RUNTIME_SOURCES if (d / name).is_file()]
        if cands:
            newest[name] = max(cands, key=_file_version)
    if not newest:
        print("  warning: no MSVC runtime DLLs found to bundle")
        return

    def fmt(v):
        return ".".join(map(str, v))

    for path in sorted(bundle.rglob("*.dll")):
        src = newest.get(path.name.lower())
        if src is None:
            continue
        old, new = _file_version(path), _file_version(src)
        if old < new:
            shutil.copy2(src, path)
            print(f"  updated {path.relative_to(bundle)}: {fmt(old)} -> {fmt(new)}")
    for name, src in newest.items():
        if not (bundle / name).exists():
            shutil.copy2(src, bundle / name)
            print(f"  added {name} ({fmt(_file_version(src))})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build ArchaeoSight Desktop with PyInstaller.")
    ap.add_argument("--onefile", action="store_true",
                    help="Bundle into a single .exe (large, slower first start).")
    ap.add_argument("--console", action="store_true",
                    help="Keep a console window open (useful for debugging).")
    ap.add_argument("--flat", action="store_true",
                    help="Put DLLs/libs directly beside the .exe instead of in an "
                         "_internal/ subfolder (onedir only).")
    ap.add_argument("--clean", action="store_true",
                    help="Remove build/, dist/ and the .spec before building.")
    ap.add_argument("--icon", default=None,
                    help="Path to a .ico file to use as the executable icon.")
    args = ap.parse_args()

    entry = ROOT / ENTRY
    if not entry.exists():
        print(f"ERROR: entry point not found: {entry}", file=sys.stderr)
        return 1

    if not _installed("PyInstaller"):
        print("ERROR: PyInstaller is not installed in this interpreter.\n"
              "Install dependencies first:  pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    if args.icon and not Path(args.icon).exists():
        print(f"ERROR: icon file not found: {args.icon}", file=sys.stderr)
        return 1

    if args.clean:
        for d in ("build", "dist"):
            shutil.rmtree(ROOT / d, ignore_errors=True)
        spec = ROOT / f"{APP_NAME}.spec"
        if spec.exists():
            spec.unlink()
        print("Cleaned build/, dist/, and .spec")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", APP_NAME,
        "--windowed" if not args.console else "--console",
        "--onefile" if args.onefile else "--onedir",
    ]
    if args.icon:
        cmd += ["--icon", str(Path(args.icon).resolve())]
    if args.flat and not args.onefile:
        # "." restores the pre-6.0 layout: supporting files sit next to the exe.
        cmd += ["--contents-directory", "."]

    # Runtime hook: preload the bundled MSVC runtime before PyQt6 can pull in
    # its stale copy, and put TensorFlow/onnxruntime package dirs on the DLL
    # search path so their lazily-loaded native DLLs initialize.
    rth = ROOT / "pyi_rth_native_dlls.py"
    if rth.exists():
        cmd += ["--runtime-hook", str(rth)]

    for pkg in _filter_installed(COLLECT_ALL, "collect-all"):
        cmd += ["--collect-all", pkg]
    for pkg in _filter_installed(COLLECT_SUBMODULES, "collect-submodules"):
        cmd += ["--collect-submodules", pkg]
    for mod in _filter_installed(HIDDEN_IMPORTS, "hidden-import"):
        cmd += ["--hidden-import", mod]

    cmd.append(str(entry))

    run(cmd)

    # One-file builds can't be patched after the fact; there the runtime
    # hook's preload of the top-level copies is the only safeguard.
    if not args.onefile:
        bundle = ROOT / "dist" / APP_NAME
        if not args.flat:
            bundle /= "_internal"
        print("\nUnifying MSVC runtime DLLs in the bundle...")
        _unify_vc_runtime(bundle)

    out =(ROOT / "dist" / f"{APP_NAME}.exe") if args.onefile \
        else (ROOT / "dist" / APP_NAME / f"{APP_NAME}.exe")
    print("\nBuild complete.")
    print(f"Executable: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
