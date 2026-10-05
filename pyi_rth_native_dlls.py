# PyInstaller runtime hook.
#
# 1. MSVC C++ runtime. PyQt6-Qt6 ships an old (14.26) msvcp140.dll in
#    PyQt6/Qt6/bin and PyQt6 puts that directory on the DLL search path when it
#    is imported. Only one msvcp140.dll can be loaded per process, and whichever
#    loads first wins. If Qt's stale copy wins, TensorFlow (built against a
#    14.40+ runtime) fails with "DLL load failed while importing
#    _pywrap_tensorflow_internal: A dynamic link library (DLL) initialization
#    routine failed". Custom runtime hooks run before PyInstaller's own hooks
#    and before main.py, so load the up-to-date copies from the top of the
#    bundle by full path here; later by-name requests resolve to them.
#
# 2. TensorFlow and onnxruntime ship native DLLs inside their package
#    directories (e.g. tensorflow/tensorflow_framework.2.dll) that are loaded
#    lazily by the Python extension modules. In a frozen one-folder app the
#    extension's sibling directory is on the DLL search path, but the package's
#    *parent* directory is not, so the delay-loaded framework DLLs can fail to
#    initialize. Adding the package directories explicitly fixes that.
#
# Harmless on non-Windows.
import os
import sys

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    base = getattr(sys, "_MEIPASS", None)
    if base:
        import ctypes

        _k32 = ctypes.WinDLL("kernel32")
        _k32.GetModuleHandleW.restype = ctypes.c_void_p
        _k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        # Dependency order, so each DLL's own imports are already resolved.
        for _name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll",
                      "msvcp140_1.dll", "msvcp140_2.dll", "concrt140.dll"):
            _p = os.path.join(base, _name)
            if os.path.isfile(_p) and not _k32.GetModuleHandleW(_name):
                try:
                    ctypes.WinDLL(_p)
                except OSError:
                    pass

        _subdirs = [
            "tensorflow",
            os.path.join("tensorflow", "python"),
            "onnxruntime",
            os.path.join("onnxruntime", "capi"),
        ]
        for _sub in _subdirs:
            _d = os.path.join(base, _sub)
            if os.path.isdir(_d):
                try:
                    os.add_dll_directory(_d)
                except OSError:
                    pass
