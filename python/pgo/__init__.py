import os
import pathlib
import sys

_pgo_dll_directory = None
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _pgo_dll_directory = os.add_dll_directory(str(pathlib.Path(__file__).resolve().parent))

from .world import StepResult, World

__all__ = ["StepResult", "World"]
