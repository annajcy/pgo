#!/usr/bin/env python3

import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class CMakeConfigTest(unittest.TestCase):
    def test_auto_mkl_probe_is_quiet_so_missing_mkl_can_fallback(self) -> None:
        content = (REPO_ROOT / "cmake/pgo_eigen.cmake").read_text(encoding="utf-8")

        self.assertIn("find_package(MKL CONFIG QUIET)", content)
        self.assertIn("PGO_SELECTED_EIGEN_ACCELERATION_BACKEND \"NONE\"", content)

    def test_tbb_implicit_linkage_is_disabled_for_msvc_consumers(self) -> None:
        content = (REPO_ROOT / "src/parallel/CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("__TBB_NO_IMPLICIT_LINKAGE=1", content)

    def test_python_wheel_installs_mkl_runtime_candidates(self) -> None:
        content = (REPO_ROOT / "src/python/CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("mkl_core.2.dll", content)
        self.assertIn("libmkl_core.so.2", content)
        self.assertIn("libiomp5", content)

    def test_python_package_adds_wheel_directory_to_windows_dll_search_path(self) -> None:
        content = (REPO_ROOT / "python/pgo/__init__.py").read_text(encoding="utf-8")

        self.assertIn("os.add_dll_directory", content)


if __name__ == "__main__":
    unittest.main()
