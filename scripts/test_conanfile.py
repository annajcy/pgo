#!/usr/bin/env python3

import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

conan_module = types.ModuleType("conan")
conan_module.ConanFile = object
sys.modules["conan"] = conan_module

conan_tools_module = types.ModuleType("conan.tools")
conan_tools_cmake_module = types.ModuleType("conan.tools.cmake")
conan_tools_cmake_module.CMakeDeps = object
conan_tools_cmake_module.CMakeToolchain = object
sys.modules["conan.tools"] = conan_tools_module
sys.modules["conan.tools.cmake"] = conan_tools_cmake_module

from conanfile import PgoRecipe


class PgoRecipeTest(unittest.TestCase):
    def test_tbb_dependency_builds_hwloc_as_shared_library(self) -> None:
        self.assertIs(PgoRecipe.default_options.get("hwloc/*:shared"), True)


if __name__ == "__main__":
    unittest.main()
