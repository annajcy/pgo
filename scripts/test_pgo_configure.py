#!/usr/bin/env python3

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pgo_configure

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class PgoConfigureTest(unittest.TestCase):
    def test_debug_preset_maps_to_debug_conan_folder(self) -> None:
        plan = pgo_configure.create_plan(
            repo_root=REPO_ROOT,
            preset_name="debug",
            host_profile=None,
            build_profile=None,
            build_missing=True,
        )

        self.assertEqual(plan.preset_name, "debug")
        self.assertEqual(plan.build_type, "Debug")
        self.assertEqual(plan.output_folder, REPO_ROOT / "build/conan/debug")
        self.assertEqual(plan.host_profile, REPO_ROOT / "conan/profiles/default")
        self.assertEqual(plan.build_profile, REPO_ROOT / "conan/profiles/default")
        self.assertEqual(plan.cmake_command, ["cmake", "--preset", "debug"])
        self.assertIn("--build=missing", plan.conan_command)

    def test_asan_reuses_debug_conan_folder(self) -> None:
        plan = pgo_configure.create_plan(
            repo_root=REPO_ROOT,
            preset_name="debug-asan",
            host_profile=None,
            build_profile=None,
            build_missing=True,
        )

        self.assertEqual(plan.build_type, "Debug")
        self.assertEqual(plan.output_folder, REPO_ROOT / "build/conan/debug")

    def test_profile_override_applies_to_host_and_build(self) -> None:
        plan = pgo_configure.create_plan(
            repo_root=REPO_ROOT,
            preset_name="debug-accel",
            host_profile=pathlib.Path("conan/profiles/macos-arm64-apple-clang"),
            build_profile=None,
            build_missing=False,
        )

        expected_profile = REPO_ROOT / "conan/profiles/macos-arm64-apple-clang"
        self.assertEqual(plan.host_profile, expected_profile)
        self.assertEqual(plan.build_profile, expected_profile)
        self.assertEqual(plan.output_folder, REPO_ROOT / "build/conan/debug-accel")
        self.assertNotIn("--build=missing", plan.conan_command)


if __name__ == "__main__":
    unittest.main()
