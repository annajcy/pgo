#!/usr/bin/env python3

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pgo_build_wheel

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class PgoBuildWheelTest(unittest.TestCase):
    def test_release_all_forwards_all_preset_defines_to_uv_build(self) -> None:
        plan = pgo_build_wheel.create_wheel_plan(
            repo_root=REPO_ROOT,
            preset_name="pypgo-release-all",
            host_profile=None,
            build_profile=None,
            build_missing=True,
            out_dir=None,
            clear=True,
            stable_abi=False,
        )

        self.assertEqual(plan.configure_plan.preset_name, "pypgo-release-all")
        self.assertIn("-Ccmake.define.PGO_ENABLE_SPDLOG=ON", plan.uv_build_command)
        self.assertIn("-Ccmake.define.PGO_ENABLE_ALEMBIC=ON", plan.uv_build_command)
        self.assertIn("-Ccmake.define.PGO_BUILD_PYTHON=ON", plan.uv_build_command)
        self.assertIn("-Ccmake.define.PGO_BUILD_C_API=ON", plan.uv_build_command)
        self.assertIn("-Ccmake.define.PGO_ENABLE_NATIVE_ARCH=OFF", plan.uv_build_command)
        self.assertNotIn("--preset", plan.uv_build_command)

    def test_accel_all_forwards_acceleration_defines_to_uv_build(self) -> None:
        plan = pgo_build_wheel.create_wheel_plan(
            repo_root=REPO_ROOT,
            preset_name="pypgo-release-accel-all",
            host_profile=None,
            build_profile=None,
            build_missing=True,
            out_dir=None,
            clear=True,
            stable_abi=False,
        )

        self.assertIn("-Ccmake.define.PGO_ENABLE_EIGEN_ACCELERATION=ON", plan.uv_build_command)
        self.assertIn("-Ccmake.define.PGO_EIGEN_ACCELERATION_BACKEND=AUTO", plan.uv_build_command)

    def test_stable_abi_adds_wheel_tag_and_overrides_cmake_define(self) -> None:
        plan = pgo_build_wheel.create_wheel_plan(
            repo_root=REPO_ROOT,
            preset_name="pypgo-release-all",
            host_profile=None,
            build_profile=None,
            build_missing=True,
            out_dir=None,
            clear=False,
            stable_abi=True,
        )

        self.assertIn("-Cwheel.py-api=cp312", plan.uv_build_command)
        self.assertIn("-Ccmake.define.PGO_ENABLE_PYTHON_STABLE_ABI=ON", plan.uv_build_command)
        self.assertNotIn("-Ccmake.define.PGO_ENABLE_PYTHON_STABLE_ABI=OFF", plan.uv_build_command)

    def test_default_out_dir_is_preset_scoped(self) -> None:
        plan = pgo_build_wheel.create_wheel_plan(
            repo_root=REPO_ROOT,
            preset_name="pypgo-release-all",
            host_profile=None,
            build_profile=None,
            build_missing=True,
            out_dir=None,
            clear=False,
            stable_abi=False,
        )

        self.assertIn("--out-dir", plan.uv_build_command)
        self.assertIn("dist/pypgo-release-all", plan.uv_build_command)


if __name__ == "__main__":
    unittest.main()
