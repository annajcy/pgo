#!/usr/bin/env python3

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pgo_release_wheels

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class PgoReleaseWheelsTest(unittest.TestCase):
    def test_default_plan_builds_single_pgo_distribution(self) -> None:
        plan = pgo_release_wheels.create_release_plan(
            repo_root=REPO_ROOT,
            preset_name="pypgo-release-all",
            source_tree=pathlib.Path("/tmp/pgo-release-src"),
            out_root=None,
            clear=True,
            stable_abi=False,
            install=False,
        )

        self.assertEqual(plan.preset, "pypgo-release-all")
        self.assertEqual(plan.distribution, "pgo")
        self.assertEqual(plan.out_dir, REPO_ROOT / "dist/pgo")

    def test_accel_preset_still_builds_pgo_distribution(self) -> None:
        plan = pgo_release_wheels.create_release_plan(
            repo_root=REPO_ROOT,
            preset_name="pypgo-release-accel-all",
            source_tree=pathlib.Path("/tmp/pgo-release-src"),
            out_root=None,
            clear=True,
            stable_abi=False,
            install=False,
        )

        self.assertEqual(plan.preset, "pypgo-release-accel-all")
        self.assertEqual(plan.distribution, "pgo")
        self.assertEqual(plan.out_dir, REPO_ROOT / "dist/pgo")

    def test_inner_command_uses_existing_wheel_wrapper(self) -> None:
        source_tree = pathlib.Path("/tmp/pgo-release-src")
        plan = pgo_release_wheels.create_release_plan(
            repo_root=REPO_ROOT,
            preset_name="pypgo-release-accel-all",
            source_tree=source_tree,
            out_root=None,
            clear=True,
            stable_abi=True,
            install=False,
        )

        self.assertEqual(
            plan.inner_command[:4],
            ["uv", "run", "python", "scripts/pgo_build_wheel.py"],
        )
        self.assertIn("pypgo-release-accel-all", plan.inner_command)
        self.assertIn("--out-dir", plan.inner_command)
        self.assertIn(str(REPO_ROOT / "dist/pgo"), plan.inner_command)
        self.assertIn("--clear", plan.inner_command)
        self.assertIn("--stable-abi", plan.inner_command)
        self.assertNotIn("uv build", " ".join(plan.inner_command))

    def test_inner_command_forwards_profile_arguments(self) -> None:
        plan = pgo_release_wheels.create_release_plan(
            repo_root=REPO_ROOT,
            preset_name="pypgo-release-all",
            source_tree=pathlib.Path("/tmp/pgo-release-src"),
            out_root=None,
            clear=False,
            stable_abi=False,
            install=False,
            host_profile=pathlib.Path("conan/profiles/ubuntu-x86_64-gcc"),
            build_profile=pathlib.Path("conan/profiles/ubuntu-x86_64-gcc"),
        )

        self.assertIn("--host-profile", plan.inner_command)
        self.assertIn("--build-profile", plan.inner_command)
        host_index = plan.inner_command.index("--host-profile")
        build_index = plan.inner_command.index("--build-profile")
        self.assertEqual(plan.inner_command[host_index + 1], "conan/profiles/ubuntu-x86_64-gcc")
        self.assertEqual(plan.inner_command[build_index + 1], "conan/profiles/ubuntu-x86_64-gcc")

    def test_inner_command_omits_profile_when_not_provided(self) -> None:
        plan = pgo_release_wheels.create_release_plan(
            repo_root=REPO_ROOT,
            preset_name="pypgo-release-all",
            source_tree=pathlib.Path("/tmp/pgo-release-src"),
            out_root=None,
            clear=False,
            stable_abi=False,
            install=False,
        )

        self.assertNotIn("--host-profile", plan.inner_command)
        self.assertNotIn("--build-profile", plan.inner_command)
        self.assertNotIn("--profile", plan.inner_command)

    def test_relative_out_root_is_resolved_against_repo_root(self) -> None:
        plan = pgo_release_wheels.create_release_plan(
            repo_root=REPO_ROOT,
            preset_name="pypgo-release-all",
            source_tree=pathlib.Path("/tmp/pgo-release-src"),
            out_root=pathlib.Path("wheelhouse"),
            clear=False,
            stable_abi=False,
            install=False,
        )

        self.assertEqual(plan.out_dir, REPO_ROOT / "wheelhouse/pgo")
        self.assertIn(str(REPO_ROOT / "wheelhouse/pgo"), plan.inner_command)

    def test_install_flag_builds_uv_pip_install_command_for_built_wheel(self) -> None:
        plan = pgo_release_wheels.create_release_plan(
            repo_root=REPO_ROOT,
            preset_name="pypgo-release-all",
            source_tree=pathlib.Path("/tmp/pgo-release-src"),
            out_root=None,
            clear=False,
            stable_abi=False,
            install=True,
        )
        wheel_path = REPO_ROOT / "dist/pgo/pgo-0.1.0-py3-none-any.whl"

        self.assertEqual(
            pgo_release_wheels.create_install_command(plan, wheel_path),
            ["uv", "pip", "install", str(wheel_path)],
        )

    def test_run_release_plan_installs_created_wheel_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_tree = pathlib.Path(temp_dir) / "src"
            out_dir = pathlib.Path(temp_dir) / "dist"
            wheel_path = out_dir / "pgo-0.1.0-py3-none-any.whl"
            plan = pgo_release_wheels.ReleasePlan(
                preset="pypgo-release-all",
                distribution="pgo",
                source_tree=source_tree,
                out_dir=out_dir,
                inner_command=["uv", "run", "python", "scripts/pgo_build_wheel.py"],
                install=True,
            )

            def write_wheel(*_args: object, **_kwargs: object) -> None:
                out_dir.mkdir(parents=True, exist_ok=True)
                with pgo_release_wheels.zipfile.ZipFile(wheel_path, "w") as wheel:
                    wheel.writestr("pgo/__init__.py", "")
                    wheel.writestr("pgo-0.1.0.dist-info/METADATA", "Name: pgo\n")

            with mock.patch.object(pgo_release_wheels.pgo_configure, "run_command") as run_command:
                run_command.side_effect = [write_wheel(), None]

                pgo_release_wheels.run_release_plan(plan, REPO_ROOT)

            self.assertEqual(run_command.call_args_list[-1].args[0], ["uv", "pip", "install", str(wheel_path)])


if __name__ == "__main__":
    unittest.main()
