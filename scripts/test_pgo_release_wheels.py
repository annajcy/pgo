#!/usr/bin/env python3

import pathlib
import sys
import tempfile
import tomllib
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pgo_release_wheels

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class PgoReleaseWheelsTest(unittest.TestCase):
    def test_default_variant_maps_to_pgo_release_all(self) -> None:
        plan = pgo_release_wheels.create_release_plan(
            repo_root=REPO_ROOT,
            variant_name="default",
            source_tree=pathlib.Path("/tmp/pgo-release-src"),
            out_root=None,
            clear=True,
            stable_abi=False,
        )

        self.assertEqual(plan.variant.distribution, "pgo")
        self.assertEqual(plan.variant.preset, "pypgo-release-all")
        self.assertEqual(plan.out_dir, REPO_ROOT / "dist/pgo")

    def test_accel_variant_maps_to_pgo_accel_release_accel_all(self) -> None:
        plan = pgo_release_wheels.create_release_plan(
            repo_root=REPO_ROOT,
            variant_name="accel",
            source_tree=pathlib.Path("/tmp/pgo-release-src"),
            out_root=None,
            clear=True,
            stable_abi=False,
        )

        self.assertEqual(plan.variant.distribution, "pgo-accel")
        self.assertEqual(plan.variant.preset, "pypgo-release-accel-all")
        self.assertEqual(plan.out_dir, REPO_ROOT / "dist/pgo-accel")

    def test_accel_patch_changes_project_name_only_in_temp_pyproject(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pyproject_path = pathlib.Path(temp_dir) / "pyproject.toml"
            pyproject_path.write_text((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8")

            pgo_release_wheels.apply_distribution_name(pyproject_path, "pgo-accel")

            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            self.assertEqual(data["project"]["name"], "pgo-accel")
            self.assertEqual(data["project"]["version"], "0.1.0")

        source_data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(source_data["project"]["name"], "pgo")

    def test_default_variant_does_not_patch_project_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pyproject_path = pathlib.Path(temp_dir) / "pyproject.toml"
            pyproject_path.write_text((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8")

            pgo_release_wheels.apply_variant_metadata(pyproject_path, pgo_release_wheels.VARIANTS["default"])

            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            self.assertEqual(data["project"]["name"], "pgo")

    def test_inner_command_uses_existing_wheel_wrapper(self) -> None:
        source_tree = pathlib.Path("/tmp/pgo-release-src")
        plan = pgo_release_wheels.create_release_plan(
            repo_root=REPO_ROOT,
            variant_name="accel",
            source_tree=source_tree,
            out_root=None,
            clear=True,
            stable_abi=True,
        )

        self.assertEqual(
            plan.inner_command[:4],
            ["uv", "run", "python", "scripts/pgo_build_wheel.py"],
        )
        self.assertIn("pypgo-release-accel-all", plan.inner_command)
        self.assertIn("--out-dir", plan.inner_command)
        self.assertIn(str(REPO_ROOT / "dist/pgo-accel"), plan.inner_command)
        self.assertIn("--clear", plan.inner_command)
        self.assertIn("--stable-abi", plan.inner_command)
        self.assertNotIn("uv build", " ".join(plan.inner_command))

    def test_inner_command_forwards_profile_arguments(self) -> None:
        plan = pgo_release_wheels.create_release_plan(
            repo_root=REPO_ROOT,
            variant_name="default",
            source_tree=pathlib.Path("/tmp/pgo-release-src"),
            out_root=None,
            clear=False,
            stable_abi=False,
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
            variant_name="default",
            source_tree=pathlib.Path("/tmp/pgo-release-src"),
            out_root=None,
            clear=False,
            stable_abi=False,
        )

        self.assertNotIn("--host-profile", plan.inner_command)
        self.assertNotIn("--build-profile", plan.inner_command)
        self.assertNotIn("--profile", plan.inner_command)

    def test_relative_out_root_is_resolved_against_repo_root(self) -> None:
        plan = pgo_release_wheels.create_release_plan(
            repo_root=REPO_ROOT,
            variant_name="default",
            source_tree=pathlib.Path("/tmp/pgo-release-src"),
            out_root=pathlib.Path("wheelhouse"),
            clear=False,
            stable_abi=False,
        )

        self.assertEqual(plan.out_dir, REPO_ROOT / "wheelhouse/pgo")
        self.assertIn(str(REPO_ROOT / "wheelhouse/pgo"), plan.inner_command)


if __name__ == "__main__":
    unittest.main()
