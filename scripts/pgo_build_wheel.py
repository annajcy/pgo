#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import pgo_configure


@dataclass(frozen=True)
class WheelBuildPlan:
    configure_plan: pgo_configure.ConfigurePlan
    uv_build_command: list[str]


def _cmake_define_arg(name: str, value: object) -> str:
    return f"-Ccmake.define.{name}={value}"


def _relative_or_absolute(repo_root: pathlib.Path, path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _resolved_pgo_cache_variables(
    *,
    repo_root: pathlib.Path,
    preset_name: str,
    stable_abi: bool,
) -> dict[str, object]:
    presets = pgo_configure.load_configure_presets(repo_root)
    if preset_name not in presets:
        visible = sorted(name for name, preset in presets.items() if not preset.get("hidden", False))
        raise ValueError(f"unknown configure preset '{preset_name}'. Available presets: {', '.join(visible)}")

    cache_variables = dict(presets[preset_name].get("cacheVariables", {}))
    pgo_variables = {
        name: value for name, value in cache_variables.items() if name.startswith("PGO_")
    }
    if pgo_variables.get("PGO_BUILD_PYTHON") != "ON":
        raise ValueError(f"preset '{preset_name}' is not a Python package preset")
    if stable_abi:
        pgo_variables["PGO_ENABLE_PYTHON_STABLE_ABI"] = "ON"
    return pgo_variables


def create_wheel_plan(
    *,
    repo_root: pathlib.Path,
    preset_name: str,
    host_profile: Optional[pathlib.Path],
    build_profile: Optional[pathlib.Path],
    build_missing: bool,
    out_dir: Optional[pathlib.Path],
    clear: bool,
    stable_abi: bool,
) -> WheelBuildPlan:
    configure_plan = pgo_configure.create_plan(
        repo_root=repo_root,
        preset_name=preset_name,
        host_profile=host_profile,
        build_profile=build_profile,
        build_missing=build_missing,
    )
    pgo_variables = _resolved_pgo_cache_variables(
        repo_root=repo_root,
        preset_name=preset_name,
        stable_abi=stable_abi,
    )

    resolved_out_dir = out_dir or pathlib.Path("dist") / preset_name
    toolchain_file = configure_plan.output_folder / "conan_toolchain.cmake"
    uv_build_command = [
        "uv",
        "build",
        "--wheel",
        "--out-dir",
        _relative_or_absolute(repo_root, resolved_out_dir),
    ]
    if clear:
        uv_build_command.append("--clear")
    if stable_abi:
        uv_build_command.append("-Cwheel.py-api=cp312")

    uv_build_command.extend(
        [
            f"-Ccmake.build-type={configure_plan.build_type}",
            f"-Ccmake.args=-DCMAKE_TOOLCHAIN_FILE={_relative_or_absolute(repo_root, toolchain_file)}",
        ]
    )
    for name in sorted(pgo_variables):
        uv_build_command.append(_cmake_define_arg(name, pgo_variables[name]))

    return WheelBuildPlan(
        configure_plan=configure_plan,
        uv_build_command=uv_build_command,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Python package preset, then build a wheel with matching scikit-build defines."
    )
    parser.add_argument(
        "preset",
        help="Python package CMake configure preset, for example pypgo-release-all",
    )
    parser.add_argument(
        "--profile",
        type=pathlib.Path,
        help="Use the same Conan profile for host and build contexts",
    )
    parser.add_argument("--host-profile", type=pathlib.Path, help="Conan host profile")
    parser.add_argument("--build-profile", type=pathlib.Path, help="Conan build profile")
    parser.add_argument(
        "--no-build-missing",
        action="store_true",
        help="Do not pass --build=missing to conan install",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        help="Wheel output directory; defaults to dist/<preset>",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Pass --clear to uv build",
    )
    parser.add_argument(
        "--stable-abi",
        action="store_true",
        help="Build an abi3 wheel by enabling PGO_ENABLE_PYTHON_STABLE_ABI and wheel.py-api=cp312",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the conan, cmake, and uv build commands without running them",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.profile and (args.host_profile or args.build_profile):
        print("error: use either --profile or --host-profile/--build-profile, not both", file=sys.stderr)
        return 2

    repo_root = pgo_configure.repo_root_from_script()
    host_profile = args.host_profile or args.profile
    build_profile = args.build_profile or args.profile
    build_missing = not args.no_build_missing

    try:
        plan = create_wheel_plan(
            repo_root=repo_root,
            preset_name=args.preset,
            host_profile=host_profile,
            build_profile=build_profile,
            build_missing=build_missing,
            out_dir=args.out_dir,
            clear=args.clear,
            stable_abi=args.stable_abi,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        pgo_configure.print_command(plan.configure_plan.conan_command)
        pgo_configure.print_command(plan.configure_plan.cmake_command)
        pgo_configure.print_command(plan.uv_build_command)
        return 0

    try:
        pgo_configure.run_command(plan.configure_plan.conan_command, repo_root)
        pgo_configure.run_command(plan.configure_plan.cmake_command, repo_root)
        pgo_configure.run_command(plan.uv_build_command, repo_root)
    except subprocess.CalledProcessError as error:
        return error.returncode

    return 0


def main_cli() -> None:
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    main_cli()
