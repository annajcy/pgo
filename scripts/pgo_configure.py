#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional


DEFAULT_PROFILE = pathlib.Path("conan/profiles/default")


@dataclass(frozen=True)
class ConfigurePlan:
    preset_name: str
    build_type: str
    output_folder: pathlib.Path
    host_profile: pathlib.Path
    build_profile: pathlib.Path
    conan_command: list[str]
    cmake_command: list[str]


def repo_root_from_script() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def resolve_repo_path(repo_root: pathlib.Path, path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else repo_root / path


def substitute_source_dir(value: str, repo_root: pathlib.Path) -> str:
    return value.replace("${sourceDir}", str(repo_root))


def load_configure_presets(repo_root: pathlib.Path) -> dict[str, dict[str, Any]]:
    presets_path = repo_root / "CMakePresets.json"
    with presets_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    raw_presets = {preset["name"]: preset for preset in data.get("configurePresets", [])}
    resolved: dict[str, dict[str, Any]] = {}

    def resolve(name: str) -> dict[str, Any]:
        if name in resolved:
            return resolved[name]
        if name not in raw_presets:
            raise ValueError(f"unknown configure preset: {name}")

        preset = raw_presets[name]
        merged: dict[str, Any] = {}
        inherited = preset.get("inherits")
        inherited_names: list[str]
        if inherited is None:
            inherited_names = []
        elif isinstance(inherited, list):
            inherited_names = inherited
        else:
            inherited_names = [inherited]

        for parent in inherited_names:
            parent_preset = resolve(parent)
            merged.update(parent_preset)
            merged["cacheVariables"] = dict(parent_preset.get("cacheVariables", {}))

        for key, value in preset.items():
            if key == "cacheVariables":
                cache_variables = dict(merged.get("cacheVariables", {}))
                cache_variables.update(value)
                merged[key] = cache_variables
            else:
                merged[key] = value

        resolved[name] = merged
        return merged

    return {name: resolve(name) for name in raw_presets}


def output_folder_from_toolchain(repo_root: pathlib.Path, toolchain_file: str) -> pathlib.Path:
    toolchain_path = pathlib.Path(substitute_source_dir(toolchain_file, repo_root))
    if toolchain_path.name != "conan_toolchain.cmake":
        raise ValueError(f"CMAKE_TOOLCHAIN_FILE must point to conan_toolchain.cmake: {toolchain_file}")
    return toolchain_path.parent


def create_plan(
    *,
    repo_root: pathlib.Path,
    preset_name: str,
    host_profile: Optional[pathlib.Path],
    build_profile: Optional[pathlib.Path],
    build_missing: bool,
) -> ConfigurePlan:
    presets = load_configure_presets(repo_root)
    if preset_name not in presets:
        visible = sorted(name for name, preset in presets.items() if not preset.get("hidden", False))
        raise ValueError(f"unknown configure preset '{preset_name}'. Available presets: {', '.join(visible)}")

    cache_variables = presets[preset_name].get("cacheVariables", {})
    build_type = cache_variables.get("CMAKE_BUILD_TYPE")
    if not build_type:
        raise ValueError(f"preset '{preset_name}' does not define CMAKE_BUILD_TYPE")

    toolchain_file = cache_variables.get("CMAKE_TOOLCHAIN_FILE")
    if not toolchain_file:
        raise ValueError(f"preset '{preset_name}' does not define CMAKE_TOOLCHAIN_FILE")

    output_folder = output_folder_from_toolchain(repo_root, toolchain_file)
    resolved_host_profile = resolve_repo_path(repo_root, host_profile or DEFAULT_PROFILE)
    resolved_build_profile = resolve_repo_path(repo_root, build_profile or host_profile or DEFAULT_PROFILE)

    conan_command = [
        "conan",
        "install",
        str(repo_root),
        f"--profile:host={resolved_host_profile}",
        f"--profile:build={resolved_build_profile}",
        f"--output-folder={output_folder}",
    ]
    if build_missing:
        conan_command.append("--build=missing")
    conan_command.extend(["-s:h", f"build_type={build_type}"])

    cmake_command = ["cmake", "--preset", preset_name]
    return ConfigurePlan(
        preset_name=preset_name,
        build_type=build_type,
        output_folder=output_folder,
        host_profile=resolved_host_profile,
        build_profile=resolved_build_profile,
        conan_command=conan_command,
        cmake_command=cmake_command,
    )


def print_command(command: list[str]) -> None:
    print(" ".join(shlex.quote(part) for part in command), flush=True)


def run_command(command: list[str], repo_root: pathlib.Path) -> None:
    print_command(command)
    subprocess.run(command, cwd=repo_root, check=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Conan install for a CMake configure preset, then run cmake --preset."
    )
    parser.add_argument("preset", help="CMake configure preset name, for example debug or release")
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
        "--dry-run",
        action="store_true",
        help="Print the conan and cmake commands without running them",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.profile and (args.host_profile or args.build_profile):
        print("error: use either --profile or --host-profile/--build-profile, not both", file=sys.stderr)
        return 2

    repo_root = repo_root_from_script()
    host_profile = args.host_profile or args.profile
    build_profile = args.build_profile or args.profile

    try:
        plan = create_plan(
            repo_root=repo_root,
            preset_name=args.preset,
            host_profile=host_profile,
            build_profile=build_profile,
            build_missing=not args.no_build_missing,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        print_command(plan.conan_command)
        print_command(plan.cmake_command)
        return 0

    try:
        run_command(plan.conan_command, repo_root)
        run_command(plan.cmake_command, repo_root)
    except subprocess.CalledProcessError as error:
        return error.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
