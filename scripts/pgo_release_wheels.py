#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from typing import Optional

import pgo_configure


@dataclass(frozen=True)
class ReleaseVariant:
    name: str
    preset: str
    distribution: str


@dataclass(frozen=True)
class ReleasePlan:
    variant: ReleaseVariant
    source_tree: pathlib.Path
    out_dir: pathlib.Path
    inner_command: list[str]


VARIANTS: dict[str, ReleaseVariant] = {
    "default": ReleaseVariant(
        name="default",
        preset="pypgo-release-all",
        distribution="pgo",
    ),
    "accel": ReleaseVariant(
        name="accel",
        preset="pypgo-release-accel-all",
        distribution="pgo-accel",
    ),
}

COPY_IGNORE = shutil.ignore_patterns(
    ".git",
    ".venv",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "*.egg-info",
    "*.pyc",
)


def apply_distribution_name(pyproject_path: pathlib.Path, distribution: str) -> None:
    original = pyproject_path.read_text(encoding="utf-8")
    data = tomllib.loads(original)
    if data.get("project", {}).get("name") is None:
        raise ValueError(f"{pyproject_path} does not define [project].name")

    lines = original.splitlines(keepends=True)
    in_project = False
    changed = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
            continue
        if in_project and stripped.startswith("[") and stripped.endswith("]"):
            break
        if in_project and stripped.startswith("name"):
            prefix = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f'{prefix}name = "{distribution}"{newline}'
            changed = True
            break

    if not changed:
        raise ValueError(f"{pyproject_path} does not have a replaceable [project].name")

    pyproject_path.write_text("".join(lines), encoding="utf-8")
    patched = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    if patched["project"]["name"] != distribution:
        raise ValueError(f"failed to patch [project].name to {distribution}")


def apply_variant_metadata(pyproject_path: pathlib.Path, variant: ReleaseVariant) -> None:
    if variant.distribution != "pgo":
        apply_distribution_name(pyproject_path, variant.distribution)


def copy_source_tree(repo_root: pathlib.Path, source_tree: pathlib.Path) -> None:
    shutil.copytree(repo_root, source_tree, ignore=COPY_IGNORE)


def create_release_plan(
    *,
    repo_root: pathlib.Path,
    variant_name: str,
    source_tree: pathlib.Path,
    out_root: Optional[pathlib.Path],
    clear: bool,
    stable_abi: bool,
    host_profile: Optional[pathlib.Path] = None,
    build_profile: Optional[pathlib.Path] = None,
) -> ReleasePlan:
    try:
        variant = VARIANTS[variant_name]
    except KeyError as error:
        raise ValueError(f"unknown release variant: {variant_name}") from error

    if out_root is None:
        resolved_out_root = repo_root / "dist"
    elif out_root.is_absolute():
        resolved_out_root = out_root
    else:
        resolved_out_root = repo_root / out_root
    out_dir = resolved_out_root / variant.distribution
    inner_command = [
        "uv",
        "run",
        "python",
        "scripts/pgo_build_wheel.py",
        variant.preset,
        "--out-dir",
        str(out_dir),
    ]
    if host_profile is not None:
        inner_command.extend(["--host-profile", str(host_profile)])
    if build_profile is not None:
        inner_command.extend(["--build-profile", str(build_profile)])
    if clear:
        inner_command.append("--clear")
    if stable_abi:
        inner_command.append("--stable-abi")

    return ReleasePlan(
        variant=variant,
        source_tree=source_tree,
        out_dir=out_dir,
        inner_command=inner_command,
    )


def find_built_wheels(out_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(out_dir.glob("*.whl"), key=lambda path: path.stat().st_mtime)


def verify_wheel_metadata(wheel_path: pathlib.Path, distribution: str) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError(f"{wheel_path} must contain exactly one METADATA file")

        metadata = Parser().parsestr(wheel.read(metadata_names[0]).decode("utf-8"))
        actual_name = metadata.get("Name")
        if actual_name != distribution:
            raise ValueError(f"{wheel_path} has Name: {actual_name}, expected {distribution}")
        if not any(name == "pgo/__init__.py" or name.startswith("pgo/") for name in names):
            raise ValueError(f"{wheel_path} does not contain the pgo import package")


def run_release_plan(plan: ReleasePlan, repo_root: pathlib.Path) -> None:
    copy_source_tree(repo_root, plan.source_tree)
    apply_variant_metadata(plan.source_tree / "pyproject.toml", plan.variant)

    before = set(find_built_wheels(plan.out_dir))
    pgo_configure.run_command(plan.inner_command, plan.source_tree)
    after = set(find_built_wheels(plan.out_dir))
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    wheels_to_check = created or find_built_wheels(plan.out_dir)[-1:]
    if not wheels_to_check:
        raise ValueError(f"no wheel was written to {plan.out_dir}")
    for wheel_path in wheels_to_check:
        verify_wheel_metadata(wheel_path, plan.variant.distribution)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build release wheel distributions from temporary source trees."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        help="Release wheel variant to build",
    )
    group.add_argument(
        "--all",
        dest="all_variants",
        action="store_true",
        help="Build all release wheel variants",
    )
    parser.add_argument(
        "--out-root",
        type=pathlib.Path,
        help="Output root; defaults to dist",
    )
    parser.add_argument(
        "--profile",
        type=pathlib.Path,
        help="Use the same Conan profile for host and build contexts",
    )
    parser.add_argument(
        "--host-profile",
        type=pathlib.Path,
        help="Conan host profile (forwarded to pgo_build_wheel.py)",
    )
    parser.add_argument(
        "--build-profile",
        type=pathlib.Path,
        help="Conan build profile (forwarded to pgo_build_wheel.py)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Pass --clear to the inner wheel build",
    )
    parser.add_argument(
        "--stable-abi",
        action="store_true",
        help="Pass --stable-abi to the inner wheel build",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print release plans without copying sources or building wheels",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary source trees after building",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.profile and (args.host_profile or args.build_profile):
        print("error: use either --profile or --host-profile/--build-profile, not both", file=sys.stderr)
        return 2

    host_profile = args.host_profile or args.profile
    build_profile = args.build_profile or args.profile

    repo_root = pgo_configure.repo_root_from_script()
    variant_names = list(VARIANTS) if args.all_variants else [args.variant]

    if args.dry_run:
        for variant_name in variant_names:
            plan = create_release_plan(
                repo_root=repo_root,
                variant_name=variant_name,
                source_tree=pathlib.Path("<temporary-source-tree>"),
                out_root=args.out_root,
                clear=args.clear,
                stable_abi=args.stable_abi,
                host_profile=host_profile,
                build_profile=build_profile,
            )
            print(f"variant: {plan.variant.name}", flush=True)
            print(f"distribution: {plan.variant.distribution}", flush=True)
            print(f"preset: {plan.variant.preset}", flush=True)
            print(f"out-dir: {plan.out_dir}", flush=True)
            pgo_configure.print_command(plan.inner_command)
        return 0

    try:
        for variant_name in variant_names:
            with tempfile.TemporaryDirectory(prefix=f"pgo-release-{variant_name}-") as temp_dir:
                source_tree = pathlib.Path(temp_dir) / "src"
                plan = create_release_plan(
                    repo_root=repo_root,
                    variant_name=variant_name,
                    source_tree=source_tree,
                    out_root=args.out_root,
                    clear=args.clear,
                    stable_abi=args.stable_abi,
                    host_profile=host_profile,
                    build_profile=build_profile,
                )
                print(f"Building {plan.variant.distribution} from {plan.variant.preset}", flush=True)
                run_release_plan(plan, repo_root)
                if args.keep_temp:
                    kept_tree = repo_root / "build/release-sources" / variant_name
                    if kept_tree.exists():
                        shutil.rmtree(kept_tree)
                    kept_tree.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source_tree, kept_tree, ignore=COPY_IGNORE)
                    print(f"Kept temporary source tree at {kept_tree}", flush=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            return error.returncode
        return 1

    return 0


def main_cli() -> None:
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    main_cli()
