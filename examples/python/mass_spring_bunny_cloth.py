#!/usr/bin/env python3
"""Bunny OBJ mass-spring simulation using the pgo Python API.

Loads a surface mesh from an OBJ file, optionally pins selected vertices,
steps the world, and writes OBJ frames.
"""

from __future__ import annotations

import argparse

import numpy as np

from pgo import World


def _parse_pinned_csv(s: str) -> np.ndarray:
    return np.array([int(x) for x in s.split(",")], dtype=np.uint64)


def _parse_pinned_file(path: str) -> np.ndarray:
    with open(path, encoding="utf-8") as f:
        values = [int(tok) for line in f for tok in line.split()]
    return np.array(values, dtype=np.uint64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mass-spring simulation from OBJ surface mesh"
    )
    parser.add_argument(
        "--input", type=str, default="assets/model/bunny.obj",
        help="OBJ input file path"
    )
    parser.add_argument(
        "--output", type=str,
        default="output/example/python/mass_spring/bunny",
        help="Output directory for OBJ frames"
    )
    parser.add_argument(
        "--export-abc", type=str, default=None,
        help="Alembic .abc output file path"
    )
    parser.add_argument(
        "--abc-fps", type=float, default=24.0,
        help="FPS for Alembic export (default 24)"
    )
    parser.add_argument(
        "--frames", type=int, default=10,
        help="Number of frames"
    )
    parser.add_argument(
        "--stiffness", type=float, default=200000.0,
        help="Spring stiffness"
    )
    parser.add_argument(
        "--gravity", type=float, default=9.81,
        help="Gravity magnitude"
    )
    parser.add_argument(
        "--dt", type=float, default=0.001,
        help="Timestep size"
    )
    parser.add_argument(
        "--pinned", type=str, default=None,
        help="Comma-separated vertex indices to pin"
    )
    parser.add_argument(
        "--pinned-file", type=str, default=None,
        help="File of whitespace/newline-separated vertex indices to pin"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=500,
        help="Max Newton iterations per step"
    )
    args = parser.parse_args()

    if args.pinned is not None and args.pinned_file is not None:
        parser.error("cannot use both --pinned and --pinned-file")

    pinned: np.ndarray | None = None
    if args.pinned is not None:
        pinned = _parse_pinned_csv(args.pinned)
    elif args.pinned_file is not None:
        pinned = _parse_pinned_file(args.pinned_file)

    world = World.from_obj(
        args.input,
        stiffness=args.stiffness,
        gravity=args.gravity,
        dt=args.dt,
        pinned_vertices=pinned,
    )

    for frame in range(args.frames):
        result = world.step(max_iterations=args.max_iterations)
        world.write_obj_frame(args.output)
        if args.export_abc is not None:
            world.write_abc_frame(args.export_abc, args.abc_fps)
        print(
            f"frame={frame} status={result.status} "
            f"iterations={result.iterations} "
            f"value={result.final_value:g} "
            f"grad={result.final_gradient_norm:g}"
        )


if __name__ == "__main__":
    main()
