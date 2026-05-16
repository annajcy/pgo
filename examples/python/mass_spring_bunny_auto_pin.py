#!/usr/bin/env python3
"""Bunny OBJ simulation with automatic height-based vertex pinning.

Demonstrates pgo.read_obj to inspect the mesh before creating a World:
reads the OBJ, auto-pins the top vertices by Y-coordinate, then simulates.
"""

from __future__ import annotations

import argparse

import numpy as np

from pgo import World, read_obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mass-spring bunny simulation with auto pin-by-height"
    )
    parser.add_argument(
        "--input", type=str, default="assets/model/bunny.obj",
        help="OBJ input file path"
    )
    parser.add_argument(
        "--output", type=str,
        default="output/example/python/mass_spring/bunny_auto_pin",
        help="Output directory for OBJ frames"
    )
    parser.add_argument(
        "--frames", type=int, default=300,
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
        "--pinned-pct", type=float, default=10.0,
        help="Percentage of vertices to pin by height (default 10)"
    )
    parser.add_argument(
        "--export-abc", type=str, default=None,
        help="Alembic .abc output file path"
    )
    parser.add_argument(
        "--abc-fps", type=float, default=24.0,
        help="FPS for Alembic export"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=500,
        help="Max Newton iterations per step"
    )
    args = parser.parse_args()

    # Read OBJ and inspect the mesh
    print(f"Reading {args.input} ...")
    vertices, triangles = read_obj(args.input)
    print(f"  vertices: {vertices.shape[0]}, triangles: {triangles.shape[0]}")

    # Auto-pin top vertices by Y-coordinate
    threshold = np.percentile(vertices[:, 1], 100.0 - args.pinned_pct)
    pinned = np.where(vertices[:, 1] >= threshold)[0].astype(np.uint64)
    print(f"  auto-pinned {len(pinned)} vertices (top {args.pinned_pct}% by Y, "
          f"threshold >= {threshold:.3f})")

    world = World.from_arrays(
        vertices,
        triangles,
        pinned_vertices=pinned,
        stiffness=args.stiffness,
        gravity=args.gravity,
        dt=args.dt,
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
