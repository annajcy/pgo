#!/usr/bin/env python3
"""Mass-spring cloth simulation using the pgo Python API.

Generates a self-contained cloth grid, steps the world, and writes OBJ frames.
"""

from __future__ import annotations

import argparse

import numpy as np

from pgo import World


def make_cloth_grid(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    vertices = []
    triangles = []
    for row in range(resolution):
        for col in range(resolution):
            scale = 1.0 / float(resolution - 1)
            vertices.append((col * scale, row * scale, 0.0))

    def vid(row: int, col: int) -> int:
        return row * resolution + col

    for row in range(resolution - 1):
        for col in range(resolution - 1):
            v00 = vid(row, col)
            v10 = vid(row + 1, col)
            v11 = vid(row + 1, col + 1)
            v01 = vid(row, col + 1)
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))

    pinned = np.array(
        [vid(resolution - 1, col) for col in range(resolution)],
        dtype=np.uint64,
    )
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(triangles, dtype=np.uint64),
        pinned,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mass-spring cloth simulation"
    )
    parser.add_argument(
        "--resolution", type=int, default=8,
        help="Cloth grid resolution (default 8, min 2)"
    )
    parser.add_argument(
        "--frames", type=int, default=10,
        help="Number of frames (default 10)"
    )
    parser.add_argument(
        "--output", type=str,
        default="output/example/python/mass_spring/cloth",
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
        "--stiffness", type=float, default=100.0,
        help="Spring stiffness"
    )
    parser.add_argument(
        "--gravity", type=float, default=9.8,
        help="Gravity magnitude"
    )
    parser.add_argument(
        "--dt", type=float, default=0.016,
        help="Timestep size"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=100,
        help="Max Newton iterations per step"
    )
    args = parser.parse_args()

    if args.resolution < 2:
        parser.error("--resolution must be at least 2")

    vertices, triangles, pinned = make_cloth_grid(args.resolution)

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
