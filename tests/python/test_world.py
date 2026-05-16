from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pgo import IOError, InvalidArgumentError, World, read_obj


def make_triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.uint64)
    return vertices, triangles


def write_triangle_obj(path: Path) -> None:
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")


def test_world_from_arrays_roundtrips_positions() -> None:
    vertices, triangles = make_triangle_mesh()
    pinned = np.array([0], dtype=np.uint64)

    world = World.from_arrays(vertices, triangles, pinned_vertices=pinned)

    assert world.vertex_count == 3
    np.testing.assert_allclose(world.positions(), vertices)


def test_step_returns_structured_result() -> None:
    vertices, triangles = make_triangle_mesh()

    world = World.from_arrays(vertices, triangles)
    result = world.step()

    assert result.status in {
        "converged",
        "max_iterations",
        "regularization_failed",
        "line_search_failed",
    }
    assert result.iterations >= 0
    assert np.isfinite(result.final_value)
    assert np.isfinite(result.final_gradient_norm)


def test_from_arrays_rejects_invalid_triangle() -> None:
    vertices, _ = make_triangle_mesh()
    triangles = np.array([[0, 1, 3]], dtype=np.uint64)
    with pytest.raises(InvalidArgumentError, match="triangle vertex index is out of range"):
        World.from_arrays(vertices, triangles)


def test_from_arrays_rejects_degenerate_triangle() -> None:
    vertices, _ = make_triangle_mesh()
    triangles = np.array([[0, 1, 1]], dtype=np.uint64)
    with pytest.raises(InvalidArgumentError, match="triangle must reference three distinct vertices"):
        World.from_arrays(vertices, triangles)


def test_step_rejects_invalid_solver_options() -> None:
    vertices, triangles = make_triangle_mesh()
    world = World.from_arrays(vertices, triangles)
    with pytest.raises(InvalidArgumentError, match="max_iterations must be positive"):
        world.step(max_iterations=0)
    with pytest.raises(InvalidArgumentError, match="gradient_tolerance must be positive"):
        world.step(gradient_tolerance=0.0)
    with pytest.raises(InvalidArgumentError, match="initial_regularization must be positive"):
        world.step(initial_regularization=0.0)


def test_from_obj_and_write_obj_frame(tmp_path: Path) -> None:
    obj_path = tmp_path / "triangle.obj"
    write_triangle_obj(obj_path)
    world = World.from_obj(str(obj_path))
    assert world.vertex_count == 3
    output_dir = tmp_path / "frames"
    world.write_obj_frame(str(output_dir))
    assert any(output_dir.glob("*.obj"))


def test_from_obj_missing_file_maps_to_exception(tmp_path: Path) -> None:
    with pytest.raises(IOError):
        World.from_obj(str(tmp_path / "missing.obj"))


def test_read_obj_roundtrips(tmp_path: Path) -> None:
    obj_path = tmp_path / "triangle.obj"
    write_triangle_obj(obj_path)
    vertices, triangles = read_obj(str(obj_path))
    assert vertices.shape == (3, 3)
    assert triangles.shape == (1, 3)
    np.testing.assert_allclose(triangles[0], [0, 1, 2])


def test_read_obj_missing_file_raises_io_error(tmp_path: Path) -> None:
    with pytest.raises(IOError):
        read_obj(str(tmp_path / "missing.obj"))
