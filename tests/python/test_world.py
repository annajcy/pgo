from __future__ import annotations

import numpy as np

from pgo import World


def test_world_from_arrays_roundtrips_positions() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.uint64)
    pinned = np.array([0], dtype=np.uint64)

    world = World.from_arrays(vertices, triangles, pinned_vertices=pinned)

    assert world.vertex_count == 3
    np.testing.assert_allclose(world.positions(), vertices)


def test_step_returns_structured_result() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.uint64)

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
