from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _pgo_ext


@dataclass(frozen=True)
class StepResult:
    status: str
    iterations: int
    final_value: float
    final_gradient_norm: float


class World:
    def __init__(self, handle: object) -> None:
        self._handle = handle

    @classmethod
    def from_arrays(
        cls,
        vertices: np.ndarray,
        triangles: np.ndarray,
        *,
        stiffness: float = 100.0,
        gravity: float = 9.8,
        dt: float = 0.016,
        pinned_vertices: np.ndarray | None = None,
    ) -> World:
        vertices64 = np.ascontiguousarray(vertices, dtype=np.float64)
        triangles64 = np.ascontiguousarray(triangles, dtype=np.uint64)
        if pinned_vertices is None:
            pinned64 = np.empty((0,), dtype=np.uint64)
        else:
            pinned64 = np.ascontiguousarray(pinned_vertices, dtype=np.uint64)
        return cls(
            _pgo_ext.create_world_from_arrays(
                vertices64,
                triangles64,
                pinned64,
                float(stiffness),
                float(gravity),
                float(dt),
            )
        )

    @classmethod
    def from_obj(
        cls,
        path: str,
        *,
        stiffness: float = 100.0,
        gravity: float = 9.8,
        dt: float = 0.016,
        pinned_vertices: np.ndarray | None = None,
    ) -> World:
        if pinned_vertices is None:
            pinned64 = np.empty((0,), dtype=np.uint64)
        else:
            pinned64 = np.ascontiguousarray(pinned_vertices, dtype=np.uint64)
        return cls(_pgo_ext.create_world_from_obj(path, pinned64, float(stiffness), float(gravity), float(dt)))

    @property
    def vertex_count(self) -> int:
        return int(_pgo_ext.vertex_count(self._handle))

    def positions(self) -> np.ndarray:
        out = np.empty((self.vertex_count, 3), dtype=np.float64)
        _pgo_ext.copy_positions(self._handle, out)
        return out

    def step(
        self,
        *,
        max_iterations: int = 100,
        gradient_tolerance: float = 1e-5,
        initial_regularization: float = 1e-4,
        commit_on_failure: bool = False,
    ) -> StepResult:
        status, iterations, final_value, final_gradient_norm = _pgo_ext.step(
            self._handle,
            int(max_iterations),
            float(gradient_tolerance),
            float(initial_regularization),
            bool(commit_on_failure),
        )
        return StepResult(
            status=str(status),
            iterations=int(iterations),
            final_value=float(final_value),
            final_gradient_norm=float(final_gradient_norm),
        )

    def write_obj_frame(self, output_dir: str) -> None:
        _pgo_ext.write_obj_frame(self._handle, output_dir)

    def write_abc_frame(self, output_path: str, fps: float = 24.0) -> None:
        _pgo_ext.write_abc_frame(self._handle, output_path, float(fps))
