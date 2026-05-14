# C99 And Python API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per user request, this plan intentionally contains no commit steps.

**Goal:** Build a stable C99 shared-library ABI for the Milestone 1 mass-spring pipeline, then expose a Python package through nanobind that forwards to that C ABI instead of binding the C++ template core directly.

**Architecture:** The C++23 header-oriented core remains the implementation layer. `pgo_c` is the binary boundary: opaque handles, POD descriptors, pointer/count arrays, explicit destroy functions, and status/error returns. The Python extension links to `pgo_c` and exposes a small Pythonic API over NumPy arrays while keeping STL, Eigen, templates, allocators, and C++ exceptions out of the public ABI.

**Tech Stack:** C++23, C99 ABI, CMake, Conan 2, Eigen, tinyobjloader, GoogleTest, scikit-build-core, nanobind, NumPy, uv, cibuildwheel for release automation.

---

## 0. Design Decisions

- C API first. Python uses C API forwarding so the Python package tests the same ABI that C users will consume.
- `pgo::core` stays header-oriented and does not depend on Python, nanobind, or the C API target.
- `pgo_c` is a shared library, not a header-only facade. It catches all C++ exceptions before returning to C/Python callers.
- Public C headers are valid C99 and do not include C++ headers.
- C strings are UTF-8 byte strings by convention. The C API accepts `const char*` paths and returns structured errors for invalid encoding/path failures.
- Input arrays passed through descriptors are borrowed only for the call duration. `pgo_c` copies mesh/topology/state into its own C++ objects during create calls.
- Output arrays are caller-owned. The C API validates output length before writing.
- Python arrays use `numpy.float64` and `numpy.uint64`/`numpy.int64` views at the boundary, then call the C API with raw pointers.
- Python API should be coarse-grained: create world, step, copy positions, write OBJ. Do not wrap every C++ energy/model/template type.
- Distribution variants:
  - `release-all`: default package, no Eigen acceleration on Linux/Windows, imports as `pgo`.
  - `release-accel-all`: accelerated package, imports as `pgo`; macOS uses Accelerate, Linux/Windows require MKL.
  - If both variants are published to the same package index, use distribution names `pgo` and `pgo-accel`; both expose the same Python import name `pgo`, and users install exactly one of them in an environment.
- `PGO_STATIC_GNU_RUNTIME` remains a packaging option for selected Linux GNU CLI/package scenarios. Do not default static `libstdc++`/`libgcc` into `pgo_c` or Python wheels; wheel repair tools should manage external shared-library policy.

## 1. Target File Structure

```text
.
  CMakeLists.txt
  CMakePresets.json
  conanfile.py
  pyproject.toml
  cmake/
    pgo_dependencies.cmake
    pgo_options.cmake
  include/
    pgo_c/
      export.h
      pgo.h
  src/
    c_api/
      CMakeLists.txt
      pgo_c.cpp
      pgo_c.symbols
      pgo_c.version
    python/
      CMakeLists.txt
      pgo_ext.cpp
  python/
    pgo/
      __init__.py
      world.py
      py.typed
  tests/
    c_api/
      CMakeLists.txt
      test_c_api.c
    python/
      test_world.py
  docs/
    api/
      c_api.md
      python_api.md
```

Responsibilities:

- `include/pgo_c/export.h`: platform export/import macro for the C ABI.
- `include/pgo_c/pgo.h`: complete public C99 API.
- `src/c_api/pgo_c.cpp`: C-to-C++ bridge, handle ownership, exception translation, status/error helpers.
- `src/c_api/pgo_c.version`: Linux version script that exports only `pgo_*`.
- `src/c_api/pgo_c.symbols`: macOS exported symbols list that exports only `_pgo_*`.
- `src/python/pgo_ext.cpp`: nanobind extension that forwards to `pgo_c`; no direct include of `pgo/core/...`.
- `python/pgo/world.py`: Pythonic wrapper, NumPy validation, lifetime management.
- `tests/c_api/test_c_api.c`: C compiler smoke/integration test.
- `tests/python/test_world.py`: uv/pytest test for the installed editable package.

## Phase 1: Build Options And Package Backend

### Task 1.1: Normalize C/Python Build Options

**Files:**
- Modify: `cmake/pgo_options.cmake`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add CMake options**

Add these options to `cmake/pgo_options.cmake`:

```cmake
option(PGO_BUILD_C_API "Build the C99 shared-library API" ON)
option(PGO_BUILD_PYTHON "Build the nanobind Python extension" OFF)
option(PGO_ENABLE_PYTHON_STABLE_ABI "Build Python extension with Python stable ABI when supported" ON)
set(PGO_PYTHON_EXTENSION_NAME "_pgo_ext" CACHE STRING "Name of the private Python extension module")
set_property(CACHE PGO_PYTHON_EXTENSION_NAME PROPERTY STRINGS "_pgo_ext")
```

Use `PGO_BUILD_C_API`, not a second spelling such as `PGO_ENABLE_C_API`, because the top-level CMake already uses `PGO_BUILD_C_API`.

- [ ] **Step 2: Wire top-level subdirectories**

In `CMakeLists.txt`, keep the existing C API gate and add the Python gate:

```cmake
if(PGO_BUILD_C_API AND EXISTS "${PROJECT_SOURCE_DIR}/src/c_api/CMakeLists.txt")
    add_subdirectory(src/c_api)
endif()

if(PGO_BUILD_PYTHON)
    if(NOT PGO_BUILD_C_API)
        message(FATAL_ERROR "PGO_BUILD_PYTHON=ON requires PGO_BUILD_C_API=ON")
    endif()
    add_subdirectory(src/python)
endif()
```

- [ ] **Step 3: Verify configuration gates**

Run:

```bash
cmake --preset debug
cmake --build --preset debug --target help
```

Expected:

- Configure succeeds when `src/c_api` does not exist yet.
- No Python target is built unless `PGO_BUILD_PYTHON=ON`.

### Task 1.2: Move Python Packaging To scikit-build-core

**Files:**
- Modify: `pyproject.toml`
- Create: `python/pgo/__init__.py`
- Create: `python/pgo/world.py`
- Create: `python/pgo/py.typed`

- [ ] **Step 1: Replace Hatchling backend with scikit-build-core**

`pyproject.toml` should keep project metadata and `pgo-configure`, then switch the build backend:

```toml
[project]
name = "pgo"
version = "0.1.0"
description = "GPU-aware C++23 physics simulation"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
]

[project.scripts]
pgo-configure = "scripts.pgo_configure:main_cli"

[tool.uv]
package = true

[build-system]
requires = [
    "scikit-build-core>=0.10",
    "nanobind>=2.0",
    "numpy>=1.24",
]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
minimum-version = "build-system.requires"
build-dir = "build/python/{wheel_tag}"
wheel.packages = ["python/pgo", "scripts"]

[tool.scikit-build.cmake.define]
PGO_BUILD_C_API = true
PGO_BUILD_PYTHON = true
```

Notes:

- scikit-build-core automatically looks for packages under `python/<package>`.
- Dynamic build choices can be passed with `uv pip install . --config-settings=cmake.define.PGO_ENABLE_EIGEN_ACCELERATION=ON` or with `SKBUILD_CMAKE_DEFINE`.

- [ ] **Step 2: Add initial Python package files**

`python/pgo/__init__.py`:

```python
from .world import StepResult, World

__all__ = ["StepResult", "World"]
```

`python/pgo/world.py`:

```python
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
    ) -> "World":
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
    ) -> "World":
        return cls(_pgo_ext.create_world_from_obj(path, float(stiffness), float(gravity), float(dt)))

    @property
    def vertex_count(self) -> int:
        return int(_pgo_ext.vertex_count(self._handle))

    def positions(self) -> np.ndarray:
        out = np.empty((self.vertex_count, 3), dtype=np.float64)
        _pgo_ext.copy_positions(self._handle, out)
        return out

    def step(self) -> StepResult:
        status, iterations, final_value, final_gradient_norm = _pgo_ext.step(self._handle)
        return StepResult(
            status=str(status),
            iterations=int(iterations),
            final_value=float(final_value),
            final_gradient_norm=float(final_gradient_norm),
        )

    def write_obj(self, path: str) -> None:
        _pgo_ext.write_obj(self._handle, path)
```

`python/pgo/py.typed` is an empty marker file.

- [ ] **Step 3: Verify Python package imports before extension exists fail clearly**

Run after file creation and before extension build:

```bash
uv run python -c "import pgo"
```

Expected while `_pgo_ext` is not implemented yet:

```text
ImportError: cannot import name '_pgo_ext'
```

That failure is acceptable at this step; Task 4 turns it into a passing import.

## Phase 2: Public C99 ABI

### Task 2.1: Add Export Macro Header

**Files:**
- Create: `include/pgo_c/export.h`

- [ ] **Step 1: Create platform export macro**

`include/pgo_c/export.h`:

```c
#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(PGO_C_BUILDING_LIBRARY)
#    define PGO_C_API __declspec(dllexport)
#  else
#    define PGO_C_API __declspec(dllimport)
#  endif
#else
#  define PGO_C_API __attribute__((visibility("default")))
#endif
```

- [ ] **Step 2: Compile header as C**

Run:

```bash
cc -std=c99 -Iinclude -x c -c include/pgo_c/export.h -o /tmp/pgo_export_header.o
```

Expected: command exits with status 0.

### Task 2.2: Add Public C API Header

**Files:**
- Create: `include/pgo_c/pgo.h`

- [ ] **Step 1: Define the API surface**

`include/pgo_c/pgo.h`:

```c
#pragma once

#include "pgo_c/export.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum pgo_status_t {
    PGO_STATUS_OK = 0,
    PGO_STATUS_INVALID_ARGUMENT = 1,
    PGO_STATUS_IO_ERROR = 2,
    PGO_STATUS_INTERNAL_ERROR = 3
} pgo_status_t;

typedef enum pgo_solver_status_t {
    PGO_SOLVER_CONVERGED = 0,
    PGO_SOLVER_MAX_ITERATIONS = 1,
    PGO_SOLVER_REGULARIZATION_FAILED = 2,
    PGO_SOLVER_LINE_SEARCH_FAILED = 3
} pgo_solver_status_t;

typedef struct pgo_error_t {
    pgo_status_t status;
    char message[512];
} pgo_error_t;

typedef struct pgo_world_t pgo_world_t;

typedef struct pgo_mesh_view_t {
    const double* positions_xyz;
    uint64_t vertex_count;
    const uint64_t* triangles;
    uint64_t triangle_count;
    const uint64_t* pinned_vertices;
    uint64_t pinned_vertex_count;
} pgo_mesh_view_t;

typedef struct pgo_mass_spring_params_t {
    double stiffness;
    double gravity;
    double dt;
} pgo_mass_spring_params_t;

typedef struct pgo_step_result_t {
    pgo_solver_status_t solver_status;
    uint64_t iterations;
    double final_value;
    double final_gradient_norm;
} pgo_step_result_t;

PGO_C_API const char* pgo_version(void);

PGO_C_API void pgo_error_clear(pgo_error_t* error);

PGO_C_API pgo_status_t pgo_world_create_mass_spring(
    const pgo_mesh_view_t* mesh,
    const pgo_mass_spring_params_t* params,
    pgo_world_t** out_world,
    pgo_error_t* error);

PGO_C_API pgo_status_t pgo_world_create_mass_spring_from_obj(
    const char* path,
    const pgo_mass_spring_params_t* params,
    pgo_world_t** out_world,
    pgo_error_t* error);

PGO_C_API void pgo_world_destroy(pgo_world_t* world);

PGO_C_API pgo_status_t pgo_world_vertex_count(
    const pgo_world_t* world,
    uint64_t* out_vertex_count,
    pgo_error_t* error);

PGO_C_API pgo_status_t pgo_world_copy_positions(
    const pgo_world_t* world,
    double* out_positions_xyz,
    uint64_t out_position_scalar_count,
    pgo_error_t* error);

PGO_C_API pgo_status_t pgo_world_step(
    pgo_world_t* world,
    pgo_step_result_t* out_result,
    pgo_error_t* error);

PGO_C_API pgo_status_t pgo_world_write_obj_frame(
    const pgo_world_t* world,
    const char* output_dir,
    pgo_error_t* error);

#ifdef __cplusplus
}
#endif
```

Design constraints:

- `positions_xyz` length is `vertex_count * 3`.
- `triangles` length is `triangle_count * 3`.
- `pgo_world_create_mass_spring` copies all borrowed arrays before returning.
- `pgo_world_copy_positions` requires `out_position_scalar_count >= vertex_count * 3`.
- `pgo_world_destroy(NULL)` is allowed and is a no-op.

- [ ] **Step 2: Verify C-only header hygiene**

Run:

```bash
cc -std=c99 -Iinclude -x c -c include/pgo_c/pgo.h -o /tmp/pgo_c_header.o
rg "std::|Eigen::|template|class|namespace|#include <vector>|#include <string>" include/pgo_c
```

Expected:

- `cc` exits with status 0.
- `rg` prints no matches.

## Phase 3: C Shared Library Bridge

### Task 3.1: Add `pgo_c` CMake Target

**Files:**
- Create: `src/c_api/CMakeLists.txt`
- Create: `src/c_api/pgo_c.version`
- Create: `src/c_api/pgo_c.symbols`

- [ ] **Step 1: Create the shared-library target**

`src/c_api/CMakeLists.txt`:

```cmake
add_library(pgo_c SHARED
    pgo_c.cpp
)
add_library(pgo::c_api ALIAS pgo_c)

target_compile_features(pgo_c PRIVATE cxx_std_23)
target_compile_definitions(pgo_c PRIVATE PGO_C_BUILDING_LIBRARY)
target_include_directories(pgo_c PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(pgo_c
    PRIVATE
        pgo::core
        pgo::io
        pgo::log
        pgo_project_warnings
        pgo_project_sanitizers
)

set_target_properties(pgo_c PROPERTIES
    OUTPUT_NAME pgo_c
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    POSITION_INDEPENDENT_CODE ON
)

if(APPLE)
    target_link_options(pgo_c PRIVATE
        "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/pgo_c.symbols"
    )
elseif(UNIX)
    target_link_options(pgo_c PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/pgo_c.version"
    )
endif()

install(TARGETS pgo_c
    RUNTIME DESTINATION pgo COMPONENT python
    LIBRARY DESTINATION pgo COMPONENT python
    ARCHIVE DESTINATION lib COMPONENT dev
)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/pgo_c"
    DESTINATION include
    COMPONENT dev
)
```

For non-wheel C SDK packaging, add a separate install component that installs `pgo_c` to `${CMAKE_INSTALL_LIBDIR}`. The Python wheel component deliberately places the runtime library next to `_pgo_ext`.

- [ ] **Step 2: Add Linux version script**

`src/c_api/pgo_c.version`:

```text
PGO_C_0.1 {
    global:
        pgo_*;
    local:
        *;
};
```

- [ ] **Step 3: Add macOS exported symbols list**

`src/c_api/pgo_c.symbols`:

```text
_pgo_version
_pgo_error_clear
_pgo_world_create_mass_spring
_pgo_world_create_mass_spring_from_obj
_pgo_world_destroy
_pgo_world_vertex_count
_pgo_world_copy_positions
_pgo_world_step
_pgo_world_write_obj_frame
```

### Task 3.2: Implement Error Helpers And Handles

**Files:**
- Create: `src/c_api/pgo_c.cpp`

- [ ] **Step 1: Add bridge skeleton**

Start `src/c_api/pgo_c.cpp` with:

```cpp
#include "pgo_c/pgo.h"

#include "pgo/dof/dirichlet_boundary.hpp"
#include "pgo/dof/dof_layout.hpp"
#include "pgo/dof/reduced_dof_map.hpp"
#include "pgo/energy/assembled_energy.hpp"
#include "pgo/energy/constant_force_energy.hpp"
#include "pgo/energy/energy_sum.hpp"
#include "pgo/energy/mass_spring_local_energy_provider.hpp"
#include "pgo/geometry/rest_mesh.hpp"
#include "pgo/integrator/backward_euler.hpp"
#include "pgo/integrator/dynamic_state.hpp"
#include "pgo/io/obj_reader.hpp"
#include "pgo/io/obj_writer.hpp"
#include "pgo/log/registry.hpp"
#include "pgo/solver/status_name.hpp"
#include "pgo/storage/host_buffer.hpp"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <memory>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void set_error(pgo_error_t* error, pgo_status_t status, std::string_view message) noexcept {
    if (error == nullptr) {
        return;
    }
    error->status = status;
    const std::size_t n = std::min(message.size(), sizeof(error->message) - 1);
    std::copy_n(message.data(), n, error->message);
    error->message[n] = '\0';
}

template <class F>
pgo_status_t call_c_api(pgo_error_t* error, F&& f) noexcept {
    try {
        if (error != nullptr) {
            pgo_error_clear(error);
        }
        return f();
    } catch (const std::bad_alloc&) {
        set_error(error, PGO_STATUS_INTERNAL_ERROR, "allocation failed");
        return PGO_STATUS_INTERNAL_ERROR;
    } catch (const std::invalid_argument& e) {
        set_error(error, PGO_STATUS_INVALID_ARGUMENT, e.what());
        return PGO_STATUS_INVALID_ARGUMENT;
    } catch (const std::runtime_error& e) {
        set_error(error, PGO_STATUS_INTERNAL_ERROR, e.what());
        return PGO_STATUS_INTERNAL_ERROR;
    } catch (const std::exception& e) {
        set_error(error, PGO_STATUS_INTERNAL_ERROR, e.what());
        return PGO_STATUS_INTERNAL_ERROR;
    } catch (...) {
        set_error(error, PGO_STATUS_INTERNAL_ERROR, "unknown C++ exception");
        return PGO_STATUS_INTERNAL_ERROR;
    }
}

void validate_params(const pgo_mass_spring_params_t& params) {
    if (!(params.stiffness > 0.0)) {
        throw std::invalid_argument{"stiffness must be positive"};
    }
    if (!(params.dt > 0.0)) {
        throw std::invalid_argument{"dt must be positive"};
    }
    if (!(params.gravity >= 0.0)) {
        throw std::invalid_argument{"gravity must be non-negative"};
    }
}

pgo_solver_status_t to_c_solver_status(const pgo::solver::SolverStatus status) {
    switch (status) {
    case pgo::solver::SolverStatus::converged:
        return PGO_SOLVER_CONVERGED;
    case pgo::solver::SolverStatus::max_iterations:
        return PGO_SOLVER_MAX_ITERATIONS;
    case pgo::solver::SolverStatus::regularization_failed:
        return PGO_SOLVER_REGULARIZATION_FAILED;
    case pgo::solver::SolverStatus::line_search_failed:
        return PGO_SOLVER_LINE_SEARCH_FAILED;
    }
    return PGO_SOLVER_LINE_SEARCH_FAILED;
}

std::vector<std::size_t> pinned_vertices_from_view(const pgo_mesh_view_t& view) {
    std::vector<std::size_t> pinned;
    pinned.reserve(static_cast<std::size_t>(view.pinned_vertex_count));
    for (std::uint64_t i = 0; i < view.pinned_vertex_count; ++i) {
        const auto vertex = view.pinned_vertices[i];
        if (vertex >= view.vertex_count) {
            throw std::invalid_argument{"pinned vertex index is out of range"};
        }
        pinned.push_back(static_cast<std::size_t>(vertex));
    }
    return pinned;
}

pgo::geometry::RestMesh<double, 3> make_rest_mesh_from_view(const pgo_mesh_view_t& view) {
    using VertexIndex = pgo::geometry::VertexIndex;

    pgo::storage::HostBuffer<double> positions;
    positions.reserve(static_cast<std::size_t>(view.vertex_count * 3));
    for (std::uint64_t i = 0; i < view.vertex_count * 3; ++i) {
        positions.push_back(view.positions_xyz[i]);
    }

    pgo::storage::HostBuffer<VertexIndex> faces;
    faces.reserve(static_cast<std::size_t>(view.triangle_count * 3));

    std::set<std::pair<VertexIndex, VertexIndex>> unique_edges;
    auto add_edge = [&](std::uint64_t a, std::uint64_t b) {
        if (a >= view.vertex_count || b >= view.vertex_count) {
            throw std::invalid_argument{"triangle vertex index is out of range"};
        }
        const auto va = static_cast<VertexIndex>(a);
        const auto vb = static_cast<VertexIndex>(b);
        unique_edges.insert(std::minmax(va, vb));
    };

    for (std::uint64_t tri = 0; tri < view.triangle_count; ++tri) {
        const std::uint64_t a = view.triangles[tri * 3 + 0];
        const std::uint64_t b = view.triangles[tri * 3 + 1];
        const std::uint64_t c = view.triangles[tri * 3 + 2];
        add_edge(a, b);
        add_edge(b, c);
        add_edge(c, a);
        faces.push_back(static_cast<VertexIndex>(a));
        faces.push_back(static_cast<VertexIndex>(b));
        faces.push_back(static_cast<VertexIndex>(c));
    }

    pgo::storage::HostBuffer<VertexIndex> edges;
    edges.reserve(unique_edges.size() * 2);
    for (const auto& [a, b] : unique_edges) {
        edges.push_back(a);
        edges.push_back(b);
    }

    return pgo::geometry::RestMesh<double, 3>{std::move(positions), std::move(edges), std::move(faces)};
}

pgo::dof::DirichletBoundary<double> make_boundary(const pgo::dof::DofLayout<3>& layout,
                                                   const std::vector<std::size_t>& pinned_vertices) {
    pgo::dof::DirichletBoundary<double> boundary;
    for (const std::size_t vertex : pinned_vertices) {
        if (vertex >= layout.num_vertices()) {
            throw std::invalid_argument{"pinned vertex index is out of range"};
        }
        pgo::dof::fix_vertex(boundary, layout, vertex);
    }
    return boundary;
}

pgo::math::DVec<double> make_gravity_force(const pgo::math::DVec<double>& lumped_mass, const double gravity) {
    pgo::math::DVec<double> force = pgo::math::DVec<double>::Zero(lumped_mass.size());
    for (pgo::math::DenseIndex dof = 0; dof < lumped_mass.size(); ++dof) {
        if (static_cast<std::size_t>(dof % 3) == 1) {
            force[dof] = -gravity * lumped_mass[dof];
        }
    }
    return force;
}

class MassSpringWorld3d {
    using Provider = pgo::energy::MassSpringLocalEnergyProvider<double, 3>;
    using Potential = pgo::energy::AssembledEnergyView<double, Provider>;

public:
    MassSpringWorld3d(pgo::geometry::RestMesh<double, 3> mesh,
                      std::vector<std::size_t> pinned_vertices,
                      const pgo_mass_spring_params_t& params)
        : m_mesh{std::move(mesh)},
          m_layout{m_mesh.num_vertices()},
          m_boundary{make_boundary(m_layout, pinned_vertices)},
          m_dof_map{m_layout.num_dofs(), m_boundary},
          m_lumped_mass{pgo::math::dense_index(m_layout.num_dofs())},
          m_provider{m_mesh, params.stiffness},
          m_potential{m_provider},
          m_gravity{params.gravity},
          m_dt{params.dt} {
        m_lumped_mass.setOnes();
        m_state.u.setZero(pgo::math::dense_index(m_layout.num_dofs()));
        m_state.v.setZero(pgo::math::dense_index(m_layout.num_dofs()));
        m_state.a.setZero(pgo::math::dense_index(m_layout.num_dofs()));
        m_newton_options.max_iterations = 100;
        m_newton_options.gradient_tolerance = 1e-5;
        m_newton_options.initial_regularization = 1e-4;
    }

    std::size_t vertex_count() const {
        return m_mesh.num_vertices();
    }

    void copy_positions(double* out_positions_xyz, const std::uint64_t out_scalar_count) const {
        const std::size_t expected = m_layout.num_dofs();
        if (out_scalar_count < expected) {
            throw std::invalid_argument{"output positions buffer is too small"};
        }
        for (std::size_t i = 0; i < expected; ++i) {
            out_positions_xyz[i] = m_mesh.rest_positions()[i] + m_state.u[pgo::math::dense_index(i)];
        }
    }

    pgo_step_result_t step() {
        const auto gravity_force = make_gravity_force(m_lumped_mass, m_gravity);
        const pgo::energy::ConstantForceEnergyView<double> gravity_energy{gravity_force};
        const pgo::energy::EnergySumView<double, Potential, decltype(gravity_energy)> step_energy{
            m_potential, gravity_energy};

        const auto result = m_integrator.step(
            step_energy,
            m_lumped_mass,
            m_dof_map,
            m_state,
            m_dt,
            m_newton_options,
            false);

        return pgo_step_result_t{
            to_c_solver_status(result.status),
            static_cast<std::uint64_t>(result.solver_iterations),
            result.final_value,
            result.final_gradient_norm,
        };
    }

    void write_obj_frame(const std::filesystem::path& output_dir) const {
        if (!m_writer || m_writer_output_dir != output_dir) {
            m_writer_output_dir = output_dir;
            m_writer = std::make_unique<pgo::io::ObjWriter3d>(m_writer_output_dir);
        }
        static_cast<void>(m_writer->write_frame(m_mesh, m_state.u));
    }

private:
    pgo::geometry::RestMesh<double, 3> m_mesh;
    pgo::dof::DofLayout<3> m_layout;
    pgo::dof::DirichletBoundary<double> m_boundary;
    pgo::dof::ReducedDofMap<double> m_dof_map;
    pgo::math::DVec<double> m_lumped_mass;
    Provider m_provider;
    Potential m_potential;
    double m_gravity;
    double m_dt;
    pgo::integrator::BackwardEuler<double> m_integrator;
    pgo::integrator::DynamicState<double> m_state;
    pgo::solver::NewtonOptions<double> m_newton_options;
    mutable std::filesystem::path m_writer_output_dir;
    mutable std::unique_ptr<pgo::io::ObjWriter3d> m_writer;
};

struct pgo_world_t {
    std::unique_ptr<MassSpringWorld3d> impl;
};

} // namespace
```

- [ ] **Step 2: Implement status utilities**

Append:

```cpp
extern "C" {

const char* pgo_version(void) {
    return "0.1.0";
}

void pgo_error_clear(pgo_error_t* error) {
    if (error == nullptr) {
        return;
    }
    error->status = PGO_STATUS_OK;
    error->message[0] = '\0';
}

} // extern "C"
```

### Task 3.3: Implement Create/Destroy/Query Functions

**Files:**
- Modify: `src/c_api/pgo_c.cpp`

- [ ] **Step 1: Implement array create function**

Append inside `extern "C"`:

```cpp
pgo_status_t pgo_world_create_mass_spring(
    const pgo_mesh_view_t* mesh,
    const pgo_mass_spring_params_t* params,
    pgo_world_t** out_world,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (mesh == nullptr || params == nullptr || out_world == nullptr) {
            throw std::invalid_argument{"mesh, params, and out_world must be non-null"};
        }
        *out_world = nullptr;
        validate_params(*params);
        if (mesh->positions_xyz == nullptr || mesh->vertex_count == 0) {
            throw std::invalid_argument{"mesh positions must be non-null and non-empty"};
        }
        if (mesh->triangles == nullptr || mesh->triangle_count == 0) {
            throw std::invalid_argument{"mesh triangles must be non-null and non-empty"};
        }
        if (mesh->pinned_vertex_count > 0 && mesh->pinned_vertices == nullptr) {
            throw std::invalid_argument{"pinned_vertices must be non-null when pinned_vertex_count is positive"};
        }

        auto world = std::make_unique<pgo_world_t>();
        world->impl = std::make_unique<MassSpringWorld3d>(
            make_rest_mesh_from_view(*mesh),
            pinned_vertices_from_view(*mesh),
            *params);

        *out_world = world.release();
        return PGO_STATUS_OK;
    });
}
```

- [ ] **Step 2: Implement OBJ create function**

Append:

```cpp
pgo_status_t pgo_world_create_mass_spring_from_obj(
    const char* path,
    const pgo_mass_spring_params_t* params,
    pgo_world_t** out_world,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (path == nullptr || params == nullptr || out_world == nullptr) {
            throw std::invalid_argument{"path, params, and out_world must be non-null"};
        }
        validate_params(*params);

        *out_world = nullptr;
        auto world = std::make_unique<pgo_world_t>();
        world->impl = std::make_unique<MassSpringWorld3d>(
            pgo::io::read_obj_rest_mesh_3d(std::filesystem::path{path}),
            std::vector<std::size_t>{},
            *params);

        *out_world = world.release();
        return PGO_STATUS_OK;
    });
}
```

- [ ] **Step 3: Implement destroy/query/copy**

Append:

```cpp
void pgo_world_destroy(pgo_world_t* world) {
    delete world;
}

pgo_status_t pgo_world_vertex_count(
    const pgo_world_t* world,
    uint64_t* out_vertex_count,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (world == nullptr || out_vertex_count == nullptr) {
            throw std::invalid_argument{"world and out_vertex_count must be non-null"};
        }
        *out_vertex_count = static_cast<std::uint64_t>(world->impl->vertex_count());
        return PGO_STATUS_OK;
    });
}

pgo_status_t pgo_world_copy_positions(
    const pgo_world_t* world,
    double* out_positions_xyz,
    uint64_t out_position_scalar_count,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (world == nullptr || out_positions_xyz == nullptr) {
            throw std::invalid_argument{"world and out_positions_xyz must be non-null"};
        }
        world->impl->copy_positions(out_positions_xyz, out_position_scalar_count);
        return PGO_STATUS_OK;
    });
}
```

### Task 3.4: Implement Step And OBJ Frame Write

**Files:**
- Modify: `src/c_api/pgo_c.cpp`

- [ ] **Step 1: Implement `pgo_world_step` with the internal mass-spring world**

Append:

```cpp
pgo_status_t pgo_world_step(
    pgo_world_t* world,
    pgo_step_result_t* out_result,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (world == nullptr || out_result == nullptr) {
            throw std::invalid_argument{"world and out_result must be non-null"};
        }

        *out_result = world->impl->step();
        return PGO_STATUS_OK;
    });
}
```

Acceptance rule:

- The implementation uses `MassSpringLocalEnergyProvider`, `AssembledEnergyView`, `ConstantForceEnergyView`, `EnergySumView`, and `BackwardEuler`.
- C function return value reports whether the C call itself succeeded.
- Solver convergence is reported in `out_result->solver_status`; non-convergence does not become a C ABI transport error.

- [ ] **Step 2: Implement `pgo_world_write_obj_frame`**

Append:

```cpp
pgo_status_t pgo_world_write_obj_frame(
    const pgo_world_t* world,
    const char* output_dir,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (world == nullptr || output_dir == nullptr) {
            throw std::invalid_argument{"world and output_dir must be non-null"};
        }
        world->impl->write_obj_frame(output_dir);
        return PGO_STATUS_OK;
    });
}
```

Acceptance rule:

- The implementation uses `pgo::io::ObjWriter3d`.
- `output_dir` is a directory path. The writer produces `frame_0000.obj`, `frame_0001.obj`, and so on.
- It writes `X + u`, not rest positions alone.
- It preserves the original triangle topology.

### Task 3.5: Add C API Tests

**Files:**
- Create: `tests/c_api/CMakeLists.txt`
- Create: `tests/c_api/test_c_api.c`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add C test source**

`tests/c_api/test_c_api.c`:

```c
#include "pgo_c/pgo.h"

#include <assert.h>
#include <stddef.h>

int main(void) {
    const double positions[] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    };
    const uint64_t triangles[] = {0, 1, 2};
    const uint64_t pinned[] = {0};

    const pgo_mesh_view_t mesh = {
        positions,
        3,
        triangles,
        1,
        pinned,
        1,
    };
    const pgo_mass_spring_params_t params = {
        100.0,
        9.8,
        0.016,
    };

    pgo_error_t error;
    pgo_error_clear(&error);

    pgo_world_t* world = NULL;
    assert(pgo_world_create_mass_spring(&mesh, &params, &world, &error) == PGO_STATUS_OK);
    assert(world != NULL);

    uint64_t vertex_count = 0;
    assert(pgo_world_vertex_count(world, &vertex_count, &error) == PGO_STATUS_OK);
    assert(vertex_count == 3);

    double out_positions[9] = {0.0};
    assert(pgo_world_copy_positions(world, out_positions, 9, &error) == PGO_STATUS_OK);
    assert(out_positions[3] == 1.0);

    pgo_step_result_t step_result;
    assert(pgo_world_step(world, &step_result, &error) == PGO_STATUS_OK);

    pgo_world_destroy(world);
    pgo_world_destroy(NULL);
    return 0;
}
```

- [ ] **Step 2: Add test target**

`tests/c_api/CMakeLists.txt`:

```cmake
add_executable(pgo_c_api_tests test_c_api.c)
target_link_libraries(pgo_c_api_tests PRIVATE pgo::c_api)
add_test(NAME pgo_c_api_tests COMMAND pgo_c_api_tests)
```

In `tests/CMakeLists.txt`:

```cmake
if(PGO_BUILD_C_API AND TARGET pgo_c)
    add_subdirectory(c_api)
endif()
```

- [ ] **Step 3: Build and run C API test**

Run:

```bash
cmake --build --preset debug --target pgo_c_api_tests
ctest --preset debug -R pgo_c_api_tests --output-on-failure
```

Expected: test passes and compiles as C, not C++.

- [ ] **Step 4: Check exported symbols**

Run on macOS:

```bash
nm -gU build/debug/src/c_api/libpgo_c.dylib | rg "pgo_|std::|Eigen::"
```

Expected: only `_pgo_*` exported API symbols are visible; no `std::` or `Eigen::` symbols appear.

Run on Linux:

```bash
nm -D --defined-only build/debug/src/c_api/libpgo_c.so | rg "pgo_|std::|Eigen::"
```

Expected: only `pgo_*` exported API symbols are visible; no `std::` or `Eigen::` symbols appear.

## Phase 4: nanobind Extension

### Task 4.1: Add Python Extension Target

**Files:**
- Create: `src/python/CMakeLists.txt`
- Create: `src/python/pgo_ext.cpp`

- [ ] **Step 1: Create extension CMake**

`src/python/CMakeLists.txt`:

```cmake
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

set(pgo_python_sources pgo_ext.cpp)

if(PGO_ENABLE_PYTHON_STABLE_ABI)
    nanobind_add_module(${PGO_PYTHON_EXTENSION_NAME}
        STABLE_ABI
        NB_STATIC
        ${pgo_python_sources}
    )
else()
    nanobind_add_module(${PGO_PYTHON_EXTENSION_NAME}
        NB_STATIC
        ${pgo_python_sources}
    )
endif()

target_link_libraries(${PGO_PYTHON_EXTENSION_NAME} PRIVATE pgo::c_api)
target_include_directories(${PGO_PYTHON_EXTENSION_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/include")

if(APPLE)
    set_target_properties(${PGO_PYTHON_EXTENSION_NAME} PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    set_target_properties(${PGO_PYTHON_EXTENSION_NAME} PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()

install(TARGETS ${PGO_PYTHON_EXTENSION_NAME}
    LIBRARY DESTINATION pgo COMPONENT python
    RUNTIME DESTINATION pgo COMPONENT python
)
```

Why this shape:

- nanobind's `STABLE_ABI` reduces wheel matrix size on Python versions where it applies.
- nanobind defaults to size-oriented extension optimization in non-debug builds, while `pgo_c` and the C++ core keep project optimization flags.
- `pgo_c` is installed next to the extension, so loader search can use `@loader_path`/`$ORIGIN`.

- [ ] **Step 2: Add extension source skeleton**

`src/python/pgo_ext.cpp`:

```cpp
#include "pgo_c/pgo.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

class WorldHandle {
public:
    explicit WorldHandle(pgo_world_t* world)
        : m_world{world} {}

    WorldHandle(const WorldHandle&) = delete;
    WorldHandle& operator=(const WorldHandle&) = delete;

    WorldHandle(WorldHandle&& other) noexcept
        : m_world{other.m_world} {
        other.m_world = nullptr;
    }

    WorldHandle& operator=(WorldHandle&& other) noexcept {
        if (this != &other) {
            pgo_world_destroy(m_world);
            m_world = other.m_world;
            other.m_world = nullptr;
        }
        return *this;
    }

    ~WorldHandle() {
        pgo_world_destroy(m_world);
    }

    pgo_world_t* get() const {
        return m_world;
    }

private:
    pgo_world_t* m_world = nullptr;
};

void throw_on_error(pgo_status_t status, const pgo_error_t& error) {
    if (status == PGO_STATUS_OK) {
        return;
    }
    throw std::runtime_error(error.message[0] == '\0' ? "pgo C API call failed" : error.message);
}

using FloatArray = nb::ndarray<nb::numpy, const double, nb::shape<-1, 3>, nb::c_contig>;
using UIntArray2 = nb::ndarray<nb::numpy, const std::uint64_t, nb::shape<-1, 3>, nb::c_contig>;
using UIntArray1 = nb::ndarray<nb::numpy, const std::uint64_t, nb::shape<-1>, nb::c_contig>;
using MutableFloatArray = nb::ndarray<nb::numpy, double, nb::shape<-1, 3>, nb::c_contig>;

std::shared_ptr<WorldHandle> create_world_from_arrays(
    FloatArray vertices,
    UIntArray2 triangles,
    UIntArray1 pinned_vertices,
    double stiffness,
    double gravity,
    double dt) {
    const pgo_mesh_view_t mesh{
        vertices.data(),
        static_cast<std::uint64_t>(vertices.shape(0)),
        triangles.data(),
        static_cast<std::uint64_t>(triangles.shape(0)),
        pinned_vertices.data(),
        static_cast<std::uint64_t>(pinned_vertices.shape(0)),
    };
    const pgo_mass_spring_params_t params{stiffness, gravity, dt};

    pgo_error_t error;
    pgo_error_clear(&error);
    pgo_world_t* raw = nullptr;
    throw_on_error(pgo_world_create_mass_spring(&mesh, &params, &raw, &error), error);
    return std::make_shared<WorldHandle>(raw);
}

std::shared_ptr<WorldHandle> create_world_from_obj(
    const std::string& path,
    double stiffness,
    double gravity,
    double dt) {
    const pgo_mass_spring_params_t params{stiffness, gravity, dt};

    pgo_error_t error;
    pgo_error_clear(&error);
    pgo_world_t* raw = nullptr;
    throw_on_error(pgo_world_create_mass_spring_from_obj(path.c_str(), &params, &raw, &error), error);
    return std::make_shared<WorldHandle>(raw);
}

std::uint64_t vertex_count(const std::shared_ptr<WorldHandle>& world) {
    pgo_error_t error;
    pgo_error_clear(&error);
    std::uint64_t count = 0;
    throw_on_error(pgo_world_vertex_count(world->get(), &count, &error), error);
    return count;
}

void copy_positions(const std::shared_ptr<WorldHandle>& world, MutableFloatArray out) {
    pgo_error_t error;
    pgo_error_clear(&error);
    throw_on_error(
        pgo_world_copy_positions(
            world->get(),
            out.data(),
            static_cast<std::uint64_t>(out.shape(0) * 3)),
        error);
}

nb::tuple step(const std::shared_ptr<WorldHandle>& world) {
    pgo_error_t error;
    pgo_error_clear(&error);
    pgo_step_result_t result{};
    throw_on_error(pgo_world_step(world->get(), &result, &error), error);
    const char* status = "line_search_failed";
    switch (result.solver_status) {
    case PGO_SOLVER_CONVERGED:
        status = "converged";
        break;
    case PGO_SOLVER_MAX_ITERATIONS:
        status = "max_iterations";
        break;
    case PGO_SOLVER_REGULARIZATION_FAILED:
        status = "regularization_failed";
        break;
    case PGO_SOLVER_LINE_SEARCH_FAILED:
        status = "line_search_failed";
        break;
    }
    return nb::make_tuple(
        status,
        result.iterations,
        result.final_value,
        result.final_gradient_norm);
}

void write_obj(const std::shared_ptr<WorldHandle>& world, const std::string& path) {
    pgo_error_t error;
    pgo_error_clear(&error);
    throw_on_error(pgo_world_write_obj_frame(world->get(), path.c_str(), &error), error);
}

} // namespace

NB_MODULE(_pgo_ext, m) {
    nb::class_<WorldHandle, std::shared_ptr<WorldHandle>>(m, "WorldHandle");

    m.def("version", &pgo_version);
    m.def("create_world_from_arrays", &create_world_from_arrays);
    m.def("create_world_from_obj", &create_world_from_obj);
    m.def("vertex_count", &vertex_count);
    m.def("copy_positions", &copy_positions);
    m.def("step", &step);
    m.def("write_obj", &write_obj);
}
```

If `PGO_PYTHON_EXTENSION_NAME` is changed away from `_pgo_ext`, the `NB_MODULE` name and `python/pgo/world.py` import must change together.

### Task 4.2: Add Python Tests

**Files:**
- Create: `tests/python/test_world.py`

- [ ] **Step 1: Add Python smoke tests**

`tests/python/test_world.py`:

```python
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
```

- [ ] **Step 2: Install editable package with uv**

Development install without build isolation:

```bash
uv pip install -e . --no-build-isolation
```

If build dependencies are missing in the active environment:

```bash
uv pip install scikit-build-core nanobind numpy pytest
uv pip install -e . --no-build-isolation
```

Expected: `_pgo_ext` builds and installs inside the editable package.

- [ ] **Step 3: Run Python tests**

```bash
uv run pytest tests/python -q
```

Expected: both Python tests pass.

## Phase 5: Package Variants And Acceleration

### Task 5.1: Define Local Install Commands

**Files:**
- Modify: `docs/api/python_api.md`

- [ ] **Step 1: Document normal local editable install**

Add:

```markdown
## Local Editable Install

```bash
uv pip install scikit-build-core nanobind numpy pytest
uv pip install -e . --no-build-isolation
uv run python -c "import pgo; print(pgo.World)"
uv run pytest tests/python -q
```
```

- [ ] **Step 2: Document accelerated local editable install**

Add:

```markdown
## Accelerated Editable Install

macOS uses Apple Accelerate through the existing Eigen acceleration option:

```bash
uv pip install -e . --no-build-isolation \
  --config-settings=cmake.define.PGO_ENABLE_EIGEN_ACCELERATION=ON
```

Linux and Windows require MKL for the accelerated variant:

```bash
uv pip install -e . --no-build-isolation \
  --config-settings=cmake.define.PGO_ENABLE_EIGEN_ACCELERATION=ON \
  --config-settings=cmake.define.PGO_EIGEN_ACCELERATION_BACKEND=MKL
```

For simulation code that also uses TBB or application-level parallel loops, set BLAS thread counts explicitly:

```bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
```
```

### Task 5.2: Define Wheel Build Commands

**Files:**
- Modify: `docs/api/python_api.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Document default wheel build**

Add:

```markdown
## Build Default Wheel

```bash
uv build --wheel
uv pip install dist/pgo-*.whl
uv run python -c "import pgo; print(pgo.World)"
```
```

- [ ] **Step 2: Document accelerated wheel build**

Add:

```markdown
## Build Accelerated Wheel

macOS:

```bash
uv build --wheel \
  --config-setting=cmake.define.PGO_ENABLE_EIGEN_ACCELERATION=ON
```

Linux/Windows with MKL:

```bash
uv build --wheel \
  --config-setting=cmake.define.PGO_ENABLE_EIGEN_ACCELERATION=ON \
  --config-setting=cmake.define.PGO_EIGEN_ACCELERATION_BACKEND=MKL
```

If both default and accelerated wheels are published, use separate distribution names:

- `pgo`: default `release-all` behavior.
- `pgo-accel`: accelerated `release-accel-all` behavior.

Both packages import as `pgo`; do not install both in one environment.
```

- [ ] **Step 3: Keep package metadata ready for variant naming**

When building the accelerated distribution, set the project name with a scikit-build metadata override or a release-time pyproject patch:

```toml
[project]
name = "pgo-accel"
```

The import package remains `python/pgo`.

### Task 5.3: Clarify MKL And Runtime Policy

**Files:**
- Modify: `docs/api/python_api.md`
- Modify: `docs/api/c_api.md`

- [ ] **Step 1: Add MKL runtime policy**

Add:

```markdown
## MKL Runtime Policy

The accelerated Linux/Windows package requires MKL. The default package does not.

Rules:

- `release-all` builds with Eigen's internal kernels unless the user explicitly enables acceleration.
- `release-accel-all` requires `PGO_ENABLE_EIGEN_ACCELERATION=ON`.
- On macOS, `release-accel-all` uses Apple Accelerate.
- On Linux and Windows, `release-accel-all` uses MKL and fails configuration if MKL is unavailable.
- Simulation-level parallelism should own the outer thread count. When TBB is enabled, prefer `MKL_NUM_THREADS=1` and `OMP_NUM_THREADS=1` unless a benchmark proves nested BLAS threads help a specific workload.
```

- [ ] **Step 2: Add static runtime policy**

Add:

```markdown
## Static GNU Runtime Policy

`PGO_STATIC_GNU_RUNTIME` is a packaging option for selected Linux GNU deliverables, not a development default.

Do not apply it blindly to `pgo_c` or Python wheels. Python wheels should be repaired/audited with the platform wheel tooling used by CI, because static C++ runtime choices interact with manylinux/musllinux policy and with dependencies such as MKL.
```

## Phase 6: Documentation

### Task 6.1: Document C API

**Files:**
- Create: `docs/api/c_api.md`

- [ ] **Step 1: Write C API overview**

`docs/api/c_api.md`:

````markdown
# C API

`pgo_c` exposes a C99 ABI over the C++ simulation core.

The public ABI contains only:

- `extern "C"` functions.
- Opaque handles such as `pgo_world_t`.
- POD descriptors such as `pgo_mesh_view_t`.
- Caller-owned output buffers.
- Explicit destroy functions.
- `pgo_status_t` plus `pgo_error_t` error reporting.

It does not expose STL, Eigen, C++ templates, C++ exceptions, namespaces, classes, or allocators.

## Minimal Example

```c
#include "pgo_c/pgo.h"

int main(void) {
    const double positions[] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    };
    const uint64_t triangles[] = {0, 1, 2};
    const pgo_mesh_view_t mesh = {positions, 3, triangles, 1, NULL, 0};
    const pgo_mass_spring_params_t params = {100.0, 9.8, 0.016};

    pgo_error_t error;
    pgo_error_clear(&error);

    pgo_world_t* world = NULL;
    if (pgo_world_create_mass_spring(&mesh, &params, &world, &error) != PGO_STATUS_OK) {
        return 1;
    }

    pgo_step_result_t result;
    pgo_world_step(world, &result, &error);
    pgo_world_destroy(world);
    return 0;
}
```
````

### Task 6.2: Document Python API

**Files:**
- Create: `docs/api/python_api.md`

- [ ] **Step 1: Write Python API overview**

`docs/api/python_api.md`:

````markdown
# Python API

The Python package uses nanobind for a private extension module named `pgo._pgo_ext`.
That extension forwards to the public C API in `pgo_c`; it does not bind the C++ template core directly.

## Minimal Example

```python
import numpy as np

from pgo import World

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
positions = world.positions()
```

The API intentionally starts coarse-grained. Exposing individual energies, solvers,
and Eigen-backed objects would make Python depend on C++ template internals and would
weaken the ABI boundary.
````

### Task 6.3: Record Upstream Build References

**Files:**
- Modify: `docs/api/python_api.md`

- [ ] **Step 1: Add references**

Add:

```markdown
## Build-System References

- nanobind CMake API: https://nanobind.readthedocs.io/en/latest/api_cmake.html
- nanobind packaging guide: https://nanobind.readthedocs.io/en/latest/packaging.html
- scikit-build-core configuration: https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html
```

## Phase 7: Verification Matrix

### Task 7.1: Local Verification Commands

**Files:**
- Modify: `docs/api/c_api.md`
- Modify: `docs/api/python_api.md`

- [ ] **Step 1: Run C++/C build and tests**

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Expected: existing C++ tests and `pgo_c_api_tests` pass.

- [ ] **Step 2: Run Python editable install and tests**

```bash
uv pip install scikit-build-core nanobind numpy pytest
uv pip install -e . --no-build-isolation
uv run pytest tests/python -q
```

Expected: Python tests pass and `import pgo` succeeds.

- [ ] **Step 3: Run accelerated package smoke on macOS**

```bash
uv pip install -e . --no-build-isolation \
  --config-settings=cmake.define.PGO_ENABLE_EIGEN_ACCELERATION=ON
uv run pytest tests/python -q
```

Expected: Python tests pass with Apple Accelerate.

- [ ] **Step 4: Run accelerated package smoke on Linux/Windows MKL machines**

```bash
uv pip install -e . --no-build-isolation \
  --config-settings=cmake.define.PGO_ENABLE_EIGEN_ACCELERATION=ON \
  --config-settings=cmake.define.PGO_EIGEN_ACCELERATION_BACKEND=MKL
uv run pytest tests/python -q
```

Expected: configure fails if MKL is unavailable; otherwise Python tests pass.

### Task 7.2: ABI Hygiene Checks

**Files:**
- Modify: `docs/api/c_api.md`

- [ ] **Step 1: Check C header stays C-only**

```bash
rg "std::|Eigen::|template|class|namespace|#include <vector>|#include <string>" include/pgo_c
cc -std=c99 -Iinclude -x c -c include/pgo_c/pgo.h -o /tmp/pgo_c_header.o
```

Expected: no `rg` matches; C compiler succeeds.

- [ ] **Step 2: Check exported symbols**

macOS:

```bash
nm -gU build/debug/src/c_api/libpgo_c.dylib | rg "pgo_|std::|Eigen::"
```

Linux:

```bash
nm -D --defined-only build/debug/src/c_api/libpgo_c.so | rg "pgo_|std::|Eigen::"
```

Expected: only public `pgo_*` C ABI symbols are exported.

- [ ] **Step 3: Check Python extension links to packaged C runtime**

macOS:

```bash
otool -L "$(python -c 'import pgo._pgo_ext as m; print(m.__file__)')"
```

Linux:

```bash
ldd "$(python -c 'import pgo._pgo_ext as m; print(m.__file__)')"
```

Windows:

```powershell
python -c "import pgo._pgo_ext as m; print(m.__file__)"
```

Expected: `_pgo_ext` can locate `pgo_c` from the installed package directory.

## Completion Criteria

- `include/pgo_c/pgo.h` compiles as C99.
- `pgo_c` builds as a shared library on macOS, Linux, and Windows.
- Exported symbols are limited to the intended `pgo_*` C ABI functions on macOS/Linux, with Windows controlled by `PGO_C_API`.
- C API tests create/destroy a world, query vertex count, copy positions, and run one step.
- Every exported C function that calls C++ code catches exceptions and returns `pgo_status_t`.
- Python package builds through scikit-build-core and nanobind.
- `uv pip install -e . --no-build-isolation` installs an importable `pgo` package.
- Python tests pass with the default package.
- Accelerated macOS package works with Apple Accelerate.
- Accelerated Linux/Windows package requires MKL and fails clearly when MKL is unavailable.
- `pgo::core` does not depend on nanobind or Python.
- `src/python/pgo_ext.cpp` includes `pgo_c/pgo.h` and does not include `pgo/core/...` headers.

## Recommended Implementation Order

1. Finish Milestone 1 Phase 5 reusable mass-spring pipeline helpers.
2. Implement Phase 1 build/package options.
3. Implement Phase 2 public C header and hygiene checks.
4. Implement Phase 3 `pgo_c` shared library and C tests.
5. Verify C step/write logic against the example simulation pipeline.
6. Implement Phase 4 nanobind extension.
7. Implement Phase 5 package variant docs and local uv install commands.
8. Implement Phase 6 docs.
9. Run Phase 7 verification matrix.

## Self-Review

- Spec coverage: C99 API split out from Milestone 1, nanobind Python API merged into this plan, uv install/build commands included, acceleration/MKL package policy included, symbol visibility/export-map policy included.
- Placeholder scan: The plan avoids open-ended placeholders; code steps define the C ABI, internal mass-spring world, nanobind bridge, tests, and verification commands directly.
- Type consistency: Public C names use `pgo_world_t`, `pgo_mesh_view_t`, `pgo_mass_spring_params_t`, `pgo_step_result_t`, and `pgo_error_t` consistently. Python wrapper uses `World` and `StepResult` consistently.
- Scope control: The plan does not expose C++ template energies, Eigen objects, collision/contact, IPC, FEM, or GPU backend APIs.
