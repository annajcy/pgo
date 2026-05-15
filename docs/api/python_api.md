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

## Local Editable Install

```bash
uv pip install scikit-build-core nanobind numpy pytest
python scripts/pgo_configure.py debug
uv pip install -e . --no-build-isolation \
  -Ccmake.build-type=Debug \
  -Ccmake.args=-DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake
uv run python -c "import pgo; print(pgo.World)"
uv run pytest tests/python -q
```

## Accelerated Editable Install

macOS uses Apple Accelerate through the existing Eigen acceleration option:

```bash
python scripts/pgo_configure.py debug-accel
uv pip install -e . --no-build-isolation \
  -Ccmake.build-type=Debug \
  -Ccmake.args=-DCMAKE_TOOLCHAIN_FILE=build/conan/debug-accel/conan_toolchain.cmake \
  -Ccmake.define.PGO_ENABLE_EIGEN_ACCELERATION=ON
```

Linux and Windows require MKL for the accelerated variant:

```bash
python scripts/pgo_configure.py debug-accel
uv pip install -e . --no-build-isolation \
  -Ccmake.build-type=Debug \
  -Ccmake.args=-DCMAKE_TOOLCHAIN_FILE=build/conan/debug-accel/conan_toolchain.cmake \
  -Ccmake.define.PGO_ENABLE_EIGEN_ACCELERATION=ON \
  -Ccmake.define.PGO_EIGEN_ACCELERATION_BACKEND=MKL
```

For simulation code that also uses TBB or application-level parallel loops, set BLAS thread counts explicitly:

```bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

## Build Default Wheel

```bash
uv run python scripts/pgo_build_wheel.py pypgo-release-all --clear
uv pip install dist/pypgo-release-all/pgo-*.whl
uv run python -c "import pgo; print(pgo.World)"
```

`uv build` does not apply `CMakePresets.json` inheritance by itself. The wheel
wrapper prepares the selected package preset and forwards all resolved `PGO_*`
cache variables to scikit-build, including the `-all` dependency toggles.

## Build Accelerated Wheel

macOS:

```bash
uv run python scripts/pgo_build_wheel.py pypgo-release-accel-all --clear
```

Linux/Windows with MKL:

```bash
uv run python scripts/pgo_build_wheel.py pypgo-release-accel-all --clear
```

Stable-ABI release wheels use Python 3.12+ and an explicit wheel tag setting:

```bash
uv run python scripts/pgo_build_wheel.py pypgo-release-all --stable-abi --clear
```

If both default and accelerated wheels are published, use separate distribution names:

- `pgo`: default `release-all` behavior.
- `pgo-accel`: accelerated `release-accel-all` behavior.

Both packages import as `pgo`; do not install both in one environment.

Use the release wrapper when building publishable distributions:

```bash
uv run python scripts/pgo_release_wheels.py --variant default --clear
uv run python scripts/pgo_release_wheels.py --variant accel --clear
uv run python scripts/pgo_release_wheels.py --all --clear
```

The release wrapper copies the source tree to a temporary directory before
building. The repository `pyproject.toml` remains `name = "pgo"`; only the
temporary accelerated tree is patched to `name = "pgo-accel"`.

## Wheel Smoke Tests

Default release wheel:

```bash
uv run python scripts/pgo_build_wheel.py pypgo-release-all --clear
uv pip install --force-reinstall dist/pypgo-release-all/pgo-*.whl
uv run python -c "import pgo; print(pgo.World)"
uv run pytest tests/python -q
```

Accelerated release wheel on supported machines:

```bash
uv run python scripts/pgo_build_wheel.py pypgo-release-accel-all --clear
uv pip install --force-reinstall dist/pypgo-release-accel-all/pgo-*.whl
uv run pytest tests/python -q
```

## MKL Runtime Policy

The accelerated Linux/Windows package requires MKL. The default package does not.

Rules:

- `release-all` builds with Eigen's internal kernels unless the user explicitly enables acceleration.
- `release-accel-all` requires `PGO_ENABLE_EIGEN_ACCELERATION=ON`.
- On macOS, `release-accel-all` uses Apple Accelerate.
- On Linux and Windows, `release-accel-all` uses MKL and fails configuration if MKL is unavailable.
- Simulation-level parallelism should own the outer thread count. When TBB is enabled, prefer `MKL_NUM_THREADS=1` and `OMP_NUM_THREADS=1` unless a benchmark proves nested BLAS threads help a specific workload.

## Build-System References

- nanobind CMake API: https://nanobind.readthedocs.io/en/latest/api_cmake.html
- nanobind packaging guide: https://nanobind.readthedocs.io/en/latest/packaging.html
- scikit-build-core configuration: https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html
