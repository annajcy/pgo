# pgo

## Requirements

- CMake 3.28 or newer
- Ninja
- Conan 2
- uv
- A C++23-capable compiler

Use uv for Python-managed tooling:

```bash
uv tool install cmake
uv tool install ninja
uv tool install conan
```

After cloning, sync Python tooling:

```bash
uv sync
```

The project sets `package = false` so uv does not attempt to auto-build `pgo`.
The Python package depends on Conan-generated CMake toolchain files and must be
installed explicitly after `pgo_configure.py` (see [Python Package](#python-package)).

## Conan Profiles

This repository keeps Conan profiles under `conan/profiles/`.

Use the platform-dispatching default profile for normal local builds:

```bash
conan/profiles/default
```

It selects one of:

- `conan/profiles/macos-arm64-apple-clang`
- `conan/profiles/ubuntu-x86_64-gcc`
- `conan/profiles/windows-x86_64-msvc`

You can also pass a platform profile explicitly with
`--profile:host=... --profile:build=...`.

## Preset Reference

Preset names follow `{debug,release}[-accel][-all][-asan]`. Each slot maps to one
orthogonal dimension:

| Slot | Present | Absent |
|------|---------|--------|
| Build type | `debug` / `release` | (required) |
| `-accel` | Eigen BLAS/LAPACK acceleration (MKL or Accelerate) | no acceleration |
| `-all` | all optional compile-time deps (spdlog, Alembic) | core deps only |
| `-asan` | address + undefined-behavior sanitizers | no sanitizers |

ASan presets share the Conan dependency folder of their non-ASan counterpart
(e.g. `debug-asan` reuses `build/conan/debug/`), because sanitizers are pure
compiler flags and do not change the dependency graph.

### Visible Presets

| Preset | Build | Accel | All | ASan |
|--------|-------|-------|-----|------|
| `debug` | Debug | | | |
| `debug-accel` | Debug | ✓ | | |
| `debug-all` | Debug | | ✓ | |
| `debug-accel-all` | Debug | ✓ | ✓ | |
| `debug-asan` | Debug | | | ✓ |
| `debug-accel-asan` | Debug | ✓ | | ✓ |
| `debug-all-asan` | Debug | | ✓ | ✓ |
| `debug-accel-all-asan` | Debug | ✓ | ✓ | ✓ |
| `release` | Release | | | |
| `release-accel` | Release | ✓ | | |
| `release-all` | Release | | ✓ | |
| `release-accel-all` | Release | ✓ | ✓ | |
| `pypgo-release-all` | Release | | ✓ | |
| `pypgo-release-accel-all` | Release | ✓ | ✓ | |
| `release-asan` | Release | | | ✓ |
| `release-accel-asan` | Release | ✓ | | ✓ |
| `release-all-asan` | Release | | ✓ | ✓ |
| `release-accel-all-asan` | Release | ✓ | ✓ | ✓ |

## Quick Start

Configure, build, and run tests:

```bash
uv run python scripts/pgo_configure.py debug --build  # install Conan deps + cmake --preset + cmake --build
ctest --preset debug                # run tests
```

Without `--build`, only configure runs — useful when you want to inspect or tweak
before compiling:

```bash
uv run python scripts/pgo_configure.py debug      # install Conan deps + cmake --preset
cmake --build --preset debug    # compile
ctest --preset debug            # run tests
```

Choose a different preset to switch configurations.
Configure only:

```bash
uv run python scripts/pgo_configure.py release
uv run python scripts/pgo_configure.py debug-asan
uv run python scripts/pgo_configure.py debug-all
uv run python scripts/pgo_configure.py debug-accel
```

Or configure + build in one step with `--build`:

```bash
uv run python scripts/pgo_configure.py release --build
uv run python scripts/pgo_configure.py debug-asan --build
uv run python scripts/pgo_configure.py debug-all --build
uv run python scripts/pgo_configure.py debug-accel --build
```

Configure every visible preset at once:

```bash
uv run python scripts/pgo_configure.py --all-presets
```

Add `--build` to compile all of them too:

```bash
uv run python scripts/pgo_configure.py --all-presets --build
```

See [Preset Reference](#preset-reference) for the full 18-preset matrix.

## Acceleration

`-accel` presets enable Eigen BLAS/LAPACK acceleration via `PGO_EIGEN_ACCELERATION_BACKEND=AUTO`:

| Platform | Backend | Setup |
|----------|---------|-------|
| macOS | Accelerate (built-in) | none |
| Linux | oneMKL | `scripts/install-onemkl/install-onemkl-linux.sh` |
| Windows | oneMKL | `.\scripts\install-onemkl\install-onemkl-windows.ps1` |

Configure + build in one step:

```bash
uv run python scripts/pgo_configure.py debug-accel --build
ctest --preset debug-accel
```

Or step by step:

```bash
uv run python scripts/pgo_configure.py debug-accel
cmake --build --preset debug-accel
ctest --preset debug-accel
```

## Benchmarks

Benchmarks use Google Benchmark and are not registered as `ctest` tests (results
are machine-dependent). Build and run with a release preset:

```bash
uv run python scripts/pgo_configure.py release
cmake --build --preset release --target pgo_benchmarks
./build/release/benchmarks/pgo_benchmarks
```

Accelerated benchmarks:

```bash
uv run python scripts/pgo_configure.py release-accel
cmake --build --preset release-accel --target pgo_benchmarks
./build/release-accel/benchmarks/pgo_benchmarks
```

## Python Package

The Python package wraps `pgo_c` via nanobind. Install in editable mode after
configuring with a C-API-enabled preset:

```bash
python scripts/pgo_configure.py debug
uv pip install -e . --no-build-isolation \
  -Ccmake.build-type=Debug \
  -Ccmake.args=-DCMAKE_TOOLCHAIN_FILE=build/conan/debug/conan_toolchain.cmake
uv run pytest tests/python -q
```

For the accelerated variant:

```bash
python scripts/pgo_configure.py debug-accel
uv pip install -e . --no-build-isolation \
  -Ccmake.build-type=Debug \
  -Ccmake.args=-DCMAKE_TOOLCHAIN_FILE=build/conan/debug-accel/conan_toolchain.cmake \
  -Ccmake.define.PGO_ENABLE_EIGEN_ACCELERATION=ON
uv run pytest tests/python -q
```

Build a release wheel:

```bash
uv run python scripts/pgo_build_wheel.py pypgo-release-all --clear
```

`uv build` does not consume `CMakePresets.json` inheritance directly. The wheel
wrapper prepares the matching Conan/CMake preset, then forwards the resolved
`PGO_*` cache variables such as `PGO_ENABLE_SPDLOG=ON` and
`PGO_ENABLE_ALEMBIC=ON` to scikit-build.

Build release distributions:

```bash
uv run python scripts/pgo_release_wheels.py --variant default --clear
uv run python scripts/pgo_release_wheels.py --variant accel --clear
uv run python scripts/pgo_release_wheels.py --all --clear
```

The release wrapper builds from temporary source trees. The source
`pyproject.toml` keeps `name = "pgo"`; the accelerated temporary tree is patched
to publish the `pgo-accel` distribution while the import package remains `pgo`.
Do not install `pgo` and `pgo-accel` in the same environment.

Pass an explicit Conan profile when the default detected profile is not what
you want (CI uses this; locally it is optional):

```bash
uv run python scripts/pgo_release_wheels.py --variant accel \
  --profile conan/profiles/ubuntu-x86_64-gcc --clear
```

Use `--host-profile` / `--build-profile` instead of `--profile` if the host and
build contexts need to differ. The flags are forwarded to `pgo_build_wheel.py`.

## Header Layout

Numerical and simulation headers live under `include/pgo/core/`. Boundary modules
remain outside core: `include/pgo/io/` and `include/pgo/log/`.

Milestone 1 intentionally made this as a breaking include-path change and does
not provide compatibility forwarding headers. The C/Python API boundary is tracked
separately in `plan/c_py_api.md`.

## Logging

`pgo::log` is a lightweight facade for examples, tools, and the C API bridge.
The numerical core (`pgo::core`) does not link to logging. spdlog is optional,
enabled via the `-all` preset (`PGO_ENABLE_SPDLOG=ON`).

## Useful Notes

- `CMakeUserPresets.json` is ignored by git. The project recipe disables Conan's
  automatic user-preset generation so multiple toolchain folders cannot create
  duplicate preset names.
- Build outputs live under `build/` and are ignored by git.
