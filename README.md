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
| `-all` | all optional compile-time deps (spdlog, Alembic, TBB) | core deps only |
| `-asan` | address + undefined-behavior sanitizers | no sanitizers |

ASan presets share the Conan dependency folder of their non-ASan counterpart
(e.g. `debug-asan` reuses `build/conan/debug/`), because sanitizers are pure
compiler flags and do not change the dependency graph.

`-all` enables optional compile-time dependencies: spdlog, Alembic, and the
TBB-backed `pgo::parallel_runtime` task runtime. `pgo::core` links it
transitively, so any consumer of `pgo::core` (examples, solver, integrator)
automatically gets the `parallel_for` abstraction with TBB or serial fallback
depending on `PGO_ENABLE_TBB`.

The `pypgo-release-all` / `pypgo-release-accel-all` presets also inherit TBB.
Both Python presets build the `pgo` distribution. The default preset exercises
the no-MKL fallback path; the accel preset enables Eigen acceleration and uses
MKL/Accelerate when present.

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
uv run python scripts/pgo_release_wheels.py --clear
uv run python scripts/pgo_release_wheels.py pypgo-release-accel-all --clear
```

The release wrapper builds from a temporary source tree and always publishes the
`pgo` distribution. Use `pypgo-release-all` for the fallback/no-MKL wheel path
and `pypgo-release-accel-all` for the accelerated wheel path.

Pass an explicit Conan profile when the default detected profile is not what
you want (CI uses this; locally it is optional):

```bash
uv run python scripts/pgo_release_wheels.py pypgo-release-accel-all \
  --profile conan/profiles/ubuntu-x86_64-gcc --clear --install
```

Use `--host-profile` / `--build-profile` instead of `--profile` if the host and
build contexts need to differ. The flags are forwarded to `pgo_build_wheel.py`.

Install the wheel with uv pip:

```bash
uv pip install dist/pgo/*.whl
```

## Examples

C API examples build as standalone C99 executables linked to `pgo_c`:

```bash
cmake --build --preset debug --target pgo_c_mass_spring_cloth
./build/debug/examples/c_api/pgo_c_mass_spring_cloth --frames 5 --resolution 8
```

Python examples use the public `pgo.World` API:

```bash
uv run python examples/python/mass_spring_cloth.py --frames 5 --resolution 8
```

Bunny OBJ examples require a local OBJ mesh and support `--pinned` / `--pinned-file`:

```bash
./build/debug/examples/c_api/pgo_c_mass_spring_bunny_cloth --input assets/model/bunny.obj --frames 5
uv run python examples/python/mass_spring_bunny_cloth.py --input assets/model/bunny.obj --frames 5
```

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

### MSVC Project Options

On MSVC builds, PGO unconditionally enables `/MP`, `/bigobj`, and
`/Zc:__cplusplus` to improve parallel compile throughput, template-heavy COFF
section limits, and `__cplusplus` macro fidelity.

CPU ISA selection is controlled by `PGO_MSVC_ARCH`:

```bash
cmake --preset release -DPGO_MSVC_ARCH=AVX512
```

Resolution order:

1. Explicit `AVX` / `AVX2` / `AVX512` always wins.
2. `DEFAULT` combined with `PGO_ENABLE_NATIVE_ARCH=ON` (the default) falls back
   to `AVX2`, preserving the previous hard-coded behavior.
3. `DEFAULT` combined with `PGO_ENABLE_NATIVE_ARCH=OFF` keeps the MSVC default
   ISA — preferred when shipping binaries to unknown Windows machines.

### Release Debug Symbols

Release builds omit extra project-level debug symbols by default. For profiling
or optimized-build crash backtraces, configure with:

```bash
cmake --preset release -DPGO_ENABLE_RELEASE_DEBUG_SYMBOLS=ON
```

This keeps optimization enabled while adding debug information. It may increase
object and binary sizes.

## TBB Parallel Runtime

`PGO_ENABLE_TBB` enables the `pgo::parallel_runtime` backend.
`pgo::core` already links it transitively, so any consumer of `pgo::core`
(examples, solver, integrator, c_api, python ext) can include
`<pgo/parallel/parallel_for.hpp>` and call `pgo::parallel::parallel_for(...)`
without an extra `target_link_libraries` line. With `PGO_ENABLE_TBB=OFF` the
same call resolves to a serial fallback.

Compile-time:

```bash
uv run pgo-configure release-all --build
```

Runtime:

```cpp
pgo::parallel::set_thread_count(0); // default scheduler
pgo::parallel::set_thread_count(1); // serial/debug mode
pgo::parallel::set_thread_count(8); // cap max parallelism
```

Threading policy:

- Use TBB for outer simulation loops such as particles, springs, elements, and contact pairs.
- Keep Eigen/BLAS acceleration controlled by the `-accel` presets.
- Avoid nested oversubscription: do not blindly combine TBB outer loops with multi-threaded BLAS kernels in the same hot path.

Python wheels (`pypgo-release-all`, `pypgo-release-accel-all`) bundle required
runtime libraries inside the `pgo/` package directory with `@loader_path` /
`$ORIGIN` RPATH, so `import pgo` works without a system-wide TBB install. MKL
runtime libraries are bundled when the accel wheel is built against oneMKL.

## Useful Notes

- `CMakeUserPresets.json` is ignored by git. The project recipe disables Conan's
  automatic user-preset generation so multiple toolchain folders cannot create
  duplicate preset names.
- Build outputs live under `build/` and are ignored by git.
