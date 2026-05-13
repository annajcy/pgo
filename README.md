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

After cloning, sync the project's Python environment (installs the `pgo-configure` entry point):

```bash
uv sync
```

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
| `release-asan` | Release | | | ✓ |
| `release-accel-asan` | Release | ✓ | | ✓ |
| `release-all-asan` | Release | | ✓ | ✓ |
| `release-accel-all-asan` | Release | ✓ | ✓ | ✓ |

## Quick Start

Configure, build, and run tests with a single wrapper:

```bash
uv run pgo-configure debug      # install Conan deps + cmake --preset
cmake --build --preset debug    # compile
ctest --preset debug            # run tests
```

Choose a different preset to switch configurations:

```bash
uv run pgo-configure release              # release build
uv run pgo-configure debug-asan           # debug + sanitizers
uv run pgo-configure debug-all            # debug + spdlog + Alembic
uv run pgo-configure debug-accel          # debug + Eigen acceleration
```

Or configure every visible preset at once:

```bash
uv run pgo-configure --all-presets
```

See [Preset Reference](#preset-reference) for the full 16-preset matrix.

## Acceleration

`-accel` presets enable Eigen BLAS/LAPACK acceleration via `PGO_EIGEN_ACCELERATION_BACKEND=AUTO`:

| Platform | Backend | Setup |
|----------|---------|-------|
| macOS | Accelerate (built-in) | none |
| Linux | oneMKL | `scripts/install-onemkl/install-onemkl-linux.sh` |
| Windows | oneMKL | `.\scripts\install-onemkl\install-onemkl-windows.ps1` |

```bash
uv run pgo-configure debug-accel
cmake --build --preset debug-accel
ctest --preset debug-accel
```

## Benchmarks

Benchmarks use Google Benchmark and are not registered as `ctest` tests (results
are machine-dependent). Build and run with a release preset:

```bash
uv run pgo-configure release
cmake --build --preset release --target pgo_benchmarks
./build/release/benchmarks/pgo_benchmarks
```

Accelerated benchmarks:

```bash
uv run pgo-configure release-accel
cmake --build --preset release-accel --target pgo_benchmarks
./build/release-accel/benchmarks/pgo_benchmarks
```

## Logging

`pgo::log` is a lightweight facade for examples, tools, and the C API bridge.
The numerical core (`pgo::core`) does not link to logging. spdlog is optional,
enabled via the `-all` preset (`PGO_ENABLE_SPDLOG=ON`).

## Useful Notes

- `CMakeUserPresets.json` is ignored by git. The project recipe disables Conan's
  automatic user-preset generation so multiple toolchain folders cannot create
  duplicate preset names.
- Build outputs live under `build/` and are ignored by git.
