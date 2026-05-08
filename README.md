# pgo

## Requirements

- CMake 3.28 or newer
- Ninja
- Conan 2
- uv
- A C++23-capable compiler

Use uv for Python-managed tooling:

```bash
uv tool install conan
uv tool install ninja
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

## Configure Wrapper

Use the repository wrapper to install Conan dependencies for a CMake preset and
then run `cmake --preset`:

```bash
uv run python scripts/pgo_configure.py debug
```

The wrapper reads `CMakePresets.json`, derives the matching Conan output folder
from `CMAKE_TOOLCHAIN_FILE`, and uses `conan/profiles/default` unless a profile
override is provided:

```bash
uv run python scripts/pgo_configure.py debug-acceleration --profile conan/profiles/macos-arm64-apple-clang
uv run python scripts/pgo_configure.py release --host-profile conan/profiles/default --build-profile conan/profiles/default
uv run python scripts/pgo_configure.py debug --dry-run
```

## Build: Debug

```bash
uv run python scripts/pgo_configure.py debug
cmake --build --preset debug
```

## Build: Release

```bash
uv run python scripts/pgo_configure.py release
cmake --build --preset release
```

## Build: ASan/UBSan

ASan/UBSan uses the debug Conan toolchain and enables `PGO_ENABLE_SANITIZERS`.

```bash
uv run python scripts/pgo_configure.py asan
cmake --build --preset asan
```

The sanitizer preset is intended for Clang/GCC-style toolchains. MSVC sanitizer
support is not configured yet.

## Eigen Acceleration

`PGO_ENABLE_EIGEN_ACCELERATION` controls Eigen's BLAS/LAPACK acceleration path.
It does not select the sparse linear solver used by Newton iterations.

The default presets keep acceleration off:

```bash
uv run python scripts/pgo_configure.py debug
```

Acceleration presets use `PGO_EIGEN_ACCELERATION_BACKEND=AUTO`. On Apple
platforms, `AUTO` selects the system Accelerate framework:

```bash
uv run python scripts/pgo_configure.py debug-acceleration --profile conan/profiles/macos-arm64-apple-clang
cmake --build --preset debug-acceleration
ctest --preset debug-acceleration -R EigenConfig
```

On Linux and Windows, `AUTO` selects MKL. Install oneMKL once through the
project setup script if the machine does not already have it. The CMake
configuration checks the standard oneMKL install locations directly.

```bash
scripts/install-onemkl/install-onemkl-linux.sh

uv run python scripts/pgo_configure.py debug-acceleration --profile conan/profiles/ubuntu-x86_64-gcc
cmake --build --preset debug-acceleration
ctest --preset debug-acceleration -R EigenConfig
```

On Windows, use `.\scripts\install-onemkl\install-onemkl-windows.ps1` for the setup step.

Note: accelerated builds link to oneMKL runtime DLLs. The setup scripts prepare
the CI environment automatically. For local Windows benchmark runs, use the same
PowerShell session after running the setup script, or make sure oneMKL's
`mkl\latest\bin` and `compiler\latest\bin` directories are on `PATH`. For local
Linux runs, make sure `mkl/latest/lib/intel64` is on `LD_LIBRARY_PATH` if your
system linker does not already know that location.

MKL is a math library suite. PARDISO is MKL's sparse direct solver.
`Eigen::PardisoLDLT` is Eigen's wrapper around MKL PARDISO. PGO will model
PARDISO as a linear solver backend, not as a MathBackend.

Future solver policy work may add options such as:

```text
PGO_CPU_LINEAR_SOLVER=EIGEN_SIMPLICIAL_LDLT
PGO_CPU_LINEAR_SOLVER=MKL_PARDISO
PGO_CPU_LINEAR_SOLVER=EIGEN_CONJUGATE_GRADIENT
```

Milestone 1 defaults to Eigen's `SimplicialLDLT` until the solver policy layer
is introduced.

## Tests

Run the configured test preset after building:

```bash
ctest --preset debug
ctest --preset asan
```

## Benchmarks

Benchmarks use Google Benchmark and are built as the standalone
`pgo_benchmarks` executable. They are not registered as `ctest` tests because
performance numbers are machine- and load-dependent.
CI runs them with a short warm-up, repeated measurements, and aggregate-only
reporting so the logs are useful for trend checks without pretending to be
dedicated performance lab results.

Build and run the baseline Eigen path:

```bash
uv run python scripts/pgo_configure.py release
cmake --build --preset release --target pgo_benchmarks
./build/release/benchmarks/pgo_benchmarks
```

Build and run the acceleration path:

```bash
uv run python scripts/pgo_configure.py release-acceleration --profile conan/profiles/macos-arm64-apple-clang
cmake --build --preset release-acceleration --target pgo_benchmarks
./build/release-acceleration/benchmarks/pgo_benchmarks
```

On Linux and Windows acceleration builds, run the matching oneMKL setup script
once if oneMKL is not already installed.

## Useful Notes

- `CMakeUserPresets.json` is ignored by git. The project recipe disables Conan's
  automatic user-preset generation so multiple toolchain folders cannot create
  duplicate preset names.
- Build outputs live under `build/` and are ignored by git.
