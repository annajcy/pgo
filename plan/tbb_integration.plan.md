# TBB Integration Implementation Plan

> **给 agentic workers:** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务执行。本计划使用 checkbox (`- [ ]`) 语法追踪进度。

**目标:** 为 PGO 引入可选 TBB task-parallel runtime，提供 `pgo::parallel` 抽象层，让 simulation/assembly 外层循环可以在 TBB 与串行 fallback 之间切换，并同步整理 Eigen threading / MKL / alignment、MSVC 编译选项、Release debug symbols 策略。

**架构:** TBB 不进入 `pgo::core` 默认依赖；需要并行能力的 target 显式链接 `pgo::parallel`。编译期只决定是否启用 TBB，线程数属于运行时配置，通过 `pgo::parallel::set_thread_count(...)` 控制。Eigen/BLAS 仍由现有 `pgo::eigen_config` 和 `release-accel` 体系负责；Eigen 内部线程默认禁用，避免与 TBB 外层任务并行抢线程。

**Tech Stack:** C++23、CMake Presets、Conan 2、oneTBB/TBB、GoogleTest、Google Benchmark、Eigen。

---

## 0. 核心设计决策

- `PGO_ENABLE_TBB` 是编译期开关，默认 `OFF`。
- `enable_tbb` 是 Conan option，默认 `False`；开启时引入 `onetbb` 包。
- TBB 归入现有 `-all` preset 家族：`debug-all`、`debug-accel-all`、`release-all`、`release-accel-all` 会启用 TBB；不新增独立 TBB-only preset 维度。
- `EIGEN_DONT_PARALLELIZE` 默认开启，表示 Eigen 自己不创建内部 worker threads；外层并行由 TBB 管。
- `EIGEN_DONT_PARALLELIZE` 不等于禁用 MKL/Accelerate。MKL/Accelerate backend 仍由 `PGO_ENABLE_EIGEN_ACCELERATION` 和 `PGO_EIGEN_ACCELERATION_BACKEND` 控制。
- `EIGEN_MKL_NO_DIRECT_CALL` 仅在 MKL backend 下可选启用，默认 `ON`，用于减少 Eigen 对 MKL direct-call path 的耦合。
- `EIGEN_MAX_ALIGN_BYTES` 不默认硬设；通过 `PGO_EIGEN_MAX_ALIGN_BYTES` cache string 显式覆盖，空字符串保留 Eigen/platform 默认。
- MSVC 默认启用 `/MP`、`/bigobj`、`/Zc:__cplusplus`，提升 Windows 构建体验和模板代码兼容性。
- MSVC CPU ISA 不默认固定 AVX2；通过 `PGO_MSVC_ARCH` 显式选择 `DEFAULT`、`AVX`、`AVX2` 或 `AVX512`。
- `PGO_MSVC_ARCH=DEFAULT` 不添加 `/arch:*`，适合可移植 Windows binary；本地性能测试可以显式使用 `-DPGO_MSVC_ARCH=AVX2` 或 `AVX512`。
- Release debug symbols 通过 `PGO_ENABLE_RELEASE_DEBUG_SYMBOLS` 显式开启，默认 `OFF`；用于 profile、crash backtrace 和优化构建调试。
- `PGO_ENABLE_RELEASE_DEBUG_SYMBOLS=ON` 在 GNU/Clang/AppleClang Release 下加 `-g`，在 MSVC Release 下加 `/Zi` 和 linker debug info；不把 `-ggdb3` 无条件塞进所有 Release 构建。
- CMake 只在 `PGO_ENABLE_TBB=ON` 时 `find_package(TBB REQUIRED CONFIG)`。
- 新增 `pgo_parallel` / `pgo::parallel` interface target。
- `pgo::parallel` 在 TBB ON 时链接 `TBB::tbb` 并定义 `PGO_ENABLE_TBB`；TBB OFF 时不链接额外库。
- 业务代码不直接依赖 `tbb::parallel_for`；统一调用 `pgo::parallel::parallel_for(...)`。
- `thread_count` 是运行时配置，不是 CMake cache variable。
- `thread_count == 0` 表示使用 runtime 默认并行度。
- `thread_count == 1` 表示强制串行执行，方便 debug 和 benchmark 对照。
- `thread_count > 1` 表示限制 TBB 最大并行度。
- TBB 后端通过 `tbb::global_control` 管理最大并行度；对象生命周期由 `.cpp` 内部静态状态持有。
- 第一阶段只提供 contiguous index range 的 `parallel_for(begin, end, func)`，不引入 task_group、flow_graph、parallel_sort、allocator 等额外能力。
- 第一阶段 benchmark 用 synthetic particle/spring style workload 测 wrapper overhead 和可扩展性，不把 benchmark 结论扩大到完整 solver。

## 1. 目标文件结构

```text
.
  CMakeLists.txt
  CMakePresets.json
  conanfile.py
  cmake/
    pgo_dependencies.cmake
    pgo_eigen.cmake
    pgo_options.cmake
    pgo_parallel.cmake
    pgo_project_options.cmake
  include/
    pgo/
      parallel/
        parallel_for.hpp
        runtime.hpp
  src/
    parallel/
      CMakeLists.txt
      runtime.cpp
  tests/
    parallel/
      test_parallel_for.cpp
  benchmarks/
    parallel/
      bench_parallel_for.cpp
  scripts/
    pgo_configure.py
```

## Phase 1: Dependency And Build Integration

### Task 1.1: Add Conan option and dependency

**Files:**
- Modify: `conanfile.py`

- [ ] **Step 1: Write the intended Conan option change**

Edit `conanfile.py` so the option blocks include `enable_tbb`:

```python
    options = {
        "enable_spdlog": [True, False],
        "enable_alembic": [True, False],
        "enable_tbb": [True, False],
    }
    default_options = {
        "enable_spdlog": False,
        "enable_alembic": False,
        "enable_tbb": False,
    }
```

Then add the optional dependency in `requirements()`:

```python
        if self.options.enable_tbb:
            self.requires("onetbb/[>=2021.12 <2023]")
```

- [ ] **Step 2: Verify Conan dry-run without TBB**

Run:

```bash
uv run pgo-configure release --dry-run
```

Expected: printed `conan install` command does not contain `enable_tbb=True`.

- [ ] **Step 3: Inspect changed files**

Run:

```bash
git diff -- conanfile.py
git status --short conanfile.py
```

Expected: only the intended Conan option and optional dependency changes appear.

### Task 1.2: Add CMake option, dependency lookup, and parallel target

**Files:**
- Modify: `cmake/pgo_options.cmake`
- Modify: `cmake/pgo_dependencies.cmake`
- Create: `cmake/pgo_parallel.cmake`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add CMake option**

Add this to `cmake/pgo_options.cmake` near the other feature options:

```cmake
option(PGO_ENABLE_TBB "Enable TBB-backed task parallelism" OFF)
```

- [ ] **Step 2: Add dependency lookup**

Add this to `cmake/pgo_dependencies.cmake`:

```cmake
if(PGO_ENABLE_TBB)
    find_package(TBB REQUIRED CONFIG)
endif()
```

- [ ] **Step 3: Create the parallel target module**

Create `cmake/pgo_parallel.cmake`:

```cmake
add_library(pgo_parallel INTERFACE)
add_library(pgo::parallel ALIAS pgo_parallel)

target_include_directories(pgo_parallel INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(pgo_parallel INTERFACE cxx_std_23)

if(PGO_ENABLE_TBB)
    target_link_libraries(pgo_parallel INTERFACE TBB::tbb)
    target_compile_definitions(pgo_parallel INTERFACE PGO_ENABLE_TBB)
endif()
```

- [ ] **Step 4: Include the module from the top-level CMake**

In `CMakeLists.txt`, include `cmake/pgo_parallel.cmake` after dependencies and options are loaded:

```cmake
include(cmake/pgo_parallel.cmake)
```

- [ ] **Step 5: Configure without TBB**

Run:

```bash
cmake --preset release
```

Expected: configure succeeds without looking for `TBB::tbb`.

- [ ] **Step 6: Configure with TBB using a temporary build folder**

Run:

```bash
conan install . \
  --profile:host=conan/profiles/default \
  --profile:build=conan/profiles/default \
  --output-folder=build/conan/tbb-probe \
  -s:h build_type=Release \
  -o:h enable_tbb=True \
  --build=missing

cmake -S . -B build/tbb-probe -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=build/conan/tbb-probe/conan_toolchain.cmake \
  -DPGO_ENABLE_TBB=ON
```

Expected: configure succeeds and `TBB::tbb` is available.

- [ ] **Step 7: Inspect changed files**

Run:

```bash
git diff -- CMakeLists.txt cmake/pgo_options.cmake cmake/pgo_dependencies.cmake cmake/pgo_parallel.cmake
git status --short CMakeLists.txt cmake/pgo_options.cmake cmake/pgo_dependencies.cmake cmake/pgo_parallel.cmake
```

Expected: only the intended CMake option, dependency lookup, and parallel target changes appear.

### Task 1.3: Teach pgo-configure to forward PGO_ENABLE_TBB

**Files:**
- Modify: `scripts/pgo_configure.py`

- [ ] **Step 1: Add option forwarding**

Update `CMAKE_TO_CONAN_OPTIONS`:

```python
CMAKE_TO_CONAN_OPTIONS: dict[str, tuple[str, dict[str, str]]] = {
    "PGO_ENABLE_SPDLOG": ("enable_spdlog", {"ON": "True", "OFF": "False"}),
    "PGO_ENABLE_ALEMBIC": ("enable_alembic", {"ON": "True", "OFF": "False"}),
    "PGO_ENABLE_TBB": ("enable_tbb", {"ON": "True", "OFF": "False"}),
}
```

- [ ] **Step 2: Verify dry-run forwarding**

After Task 1.4 adds presets, run:

```bash
uv run pgo-configure release-all --dry-run
```

Expected: printed `conan install` command includes `-o:h enable_tbb=True`.

- [ ] **Step 3: Inspect changed files**

Run:

```bash
git diff -- scripts/pgo_configure.py
git status --short scripts/pgo_configure.py
```

Expected: only the intended `PGO_ENABLE_TBB` forwarding change appears.

### Task 1.4: Fold TBB into the existing all presets

**Files:**
- Modify: `CMakePresets.json`
- Modify: `README.md`

- [ ] **Step 1: Add TBB to the all preset fragment**

Update the existing hidden `all-opt` fragment in `CMakePresets.json`:

```json
{
  "name": "all-opt",
  "hidden": true,
  "cacheVariables": {
    "PGO_ENABLE_SPDLOG": "ON",
    "PGO_ENABLE_ALEMBIC": "ON",
    "PGO_ENABLE_TBB": "ON"
  }
}
```

This makes the existing presets inherit TBB automatically:

```text
debug-all
debug-accel-all
debug-all-asan
debug-accel-all-asan
release-all
release-accel-all
release-all-asan
release-accel-all-asan
```

- [ ] **Step 2: Document TBB as part of all presets**

In `README.md`, update the preset matrix and `-all` description:

```markdown
`-all` enables optional compile-time dependencies: spdlog, Alembic, and the
TBB-backed `pgo::parallel` runtime. Code that needs task parallelism should link
`pgo::parallel_runtime`; `pgo::core` remains usable without TBB.
```

- [ ] **Step 3: Verify all preset Conan forwarding**

Run:

```bash
uv run pgo-configure release-all --dry-run
```

Expected: printed `conan install` command includes `-o:h enable_tbb=True`.

- [ ] **Step 4: Verify preset listing remains compact**

Run:

```bash
cmake --list-presets
```

Expected: no new TBB-specific visible presets are required; existing `debug-all` and `release-all` remain visible and now inherit TBB through `all-opt`.

- [ ] **Step 5: Inspect changed files**

Run:

```bash
git diff -- CMakePresets.json README.md
git status --short CMakePresets.json README.md
```

Expected: only the intended preset and README changes appear.

## Phase 2: MSVC Project Options Policy

### Task 2.1: Add configurable MSVC CPU ISA option

**Files:**
- Modify: `cmake/pgo_project_options.cmake`

- [ ] **Step 1: Add the MSVC architecture cache variable**

Near the top of `cmake/pgo_project_options.cmake`, after `PGO_ENABLE_NATIVE_ARCH`, add:

```cmake
set(PGO_MSVC_ARCH "DEFAULT" CACHE STRING "MSVC CPU ISA: DEFAULT, AVX, AVX2, AVX512")
set_property(CACHE PGO_MSVC_ARCH PROPERTY STRINGS DEFAULT AVX AVX2 AVX512)
```

- [ ] **Step 2: Replace the hard-coded MSVC AVX2 branch**

Replace the existing MSVC branch:

```cmake
elseif(MSVC)
    target_compile_options(pgo_project_options INTERFACE
        $<$<CONFIG:Release>:/arch:AVX2>
    )
endif()
```

with:

```cmake
elseif(MSVC)
    target_compile_options(pgo_project_options INTERFACE
        /MP
        /bigobj
        /Zc:__cplusplus
    )

    if(PGO_MSVC_ARCH STREQUAL "AVX")
        target_compile_options(pgo_project_options INTERFACE
            $<$<CONFIG:Release>:/arch:AVX>
        )
    elseif(PGO_MSVC_ARCH STREQUAL "AVX2")
        target_compile_options(pgo_project_options INTERFACE
            $<$<CONFIG:Release>:/arch:AVX2>
        )
    elseif(PGO_MSVC_ARCH STREQUAL "AVX512")
        target_compile_options(pgo_project_options INTERFACE
            $<$<CONFIG:Release>:/arch:AVX512>
        )
    elseif(PGO_MSVC_ARCH STREQUAL "DEFAULT")
        # Keep the MSVC compiler default ISA for portable Windows binaries.
    else()
        message(FATAL_ERROR "Unknown PGO_MSVC_ARCH=${PGO_MSVC_ARCH}")
    endif()
endif()
```

- [ ] **Step 3: Verify non-MSVC configure still works**

Run:

```bash
cmake --preset release
```

Expected: configure succeeds on non-MSVC platforms and ignores `PGO_MSVC_ARCH`.

- [ ] **Step 4: Inspect changed files**

Run:

```bash
git diff -- cmake/pgo_project_options.cmake
git status --short cmake/pgo_project_options.cmake
```

Expected: hard-coded `/arch:AVX2` is gone; `/MP`, `/bigobj`, `/Zc:__cplusplus`, and `PGO_MSVC_ARCH` mapping are present.

### Task 2.2: Document Windows compiler option policy

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add Windows compiler options note**

Add this note near the build/preset documentation:

```markdown
### MSVC Project Options

On MSVC builds, PGO enables `/MP`, `/bigobj`, and `/Zc:__cplusplus` by default.
CPU ISA flags are explicit rather than hard-coded. Use:

```bash
cmake --preset release -DPGO_MSVC_ARCH=AVX2
```

Valid values are `DEFAULT`, `AVX`, `AVX2`, and `AVX512`. `DEFAULT` keeps the
compiler's portable default ISA and is preferred for binaries distributed to
unknown Windows machines.
```

- [ ] **Step 2: Inspect changed files**

Run:

```bash
git diff -- README.md
git status --short README.md
```

Expected: README describes MSVC default engineering flags and explicit CPU ISA selection.

### Task 2.3: Add optional Release debug symbols

**Files:**
- Modify: `cmake/pgo_project_options.cmake`
- Modify: `README.md`

- [ ] **Step 1: Add the Release debug symbols option**

Near the other project options in `cmake/pgo_project_options.cmake`, add:

```cmake
option(PGO_ENABLE_RELEASE_DEBUG_SYMBOLS "Emit debug symbols in release builds" OFF)
```

- [ ] **Step 2: Add compiler-specific Release debug symbol flags**

In `cmake/pgo_project_options.cmake`, after the compiler-specific native/MSVC option logic, add:

```cmake
if(PGO_ENABLE_RELEASE_DEBUG_SYMBOLS)
    if(MSVC)
        target_compile_options(pgo_project_options INTERFACE
            $<$<CONFIG:Release>:/Zi>
        )
        target_link_options(pgo_project_options INTERFACE
            $<$<CONFIG:Release>:/DEBUG>
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(pgo_project_options INTERFACE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C>>:-g>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-g>
        )
        target_link_options(pgo_project_options INTERFACE
            $<$<CONFIG:Release>:-g>
        )
    endif()
endif()
```

Use `-g` instead of unconditional `-ggdb3` so AppleClang/GNU/Clang all get portable debug info. If a Linux-only workflow later needs heavier GDB metadata, add a separate `PGO_RELEASE_DEBUG_SYMBOL_LEVEL` option rather than changing this default.

- [ ] **Step 3: Verify default release does not emit project-level debug symbol flags**

Run:

```bash
cmake --preset release
node - <<'NODE'
const fs = require('fs');
const commands = JSON.parse(fs.readFileSync('build/release/compile_commands.json', 'utf8'));
const hasDebug = commands.some((entry) => {
  const command = entry.command || entry.arguments.join(' ');
  return /(^| )-g($| )/.test(command) || command.includes('/Zi');
});
if (hasDebug) {
  console.error('release debug symbols should be disabled by default');
  process.exit(1);
}
console.log('release debug symbols disabled by default');
NODE
```

Expected: script prints `release debug symbols disabled by default`.

- [ ] **Step 4: Verify opt-in release debug symbols**

Run:

```bash
cmake -S . -B build/release-debug-symbols-probe -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=build/conan/release/conan_toolchain.cmake \
  -DPGO_ENABLE_RELEASE_DEBUG_SYMBOLS=ON

node - <<'NODE'
const fs = require('fs');
const commands = JSON.parse(fs.readFileSync('build/release-debug-symbols-probe/compile_commands.json', 'utf8'));
const hit = commands.some((entry) => {
  const command = entry.command || entry.arguments.join(' ');
  return /(^| )-g($| )/.test(command) || command.includes('/Zi');
});
if (!hit) {
  console.error('release debug symbols were not found after opt-in');
  process.exit(1);
}
console.log('release debug symbols found after opt-in');
NODE
```

Expected: script prints `release debug symbols found after opt-in`.

- [ ] **Step 5: Document the option**

Add this README note near the build options section:

```markdown
### Release Debug Symbols

Release builds omit extra project-level debug symbols by default. For profiling
or optimized-build crash backtraces, configure with:

```bash
cmake --preset release -DPGO_ENABLE_RELEASE_DEBUG_SYMBOLS=ON
```

This keeps optimization enabled while adding debug information. It may increase
object and binary sizes.
```

- [ ] **Step 6: Inspect changed files**

Run:

```bash
git diff -- cmake/pgo_project_options.cmake README.md
git status --short cmake/pgo_project_options.cmake README.md
```

Expected: only the intended optional Release debug symbols logic and README note appear.

## Phase 3: Eigen Threading And Alignment Policy

### Task 3.1: Add Eigen threading and alignment options

**Files:**
- Modify: `cmake/pgo_options.cmake`

- [ ] **Step 1: Add Eigen policy options**

Add these options near the existing Eigen acceleration option:

```cmake
option(PGO_EIGEN_DONT_PARALLELIZE "Disable Eigen internal thread parallelism" ON)
option(PGO_EIGEN_MKL_NO_DIRECT_CALL "Disable Eigen direct MKL calls when MKL acceleration is enabled" ON)

set(PGO_EIGEN_MAX_ALIGN_BYTES "" CACHE STRING "Override Eigen max alignment bytes; empty keeps Eigen default")
```

- [ ] **Step 2: Inspect changed files**

Run:

```bash
git diff -- cmake/pgo_options.cmake
git status --short cmake/pgo_options.cmake
```

Expected: only the intended Eigen policy options appear.

### Task 3.2: Apply Eigen policy definitions on pgo_eigen_config

**Files:**
- Modify: `cmake/pgo_eigen.cmake`

- [ ] **Step 1: Add common Eigen definitions before backend selection**

After `target_link_libraries(pgo_eigen_config INTERFACE Eigen3::Eigen)`, add:

```cmake
if(PGO_EIGEN_DONT_PARALLELIZE)
    target_compile_definitions(pgo_eigen_config INTERFACE
        EIGEN_DONT_PARALLELIZE
    )
endif()

if(NOT PGO_EIGEN_MAX_ALIGN_BYTES STREQUAL "")
    target_compile_definitions(pgo_eigen_config INTERFACE
        EIGEN_MAX_ALIGN_BYTES=${PGO_EIGEN_MAX_ALIGN_BYTES}
    )
endif()
```

- [ ] **Step 2: Add MKL direct-call policy only in the MKL branch**

Inside the `PGO_SELECTED_EIGEN_ACCELERATION_BACKEND STREQUAL "MKL"` branch, after the existing `EIGEN_USE_MKL_ALL` definition, add:

```cmake
    if(PGO_EIGEN_MKL_NO_DIRECT_CALL)
        target_compile_definitions(pgo_eigen_config INTERFACE
            EIGEN_MKL_NO_DIRECT_CALL
        )
    endif()
```

Keep `EIGEN_MKL_NO_DIRECT_CALL` out of the Accelerate and NONE branches.

- [ ] **Step 3: Configure default release and inspect compile commands**

Run:

```bash
cmake --preset release
node - <<'NODE'
const fs = require('fs');
const commands = JSON.parse(fs.readFileSync('build/release/compile_commands.json', 'utf8'));
const eigenUsers = commands.filter((entry) => {
  const command = entry.command || entry.arguments.join(' ');
  return command.includes('PGO_EIGEN_ACCELERATION_NONE') || command.includes('EIGEN_USE_BLAS') || command.includes('EIGEN_USE_MKL_ALL');
});
const missing = eigenUsers.filter((entry) => {
  const command = entry.command || entry.arguments.join(' ');
  return !command.includes('EIGEN_DONT_PARALLELIZE');
});
if (missing.length !== 0) {
  console.error(`EIGEN_DONT_PARALLELIZE missing from ${missing.length} Eigen compile commands`);
  process.exit(1);
}
console.log(`EIGEN_DONT_PARALLELIZE found in ${eigenUsers.length} Eigen compile commands`);
NODE
```

Expected: script prints that `EIGEN_DONT_PARALLELIZE` is present in Eigen-using compile commands.

- [ ] **Step 4: Configure explicit alignment override and inspect compile commands**

Run:

```bash
cmake -S . -B build/eigen-align-probe -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=build/conan/release/conan_toolchain.cmake \
  -DPGO_EIGEN_MAX_ALIGN_BYTES=32

node - <<'NODE'
const fs = require('fs');
const commands = JSON.parse(fs.readFileSync('build/eigen-align-probe/compile_commands.json', 'utf8'));
const hit = commands.some((entry) => {
  const command = entry.command || entry.arguments.join(' ');
  return command.includes('EIGEN_MAX_ALIGN_BYTES=32');
});
if (!hit) {
  console.error('EIGEN_MAX_ALIGN_BYTES=32 was not found in compile commands');
  process.exit(1);
}
console.log('EIGEN_MAX_ALIGN_BYTES=32 found in compile commands');
NODE
```

Expected: alignment override is present only when explicitly requested.

- [ ] **Step 5: Inspect changed files**

Run:

```bash
git diff -- cmake/pgo_eigen.cmake
git status --short cmake/pgo_eigen.cmake
```

Expected: only the intended Eigen compile definitions appear.

### Task 3.3: Add Eigen policy tests

**Files:**
- Modify: `tests/math/test_eigen_config.cpp`

- [ ] **Step 1: Add macro visibility assertions**

Append these tests to `tests/math/test_eigen_config.cpp`:

```cpp
TEST(EigenConfig, EigenInternalParallelismIsDisabled) {
#if defined(EIGEN_DONT_PARALLELIZE)
    SUCCEED();
#else
    FAIL() << "EIGEN_DONT_PARALLELIZE should be defined by pgo::eigen_config";
#endif
}

TEST(EigenConfig, EigenAlignmentOverrideIsOptional) {
#if defined(EIGEN_MAX_ALIGN_BYTES)
    EXPECT_GT(EIGEN_MAX_ALIGN_BYTES, 0);
#else
    SUCCEED() << "default configuration keeps Eigen/platform alignment policy";
#endif
}
```

- [ ] **Step 2: Add MKL no-direct-call assertion guarded by backend**

Append this test to `tests/math/test_eigen_config.cpp`:

```cpp
TEST(EigenConfig, MklNoDirectCallOnlyAppliesToMklBackend) {
#if defined(PGO_EIGEN_ACCELERATION_MKL)
    #if defined(EIGEN_MKL_NO_DIRECT_CALL)
        SUCCEED();
    #else
        FAIL() << "MKL backend should define EIGEN_MKL_NO_DIRECT_CALL by default";
    #endif
#else
    #if defined(EIGEN_MKL_NO_DIRECT_CALL)
        FAIL() << "EIGEN_MKL_NO_DIRECT_CALL should not be defined outside the MKL backend";
    #else
        SUCCEED();
    #endif
#endif
}
```

- [ ] **Step 3: Run Eigen tests**

Run:

```bash
cmake --build --preset release --target pgo_tests
ctest --preset release --output-on-failure -R "EigenConfig"
```

Expected: Eigen config tests pass in the default non-accelerated configuration.

- [ ] **Step 4: Run all+accel Eigen tests**

Run:

```bash
uv run pgo-configure release-accel-all --build
ctest --preset release-accel-all --output-on-failure -R "EigenConfig"
```

Expected: Eigen config tests pass with acceleration and all optional dependencies enabled.

- [ ] **Step 5: Inspect changed files**

Run:

```bash
git diff -- tests/math/test_eigen_config.cpp
git status --short tests/math/test_eigen_config.cpp
```

Expected: only the intended Eigen policy tests appear.

## Phase 4: Runtime API And Serial Fallback

### Task 4.1: Add runtime thread-count API

**Files:**
- Create: `include/pgo/parallel/runtime.hpp`
- Create: `src/parallel/CMakeLists.txt`
- Create: `src/parallel/runtime.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the public header**

Create `include/pgo/parallel/runtime.hpp`:

```cpp
#pragma once

namespace pgo::parallel {

[[nodiscard]] bool is_tbb_enabled() noexcept;

void set_thread_count(int thread_count);
[[nodiscard]] int configured_thread_count() noexcept;

} // namespace pgo::parallel
```

- [ ] **Step 2: Add compiled runtime library**

Create `src/parallel/CMakeLists.txt`:

```cmake
add_library(pgo_parallel_runtime STATIC
    runtime.cpp
)
add_library(pgo::parallel_runtime ALIAS pgo_parallel_runtime)

target_compile_features(pgo_parallel_runtime PUBLIC cxx_std_23)
target_include_directories(pgo_parallel_runtime PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(pgo_parallel_runtime PUBLIC pgo::parallel)
```

Add this to top-level `CMakeLists.txt` after `add_subdirectory(src/log)`:

```cmake
add_subdirectory(src/parallel)
```

- [ ] **Step 3: Implement serial/TBB runtime state**

Create `src/parallel/runtime.cpp`:

```cpp
#include <pgo/parallel/runtime.hpp>

#if defined(PGO_ENABLE_TBB)
#include <oneapi/tbb/global_control.h>
#endif

#include <memory>
#include <mutex>
#include <stdexcept>

namespace pgo::parallel {
namespace {

std::mutex runtime_mutex;
int runtime_thread_count = 0;

#if defined(PGO_ENABLE_TBB)
std::unique_ptr<oneapi::tbb::global_control> runtime_global_control;
#endif

} // namespace

bool is_tbb_enabled() noexcept {
#if defined(PGO_ENABLE_TBB)
    return true;
#else
    return false;
#endif
}

void set_thread_count(int thread_count) {
    if (thread_count < 0) {
        throw std::invalid_argument("pgo::parallel::set_thread_count requires a non-negative thread count");
    }

    std::lock_guard<std::mutex> lock(runtime_mutex);
    runtime_thread_count = thread_count;

#if defined(PGO_ENABLE_TBB)
    runtime_global_control.reset();
    if (thread_count > 0) {
        runtime_global_control = std::make_unique<oneapi::tbb::global_control>(
            oneapi::tbb::global_control::max_allowed_parallelism,
            static_cast<std::size_t>(thread_count));
    }
#endif
}

int configured_thread_count() noexcept {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    return runtime_thread_count;
}

} // namespace pgo::parallel
```

- [ ] **Step 4: Build serial runtime**

Run:

```bash
cmake --build --preset release --target pgo_parallel_runtime
```

Expected: target builds with no TBB dependency.

- [ ] **Step 5: Inspect changed files**

Run:

```bash
git diff -- CMakeLists.txt include/pgo/parallel/runtime.hpp src/parallel/CMakeLists.txt src/parallel/runtime.cpp
git status --short CMakeLists.txt include/pgo/parallel/runtime.hpp src/parallel/CMakeLists.txt src/parallel/runtime.cpp
```

Expected: only the intended runtime API, runtime implementation, and source subdirectory changes appear.

### Task 4.2: Add parallel_for wrapper

**Files:**
- Create: `include/pgo/parallel/parallel_for.hpp`

- [ ] **Step 1: Write the wrapper header**

Create `include/pgo/parallel/parallel_for.hpp`:

```cpp
#pragma once

#include <pgo/parallel/runtime.hpp>

#if defined(PGO_ENABLE_TBB)
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#endif

#include <type_traits>

namespace pgo::parallel {

template <class Index, class Func>
void serial_for(Index begin, Index end, Func&& func) {
    static_assert(std::is_integral_v<Index>, "pgo::parallel::serial_for requires an integral index type");
    for (Index i = begin; i < end; ++i) {
        func(i);
    }
}

template <class Index, class Func>
void parallel_for(Index begin, Index end, Func&& func) {
    static_assert(std::is_integral_v<Index>, "pgo::parallel::parallel_for requires an integral index type");

    if (end <= begin) {
        return;
    }

    if (configured_thread_count() == 1) {
        serial_for(begin, end, static_cast<Func&&>(func));
        return;
    }

#if defined(PGO_ENABLE_TBB)
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<Index>(begin, end),
        [&](const oneapi::tbb::blocked_range<Index>& range) {
            for (Index i = range.begin(); i < range.end(); ++i) {
                func(i);
            }
        });
#else
    serial_for(begin, end, static_cast<Func&&>(func));
#endif
}

} // namespace pgo::parallel
```

- [ ] **Step 2: Build a target that consumes the header indirectly**

Run:

```bash
cmake --build --preset release --target pgo_parallel_runtime
```

Expected: build succeeds.

- [ ] **Step 3: Inspect changed files**

Run:

```bash
git diff -- include/pgo/parallel/parallel_for.hpp
git status --short include/pgo/parallel/parallel_for.hpp
```

Expected: only the intended `parallel_for` wrapper change appears.

## Phase 5: Tests

### Task 5.1: Add serial fallback tests

**Files:**
- Create: `tests/parallel/test_parallel_for.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write tests before implementation is considered complete**

Create `tests/parallel/test_parallel_for.cpp`:

```cpp
#include <pgo/parallel/parallel_for.hpp>
#include <pgo/parallel/runtime.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

TEST(ParallelFor, EmptyRangeDoesNothing) {
    int count = 0;
    pgo::parallel::parallel_for(5, 5, [&](int) { ++count; });
    EXPECT_EQ(count, 0);
}

TEST(ParallelFor, VisitsEachIndexOnce) {
    std::vector<int> visits(128, 0);
    pgo::parallel::set_thread_count(1);

    pgo::parallel::parallel_for(0, static_cast<int>(visits.size()), [&](int index) {
        visits[static_cast<std::size_t>(index)] += 1;
    });

    for (int visit : visits) {
        EXPECT_EQ(visit, 1);
    }
}

TEST(ParallelRuntime, RejectsNegativeThreadCount) {
    EXPECT_THROW(pgo::parallel::set_thread_count(-1), std::invalid_argument);
}

TEST(ParallelRuntime, StoresConfiguredThreadCount) {
    pgo::parallel::set_thread_count(0);
    EXPECT_EQ(pgo::parallel::configured_thread_count(), 0);

    pgo::parallel::set_thread_count(1);
    EXPECT_EQ(pgo::parallel::configured_thread_count(), 1);
}

} // namespace
```

- [ ] **Step 2: Add the test to CMake**

In `tests/CMakeLists.txt`, add `parallel/test_parallel_for.cpp` to `pgo_tests` sources:

```cmake
add_executable(pgo_tests
    assembly/test_cpu_assembler.cpp
    base/test_assert.cpp
    dof/test_dof.cpp
    energy/test_assembled_energy.cpp
    energy/test_constant_force_energy.cpp
    energy/test_energy_concepts.cpp
    energy/test_energy_sum.cpp
    energy/test_mass_spring_local_energy_provider.cpp
    energy/test_reduced_energy.cpp
    geometry/test_rest_mesh.cpp
    io/test_obj_io.cpp
    math/test_backend.cpp
    math/test_eigen_config.cpp
    math/test_finite_difference.cpp
    parallel/test_parallel_for.cpp
    solver/test_line_search.cpp
    integrator/test_backward_euler.cpp
    solver/test_solver.cpp
    storage/test_storage.cpp
)
```

Link `pgo::parallel_runtime` into `pgo_tests`:

```cmake
target_link_libraries(pgo_tests
    PRIVATE
        pgo::core
        pgo::io
        pgo::parallel_runtime
        GTest::gtest_main
)
```

- [ ] **Step 3: Run tests without TBB**

Run:

```bash
cmake --build --preset release --target pgo_tests
ctest --preset release --output-on-failure -R "Parallel"
```

Expected: all `ParallelFor` and `ParallelRuntime` tests pass in serial fallback mode.

- [ ] **Step 4: Inspect changed files**

Run:

```bash
git diff -- tests/CMakeLists.txt tests/parallel/test_parallel_for.cpp
git status --short tests/CMakeLists.txt tests/parallel/test_parallel_for.cpp
```

Expected: only the intended serial fallback tests and test target link changes appear.

### Task 5.2: Add TBB-enabled test coverage

**Files:**
- Modify: `tests/parallel/test_parallel_for.cpp`

- [ ] **Step 1: Add TBB backend assertions**

Append these tests to `tests/parallel/test_parallel_for.cpp`:

```cpp
TEST(ParallelRuntime, ReportsCompiledBackend) {
#if defined(PGO_ENABLE_TBB)
    EXPECT_TRUE(pgo::parallel::is_tbb_enabled());
#else
    EXPECT_FALSE(pgo::parallel::is_tbb_enabled());
#endif
}

TEST(ParallelFor, DefaultThreadCountProducesCorrectReduction) {
    pgo::parallel::set_thread_count(0);

    std::vector<int> values(4096, 0);
    pgo::parallel::parallel_for(0, static_cast<int>(values.size()), [&](int index) {
        values[static_cast<std::size_t>(index)] = index % 7;
    });

    const int sum = std::accumulate(values.begin(), values.end(), 0);
    int expected = 0;
    for (int index = 0; index < static_cast<int>(values.size()); ++index) {
        expected += index % 7;
    }
    EXPECT_EQ(sum, expected);
}
```

- [ ] **Step 2: Run tests with TBB**

Run:

```bash
uv run pgo-configure release-all --build
ctest --preset release-all --output-on-failure -R "Parallel"
```

Expected: tests pass, and `ParallelRuntime.ReportsCompiledBackend` observes `PGO_ENABLE_TBB`.

- [ ] **Step 3: Inspect changed files**

Run:

```bash
git diff -- tests/parallel/test_parallel_for.cpp
git status --short tests/parallel/test_parallel_for.cpp
```

Expected: only the intended TBB backend test additions appear.

## Phase 6: Benchmark

### Task 6.1: Add synthetic particle/spring parallel benchmark

**Files:**
- Create: `benchmarks/parallel/bench_parallel_for.cpp`
- Modify: `benchmarks/CMakeLists.txt`

- [ ] **Step 1: Add benchmark source**

Create `benchmarks/parallel/bench_parallel_for.cpp`:

```cpp
#include <pgo/parallel/parallel_for.hpp>
#include <pgo/parallel/runtime.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

struct Particle {
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
};

std::vector<Particle> make_particles(std::size_t count) {
    std::vector<Particle> particles(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto t = static_cast<double>(i);
        particles[i] = Particle{t * 0.001, t * 0.002, t * 0.003, 0.0, 0.0, 0.0};
    }
    return particles;
}

void benchmark_particle_update(benchmark::State& state) {
    const auto particle_count = static_cast<std::size_t>(state.range(0));
    const auto thread_count = static_cast<int>(state.range(1));
    auto particles = make_particles(particle_count);

    pgo::parallel::set_thread_count(thread_count);

    for (auto _ : state) {
        pgo::parallel::parallel_for(std::size_t{0}, particles.size(), [&](std::size_t i) {
            auto& p = particles[i];
            const double force = std::sin(p.x) + std::cos(p.y) + std::sqrt(p.z + 1.0);
            p.vx += 0.001 * force;
            p.vy += 0.002 * force;
            p.vz += 0.003 * force;
            p.x += p.vx;
            p.y += p.vy;
            p.z += p.vz;
        });
        benchmark::DoNotOptimize(particles.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(particle_count));
}

BENCHMARK(benchmark_particle_update)
    ->Name("ParallelParticleUpdate")
    ->Args({1 << 12, 1})
    ->Args({1 << 12, 0})
    ->Args({1 << 18, 1})
    ->Args({1 << 18, 0});

} // namespace
```

- [ ] **Step 2: Wire benchmark source**

Modify `benchmarks/CMakeLists.txt`:

```cmake
add_executable(pgo_benchmarks
    math/bench_eigen_config.cpp
    parallel/bench_parallel_for.cpp
)

target_link_libraries(pgo_benchmarks
    PRIVATE
        pgo::core
        pgo::parallel_runtime
        benchmark::benchmark
)
```

- [ ] **Step 3: Run benchmark without TBB**

Run:

```bash
cmake --build --preset release --target pgo_benchmarks
./build/release/benchmarks/pgo_benchmarks --benchmark_filter=ParallelParticleUpdate --benchmark_min_time=0.2s
```

Expected: `thread_count=1` and `thread_count=0` produce similar timings because fallback is serial.

- [ ] **Step 4: Run benchmark with TBB**

Run:

```bash
cmake --build --preset release-all --target pgo_benchmarks
./build/release-all/benchmarks/pgo_benchmarks --benchmark_filter=ParallelParticleUpdate --benchmark_min_time=0.2s
```

Expected: large workload with `thread_count=0` is faster than `thread_count=1` when the machine has available cores. If system load is high, rerun after closing other CPU-heavy tasks.

- [ ] **Step 5: Inspect changed files**

Run:

```bash
git diff -- benchmarks/CMakeLists.txt benchmarks/parallel/bench_parallel_for.cpp
git status --short benchmarks/CMakeLists.txt benchmarks/parallel/bench_parallel_for.cpp
```

Expected: only the intended benchmark source and benchmark target changes appear.

## Phase 7: Documentation And Final Verification

### Task 7.1: Document usage and threading policy

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add TBB usage section**

Add this section:

```markdown
## TBB Parallel Runtime

`PGO_ENABLE_TBB` enables the optional `pgo::parallel` backend. Code that needs
task parallelism should link `pgo::parallel_runtime` and include
`<pgo/parallel/parallel_for.hpp>`.

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
```

- [ ] **Step 2: Inspect changed files**

Run:

```bash
git diff -- README.md
git status --short README.md
```

Expected: only the intended TBB documentation changes appear.

### Task 7.2: Run final verification matrix

**Files:**
- No source edits.

- [ ] **Step 1: Verify serial configuration**

Run:

```bash
uv run pgo-configure release --build
ctest --preset release --output-on-failure -R "Parallel|Eigen"
```

Expected: configure, build, and selected tests pass.

- [ ] **Step 2: Verify TBB configuration**

Run:

```bash
uv run pgo-configure release-all --build
ctest --preset release-all --output-on-failure -R "Parallel|Eigen"
```

Expected: configure, build, and selected tests pass with TBB enabled.

- [ ] **Step 3: Verify benchmark executable**

Run:

```bash
./build/release-all/benchmarks/pgo_benchmarks --benchmark_filter=ParallelParticleUpdate --benchmark_min_time=0.2s
```

Expected: benchmark runs and reports `ParallelParticleUpdate` rows.

- [ ] **Step 4: Inspect compile commands**

Run:

```bash
node - <<'NODE'
const fs = require('fs');
const commands = JSON.parse(fs.readFileSync('build/release-all/compile_commands.json', 'utf8'));
const hit = commands.some((entry) => {
  const command = entry.command || entry.arguments.join(' ');
  return command.includes('PGO_ENABLE_TBB') && entry.file.includes('/parallel/');
});
if (!hit) {
  console.error('PGO_ENABLE_TBB was not found in parallel compile commands');
  process.exit(1);
}
console.log('PGO_ENABLE_TBB found in parallel compile commands');
NODE
```

Expected: script prints `PGO_ENABLE_TBB found in parallel compile commands`.

- [ ] **Step 5: Inspect Eigen policy compile definitions**

Run:

```bash
node - <<'NODE'
const fs = require('fs');
const commands = JSON.parse(fs.readFileSync('build/release-all/compile_commands.json', 'utf8'));
const eigenUsers = commands.filter((entry) => {
  const command = entry.command || entry.arguments.join(' ');
  return command.includes('PGO_EIGEN_ACCELERATION') || command.includes('EIGEN_USE_BLAS') || command.includes('EIGEN_USE_MKL_ALL');
});
if (!eigenUsers.every((entry) => (entry.command || entry.arguments.join(' ')).includes('EIGEN_DONT_PARALLELIZE'))) {
  console.error('EIGEN_DONT_PARALLELIZE missing from at least one Eigen compile command');
  process.exit(1);
}
if (eigenUsers.some((entry) => (entry.command || entry.arguments.join(' ')).includes('EIGEN_MAX_ALIGN_BYTES='))) {
  console.error('EIGEN_MAX_ALIGN_BYTES should not be defined unless explicitly requested');
  process.exit(1);
}
console.log('Eigen threading policy is present and alignment override is absent by default');
NODE
```

Expected: script prints `Eigen threading policy is present and alignment override is absent by default`.

- [ ] **Step 6: Inspect final working tree**

Run:

```bash
node - <<'NODE'
const fs = require('fs');
const text = fs.readFileSync('cmake/pgo_project_options.cmake', 'utf8');
if (!text.includes('set(PGO_MSVC_ARCH "DEFAULT"')) {
  console.error('PGO_MSVC_ARCH default option is missing');
  process.exit(1);
}
if (!text.includes('/MP') || !text.includes('/bigobj') || !text.includes('/Zc:__cplusplus')) {
  console.error('MSVC engineering flags are missing');
  process.exit(1);
}
if (!text.includes('PGO_MSVC_ARCH STREQUAL "AVX2"')) {
  console.error('PGO_MSVC_ARCH=AVX2 mapping is missing');
  process.exit(1);
}
console.log('MSVC project options policy is present');
NODE
```

Expected: script prints `MSVC project options policy is present`.

- [ ] **Step 7: Inspect Release debug symbols policy**

Run:

```bash
node - <<'NODE'
const fs = require('fs');
const text = fs.readFileSync('cmake/pgo_project_options.cmake', 'utf8');
if (!text.includes('PGO_ENABLE_RELEASE_DEBUG_SYMBOLS')) {
  console.error('PGO_ENABLE_RELEASE_DEBUG_SYMBOLS option is missing');
  process.exit(1);
}
if (!text.includes('$<$<CONFIG:Release>:-g>') && !text.includes('$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-g>')) {
  console.error('Release -g mapping is missing');
  process.exit(1);
}
console.log('Release debug symbols policy is present');
NODE
```

Expected: script prints `Release debug symbols policy is present`.

- [ ] **Step 8: Inspect final working tree**

Run:

```bash
git status --short
```

Expected: source changes are visible for the user to review and commit manually; generated build outputs remain untracked or ignored.

## Self-Review

- Spec coverage: The plan covers the requested first phase: build option, Conan option, `pgo::parallel` target, serial/TBB wrapper, runtime thread count, tests, benchmark, and docs.
- Placeholder scan: The plan contains no unresolved placeholder tokens, no open-ended "handle later" steps, and each code step includes concrete snippets.
- Type consistency: Public API names are consistent across tasks: `pgo::parallel::is_tbb_enabled`, `set_thread_count`, `configured_thread_count`, `serial_for`, and `parallel_for`.
- Dependency boundary: `pgo::core` remains independent of TBB; only `pgo::parallel` / `pgo::parallel_runtime` carry the optional backend.
