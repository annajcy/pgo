# TBB Integration Implementation Plan

> **给 agentic workers:** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务执行。本计划使用 checkbox (`- [ ]`) 语法追踪进度。

**目标:** 为 PGO 引入可选 TBB task-parallel runtime，提供 `pgo::parallel` 抽象层，让 simulation/assembly 外层循环可以在 TBB 与串行 fallback 之间切换，并同步整理 Eigen threading / MKL / alignment、MSVC 编译选项、Release debug symbols 策略。

**架构:** `pgo::core` 透明 INTERFACE 链接 `pgo::parallel_runtime`，下游 examples/solver/integrator 通过 `pgo::core` 间接拿到 `parallel_for` 抽象，不再单独 link。编译期只决定是否启用 TBB，线程数属于运行时配置，通过 `pgo::parallel::set_thread_count(...)` 控制。Eigen/BLAS 仍由现有 `pgo::eigen_config` 和 `release-accel` 体系负责；Eigen 内部线程默认禁用，避免与 TBB 外层任务并行抢线程。

**Tech Stack:** C++23、CMake Presets、Conan 2、oneTBB/TBB、GoogleTest、Google Benchmark、Eigen。

---

## 0. 核心设计决策

- `PGO_ENABLE_TBB` 是编译期开关，默认 `OFF`。
- `enable_tbb` 是 Conan option，默认 `False`；开启时引入 `onetbb` 包。
- TBB 归入现有 `-all` preset 家族：`debug-all`、`debug-accel-all`、`release-all`、`release-accel-all` 会启用 TBB；不新增独立 TBB-only preset 维度。
- `EIGEN_DONT_PARALLELIZE` 默认开启，表示 Eigen 自己不创建内部 worker threads；外层并行由 TBB 管。
- `EIGEN_DONT_PARALLELIZE` 不等于禁用 MKL/Accelerate。MKL/Accelerate backend 仍由 `PGO_ENABLE_EIGEN_ACCELERATION` 和 `PGO_EIGEN_ACCELERATION_BACKEND` 控制。
- `EIGEN_MKL_NO_DIRECT_CALL` 仅在 MKL backend 下可选启用，**默认 `OFF`**；保留 Eigen direct-MKL-call path 以避免性能回退，需要时再 `-DPGO_EIGEN_MKL_NO_DIRECT_CALL=ON` 显式 opt-in。
- `EIGEN_MAX_ALIGN_BYTES` 不默认硬设；通过 `PGO_EIGEN_MAX_ALIGN_BYTES` cache string 显式覆盖，空字符串保留 Eigen/platform 默认。
- MSVC 默认启用 `/MP`、`/bigobj`、`/Zc:__cplusplus`，提升 Windows 构建体验和模板代码兼容性；这三项**无条件**应用于 MSVC，不受 `PGO_ENABLE_NATIVE_ARCH` 控制。
- MSVC CPU ISA 通过 `PGO_MSVC_ARCH` 显式选择 `DEFAULT`、`AVX`、`AVX2` 或 `AVX512`。
- `PGO_MSVC_ARCH` 解析顺序：(1) 显式 `AVX`/`AVX2`/`AVX512` 永远生效；(2) `DEFAULT` 且 `PGO_ENABLE_NATIVE_ARCH=ON` 回退到 `AVX2`（保留旧版默认行为）；(3) `DEFAULT` 且 `PGO_ENABLE_NATIVE_ARCH=OFF` 不加 `/arch:*`，生成可移植 Windows binary。
- Release debug symbols 通过 `PGO_ENABLE_RELEASE_DEBUG_SYMBOLS` 显式开启，默认 `OFF`；用于 profile、crash backtrace 和优化构建调试。
- `PGO_ENABLE_RELEASE_DEBUG_SYMBOLS=ON` 在 GNU/Clang/AppleClang Release 下加 `-g`，在 MSVC Release 下加 `/Zi` 和 linker debug info；不把 `-ggdb3` 无条件塞进所有 Release 构建。
- CMake 只在 `PGO_ENABLE_TBB=ON` 时 `find_package(TBB REQUIRED CONFIG)`。
- 只新增 `pgo_parallel_runtime` / `pgo::parallel_runtime` STATIC target（含 `runtime.cpp` 实现 + header propagation）；不再额外引入 INTERFACE `pgo::parallel`，简化命名与 link 关系。
- `pgo::parallel_runtime` 在 TBB ON 时 **PUBLIC** 链接 `TBB::tbb` 并 PUBLIC 定义 `PGO_ENABLE_TBB`，让下游头文件展开 `parallel_for` 时也能看到该 macro；TBB OFF 时只编 serial fallback，不链接额外库。
- 业务代码不直接依赖 `tbb::parallel_for`；统一调用 `pgo::parallel::parallel_for(...)`。
- pypgo wheel 在 `PGO_ENABLE_TBB=ON` 时通过 `install(IMPORTED_RUNTIME_ARTIFACTS TBB::tbb)` 把 TBB 动态库随 `_pgo_ext` 打进 `pgo/` 包目录，并配 `@loader_path`/`$ORIGIN` RPATH，使 `import pgo` 不依赖系统 TBB。
- `runtime_thread_count` 用 `std::atomic<int>` 读写，无需 mutex；mutex 只用于保护 `oneapi::tbb::global_control` 的 `unique_ptr` 重建。
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
  CMakeLists.txt                    # modify: add_subdirectory(src/parallel) + pgo_core 链接 parallel_runtime
  CMakePresets.json                 # modify: all-opt 注入 PGO_ENABLE_TBB
  conanfile.py                      # modify: enable_tbb option + 可选 onetbb 依赖
  cmake/
    pgo_dependencies.cmake          # modify: find_package(TBB) gated by PGO_ENABLE_TBB
    pgo_eigen.cmake                 # modify: EIGEN_DONT_PARALLELIZE / MAX_ALIGN_BYTES / MKL_NO_DIRECT_CALL
    pgo_options.cmake               # modify: PGO_ENABLE_TBB, PGO_EIGEN_* policy options
    pgo_project_options.cmake       # modify: PGO_MSVC_ARCH, PGO_ENABLE_RELEASE_DEBUG_SYMBOLS
  include/
    pgo/
      parallel/
        parallel_for.hpp            # new
        runtime.hpp                 # new
  src/
    parallel/
      CMakeLists.txt                # new: 直接 add_library(pgo_parallel_runtime STATIC)，不通过 cmake/pgo_parallel.cmake 中转
      runtime.cpp                   # new
    python/
      CMakeLists.txt                # modify: install(IMPORTED_RUNTIME_ARTIFACTS TBB::tbb) when PGO_ENABLE_TBB
  tests/
    parallel/
      test_parallel_for.cpp         # new
    math/
      test_eigen_config.cpp         # modify: Eigen policy assertions
  benchmarks/
    parallel/
      bench_parallel_for.cpp        # new
  scripts/
    pgo_configure.py                # modify: PGO_ENABLE_TBB -> enable_tbb forwarding
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

### Task 1.2: Add CMake option and TBB dependency lookup

**Files:**
- Modify: `cmake/pgo_options.cmake`
- Modify: `cmake/pgo_dependencies.cmake`

注：parallel target 本身在 Phase 4 才创建（直接在 `src/parallel/CMakeLists.txt` 里 `add_library(pgo_parallel_runtime STATIC ...)`），本期不再单独引入 `cmake/pgo_parallel.cmake` 或 INTERFACE `pgo::parallel` 中间层。

- [ ] **Step 1: Add CMake option**

Add this to `cmake/pgo_options.cmake` near the other feature options:

```cmake
option(PGO_ENABLE_TBB "Enable TBB-backed task parallelism" OFF)
```

- [ ] **Step 2: Add dependency lookup**

紧跟 `cmake/pgo_dependencies.cmake` 现有 `find_package` 块尾部追加：

```cmake
if(PGO_ENABLE_TBB)
    find_package(TBB REQUIRED CONFIG)
endif()
```

`pgo_dependencies.cmake` 在顶层 `CMakeLists.txt` 第 8 行被 include，早于 `add_subdirectory(src/parallel)`，所以 Phase 4 创建 `pgo_parallel_runtime` 时 `TBB::tbb` 已 ready。

- [ ] **Step 3: Configure without TBB**

Run:

```bash
cmake --preset release
```

Expected: configure succeeds without looking for `TBB::tbb`.

- [ ] **Step 4: Configure with TBB using a temporary build folder**

Run（`conan/profiles/default` 是 Jinja 模板，按 `platform.system()` 自动 include 对应平台 profile，所以三大平台都可直接用）：

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

Expected: configure succeeds and `TBB::tbb` is available。macOS arm64 首次 CI 若 conancenter 缺 prebuilt binary，`--build=missing` 会自动 fallback 到 source build；后续 conan cache 命中后可省略该 flag。

- [ ] **Step 5: Inspect changed files**

Run:

```bash
git diff -- cmake/pgo_options.cmake cmake/pgo_dependencies.cmake
git status --short cmake/pgo_options.cmake cmake/pgo_dependencies.cmake
```

Expected: only the intended CMake option and dependency lookup changes appear; no `cmake/pgo_parallel.cmake` is created.

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

- [ ] **Step 2: Verify dry-run forwarding** *(执行顺序：先做完 Task 1.4 再回来跑这一步；`release-all` 预设里加上 TBB 之后这条 dry-run 才能观察到 forwarding 行为)*

Run:

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
pypgo-release-all
pypgo-release-accel-all
```

The Python wheel wrapper (`uv run python scripts/pgo_build_wheel.py ...`) reads
the resolved package preset and forwards every `PGO_*` cache variable to
scikit-build, so adding `PGO_ENABLE_TBB=ON` to `all-opt` also makes the
`pypgo-*` wheel commands pick up TBB without duplicating flags in the docs.

- [ ] **Step 2: Document TBB as part of all presets**

In `README.md`, update the preset matrix and `-all` description:

```markdown
`-all` enables optional compile-time dependencies: spdlog, Alembic, and the
TBB-backed `pgo::parallel_runtime` task runtime. `pgo::core` links it
transitively, so any consumer of `pgo::core` (examples, solver, integrator)
automatically gets the `parallel_for` abstraction with TBB or serial fallback
depending on `PGO_ENABLE_TBB`.

The `pypgo-release-all` / `pypgo-release-accel-all` presets also inherit TBB.
The Python wheel bundles the TBB shared library next to `_pgo_ext` and sets
`@loader_path`/`$ORIGIN` RPATH so `import pgo` does not need a system-wide
TBB install (see Phase 4 wheel-bundling task).
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

按下面三个改动重写 `cmake/pgo_project_options.cmake` 里 MSVC 相关逻辑：

1. **删掉** 现有 `if(PGO_ENABLE_NATIVE_ARCH) ... elseif(MSVC) ... $<$<CONFIG:Release>:/arch:AVX2> ...` 整段中的 MSVC 分支（保留 GNU/Clang 分支不动）。

2. 在 `if(PGO_ENABLE_NATIVE_ARCH)` 块**之外**追加 MSVC 工程类 flag（这三项跟 ISA 无关，无条件应用）：

```cmake
if(MSVC)
    target_compile_options(pgo_project_options INTERFACE
        /MP
        /bigobj
        /Zc:__cplusplus
    )
endif()
```

3. 在上面之后追加 MSVC ISA 解析逻辑；`DEFAULT` + `PGO_ENABLE_NATIVE_ARCH=ON` 回退到 `AVX2` 以保留旧版默认行为：

```cmake
if(MSVC)
    set(_pgo_msvc_resolved_arch "${PGO_MSVC_ARCH}")
    if(_pgo_msvc_resolved_arch STREQUAL "DEFAULT" AND PGO_ENABLE_NATIVE_ARCH)
        set(_pgo_msvc_resolved_arch "AVX2")
        message(STATUS "PGO_MSVC_ARCH=DEFAULT with PGO_ENABLE_NATIVE_ARCH=ON resolves to AVX2 (legacy default)")
    endif()

    if(_pgo_msvc_resolved_arch STREQUAL "AVX")
        target_compile_options(pgo_project_options INTERFACE $<$<CONFIG:Release>:/arch:AVX>)
    elseif(_pgo_msvc_resolved_arch STREQUAL "AVX2")
        target_compile_options(pgo_project_options INTERFACE $<$<CONFIG:Release>:/arch:AVX2>)
    elseif(_pgo_msvc_resolved_arch STREQUAL "AVX512")
        target_compile_options(pgo_project_options INTERFACE $<$<CONFIG:Release>:/arch:AVX512>)
    elseif(_pgo_msvc_resolved_arch STREQUAL "DEFAULT")
        # NATIVE_ARCH=OFF + DEFAULT：保留 MSVC 编译器默认 ISA，生成可移植 Windows binary
    else()
        message(FATAL_ERROR "Unknown PGO_MSVC_ARCH=${PGO_MSVC_ARCH}")
    endif()
endif()
```

这样 `PGO_ENABLE_NATIVE_ARCH=ON`（默认）+ MSVC 用户不显式设 `PGO_MSVC_ARCH` 时，仍然走 AVX2，性能不会回退；显式设 `-DPGO_MSVC_ARCH=AVX512` 或在 `NATIVE_ARCH=OFF` 下设 `DEFAULT` 才会改变行为。

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
uv run python - <<'PY'
import json, re, sys
commands = json.loads(open("build/release/compile_commands.json").read())
def args(e): return e.get("command") or " ".join(e.get("arguments", []))
hit = any(re.search(r"(^| )-g($| )", args(e)) or "/Zi" in args(e) for e in commands)
if hit:
    sys.exit("release debug symbols should be disabled by default")
print("release debug symbols disabled by default")
PY
```

Expected: script prints `release debug symbols disabled by default`.

- [ ] **Step 4: Verify opt-in release debug symbols**

Run:

```bash
cmake -S . -B build/release-debug-symbols-probe -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=build/conan/release/conan_toolchain.cmake \
  -DPGO_ENABLE_RELEASE_DEBUG_SYMBOLS=ON

uv run python - <<'PY'
import json, re, sys
commands = json.loads(open("build/release-debug-symbols-probe/compile_commands.json").read())
def args(e): return e.get("command") or " ".join(e.get("arguments", []))
hit = any(re.search(r"(^| )-g($| )", args(e)) or "/Zi" in args(e) for e in commands)
if not hit:
    sys.exit("release debug symbols were not found after opt-in")
print("release debug symbols found after opt-in")
PY
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
option(PGO_EIGEN_MKL_NO_DIRECT_CALL "Disable Eigen direct MKL calls when MKL acceleration is enabled" OFF)

set(PGO_EIGEN_MAX_ALIGN_BYTES "" CACHE STRING "Override Eigen max alignment bytes; empty keeps Eigen default")
```

`PGO_EIGEN_MKL_NO_DIRECT_CALL` 默认 `OFF`：保留 Eigen 走 MKL direct-call path 的性能；如果发现某些 kernel 与 direct-call 兼容性差再 opt-in。

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
uv run python - <<'PY'
import json, sys
commands = json.loads(open("build/release/compile_commands.json").read())
def args(e): return e.get("command") or " ".join(e.get("arguments", []))
eigen_users = [e for e in commands if any(tok in args(e) for tok in ("PGO_EIGEN_ACCELERATION_NONE", "EIGEN_USE_BLAS", "EIGEN_USE_MKL_ALL"))]
missing = [e for e in eigen_users if "EIGEN_DONT_PARALLELIZE" not in args(e)]
if missing:
    sys.exit(f"EIGEN_DONT_PARALLELIZE missing from {len(missing)} Eigen compile commands")
print(f"EIGEN_DONT_PARALLELIZE found in {len(eigen_users)} Eigen compile commands")
PY
```

Expected: script prints that `EIGEN_DONT_PARALLELIZE` is present in Eigen-using compile commands.

- [ ] **Step 4: Configure explicit alignment override and inspect compile commands**

Run:

```bash
cmake -S . -B build/eigen-align-probe -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=build/conan/release/conan_toolchain.cmake \
  -DPGO_EIGEN_MAX_ALIGN_BYTES=32

uv run python - <<'PY'
import json, sys
commands = json.loads(open("build/eigen-align-probe/compile_commands.json").read())
def args(e): return e.get("command") or " ".join(e.get("arguments", []))
hit = any("EIGEN_MAX_ALIGN_BYTES=32" in args(e) for e in commands)
if not hit:
    sys.exit("EIGEN_MAX_ALIGN_BYTES=32 was not found in compile commands")
print("EIGEN_MAX_ALIGN_BYTES=32 found in compile commands")
PY
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
TEST(EigenConfig, MklNoDirectCallIsOptInAndScopedToMklBackend) {
    // 默认 PGO_EIGEN_MKL_NO_DIRECT_CALL=OFF：无论 MKL 是否启用，EIGEN_MKL_NO_DIRECT_CALL 都不应被定义。
    // 如果未来在 CI 里 opt-in，需要把这个测试改成读 cmake 注入的 PGO_EIGEN_MKL_NO_DIRECT_CALL 配置宏并分支检查。
#if defined(EIGEN_MKL_NO_DIRECT_CALL)
    #if defined(PGO_EIGEN_ACCELERATION_MKL)
        SUCCEED() << "MKL backend explicitly opted into EIGEN_MKL_NO_DIRECT_CALL";
    #else
        FAIL() << "EIGEN_MKL_NO_DIRECT_CALL must not leak outside the MKL backend";
    #endif
#else
    SUCCEED() << "default configuration keeps Eigen direct-MKL-call path enabled";
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

if(PGO_ENABLE_TBB)
    target_link_libraries(pgo_parallel_runtime PUBLIC TBB::tbb)
    target_compile_definitions(pgo_parallel_runtime PUBLIC PGO_ENABLE_TBB)
endif()
```

PUBLIC link + PUBLIC define 让所有透过 `pgo::core` 间接拿到 `pgo::parallel_runtime` 的 target（examples、tests、benchmarks、c_api 等）在编译 `parallel_for.hpp` 时都能看见 `PGO_ENABLE_TBB` macro 和 TBB 头。

在顶层 `CMakeLists.txt` 中**无条件**追加（紧跟现有 `add_subdirectory(src/log)` 之后；TBB OFF 时也要存在以便 serial fallback）：

```cmake
add_subdirectory(src/parallel)
```

并把 `pgo_core` 的 `target_link_libraries` 扩展为透明 INTERFACE 依赖 `pgo::parallel_runtime`，让上层 solver/integrator/examples 不必单独 link：

```cmake
target_link_libraries(pgo_core INTERFACE pgo::eigen_config pgo::project_options pgo::parallel_runtime)
```

注意原 line 32 的 `pgo_project_warnings`/`pgo_project_sanitizers` 链接保持不变。

- [ ] **Step 3: Implement serial/TBB runtime state**

Create `src/parallel/runtime.cpp`：`runtime_thread_count` 用 `std::atomic<int>` 直接读写，mutex 只保护 `global_control` 的 `unique_ptr` 重建（避免热路径每次 `parallel_for` 取 thread count 时加锁）：

```cpp
#include <pgo/parallel/runtime.hpp>

#if defined(PGO_ENABLE_TBB)
#include <oneapi/tbb/global_control.h>
#endif

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace pgo::parallel {
namespace {

std::atomic<int> runtime_thread_count{0};

#if defined(PGO_ENABLE_TBB)
std::mutex global_control_mutex;
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

    runtime_thread_count.store(thread_count, std::memory_order_release);

#if defined(PGO_ENABLE_TBB)
    std::lock_guard<std::mutex> lock(global_control_mutex);
    runtime_global_control.reset();
    if (thread_count > 0) {
        runtime_global_control = std::make_unique<oneapi::tbb::global_control>(
            oneapi::tbb::global_control::max_allowed_parallelism,
            static_cast<std::size_t>(thread_count));
    }
#endif
}

int configured_thread_count() noexcept {
    return runtime_thread_count.load(std::memory_order_acquire);
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

Expected: 顶层 `CMakeLists.txt` 显示 `add_subdirectory(src/parallel)` 新增 + `pgo_core` 的 `target_link_libraries(INTERFACE ...)` 加上 `pgo::parallel_runtime`；新建文件 runtime.hpp / src/parallel/CMakeLists.txt / src/parallel/runtime.cpp 出现；没有 INTERFACE `pgo::parallel` 残留。

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

- [ ] **Step 2: Defer header compile-check to Phase 5 tests**

`parallel_for.hpp` 是模板头，`runtime.cpp` 不会 include 它，所以单 build `pgo_parallel_runtime` 不能验证语法。真正的 compile-check 留给 Phase 5 的 `tests/parallel/test_parallel_for.cpp`（它直接 `#include <pgo/parallel/parallel_for.hpp>`）。

如果想在 Phase 4 阶段就提早 catch 语法错误，可以快速做一次：

```bash
cat > /tmp/pgo_parallel_for_probe.cpp <<'CPP'
#include <pgo/parallel/parallel_for.hpp>
int main() {
    pgo::parallel::parallel_for(0, 4, [](int){});
    return 0;
}
CPP
${CXX:-c++} -std=c++23 -Iinclude -fsyntax-only /tmp/pgo_parallel_for_probe.cpp
```

Expected: 命令静默退出（`-fsyntax-only` 不产物，只做 parse + sema）。TBB ON 模式想 catch `oneapi/tbb/*.h` 的 include 路径，要把 conan toolchain 的 include dir 也 `-I` 进去；省事的做法直接跳到 Phase 5 跑测试。

- [ ] **Step 3: Inspect changed files**

Run:

```bash
git diff -- include/pgo/parallel/parallel_for.hpp
git status --short include/pgo/parallel/parallel_for.hpp
```

Expected: only the intended `parallel_for` wrapper change appears.

### Task 4.3: Bundle TBB shared library into pypgo wheels

**Files:**
- Modify: `src/python/CMakeLists.txt`

`pypgo-release-all` / `pypgo-release-accel-all` 通过 `all-opt` 间接打开 `PGO_ENABLE_TBB`。wheel 必须把 TBB 动态库随 `_pgo_ext` 一起打进去，并设好 RPATH/loader path，否则 `import pgo` 在没装系统 TBB 的机器上会报 missing library。

- [ ] **Step 1: Install TBB runtime alongside `_pgo_ext`**

在 `src/python/CMakeLists.txt` 末尾追加（gate 在 `PGO_BUILD_PYTHON AND PGO_ENABLE_TBB`）：

```cmake
if(PGO_ENABLE_TBB)
    install(IMPORTED_RUNTIME_ARTIFACTS TBB::tbb
        RUNTIME DESTINATION pgo
        LIBRARY DESTINATION pgo
        COMPONENT python
    )
endif()
```

`IMPORTED_RUNTIME_ARTIFACTS` 跨平台行为：macOS 安装 `.dylib`，Linux 安装 `.so`，Windows 安装 `.dll`，路径都进 wheel 里的 `pgo/`。

- [ ] **Step 2: Confirm `_pgo_ext` RPATH already resolves to `pgo/`**

`src/python/CMakeLists.txt:59-69` 现有逻辑已经给 `${PGO_PYTHON_EXTENSION_NAME}` 设置好：

- macOS：`BUILD_RPATH` / `INSTALL_RPATH` = `@loader_path`
- Linux/其它 UNIX：`BUILD_RPATH` / `INSTALL_RPATH` = `$ORIGIN`
- Windows：不需要 RPATH，Python 3.8+ DLL search path 默认包含 extension 自己所在的目录

所以这一步**不需要新增代码**，只在 Step 1 之前确认这段 `set_target_properties` 没被改动；如果未来有人精简了 RPATH 逻辑，TBB 打包会同步失效，这里是排错锚点。

- [ ] **Step 3: Verify bundled TBB shows up in the built wheel**

Run:

```bash
uv run python scripts/pgo_build_wheel.py pypgo-release-all --clear
uv run python - <<'PY'
import zipfile, pathlib, sys
wheels = sorted(pathlib.Path("dist/pypgo-release-all").glob("*.whl"))
if not wheels:
    sys.exit("no wheel built")
with zipfile.ZipFile(wheels[-1]) as z:
    members = z.namelist()
ext = [m for m in members if "_pgo_ext" in m]
tbb = [m for m in members if "tbb" in m.lower() and m.startswith("pgo/")]
print("ext:", ext)
print("tbb:", tbb)
if not ext:
    sys.exit("_pgo_ext is missing from the wheel")
if not tbb:
    sys.exit("TBB runtime library is not bundled inside pgo/")
print("TBB runtime is bundled alongside _pgo_ext")
PY
```

Expected: 脚本打印 `TBB runtime is bundled alongside _pgo_ext`，wheel 内 `pgo/` 目录里能看到 TBB 动态库。

- [ ] **Step 4: Sanity check `import pgo` against the installed wheel**

Run（在一个不安装 TBB 的临时 venv 中）：

```bash
uv venv build/tbb-bundle-probe
uv pip install --python build/tbb-bundle-probe/bin/python "dist/pypgo-release-all"/*.whl
build/tbb-bundle-probe/bin/python -c "import pgo; print(pgo.__file__)"
```

Expected: import 成功，输出 wheel 里的 `pgo/__init__.py` 路径；没有 `library not loaded` 之类的报错。Linux/macOS 上如果失败，多半是 RPATH 没生效，按顺序回头确认：
1. `src/python/CMakeLists.txt:59-69` 的 `INSTALL_RPATH = @loader_path / $ORIGIN` 块仍然存在；
2. scikit-build 的 install 阶段确实执行了 `IMPORTED_RUNTIME_ARTIFACTS TBB::tbb`（用 Step 3 的 zipfile 脚本核对 wheel 内容）；
3. 用 `otool -l _pgo_ext*.so | grep -A2 LC_RPATH`（macOS）或 `readelf -d _pgo_ext*.so | grep RUNPATH`（Linux）查 wheel 内 extension 的 RPATH 字段是否真的被写进二进制。

- [ ] **Step 5: Inspect changed files**

Run:

```bash
git diff -- src/python/CMakeLists.txt
git status --short src/python/CMakeLists.txt
```

Expected: 只增加 `if(PGO_ENABLE_TBB) install(IMPORTED_RUNTIME_ARTIFACTS TBB::tbb ...)` 这一段；RPATH 块保持原样不动。

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

// 用 fixture + TearDown 把 thread count 还原为 0，避免一个 case 改了全局 runtime 影响后续 case；
// gtest_discover_tests 不保证执行顺序，所以每个 case 都假定起点是默认状态。
class ParallelFixture : public ::testing::Test {
protected:
    void SetUp() override {
        pgo::parallel::set_thread_count(0);
    }
    void TearDown() override {
        pgo::parallel::set_thread_count(0);
    }
};

using ParallelFor = ParallelFixture;
using ParallelRuntime = ParallelFixture;

TEST_F(ParallelFor, EmptyRangeDoesNothing) {
    int count = 0;
    pgo::parallel::parallel_for(5, 5, [&](int) { ++count; });
    EXPECT_EQ(count, 0);
}

TEST_F(ParallelFor, VisitsEachIndexOnce) {
    std::vector<int> visits(128, 0);
    pgo::parallel::set_thread_count(1);

    pgo::parallel::parallel_for(0, static_cast<int>(visits.size()), [&](int index) {
        visits[static_cast<std::size_t>(index)] += 1;
    });

    for (int visit : visits) {
        EXPECT_EQ(visit, 1);
    }
}

TEST_F(ParallelRuntime, RejectsNegativeThreadCount) {
    EXPECT_THROW(pgo::parallel::set_thread_count(-1), std::invalid_argument);
}

TEST_F(ParallelRuntime, StoresConfiguredThreadCount) {
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

`pgo_tests` 已经 PRIVATE 链接 `pgo::core`，而 `pgo::core` 在 Phase 4 之后 INTERFACE 透传 `pgo::parallel_runtime`，所以 link 列表**不需要再加** `pgo::parallel_runtime`，保持现有：

```cmake
target_link_libraries(pgo_tests
    PRIVATE
        pgo::core
        pgo::io
        GTest::gtest_main
)
```

如果以后某个测试 binary 只想链 `pgo::parallel_runtime` 而不要 `pgo::core`，再单独加。

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

Expected: `tests/CMakeLists.txt` 只在 `pgo_tests` 源文件列表里多一行 `parallel/test_parallel_for.cpp`，`target_link_libraries` 块**不动**（pgo::parallel_runtime 通过 pgo::core 透传）；`tests/parallel/test_parallel_for.cpp` 新文件包含 `ParallelFixture` + 四个 TEST_F case。

### Task 5.2: Add TBB-enabled test coverage

**Files:**
- Modify: `tests/parallel/test_parallel_for.cpp`

- [ ] **Step 1: Add TBB backend assertions**

把下面两个 case **插入 Task 5.1 Step 1 的匿名 namespace 里**（紧贴 `TEST_F(ParallelRuntime, StoresConfiguredThreadCount)` 之后、`} // namespace` 之前）；`ParallelFor` / `ParallelRuntime` 别名只在那个 namespace 内可见，写到文件末尾会编不过。

```cpp
TEST_F(ParallelRuntime, ReportsCompiledBackend) {
#if defined(PGO_ENABLE_TBB)
    EXPECT_TRUE(pgo::parallel::is_tbb_enabled());
#else
    EXPECT_FALSE(pgo::parallel::is_tbb_enabled());
#endif
}

TEST_F(ParallelFor, DefaultThreadCountProducesCorrectReduction) {
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

Modify `benchmarks/CMakeLists.txt`（`pgo::parallel_runtime` 通过 `pgo::core` 透传，不需要重复 link）：

```cmake
add_executable(pgo_benchmarks
    math/bench_eigen_config.cpp
    parallel/bench_parallel_for.cpp
)

target_link_libraries(pgo_benchmarks
    PRIVATE
        pgo::core
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

Python wheels (`pypgo-release-all`, `pypgo-release-accel-all`) bundle the TBB
shared library inside the `pgo/` package directory with `@loader_path`/`$ORIGIN`
RPATH, so `import pgo` works on machines without a system-wide TBB install.
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
uv run python - <<'PY'
import json, sys
commands = json.loads(open("build/release-all/compile_commands.json").read())
def args(e): return e.get("command") or " ".join(e.get("arguments", []))
hit = any("PGO_ENABLE_TBB" in args(e) and "/parallel/" in e.get("file", "") for e in commands)
if not hit:
    sys.exit("PGO_ENABLE_TBB was not found in parallel compile commands")
print("PGO_ENABLE_TBB found in parallel compile commands")
PY
```

Expected: script prints `PGO_ENABLE_TBB found in parallel compile commands`.

- [ ] **Step 5: Inspect Eigen policy compile definitions**

Run:

```bash
uv run python - <<'PY'
import json, sys
commands = json.loads(open("build/release-all/compile_commands.json").read())
def args(e): return e.get("command") or " ".join(e.get("arguments", []))
eigen_users = [e for e in commands if any(tok in args(e) for tok in ("PGO_EIGEN_ACCELERATION", "EIGEN_USE_BLAS", "EIGEN_USE_MKL_ALL"))]
if not all("EIGEN_DONT_PARALLELIZE" in args(e) for e in eigen_users):
    sys.exit("EIGEN_DONT_PARALLELIZE missing from at least one Eigen compile command")
if any("EIGEN_MAX_ALIGN_BYTES=" in args(e) for e in eigen_users):
    sys.exit("EIGEN_MAX_ALIGN_BYTES should not be defined unless explicitly requested")
print("Eigen threading policy is present and alignment override is absent by default")
PY
```

Expected: script prints `Eigen threading policy is present and alignment override is absent by default`.

- [ ] **Step 6: Inspect MSVC project options policy**

Run:

```bash
uv run python - <<'PY'
import sys
text = open("cmake/pgo_project_options.cmake").read()
if 'set(PGO_MSVC_ARCH "DEFAULT"' not in text:
    sys.exit("PGO_MSVC_ARCH default option is missing")
if any(flag not in text for flag in ("/MP", "/bigobj", "/Zc:__cplusplus")):
    sys.exit("MSVC engineering flags are missing")
if '"AVX2"' not in text:
    sys.exit("PGO_MSVC_ARCH=AVX2 mapping is missing")
if "PGO_ENABLE_NATIVE_ARCH" not in text or "_pgo_msvc_resolved_arch" not in text:
    sys.exit("NATIVE_ARCH -> AVX2 fallback for MSVC DEFAULT is missing")
print("MSVC project options policy is present")
PY
```

Expected: script prints `MSVC project options policy is present`.

- [ ] **Step 7: Inspect Release debug symbols policy**

Run:

```bash
uv run python - <<'PY'
import sys
text = open("cmake/pgo_project_options.cmake").read()
if "PGO_ENABLE_RELEASE_DEBUG_SYMBOLS" not in text:
    sys.exit("PGO_ENABLE_RELEASE_DEBUG_SYMBOLS option is missing")
if "$<$<CONFIG:Release>:-g>" not in text and "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-g>" not in text:
    sys.exit("Release -g mapping is missing")
print("Release debug symbols policy is present")
PY
```

Expected: script prints `Release debug symbols policy is present`.

- [ ] **Step 8: Inspect final working tree**

Run:

```bash
git status --short
```

Expected: source changes are visible for the user to review and commit manually; generated build outputs remain untracked or ignored.

## Self-Review

- Spec coverage: build option, Conan option, MSVC project flag policy, Eigen threading/alignment policy, Release debug symbols, `pgo::parallel_runtime` STATIC target with serial/TBB wrapper, runtime thread count, wheel TBB bundling, tests with TEST_F teardown, benchmark, and docs.
- Single target naming: only `pgo::parallel_runtime` (STATIC) exists; no INTERFACE `pgo::parallel` intermediate.
- `pgo::core` INTERFACE-links `pgo::parallel_runtime`, so examples/solver/integrator/tests/benchmarks pick up `parallel_for` transitively without per-target link changes.
- MSVC ISA resolution: explicit `AVX/AVX2/AVX512` wins; `DEFAULT + NATIVE_ARCH=ON` falls back to AVX2 to preserve legacy behavior; `DEFAULT + NATIVE_ARCH=OFF` keeps the compiler default for portable binaries.
- Eigen policy: `EIGEN_DONT_PARALLELIZE` default ON; `EIGEN_MKL_NO_DIRECT_CALL` default OFF to avoid MKL perf regressions; `EIGEN_MAX_ALIGN_BYTES` opt-in only.
- Wheel packaging: `pypgo-release-*` presets bundle TBB shared library next to `_pgo_ext` via `install(IMPORTED_RUNTIME_ARTIFACTS TBB::tbb)` + `@loader_path`/`$ORIGIN` RPATH.
- Runtime thread state: `std::atomic<int>` for `runtime_thread_count`; mutex only protects `tbb::global_control` `unique_ptr` rebuild.
- Verification scripts: all `compile_commands.json` inspections use `uv run python` (no Node toolchain assumed).
- Test isolation: `ParallelFixture::SetUp/TearDown` resets thread count to 0 around every gtest case.
