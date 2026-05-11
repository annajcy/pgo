# GPU-Aware Mass Spring CPU 实现计划

> **给 agentic workers:** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务执行。本计划使用 checkbox (`- [ ]`) 语法追踪进度。

**目标:** 完成 Milestone 1：实现一个 CPU-only、GPU-aware 的 C++23 header-oriented mass-spring simulation core。它读取 rest mesh，优化 displacement `u`，输出 OBJ frame sequence，并提供最小稳定的 C99 动态库 API。

**架构:** 内部核心是现代 C++23 template/header-only library；对外二进制接口是 compiled C99 ABI dynamic library。CPU 实现优先，但数据布局和 energy interface 从第一天就为 Milestone 2 的 Vulkan/Slang GPU backend 留好边界：flat storage、显式 `X + u`、local contribution API、独立 assembly 层、geometry/storage 不持有 Eigen object。

**技术栈:** C++23、C99 ABI、CMake Presets、Conan 2、Eigen、GoogleTest、CLI11、Alembic、clang-format、GitHub Actions。

---

## 0. 核心设计决策

- Solver 的未知量始终是 displacement `u`，不是 current position `x`。
- Current position 只通过 `x = X + u` 得到，其中 `X` 是 immutable rest position。
- Rest positions `X` 和 topology 属于 `geometry::RestMesh<T, Dim>`。
- Mesh storage 使用 flat scalar/index arrays：positions 长度为 `num_vertices * Dim`，topology 使用 index buffer。
- Eigen 只通过 `pgo::math::eigen::EigenBackend` 和 `pgo::math` 默认 aliases 用于 CPU 数值计算和 sparse solve；`geometry/` 与 `storage/` 不存储 `Eigen::Vector3d`、`Eigen::MatrixXd` 等对象。
- Eigen CPU acceleration 是构建配置层：`PGO_ENABLE_EIGEN_ACCELERATION` 只控制 Eigen 调用 BLAS/LAPACK 类 backend（Apple Accelerate 或 MKL），不等同于选择 sparse linear solver。
- PARDISO 属于 linear solver backend 候选，不属于 `MathBackend`。后续应通过 `PGO_CPU_LINEAR_SOLVER` 或 solver policy 选择 `Eigen::PardisoLDLT` / `Eigen::SimplicialLDLT` / CG，而不是塞进 Eigen acceleration 开关。
- 不引入 `fmt` 第三方依赖；需要格式化字符串时使用标准库 `<format>` / `std::format`。核心库应尽量少做格式化，C API 错误消息用固定 buffer 写入。
- Energy model 必须暴露 local contribution API，使 CPU assembly 和未来 GPU kernel 能共享同一语义边界。
- Milestone 1 只实现 CPU assembly、CPU Newton solver、OBJ input/output、C99 ABI facade。
- C++ template、STL、Eigen、异常、allocator 内部细节不能越过 C ABI 边界。
- C API 只暴露 `extern "C"`、opaque handles、POD descriptors、pointer/count arrays、status code、explicit destroy/copy functions。
- C bridge 的 `.cpp` 内部可以使用现代 C++、STL、RAII、Eigen，但所有 exported C function 必须 catch exceptions 并转换为 `pgo_status_t` + `pgo_error_t`。
- Alembic 不进入 C++ core。`.abc` 由独立 C++ tool 消费 OBJ frames 后生成，并通过 Conan 管理 Alembic/Imath 依赖。
- Vulkan、Slang、GPU reductions、GPU linear solver、contact、IPC、FEM、time integrator 不属于 Milestone 1。

## 1. 目标目录结构

```text
.
  CMakeLists.txt
  CMakePresets.json
  conanfile.py
  .clang-format
  .gitignore
  README.md
  cmake/
    pgo_dependencies.cmake
    pgo_eigen.cmake
    pgo_options.cmake
    pgo_sanitizers.cmake
    pgo_warnings.cmake
  conan/
    profiles/
      default
      macos-arm64-apple-clang
      ubuntu-x86_64-gcc
      windows-x86_64-msvc
  include/
    pgo/
      assembly/
        cpu_assembler.hpp
        local_matrix.hpp
      base/
        assert.hpp
      dof/
        dirichlet_boundary.hpp
        displacement.hpp
        dof_layout.hpp
        reduced_dof_map.hpp
      energy/
        energy_concepts.hpp
        energy_sum.hpp
        mass_spring_local_energy_provider.hpp
        reduced_energy.hpp
      geometry/
        rest_mesh.hpp
        topology.hpp
      io/
        obj_frame_writer.hpp
        obj_reader.hpp
      math/
        backend.hpp
        eigen_backend.hpp
        finite_difference.hpp
        scalar.hpp
      solver/
        line_search.hpp
        newton_solver.hpp
        solver_result.hpp
      storage/
        array_view.hpp
        host_buffer.hpp
    pgo_c/
      export.h
      pgo.h
  src/
    c_api/
      CMakeLists.txt
      pgo_c.cpp
  examples/
    assets/
      cloth_grid.obj
    CMakeLists.txt
    mass_spring_cloth.cpp
  tests/
    CMakeLists.txt
    assembly/
      test_cpu_assembler.cpp
    base/
      test_assert.cpp
    dof/
      test_dof.cpp
    energy/
      test_energy_concepts.cpp
      test_mass_spring_local_energy_provider.cpp
    geometry/
      test_rest_mesh.cpp
    io/
      test_obj_io.cpp
    math/
      test_backend.cpp
      test_eigen_config.cpp
      test_finite_difference.cpp
    pgo_c/
      test_c_api.c
    solver/
      test_solver.cpp
  tools/
    CMakeLists.txt
    obj_frames_to_abc.cpp
  .github/
    workflows/
      ci.yml
```

## Phase 0: 仓库、构建系统、格式化和 CI 基础

**当前状态:** 已完成。实际落地版本相较原始计划做了三处合理调整：测试框架从 Catch2 改为 GoogleTest；移除 `fmt` 依赖，后续使用标准库格式化能力；Conan profile 放入仓库并由 CI 显式选择，避免依赖开发者本机 default profile。

**已验证命令:**

```bash
cmake --preset debug
cmake --build --preset debug
cmake --preset asan
cmake --build --preset asan
cmake --preset release
cmake --build --preset release
```

结果：三个 preset 均 configure/build 成功。Phase 0 尚无测试 target，所以 `ctest` 从 Phase 1 之后开始作为硬验证。

### Task 0.1: 初始化 git 和忽略文件（已完成）

**文件:**
- 创建: `.gitignore`

- [x] **Step 1: 如果当前目录还不是 git repository，初始化它**

```bash
git rev-parse --is-inside-work-tree || git init
```

期望：已有仓库时输出 `true`，否则初始化新仓库。

- [x] **Step 2: 创建 `.gitignore`**

```gitignore
build/
.cache/
.DS_Store
compile_commands.json
CMakeUserPresets.json
*.o
*.a
*.so
*.dylib
*.dll
*.exe
*.pyc
__pycache__/
frames/
*.abc
plan/
```


### Task 0.2: 添加 Conan 2 recipe 和仓库内 profiles（已完成）

**文件:**
- 创建: `conanfile.py`
- 创建: `conan/profiles/default`
- 创建: `conan/profiles/macos-arm64-apple-clang`
- 创建: `conan/profiles/ubuntu-x86_64-gcc`
- 创建: `conan/profiles/windows-x86_64-msvc`

- [x] **Step 1: 创建 `conanfile.py`**

```python
from conan import ConanFile


class PgoRecipe(ConanFile):
    name = "pgo"
    version = "0.1.0"
    package_type = "header-library"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("eigen/3.4.0")
        self.requires("cli11/[>=2.4 <3]")

    def build_requirements(self):
        self.test_requires("gtest/[>=1.14 <2]")
```

- [x] **Step 2: 添加仓库内 Conan profiles**

实际采用仓库内 profiles，而不是依赖用户本机的 Conan default profile：

```text
conan/profiles/default
conan/profiles/macos-arm64-apple-clang
conan/profiles/ubuntu-x86_64-gcc
conan/profiles/windows-x86_64-msvc
```

`conan/profiles/default` 根据平台 include 对应 profile：

```jinja
{% set sys = platform.system() %}

{% if sys == "Darwin" %}
include(macos-arm64-apple-clang)
{% elif sys == "Linux" %}
include(ubuntu-x86_64-gcc)
{% elif sys == "Windows" %}
include(windows-x86_64-msvc)
{% else %}
[settings]
os={{ sys }}
{% endif %}
```

- [x] **Step 3: 安装 Debug 依赖**

```bash
conan install . \
  --profile:host=conan/profiles/default \
  --profile:build=conan/profiles/default \
  --output-folder=build/conan/debug \
  --build=missing \
  -s:h build_type=Debug
```

期望：生成 `build/conan/debug/conan_toolchain.cmake`。

- [x] **Step 4: 安装 Release 依赖**

```bash
conan install . \
  --profile:host=conan/profiles/default \
  --profile:build=conan/profiles/default \
  --output-folder=build/conan/release \
  --build=missing \
  -s:h build_type=Release
```

期望：生成 `build/conan/release/conan_toolchain.cmake`。


### Task 0.3: 添加现代 CMake skeleton（已完成）

**文件:**
- 创建: `CMakeLists.txt`
- 创建: `CMakePresets.json`
- 创建: `cmake/pgo_options.cmake`
- 创建: `cmake/pgo_dependencies.cmake`
- 创建: `cmake/pgo_warnings.cmake`
- 创建: `cmake/pgo_sanitizers.cmake`

- [x] **Step 1: 创建 `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.28)

project(pgo VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/pgo_options.cmake)
include(cmake/pgo_dependencies.cmake)
include(cmake/pgo_warnings.cmake)
include(cmake/pgo_sanitizers.cmake)

add_library(pgo_core INTERFACE)
add_library(pgo::core ALIAS pgo_core)

target_compile_features(pgo_core INTERFACE cxx_std_23)
target_include_directories(pgo_core INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(pgo_core INTERFACE Eigen3::Eigen)
target_link_libraries(pgo_core INTERFACE pgo_project_warnings pgo_project_sanitizers)

if(PGO_BUILD_C_API AND EXISTS "${PROJECT_SOURCE_DIR}/src/c_api/CMakeLists.txt")
    add_subdirectory(src/c_api)
endif()

if(PGO_BUILD_EXAMPLES AND EXISTS "${PROJECT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

if(PGO_BUILD_TESTS AND EXISTS "${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()
```

- [x] **Step 2: 创建 `cmake/pgo_options.cmake`**

```cmake
option(PGO_BUILD_TESTS "Build pgo tests" ON)
option(PGO_BUILD_EXAMPLES "Build pgo examples" ON)
option(PGO_BUILD_C_API "Build C99 shared-library API" ON)
option(PGO_ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
option(PGO_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(PGO_ENABLE_GPU "Enable GPU backend targets" OFF)
```

- [x] **Step 3: 创建 `cmake/pgo_dependencies.cmake`**

```cmake
find_package(Eigen3 REQUIRED CONFIG)
if(PGO_BUILD_EXAMPLES)
    find_package(CLI11 REQUIRED CONFIG)
endif()

if(PGO_BUILD_TESTS)
    find_package(GTest REQUIRED CONFIG)
endif()
```

- [x] **Step 4: 创建 warnings/sanitizers helper**

`cmake/pgo_warnings.cmake`:

```cmake
add_library(pgo_project_warnings INTERFACE)

if(MSVC)
    target_compile_options(pgo_project_warnings INTERFACE /W4)
    if(PGO_WARNINGS_AS_ERRORS)
        target_compile_options(pgo_project_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(pgo_project_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
    )
    if(PGO_WARNINGS_AS_ERRORS)
        target_compile_options(pgo_project_warnings INTERFACE -Werror)
    endif()
endif()
```

`cmake/pgo_sanitizers.cmake`:

```cmake
add_library(pgo_project_sanitizers INTERFACE)

if(PGO_ENABLE_SANITIZERS)
    if(MSVC)
        message(WARNING "PGO_ENABLE_SANITIZERS is currently configured for Clang/GCC style sanitizers only")
    else()
        target_compile_options(pgo_project_sanitizers INTERFACE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(pgo_project_sanitizers INTERFACE
            -fsanitize=address,undefined
        )
    endif()
endif()
```

- [x] **Step 5: 创建 `CMakePresets.json`**

包含 `debug`、`asan`、`release` 三个 configure/build preset。`debug` 和 `asan` 使用 `build/conan/debug/conan_toolchain.cmake`，`release` 使用 `build/conan/release/conan_toolchain.cmake`。

- [x] **Step 6: 验证 configure/build**

```bash
cmake --preset debug
cmake --build --preset debug
cmake --preset asan
cmake --build --preset asan
cmake --preset release
cmake --build --preset release
```

期望：三个 preset 都 configure/build 成功。Phase 0 尚无 `tests/CMakeLists.txt`，因此此阶段不要求 `ctest`。


### Task 0.4: 添加 clang-format（已完成）

**文件:**
- 创建: `.clang-format`

- [x] **Step 1: 创建 `.clang-format`**

```yaml
BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 120
PointerAlignment: Left
ReferenceAlignment: Left
BreakBeforeBraces: Attach
AllowShortFunctionsOnASingleLine: Empty
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
NamespaceIndentation: None
SortIncludes: CaseSensitive
```


### Task 0.5: 添加 Phase 0 README 和 build-only CI（已完成）

**文件:**
- 创建: `README.md`
- 创建: `.github/workflows/ci.yml`

- [x] **Step 1: README 记录工具和构建命令**

README 使用 `uv tool install conan` / `uv tool install ninja` 管理 Python tooling，并记录 Debug、Release、ASan/UBSan 三套本地构建命令。

- [x] **Step 2: CI 使用仓库内 Conan profiles**

`.github/workflows/ci.yml` 当前包含：

- `ubuntu-latest` + `conan/profiles/ubuntu-x86_64-gcc`
- `macos-latest` + `conan/profiles/macos-arm64-apple-clang`
- `windows-latest` + `conan/profiles/windows-x86_64-msvc`
- `ubuntu-latest` ASan/UBSan

Phase 0 CI 只做 configure/build。等 Phase 1 创建 `tests/CMakeLists.txt` 后，Phase 7 再把 `ctest` 加回 CI。


## Phase 0.5: Eigen CPU acceleration 配置层

**目标:** 在进入 math backend 代码前，先把 Eigen 的 CPU acceleration 配置体系化。这个 phase 只管理 Eigen 与 BLAS/LAPACK 后端的编译/链接配置，不负责选择 Newton 线性求解器。

**设计边界:**

- `PGO_ENABLE_EIGEN_ACCELERATION` 控制 Eigen 是否启用平台 BLAS/LAPACK acceleration。
- `PGO_EIGEN_ACCELERATION_BACKEND` 控制 acceleration backend：`AUTO` / `MKL` / `ACCELERATE` / `NONE`。
- Apple 平台 `AUTO` 选择 Accelerate。
- Ubuntu/Windows 平台 `AUTO` 选择 MKL。
- 默认 `PGO_ENABLE_EIGEN_ACCELERATION=OFF`，保证普通开发和 CI 不要求 MKL/Accelerate 环境。
- Apple Accelerate 是系统 framework，不通过 Conan 管理。
- oneMKL 不通过 Conan 管理；Ubuntu/Windows acceleration 配置先确保系统 oneMKL 已安装，CMake 会检查 oneMKL 默认安装位置，也允许用户通过 `MKL_DIR` 或 `CMAKE_PREFIX_PATH` 指定自定义位置。
- PARDISO 是 MKL 中的 sparse direct solver，但它不是 Eigen 的 `MathBackend`，也不是这个 acceleration option 的语义。后续 solver 层应单独引入 `PGO_CPU_LINEAR_SOLVER=EIGEN_SIMPLICIAL_LDLT/MKL_PARDISO/...`。
- 选项名使用 `PGO_ENABLE_EIGEN_ACCELERATION`，不要使用拼写错误的 `acceloration`。

### Task 0.5.1: 添加系统 oneMKL 平台边界

**文件:**
- 修改: `conanfile.py`
- 创建: `scripts/install-onemkl/install-onemkl-linux.sh`
- 创建: `scripts/install-onemkl/install-onemkl-windows.ps1`

- [x] **Step 1: 保持 Conan 依赖不包含 MKL**

`conanfile.py` 只管理跨平台 C++ 包和测试/benchmark 依赖：

```python
from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain


class PgoRecipe(ConanFile):
    name = "pgo"
    version = "0.1.0"
    package_type = "header-library"
    settings = "os", "compiler", "build_type", "arch"

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        toolchain = CMakeToolchain(self, generator="Ninja")
        toolchain.user_presets_path = None
        toolchain.generate()

    def requirements(self):
        self.requires("eigen/3.4.0")
        self.requires("cli11/[>=2.4 <3]")

    def build_requirements(self):
        self.test_requires("benchmark/[>=1.9 <2]")
        self.test_requires("gtest/[>=1.14 <2]")
```

设计约束：

- macOS 不通过 Conan 拉取 MKL；macOS acceleration 只使用系统 Accelerate。`requirements()` 里不要出现 MKL requirement。
- Linux/Windows MKL acceleration jobs 必须在 configure 前安装系统 oneMKL，并让 `find_package(MKL CONFIG REQUIRED)` 能找到 `MKLConfig.cmake`。
- CMake 仍只使用 `find_package(MKL CONFIG REQUIRED)`，由系统 oneMKL 提供 `MKLConfig.cmake`。

- [x] **Step 2: 验证默认 Conan 依赖不拉 MKL**

```bash
conan install . \
  --profile:host=conan/profiles/default \
  --profile:build=conan/profiles/default \
  --output-folder=build/conan/debug \
  --build=missing \
  -s:h build_type=Debug
```

期望：依赖图包含 Eigen/CLI11/GTest，不包含 MKL。

- [x] **Step 3: 添加 Linux/Windows oneMKL 安装脚本**

Linux 安装脚本通过 Intel APT repository 安装 `intel-oneapi-mkl-devel`，并校验默认安装位置提供 `MKLConfig.cmake`。在 GitHub Actions 中额外写入 `MKLROOT`、`MKL_DIR`、`CMAKE_PREFIX_PATH`、`LD_LIBRARY_PATH`、`LIBRARY_PATH`。

Windows 安装脚本通过 winget 安装 `Intel.oneMKL`，并校验默认安装位置提供 `MKLConfig.cmake`。脚本写入 `MKLROOT`、`MKL_DIR`、`CMAKE_PREFIX_PATH`、`LIB`，同时把 `mkl/latest/bin`、`mkl/latest/redist/intel64`、`mkl/latest/lib/intel64`、`compiler/latest/bin` 中存在的目录加入运行时 `PATH`；在 GitHub Actions 中通过 `GITHUB_PATH` 暴露这些 DLL 目录。

- [x] **Step 4: 验证 Linux/Windows 系统 oneMKL 配置**

Linux:

```bash
scripts/install-onemkl/install-onemkl-linux.sh
conan install . \
  --profile:host=conan/profiles/ubuntu-x86_64-gcc \
  --profile:build=conan/profiles/ubuntu-x86_64-gcc \
  --output-folder=build/conan/debug-acceleration \
  --build=missing \
  -s:h build_type=Debug
```

Windows:

```powershell
.\scripts\install-onemkl\install-onemkl-windows.ps1
conan install . `
  --profile:host=conan/profiles/windows-x86_64-msvc `
  --profile:build=conan/profiles/windows-x86_64-msvc `
  --output-folder=build/conan/debug-acceleration `
  --build=missing `
  -s:h build_type=Debug
```

期望：Conan 依赖图不包含 MKL；系统 oneMKL 提供 `MKLConfig.cmake`，CMake configure 阶段能成功链接 `MKL::MKL`。

### Task 0.5.2: 添加 Eigen 配置 target

**文件:**
- 创建: `cmake/pgo_eigen.cmake`
- 修改: `cmake/pgo_options.cmake`
- 修改: `CMakeLists.txt`

- [x] **Step 1: 在 `cmake/pgo_options.cmake` 中添加选项**

```cmake
option(PGO_ENABLE_EIGEN_ACCELERATION "Enable Eigen BLAS/LAPACK acceleration" OFF)

set(PGO_EIGEN_ACCELERATION_BACKEND "AUTO" CACHE STRING "Eigen acceleration backend: AUTO, MKL, ACCELERATE, NONE")
set_property(CACHE PGO_EIGEN_ACCELERATION_BACKEND PROPERTY STRINGS AUTO MKL ACCELERATE NONE)
```

- [x] **Step 2: 创建 `cmake/pgo_eigen.cmake`**

```cmake
add_library(pgo_eigen_config INTERFACE)
add_library(pgo::eigen_config ALIAS pgo_eigen_config)

target_link_libraries(pgo_eigen_config INTERFACE Eigen3::Eigen)

set(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND "NONE")

if(PGO_ENABLE_EIGEN_ACCELERATION)
    if(PGO_EIGEN_ACCELERATION_BACKEND STREQUAL "AUTO")
        if(APPLE)
            set(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND "ACCELERATE")
        elseif(UNIX OR WIN32)
            set(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND "MKL")
        else()
            set(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND "NONE")
        endif()
    else()
        set(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND "${PGO_EIGEN_ACCELERATION_BACKEND}")
    endif()
endif()

if(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND STREQUAL "MKL")
    find_package(MKL CONFIG REQUIRED)

    target_compile_definitions(pgo_eigen_config INTERFACE
        EIGEN_USE_MKL_ALL
        PGO_EIGEN_ACCELERATION_MKL
    )

    target_link_libraries(pgo_eigen_config INTERFACE MKL::MKL)
elseif(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND STREQUAL "ACCELERATE")
    if(NOT APPLE)
        message(FATAL_ERROR "PGO_EIGEN_ACCELERATION_BACKEND=ACCELERATE is only supported on Apple platforms.")
    endif()

    find_library(PGO_ACCELERATE_FRAMEWORK Accelerate REQUIRED)

    target_compile_definitions(pgo_eigen_config INTERFACE
        EIGEN_USE_BLAS
        PGO_EIGEN_ACCELERATION_ACCELERATE
    )

    target_link_libraries(pgo_eigen_config INTERFACE "${PGO_ACCELERATE_FRAMEWORK}")
elseif(PGO_SELECTED_EIGEN_ACCELERATION_BACKEND STREQUAL "NONE")
    target_compile_definitions(pgo_eigen_config INTERFACE
        PGO_EIGEN_ACCELERATION_NONE
    )
else()
    message(FATAL_ERROR "Unknown PGO_SELECTED_EIGEN_ACCELERATION_BACKEND=${PGO_SELECTED_EIGEN_ACCELERATION_BACKEND}")
endif()
```

设计约束：

- `EIGEN_USE_MKL_ALL` / `EIGEN_USE_BLAS` 必须通过 `pgo::eigen_config` target 传播，不能散落在源码里。
- Apple 第一版只定义 `EIGEN_USE_BLAS`，不默认定义 `EIGEN_USE_LAPACKE`。Accelerate 的 LAPACK/LAPACKE 接口后续用单独测试确认后再扩展。
- `PGO_ENABLE_EIGEN_ACCELERATION=ON` 且 backend 选到 MKL 时，找不到 MKL 应直接 configure 失败，不静默降级。

- [x] **Step 3: 修改顶层 `CMakeLists.txt`**

include 顺序改为：

```cmake
include(cmake/pgo_options.cmake)
include(cmake/pgo_dependencies.cmake)
include(cmake/pgo_eigen.cmake)
include(cmake/pgo_warnings.cmake)
include(cmake/pgo_sanitizers.cmake)
```

`pgo_core` 不再直接链接 `Eigen3::Eigen`，而是链接 `pgo::eigen_config`：

```cmake
target_link_libraries(pgo_core INTERFACE pgo::eigen_config)
target_link_libraries(pgo_core INTERFACE pgo_project_warnings pgo_project_sanitizers)
```

### Task 0.5.3: 添加 acceleration presets、smoke test 和验证命令

**文件:**
- 修改: `CMakePresets.json`
- 创建: `tests/math/test_eigen_config.cpp`
- 修改: `tests/CMakeLists.txt`

- [x] **Step 1: 添加测试文件**

`tests/math/test_eigen_config.cpp` 使用 `namespace pgo::math::test`。测试内容：

```cpp
#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace pgo::math::test {

TEST(EigenConfig, DenseMatrixMultiplyWorks) {
    Eigen::Matrix2d a;
    a << 1.0, 2.0,
         3.0, 4.0;

    const Eigen::Matrix2d b = Eigen::Matrix2d::Identity();
    const Eigen::Matrix2d c = a * b;

    EXPECT_DOUBLE_EQ(c(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(c(1, 1), 4.0);
}

} // namespace pgo::math::test
```

- [x] **Step 2: 把测试加入 `tests/CMakeLists.txt`**

```cmake
add_executable(pgo_tests
    base/test_assert.cpp
    math/test_backend.cpp
    math/test_eigen_config.cpp
    geometry/test_rest_mesh.cpp
)
```

- [x] **Step 3: 添加 preset-driven acceleration 配置**

`CMakePresets.json` 保持 baseline presets：

```text
debug
release
asan
```

并新增 acceleration presets：

```text
debug-acceleration
release-acceleration
```

`debug-acceleration` 使用：

```json
{
  "name": "debug-acceleration",
  "inherits": "base",
  "binaryDir": "${sourceDir}/build/debug-acceleration",
  "cacheVariables": {
    "CMAKE_BUILD_TYPE": "Debug",
    "CMAKE_TOOLCHAIN_FILE": "${sourceDir}/build/conan/debug-acceleration/conan_toolchain.cmake",
    "PGO_ENABLE_EIGEN_ACCELERATION": "ON",
    "PGO_EIGEN_ACCELERATION_BACKEND": "AUTO"
  }
}
```

`release-acceleration` 使用：

```json
{
  "name": "release-acceleration",
  "inherits": "base",
  "binaryDir": "${sourceDir}/build/release-acceleration",
  "cacheVariables": {
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_TOOLCHAIN_FILE": "${sourceDir}/build/conan/release-acceleration/conan_toolchain.cmake",
    "PGO_ENABLE_EIGEN_ACCELERATION": "ON",
    "PGO_EIGEN_ACCELERATION_BACKEND": "AUTO"
  }
}
```

要求同时添加同名 build/test presets：

```text
cmake --build --preset debug-acceleration
ctest --preset debug-acceleration
cmake --build --preset release-acceleration
ctest --preset release-acceleration
```

设计约束：

- Preset 不写死 `MKL` 或 `ACCELERATE`，统一使用 `PGO_EIGEN_ACCELERATION_BACKEND=AUTO`。
- 平台差异由 CMake 自动选择：macOS -> Accelerate，Linux/Windows -> MKL。
- 平台依赖由系统环境决定：macOS 使用系统 Accelerate，Linux/Windows acceleration job 在 configure 前安装 oneMKL。

- [x] **Step 4: 验证默认配置**

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug -R EigenConfig
```

期望：不需要 MKL 或 Accelerate 额外配置，测试通过。

- [x] **Step 5: 验证 Apple Accelerate 配置**

仅在 macOS 上运行：

```bash
conan install . \
  --profile:host=conan/profiles/macos-arm64-apple-clang \
  --profile:build=conan/profiles/macos-arm64-apple-clang \
  --output-folder=build/conan/debug-acceleration \
  --build=missing \
  -s:h build_type=Debug
cmake --preset debug-acceleration
cmake --build --preset debug-acceleration
ctest --preset debug-acceleration -R EigenConfig
```

期望：`AUTO` 选择 Accelerate，链接 Accelerate framework，测试通过。

- [x] **Step 6: 验证 MKL 配置**

仅在 Ubuntu/Windows 环境运行。先安装系统 oneMKL 并暴露 `MKLConfig.cmake`：

```bash
scripts/install-onemkl/install-onemkl-linux.sh
conan install . \
  --profile:host=conan/profiles/ubuntu-x86_64-gcc \
  --profile:build=conan/profiles/ubuntu-x86_64-gcc \
  --output-folder=build/conan/debug-acceleration \
  --build=missing \
  -s:h build_type=Debug
```

然后运行：

```bash
cmake --preset debug-acceleration
cmake --build --preset debug-acceleration
ctest --preset debug-acceleration -R EigenConfig
```

期望：`AUTO` 选择 MKL，链接 `MKL::MKL`，测试通过。

### Task 0.5.4: 记录 PARDISO 与 solver backend 边界

**文件:**
- 修改: `README.md`
- 修改: `plan/milestion1_mass_spring_cpu.plan.md`

- [x] **Step 1: 在 README 记录 Eigen acceleration 语义**

必须写明：

```text
PGO_ENABLE_EIGEN_ACCELERATION controls Eigen's BLAS/LAPACK acceleration path.
It does not select the sparse linear solver used by Newton iterations.
```

- [x] **Step 2: 记录 PARDISO 的位置**

必须写明：

```text
MKL is a math library suite. PARDISO is MKL's sparse direct solver.
Eigen::PardisoLDLT is Eigen's wrapper around MKL PARDISO.
PGO will model PARDISO as a linear solver backend, not as a MathBackend.
```

- [x] **Step 3: 预留后续 solver option，不在 Phase 0.5 实现**

后续 solver 层可以引入：

```text
PGO_CPU_LINEAR_SOLVER=EIGEN_SIMPLICIAL_LDLT
PGO_CPU_LINEAR_SOLVER=MKL_PARDISO
PGO_CPU_LINEAR_SOLVER=EIGEN_CONJUGATE_GRADIENT
```

Milestone 1 默认仍使用 Eigen `SimplicialLDLT`，直到 solver policy 文件稳定后再引入 PARDISO。


## Phase 1: Base、Math、Storage、Geometry、DOF 核心

**执行策略:** Phase 1 不要一口气把 geometry 和 DOF 全塞进去。先做 `base/`、`math/`、`storage/`、`geometry/` 的最小可编译/可测试闭环，跑通 `test_rest_mesh`；然后再进入 DOF/reduced map。这样如果后续出错，问题会落在很小的边界里。

**测试目录和命名空间规范:** `tests/` 下的测试源文件目录要和主代码模块对齐，不把所有测试平铺在 `tests/` 根目录。比如 `include/pgo/math/...` 对应 `tests/math/...`，`include/pgo/geometry/...` 对应 `tests/geometry/...`。C++ 测试文件里的 `TEST`/helper 放在对应模块的 `pgo::<module>::test` 命名空间中。例如 base 测试使用 `namespace pgo::base::test`，math backend 测试使用 `namespace pgo::math::test`，geometry 测试使用 `namespace pgo::geometry::test`。纯 C API 测试 `tests/pgo_c/test_c_api.c` 不适用这个 C++ namespace 规范。

### Task 1.1: 添加基础 assert

**文件:**
- 创建: `include/pgo/base/assert.hpp`
- 创建/移动: `tests/base/test_assert.cpp`
- 修改: `tests/CMakeLists.txt`

- [x] **Step 1: 实现 `pgo::base::require`**

`require(condition, message)` 在 condition 为 false 时抛出 `std::runtime_error`。这是 C++ core 内部使用的错误机制，不能越过 C API 边界。

如果当前已有根目录平铺的 base assert 测试文件，将它整理到 `tests/base/test_assert.cpp`，并使用 `namespace pgo::base::test`。


### Task 1.2: 添加基本数学类型 (Eigen Aliases)

**文件:**
- 创建: `include/pgo/math/types.hpp`
- 创建: `tests/math/test_backend.cpp`
- 修改: `tests/CMakeLists.txt`

- [x] **Step 1: 定义扁平化数学类型**

`types.hpp` 提供对 Eigen 类型的直接 alias：

```cpp
namespace pgo::math {

using Index = std::uint32_t;
using DenseIndex = Eigen::Index;

[[nodiscard]] constexpr DenseIndex dense_index(const std::size_t index) {
    return static_cast<DenseIndex>(index);
}

template <typename T, int Dim>
using Vec = Eigen::Matrix<T, Dim, 1>;

template <typename T, int Rows, int Cols>
using Mat = Eigen::Matrix<T, Rows, Cols>;

template <typename T>
using DVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using DMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using SparseMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;

template <typename T>
using Triplet = Eigen::Triplet<T>;

} // namespace pgo::math
```

设计约束：

- 移除了复杂的 `MathBackend` 概念和 `scalar.hpp` 类型约束，转而直接在 `types.hpp` 中通过 `typename T` 提供透明的 Eigen alias。
- 业务代码优先使用 `pgo::math::Vec`、`DVec`、`SparseMat` 等别名，保持代码整洁。
- `SparseMat<T>` 和 `Triplet<T>` 约定只接受浮点数（受限于 Eigen）。

- [x] **Step 2: 添加 math types 测试**

在 `tests/CMakeLists.txt` 中加入 `tests/math/test_backend.cpp`。测试文件命名空间使用：

```cpp
namespace pgo::math::test {
// TEST(...)
}
```

测试内容：

- `pgo::math::Vec<double, 3>` 的 size 是 3。
- `pgo::math::DVec<double>` 可以 resize 到 4。
- `pgo::math::SparseMat<double>` 是 row-major sparse matrix。


### Task 1.3: 添加 GPU-aware host storage primitive

**文件:**
- 创建: `include/pgo/storage/array_view.hpp`
- 创建: `include/pgo/storage/host_buffer.hpp`

- [x] **Step 1: 定义 view/buffer**

```cpp
namespace pgo::storage {
template <typename T>
using ArrayView = std::span<T>;

template <typename T>
using ConstArrayView = std::span<const T>;

template <typename T>
using HostBuffer = std::vector<T>;
}
```

要求：这些类型只表示 host storage 语义。Milestone 2 可以平行添加 `DeviceBuffer` / `DeviceArrayView`，不需要重写 geometry 层。


### Task 1.4: 添加 RestMesh、topology 和第一组测试闭环

**文件:**
- 创建: `include/pgo/geometry/topology.hpp`
- 创建: `include/pgo/geometry/rest_mesh.hpp`
- 创建/修改: `tests/CMakeLists.txt`
- 创建: `tests/geometry/test_rest_mesh.cpp`

- [x] **Step 1: 定义 topology**

```cpp
namespace pgo::geometry {
using VertexIndex = pgo::math::Index;

inline constexpr std::size_t kEdgeArity = 2;
inline constexpr std::size_t kFaceArity = 3;
}
```

设计约束：topology 使用 flat index buffers，而不是 `std::array<VertexIndex, 2>` / `std::array<VertexIndex, 3>` object buffers。这样 CPU 访问公式和未来 Vulkan/Slang buffer 访问公式一致：

```text
edge vertex = edge_indices[kEdgeArity * edge_id + local_vertex]
face vertex = face_indices[kFaceArity * face_id + local_vertex]
```

- [x] **Step 2: 定义 `RestMesh<T, Dim>`**

要求：

- `rest_positions` 是 vertex-major flat buffer。
- `num_vertices() == rest_positions.size() / Dim`。
- `rest_position(i)` 返回 `math::Vec<T, Dim>`。
- 保存 `edge_indices()` 和 `face_indices()`，长度分别是 `2 * num_edges()` 和 `3 * num_faces()`。
- 提供 `edge_vertex(edge_id, local_vertex)` 和 `face_vertex(face_id, local_vertex)` helper。
- 不在 storage 中保存 Eigen vector object。
- 不在 topology storage 中保存 `std::array`、指针、对象图或 per-edge/per-face 动态分配。

- [x] **Step 3: 建立测试 target**

`tests/CMakeLists.txt` 创建 `pgo_tests`，包含当前已经存在的 base/math/geometry 测试：

```cmake
add_executable(pgo_tests
    base/test_assert.cpp
    math/test_backend.cpp
    math/test_eigen_config.cpp
    geometry/test_rest_mesh.cpp
)

target_link_libraries(pgo_tests PRIVATE pgo::core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(pgo_tests)
```

`tests/geometry/test_rest_mesh.cpp` 使用 `namespace pgo::geometry::test`，至少测试：

- `RestMesh<double, 3>` 能从 flat position buffer 构造。
- `num_vertices()`、`num_edges()`、`num_faces()` 正确。
- `edge_indices().size() == 2 * num_edges()`。
- `face_indices().size() == 3 * num_faces()`。
- `edge_vertex(e, local)` 和 `face_vertex(f, local)` 返回正确 vertex index。
- `rest_position(i)` 返回正确坐标。
- position buffer 长度不是 `Dim` 的倍数时抛出异常。
- `edge_indices` 长度不是 `kEdgeArity` 的倍数时抛出异常。
- `face_indices` 长度不是 `kFaceArity` 的倍数时抛出异常。

- [x] **Step 4: 运行测试**

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```


### Task 1.5: 添加 DOF layout、displacement、Dirichlet boundary 和 reduced map

**文件:**
- 创建: `include/pgo/dof/dof_layout.hpp`
- 创建: `include/pgo/dof/displacement.hpp`
- 创建: `include/pgo/dof/dirichlet_boundary.hpp`
- 创建: `include/pgo/dof/reduced_dof_map.hpp`
- 修改: `tests/CMakeLists.txt`
- 创建: `tests/dof/test_dof.cpp`

- [x] **Step 1: 实现 `DofLayout<Dim>`**

职责：把 `(vertex, component)` 映射到 full displacement dof index：

```text
dof = vertex * Dim + component
```

- [x] **Step 2: 实现 `Displacement<T, Dim>`**

职责：持有 full displacement vector `u`，提供：

- `layout()`
- `vector()`
- `at(vertex)` 返回该 vertex 的 displacement vector

- [x] **Step 3: 实现 `DirichletBoundary<T>` 和 vertex helper free functions**

`DirichletBoundary<T>` 是稀疏的边界条件 builder / prescribed value store，底层使用 `std::unordered_map<std::size_t, T>` 保存 fixed DOF values。它不需要稳定遍历顺序；`ReducedDofMap` 通过遍历 `[0, full_dofs)` 构造 deterministic free/fixed mapping。

只提供核心的标量 DOF 操作：

- `prescribe_dof(dof, value)`：固定单个 scalar displacement DOF 到指定值。
- `fix_dof(dof)`：固定单个 scalar displacement DOF 到 `0`，dispatch 到 `prescribe_dof(dof, T{0})`。
- `is_fixed(dof)`：查询 full DOF 是否固定。
- `fixed_value(dof)`：读取 fixed displacement value；如果不是 fixed DOF，返回 `std::nullopt`。
- `value(dof)`：读取 fixed displacement value 的 throwing convenience API。

实现要求：

- `prescribe_dof(dof, value)` 直接写入/覆盖 `unordered_map`。
- `fix_dof(dof)` 调用 `prescribe_dof(dof, T{0})`。
- `is_fixed(dof)` 使用 `contains`。
- `fixed_value(dof)` 找不到时返回 `std::nullopt`。
- `value(dof)` 找不到时抛出 `std::runtime_error`。
- `DirichletBoundary` 是 host-side builder，不能进入 geometry/storage/GPU-facing flat buffers。

`DirichletBoundaryLike<Boundary, T>` concept 只要求 `fixed_value(dof) -> std::optional<T>`。`is_fixed(dof)` 和 `value(dof)` 是具体 boundary 类型可提供的 convenience API，但 solver/reduced map 不依赖它们。`DirichletBoundary` 和 `ReducedDofMap` 均满足此 concept。

Vertex-level convenience helpers 提取为 **namespace-scope free functions**，对任何支持 `prescribe_dof`/`fix_dof` 的 boundary 类型通用：

- `prescribe_component(boundary, layout, vertex, component, value)`
- `fix_component(boundary, layout, vertex, component)`
- `prescribe_vertex(boundary, layout, vertex, value)`
- `fix_vertex(boundary, layout, vertex)`
- `prescribe_vertices(boundary, layout, vertex_indices, value)`
- `fix_vertices(boundary, layout, vertex_indices)`
- `prescribe_vertices_by_list(boundary, layout, vertex_indices, values)`

设计约束：solver/reduced map 只依赖 `DirichletBoundaryLike` concept；vertex helpers 是 free functions，不与具体 boundary 类绑定。

- [x] **Step 4: 实现语义化 `ReducedDofMap<T>` API**

职责：从 `DirichletBoundaryLike` boundary snapshot 构建 immutable 的 DOF 映射，同时自身也满足 `DirichletBoundaryLike` concept（可作为 dense boundary snapshot 使用）。

核心 API：

- 构造：`ReducedDofMap(full_dofs, boundary)` — 从 boundary 生成 `free_to_full` / `full_to_free` 双向映射并快照 prescribed values。
- 查询：`full_dofs()`, `free_dofs()`, `full_dof(free)`, `free_dof(full)`, `is_free(full)`, `is_fixed(full)`, `fixed_value(full) -> std::optional<T>`, `value(full)`。

必须按不同数学对象拆分 API，不允许用一个 `pack/unpack/reduce` 名字同时表达多种语义：

```cpp
[[nodiscard]] pgo::math::DVec<T>
scatter_solution(const pgo::math::DVec<T>& free_solution) const;

[[nodiscard]] pgo::math::DVec<T>
scatter_direction(const pgo::math::DVec<T>& free_direction) const;

[[nodiscard]] pgo::math::DVec<T>
restrict_vector_to_free(const pgo::math::DVec<T>& full_vector) const;

[[nodiscard]] pgo::math::SparseMat<T>
restrict_matrix_to_free(const pgo::math::SparseMat<T>& full_matrix) const;

[[nodiscard]] pgo::math::DVec<T>
eliminate_rhs_for_dirichlet(
    const pgo::math::DVec<T>& full_rhs,
    const pgo::math::SparseMat<T>& full_matrix) const;
```

数学语义：

- `scatter_solution(free_solution)` 对应 solution/state：

```text
u = P y + G g
```

free DOF 填入 `free_solution`，fixed DOF 填入 snapshot prescribed values。

- `scatter_direction(free_direction)` 对应 direction/increment：

```text
du = P dy
```

free DOF 填入 `free_direction`，fixed DOF 必须填 `0`。不要用 `scatter_solution(delta_y)` 处理 Newton direction、line-search direction、CCD direction 或 debug step direction。

- `restrict_vector_to_free(full_vector)` 对应 vector restriction：

```text
v_red = P^T v_full
```

它只是提取 full vector 的 free DOF，不做 `K_fc g` 修正。IPC / energy reduced gradient 必须使用这个 API。

- `restrict_matrix_to_free(full_matrix)` 对应 matrix restriction：

```text
A_red = P^T A_full P
```

它只提取 free-free block。IPC / energy reduced Hessian 必须使用这个 API。

- `eliminate_rhs_for_dirichlet(full_rhs, full_matrix)` 对应直接线性系统 Dirichlet 消元：

```text
A u = b
A_ff y = b_f - A_fc g
```

它只适合直接线性系统 RHS 消元，不能用于 energy gradient projection。

设计约束：

- 不提供 `pack_vector`、`reduce_vector`、`reduce_sparse_mat` 这类含糊 API。
- 不保留 `pack_rhs`、`pack_matrix`、`unpack_solution` 旧命名。
- `unpack_matrix` 不作为 Phase 1 核心 API 保留；如果后续确实需要 full-size debug/visualization matrix，再单独设计 `scatter_matrix_with_dirichlet_identity`。
- `ReducedDofMap` 是 construct-once immutable snapshot，不提供 mutable 的 `prescribe_dof`。
- 当前 Milestone 1 保留 snapshot prescribed values；如果 prescribed displacement 后续随 timestep 或 animation 高频变化，再升级为 pattern map 与 value provider 分离的 API，例如 `scatter_solution(free_solution, boundary)`。

- [x] **Step 5: 把 DOF 测试加入测试 target**

修改 `tests/CMakeLists.txt`：

```cmake
add_executable(pgo_tests
    base/test_assert.cpp
    math/test_backend.cpp
    math/test_eigen_config.cpp
    geometry/test_rest_mesh.cpp
    dof/test_dof.cpp
)
```

- [x] **Step 6: 添加 DOF 测试**

`tests/dof/test_dof.cpp` 使用 `namespace pgo::dof::test`。测试内容：

- `DofLayout<3>(4)` 有 12 个 DOF。
- `layout.index(2, 1) == 7`。
- `DirichletBoundary` 核心 API：`fix_dof`、`prescribe_dof`、`is_fixed`、`value`。
- `DirichletBoundary` 覆盖写入：同一个 DOF 多次 `prescribe_dof` 后返回最后一次 prescribed value。
- Vertex helper free functions：`fix_vertex`、`prescribe_vertex`、`fix_vertices`、`prescribe_vertices`、`prescribe_vertices_by_list`。
- `ReducedDofMap` 满足 `DirichletBoundaryLike` concept（`fixed_value` 返回 `std::optional<T>`）。
- `scatter_solution(free_solution)` 还原 solution/state：free DOF 填 reduced solution，fixed DOF 填 snapshot prescribed values。
- `scatter_direction(free_direction)` 还原 direction/increment：free DOF 填 reduced direction，fixed DOF 填 `0`。
- `restrict_vector_to_free(full_vector)` 正确实现 `P^T v`。
- `restrict_matrix_to_free(full_matrix)` 正确实现 `P^T A P`，即提取 free-free sparse block。
- `eliminate_rhs_for_dirichlet(full_rhs, full_matrix)` 正确实现直接线性系统 RHS 消元（`b_f − A_fc · g`），非零 prescribed values 下验证。
- End-to-end elimination test：构建 SPD 系统 → `restrict_matrix_to_free(A)` → `eliminate_rhs_for_dirichlet(b, A)` → solve → `scatter_solution(y)` → 验证 `A·u = b` 的 free rows 残差为零。

- [x] **Step 7: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R dof
```

## Phase 2: OBJ Mesh 输入和 OBJ Frame 输出

### Task 2.1: 添加 OBJ reader

**文件:**
- 创建: `include/pgo/io/obj_reader.hpp`
- 修改: `tests/io/test_obj_io.cpp`

- [x] **Step 1: 实现 `read_obj_rest_mesh<T, Dim>(path)`**

支持：

- `v x y z`
- `l i j`
- triangular `f i j k`
- `f` 中的 `i/j/k` 或 `i//k` token，只读取第一个 vertex index
- OBJ positive one-based indices

要求：

- Faces 自动抽取 undirected unique edges，并写入 flat `edge_indices`。
- OBJ `l` records 也写入 flat `edge_indices`。
- Triangular faces 写入 flat `face_indices`。
- 读取为 `RestMesh<T, Dim>`。
- `Dim == 2` 时丢弃 OBJ z 坐标。

- [x] **Step 2: 添加测试**

创建临时 OBJ：4 个 vertices、2 个 triangles。读取后断言：

- `num_vertices() == 4`
- `num_faces() == 2`
- `num_edges() == 5`
- `face_indices().size() == 6`
- `edge_indices().size() == 10`

- [x] **Step 3: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R obj
```

### Task 2.2: 添加 OBJ frame writer

**文件:**
- 创建: `include/pgo/io/obj_frame_writer.hpp`
- 修改: `tests/io/test_obj_io.cpp`

- [x] **Step 1: 实现 `ObjFrameWriter<T, Dim>`**

职责：

- 输入 `RestMesh<T, Dim>` 和 full displacement vector `u`。
- 输出 `frame_0000.obj`、`frame_0001.obj` 等。
- 写出的 vertex position 是 `X + u`。
- 通过 `face_indices` 保留原 faces。
- 如果 mesh 没有 faces，则通过 `edge_indices` 写 `l` records。

- [x] **Step 2: 添加 writer 测试**

创建两点 line mesh，设置第二个点的 displacement，写出 frame 后检查 OBJ 文本包含 displaced vertex。

- [x] **Step 3: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R obj
```

## Phase 3: Local-Contribution Energy 和 CPU Assembly

**当前状态:** 已完成。实际代码已经落地 `finite_difference.hpp`、energy concepts、local matrix aliases、CPU assembler、`MassSpringLocalEnergyModel` 和 `MassSpringLocalEnergyProvider`，并补齐 finite difference、concept、assembler、mass-spring local/global derivative 测试。plan checkbox 在本次更新中按代码现状回填。

**已验证命令:**

```bash
ctest --preset debug --output-on-failure
```

结果：41/41 tests passed。Phase 3 相关测试覆盖 `finite_difference`、`EnergyConcept`、`CPUAssembler`、`MassSpringLocalEnergyModel`、`MassSpringLocalEnergyProvider`，包括 local derivative finite difference、global assembly derivative finite difference 和 shared DOF contribution 累加。

**目标:** 把“局部物理公式”变成“全局可求导 sparse optimization system”。Phase 1/2 已经建立了 flat mesh、full displacement DOF、reduced DOF 和 OBJ pipeline；Phase 3 第一次引入真正的物理 energy，并让系统能够计算 full-space `E(u)`、`grad E(u)`、`H(u)`。

**核心设计决策:**

- Phase 3 的核心抽象是：

```text
E_full(u) = sum_i E_i(u)
```

其中 `local_id` 可以是一条 spring、一个 vertex term、一个 triangle/tet element、一个 contact stencil，或者未来 FEM mass/inertia contribution；不要假设 local term 一定是 edge 或 vertex。

- Phase 3 的 energy 分层固定为：

```text
LocalEnergyModel -> LocalEnergyProvider -> CPUAssembler -> FullEnergy quantities
```

`LocalEnergyModel` 是 stateless static formula / material-law policy：它接收 local data 和 local displacement，不知道 full DOF、mesh topology、`local_id`、sparse matrix、boundary、solver 或 GPU backend。
- `LocalEnergyProvider` 保持 **full-space local contribution provider** 语义：它接收 full displacement vector `u`，通过自己的 topology/rest/material data gather 需要的 local state，并返回 local value/gradient/Hessian。`local_dofs(local_id, dofs)` 返回的是 full DOF indices。
- `LocalEnergyProvider` 只回答三件事：local term 数量、某个 local term 作用在哪些 full DOF 上、这个 local term 对 full-space energy 的 value/gradient/Hessian contribution 是什么。
- `CPUAssembler` 是 adapter：`LocalEnergyProvider + CPUAssembler = FullEnergy quantities`。solver 和 `ReducedEnergyView` 面向的是 full-space energy，不直接依赖 local provider 或 local model。
- finite difference helper 必须放在 Phase 3 最前面。`MassSpringLocalEnergyModel` 是第一个 nonlinear local energy model，解析 gradient/Hessian 容易写错；先建立 derivative oracle，可以在进入 Newton solver 前单独验证 local derivative 和 global assembly。
- `local_hessian()` 默认只表示 **true analytic Hessian**。PSD projection、Gauss-Newton、diagonal regularization、modified Cholesky 都属于 Phase 4 solver/optimization strategy，不属于 energy definition。不要在 energy 层偷偷把 Hessian 改成 PSD，否则 finite difference Hessian test 会失去语义。
- rest length 为零或过小时应在 `MassSpringLocalEnergyProvider` 构造或 `MassSpringLocalEnergyModel` evaluate 时直接 reject；current length 过小时只做数值防护以避免 NaN，derivative tests 不覆盖不可导的 singular current edge 构型。

**实现顺序:** 先 `finite_difference.hpp`，再 `energy_concepts.hpp` / `local_matrix.hpp`，再 `cpu_assembler.hpp`，最后 `mass_spring_local_energy_provider.hpp`。这条顺序的目标是：不要在没有导数验算工具的情况下写 nonlinear Hessian。

### Task 3.1: 添加 finite difference derivative oracle

**文件:**
- 创建: `include/pgo/math/finite_difference.hpp`
- 创建: `tests/math/test_finite_difference.cpp`
- 修改: `tests/CMakeLists.txt`

- [x] **Step 1: 实现 `finite_difference.hpp`**

提供 central difference helper，只用于测试和诊断，不进入 runtime solver：

```cpp
namespace pgo::math {

template <typename T, typename F>
[[nodiscard]] DVec<T> finite_difference_gradient(
    F&& value_function,
    const DVec<T>& x,
    T eps);

template <typename T, typename Grad>
[[nodiscard]] DMat<T> finite_difference_hessian_from_gradient(
    Grad&& gradient_function,
    const DVec<T>& x,
    T eps);

} // namespace pgo::math
```

语义要求：

- `finite_difference_gradient(f, x, eps)` 使用：

```text
g_i = (f(x + eps e_i) - f(x - eps e_i)) / (2 eps)
```

- `finite_difference_hessian_from_gradient(grad, x, eps)` 使用：

```text
H_col_i = (grad(x + eps e_i) - grad(x - eps e_i)) / (2 eps)
```

- helper 内部可以复制 `x` 生成 `x_plus` / `x_minus`，但不能修改 caller 传入的 `x`。
- `eps <= 0` 时抛出 `std::runtime_error`。
- `gradient_function` 返回 `DVec<T>` 或写入 output buffer 这两种形式二选一即可；Phase 3 推荐先实现返回 `DVec<T>` 的版本，保持测试代码简单。

- [x] **Step 2: 添加 finite difference 自测**

`tests/math/test_finite_difference.cpp` 使用 `namespace pgo::math::test`。用二次函数验证 FD helper 本身：

```text
f(x, y) = x^2 + 3xy + 2y^2
grad = [2x + 3y, 3x + 4y]
H = [[2, 3], [3, 4]]
```

测试内容：

- `finite_difference_gradient` 在 `x = [0.7, -1.2]` 附近与解析 gradient 匹配。
- `finite_difference_hessian_from_gradient` 在同一点与解析 Hessian 匹配。
- `eps <= 0` 抛出异常。

- [x] **Step 3: 把测试加入 `tests/CMakeLists.txt`**

```cmake
add_executable(pgo_tests
    base/test_assert.cpp
    math/test_backend.cpp
    math/test_eigen_config.cpp
    math/test_finite_difference.cpp
    geometry/test_rest_mesh.cpp
    dof/test_dof.cpp
    io/test_obj_io.cpp
)
```

- [x] **Step 4: 运行 finite difference 测试**

```bash
cmake --build --preset debug
ctest --preset debug -R finite
```

期望：`finite_difference` 相关测试全部通过。

### Task 3.2: 添加 energy concepts 和 local matrix types

**文件:**
- 创建: `include/pgo/assembly/local_matrix.hpp`
- 创建: `include/pgo/energy/energy_concepts.hpp`
- 创建: `tests/energy/test_energy_concepts.cpp`
- 修改: `tests/CMakeLists.txt`

- [x] **Step 1: 定义 local aliases**

`include/pgo/assembly/local_matrix.hpp` 提供：

```cpp
namespace pgo::assembly {

template <typename T>
using LocalVector = pgo::math::DVec<T>;

template <typename T>
using LocalMatrix = pgo::math::DMat<T>;

} // namespace pgo::assembly
```

设计约束：

- Phase 3 的 local vector/matrix 可以使用动态 Eigen 类型。每条 spring 的 local size 是 `2 * Dim`，后续 GPU 不复用 Eigen local matrix，只复用 local contribution API 语义。
- local aliases 属于 `assembly/`，因为它们描述 local-to-global assembly 的中间数据，不是 geometry storage。

- [x] **Step 2: 定义 `FullEnergy` concept**

`include/pgo/energy/energy_concepts.hpp` 提供 solver-facing full-space energy concept：

```cpp
namespace pgo::energy {

template <typename Energy, typename T>
concept FullEnergy = typename T && requires(
    const Energy& energy,
    const pgo::math::DVec<T>& u,
    T& value,
    pgo::math::DVec<T>& gradient,
    pgo::math::SparseMat<T>& hessian
) {
    { energy.value(u) } -> std::same_as<T>;
    energy.gradient(u, gradient);
    energy.hessian(u, hessian);
    energy.value_gradient_hessian(u, value, gradient, hessian);
};

} // namespace pgo::energy
```

设计约束：

- `FullEnergy` 工作在 full displacement space，不知道 reduced variables。
- `value_gradient_hessian` 是 solver 主路径：Newton 每轮通常同时需要三者。单独的 `value` / `gradient` / `hessian` 用于测试、调试和 line search。

- [x] **Step 3: 定义 `LocalEnergyModel`、`FusedLocalEnergyModel`、`LocalEnergyProvider` 和 `FusedLocalEnergyProvider` concepts**

`LocalEnergyModel` 是单个局部项公式 / material law 层，最小接口：

```cpp
namespace pgo::energy {

template <typename Model, typename T, typename LocalData>
concept LocalEnergyModel = typename T && requires(
    const LocalData& local_data,
    const pgo::assembly::LocalVector<T>& local_u,
    pgo::assembly::LocalVector<T>& local_g,
    pgo::assembly::LocalMatrix<T>& local_H
) {
    { Model::local_dof_count(local_data) } -> std::convertible_to<std::size_t>;
    { Model::value(local_data, local_u) } -> std::same_as<T>;
    Model::gradient(local_data, local_u, local_g);
    Model::hessian(local_data, local_u, local_H);
};

template <typename Model, typename T, typename LocalData>
concept FusedLocalEnergyModel = LocalEnergyModel<Model, T, LocalData> && requires(
    const LocalData& local_data,
    const pgo::assembly::LocalVector<T>& local_u,
    T& value,
    pgo::assembly::LocalVector<T>& local_g,
    pgo::assembly::LocalMatrix<T>& local_H
) {
    Model::value_gradient_hessian(local_data, local_u, value, local_g, local_H);
};

} // namespace pgo::energy
```

`LocalEnergyProvider` 是 assembler-facing indexed collection / gather 层，最小接口：

```cpp
namespace pgo::energy {

template <typename EnergyProvider, typename T>
concept LocalEnergyProvider = typename T && requires(
    const EnergyProvider& energy_provider,
    std::size_t local_id,
    const pgo::math::DVec<T>& full_u,
    std::vector<std::size_t>& dofs,
    pgo::assembly::LocalVector<T>& local_g,
    pgo::assembly::LocalMatrix<T>& local_H
) {
    { energy_provider.local_count() } -> std::convertible_to<std::size_t>;
    energy_provider.local_dofs(local_id, dofs);
    { energy_provider.local_value(local_id, full_u) } -> std::same_as<T>;
    energy_provider.local_gradient(local_id, full_u, local_g);
    energy_provider.local_hessian(local_id, full_u, local_H);
};

template <typename EnergyProvider, typename T>
concept FusedLocalEnergyProvider = LocalEnergyProvider<EnergyProvider, T> && requires(
    const EnergyProvider& energy_provider,
    std::size_t local_id,
    const pgo::math::DVec<T>& full_u,
    T& value,
    pgo::assembly::LocalVector<T>& local_g,
    pgo::assembly::LocalMatrix<T>& local_H
) {
    energy_provider.local_value_gradient_hessian(local_id, full_u, value, local_g, local_H);
};

} // namespace pgo::energy
```

语义要求：

- `LocalEnergyModel` 是 stateless static API，不知道 full displacement vector、`local_id`、full DOF indices、mesh topology 或 sparse assembly。
- `LocalEnergyProvider` 负责从 full `u` gather local displacement，并从 mesh/rest/material arrays gather model 需要的 `LocalData`。
- `local_dofs(local_id, dofs)` 必须写入 full DOF indices，例如 2D 三点两弹簧里 spring 0 返回 `[0, 1, 2, 3]`，spring 1 返回 `[2, 3, 4, 5]`。
- `local_gradient(local_id, full_u, local_g)` 返回 `dE_i / du_local`，并满足 `local_g.size() == dofs.size()`。
- `local_hessian(local_id, full_u, local_H)` 返回 true local Hessian，并满足 `local_H.rows() == dofs.size()`、`local_H.cols() == dofs.size()`。
- `local_value_gradient_hessian` 是 provider 的可选 performance API；assembler 优先使用 fused provider API，避免重复计算 current positions、edge vector、edge length、direction 等中间量。
- 不要在 model/provider concepts 中出现 `assemble`、`global_gradient`、`global_hessian`、`reduced_gradient`、`boundary`、`dof_map` 等职责。

- [x] **Step 4: 添加 concept smoke tests**

`tests/energy/test_energy_concepts.cpp` 使用 `namespace pgo::energy::test`。定义一个最小 toy local energy：

```cpp
class ToyEdgeEnergy {
public:
    [[nodiscard]] std::size_t local_count() const { return 1; }

    void local_dofs(std::size_t, std::vector<std::size_t>& dofs) const {
        dofs = {0, 1};
    }

    [[nodiscard]] double local_value(std::size_t, const pgo::math::DVec<double>& u) const {
        return 0.5 * (u[0] - u[1]) * (u[0] - u[1]);
    }

    void local_gradient(std::size_t, const pgo::math::DVec<double>& u, pgo::assembly::LocalVector<double>& g) const {
        g.resize(2);
        g[0] = u[0] - u[1];
        g[1] = u[1] - u[0];
    }

    void local_hessian(std::size_t, const pgo::math::DVec<double>&, pgo::assembly::LocalMatrix<double>& H) const {
        H.resize(2, 2);
        H << 1.0, -1.0,
            -1.0, 1.0;
    }
};
```

测试内容：

- `static_assert(pgo::energy::LocalEnergyProvider<ToyEdgeEnergy, double>)`。
- `static_assert(!pgo::energy::FusedLocalEnergyProvider<ToyEdgeEnergy, double>)`。
- local gradient/Hessian size 与 `local_dofs` size 一致。

- [x] **Step 5: 运行 energy concept 测试**

```bash
cmake --build --preset debug
ctest --preset debug -R EnergyConcept
```

### Task 3.3: 添加 CPU local energy assembler

**文件:**
- 创建: `include/pgo/assembly/cpu_assembler.hpp`
- 创建: `tests/assembly/test_cpu_assembler.cpp`
- 修改: `tests/CMakeLists.txt`

- [x] **Step 1: 实现 stateless assembly functions**

`include/pgo/assembly/cpu_assembler.hpp` 提供：

```cpp
namespace pgo::assembly {

template <typename T, typename EnergyProvider>
    requires pgo::energy::LocalEnergyProvider<EnergyProvider, T>
[[nodiscard]] T assemble_value(const EnergyProvider& energy_provider, const pgo::math::DVec<T>& full_u);

template <typename T, typename EnergyProvider>
    requires pgo::energy::LocalEnergyProvider<EnergyProvider, T>
void assemble_gradient(
    const EnergyProvider& energy_provider,
    const pgo::math::DVec<T>& full_u,
    pgo::math::DVec<T>& full_gradient);

template <typename T, typename EnergyProvider>
    requires pgo::energy::LocalEnergyProvider<EnergyProvider, T>
void assemble_hessian(
    const EnergyProvider& energy_provider,
    const pgo::math::DVec<T>& full_u,
    pgo::math::SparseMat<T>& full_hessian);

template <typename T, typename EnergyProvider>
    requires pgo::energy::LocalEnergyProvider<EnergyProvider, T>
void assemble_value_gradient_hessian(
    const EnergyProvider& energy_provider,
    const pgo::math::DVec<T>& full_u,
    T& value,
    pgo::math::DVec<T>& full_gradient,
    pgo::math::SparseMat<T>& full_hessian);

} // namespace pgo::assembly
```

如果 C++ concept 写法中 requires 子句不方便表达，允许改成普通模板加 `static_assert(pgo::energy::LocalEnergyProvider<EnergyProvider, T>)`，但外部 API 名字保持不变。

实现要求：

- assembler 是 stateless free-function utility，不拥有 mesh、不拥有 boundary、不缓存 solver state。
- `assemble_value` 遍历 `[0, energy.local_count())` 并累加 `local_value`。
- `assemble_gradient` 先把 `full_gradient` resize 到 `full_u.size()` 并置零，再通过 `local_dofs` scatter：

```cpp
full_gradient[dofs[a]] += local_g[a];
```

- `assemble_hessian` 使用 `std::vector<pgo::math::Triplet<T>>` scatter local Hessian：

```cpp
triplets.emplace_back(dofs[a], dofs[b], local_H(a, b));
```

然后 `SparseMat<T>(full_u.size(), full_u.size())` + `setFromTriplets`。重复 triplets 必须由 Eigen 累加。
- `assemble_value_gradient_hessian` 优先调用 `FusedLocalEnergyProvider` 的 `local_value_gradient_hessian`；如果 provider 没有 fused API，则 fallback 到 `local_value`、`local_gradient`、`local_hessian` 三个接口。
- 每个 local term assembly 前后都要验证尺寸：

```text
dofs.size() == local_g.size()
local_H.rows() == dofs.size()
local_H.cols() == dofs.size()
```

不满足时抛出 `std::runtime_error`。
- 对每个 `dofs[a]` 验证 `dofs[a] < full_u.size()`，防止坏 local map 进入 sparse assembly。

- [x] **Step 2: 添加 toy assembly 测试**

`tests/assembly/test_cpu_assembler.cpp` 使用 `namespace pgo::assembly::test`。用 Task 3.2 的 `ToyEdgeEnergy` 或等价测试类型验证：

- `assemble_value` 对 `u = [2, -1]` 返回 `0.5 * 9 = 4.5`。
- `assemble_gradient` 返回 `[3, -3]`。
- `assemble_hessian` 返回：

```text
[[1, -1],
 [-1, 1]]
```

- `assemble_value_gradient_hessian` 在 non-fused energy 上能 fallback，结果与单独 assembly 一致。

- [x] **Step 3: 添加 shared DOF scatter 测试**

构造一个 2D 三点两弹簧形状的 toy local energy，`local_dofs(0) = [0,1,2,3]`，`local_dofs(1) = [2,3,4,5]`。每个 local term 给固定 local gradient `ones(4)`。

验证 assembled full gradient：

```text
[1, 1, 2, 2, 1, 1]
```

这个测试确保共享 vertex 的 DOF contribution 会累加，而不是覆盖。

- [x] **Step 4: 添加 bad local energy 防御测试**

测试以下错误会抛出：

- `local_gradient` size 与 `local_dofs` size 不一致。
- `local_hessian` shape 与 `local_dofs` size 不一致。
- `local_dofs` 返回超出 `full_u.size()` 的 full DOF index。

- [x] **Step 5: 运行 assembler 测试**

```bash
cmake --build --preset debug
ctest --preset debug -R CPUAssembler
```

### Task 3.4: 添加 MassSpringLocalEnergyModel 和 MassSpringLocalEnergyProvider

**文件:**
- 创建: `include/pgo/energy/mass_spring_local_energy_provider.hpp`
- 创建: `tests/energy/test_mass_spring_local_energy_provider.cpp`
- 修改: `tests/CMakeLists.txt`

- [x] **Step 1: 实现 `MassSpringLocalData<T, Dim>` 和 `MassSpringLocalEnergyModel<T, Dim>`**

单条 spring 的 local energy model：

```text
E_e(u) = 0.5 * k * (||x_i - x_j|| - L)^2
x_i = X_i + u_i
x_j = X_j + u_j
L = ||X_i - X_j||
```

推荐 API：

```cpp
namespace pgo::energy {

template <typename T, int Dim>
struct MassSpringLocalData {
    pgo::math::Vec<T, Dim> rest_i;
    pgo::math::Vec<T, Dim> rest_j;
    T stiffness;
};

template <typename T, int Dim>
class MassSpringLocalEnergyModel {
public:
    [[nodiscard]] static std::size_t local_dof_count(const MassSpringLocalData<T, Dim>& data);
    [[nodiscard]] static T value(const MassSpringLocalData<T, Dim>& data,
                                 const pgo::assembly::LocalVector<T>& local_u);
    static void gradient(const MassSpringLocalData<T, Dim>& data, const pgo::assembly::LocalVector<T>& local_u,
                         pgo::assembly::LocalVector<T>& local_g);
    static void hessian(const MassSpringLocalData<T, Dim>& data, const pgo::assembly::LocalVector<T>& local_u,
                        pgo::assembly::LocalMatrix<T>& local_H);
    static void value_gradient_hessian(const MassSpringLocalData<T, Dim>& data,
                                       const pgo::assembly::LocalVector<T>& local_u, T& value,
                                       pgo::assembly::LocalVector<T>& local_g,
                                       pgo::assembly::LocalMatrix<T>& local_H);
};
```

实现要求：

- `local_u` 的 order 固定为 `[u_i components..., u_j components...]`。
- `local_dof_count(data) == 2 * Dim`。
- `data.stiffness < 0`、rest length `<= kMinLength` 时抛出 `std::runtime_error`。
- `local_hessian` 返回 true analytic Hessian，不做 PSD projection。
- `value` 和 `gradient` 拥有独立的实现，以避免不必要的开销，而不是直接 fallback 到 `value_gradient_hessian`。
- 退化情况 (degenerate edge) 使用内部的 `static constexpr T kMinLength` 进行数值防护。

- [x] **Step 2: 实现 `MassSpringLocalEnergyProvider<T, Dim>`**

Provider 是 mesh/full-u/material gather 层：

```cpp
template <typename T, int Dim>
class MassSpringLocalEnergyProvider {
public:
    MassSpringLocalEnergyProvider(const pgo::geometry::RestMesh<T, Dim>& mesh, T uniform_stiffness);
    MassSpringLocalEnergyProvider(const pgo::geometry::RestMesh<T, Dim>& mesh,
                                  pgo::storage::ConstArrayView<T> stiffnesses);

    [[nodiscard]] std::size_t local_count() const;
    [[nodiscard]] std::size_t max_local_dofs() const;

    void local_dofs(std::size_t edge_id, std::vector<std::size_t>& dofs) const;

    [[nodiscard]] T local_value(std::size_t edge_id, const pgo::math::DVec<T>& full_u) const;
    void local_gradient(std::size_t edge_id, const pgo::math::DVec<T>& full_u, pgo::assembly::LocalVector<T>& local_g) const;
    void local_hessian(std::size_t edge_id, const pgo::math::DVec<T>& full_u, pgo::assembly::LocalMatrix<T>& local_H) const;

    void local_value_gradient_hessian(
        std::size_t edge_id,
        const pgo::math::DVec<T>& full_u,
        T& value,
        pgo::assembly::LocalVector<T>& local_g,
        pgo::assembly::LocalMatrix<T>& local_H) const;
};
} // namespace pgo::energy
```

Provider 实现要求：

- `local_count() == mesh.num_edges()`。
- `max_local_dofs() == 2 * Dim`。
- uniform stiffness constructor 将同一个 stiffness 复制到所有 edges。
- per-edge stiffness constructor 将 input view 复制到 provider 内部，要求数量等于 `mesh.num_edges()`。
- 任意 stiffness `< 0` 时抛出 `std::runtime_error`。
- `local_dofs(edge_id, dofs)` 通过 `mesh.edge_vertex(edge_id, 0/1)` 取端点，并返回：

```text
[i0, i1, ..., i(Dim-1), j0, j1, ..., j(Dim-1)]
```

其中 full DOF index 使用 `vertex * Dim + component`，和 `DofLayout<Dim>` 一致。
- 构造时遍历所有 edges，计算 rest length `L`。如果 `L <= kMinLength`，抛出 `std::runtime_error`，不要让零长度 rest spring 进入 solver。
- provider 的 `local_value`、`local_gradient`、`local_hessian` 使用 full displacement `u` gather local displacement，然后调用 `MassSpringLocalEnergyModel` static API。

- [x] **Step 3: 明确 mass-spring 解析导数公式**

对单条 edge，令：

```text
d = x_i - x_j
r = ||d||
L = ||X_i - X_j||
n = d / r
```

当 `r > kMinLength` 时，对 `d` 的导数为：

```text
grad_d = k * (r - L) * n
H_d = k * (n n^T + (1 - L / r) * (I - n n^T))
```

映射到 local DOF order `[u_i, u_j]`：

```text
local_g = [ grad_d, -grad_d ]

local_H = [  H_d, -H_d
            -H_d,  H_d ]
```

当 `r <= kMinLength` 时，能量仍按 clamped `r_safe = kMinLength` 做数值防护，gradient/Hessian 使用稳定 fallback，保证不产生 NaN。这个分支只用于避免 solver 崩溃；finite difference derivative tests 应选择远离 `r = 0` 的构型。

- [x] **Step 4: 添加 local model/provider smoke tests**

`tests/energy/test_mass_spring_local_energy_provider.cpp` 使用 `namespace pgo::energy::test`。测试：

- `static_assert(pgo::energy::LocalEnergyModel<MassSpringLocalEnergyModel<double, 2>, double, MassSpringLocalData<double, 2>>)`。
- `static_assert(pgo::energy::FusedLocalEnergyModel<MassSpringLocalEnergyModel<double, 2>, double, MassSpringLocalData<double, 2>>)`。
- `static_assert(pgo::energy::LocalEnergyProvider<MassSpringLocalEnergyProvider<double, 2>, double>)`。
- `static_assert(pgo::energy::FusedLocalEnergyProvider<MassSpringLocalEnergyProvider<double, 2>, double>)`。
- `local_count() == mesh.num_edges()`。
- `max_local_dofs() == 2 * Dim`。
- `local_dofs(0, dofs)` 返回 `[0, 1, 2, 3]`（2D）或 `[0, 1, 2, 3, 4, 5]`（3D）。
- rest state 下 `local_value == 0`，`local_gradient` norm 为 0。
- 拉伸一个端点后 `local_value > 0`。
- 构造包含零长度 rest edge 的 mesh 时抛出异常。
- per-edge stiffness 数量不匹配时抛出异常。
- per-edge stiffness 包含负值时抛出异常。
- 两条 spring 使用不同 stiffness 时，各自 `local_value(edge_id, full_u)` 按对应 stiffness 缩放。

- [x] **Step 5: 添加 finite-difference local derivative tests**

用单条 2D spring，选择远离 singularity 的 displacement，例如：

```text
X0 = (0, 0)
X1 = (1, 0)
u0 = (0.1, 0.2)
u1 = (0.35, -0.15)
```

测试：

- `MassSpringLocalEnergyModel::gradient(data, local_u, analytic_g)` 与 `finite_difference_gradient(local_value_lambda, local_u, eps)` 匹配。
- `MassSpringLocalEnergyModel::hessian(data, local_u, analytic_H)` 与 `finite_difference_hessian_from_gradient(local_gradient_lambda, local_u, eps)` 匹配。
- `MassSpringLocalEnergyProvider::local_gradient(0, full_u, analytic_g)` 与同一 local order 的 finite difference 匹配。
- `MassSpringLocalEnergyProvider::local_hessian(0, full_u, analytic_H)` 与同一 local order 的 finite difference 匹配。
- 至少覆盖一个压缩但非 singular 的状态，确认 true Hessian 允许 indefinite，不要求 PSD。

测试里的 `local_u` 可以通过 helper scatter 到 full `u`，但比较对象必须是 local DOF order 对应的 local gradient/Hessian。

- [x] **Step 6: 添加 global assembly derivative tests**

创建三点两弹簧 2D mesh，通过 `CPUAssembler` 组装 full energy/gradient/Hessian：

- 用 `assemble_value` 和 `finite_difference_gradient` 验证 assembled full gradient。
- 用 `assemble_gradient` 和 `finite_difference_hessian_from_gradient` 验证 assembled sparse Hessian。
- 验证共享 vertex 的 gradient contribution 会累加。

这个测试用于区分：

```text
local derivative bug
local_dofs ordering bug
local -> global scatter bug
triplet accumulation bug
```

- [x] **Step 7: 运行 mass-spring 和 assembler 相关测试**

```bash
cmake --build --preset debug
ctest --preset debug -R "MassSpring|CPUAssembler|finite"
```

期望：finite difference、CPU assembler、MassSpringLocalEnergyModel / MassSpringLocalEnergyProvider 测试全部通过。

## Phase 4: Reduced Energy、Newton Solver 和 Minimal Backward Euler

**当前状态:** 未开始，是下一步主线。Phase 4 基于 Phase 3 已验证的 local contribution / CPU assembly，把系统推进到可求解的 reduced optimization problem，并补上 Milestone 1 最小动态 step。

**Phase 4 目标:** 跑通以下 pipeline：

```text
LocalEnergyProvider
  -> AssembledEnergy
  -> EnergySum
  -> ReducedEnergyView
  -> NewtonSolver + LineSearch + AlwaysFeasible
  -> minimal BackwardEuler step
```

**设计决策:**

- `CPUAssembler` 继续保持 stateless free functions；它只计算 full-space quantities，不直接满足 `FullEnergy`。
- `AssembledEnergy<T, Provider>` 是 local -> full 的薄 adapter：`LocalEnergyProvider + CPUAssembler -> FullEnergy`。
- `EnergySum<T, Energies...>` 是 full -> full composer：多个 `FullEnergy` 相加后仍然是 `FullEnergy`。它不认识 `LocalEnergyProvider`，也不 include `cpu_assembler.hpp`。
- 新增 `DifferentiableEnergy` concept 作为 solver-facing 最小接口。`FullEnergy` 是 full-space `DifferentiableEnergy`；`ReducedEnergyView` 是 reduced-space `DifferentiableEnergy`，不是 `FullEnergy`。
- `ReducedEnergyView` 是变量代换层，不是线性系统消元层。它使用 `scatter_solution`、`restrict_vector_to_free`、`restrict_matrix_to_free`，绝不使用 `eliminate_rhs_for_dirichlet`。
- Energy / assembly / reduced energy 全部保持 true analytic Hessian。`H + lambda I`、descent direction check、Armijo line search 属于 solver 层。
- Feasible line search 的位置在 solver 层，但 Milestone 1 只实现 `AlwaysFeasible`。IPC/contact/CCD 的真实 feasibility 不进入 Milestone 1。
- 引入 `integrator/` 模块，但不引入 `IntegratorBase`、runtime polymorphism、Newmark、TF-BDF2 或 adaptive timestep。Milestone 1 只实现 concrete `BackwardEuler<T>::step(...)`。
- Backward Euler 通过普通 full-space inertia energy 接入：`step_energy = EnergySum(potential_energy, inertial_energy)`。

### Task 4.1: 补齐 energy concepts 和 AssembledEnergy

**文件:**
- 修改: `include/pgo/energy/energy_concepts.hpp`
- 创建: `include/pgo/energy/assembled_energy.hpp`
- 修改: `tests/energy/test_energy_concepts.cpp`
- 创建: `tests/energy/test_assembled_energy.cpp`
- 修改: `tests/CMakeLists.txt`

- [x] **Step 1: 添加 `DifferentiableEnergy` concept**

`DifferentiableEnergy` 是 solver-facing contract，不承诺变量是 full displacement 还是 reduced free variables：

```cpp
namespace pgo::energy {

template <typename Energy, typename T>
concept DifferentiableEnergy =
    requires(const Energy& energy, const pgo::math::DVec<T>& z, T& value,
             pgo::math::DVec<T>& gradient, pgo::math::SparseMat<T>& hessian) {
        { energy.value(z) } -> std::same_as<T>;
        energy.gradient(z, gradient);
        energy.hessian(z, hessian);
        energy.value_gradient_hessian(z, value, gradient, hessian);
    };

} // namespace pgo::energy
```

- [x] **Step 2: 让 `FullEnergy` 基于 `DifferentiableEnergy`**

保留 `FullEnergy` 名字，但把语义明确为 full displacement space 上的 energy contract：

```cpp
template <typename Energy, typename T>
concept FullEnergy = DifferentiableEnergy<Energy, T>;
```

设计约束：

- `FullEnergy` 用于 `AssembledEnergy`、`EnergySum`、`ReducedEnergyView` 的输入边界。
- `NewtonSolver` 不使用 `FullEnergy` 约束；它只要求 `DifferentiableEnergy`。
- C++ concept 无法检查 vector 维度，`FullEnergy` 和 `DifferentiableEnergy` 的区别主要是架构语义：full-space vs arbitrary optimization variable。

- [x] **Step 3: 实现 `AssembledEnergy<T, Provider>`**

`include/pgo/energy/assembled_energy.hpp` 提供：

```cpp
namespace pgo::energy {

template <typename T, typename Provider>
    requires LocalEnergyProvider<Provider, T>
class AssembledEnergy {
public:
    explicit AssembledEnergy(const Provider& provider);

    [[nodiscard]] T value(const pgo::math::DVec<T>& full_u) const;
    void gradient(const pgo::math::DVec<T>& full_u, pgo::math::DVec<T>& full_g) const;
    void hessian(const pgo::math::DVec<T>& full_u, pgo::math::SparseMat<T>& full_H) const;
    void value_gradient_hessian(const pgo::math::DVec<T>& full_u, T& value,
                                pgo::math::DVec<T>& full_g,
                                pgo::math::SparseMat<T>& full_H) const;
};

} // namespace pgo::energy
```

实现要求：

- non-owning 保存 `Provider` 指针或引用；不复制 mesh/provider/material arrays。
- `value` 调用 `assembly::assemble_value<T>`。
- `gradient` 调用 `assembly::assemble_gradient<T>`。
- `hessian` 调用 `assembly::assemble_hessian<T>`。
- `value_gradient_hessian` 调用 `assembly::assemble_value_gradient_hessian<T>`。
- `AssembledEnergy` 是 `FullEnergy`。`CPUAssembler` 本身不变成 class、不拥有 state、不知道 solver/reduced boundary。

- [x] **Step 4: 添加 concept 和 adapter 测试**

测试内容：

- `static_assert(pgo::energy::DifferentiableEnergy<QuadraticEnergy, double>)`。
- `static_assert(pgo::energy::FullEnergy<QuadraticEnergy, double>)`。
- 使用 `tests/assembly/test_cpu_assembler.cpp` 中等价的 toy provider，验证 `AssembledEnergy` 满足 `FullEnergy`。
- 验证 `AssembledEnergy::value/gradient/hessian/value_gradient_hessian` 与直接调用 `assembly::assemble_*` 的结果一致。

- [x] **Step 5: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R "EnergyConcept|AssembledEnergy|CPUAssembler"
```

验证结果：`cmake --build --preset debug` 成功；`ctest --preset debug -R "EnergyConcept|AssembledEnergy|CPUAssembler" --output-on-failure` 运行 5 个测试，全部通过；`ctest --preset debug --output-on-failure` 运行 42 个测试，全部通过。

### Task 4.2: 添加 EnergySum 和 ReducedEnergyView

**文件:**
- 创建: `include/pgo/energy/energy_sum.hpp`
- 创建: `include/pgo/energy/reduced_energy.hpp`
- 创建: `tests/energy/test_energy_sum.cpp`
- 创建: `tests/energy/test_reduced_energy.cpp`
- 修改: `tests/CMakeLists.txt`

- [ ] **Step 1: 实现 tuple-based `EnergySum<T, Energies...>`**

组合多个 full-space energies：

- value 相加。
- gradient 相加。
- Hessian 相加。

推荐 API：

```cpp
namespace pgo::energy {

template <typename T, typename... Energies>
    requires(FullEnergy<Energies, T> && ...)
class EnergySum {
public:
    explicit EnergySum(const Energies&... energies);

    [[nodiscard]] T value(const pgo::math::DVec<T>& full_u) const;
    void gradient(const pgo::math::DVec<T>& full_u, pgo::math::DVec<T>& full_g) const;
    void hessian(const pgo::math::DVec<T>& full_u, pgo::math::SparseMat<T>& full_H) const;
    void value_gradient_hessian(const pgo::math::DVec<T>& full_u, T& value,
                                pgo::math::DVec<T>& full_g,
                                pgo::math::SparseMat<T>& full_H) const;
};

} // namespace pgo::energy
```

实现要求：

- non-owning 保存 input energies 的指针/reference tuple。
- `EnergySum` 的输入必须已经是 `FullEnergy`；不要 special-case `AssembledEnergy` 或 `LocalEnergyProvider`。
- gradient 先 resize 到 input gradient size 并置零，再累加每个 energy 的 gradient。
- Hessian 可以先让每个 energy 输出 sparse matrix，然后相加。Milestone 1 接受 sparse temporary 开销。
- `value_gradient_hessian` 优先每个子 energy 调用 fused API，累加 value / gradient / Hessian。
- 空 `EnergySum` 暂不支持；如果 `Energies...` 为空应在 compile-time 或 constructor 中拒绝。

- [ ] **Step 2: 添加 `EnergySum` 测试**

使用两个 quadratic full energies：

```text
E_i(u) = 0.5 * u^T A_i u - b_i^T u + c_i
```

验证：

- `EnergySum::value == value_0 + value_1`。
- `EnergySum::gradient == gradient_0 + gradient_1`。
- `EnergySum::hessian == hessian_0 + hessian_1`。
- `EnergySum::value_gradient_hessian` 与单独接口一致。
- `static_assert(pgo::energy::FullEnergy<EnergySum<...>, double>)`。

- [ ] **Step 3: 实现 `ReducedEnergyView<T, Energy>`**

职责：

```text
free_u -> scatter_solution 成 full_u
full energy evaluate
full gradient 通过 restrict_vector_to_free 变成 reduced gradient
full Hessian 通过 restrict_matrix_to_free 变成 reduced Hessian
```

实现语义：

```cpp
const auto full_u = dof_map.scatter_solution(free_u);

full_energy.value_gradient_hessian(
    full_u,
    value,
    full_gradient,
    full_hessian);

reduced_gradient = dof_map.restrict_vector_to_free(full_gradient);
reduced_hessian = dof_map.restrict_matrix_to_free(full_hessian);
```

数学语义：

```text
E_red(y) = E_full(P y + G g)
grad_red = P^T grad_full(P y + G g)
H_red = P^T H_full(P y + G g) P
```

重要约束：`eliminate_rhs_for_dirichlet(full_rhs, full_matrix)` 只用于直接线性系统 `A u = b` 的 RHS 消元，不用于 reduced energy gradient。energy gradient 已经在 `full_u = P y + G g` 上 evaluate 过，prescribed displacement 的影响已经包含在 `full_gradient` 中。

推荐 API：

```cpp
namespace pgo::energy {

template <typename T, typename Energy>
    requires FullEnergy<Energy, T>
class ReducedEnergyView {
public:
    ReducedEnergyView(const Energy& full_energy,
                      const pgo::dof::ReducedDofMap<T>& dof_map);

    [[nodiscard]] std::size_t full_dofs() const;
    [[nodiscard]] std::size_t free_dofs() const;

    [[nodiscard]] T value(const pgo::math::DVec<T>& free_u) const;
    void gradient(const pgo::math::DVec<T>& free_u, pgo::math::DVec<T>& reduced_g) const;
    void hessian(const pgo::math::DVec<T>& free_u, pgo::math::SparseMat<T>& reduced_H) const;
    void value_gradient_hessian(const pgo::math::DVec<T>& free_u, T& value,
                                pgo::math::DVec<T>& reduced_g,
                                pgo::math::SparseMat<T>& reduced_H) const;

    [[nodiscard]] pgo::math::DVec<T> scatter_solution(const pgo::math::DVec<T>& free_u) const;
    [[nodiscard]] pgo::math::DVec<T> scatter_direction(const pgo::math::DVec<T>& free_du) const;
};

} // namespace pgo::energy
```

实现要求：

- non-owning 保存 full energy 和 `ReducedDofMap`。
- 不缓存 `full_u`、`full_gradient`、`full_hessian`；Milestone 1 优先简单和线程语义清晰。
- `value(free_u)` 必须先 `scatter_solution(free_u)`，不能要求调用者传 full `u`。
- `gradient/hessian/value_gradient_hessian` 只做 `P^T` / `P^T H P` projection。
- `scatter_solution` / `scatter_direction` 是 convenience API，直接转发 dof map，方便 solver debug 和 example 输出。
- `ReducedEnergyView` 满足 `DifferentiableEnergy`，但不要把它当作 `FullEnergy`。

- [ ] **Step 4: 添加 reduced energy 测试**

使用 quadratic energy：

```text
E(u) = 0.5 * u^T A u - b^T u
```

固定一个 DOF，验证 reduced gradient/Hessian 分别等于：

```text
restrict_vector_to_free(full_gradient)
restrict_matrix_to_free(full_hessian)
```

测试必须覆盖非零 prescribed displacement，用于防止错误地把 `eliminate_rhs_for_dirichlet(full_gradient, full_hessian)` 当成 reduced gradient projection。

额外测试：

- `ReducedEnergyView::value(free_u) == full_energy.value(dof_map.scatter_solution(free_u))`。
- `ReducedEnergyView::scatter_solution(free_u)` 与 dof map 结果一致。
- `ReducedEnergyView::scatter_direction(free_du)` 的 fixed DOF 为 0。
- `static_assert(pgo::energy::DifferentiableEnergy<ReducedEnergyView<...>, double>)`。

- [ ] **Step 5: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R "EnergySum|ReducedEnergy|solver"
```

### Task 4.3: 添加 solver result、options、feasible set 和 Armijo line search

**文件:**
- 创建: `include/pgo/solver/solver_result.hpp`
- 创建: `include/pgo/solver/solver_options.hpp`
- 创建: `include/pgo/solver/feasible_set.hpp`
- 创建: `include/pgo/solver/line_search.hpp`
- 创建: `tests/solver/test_line_search.cpp`
- 修改: `tests/CMakeLists.txt`

- [ ] **Step 1: 定义 `SolverResult<T>`**

字段：

- `bool converged`
- `std::size_t iterations`
- `T final_value`
- `T final_gradient_norm`
- `std::string message`

- [ ] **Step 2: 定义 `NewtonOptions<T>` 和 `LineSearchOptions<T>`**

`include/pgo/solver/solver_options.hpp` 提供：

```cpp
namespace pgo::solver {

template <typename T>
struct LineSearchOptions {
    T armijo_c = T{1e-4};
    T shrink = T{0.5};
    T min_step = T{1e-12};
    T feasibility_safety = T{0.99};
};

template <typename T>
struct NewtonOptions {
    std::size_t max_iterations = 50;
    T gradient_tolerance = T{1e-8};
    T initial_regularization = T{0};
    T min_regularization = T{1e-12};
    T regularization_growth = T{10};
    T max_regularization = T{1e8};
    LineSearchOptions<T> line_search;
};

} // namespace pgo::solver
```

- [ ] **Step 3: 定义 `AlwaysFeasible<T>` 和 feasible-set concept**

`include/pgo/solver/feasible_set.hpp` 提供：

```cpp
namespace pgo::solver {

template <typename FeasibleSet, typename T>
concept FeasibleSetLike = requires(const FeasibleSet& feasible,
                                   const pgo::math::DVec<T>& z,
                                   const pgo::math::DVec<T>& dz) {
    { feasible.is_feasible(z) } -> std::same_as<bool>;
    { feasible.max_step(z, dz) } -> std::same_as<T>;
};

template <typename T>
struct AlwaysFeasible {
    [[nodiscard]] bool is_feasible(const pgo::math::DVec<T>&) const;
    [[nodiscard]] T max_step(const pgo::math::DVec<T>&,
                             const pgo::math::DVec<T>&) const;
};

} // namespace pgo::solver
```

Milestone 1 只实现 `AlwaysFeasible`。`ReducedFeasibleSet`、IPC/CCD feasibility、barrier domain check 留到后续 milestone。

- [ ] **Step 4: 实现 feasible Armijo backtracking line search**

输入：

- energy
- feasible set
- current optimization variable `z`
- direction `dz`
- gradient `g`
- current value
- line-search options

输出 accepted step size。推荐 API：

```cpp
template <typename T, typename Energy, typename FeasibleSet>
    requires pgo::energy::DifferentiableEnergy<Energy, T> &&
             FeasibleSetLike<FeasibleSet, T>
[[nodiscard]] T feasible_armijo_line_search(
    const Energy& energy,
    const FeasibleSet& feasible,
    const pgo::math::DVec<T>& z,
    const pgo::math::DVec<T>& dz,
    const pgo::math::DVec<T>& gradient,
    T current_value,
    const LineSearchOptions<T>& options);
```

实现语义：

```text
alpha = min(1, feasibility_safety * feasible.max_step(z, dz))
while alpha >= min_step:
  trial = z + alpha * dz
  if !feasible.is_feasible(trial):
    alpha *= shrink
    continue
  if energy.value(trial) <= current_value + armijo_c * alpha * gradient.dot(dz):
    return alpha
  alpha *= shrink
return 0
```

设计约束：

- Dirichlet feasibility 已经由 `ReducedEnergyView` 的 reduced variables 保证，不需要 line search 额外检查。
- Line search 不修改 Hessian，不负责 PSD projection。
- 如果 `gradient.dot(dz) >= 0`，调用者应该先拒绝 direction；line search 可以 assert/require descent 或返回 0。

- [ ] **Step 5: 添加 line search 测试**

测试内容：

- 对一维 quadratic，descent direction 接受 `alpha = 1`。
- 非充分下降时会 backtrack。
- `AlwaysFeasible::max_step == 1` 且 `is_feasible == true`。
- 构造一个 toy feasible set，其 `max_step` 返回 `0.25`，验证初始 alpha 被 feasibility cap 限制。

- [ ] **Step 6: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R "LineSearch|solver"
```

### Task 4.4: 添加 damped Newton solver

**文件:**
- 创建: `include/pgo/solver/newton_solver.hpp`
- 修改: `tests/solver/test_solver.cpp`
- 修改: `tests/CMakeLists.txt`

- [ ] **Step 1: 实现 Newton solver**

行为：

- 输入 generic `DifferentiableEnergy` 和 initial optimization variable `z`；在 reduced problem 中这个 `z` 就是 `free_u`。
- 每轮计算 value、gradient、Hessian。
- 解 `(H + lambda I) du = -g`，使用 Eigen `SimplicialLDLT`。
- factorization 失败或 direction 不是 descent 时增加 diagonal regularization。
- 使用 feasible Armijo line search；默认 feasible set 是 `AlwaysFeasible<T>`。
- gradient norm 小于 tolerance 或达到 max iterations 时停止。

推荐 API：

```cpp
namespace pgo::solver {

template <typename T, typename Energy>
    requires pgo::energy::DifferentiableEnergy<Energy, T>
SolverResult<T> solve_newton(const Energy& energy,
                             pgo::math::DVec<T>& z,
                             const NewtonOptions<T>& options = {});

template <typename T, typename Energy, typename FeasibleSet>
    requires pgo::energy::DifferentiableEnergy<Energy, T> &&
             FeasibleSetLike<FeasibleSet, T>
SolverResult<T> solve_newton(const Energy& energy,
                             const FeasibleSet& feasible,
                             pgo::math::DVec<T>& z,
                             const NewtonOptions<T>& options = {});

} // namespace pgo::solver
```

实现细节：

- 每次迭代先调用 `energy.value_gradient_hessian(z, value, g, H)`。
- 若 `||g|| <= gradient_tolerance`，返回 converged。
- 对每轮 Newton system，尝试 `lambda = initial_regularization`；当 `lambda == 0` 失败时下一次使用 `min_regularization`。
- 构造 `H_mod = H + lambda I`。Milestone 1 只做 diagonal shift，不实现完整 modified Cholesky 或 PSD projection policy。
- `SimplicialLDLT` factorization 或 solve 失败时增大 `lambda`。
- 如果 `du` 有 NaN/Inf，或 `g.dot(du) >= 0`，增大 `lambda`。
- 找到 descent direction 后调用 `feasible_armijo_line_search`。
- line search 返回 0 或低于 `min_step` 时返回 failure message。
- 成功接受 step 后执行 `z += alpha * du`。

- [ ] **Step 2: 添加 quadratic convergence 测试**

使用 positive definite quadratic energy，验证 Newton 收敛到解析 minimizer。

- [ ] **Step 3: 添加 diagonal regularization 测试**

使用一个 Hessian 在初始点不可直接给出 descent direction 的 toy differentiable energy，验证 solver 会增加 `lambda` 并最终下降。

要求：

- energy 层返回 true Hessian。
- 测试只检查 solver 能得到下降并收敛到合理点；不要要求 energy Hessian PSD。

- [ ] **Step 4: 添加 reduced mass-spring smoke test**

创建三点 chain，固定 vertex 0，对末端施加简单 external force energy，求解 reduced energy，验证末端 displacement 朝 force 方向。

为了避免 2D chain 的自由转动/零模导致 `SimplicialLDLT` 不稳定，测试建议：

- 使用 1D chain；或
- 使用 2D chain 但固定所有 y 分量，只允许 x 方向自由。

测试 pipeline：

```text
MassSpringLocalEnergyProvider
  -> AssembledEnergy
  -> ExternalForceEnergy / Quadratic test energy
  -> EnergySum
  -> ReducedEnergyView
  -> solve_newton
```

- [ ] **Step 5: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R solver
```

### Task 4.5: 添加 minimal inertia energy 和 BackwardEuler integrator

**文件:**
- 创建: `include/pgo/energy/inertial_energy.hpp`
- 创建: `include/pgo/integrator/dynamic_state.hpp`
- 创建: `include/pgo/integrator/time_step_result.hpp`
- 创建: `include/pgo/integrator/backward_euler.hpp`
- 创建: `tests/integrator/test_backward_euler.cpp`
- 修改: `tests/CMakeLists.txt`

- [ ] **Step 1: 实现 `DynamicState<T>` 和 `TimeStepResult<T>`**

`include/pgo/integrator/dynamic_state.hpp`:

```cpp
namespace pgo::integrator {

template <typename T>
struct DynamicState {
    pgo::math::DVec<T> u;
    pgo::math::DVec<T> v;
    pgo::math::DVec<T> a;
};

} // namespace pgo::integrator
```

`a` 在 Milestone 1 的 Backward Euler 中可以不使用，但保留给 Newmark。

`include/pgo/integrator/time_step_result.hpp`:

```cpp
namespace pgo::integrator {

template <typename T>
struct TimeStepResult {
    bool converged = false;
    std::size_t solver_iterations = 0;
    T final_value{};
    T final_gradient_norm{};
    T dt{};
    std::string message;
};

} // namespace pgo::integrator
```

- [ ] **Step 2: 实现 `LumpedInertialEnergy<T>`**

`include/pgo/energy/inertial_energy.hpp` 提供 full-space inertia energy：

```text
E_inertia(u) = 0.5 / (dt * dt) * sum_i mass_i * (u_i - u_hat_i)^2
gradient_i = mass_i * (u_i - u_hat_i) / (dt * dt)
H_ii = mass_i / (dt * dt)
```

推荐 API：

```cpp
namespace pgo::energy {

template <typename T>
class LumpedInertialEnergy {
public:
    LumpedInertialEnergy(const pgo::math::DVec<T>& lumped_mass,
                         const pgo::math::DVec<T>& u_hat,
                         T dt);

    [[nodiscard]] T value(const pgo::math::DVec<T>& full_u) const;
    void gradient(const pgo::math::DVec<T>& full_u, pgo::math::DVec<T>& full_g) const;
    void hessian(const pgo::math::DVec<T>& full_u, pgo::math::SparseMat<T>& full_H) const;
    void value_gradient_hessian(const pgo::math::DVec<T>& full_u, T& value,
                                pgo::math::DVec<T>& full_g,
                                pgo::math::SparseMat<T>& full_H) const;
};

} // namespace pgo::energy
```

实现要求：

- `lumped_mass.size() == u_hat.size()`。
- `full_u.size() == u_hat.size()`。
- `dt > 0`。
- mass 不能为负。零 mass 暂时允许，但可能导致 singular system；测试应使用正 mass。
- `LumpedInertialEnergy` 满足 `FullEnergy`。

- [ ] **Step 3: 实现 concrete `BackwardEuler<T>`**

`include/pgo/integrator/backward_euler.hpp` 提供：

```cpp
namespace pgo::integrator {

template <typename T>
class BackwardEuler {
public:
    template <typename PotentialEnergy>
        requires pgo::energy::FullEnergy<PotentialEnergy, T>
    TimeStepResult<T> step(const PotentialEnergy& potential_energy,
                           const pgo::math::DVec<T>& lumped_mass,
                           const pgo::dof::ReducedDofMap<T>& dof_map,
                           DynamicState<T>& state,
                           T dt,
                           const pgo::solver::NewtonOptions<T>& options = {}) const;
};

} // namespace pgo::integrator
```

内部流程：

```text
u_old = state.u
u_hat = state.u + dt * state.v
inertia_energy = LumpedInertialEnergy(lumped_mass, u_hat, dt)
step_energy = EnergySum(potential_energy, inertia_energy)
reduced_energy = ReducedEnergyView(step_energy, dof_map)
free_u = dof_map.restrict_vector_to_free(state.u)
solver_result = solve_newton(reduced_energy, AlwaysFeasible, free_u, options)
u_next = dof_map.scatter_solution(free_u)
v_next = (u_next - u_old) / dt
state.u = u_next
state.v = v_next
return TimeStepResult copied from solver_result
```

设计约束：

- 不引入 `IntegratorBase`、`TimeIntegrator` concept、runtime polymorphism 或 solver policy。
- `BackwardEuler` 是 orchestration layer：构造 step energy、调用 solver、commit state。
- Integrator 不实现 line search，不处理 Hessian regularization，不知道 local provider。

- [ ] **Step 4: 添加 inertia energy 测试**

测试：

- `LumpedInertialEnergy` value / gradient / Hessian 与手算结果一致。
- `value_gradient_hessian` 与单独接口一致。
- `static_assert(pgo::energy::FullEnergy<LumpedInertialEnergy<double>, double>)`。

- [ ] **Step 5: 添加 Backward Euler smoke test**

使用单自由度或小 chain：

- 构造 `DynamicState`，`u = 0`，`v = 0`。
- potential energy 使用简单 external force / quadratic energy。
- 设置正 lumped mass 和 `dt`。
- 固定部分 DOF，验证 fixed DOF 在 step 后仍为 prescribed value。
- 验证自由 DOF 朝 external force 方向更新。
- 验证 `state.v == (state.u_new - state.u_old) / dt`。

- [ ] **Step 6: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R "Inertial|BackwardEuler|solver"
```

## Phase 5: Example Simulation 和 OBJ Frame Pipeline

### Task 5.1: 添加 example asset 和 CMake

**文件:**
- 创建: `examples/CMakeLists.txt`
- 创建: `examples/assets/cloth_grid.obj`

- [ ] **Step 1: 创建 example target**

```cmake
add_executable(pgo_mass_spring_cloth mass_spring_cloth.cpp)
target_link_libraries(pgo_mass_spring_cloth PRIVATE pgo::core CLI11::CLI11)
```

- [ ] **Step 2: 创建 4x4 cloth grid OBJ**

要求：

- 使用 triangular faces。
- 顶部一行可以通过最大 `y` 坐标识别。
- mesh edge indices 可以从 faces 自动抽取。


### Task 5.2: 添加 mass-spring cloth example

**文件:**
- 创建: `examples/mass_spring_cloth.cpp`

- [ ] **Step 1: 实现 CLI**

参数：

```text
--input examples/assets/cloth_grid.obj
--output frames
--frames 40
--stiffness 100
--gravity 9.8
--dt 0.016
```

执行流程：

- 读取 OBJ 为 rest mesh。
- 从 `mesh.edge_indices()` / `mesh.edge_vertex()` 构造 mass-spring energy。
- 固定 top-row vertices 的 displacement DOFs 为 0。
- 每一帧将 gravity 从 0 ramp 到目标值。
- 使用 Phase 4 的 `BackwardEuler<T>::step(...)` 求解一个动态 timestep。
- 使用 `ObjFrameWriter` 写出 OBJ frame。

- [ ] **Step 2: 构建并运行 example**

```bash
cmake --build --preset debug --target pgo_mass_spring_cloth
./build/debug/examples/pgo_mass_spring_cloth --input examples/assets/cloth_grid.obj --output frames --frames 5
```

期望：生成 `frames/frame_0000.obj` 到 `frames/frame_0004.obj`。


### Task 5.3: 添加 OBJ frames -> Alembic C++ tool

**文件:**
- 创建: `tools/CMakeLists.txt`
- 创建: `tools/obj_frames_to_abc.cpp`
- 修改: `conanfile.py`
- 修改: `cmake/pgo_options.cmake`
- 修改: `cmake/pgo_dependencies.cmake`
- 修改: `CMakeLists.txt`

- [x] **Step 1: 实现工具**

行为：

- 接收 `--frames-dir`、`--output`、`--fps`。
- 按字典序读取 `frame_*.obj`。
- 要求所有 frame topology 一致。
- 通过 Conan 依赖 `alembic/1.8.8`，Imath 由 Alembic 传递引入。
- 构建 target: `pgo_obj_frames_to_abc`。
- 成功时写出 animated polymesh `.abc`。

- [x] **Step 2: 构建和运行验证**

```bash
cmake --build --preset debug --target pgo_obj_frames_to_abc
./build/debug/tools/pgo_obj_frames_to_abc --frames-dir output/example/obj_io --output output/example/bunny.abc --fps 24
```

期望：写出 `output/example/bunny.abc`，并打印 frame、vertex、face、fps、output summary。


## Phase 6: C99 ABI 动态库桥接层

### Task 6.1: 添加 C API 导出宏和 public header

**文件:**
- 创建: `include/pgo_c/export.h`
- 创建: `include/pgo_c/pgo.h`

- [ ] **Step 1: 创建 `include/pgo_c/export.h`**

```c
#ifndef PGO_C_EXPORT_H
#define PGO_C_EXPORT_H

#if defined(_WIN32)
#  if defined(PGO_C_BUILDING_LIBRARY)
#    define PGO_C_API __declspec(dllexport)
#  else
#    define PGO_C_API __declspec(dllimport)
#  endif
#else
#  define PGO_C_API __attribute__((visibility("default")))
#endif

#endif /* PGO_C_EXPORT_H */
```

- [ ] **Step 2: 创建 `include/pgo_c/pgo.h`**

要求：

- 必须是 C99-compatible。
- 不 include C++/Eigen/STL headers。
- 使用 `extern "C"` guard。
- 所有 handle 都 opaque。
- 所有 descriptor 都有 `size` 和 `version`。

最小 API：

```c
typedef struct pgo_world_t pgo_world_t;

PGO_C_API uint32_t pgo_api_version(void);
PGO_C_API void pgo_error_clear(pgo_error_t* error);
PGO_C_API const char* pgo_status_name(pgo_status_t status);

PGO_C_API pgo_status_t pgo_world_create_mass_spring(const pgo_mesh_desc_t* mesh_desc,
                                                    const pgo_mass_spring_desc_t* sim_desc,
                                                    pgo_world_t** out_world,
                                                    pgo_error_t* error);

PGO_C_API void pgo_world_destroy(pgo_world_t* world);
PGO_C_API pgo_status_t pgo_world_fix_vertex(pgo_world_t* world, uint32_t vertex, pgo_error_t* error);
PGO_C_API pgo_status_t pgo_world_solve_static(pgo_world_t* world, double gravity_scale, pgo_error_t* error);
PGO_C_API pgo_status_t pgo_world_copy_displacements(const pgo_world_t* world,
                                                    double* out_displacements,
                                                    size_t displacement_count,
                                                    pgo_error_t* error);
```


### Task 6.2: 添加 C API shared library target

**文件:**
- 创建: `src/c_api/CMakeLists.txt`
- 创建: `src/c_api/pgo_c.cpp`

- [ ] **Step 1: 创建 `src/c_api/CMakeLists.txt`**

```cmake
add_library(pgo_c SHARED pgo_c.cpp)
add_library(pgo::c ALIAS pgo_c)

target_compile_features(pgo_c PRIVATE cxx_std_23)
target_link_libraries(pgo_c PRIVATE pgo::core)
target_include_directories(pgo_c PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(pgo_c PRIVATE PGO_C_BUILDING_LIBRARY)

set_target_properties(pgo_c PROPERTIES
    OUTPUT_NAME pgo
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if(APPLE)
    set_target_properties(pgo_c PROPERTIES INSTALL_NAME_DIR "@rpath")
elseif(UNIX)
    set_target_properties(pgo_c PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
```

- [ ] **Step 2: 创建可构建 stub `src/c_api/pgo_c.cpp`**

stub 只保证 target 立刻可构建。Task 6.3 会替换为真实 bridge。

要求：

- include `pgo_c/pgo.h`。
- 实现 `pgo_api_version`、`pgo_error_clear`、`pgo_status_name`。
- 其他 API 返回 `PGO_STATUS_INTERNAL_ERROR`，并写入错误消息 `"pgo C API stub called before bridge task completed"`。
- 每个 exported function 使用 `extern "C" PGO_C_API`。

- [ ] **Step 3: 构建 C API target**

```bash
cmake --preset debug
cmake --build --preset debug --target pgo_c
```

期望：生成 `libpgo.dylib`、`libpgo.so` 或 `pgo.dll`。


### Task 6.3: 实现 C++ bridge，不暴露 C++ ABI

**文件:**
- 修改: `src/c_api/pgo_c.cpp`

- [ ] **Step 1: 实现真实 bridge**

规则：

- `pgo_c/pgo.h` 必须第一个 include。
- `struct pgo_world_t` 只定义在 `.cpp` 内。
- `pgo_world_t` 内部可以用 `std::unique_ptr`、`std::variant` 等 RAII 类型。
- Milestone 1 只支持 `PGO_SCALAR_FLOAT64` 与 `PGO_DIM_2`/`PGO_DIM_3`。
- descriptor 校验使用 `desc->size >= sizeof(desc_type)` 和 `desc->version == PGO_C_API_VERSION`。
- exported C function 边界必须 catch `std::exception` 和 `catch (...)`。
- 不返回临时 `std::string` 的 `c_str()`。
- 错误信息写入 caller-provided `pgo_error_t::message` 固定数组。

内部可显式实例化或分派到：

```text
MassSpringWorld<double, 2>
MassSpringWorld<double, 3>
```

这些 C++ 类型不得出现在 `include/pgo_c/pgo.h`。

- [ ] **Step 2: 构建 C API target**

```bash
cmake --preset debug
cmake --build --preset debug --target pgo_c
```


### Task 6.4: 添加纯 C API smoke test

**文件:**
- 创建: `tests/pgo_c/test_c_api.c`
- 修改: `tests/CMakeLists.txt`

- [ ] **Step 1: 创建 `tests/pgo_c/test_c_api.c`**

要求：

- 文件用 C 编译，不是 C++。
- include `pgo_c/pgo.h`。
- 检查 `pgo_api_version() == PGO_C_API_VERSION`。
- 构造两点一边的 3D mesh descriptor。
- 调用 `pgo_world_create_mass_spring`。
- 调用 `pgo_world_fix_vertex(world, 0)`。
- 调用 `pgo_world_destroy(world)`。

- [ ] **Step 2: 修改 `tests/CMakeLists.txt`**

追加：

```cmake
if(TARGET pgo::c)
    add_executable(pgo_c_api_tests pgo_c/test_c_api.c)
    target_link_libraries(pgo_c_api_tests PRIVATE pgo::c)
    add_test(NAME pgo_c_api_tests COMMAND pgo_c_api_tests)
endif()
```

- [ ] **Step 3: 运行 C API 测试**

```bash
cmake --build --preset debug --target pgo_c_api_tests
ctest --preset debug -R pgo_c_api_tests
```

- [ ] **Step 4: 检查 exported symbols**

macOS/Linux:

```bash
nm -gU build/debug/src/c_api/libpgo.dylib 2>/dev/null || nm -D --defined-only build/debug/src/c_api/libpgo.so
```

期望：只导出 `pgo_*` C API symbols，不导出 C++ template internals。


## Phase 7: CI 和验证

### Task 7.1: 升级 GitHub Actions CI 为全平台 build/test

**文件:**
- 修改: `.github/workflows/ci.yml`

- [x] **Step 1: 在 Phase 0 build-only CI 基础上加入 Linux/macOS/Windows 测试步骤**

CI 至少包含：

- `ubuntu-latest` Debug build/test。
- `macos-latest` Debug build/test。
- `windows-latest` Debug build/test。
- `ubuntu-latest` ASan/UBSan build/test。

Debug job 使用对应仓库内 profile：

```text
ubuntu-latest  -> conan/profiles/ubuntu-x86_64-gcc
macos-latest   -> conan/profiles/macos-arm64-apple-clang
windows-latest -> conan/profiles/windows-x86_64-msvc
```

Linux/macOS Debug job 执行：

```bash
uv tool install conan
conan install . \
  --profile:host=conan/profiles/${PROFILE_NAME} \
  --profile:build=conan/profiles/${PROFILE_NAME} \
  --output-folder=build/conan/debug \
  --build=missing \
  -s:h build_type=Debug
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Windows Debug job 执行同样的 configure/build/test 流程，但 shell 使用 PowerShell 或 bash 均可；路径变量不要写死 Unix-only 路径。命令语义是：

```powershell
uv tool install conan
conan install . `
  --profile:host=conan/profiles/windows-x86_64-msvc `
  --profile:build=conan/profiles/windows-x86_64-msvc `
  --output-folder=build/conan/debug `
  --build=missing `
  -s:h build_type=Debug
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

ASan job 使用：

```bash
cmake --preset asan
cmake --build --preset asan
ctest --preset asan
```

要求：Phase 0 已经存在 `build-debug` 和 `sanitize` jobs；本任务只是在 tests/examples/C API targets 存在后恢复 `ctest`，不要退回到本机 Conan default profile。

- [x] **Step 2: 添加 preset-driven `PGO_ENABLE_EIGEN_ACCELERATION=ON` CI 验证**

默认全平台 CI 必须保持：

```text
PGO_ENABLE_EIGEN_ACCELERATION=OFF
```

这是 baseline job，保证没有 acceleration dependency 的用户也能稳定构建。

同时必须添加 acceleration-on jobs，全部使用 preset：

- `macos-latest` + `debug-acceleration`，CMake `AUTO` 选择 Accelerate。
- `ubuntu-latest` + `debug-acceleration`，CMake `AUTO` 选择 MKL。
- `windows-latest` + `debug-acceleration`，CMake `AUTO` 选择 MKL。

MKL jobs 必须在 CMake configure 前安装系统 oneMKL，并保证 `find_package(MKL CONFIG REQUIRED)` 能找到 `MKLConfig.cmake`。Conan 不负责 MKL；CI 复用 `scripts/install-onemkl/install-onemkl-linux.sh` 和 `scripts/install-onemkl/install-onemkl-windows.ps1` 完成平台安装。

- [x] **Step 3: 添加 acceleration-on CI 命令**

macOS Accelerate job：

```bash
uv tool install conan
conan install . \
  --profile:host=conan/profiles/macos-arm64-apple-clang \
  --profile:build=conan/profiles/macos-arm64-apple-clang \
  --output-folder=build/conan/debug-acceleration \
  --build=missing \
  -s:h build_type=Debug
cmake --preset debug-acceleration
cmake --build --preset debug-acceleration
ctest --preset debug-acceleration -R EigenConfig
```

Ubuntu MKL job：

```bash
uv tool install conan
scripts/install-onemkl/install-onemkl-linux.sh
conan install . \
  --profile:host=conan/profiles/ubuntu-x86_64-gcc \
  --profile:build=conan/profiles/ubuntu-x86_64-gcc \
  --output-folder=build/conan/debug-acceleration \
  --build=missing \
  -s:h build_type=Debug
cmake --preset debug-acceleration
cmake --build --preset debug-acceleration
ctest --preset debug-acceleration -R EigenConfig
```

Windows MKL job：

```powershell
uv tool install conan
.\scripts\install-onemkl\install-onemkl-windows.ps1
conan install . `
  --profile:host=conan/profiles/windows-x86_64-msvc `
  --profile:build=conan/profiles/windows-x86_64-msvc `
  --output-folder=build/conan/debug-acceleration `
  --build=missing `
  -s:h build_type=Debug
cmake --preset debug-acceleration
cmake --build --preset debug-acceleration
ctest --preset debug-acceleration -R EigenConfig
```

- [x] **Step 4: 添加 CI matrix 设计说明**

`.github/workflows/ci.yml` 推荐使用 matrix 表达全平台 Debug jobs：

```yaml
strategy:
  fail-fast: false
  matrix:
    include:
      - os: ubuntu-latest
        profile: conan/profiles/ubuntu-x86_64-gcc
        preset: debug
      - os: macos-latest
        profile: conan/profiles/macos-arm64-apple-clang
        preset: debug
      - os: windows-latest
        profile: conan/profiles/windows-x86_64-msvc
        preset: debug
```

ASan job 单独留在 Ubuntu。Windows 不跑 sanitizer job。

Acceleration-on jobs 可以使用单独 matrix：

```yaml
strategy:
  fail-fast: false
  matrix:
    include:
      - os: macos-latest
        profile: conan/profiles/macos-arm64-apple-clang
        preset: debug-acceleration
        needs_mkl: false
      - os: ubuntu-latest
        profile: conan/profiles/ubuntu-x86_64-gcc
        preset: debug-acceleration
        needs_mkl: true
      - os: windows-latest
        profile: conan/profiles/windows-x86_64-msvc
        preset: debug-acceleration
        needs_mkl: true
```


### Task 7.2: 添加 README 构建说明

**文件:**
- 创建: `README.md`

- [ ] **Step 1: 写入本地构建命令**

README 至少包含：

````markdown
# pgo

GPU-aware C++23 PGO learning and refactoring project.

## Milestone 1

Milestone 1 builds a CPU mass-spring solver around explicit rest positions `X`,
displacement unknowns `u`, local energy assembly, Newton solving, OBJ frame output,
and a C99 dynamic-library API.

## Local Build

```bash
uv tool install conan
conan install . \
  --profile:host=conan/profiles/default \
  --profile:build=conan/profiles/default \
  --output-folder=build/conan/debug \
  --build=missing \
  -s:h build_type=Debug
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

## Example

```bash
./build/debug/examples/pgo_mass_spring_cloth --input examples/assets/cloth_grid.obj --output frames --frames 20
```
````


## Phase 8: GPU-Awareness 和 ABI Review Gate

### Task 8.1: 检查 GPU-aware 约束

**文件:**
- 修改: `README.md`

- [ ] **Step 1: 检查 geometry/storage 不泄漏 Eigen object storage**

```bash
rg "std::vector<.*Eigen|Eigen::Vector[234]|Eigen::Matrix<.*Dynamic" include/pgo/geometry include/pgo/storage
```

期望：无匹配，除了 `include/pgo/math` 中允许的 math aliases。

- [ ] **Step 2: 检查 solver variable 命名**

```bash
rg "current.*unknown|position.*unknown|optimize.*position|solver.* x" include examples tests
```

期望：没有文档或代码把 position 说成 solver unknown。未知量应命名为 `u`、`free_u`、`du`。

- [ ] **Step 3: 检查 local energy API**

```bash
rg "local_count|local_dofs|local_value|local_gradient|local_hessian" include/pgo/energy include/pgo/assembly
```

期望：energy/assembly 中存在 local contribution API。

- [ ] **Step 4: README 记录 GPU backend 方向**

添加：

```markdown
## GPU Backend Direction

The CPU implementation keeps mesh data flat and energy evaluation local-contribution based.
This is intentional: a Vulkan/Slang backend can dispatch one work item per spring, element,
or contact candidate while reusing the same semantic model.
```


### Task 8.2: 检查 C ABI 边界

**文件:**
- 修改: `README.md`

- [ ] **Step 1: 检查 public C header 不泄漏 C++**

```bash
rg "std::|Eigen::|template|class|namespace|#include <vector>|#include <string>" include/pgo_c
```

期望：public C headers 无匹配。

- [ ] **Step 2: 检查 C API 实现 catch exceptions**

```bash
rg "catch \\(const std::exception&|catch \\(\\.\\.\\.\\)" src/c_api/pgo_c.cpp
```

期望：每个会调用 C++ core 的 exported function 都有 exception handling。

- [ ] **Step 3: 检查 exported symbols**

```bash
nm -gU build/debug/src/c_api/libpgo.dylib 2>/dev/null || nm -D --defined-only build/debug/src/c_api/libpgo.so
```

期望：只导出 `pgo_*` API functions 和平台 runtime symbols。

- [ ] **Step 4: README 记录 ABI 边界**

添加：

```markdown
## C ABI Boundary

The template C++ core is not the binary interface. The shared library exposes only
a C99 ABI in `include/pgo_c/pgo.h`: opaque handles, POD descriptors, pointer/count
arrays, explicit destroy functions, and status/error returns. C++ exceptions,
STL types, Eigen types, and template types do not cross this boundary.
```


## Milestone 1 完成标准

- `cmake --preset debug` 在 Conan 依赖安装后成功。
- `cmake --build --preset debug` 成功。
- `ctest --preset debug` 通过。
- Linux 或本地 Clang/GCC 环境中，`cmake --preset asan`、`cmake --build --preset asan`、`ctest --preset asan` 通过。
- Mass-spring cloth example 能写出 OBJ frames。
- Solver 优化 free displacement variables，而不是 current positions。
- Rest positions `X` 在求解过程中保持 immutable。
- OBJ frame output 写出 `X + u`。
- Mesh/geometry storage 保持 flat、index-based。
- Energy model/provider 暴露 local contribution API，适合 CPU assembly 和未来 GPU dispatch。
- `pgo_c` 构建为 shared library，并通过 selected concrete C++ template instantiations 暴露 C99 ABI。
- `include/pgo_c/pgo.h` 能作为 C 编译，不暴露 STL、Eigen、C++ templates、exceptions、namespaces 或 C++ classes。
- C API ownership/error 规则明确：handles 由 matching destroy functions 释放，descriptors 是 borrowed，错误通过 `pgo_status_t` + `pgo_error_t` 返回。
- `PGO_ENABLE_EIGEN_ACCELERATION=OFF` 默认构建通过；打开后 Apple 可走 Accelerate，Ubuntu/Windows 可在 oneMKL 可用时走 MKL。
- PARDISO 被记录为 future linear solver backend，而不是 Eigen `MathBackend` 或 Eigen acceleration 开关的一部分。
- GitHub Actions CPU build/test jobs 在 Ubuntu、macOS、Windows 三个平台通过；Ubuntu ASan/UBSan job 通过。
- Eigen acceleration CI jobs 是可选验证，不作为默认全平台 CI 通过门槛。

## 推荐实现顺序

1. 完成 Phase 0，先让构建和 CI 骨架站起来。
2. 完成 Phase 0.5，把 Eigen acceleration、MKL/Accelerate 和 PARDISO/solver backend 的边界写清楚。
3. 完成 Phase 1，建立 flat storage、rest mesh、DOF/reduced DOF。
4. 完成 Phase 2，尽早获得可视化输出能力。
5. 完成 Phase 3，用 derivative tests 保护 energy 实现。
6. 完成 Phase 4，先用 quadratic system 验证 solver，再跑 mass-spring。
7. 完成 Phase 5，生成 OBJ frames。
8. 完成 Phase 6，暴露稳定的 C99 shared-library facade。
9. 完成 Phase 7 和 Phase 8 后，再进入 Milestone 2 GPU backend。

## 自检记录

- 覆盖范围：计划覆盖 build system、CI、Eigen acceleration 配置、C++23 header-oriented core、Eigen backend layer、GPU-aware storage、rest/displacement 分离、DOF reduction、OBJ input/output、local energy model/provider assembly、mass-spring energy、Newton solver、example frames、C99 dynamic-library API、Alembic 后处理。
- 占位扫描：计划不包含 `TBD`、`TODO`、`implement later` 等未落实占位。
- 类型一致性：核心名称统一使用 `RestMesh`、`DofLayout`、`Displacement`、`DirichletBoundary`、`ReducedDofMap`、`MassSpringLocalEnergyModel`、`MassSpringLocalEnergyProvider`、`ReducedEnergyView`、`ObjFrameWriter`、`NewtonSolver`、`pgo_world_t`、`pgo_error_t`。
- 范围控制：Vulkan、Slang、FEM、contact、IPC、GPU solvers 不进入 Milestone 1，但数据布局和 API 边界保持兼容。
