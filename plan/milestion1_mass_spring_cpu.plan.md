# GPU-Aware Mass Spring CPU 实现计划

> **给 agentic workers:** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务执行。本计划使用 checkbox (`- [ ]`) 语法追踪进度。

**目标:** 完成 Milestone 1：实现一个 CPU-only、GPU-aware 的 C++23 header-oriented mass-spring simulation core。它读取 rest mesh，优化 displacement `u`，输出 OBJ frame sequence，并提供最小稳定的 C99 动态库 API。

**架构:** 内部核心是现代 C++23 template/header-only library；对外二进制接口是 compiled C99 ABI dynamic library。CPU 实现优先，但数据布局和 energy interface 从第一天就为 Milestone 2 的 Vulkan/Slang GPU backend 留好边界：flat storage、显式 `X + u`、local contribution API、独立 assembly 层、geometry/storage 不持有 Eigen object。

**技术栈:** C++23、C99 ABI、CMake Presets、Conan 2、Eigen、GoogleTest、CLI11、clang-format、GitHub Actions、Python Alembic binding（可选，用于 OBJ frames 后处理）。

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
- Alembic 不进入 C++ core。`.abc` 由 Python tool 消费 OBJ frames 后生成。
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
        mass_spring_energy.hpp
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
    base/
      test_assert.cpp
    dof/
      test_dof.cpp
    energy/
      test_mass_spring_energy.cpp
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
    obj_frames_to_abc.py
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

Windows 安装脚本通过 winget 安装 `Intel.oneMKL`，并校验默认安装位置提供 `MKLConfig.cmake`。在 GitHub Actions 中额外写入 `MKLROOT`、`MKL_DIR`、`CMAKE_PREFIX_PATH`、`LIB`，同时用 `GITHUB_PATH` 暴露运行时 DLL 目录。

- [ ] **Step 4: 验证 Linux/Windows 系统 oneMKL 配置**

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

- [ ] **Step 6: 验证 MKL 配置**

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


### Task 1.2: 添加 math backend contracts 和 Eigen backend

**文件:**
- 创建: `include/pgo/math/scalar.hpp`
- 创建: `include/pgo/math/eigen_backend.hpp`
- 创建: `include/pgo/math/backend.hpp`
- 创建: `tests/math/test_backend.cpp`
- 修改: `tests/CMakeLists.txt`

- [x] **Step 1: 定义 scalar concepts**

`scalar.hpp` 提供：

```cpp
namespace pgo::math {
template <class T>
concept ScalarLike = requires(T a, T b) {
    T{0};
    T{1};
    a + b;
    a - b;
    a * b;
    a / b;
    -a;
};

template <class T>
concept RealScalar = std::floating_point<T>;
} // namespace pgo::math
```

设计约束：

- `ScalarLike` 用于 local formula、小型向量/矩阵和 energy 表达式，允许 `double`、`float`、Eigen `AutoDiffScalar`，以及未来 Slang/Vulkan 侧的 dual/jet scalar 复刻同一数学形状。
- `RealScalar` 用于 global sparse assembly、linear solve、line search 这类 Milestone 1 中只支持 real floating-point 的算法。
- 如果某个 `ScalarLike` 类型需要 Eigen 支持，后续可以为它补 `Eigen::NumTraits<T>` specialization。Milestone 1 只要求 `double` 跑通。

- [x] **Step 2: 定义 `EigenBackend`**

`eigen_backend.hpp` 是 Milestone 1 唯一 math backend implementation：

```cpp
namespace pgo::math::eigen {

struct EigenBackend {
    template <pgo::math::ScalarLike T, int Dim>
    using Vec = Eigen::Matrix<T, Dim, 1>;

    template <pgo::math::ScalarLike T, int Rows, int Cols>
    using Mat = Eigen::Matrix<T, Rows, Cols>;

    template <pgo::math::ScalarLike T>
    using DVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    template <pgo::math::ScalarLike T>
    using DMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    template <pgo::math::RealScalar T>
    using SparseMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;

    template <pgo::math::RealScalar T>
    using Triplet = Eigen::Triplet<T>;
};

} // namespace pgo::math::eigen
```

设计约束：

- Eigen 是明确 backend，不是 `pgo::math` 本身。
- `SparseMat<T>` 和 `Triplet<T>` 暂时只接受 `RealScalar`。不要把 autodiff scalar 放进全局 sparse matrix/factorization；AD 应优先用于 local derivative experiment 或未来 GPU-side local formula。
- 这是 backend-aware alias layer，不是 runtime polymorphism，也不是完整 backend-agnostic 大抽象。

- [x] **Step 3: 定义 backend contract 和默认 aliases**

`backend.hpp` 提供 `MathBackend` concept、`DefaultBackend` 和默认 aliases：

```cpp
namespace pgo::math {

using Index = std::uint32_t;

template <class Backend>
concept MathBackend = requires {
    typename Backend::template Vec<double, 3>;
    typename Backend::template Mat<double, 3, 3>;
    typename Backend::template DVec<double>;
    typename Backend::template DMat<double>;
    typename Backend::template SparseMat<double>;
    typename Backend::template Triplet<double>;
};

using DefaultBackend = pgo::math::eigen::EigenBackend;
static_assert(MathBackend<DefaultBackend>);

template <ScalarLike T, int Dim>
using Vec = DefaultBackend::template Vec<T, Dim>;

template <ScalarLike T, int Rows, int Cols>
using Mat = DefaultBackend::template Mat<T, Rows, Cols>;

template <ScalarLike T>
using DVec = DefaultBackend::template DVec<T>;

template <ScalarLike T>
using DMat = DefaultBackend::template DMat<T>;

template <RealScalar T>
using SparseMat = DefaultBackend::template SparseMat<T>;

template <RealScalar T>
using Triplet = DefaultBackend::template Triplet<T>;

} // namespace pgo::math
```

设计约束：

- Milestone 1 的业务代码优先使用 `pgo::math::Vec`、`DVec`、`SparseMat` 等默认 aliases。
- 如果某个算法需要显式依赖 backend，可以模板化为 `template <class Backend = pgo::math::DefaultBackend>`。
- 未来 GPU backend 不一定实现 host-side `DVec/SparseMat`，但 local formula 的 `ScalarLike + small Vec/Mat` 形状应保持可迁移。

- [x] **Step 4: 添加 math backend 测试**

在 `tests/CMakeLists.txt` 中加入 `tests/math/test_backend.cpp`。测试文件命名空间使用：

```cpp
namespace pgo::math::test {
// TEST(...)
}
```

测试内容：

- `static_assert(pgo::math::MathBackend<pgo::math::DefaultBackend>)`。
- `pgo::math::Vec<double, 3>` 的 size 是 3。
- `pgo::math::DVec<double>` 可以 resize 到 4。
- `pgo::math::SparseMat<double>` 是 row-major sparse matrix。
- `ScalarLike` 接受 `double`。
- `RealScalar` 接受 `double`。


### Task 1.3: 添加 GPU-aware host storage primitive

**文件:**
- 创建: `include/pgo/storage/array_view.hpp`
- 创建: `include/pgo/storage/host_buffer.hpp`

- [x] **Step 1: 定义 view/buffer**

```cpp
namespace pgo::storage {
template <class T>
using ArrayView = std::span<T>;

template <class T>
using ConstArrayView = std::span<const T>;

template <class T>
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

- [ ] **Step 1: 定义 topology**

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

- [ ] **Step 2: 定义 `RestMesh<T, Dim>`**

要求：

- `rest_positions` 是 vertex-major flat buffer。
- `num_vertices() == rest_positions.size() / Dim`。
- `rest_position(i)` 返回 `math::Vec<T, Dim>`。
- 保存 `edge_indices()` 和 `face_indices()`，长度分别是 `2 * num_edges()` 和 `3 * num_faces()`。
- 提供 `edge_vertex(edge_id, local_vertex)` 和 `face_vertex(face_id, local_vertex)` helper。
- 不在 storage 中保存 Eigen vector object。
- 不在 topology storage 中保存 `std::array`、指针、对象图或 per-edge/per-face 动态分配。

- [ ] **Step 3: 建立测试 target**

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

- [ ] **Step 4: 运行测试**

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

- [ ] **Step 1: 实现 `DofLayout<Dim>`**

职责：把 `(vertex, component)` 映射到 full displacement dof index：

```text
dof = vertex * Dim + component
```

- [ ] **Step 2: 实现 `Displacement<T, Dim>`**

职责：持有 full displacement vector `u`，提供：

- `layout()`
- `vector()`
- `at(vertex)` 返回该 vertex 的 displacement vector

- [ ] **Step 3: 实现 `DirichletBoundary<T>`**

职责：底层保存 fixed full DOF values。默认固定值为 `0`。

底层表达：

```text
full_dof_index -> fixed displacement value
```

必须提供的基础 API：

- `prescribe_dof(dof, value)`：固定单个 scalar displacement DOF 到指定值，这是唯一直接写入底层 map 的 API。
- `fix_dof(dof)`：固定单个 scalar displacement DOF 到 `0`，内部 dispatch 到 `prescribe_dof(dof, T{0})`。
- `is_fixed(dof)`：查询 full DOF 是否固定。
- `value(dof)`：读取 fixed displacement value。

必须提供的 component-level API：

- `prescribe_component(layout, vertex, component, value)`：通过 `layout.index(vertex, component)` dispatch 到 `prescribe_dof`。
- `fix_component(layout, vertex, component)`：dispatch 到 `prescribe_component(..., T{0})`。

必须提供的 vertex/list convenience helpers：

- `prescribe_vertex(layout, vertex, value)`：`value` 是 `math::Vec<T, Dim>`，逐 component dispatch 到 `prescribe_component`。
- `fix_vertex(layout, vertex)`：固定某个 vertex 的所有 displacement components 到 `0`，逐 component dispatch 到 `fix_component`。
- `prescribe_vertices(layout, vertex_indices, value)`：对一组 vertices 使用同一个 prescribed displacement `value`。
- `fix_vertices(layout, vertex_indices)`：从 fixed vertex list 构造 zero displacement 边界，内部 dispatch 到 `fix_vertex`。
- `prescribe_vertices_by_list(layout, vertex_indices, values)`：每个 vertex 使用自己的 prescribed displacement。`values` 使用 flat buffer/view，要求 `values.size() == vertex_indices.size() * Dim`，布局是 `values[local_vertex * Dim + component]`。

dispatch 规则：

```text
prescribe_dof        -> 写入 fixed map
fix_dof              -> prescribe_dof(dof, 0)

prescribe_component  -> layout.index(vertex, component) -> prescribe_dof
fix_component        -> prescribe_component(..., 0)

prescribe_vertex     -> loop components -> prescribe_component
fix_vertex           -> loop components -> fix_component

prescribe_vertices   -> loop vertices -> prescribe_vertex
fix_vertices         -> loop vertices -> fix_vertex
prescribe_vertices_by_list -> loop vertices/components -> prescribe_component
```

设计约束：solver/reduced map 只依赖底层 fixed DOF 表达；example/C API 可以使用 fixed vertex list 这种更符合用户直觉的高层入口。

- [ ] **Step 4: 实现 `ReducedDofMap<T>`**

职责：

- 从 full dofs 和 boundary 生成 `free_to_full` / `full_to_free`。
- `pack_displacement(full_u)`：从 full displacement vector 选取 free DOFs，得到 `free_u`。
- `unpack_displacement(free_u, boundary)`：把 reduced/free displacement 展开回 full displacement；fixed DOFs 使用 `DirichletBoundary` 中的 prescribed values。
- `reduce_vector(full_v)`：从任意 full-space vector 选取 free DOFs，得到 reduced vector。可用于 gradient、force、residual、velocity、search direction 等。
- `reduce_sparse_mat(full_A)`：从任意 full-space sparse matrix 选取 free-free block，得到 reduced sparse matrix。可用于 Hessian、mass matrix、stiffness matrix、Jacobian normal matrix 等。

设计约束：

- displacement 的 unpack 需要 boundary value，所以命名为 `unpack_displacement`。
- 普通 vector reduction 不应该填 prescribed displacement value，因此只提供 `reduce_vector(full_v)`。
- `ReducedDofMap` 是 DOF space mapping，不应该把 API 命名绑定到 gradient/Hessian。

- [ ] **Step 5: 把 DOF 测试加入测试 target**

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

- [ ] **Step 6: 添加 DOF 测试**

`tests/dof/test_dof.cpp` 使用 `namespace pgo::dof::test`。测试内容：

- `DofLayout<3>(4)` 有 12 个 DOF。
- `layout.index(2, 1) == 7`。
- `DirichletBoundary` 可以通过 `fix_dof` 固定单个 scalar DOF 到 0。
- `DirichletBoundary` 可以通过 `prescribe_dof` 固定单个 scalar DOF 到指定值。
- `DirichletBoundary` 可以通过 `fix_vertex` 固定某个 vertex 的所有 components 到 0。
- `DirichletBoundary` 可以通过 `prescribe_vertex` 固定某个 vertex 到指定 displacement vector。
- `DirichletBoundary` 可以通过 `fix_vertices` 从 fixed vertex list 构造 zero displacement 边界。
- `DirichletBoundary` 可以通过 `prescribe_vertices` 给一组 vertices 设置同一个 prescribed displacement。
- `DirichletBoundary` 可以通过 `prescribe_vertices_by_list` 给一组 vertices 设置逐 vertex prescribed displacement，并校验 flat values 长度。
- 固定 DOF 后，`ReducedDofMap` 能正确 `pack_displacement` / `unpack_displacement`。
- `reduce_vector(full_v)` 能正确选取 free DOF entries。
- `reduce_sparse_mat(full_A)` 能正确选取 free-free sparse block。

- [ ] **Step 7: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R dof
```

## Phase 2: OBJ Mesh 输入和 OBJ Frame 输出

### Task 2.1: 添加 OBJ reader

**文件:**
- 创建: `include/pgo/io/obj_reader.hpp`
- 修改: `tests/io/test_obj_io.cpp`

- [ ] **Step 1: 实现 `read_obj_rest_mesh<T, Dim>(path)`**

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

- [ ] **Step 2: 添加测试**

创建临时 OBJ：4 个 vertices、2 个 triangles。读取后断言：

- `num_vertices() == 4`
- `num_faces() == 2`
- `num_edges() == 5`
- `face_indices().size() == 6`
- `edge_indices().size() == 10`

- [ ] **Step 3: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R obj
```

### Task 2.2: 添加 OBJ frame writer

**文件:**
- 创建: `include/pgo/io/obj_frame_writer.hpp`
- 修改: `tests/io/test_obj_io.cpp`

- [ ] **Step 1: 实现 `ObjFrameWriter<T, Dim>`**

职责：

- 输入 `RestMesh<T, Dim>` 和 full displacement vector `u`。
- 输出 `frame_0000.obj`、`frame_0001.obj` 等。
- 写出的 vertex position 是 `X + u`。
- 通过 `face_indices` 保留原 faces。
- 如果 mesh 没有 faces，则通过 `edge_indices` 写 `l` records。

- [ ] **Step 2: 添加 writer 测试**

创建两点 line mesh，设置第二个点的 displacement，写出 frame 后检查 OBJ 文本包含 displaced vertex。

- [ ] **Step 3: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R obj
```

## Phase 3: Local-Contribution Energy 和 CPU Assembly

### Task 3.1: 添加 energy concepts 和 local matrix types

**文件:**
- 创建: `include/pgo/assembly/local_matrix.hpp`
- 创建: `include/pgo/energy/energy_concepts.hpp`

- [ ] **Step 1: 定义 local aliases**

```cpp
template <pgo::math::ScalarLike T>
using LocalVector = pgo::math::DVec<T>;

template <pgo::math::ScalarLike T>
using LocalMatrix = pgo::math::DMat<T>;
```

- [ ] **Step 2: 定义 concepts**

`FullEnergy` 要求：

- `value(u) -> T`
- `gradient(u, g)`
- `hessian(u, H)`

`LocalEnergy` 要求：

- `local_count()`
- `local_dofs(local_id, dofs)`
- `local_value(local_id, u)`
- `local_gradient(local_id, u, local_g)`
- `local_hessian(local_id, u, local_H)`


### Task 3.2: 添加 CPU local energy assembler

**文件:**
- 创建: `include/pgo/assembly/cpu_assembler.hpp`

- [ ] **Step 1: 实现 assembler**

提供：

- `assemble_value(energy, u)`
- `assemble_gradient(energy, u, g)`
- `assemble_hessian(energy, u, H)`

实现要求：

- 遍历 `local_id`。
- 通过 `local_dofs` scatter local gradient。
- 通过 triplets scatter local Hessian。
- Hessian 使用 `math::SparseMat<T>`。


### Task 3.3: 添加 MassSpringEnergy

**文件:**
- 创建: `include/pgo/energy/mass_spring_energy.hpp`
- 修改: `tests/energy/test_mass_spring_energy.cpp`

- [ ] **Step 1: 实现 `MassSpringEnergy<T, Dim>`**

单条 spring 的公式：

```text
E_e(u) = 0.5 * k * (||x_i - x_j|| - L)^2
x_i = X_i + u_i
x_j = X_j + u_j
L = ||X_i - X_j||
```

要求：

- `local_count()` 等于 `mesh.num_edges()`。
- `local_dofs(edge_id, dofs)` 通过 `mesh.edge_vertex(edge_id, 0/1)` 取端点，并返回 `[i0, i1, ..., j0, j1, ...]` 对应的 full displacement DOFs。
- `local_gradient` 和 `local_hessian` 使用解析导数。
- 对零长度或极短 current edge 做 epsilon clamp，避免 NaN。

- [ ] **Step 2: 添加 derivative tests**

测试：

- rest state energy 为 0。
- rest state gradient 为 0。
- 拉伸一个端点后 energy 增大。
- finite difference gradient 与解析 gradient 匹配。
- finite difference Hessian 与解析 Hessian 匹配。

- [ ] **Step 3: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R mass_spring
```

### Task 3.4: 添加 finite difference helper

**文件:**
- 创建: `include/pgo/math/finite_difference.hpp`
- 修改: `tests/math/test_finite_difference.cpp`

- [ ] **Step 1: 实现 helper**

提供：

- `finite_difference_gradient(f, x, eps)`
- `finite_difference_hessian_from_gradient(grad, x, eps)`

使用 central difference。

- [ ] **Step 2: 添加测试**

用函数：

```text
f(x, y) = x^2 + 3xy + 2y^2
grad = [2x + 3y, 3x + 4y]
H = [[2, 3], [3, 4]]
```

- [ ] **Step 3: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R finite
```

## Phase 4: Reduced Energy 和 Newton Solver

### Task 4.1: 添加 EnergySum 和 ReducedEnergyView

**文件:**
- 创建: `include/pgo/energy/energy_sum.hpp`
- 创建: `include/pgo/energy/reduced_energy.hpp`
- 修改: `tests/solver/test_solver.cpp`

- [ ] **Step 1: 实现 `EnergySum<T>`**

组合多个 full-space energies：

- value 相加。
- gradient 相加。
- Hessian 相加。

- [ ] **Step 2: 实现 `ReducedEnergyView<T, Energy>`**

职责：

```text
free_u -> unpack_displacement 成 full_u
full energy evaluate
full gradient 通过 reduce_vector 变成 reduced gradient
full Hessian 通过 reduce_sparse_mat 变成 reduced Hessian
```

- [ ] **Step 3: 添加 reduced energy 测试**

使用 quadratic energy：

```text
E(u) = 0.5 * u^T A u - b^T u
```

固定一个 DOF，验证 reduced gradient/Hessian 分别等于 `reduce_vector(full_gradient)` 和 `reduce_sparse_mat(full_hessian)` 的结果。

- [ ] **Step 4: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R solver
```

### Task 4.2: 添加 line search 和 solver result

**文件:**
- 创建: `include/pgo/solver/solver_result.hpp`
- 创建: `include/pgo/solver/line_search.hpp`

- [ ] **Step 1: 定义 `SolverResult<T>`**

字段：

- `bool converged`
- `std::size_t iterations`
- `T final_value`
- `T final_gradient_norm`
- `std::string message`

- [ ] **Step 2: 实现 Armijo backtracking line search**

输入：

- energy
- current `u`
- direction `du`
- gradient `g`

输出 accepted step size。


### Task 4.3: 添加 damped Newton solver

**文件:**
- 创建: `include/pgo/solver/newton_solver.hpp`
- 修改: `tests/solver/test_solver.cpp`

- [ ] **Step 1: 实现 Newton solver**

行为：

- 输入 free-space energy 和 initial `free_u`。
- 每轮计算 value、gradient、Hessian。
- 解 `(H + lambda I) du = -g`，使用 Eigen `SimplicialLDLT`。
- factorization 失败或 direction 不是 descent 时增加 diagonal regularization。
- 使用 Armijo line search。
- gradient norm 小于 tolerance 或达到 max iterations 时停止。

- [ ] **Step 2: 添加 quadratic convergence 测试**

使用 positive definite quadratic energy，验证 Newton 收敛到解析 minimizer。

- [ ] **Step 3: 添加 mass-spring smoke test**

创建三点 chain，固定 vertex 0，对末端施加简单 external force energy，求解 reduced energy，验证末端 displacement 朝 force 方向。

- [ ] **Step 4: 运行测试**

```bash
cmake --build --preset debug
ctest --preset debug -R solver
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
```

执行流程：

- 读取 OBJ 为 rest mesh。
- 从 `mesh.edge_indices()` / `mesh.edge_vertex()` 构造 mass-spring energy。
- 固定 top-row vertices 的 displacement DOFs 为 0。
- 每一帧将 gravity 从 0 ramp 到目标值。
- 求解 quasi-static equilibrium。
- 使用 `ObjFrameWriter` 写出 OBJ frame。

- [ ] **Step 2: 构建并运行 example**

```bash
cmake --build --preset debug --target pgo_mass_spring_cloth
./build/debug/examples/pgo_mass_spring_cloth --input examples/assets/cloth_grid.obj --output frames --frames 5
```

期望：生成 `frames/frame_0000.obj` 到 `frames/frame_0004.obj`。


### Task 5.3: 添加 OBJ frames -> Alembic Python tool

**文件:**
- 创建: `tools/obj_frames_to_abc.py`

- [ ] **Step 1: 实现工具**

行为：

- 接收 `--frames-dir`、`--output`、`--fps`。
- 按字典序读取 `frame_*.obj`。
- 要求所有 frame topology 一致。
- 在 `main()` 内部 import Alembic Python binding。
- Alembic import 失败时打印明确安装提示并返回 exit code `2`。
- 成功时写出 animated polymesh `.abc`。

- [ ] **Step 2: 测试 Alembic 缺失路径**

```bash
python tools/obj_frames_to_abc.py --frames-dir frames --output cloth.abc
```

没有 Alembic binding 时，期望返回 code `2` 并打印清晰提示。


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
- Energy model 暴露 local contribution API，适合 CPU assembly 和未来 GPU dispatch。
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

- 覆盖范围：计划覆盖 build system、CI、Eigen acceleration 配置、C++23 header-oriented core、Eigen backend layer、GPU-aware storage、rest/displacement 分离、DOF reduction、OBJ input/output、local energy assembly、mass-spring energy、Newton solver、example frames、C99 dynamic-library API、Alembic 后处理。
- 占位扫描：计划不包含 `TBD`、`TODO`、`implement later` 等未落实占位。
- 类型一致性：核心名称统一使用 `RestMesh`、`DofLayout`、`Displacement`、`DirichletBoundary`、`ReducedDofMap`、`MassSpringEnergy`、`ReducedEnergyView`、`ObjFrameWriter`、`NewtonSolver`、`pgo_world_t`、`pgo_error_t`。
- 范围控制：Vulkan、Slang、FEM、contact、IPC、GPU solvers 不进入 Milestone 1，但数据布局和 API 边界保持兼容。
