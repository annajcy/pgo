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
- Eigen 只通过 `pgo::math` facade 用于 CPU 数值计算和 sparse solve；`geometry/` 与 `storage/` 不存储 `Eigen::Vector3d`、`Eigen::MatrixXd` 等对象。
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
        macros.hpp
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
        finite_difference.hpp
        sparse.hpp
        types.hpp
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
    test_c_api.c
    test_dof.cpp
    test_finite_difference.cpp
    test_mass_spring_energy.cpp
    test_obj_io.cpp
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

- [x] **Step 3: 提交**

```bash
git add .gitignore
git commit -m "chore: initialize repository ignores"
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

- [x] **Step 5: 提交**

```bash
git add conanfile.py conan/profiles
git commit -m "build: add repository conan profile"
```

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

- [x] **Step 7: 提交**

```bash
git add CMakeLists.txt CMakePresets.json cmake
git commit -m "build: add cmake project skeleton"
```

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

- [x] **Step 2: 提交**

```bash
git add .clang-format
git commit -m "style: add clang-format configuration"
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
- `ubuntu-latest` ASan/UBSan

Phase 0 CI 只做 configure/build。等 Phase 1 创建 `tests/CMakeLists.txt` 后，Phase 7 再把 `ctest` 加回 CI。

- [x] **Step 3: 提交**

```bash
git add README.md .github/workflows/ci.yml
git commit -m "build: update CI configuration and add Conan profiles"
```

## Phase 1: Base、Math、Storage、Geometry、DOF 核心

### Task 1.1: 添加基础 assert/macros

**文件:**
- 创建: `include/pgo/base/assert.hpp`
- 创建: `include/pgo/base/macros.hpp`

- [ ] **Step 1: 实现 `pgo::base::require`**

`require(condition, message)` 在 condition 为 false 时抛出 `std::runtime_error`。这是 C++ core 内部使用的错误机制，不能越过 C API 边界。

- [ ] **Step 2: 添加 `PGO_NODISCARD` macro**

`include/pgo/base/macros.hpp` 定义：

```cpp
#define PGO_NODISCARD [[nodiscard]]
```

- [ ] **Step 3: 提交**

```bash
git add include/pgo/base
git commit -m "feat: add base utilities"
```

### Task 1.2: 添加 Eigen math facade

**文件:**
- 创建: `include/pgo/math/types.hpp`
- 创建: `include/pgo/math/sparse.hpp`

- [ ] **Step 1: 定义基础类型别名**

`types.hpp` 提供：

```cpp
namespace pgo::math {
template <class T>
concept Scalar = std::floating_point<T>;

template <Scalar T, int Dim>
using Vec = Eigen::Matrix<T, Dim, 1>;

template <Scalar T, int Rows, int Cols>
using Mat = Eigen::Matrix<T, Rows, Cols>;

template <Scalar T>
using DVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <Scalar T>
using DMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

using Index = std::uint32_t;
}
```

- [ ] **Step 2: 定义 sparse aliases**

`sparse.hpp` 提供：

```cpp
template <Scalar T>
using SparseMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;

template <Scalar T>
using Triplet = Eigen::Triplet<T>;
```

- [ ] **Step 3: 提交**

```bash
git add include/pgo/math
git commit -m "feat: add math type facade"
```

### Task 1.3: 添加 GPU-aware host storage primitive

**文件:**
- 创建: `include/pgo/storage/array_view.hpp`
- 创建: `include/pgo/storage/host_buffer.hpp`

- [ ] **Step 1: 定义 view/buffer**

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

- [ ] **Step 2: 提交**

```bash
git add include/pgo/storage
git commit -m "feat: add host storage primitives"
```

### Task 1.4: 添加 RestMesh 和 topology

**文件:**
- 创建: `include/pgo/geometry/topology.hpp`
- 创建: `include/pgo/geometry/rest_mesh.hpp`
- 创建/修改: `tests/CMakeLists.txt`
- 创建/修改: `tests/test_dof.cpp`

- [ ] **Step 1: 定义 topology**

```cpp
namespace pgo::geometry {
using VertexIndex = pgo::math::Index;
using Edge = std::array<VertexIndex, 2>;
using Face = std::array<VertexIndex, 3>;
}
```

- [ ] **Step 2: 定义 `RestMesh<T, Dim>`**

要求：

- `rest_positions` 是 vertex-major flat buffer。
- `num_vertices() == rest_positions.size() / Dim`。
- `rest_position(i)` 返回 `math::Vec<T, Dim>`。
- 保存 `edges()` 和 `faces()`。
- 不在 storage 中保存 Eigen vector object。

- [ ] **Step 3: 建立测试 target**

`tests/CMakeLists.txt` 创建 `pgo_tests`，链接 `pgo::core` 和 `GTest::gtest_main`，并使用 `include(GoogleTest)` + `gtest_discover_tests(pgo_tests)`。

- [ ] **Step 4: 运行测试**

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

- [ ] **Step 5: 提交**

```bash
git add include/pgo/geometry tests
git commit -m "feat: add rest mesh topology"
```

### Task 1.5: 添加 DOF layout、displacement、Dirichlet boundary 和 reduced map

**文件:**
- 创建: `include/pgo/dof/dof_layout.hpp`
- 创建: `include/pgo/dof/displacement.hpp`
- 创建: `include/pgo/dof/dirichlet_boundary.hpp`
- 创建: `include/pgo/dof/reduced_dof_map.hpp`
- 修改: `tests/test_dof.cpp`

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

职责：保存 fixed full DOF values。默认固定值为 `0`。

- [ ] **Step 4: 实现 `ReducedDofMap<T>`**

职责：

- 从 full dofs 和 boundary 生成 `free_to_full` / `full_to_free`。
- `pack(full_u)` 得到 `free_u`。
- `unpack(free_u, boundary)` 得到 full `u`。

- [ ] **Step 5: 添加 DOF 测试**

测试内容：

- `DofLayout<3>(4)` 有 12 个 DOF。
- `layout.index(2, 1) == 7`。
- 固定 DOF 后，`ReducedDofMap` 能正确 pack/unpack。

- [ ] **Step 6: 运行测试并提交**

```bash
cmake --build --preset debug
ctest --preset debug -R dof
git add include/pgo/dof tests/test_dof.cpp
git commit -m "feat: add displacement dof mapping"
```

## Phase 2: OBJ Mesh 输入和 OBJ Frame 输出

### Task 2.1: 添加 OBJ reader

**文件:**
- 创建: `include/pgo/io/obj_reader.hpp`
- 修改: `tests/test_obj_io.cpp`

- [ ] **Step 1: 实现 `read_obj_rest_mesh<T, Dim>(path)`**

支持：

- `v x y z`
- `l i j`
- triangular `f i j k`
- `f` 中的 `i/j/k` 或 `i//k` token，只读取第一个 vertex index
- OBJ positive one-based indices

要求：

- Faces 自动抽取 undirected unique edges。
- 读取为 `RestMesh<T, Dim>`。
- `Dim == 2` 时丢弃 OBJ z 坐标。

- [ ] **Step 2: 添加测试**

创建临时 OBJ：4 个 vertices、2 个 triangles。读取后断言：

- `num_vertices() == 4`
- `num_faces() == 2`
- `num_edges() == 5`

- [ ] **Step 3: 运行测试并提交**

```bash
cmake --build --preset debug
ctest --preset debug -R obj
git add include/pgo/io/obj_reader.hpp tests/test_obj_io.cpp
git commit -m "feat: add obj rest mesh reader"
```

### Task 2.2: 添加 OBJ frame writer

**文件:**
- 创建: `include/pgo/io/obj_frame_writer.hpp`
- 修改: `tests/test_obj_io.cpp`

- [ ] **Step 1: 实现 `ObjFrameWriter<T, Dim>`**

职责：

- 输入 `RestMesh<T, Dim>` 和 full displacement vector `u`。
- 输出 `frame_0000.obj`、`frame_0001.obj` 等。
- 写出的 vertex position 是 `X + u`。
- 保留原 faces。
- 如果 mesh 没有 faces，则写 `l` edges。

- [ ] **Step 2: 添加 writer 测试**

创建两点 line mesh，设置第二个点的 displacement，写出 frame 后检查 OBJ 文本包含 displaced vertex。

- [ ] **Step 3: 运行测试并提交**

```bash
cmake --build --preset debug
ctest --preset debug -R obj
git add include/pgo/io/obj_frame_writer.hpp tests/test_obj_io.cpp
git commit -m "feat: add obj frame writer"
```

## Phase 3: Local-Contribution Energy 和 CPU Assembly

### Task 3.1: 添加 energy concepts 和 local matrix types

**文件:**
- 创建: `include/pgo/assembly/local_matrix.hpp`
- 创建: `include/pgo/energy/energy_concepts.hpp`

- [ ] **Step 1: 定义 local aliases**

```cpp
template <pgo::math::Scalar T>
using LocalVector = pgo::math::DVec<T>;

template <pgo::math::Scalar T>
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

- [ ] **Step 3: 提交**

```bash
git add include/pgo/assembly/local_matrix.hpp include/pgo/energy/energy_concepts.hpp
git commit -m "feat: define energy concepts"
```

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

- [ ] **Step 2: 提交**

```bash
git add include/pgo/assembly/cpu_assembler.hpp
git commit -m "feat: add cpu local energy assembler"
```

### Task 3.3: 添加 MassSpringEnergy

**文件:**
- 创建: `include/pgo/energy/mass_spring_energy.hpp`
- 修改: `tests/test_mass_spring_energy.cpp`

- [ ] **Step 1: 实现 `MassSpringEnergy<T, Dim>`**

单条 spring 的公式：

```text
E_e(u) = 0.5 * k * (||x_i - x_j|| - L)^2
x_i = X_i + u_i
x_j = X_j + u_j
L = ||X_i - X_j||
```

要求：

- `local_count()` 等于 edge 数量。
- `local_dofs(edge_id, dofs)` 返回 `[i0, i1, ..., j0, j1, ...]` 对应的 full displacement DOFs。
- `local_gradient` 和 `local_hessian` 使用解析导数。
- 对零长度或极短 current edge 做 epsilon clamp，避免 NaN。

- [ ] **Step 2: 添加 derivative tests**

测试：

- rest state energy 为 0。
- rest state gradient 为 0。
- 拉伸一个端点后 energy 增大。
- finite difference gradient 与解析 gradient 匹配。
- finite difference Hessian 与解析 Hessian 匹配。

- [ ] **Step 3: 运行测试并提交**

```bash
cmake --build --preset debug
ctest --preset debug -R mass_spring
git add include/pgo/energy/mass_spring_energy.hpp tests/test_mass_spring_energy.cpp
git commit -m "feat: add mass spring energy"
```

### Task 3.4: 添加 finite difference helper

**文件:**
- 创建: `include/pgo/math/finite_difference.hpp`
- 修改: `tests/test_finite_difference.cpp`

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

- [ ] **Step 3: 运行测试并提交**

```bash
cmake --build --preset debug
ctest --preset debug -R finite
git add include/pgo/math/finite_difference.hpp tests/test_finite_difference.cpp
git commit -m "test: add finite difference derivative checks"
```

## Phase 4: Reduced Energy 和 Newton Solver

### Task 4.1: 添加 EnergySum 和 ReducedEnergyView

**文件:**
- 创建: `include/pgo/energy/energy_sum.hpp`
- 创建: `include/pgo/energy/reduced_energy.hpp`
- 修改: `tests/test_solver.cpp`

- [ ] **Step 1: 实现 `EnergySum<T>`**

组合多个 full-space energies：

- value 相加。
- gradient 相加。
- Hessian 相加。

- [ ] **Step 2: 实现 `ReducedEnergyView<T, Energy>`**

职责：

```text
free_u -> unpack 成 full_u
full energy evaluate
full gradient/Hessian reduce 到 free DOFs
```

- [ ] **Step 3: 添加 reduced energy 测试**

使用 quadratic energy：

```text
E(u) = 0.5 * u^T A u - b^T u
```

固定一个 DOF，验证 reduced gradient/Hessian 等于选取 free rows/cols。

- [ ] **Step 4: 运行测试并提交**

```bash
cmake --build --preset debug
ctest --preset debug -R solver
git add include/pgo/energy/energy_sum.hpp include/pgo/energy/reduced_energy.hpp tests/test_solver.cpp
git commit -m "feat: add reduced energy view"
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

- [ ] **Step 3: 提交**

```bash
git add include/pgo/solver/solver_result.hpp include/pgo/solver/line_search.hpp
git commit -m "feat: add solver result and line search"
```

### Task 4.3: 添加 damped Newton solver

**文件:**
- 创建: `include/pgo/solver/newton_solver.hpp`
- 修改: `tests/test_solver.cpp`

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

- [ ] **Step 4: 运行测试并提交**

```bash
cmake --build --preset debug
ctest --preset debug -R solver
git add include/pgo/solver/newton_solver.hpp tests/test_solver.cpp
git commit -m "feat: add damped newton solver"
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
- mesh edges 可以从 faces 自动抽取。

- [ ] **Step 3: 提交**

```bash
git add examples/CMakeLists.txt examples/assets/cloth_grid.obj
git commit -m "feat: add mass spring example asset"
```

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
- 从 mesh edges 构造 mass-spring energy。
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

- [ ] **Step 3: 提交**

```bash
git add examples/mass_spring_cloth.cpp
git commit -m "feat: add mass spring cloth example"
```

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

- [ ] **Step 3: 提交**

```bash
git add tools/obj_frames_to_abc.py
git commit -m "tool: add obj frames to alembic converter"
```

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

- [ ] **Step 3: 提交**

```bash
git add include/pgo_c
git commit -m "feat: add public c api header"
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

- [ ] **Step 4: 提交**

```bash
git add src/c_api/CMakeLists.txt src/c_api/pgo_c.cpp
git commit -m "build: add c api shared library target"
```

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

- [ ] **Step 3: 提交**

```bash
git add src/c_api/pgo_c.cpp
git commit -m "feat: implement c api bridge"
```

### Task 6.4: 添加纯 C API smoke test

**文件:**
- 创建: `tests/test_c_api.c`
- 修改: `tests/CMakeLists.txt`

- [ ] **Step 1: 创建 `tests/test_c_api.c`**

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
    add_executable(pgo_c_api_tests test_c_api.c)
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

- [ ] **Step 5: 提交**

```bash
git add tests/test_c_api.c tests/CMakeLists.txt
git commit -m "test: add pure c api smoke test"
```

## Phase 7: CI 和验证

### Task 7.1: 升级 GitHub Actions CI 为 build/test

**文件:**
- 修改: `.github/workflows/ci.yml`

- [ ] **Step 1: 在 Phase 0 build-only CI 基础上加入测试步骤**

CI 至少包含：

- `ubuntu-latest` Debug build/test。
- `macos-latest` Debug build/test。
- `ubuntu-latest` ASan/UBSan build/test。

每个 job 执行：

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

ASan job 使用：

```bash
cmake --preset asan
cmake --build --preset asan
ctest --preset asan
```

要求：Phase 0 已经存在 `build-debug` 和 `sanitize` jobs；本任务只是在 tests/examples/C API targets 存在后恢复 `ctest`，不要退回到本机 Conan default profile。

- [ ] **Step 2: 提交**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add build and sanitizer workflow"
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

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs: add milestone 1 build instructions"
```

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

- [ ] **Step 5: 提交**

```bash
git add README.md
git commit -m "docs: record gpu-aware architecture constraints"
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

- [ ] **Step 5: 提交**

```bash
git add README.md
git commit -m "docs: record c abi boundary constraints"
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
- GitHub Actions CPU build/test jobs 通过。

## 推荐实现顺序

1. 完成 Phase 0，先让构建和 CI 骨架站起来。
2. 完成 Phase 1，建立 flat storage、rest mesh、DOF/reduced DOF。
3. 完成 Phase 2，尽早获得可视化输出能力。
4. 完成 Phase 3，用 derivative tests 保护 energy 实现。
5. 完成 Phase 4，先用 quadratic system 验证 solver，再跑 mass-spring。
6. 完成 Phase 5，生成 OBJ frames。
7. 完成 Phase 6，暴露稳定的 C99 shared-library facade。
8. 完成 Phase 7 和 Phase 8 后，再进入 Milestone 2 GPU backend。

## 自检记录

- 覆盖范围：计划覆盖 build system、CI、C++23 header-oriented core、Eigen facade、GPU-aware storage、rest/displacement 分离、DOF reduction、OBJ input/output、local energy assembly、mass-spring energy、Newton solver、example frames、C99 dynamic-library API、Alembic 后处理。
- 占位扫描：计划不包含 `TBD`、`TODO`、`implement later` 等未落实占位。
- 类型一致性：核心名称统一使用 `RestMesh`、`DofLayout`、`Displacement`、`DirichletBoundary`、`ReducedDofMap`、`MassSpringEnergy`、`ReducedEnergyView`、`ObjFrameWriter`、`NewtonSolver`、`pgo_world_t`、`pgo_error_t`。
- 范围控制：Vulkan、Slang、FEM、contact、IPC、GPU solvers 不进入 Milestone 1，但数据布局和 API 边界保持兼容。
