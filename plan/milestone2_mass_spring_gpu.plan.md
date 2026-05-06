# GPU Mass Spring Vulkan/Slang 实现计划草案

> **给 agentic workers:** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务执行。本计划使用 checkbox (`- [ ]`) 语法追踪进度。

**目标:** 完成 Milestone 2：在 Milestone 1 CPU mass-spring core 之上实现一个 Vulkan compute + Slang 的 GPU mass-spring solver 原型，包含 matrix-free Hessian-vector product、GPU PCG、基础 preconditioner、CPU/GPU 对照测试和 OBJ frame 输出。

**架构:** Milestone 2 不把 GPU 化做成 `MathBackend` 替换。`MathBackend` 继续只是 CPU 数值类型边界；GPU 化替换的是 assembly/operator、linear solver、preconditioner、reduction 和 device storage。Vulkan RHI 不在 PGO 仓库里从零重写，而是在 `/Users/jinceyang/Desktop/codebase/rtr2` 中正式分成 `rtr::rhi_core` 与 `rtr::rhi_graphics` 两层：core 是 headless Vulkan 基础和 compute 能力，graphics 依赖 core 并提供完整图形栈。PGO 只消费 `rtr::rhi_core`，并在其上实现 mass-spring 数值 backend。整体路线以 Stiff-GIPC 为标杆，但先做 mass-spring 版本的最小闭环：edge-local contribution -> vertex gather -> PCG -> line search/newton step。

**技术栈:** C++23、RTR2 RHI core library、Vulkan compute、Slang、CMake、Conan 2、Eigen CPU reference、GoogleTest、OBJ frame pipeline。

---

## 0. 核心设计决策

- GPU backend 不实现为 `VulkanMathBackend`。
- GPU backend 新增 `gpu/`、`linear_system/`、`assembly/gpu_*` 层，替换执行热路径，而不是替换 `pgo::math::Vec` / `SparseMat` aliases。
- Milestone 2 默认 GPU scalar 使用 `float32`。`float64` 在 Vulkan 上依赖 `shaderFloat64`，不作为跨平台默认要求。
- CPU reference 继续用 `double`，GPU 测试用 relaxed tolerance 对比 CPU 结果。
- 第一版 PCG 由 CPU 控制迭代：每轮 GPU dispatch 后把少量 reduction scalar 拷回 CPU 判断收敛。
- 第一版不依赖 float atomic add。edge-local kernel 先写 per-edge contribution buffer，再由 vertex gather kernel 汇总到 vertex DOF。
- 第一版优先 matrix-free operator，不强制组装全局 sparse matrix。
- 第一版 preconditioner 做 diagonal / block-diagonal；MAS / Schwarz 作为后续扩展，不进入 Milestone 2 完成标准。
- Dirichlet boundary 在 GPU 侧先用 full-space active DOF mask 表达：fixed DOF 的 residual/search direction/update 都置零。
- C API 暂不强制暴露 GPU backend；Milestone 2 先通过 C++ example 和 tests 验证。
- Vulkan RHI 参考 `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/rhi` 与 `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/system/render` 的设计形状，并且 `rtr::rhi_core` / `rtr::rhi_graphics` 分层工作在 `/Users/jinceyang/Desktop/codebase/rtr2` 仓库中完成和验证。
- 允许参考 RTR2 的 `Device`/`PhysicalDeviceSelector`、`Buffer`、`CommandPool`/`CommandBuffer`、`DescriptorSetLayout`/`DescriptorPool`/`DescriptorWriter`、`ShaderModule`、Slang-to-SPIR-V compiler 和 shadertoy compute pass 的 dispatch 组织。
- `rtr::rhi_core` 不引入 RTR2 的 window、swap chain、imgui、render pass、frame scheduler、present pass、forward pipeline 等图形渲染层；这些属于 `rtr::rhi_graphics`。
- `rtr::rhi_graphics` 依赖 `rtr::rhi_core`。RTR2 自己的图形应用链接 graphics target，PGO 链接 core target。
- PGO 仓库不复制 RTR2 的低层 Vulkan RAII 文件，也不把 RHI 作为 PGO submodule。PGO 只依赖 CMake target：superproject 已经提供 `rtr::rhi_core` 时直接使用，否则通过 `find_package(rtr_rhi CONFIG REQUIRED)` 消费 RTR2 install/export 出来的 package。

## 1. 目标目录结构

```text
include/
  pgo/
    gpu/
      gpu_options.hpp
      device_span.hpp
      rtr_rhi_core_context.hpp
    assembly/
      gpu_mass_spring_operator.hpp
      gpu_mass_spring_system.hpp
    linear_system/
      linear_operator.hpp
      pcg_result.hpp
      preconditioner.hpp
      vulkan_pcg_solver.hpp
      vulkan_pcg_state.hpp
      vulkan_diagonal_preconditioner.hpp
    solver/
      gpu_newton_solver.hpp
shaders/
  pgo/
    common.slang
    vector_ops.slang
    reductions.slang
    mass_spring_operator.slang
    pcg.slang
src/
  gpu/
    CMakeLists.txt
    rtr_rhi_core_context.cpp
examples/
  gpu_mass_spring_cloth.cpp
tests/
  gpu/
    vulkan/
      test_device.cpp
      test_vector_ops.cpp
      test_reductions.cpp
      test_mass_spring_operator.cpp
      test_vulkan_pcg.cpp
      test_gpu_mass_spring_solver.cpp
```

## Phase 0: RTR2 RHI core/graphics 分层和 PGO dependency boundary

### Task 0.0: 在 RTR2 仓库拆分 RHI core 和 RHI graphics

**工作仓库:**
- `/Users/jinceyang/Desktop/codebase/rtr2`

**参考文件:**
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/rhi/device.hpp`
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/rhi/buffer.hpp`
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/rhi/command.hpp`
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/rhi/descriptor.hpp`
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/rhi/shader_module.hpp`
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/system/render/utils/shader_compiler.hpp`
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/system/render/pipeline/shadertoy/shadertoy_compute_pass.hpp`

**目标产物:**

```text
rtr::rhi_core
rtr::rhi_graphics
```

`rtr::rhi_core` 是可被 PGO 消费的 headless Vulkan target，`rtr::rhi_graphics` 是 RTR2 自己的完整图形栈 target，并且依赖 `rtr::rhi_core`。

PGO 最好可以通过以下方式接入：

```cmake
if(NOT TARGET rtr::rhi_core)
    find_package(rtr_rhi CONFIG REQUIRED)
endif()

target_link_libraries(pgo_gpu PRIVATE rtr::rhi_core)
```

本地 PGO 开发阶段通过 `CMAKE_PREFIX_PATH=/Users/jinceyang/Desktop/codebase/rtr2/install` 指向 RTR2 安装目录。RTR2 作为 superproject 消费 PGO 时，则先 `add_subdirectory(external/rtr-rhi 或 rtr2 内部 rhi)` 让 `rtr::rhi_core` target 已经存在，再 `add_subdirectory(external/pgo)`，避免 RHI 源码出现两份。

- [ ] **Step 1: 在 RTR2 中定义 `rtr::rhi_core` public surface**

`rtr::rhi_core` 保留：

```text
PhysicalDeviceSelector
Device
Buffer
CommandPool
CommandBuffer
DescriptorSetLayout
DescriptorPool
DescriptorWriter
ShaderModule
ComputePipeline
Slang shader compiler
```

`rtr::rhi_core` 不暴露：

```text
Window
SwapChain
ImguiContext
RenderPass
FrameContext
FrameScheduler
PresentPass
ForwardPipeline
Shadertoy image output pipeline
```

`Image` / `Texture` / `ImageReadback` 是否放入 core 需要谨慎：如果只服务 swapchain/present/rendering，放 graphics；如果作为 storage image compute primitive 被保留，也必须不依赖 window/swapchain/render pass。Milestone 2 的 PGO 不依赖 image API。

- [ ] **Step 2: 在 RTR2 中定义 `rtr::rhi_graphics` target**

`rtr::rhi_graphics` 依赖：

```text
rtr::rhi_core
```

`rtr::rhi_graphics` 拥有：

```text
Window
Surface
SwapChain
Texture/Image for rendering
ImageReadback for rendering/debugging
ImguiContext
Mesh/DynamicMesh rendering helpers
RenderPass
FrameContext
FrameScheduler
PresentPass
ForwardPipeline
Shadertoy image output pipeline
```

设计约束：

```text
rtr::rhi_core -> no dependency on rtr::rhi_graphics
rtr::rhi_graphics -> depends on rtr::rhi_core
PGO -> depends only on rtr::rhi_core
```

- [ ] **Step 3: 在 RTR2 中添加 headless core tests**

必须验证：

```text
create compute device
create host-visible buffer
create device-local storage buffer
upload/download buffer
compile Slang compute shader to SPIR-V
create compute pipeline
bind storage buffer descriptor
dispatch a simple vector kernel
download and verify result
```

- [ ] **Step 4: 在 RTR2 中验证 graphics 继续可用**

RTR2 原有 window/swapchain/present/forward/shadertoy graphics pipeline 应改为链接：

```text
rtr::rhi_graphics
```

期望：原有图形 demo 或渲染测试仍能 configure/build/run。

- [ ] **Step 5: 在 RTR2 中导出可消费 package**

期望安装后存在：

```text
<install-prefix>/lib/cmake/rtr_rhi/rtr_rhiConfig.cmake
<install-prefix>/include/...
<install-prefix>/lib/...
```

CMake target 名称固定为：

```text
rtr::rhi_core
rtr::rhi_graphics
```

- [ ] **Step 6: 在 RTR2 中验证 install-tree consumption**

在 RTR2 仓库中添加两个最小 external consumer 测试：

```text
headless consumer: find_package(rtr_rhi CONFIG REQUIRED), link rtr::rhi_core, create device, run simple compute shader.
graphics consumer: find_package(rtr_rhi CONFIG REQUIRED), link rtr::rhi_graphics, create minimal graphics app or smoke target.
```

### Task 0.1: 添加 GPU build option

**文件:**
- 修改: `cmake/pgo_options.cmake`
- 修改: `CMakeLists.txt`

- [ ] **Step 1: 添加 option**

```cmake
option(PGO_BUILD_GPU "Build Vulkan/Slang GPU backend" OFF)
```

- [ ] **Step 2: 只在 option 开启且目录存在时加入 GPU target**

```cmake
if(PGO_BUILD_GPU AND EXISTS "${PROJECT_SOURCE_DIR}/src/gpu/CMakeLists.txt")
    add_subdirectory(src/gpu)
endif()
```

- [ ] **Step 3: 验证默认构建不需要 Vulkan/Slang**

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

期望：`PGO_BUILD_GPU=OFF` 时，普通 CPU build/test 不受 GPU SDK 影响。

### Task 0.2: 添加 RTR2 RHI core target discovery

**文件:**
- 创建: `cmake/pgo_gpu_dependencies.cmake`
- 修改: `CMakeLists.txt`

- [ ] **Step 1: 优先使用 superproject 已有 target**

```cmake
if(NOT TARGET rtr::rhi_core)
    find_package(rtr_rhi CONFIG REQUIRED)
endif()
```

- [ ] **Step 2: 不在 PGO 中直接查找 Vulkan/Slang 或 vendored RHI 源码**

Vulkan 和 Slang 是 `rtr::rhi_core` 的 implementation dependency。PGO 只依赖 `rtr::rhi_core` 的 public target。如果 RTR2 package 需要向 consumer 传递 Vulkan headers 或 Slang runtime，应由 `rtr_rhiConfig.cmake` 处理。

PGO 不做：

```cmake
add_subdirectory(external/rtr-rhi)
```

这样 RTR2 作为 superproject 时不会出现：

```text
rtr2/external/rtr-rhi
rtr2/external/pgo/external/rtr-rhi
```

- [ ] **Step 3: GPU target 链接 RTR2 RHI core**

`src/gpu/CMakeLists.txt` 中：

```cmake
add_library(pgo_gpu STATIC
    rtr_rhi_core_context.cpp
)

add_library(pgo::gpu ALIAS pgo_gpu)

target_link_libraries(pgo_gpu
    PUBLIC
        pgo::core
        rtr::rhi_core
)
```

- [ ] **Step 4: 记录 PGO standalone 开发命令**

先在 RTR2 安装 RHI package：

```bash
cmake --preset release
cmake --build --preset release
cmake --install build/release --prefix /Users/jinceyang/Desktop/codebase/rtr2/install
```

再在 PGO 中消费：

```bash
cmake --preset debug \
  -D PGO_BUILD_GPU=ON \
  -D CMAKE_PREFIX_PATH=/Users/jinceyang/Desktop/codebase/rtr2/install
cmake --build --preset debug
```

- [ ] **Step 5: GPU target 只在 `PGO_BUILD_GPU=ON` 时要求依赖**

```cmake
if(PGO_BUILD_GPU)
    include(cmake/pgo_gpu_dependencies.cmake)
endif()
```

## Phase 1: PGO GPU adapter over RTR2 RHI core

### Task 1.0: 锁定 PGO 与 RTR2 RHI core 的责任边界

**参考文件:**
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/rhi/device.hpp`
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/rhi/buffer.hpp`
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/rhi/command.hpp`
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/rhi/descriptor.hpp`
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/rhi/shader_module.hpp`
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/system/render/utils/shader_compiler.hpp`
- 参考: `/Users/jinceyang/Desktop/codebase/rtr2/src/rtr/system/render/pipeline/shadertoy/shadertoy_compute_pass.hpp`

**`rtr::rhi_core` 保留:**

```text
PhysicalDeviceSelector: 只需要 compute queue 和可选 shaderFloat64 feature check。
Device: instance / physical device / logical device / compute queue。
Buffer: host-visible buffer、device-local buffer、staging upload/download。
CommandPool + CommandBuffer: one-time compute command recording/submission。
DescriptorSetLayout + DescriptorPool + DescriptorWriter: storage buffer 和少量 uniform/push constants。
ShaderModule: 从 Slang 编译结果创建 compute shader module。
ComputePipeline: descriptor layout + pipeline layout + compute pipeline + dispatch。
Slang compiler: 编译单个 compute entry point 到 SPIR-V。
```

**`rtr::rhi_core` 不引入:**

```text
Window
SwapChain
Texture/Image
ImageReadback
ImguiContext
Mesh/DynamicMesh rendering helpers
RenderPass
FrameContext
FrameScheduler
PresentPass
ForwardPipeline
Shadertoy image output pipeline
```

设计约束：RHI 只服务数值 compute，不承担渲染生命周期。`CommandContext` 可以借鉴 RTR2 的 `CommandPool`/`CommandBuffer`，但 API 面向 single compute queue + storage buffer barrier。

`rtr::rhi_graphics` 承担完整 Vulkan 图形栈，并依赖 `rtr::rhi_core`。PGO 不依赖 graphics target。

**PGO 只负责:**

```text
GPU mass-spring buffer layout
shader dispatch sequence for vector ops/reductions/operator/PCG
CPU/GPU validation tests
solver/operator/preconditioner abstractions
```

**RTR2 RHI core 负责:**

```text
Vulkan instance/device/queue
buffer allocation and staging
command recording/submission
descriptor set layout/pool/write
shader compilation and shader module creation
compute pipeline creation
```

### Task 1.1: 添加 GPU options 和 device span

**文件:**
- 创建: `include/pgo/gpu/gpu_options.hpp`
- 创建: `include/pgo/gpu/device_span.hpp`
- 创建: `include/pgo/gpu/rtr_rhi_core_context.hpp`
- 创建: `src/gpu/rtr_rhi_core_context.cpp`
- 创建: `tests/gpu/vulkan/test_device.cpp`

- [ ] **Step 1: 定义 GPU options**

```cpp
namespace pgo::gpu {

enum class GpuScalarType {
    Float32,
    Float64,
};

struct GpuOptions {
    GpuScalarType scalar_type = GpuScalarType::Float32;
    bool enable_validation = true;
    bool require_float64 = false;
};

} // namespace pgo::gpu
```

- [ ] **Step 2: 定义 device span descriptor**

```cpp
namespace pgo::gpu {

template <class T>
struct DeviceSpan {
    T* data = nullptr;
    std::size_t size = 0;
};

template <class T>
struct ConstDeviceSpan {
    const T* data = nullptr;
    std::size_t size = 0;
};

} // namespace pgo::gpu
```

- [ ] **Step 3: 添加 PGO compute context wrapper**

`RtrRhiCoreContext` 是 PGO 对 `rtr::rhi_core` 的薄封装，只保存：

```text
rtr rhi core device/context
default command context
scratch buffer allocator policy
```

它不重新包装所有 Vulkan object，也不暴露 window/swapchain/image API。

- [ ] **Step 4: 添加 smoke test**

`tests/gpu/vulkan/test_device.cpp` 使用 `namespace pgo::gpu::vulkan::test`，测试 `GpuOptions` 默认 scalar 是 `Float32`。

### Task 1.2: 添加 RTR2 RHI core device/context smoke test

**文件:**
- 修改: `include/pgo/gpu/rtr_rhi_core_context.hpp`
- 修改: `src/gpu/rtr_rhi_core_context.cpp`
- 修改: `tests/gpu/vulkan/test_device.cpp`

- [ ] **Step 1: 通过 `rtr::rhi_core` 创建 compute device**

`RtrRhiCoreContext` 通过 RTR2 RHI core 创建：

```text
instance/device/compute queue
command pool/command buffer
```

PGO 侧只传递 `GpuOptions`。

- [ ] **Step 2: 暴露最小访问接口**

提供：

```text
device()
command_context()
submit_and_wait(...)
```

- [ ] **Step 3: 参考 RTR2 的最小化取舍**

RTR2 RHI core 只要求：

```text
require_api_version(VK_API_VERSION_1_2 或 1_3)
require_queue_flags(vk::QueueFlagBits::eCompute)
require shaderFloat64 only when GpuOptions::require_float64 == true
```

不要求 surface、swapchain extension、dynamic rendering、present queue。

- [ ] **Step 4: 添加 device skip test**

测试规则：

- 如果当前机器没有 Vulkan device，GPU tests 使用 `GTEST_SKIP()`。
- 如果有 Vulkan device，能创建 `VulkanDevice` 和 `CommandContext`。

## Phase 2: Device buffer、shader pipeline 和 Slang 编译

### Task 2.1: 添加 PGO device buffer vocabulary

**文件:**
- 修改: `include/pgo/gpu/rtr_rhi_core_context.hpp`
- 修改: `src/gpu/rtr_rhi_core_context.cpp`
- 修改: `tests/gpu/vulkan/test_device.cpp`

- [ ] **Step 1: 基于 RTR2 RHI core 创建 typed buffer helpers**

职责：

- typed storage buffer allocation。
- host upload/download。
- 提供 `device_span()` 和 `size()`。
- 低层 `VkBuffer` / `VkDeviceMemory` 生命周期归 RTR2 RHI core 管。

- [ ] **Step 2: 添加 upload/download 测试**

上传 `{1.0f, 2.0f, 3.0f}` 到 GPU buffer，再下载，验证值不变。

### Task 2.2: 添加 shader/pipeline consumption helpers

**文件:**
- 修改: `include/pgo/gpu/rtr_rhi_core_context.hpp`
- 修改: `src/gpu/rtr_rhi_core_context.cpp`
- 创建: `shaders/pgo/common.slang`

- [ ] **Step 1: 使用 RTR2 RHI core descriptor helpers**

PGO shader dispatch 只使用 RTR2 RHI core 暴露的：

```text
add_storage_buffer(binding)
add_uniform_buffer(binding)
write_storage_buffer(binding, buffer, offset, range)
write_uniform_buffer(binding, buffer, offset, range)
```

不要在 PGO 中新增 image/sampler descriptor API。

- [ ] **Step 2: 使用 RTR2 RHI core 编译 Slang 到 SPIR-V**

输入：

- shader path
- entry point
- target profile，例如 `spirv_1_5`
- expected stage 固定为 compute

输出：

- SPIR-V bytecode
- diagnostics
- 编译得到的 Vulkan entry point 名称，默认 `"main"`

- [ ] **Step 3: 使用 RTR2 RHI core 创建 compute pipeline**

`ComputePipeline` 负责：

- descriptor set layout。
- pipeline layout。
- compute pipeline。
- bind + dispatch。
- dispatch 顺序参考 RTR2 shadertoy `ComputePass` 的 `bindPipeline`、`bindDescriptorSets`、`dispatch`，但 PGO 只操作 storage buffer，不处理 image layout transition。

- [ ] **Step 4: 添加 shader compile smoke test**

编译一个空写 buffer kernel，验证 pipeline 创建成功。

## Phase 3: GPU vector ops 和 reductions

### Task 3.1: 添加 vector ops kernels

**文件:**
- 创建: `shaders/pgo/vector_ops.slang`
- 创建: `tests/gpu/vulkan/test_vector_ops.cpp`

- [ ] **Step 1: 实现 kernels**

必须提供：

```text
clear_vector(x)
copy_vector(dst, src)
axpy(y, alpha, x)        // y = y + alpha * x
axpby(y, alpha, x, beta) // y = alpha * x + beta * y
mask_vector(x, active)   // fixed DOF set to 0
```

- [ ] **Step 2: 添加测试**

输入小数组：

```text
x = [1, 2, 3, 4]
y = [10, 20, 30, 40]
active = [1, 0, 1, 0]
```

验证 `axpy`、`axpby`、`mask_vector` 的结果。

### Task 3.2: 添加 reduction kernels

**文件:**
- 创建: `shaders/pgo/reductions.slang`
- 创建: `tests/gpu/vulkan/test_reductions.cpp`

- [ ] **Step 1: 实现 staged dot reduction**

第一阶段：

```text
partial_sums[workgroup_id] = sum_i a[i] * b[i]
```

第二阶段重复调用，直到 partial buffer 只剩一个值。

- [ ] **Step 2: 实现 norm reduction**

`norm2(x)` 复用 `dot(x, x)`。

- [ ] **Step 3: 添加测试**

测试：

```text
dot([1,2,3], [4,5,6]) == 32
norm2([3,4]) == 25
```

容忍误差：`1e-5f`。

## Phase 4: Matrix-free mass-spring operator

### Task 4.1: 添加 GPU mass-spring system descriptor

**文件:**
- 创建: `include/pgo/assembly/gpu_mass_spring_system.hpp`
- 创建: `tests/gpu/vulkan/test_mass_spring_operator.cpp`

- [ ] **Step 1: 定义 host-side GPU system builder**

从 Milestone 1 的 `geometry::RestMesh<T, Dim>` 构建：

```text
rest_positions_f32
edge_indices
edge_offsets
incident_edge_ids
incident_edge_local_vertices
active_dof_mask
spring_stiffness
```

设计要求：

- `edge_indices` 使用 flat `[i0, j0, i1, j1, ...]`。
- `edge_offsets[v]..edge_offsets[v+1]` 给出 vertex `v` 的 incident edges。
- `incident_edge_local_vertices` 是 0 或 1，用于说明当前 vertex 是 edge 的哪一端。
- 不使用 GPU float atomic add。

- [ ] **Step 2: 添加 adjacency 构建测试**

三点 chain，edges `(0,1), (1,2)`：

```text
edge_offsets = [0, 1, 3, 4]
vertex 0 incident = edge 0 local 0
vertex 1 incident = edge 0 local 1, edge 1 local 0
vertex 2 incident = edge 1 local 1
```

### Task 4.2: 添加 Hessian-vector product kernels

**文件:**
- 创建: `include/pgo/assembly/gpu_mass_spring_operator.hpp`
- 创建: `shaders/pgo/mass_spring_operator.slang`
- 修改: `tests/gpu/vulkan/test_mass_spring_operator.cpp`

- [ ] **Step 1: edge-local contribution kernel**

每条 spring 一个 work item：

```text
input: X, u, p, edge_indices, stiffness
output: edge_Ap_contrib[edge_id][2 * Dim]
```

语义：

```text
edge_Ap_contrib = H_edge(u) * [p_i, p_j]
```

- [ ] **Step 2: vertex gather kernel**

每个 vertex 一个 work item：

```text
Ap_vertex = sum incident edge contribution for this vertex
```

然后对 fixed DOF 应用 active mask。

- [ ] **Step 3: CPU/GPU operator 对照测试**

用 3 点 chain：

- CPU 侧用 Milestone 1 mass-spring analytic Hessian 计算 `H * p`。
- GPU 侧用 matrix-free operator 计算 `Ap`。
- 逐 DOF 对比，误差小于 `1e-4f`。

## Phase 5: GPU preconditioner

### Task 5.1: 添加 diagonal preconditioner

**文件:**
- 创建: `include/pgo/linear_system/preconditioner.hpp`
- 创建: `include/pgo/linear_system/vulkan_diagonal_preconditioner.hpp`
- 修改: `shaders/pgo/mass_spring_operator.slang`
- 修改: `tests/gpu/vulkan/test_mass_spring_operator.cpp`

- [ ] **Step 1: 定义 preconditioner concept**

```cpp
namespace pgo::linear_system {

template <class P>
concept GpuPreconditioner = requires(P p) {
    p.assemble();
    p.apply();
};

} // namespace pgo::linear_system
```

- [ ] **Step 2: 组装 diagonal**

edge-local kernel 输出每条 edge 对两个端点 DOF 的 diagonal contribution，vertex gather kernel 汇总到 `diag`。

- [ ] **Step 3: apply preconditioner**

```text
z[i] = active[i] ? r[i] / max(diag[i], epsilon) : 0
```

- [ ] **Step 4: 添加测试**

构造一个 diag buffer：

```text
diag = [2, 4, 8]
r = [2, 8, 24]
active = [1, 1, 0]
```

期望：

```text
z = [1, 2, 0]
```

## Phase 6: Vulkan PCG solver

### Task 6.1: 添加 linear operator 和 PCG state

**文件:**
- 创建: `include/pgo/linear_system/linear_operator.hpp`
- 创建: `include/pgo/linear_system/pcg_result.hpp`
- 创建: `include/pgo/linear_system/vulkan_pcg_state.hpp`
- 创建: `include/pgo/linear_system/vulkan_pcg_solver.hpp`

- [ ] **Step 1: 定义 operator interface**

```cpp
namespace pgo::linear_system {

struct PcgResult {
    bool converged = false;
    std::size_t iterations = 0;
    float initial_residual2 = 0.0f;
    float final_residual2 = 0.0f;
};

} // namespace pgo::linear_system
```

GPU operator 必须提供：

```text
apply(p, Ap)
```

preconditioner 必须提供：

```text
apply(r, z)
```

- [ ] **Step 2: 定义 PCG state buffers**

```text
x
r
z
p
Ap
partial_sums
```

### Task 6.2: 实现 CPU-controlled GPU PCG loop

**文件:**
- 修改: `include/pgo/linear_system/vulkan_pcg_solver.hpp`
- 创建: `shaders/pgo/pcg.slang`
- 创建: `tests/gpu/vulkan/test_vulkan_pcg.cpp`

- [ ] **Step 1: 初始化**

```text
x = 0
r = b
mask(r)
z = M^-1 r
p = z
rz0 = dot(r, z)
```

- [ ] **Step 2: 每轮迭代**

```text
Ap = A p
pAp = dot(p, Ap)
alpha = rz / pAp
x = x + alpha * p
r = r - alpha * Ap
mask(r)
z = M^-1 r
rz_new = dot(r, z)
beta = rz_new / rz
p = z + beta * p
mask(p)
rz = rz_new
```

- [ ] **Step 3: 收敛条件**

```text
rz <= tolerance * tolerance * rz0
```

默认：

```text
tolerance = 1e-4
max_iterations = min(1000, 2 * dof_count)
```

- [ ] **Step 4: 添加 PCG 测试**

先用一个 mock diagonal operator：

```text
A = diag([2, 4, 8])
b = [2, 8, 24]
```

期望：

```text
x ~= [1, 2, 3]
```

误差小于 `1e-4f`。

## Phase 7: GPU Newton / line search integration

### Task 7.1: 添加 GPU mass-spring solver

**文件:**
- 创建: `include/pgo/solver/gpu_newton_solver.hpp`
- 创建: `tests/gpu/vulkan/test_gpu_mass_spring_solver.cpp`

- [ ] **Step 1: GPU solver state**

持有：

```text
u
gradient
search_direction
trial_u
energy_partial_sums
active_dof_mask
mass_spring_operator
diagonal_preconditioner
vulkan_pcg_solver
```

- [ ] **Step 2: 每个 Newton iteration**

```text
compute energy and gradient on GPU
rhs = -gradient
mask(rhs)
assemble/apply diagonal preconditioner
solve H du = rhs using VulkanPcgSolver
line search:
    trial_u = u + alpha * du
    evaluate energy(trial_u)
    accept if sufficient decrease
```

- [ ] **Step 3: CPU-controlled line search**

第一版每次 line search evaluation 把 energy scalar 拷回 CPU。默认最多 12 次 backtracking。

- [ ] **Step 4: end-to-end 测试**

三点 chain：

- 固定 vertex 0。
- 对 vertex 2 施加外力。
- GPU solver 得到的 displacement 方向和 CPU solver 一致。
- GPU/CPU final displacement 相对误差小于 `1e-3f`。

## Phase 8: Example 和 frame pipeline

### Task 8.1: 添加 GPU cloth example

**文件:**
- 创建: `examples/gpu_mass_spring_cloth.cpp`
- 修改: `examples/CMakeLists.txt`

- [ ] **Step 1: CLI 参数**

支持：

```text
--input examples/assets/cloth_grid.obj
--frames-dir frames_gpu
--frames 5
--stiffness 1000
--gravity -9.8
--pcg-tolerance 1e-4
--gpu-device 0
```

- [ ] **Step 2: 执行流程**

```text
read OBJ rest mesh
build GPU mass-spring system
fix top-row vertices
for each frame:
    update external force
    solve GPU quasi-static state
    download u
    write OBJ frame using ObjFrameWriter
```

- [ ] **Step 3: 运行 example**

```bash
cmake --build --preset debug --target pgo_gpu_mass_spring_cloth
./build/debug/examples/pgo_gpu_mass_spring_cloth \
  --input examples/assets/cloth_grid.obj \
  --frames-dir frames_gpu \
  --frames 5
```

期望：生成 `frames_gpu/frame_0000.obj` 到 `frames_gpu/frame_0004.obj`。

## Phase 9: Tests、CI 和文档

### Task 9.1: 添加 GPU test gating

**文件:**
- 修改: `tests/CMakeLists.txt`
- 修改: `.github/workflows/ci.yml`

- [ ] **Step 1: GPU tests 只在 `PGO_BUILD_GPU=ON` 时编译**

```cmake
if(PGO_BUILD_GPU AND TARGET pgo::gpu)
    add_executable(pgo_gpu_tests
        gpu/vulkan/test_device.cpp
        gpu/vulkan/test_vector_ops.cpp
        gpu/vulkan/test_reductions.cpp
        gpu/vulkan/test_mass_spring_operator.cpp
        gpu/vulkan/test_vulkan_pcg.cpp
        gpu/vulkan/test_gpu_mass_spring_solver.cpp
    )
endif()
```

- [ ] **Step 2: CI 默认不要求 GPU**

普通 CI 继续跑 CPU build/test。GPU CI 作为可选 job，只有 runner 明确安装 Vulkan/Slang 并能提供 RTR2 `rtr::rhi_core` package 时开启。

### Task 9.2: 添加 GPU architecture note

**文件:**
- 修改: `README.md`
- 创建: `docs/gpu_backend.md`

- [ ] **Step 1: 记录 MathBackend 和 GPU backend 的边界**

必须写明：

```text
MathBackend is a CPU-side type vocabulary.
GPU acceleration replaces assembly/operator, reductions, linear solver,
preconditioner, and device storage. It is not modeled as VulkanMathBackend.
```

- [ ] **Step 2: 记录 Stiff-GIPC inspired roadmap**

```text
Milestone 2: matrix-free mass-spring operator + GPU PCG + diagonal/block preconditioner.
Future: BSR/BCOO assembled operator, block preconditioner, MAS/Schwarz preconditioner, GPU-side nonlinear loop.
```

## Milestone 2 完成标准

1. `PGO_BUILD_GPU=OFF` 时，CPU build/test 完全不依赖 Vulkan/Slang。
2. `/Users/jinceyang/Desktop/codebase/rtr2` 中已经拆分并验证 `rtr::rhi_core` / `rtr::rhi_graphics`，且 PGO 可通过既有 `rtr::rhi_core` target 或 `find_package(rtr_rhi CONFIG REQUIRED)` 消费它。
3. `PGO_BUILD_GPU=ON` 且机器有 Vulkan/Slang/RTR2 RHI core package 时，GPU tests 可以 configure/build/run。
4. GPU vector ops 和 reductions 通过测试。
5. GPU matrix-free mass-spring Hessian-vector product 与 CPU analytic reference 匹配。
6. GPU PCG 能解 diagonal SPD test。
7. GPU PCG 能解小型 mass-spring Hessian system，并与 CPU reference 匹配。
8. GPU mass-spring example 能输出 OBJ frame sequence。
9. 文档明确说明：GPU 化替换 solver/operator/preconditioner/reduction，不是替换 MathBackend。

## 后续 Milestone 3 候选方向

- BSR/BCOO assembled sparse operator。
- Block-diagonal preconditioner。
- MAS / additive Schwarz preconditioner。
- GPU-side line search loop，减少 CPU/GPU 同步。
- GPU contact candidate pipeline。
- IPC barrier energy 和 CCD。
- Slang shader autodiff experiment。

## 自检记录

- 覆盖范围：计划覆盖 RTR2 RHI core library 抽取、PGO dependency boundary、device buffers、Slang shader pipeline、vector ops、reductions、matrix-free mass-spring operator、diagonal preconditioner、GPU PCG、GPU Newton integration、example、tests、docs。
- 范围控制：不把 MAS、IPC contact、CCD、GPU-side nonlinear loop、BSR assembled sparse matrix 放入 Milestone 2 完成标准。
- 架构一致性：MathBackend 只保留 CPU type vocabulary 定位；GPU backend 通过 operator/solver/preconditioner/reduction/device storage 实现。
- 测试结构：测试目录与模块对齐，GPU 测试放在 `tests/gpu/vulkan/`。
