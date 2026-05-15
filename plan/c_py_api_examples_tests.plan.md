# C/Python API Examples And Test Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per current project convention, this plan intentionally contains no commit steps.

**Goal:** 补齐 C/Python API 的负路径测试与完整 mass-spring examples，让外部用户能用 C99 API 或 Python API 跑通 cloth grid 与 bunny OBJ simulation。

**Architecture:** C API examples 只 include `pgo_c/pgo.h`，不接触 C++ core headers；Python examples 只使用 public `pgo.World` API。测试分两层：C API 精确断言 `pgo_status_t` 和 `pgo_error_t`，Python API 断言异常与用户级行为。`size/version` ABI 字段本阶段不加入，只记录为 first stable C ABI 前的后续任务。

**Tech Stack:** C99, C++23 bridge, CMake/CTest, nanobind Python package, NumPy, pytest, OBJ frame output.

---

## 0. Design Decisions

- 暂不为 `pgo_mesh_view_t`、`pgo_mass_spring_params_t`、`pgo_solver_options_t`、`pgo_step_result_t` 加 `size/version` 字段。当前优先级是把行为和错误映射测扎实。
- C API tests 负责精确 status mapping：`PGO_STATUS_INVALID_ARGUMENT`、`PGO_STATUS_IO_ERROR`、`PGO_STATUS_OK`。
- Python binding 当前把 C API error 转为 `RuntimeError`；Python tests 断言异常 message，不要求 Python 层暴露 status enum。
- `mass_spring_cloth` examples 自生成小网格，不依赖外部 assets，适合 smoke test / CI。
- `mass_spring_bunny_cloth` examples 使用 `from_obj` / `pgo_world_create_mass_spring_from_obj`，展示真实 IO pipeline，默认 manual demo，不要求常规 CI 跑外部 asset。
- Bunny examples 不自动按高度 pin 顶点，因为 C/Python public API 目前没有 standalone OBJ read helper。第一版支持 `--pinned 0,1,2` 和 `--pinned-file pins.txt`，默认不 pin。
- C examples 不引入 CLI11；使用小型 C99 argument parser，避免 C API cookbook 依赖 C++ example tooling。
- `pgo_solver_options_t::commit_on_failure` 不在本期负路径测试覆盖范围内：3 顶点 mesh 上无法稳定触发 solver failure，验证 happy path 的 `commit_on_failure=0` 已足够。日后等真实 failing fixture 出现再补。
- pypgo CI 统一走 `scripts/pgo_release_wheels.py` 的 `default` / `accel` variant；所有平台都用 `pypgo-release-accel-all` 预设里的 `PGO_EIGEN_ACCELERATION_BACKEND=AUTO`，不再用 matrix 级 `extra_config` 强制 MKL。

## 1. Target File Structure

```text
examples/
  CMakeLists.txt
  c_api/
    CMakeLists.txt
    mass_spring_cloth.c
    mass_spring_bunny_cloth.c
  python/
    mass_spring_cloth.py
    mass_spring_bunny_cloth.py
tests/
  c_api/
    CMakeLists.txt
    test_c_api.c
  python/
    test_world.py
docs/
  api/
    c_api.md
    python_api.md
README.md
plan/
  c_py_api.md
```

Responsibilities:

- `examples/c_api/mass_spring_cloth.c`: C99 cookbook example that builds a cloth grid in memory, steps the world, and writes OBJ frames.
- `examples/c_api/mass_spring_bunny_cloth.c`: C99 cookbook example that loads an OBJ through the C API and writes OBJ frames.
- `examples/python/mass_spring_cloth.py`: Python cookbook example mirroring the C cloth grid behavior.
- `examples/python/mass_spring_bunny_cloth.py`: Python cookbook example for OBJ input via `World.from_obj`.
- `tests/c_api/test_c_api.c`: success and negative status coverage for the C ABI.
- `tests/python/test_world.py`: Python-level success, error, OBJ IO, and solver-option coverage.

## Phase 1: C API Negative Tests

### Task 1.1: Add C test helpers and fixture builders

**Files:**
- Modify: `tests/c_api/test_c_api.c`

- [ ] **Step 1: Add assertion helpers**

Add local helpers near the top of `tests/c_api/test_c_api.c`:

```c
static void expect_status(
    const pgo_status_t actual,
    const pgo_status_t expected,
    const pgo_error_t* error) {
    assert(actual == expected);
    if (expected == PGO_STATUS_OK) {
        assert(error == NULL || error->status == PGO_STATUS_OK);
    } else {
        assert(error != NULL);
        assert(error->status == expected);
        assert(error->message[0] != '\0');
    }
}

static pgo_mesh_view_t make_triangle_mesh(
    const double* positions,
    const uint64_t vertex_count,
    const uint64_t* triangles,
    const uint64_t triangle_count,
    const uint64_t* pinned,
    const uint64_t pinned_count) {
    pgo_mesh_view_t mesh;
    mesh.positions_xyz = positions;
    mesh.vertex_count = vertex_count;
    mesh.triangles = triangles;
    mesh.triangle_count = triangle_count;
    mesh.pinned_vertices = pinned;
    mesh.pinned_vertex_count = pinned_count;
    return mesh;
}
```

Expected: existing happy path still compiles after switching to helper-based mesh construction.

- [ ] **Step 2: Refactor existing happy path into `test_create_step_destroy`**

Move the current `main` body into:

```c
static void test_create_step_destroy(void) {
    /* existing create/query/copy/step/destroy assertions */
}
```

Then make `main` call `test_create_step_destroy();`.

- [ ] **Step 3: Run C API test target**

Run:

```bash
cmake --build --preset debug --target pgo_c_api_tests
ctest --test-dir build/debug --output-on-failure -R pgo_c_api_tests
```

Expected: `pgo_c_api_tests` passes.

### Task 1.2: Test invalid mesh and buffer inputs

**Files:**
- Modify: `tests/c_api/test_c_api.c`

- [ ] **Step 1: Add invalid triangle tests**

Add:

```c
static void test_invalid_triangles_return_invalid_argument(void) {
    const double positions[] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    };
    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);

    pgo_error_t error;
    pgo_world_t* world = NULL;

    const uint64_t degenerate_triangles[] = {0, 1, 1};
    pgo_error_clear(&error);
    pgo_mesh_view_t degenerate =
        make_triangle_mesh(positions, 3, degenerate_triangles, 1, NULL, 0);
    expect_status(
        pgo_world_create_mass_spring(&degenerate, &params, &world, &error),
        PGO_STATUS_INVALID_ARGUMENT,
        &error);
    assert(world == NULL);

    const uint64_t out_of_range_triangles[] = {0, 1, 3};
    pgo_error_clear(&error);
    pgo_mesh_view_t out_of_range =
        make_triangle_mesh(positions, 3, out_of_range_triangles, 1, NULL, 0);
    expect_status(
        pgo_world_create_mass_spring(&out_of_range, &params, &world, &error),
        PGO_STATUS_INVALID_ARGUMENT,
        &error);
    assert(world == NULL);
}
```

Call it from `main`.

- [ ] **Step 2: Add pinned vertex and small output buffer tests**

Add:

```c
static void test_pinned_and_output_buffer_validation(void) {
    const double positions[] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    };
    const uint64_t triangles[] = {0, 1, 2};
    const uint64_t invalid_pinned[] = {9};

    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);

    pgo_error_t error;
    pgo_world_t* world = NULL;
    pgo_mesh_view_t invalid_pin =
        make_triangle_mesh(positions, 3, triangles, 1, invalid_pinned, 1);
    pgo_error_clear(&error);
    expect_status(
        pgo_world_create_mass_spring(&invalid_pin, &params, &world, &error),
        PGO_STATUS_INVALID_ARGUMENT,
        &error);
    assert(world == NULL);

    pgo_mesh_view_t mesh = make_triangle_mesh(positions, 3, triangles, 1, NULL, 0);
    pgo_error_clear(&error);
    expect_status(
        pgo_world_create_mass_spring(&mesh, &params, &world, &error),
        PGO_STATUS_OK,
        &error);

    double too_small[8] = {0.0};
    pgo_error_clear(&error);
    expect_status(
        pgo_world_copy_positions(world, too_small, 8, &error),
        PGO_STATUS_INVALID_ARGUMENT,
        &error);

    pgo_world_destroy(world);
}
```

Call it from `main`.

- [ ] **Step 3: Verify invalid input tests**

Run:

```bash
cmake --build --preset debug --target pgo_c_api_tests
ctest --test-dir build/debug --output-on-failure -R pgo_c_api_tests
```

Expected: C API tests pass.

### Task 1.3: Test null args, invalid params, and solver options

**Files:**
- Modify: `tests/c_api/test_c_api.c`

- [ ] **Step 1: Add null-argument coverage**

Add the test wrapper with explicit fixtures so the snippet compiles:

```c
static void test_null_arguments_return_invalid_argument(void) {
    const double positions[] = {0,0,0, 1,0,0, 0,1,0};
    const uint64_t triangles[] = {0, 1, 2};
    pgo_mesh_view_t mesh = make_triangle_mesh(positions, 3, triangles, 1, NULL, 0);

    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);
    pgo_solver_options_t options;
    pgo_solver_options_default(&options);

    pgo_error_t error;
    pgo_world_t* world = NULL;
    pgo_error_clear(&error);
    expect_status(
        pgo_world_create_mass_spring(&mesh, &params, &world, &error),
        PGO_STATUS_OK,
        &error);

    uint64_t vertex_count = 0;
    double out_positions[9] = {0};
    pgo_step_result_t result;

    expect_status(pgo_world_create_mass_spring(NULL, &params, &world, &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_create_mass_spring(&mesh, NULL, &world, &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_create_mass_spring(&mesh, &params, NULL, &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_vertex_count(NULL, &vertex_count, &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_vertex_count(world, NULL, &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_copy_positions(NULL, out_positions, 9, &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_copy_positions(world, NULL, 9, &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_step(NULL, &options, &result, &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_step(world, NULL, &result, &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_step(world, &options, NULL, &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_write_obj_frame(NULL, "unused", &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_write_obj_frame(world, NULL, &error), PGO_STATUS_INVALID_ARGUMENT, &error);

    pgo_world_destroy(world);
}
```

注意：`pgo_world_create_mass_spring(&mesh, &params, NULL, &error)` 的语义是 `out_world` 为 NULL，跟局部 `world` 变量无关；测试中创建一份新 valid world 复用就好。

- [ ] **Step 2: Add invalid parameter coverage**

Add checks for:

```c
params.stiffness = 0.0;
params.dt = 0.0;
params.gravity = -1.0;
```

Each should make `pgo_world_create_mass_spring(...)` return `PGO_STATUS_INVALID_ARGUMENT`.

- [ ] **Step 3: Add solver option coverage**

Create a valid `world`, then check each invalid option independently — reset to defaults between cases so the previous mutation doesn't mask the next:

```c
pgo_solver_options_t options;
pgo_step_result_t result;

pgo_solver_options_default(&options);
options.max_iterations = 0;
expect_status(pgo_world_step(world, &options, &result, &error),
              PGO_STATUS_INVALID_ARGUMENT, &error);

pgo_solver_options_default(&options);
options.gradient_tolerance = 0.0;
expect_status(pgo_world_step(world, &options, &result, &error),
              PGO_STATUS_INVALID_ARGUMENT, &error);

pgo_solver_options_default(&options);
options.initial_regularization = 0.0;
expect_status(pgo_world_step(world, &options, &result, &error),
              PGO_STATUS_INVALID_ARGUMENT, &error);
```

- [ ] **Step 4: Verify null and option tests**

Run:

```bash
cmake --build --preset debug --target pgo_c_api_tests
ctest --test-dir build/debug --output-on-failure -R pgo_c_api_tests
```

Expected: C API tests pass.

## Phase 2: C API OBJ IO Tests

### Task 2.1: Add small OBJ fixture utilities

**Files:**
- Modify: `tests/c_api/test_c_api.c`

- [ ] **Step 1: Add portable fixture writers**

Add:

```c
static void write_text_file(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    assert(file != NULL);
    const size_t written = fwrite(text, 1, strlen(text), file);
    assert(written == strlen(text));
    assert(fclose(file) == 0);
}
```

Also add required includes:

```c
#include <stdio.h>
#include <string.h>
```

- [ ] **Step 2: Use `PGO_TEST_TMPDIR` for generated files**

Read the temp directory from `getenv("PGO_TEST_TMPDIR")`; fall back to `"."` if unset. Build file paths with `snprintf`.

```c
static const char* test_tmpdir(void) {
    const char* value = getenv("PGO_TEST_TMPDIR");
    return value == NULL || value[0] == '\0' ? "." : value;
}
```

Add required include:

```c
#include <stdlib.h>
```

### Task 2.2: Test C `from_obj` and `write_obj_frame`

**Files:**
- Modify: `tests/c_api/test_c_api.c`
- Modify: `tests/c_api/CMakeLists.txt`

- [ ] **Step 1: Add `from_obj` success test**

Write a tiny OBJ fixture:

```obj
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
```

Then call `pgo_world_create_mass_spring_from_obj(path, &params, NULL, 0, &world, &error)` and verify:

```c
expect_status(status, PGO_STATUS_OK, &error);
assert(world != NULL);
```

- [ ] **Step 2: Add missing OBJ IO error test**

Call `pgo_world_create_mass_spring_from_obj` with a path under `PGO_TEST_TMPDIR` that does not exist. Assert `PGO_STATUS_IO_ERROR`.

- [ ] **Step 3: Add `write_obj_frame` success test**

Create a world from arrays, call:

```c
expect_status(pgo_world_write_obj_frame(world, output_dir, &error), PGO_STATUS_OK, &error);
```

Then verify a frame file exists. Use the actual writer naming by checking at least one regular file in the output directory, not by hardcoding `frame_0000.obj` unless current writer behavior requires it.

- [ ] **Step 4: Add `write_obj_frame` IO error test**

预先创建一个 regular file `<PGO_TEST_TMPDIR>/blocked_dir`，然后把这个 *file* 路径当 `output_dir` 传进 `pgo_world_write_obj_frame`。`ObjWriter3d` 构造时调用 `std::filesystem::create_directories(path)`，遇到已存在的 regular file 会抛 `filesystem_error`，对应到 `PGO_STATUS_IO_ERROR`。

```c
char blocked[512];
snprintf(blocked, sizeof(blocked), "%s/blocked_dir", test_tmpdir());
write_text_file(blocked, "");
pgo_error_clear(&error);
expect_status(
    pgo_world_write_obj_frame(world, blocked, &error),
    PGO_STATUS_IO_ERROR,
    &error);
```

Windows 上 `create_directories` 对 regular file 的语义历史上不稳定；如果在 windows-latest 上偶发返回 OK，把这个子测试改成 `output_dir = "" `（空字符串，`obj_writer` 直接 reject）作为兜底，或在该平台 `#if defined(_WIN32)` skip 这一段并留 TODO。

- [ ] **Step 5: Configure test temp directory**

In `tests/c_api/CMakeLists.txt`，把 `ENVIRONMENT_MODIFICATION` 写成 CMake list（每条变量一个元素，跨平台都正确）；temp dir 用 `${CMAKE_CURRENT_BINARY_DIR}` 而不是 `$<TARGET_FILE_DIR:pgo_c_api_tests>`，避免 Windows 上把 frame 输出写进存放 DLL 的 runtime 目录：

```cmake
set_tests_properties(pgo_c_api_tests PROPERTIES
    ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:pgo_c>;PGO_TEST_TMPDIR=set:${CMAKE_CURRENT_BINARY_DIR}"
)
```

Test 启动前清空 `${CMAKE_CURRENT_BINARY_DIR}` 下上次跑遗留的 fixture 文件（`blocked_dir` 等）：在 C 测试代码里每次都 `remove()` 旧文件再 `fopen()`，避免上一次 run 留下的 directory 干扰下一次 IO_ERROR 测试。

- [ ] **Step 6: Verify C OBJ IO tests**

Run:

```bash
cmake --build --preset debug --target pgo_c_api_tests
ctest --test-dir build/debug --output-on-failure -R pgo_c_api_tests
```

Expected: C API tests pass and leave only ignored build-tree artifacts.

## Phase 3: Python API Tests

### Task 3.1: Add Python fixture helpers

**Files:**
- Modify: `tests/python/test_world.py`

- [ ] **Step 1: Add small mesh helper**

Add:

```python
def make_triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.uint64)
    return vertices, triangles
```

Refactor existing tests to use it.

- [ ] **Step 2: Add OBJ fixture helper**

Add:

```python
def write_triangle_obj(path: Path) -> None:
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
```

Add `from pathlib import Path`.

### Task 3.2: Test Python error behavior

**Files:**
- Modify: `tests/python/test_world.py`

- [ ] **Step 1: Add invalid array tests**

Add pytest cases:

```python
def test_from_arrays_rejects_invalid_triangle() -> None:
    vertices, _ = make_triangle_mesh()
    triangles = np.array([[0, 1, 3]], dtype=np.uint64)
    with pytest.raises(RuntimeError, match="triangle vertex index is out of range"):
        World.from_arrays(vertices, triangles)


def test_from_arrays_rejects_degenerate_triangle() -> None:
    vertices, _ = make_triangle_mesh()
    triangles = np.array([[0, 1, 1]], dtype=np.uint64)
    with pytest.raises(RuntimeError, match="triangle must reference three distinct vertices"):
        World.from_arrays(vertices, triangles)
```

Add `import pytest`.

- [ ] **Step 2: Add invalid solver option tests**

Add:

```python
def test_step_rejects_invalid_solver_options() -> None:
    vertices, triangles = make_triangle_mesh()
    world = World.from_arrays(vertices, triangles)
    with pytest.raises(RuntimeError, match="max_iterations must be positive"):
        world.step(max_iterations=0)
    with pytest.raises(RuntimeError, match="gradient_tolerance must be positive"):
        world.step(gradient_tolerance=0.0)
    with pytest.raises(RuntimeError, match="initial_regularization must be positive"):
        world.step(initial_regularization=0.0)
```

- [ ] **Step 3: Add Python OBJ IO tests**

Add:

```python
def test_from_obj_and_write_obj_frame(tmp_path: Path) -> None:
    obj_path = tmp_path / "triangle.obj"
    write_triangle_obj(obj_path)
    world = World.from_obj(str(obj_path))
    assert world.vertex_count == 3
    output_dir = tmp_path / "frames"
    world.write_obj_frame(str(output_dir))
    assert any(output_dir.glob("*.obj"))


def test_from_obj_missing_file_maps_to_exception(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        World.from_obj(str(tmp_path / "missing.obj"))
```

- [ ] **Step 4: Verify Python tests**

Run:

```bash
uv run pytest tests/python/test_world.py -q
```

Expected: all Python API tests pass in an environment where the editable package is built.

## Phase 4: C API Examples

### Task 4.1: Add C API example CMake targets

**Files:**
- Modify: `examples/CMakeLists.txt`
- Create: `examples/c_api/CMakeLists.txt`

- [ ] **Step 1: Gate C API examples on `PGO_BUILD_C_API`**

Append to `examples/CMakeLists.txt`:

```cmake
if(PGO_BUILD_C_API AND TARGET pgo_c)
    add_subdirectory(c_api)
endif()
```

- [ ] **Step 2: Add C API example targets**

Create `examples/c_api/CMakeLists.txt`:

```cmake
add_executable(pgo_c_mass_spring_cloth mass_spring_cloth.c)
target_link_libraries(pgo_c_mass_spring_cloth PRIVATE pgo::c_api)

add_executable(pgo_c_mass_spring_bunny_cloth mass_spring_bunny_cloth.c)
target_link_libraries(pgo_c_mass_spring_bunny_cloth PRIVATE pgo::c_api)
```

Expected: examples compile only when both examples and C API are enabled.

### Task 4.2: Implement `mass_spring_cloth.c`

**Files:**
- Create: `examples/c_api/mass_spring_cloth.c`

- [ ] **Step 1: Implement options and usage**

Support:

```text
--resolution <uint>    default 8, minimum 2
--frames <uint>        default 10, minimum 1
--output <path>        default output/example/c_api/mass_spring/cloth
--stiffness <double>   default from pgo_mass_spring_params_default
--gravity <double>     default from pgo_mass_spring_params_default
--dt <double>          default from pgo_mass_spring_params_default
--max-iterations <uint> default from pgo_solver_options_default
--help
```

Use `strtoull` / `strtod` and print usage to `stderr` on invalid flags.

- [ ] **Step 2: Generate a cloth grid**

Generate positions for an `N x N` grid in the XY plane, two triangles per quad, and pinned vertices on the top row.

Memory ownership:

```c
double* positions = malloc(vertex_count * 3 * sizeof(double));
uint64_t* triangles = malloc(triangle_count * 3 * sizeof(uint64_t));
uint64_t* pinned = malloc(resolution * sizeof(uint64_t));
```

Free all arrays before exit.

- [ ] **Step 3: Run C API simulation**

Create the world, then:

```c
for (uint64_t frame = 0; frame < frames; ++frame) {
    pgo_step_result_t result;
    status = pgo_world_step(world, &solver_options, &result, &error);
    if (status != PGO_STATUS_OK) { /* print and exit */ }
    status = pgo_world_write_obj_frame(world, options.output, &error);
    if (status != PGO_STATUS_OK) { /* print and exit */ }
    printf("frame=%llu status=%d iterations=%llu value=%g grad=%g\n", ...);
}
```

Destroy `world` on every exit path.

- [ ] **Step 4: Build and smoke run**

Run:

```bash
cmake --build --preset debug --target pgo_c_mass_spring_cloth
./build/debug/examples/c_api/pgo_c_mass_spring_cloth --frames 2 --resolution 3 --output output/example/c_api/smoke-cloth
```

Expected: executable returns 0 and writes OBJ frame files.

### Task 4.3: Implement `mass_spring_bunny_cloth.c`

**Files:**
- Create: `examples/c_api/mass_spring_bunny_cloth.c`

- [ ] **Step 1: Implement options**

Support:

```text
--input <path>         default assets/model/bunny.obj
--output <path>        default output/example/c_api/mass_spring/bunny
--frames <uint>        default 10
--stiffness <double>   default 200000
--gravity <double>     default 9.81
--dt <double>          default 0.001
--pinned <csv>         optional comma-separated uint64 vertex indices
--pinned-file <path>   optional whitespace/newline-separated uint64 vertex indices
--max-iterations <uint> default 500
--help
```

If both `--pinned` and `--pinned-file` are provided, print an error and exit 2.

- [ ] **Step 2: Parse pinned vertices**

Implement a small dynamic array:

```c
typedef struct pinned_list_t {
    uint64_t* values;
    uint64_t count;
    uint64_t capacity;
} pinned_list_t;
```

Support CSV and file parsing with `strtoull`; reject malformed tokens.

- [ ] **Step 3: Run OBJ-backed C API simulation**

Use:

```c
pgo_world_create_mass_spring_from_obj(
    options.input,
    &params,
    pinned.values,
    pinned.count,
    &world,
    &error);
```

Then step/write frames as in the cloth example.

- [ ] **Step 4: Build only smoke**

Run:

```bash
cmake --build --preset debug --target pgo_c_mass_spring_bunny_cloth
```

Expected: target builds. Do not require runtime smoke unless `assets/model/bunny.obj` exists.

## Phase 5: Python Examples

### Task 5.1: Implement Python cloth example

**Files:**
- Create: `examples/python/mass_spring_cloth.py`

- [ ] **Step 1: Implement mesh generation**

Add:

```python
def make_cloth_grid(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    vertices = []
    triangles = []
    for row in range(resolution):
        for col in range(resolution):
            scale = 1.0 / float(resolution - 1)
            vertices.append((col * scale, row * scale, 0.0))
    def vid(row: int, col: int) -> int:
        return row * resolution + col
    for row in range(resolution - 1):
        for col in range(resolution - 1):
            v00 = vid(row, col)
            v10 = vid(row + 1, col)
            v11 = vid(row + 1, col + 1)
            v01 = vid(row, col + 1)
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))
    pinned = np.array([vid(resolution - 1, col) for col in range(resolution)], dtype=np.uint64)
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(triangles, dtype=np.uint64),
        pinned,
    )
```

- [ ] **Step 2: Implement CLI and simulation**

Use `argparse` with the same options as C cloth where practical. Create `World.from_arrays(...)`, loop `world.step(...)`, call `world.write_obj_frame(...)`, and print one summary line per frame.

- [ ] **Step 3: Smoke run**

Run:

```bash
uv run python examples/python/mass_spring_cloth.py --frames 2 --resolution 3 --output output/example/python/smoke-cloth
```

Expected: script returns 0 and writes OBJ frame files after editable package setup.

### Task 5.2: Implement Python bunny example

**Files:**
- Create: `examples/python/mass_spring_bunny_cloth.py`

- [ ] **Step 1: Implement pinned parsing**

Support:

```text
--pinned 0,1,2
--pinned-file pins.txt
```

Return `np.ndarray(dtype=np.uint64)`.

- [ ] **Step 2: Implement OBJ-backed simulation**

Use:

```python
world = World.from_obj(
    args.input,
    stiffness=args.stiffness,
    gravity=args.gravity,
    dt=args.dt,
    pinned_vertices=pinned,
)
```

Loop `step` and `write_obj_frame`, printing solver status summary.

- [ ] **Step 3: Manual build/run guidance**

Do not add this example to default pytest unless a local OBJ asset exists. Document:

```bash
uv run python examples/python/mass_spring_bunny_cloth.py --input assets/model/bunny.obj --frames 5
```

## Phase 6: Example Smoke Tests

### Task 6.1: Add C example smoke to CTest

**Files:**
- Modify: `examples/c_api/CMakeLists.txt`

- [ ] **Step 1: Add cloth smoke test**

When `PGO_BUILD_TESTS` is ON, add；输出路径放在 `${CMAKE_CURRENT_BINARY_DIR}` 下并按 `$<CONFIG>` 分隔，避免 multi-config 生成器（Xcode/MSVC）里几个 config 共用同一个目录互相覆盖：

```cmake
if(PGO_BUILD_TESTS)
    add_test(
        NAME pgo_c_mass_spring_cloth_example_smoke
        COMMAND pgo_c_mass_spring_cloth
            --frames 2
            --resolution 3
            --output "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/smoke-cloth"
    )
    set_tests_properties(pgo_c_mass_spring_cloth_example_smoke PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:pgo_c>"
    )
endif()
```

Expected: CI can run the self-contained C cloth example.

### Task 6.2: Add Python example smoke test

**Files:**
- Create: `tests/python/test_examples.py`

- [ ] **Step 1: Test pure mesh helper**

Import `examples/python/mass_spring_cloth.py` with `runpy.run_path` or add `examples/python` to `sys.path`, then verify `make_cloth_grid(3)` returns:

```python
vertices.shape == (9, 3)
triangles.shape == (8, 3)
pinned.shape == (3,)
```

- [ ] **Step 2: Test script smoke via subprocess**

Add:

```python
def test_python_cloth_example_smoke(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "examples/python/mass_spring_cloth.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--frames",
            "2",
            "--resolution",
            "3",
            "--output",
            str(tmp_path / "frames"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert any((tmp_path / "frames").glob("*.obj"))
```

Use `sys.executable` so the test runs in the same environment as the installed package.

- [ ] **Step 3: Verify Python example tests**

Run:

```bash
uv run pytest tests/python/test_world.py tests/python/test_examples.py -q
```

Expected: Python API and example smoke tests pass.

## Phase 7: Docs And Plan Sync

### Task 7.1: Document C/Python API examples

**Files:**
- Modify: `README.md`
- Modify: `docs/api/c_api.md`
- Modify: `docs/api/python_api.md`
- Modify: `plan/c_py_api.md`

- [ ] **Step 1: Add README example commands**

Add a short section with:

```bash
./build/debug/examples/c_api/pgo_c_mass_spring_cloth --frames 5 --resolution 8
uv run python examples/python/mass_spring_cloth.py --frames 5 --resolution 8
```

Mention bunny examples as manual demos requiring an OBJ path.

- [ ] **Step 2: Update C API docs**

Add C example build/run commands and clarify:

- C examples are C99 API consumers.
- `mass_spring_cloth.c` is self-contained.
- `mass_spring_bunny_cloth.c` exercises `pgo_world_create_mass_spring_from_obj`.

- [ ] **Step 3: Update Python API docs**

Add Python example commands and clarify:

- `mass_spring_cloth.py` is the recommended first example.
- `mass_spring_bunny_cloth.py` requires an OBJ file and optional pinned indices.

- [ ] **Step 4: Update `plan/c_py_api.md` completion criteria**

Add:

- C API negative tests cover invalid mesh, null args, output buffer size, solver options, OBJ IO, and IO error mapping.
- Python API tests cover invalid mesh, solver options, `from_obj`, and `write_obj_frame`.
- C/Python examples include self-contained cloth and OBJ-backed bunny demos.

## Phase 8: Full Verification

### Task 8.1: Run targeted verification

**Files:**
- No source edits.

- [ ] **Step 1: Build C API examples and tests**

Run:

```bash
cmake --build --preset debug --target pgo_c_api_tests pgo_c_mass_spring_cloth pgo_c_mass_spring_bunny_cloth
```

Expected: all targets build.

- [ ] **Step 2: Run CTest subset**

Run:

```bash
ctest --test-dir build/debug --output-on-failure -R "pgo_c_api_tests|pgo_c_mass_spring_cloth_example_smoke"
```

Expected: C API tests and C cloth smoke pass.

- [ ] **Step 3: Run Python tests**

Run:

```bash
uv run pytest tests/python/test_world.py tests/python/test_examples.py -q
```

Expected: Python API and example smoke tests pass.

- [ ] **Step 4: Run manual demo commands when assets are available**

If `assets/model/bunny.obj` exists, run:

```bash
./build/debug/examples/c_api/pgo_c_mass_spring_bunny_cloth --input assets/model/bunny.obj --frames 2 --output output/example/c_api/smoke-bunny
uv run python examples/python/mass_spring_bunny_cloth.py --input assets/model/bunny.obj --frames 2 --output output/example/python/smoke-bunny
```

Expected: both commands return 0 and write OBJ frames.

## Phase 9: pypgo CI Sync To `pgo_release_wheels.py`

目标：把 `.github/workflows/ci.yml` 的 `python-api` job 切换到 `scripts/pgo_release_wheels.py`，统一所有平台用 `pypgo-release-accel-all` preset 自带的 `PGO_EIGEN_ACCELERATION_BACKEND=AUTO`，删掉 matrix-level `extra_config` 强制 MKL 的 override。

### Task 9.1: Add Conan profile passthrough to `pgo_release_wheels.py`

**Files:**
- Modify: `scripts/pgo_release_wheels.py`
- Modify: `scripts/test_pgo_release_wheels.py`

- [ ] **Step 1: Extend CLI**

新增三个互不冲突的参数：

```text
--profile <path>          Use the same Conan profile for host and build contexts
--host-profile <path>     Conan host profile
--build-profile <path>    Conan build profile
```

和 `pgo_build_wheel.py` 的同名参数语义保持一致；`--profile` 与 `--host-profile`/`--build-profile` 互斥。

- [ ] **Step 2: Forward profiles into the inner build command**

`create_release_plan` 接收 `host_profile`/`build_profile`，构造 `inner_command` 时把它们追加为 `pgo_build_wheel.py` 的参数（只在非 None 时附加）。

- [ ] **Step 3: Update tests**

在 `scripts/test_pgo_release_wheels.py` 中加：

```python
def test_inner_command_forwards_profile(self) -> None:
    plan = pgo_release_wheels.create_release_plan(
        repo_root=REPO_ROOT,
        variant_name="default",
        source_tree=pathlib.Path("/tmp/pgo-release-src"),
        out_root=None,
        clear=False,
        stable_abi=False,
        host_profile=pathlib.Path("conan/profiles/ubuntu-x86_64-gcc"),
        build_profile=pathlib.Path("conan/profiles/ubuntu-x86_64-gcc"),
    )
    self.assertIn("--host-profile", plan.inner_command)
    self.assertIn("conan/profiles/ubuntu-x86_64-gcc", " ".join(plan.inner_command))
```

并扩展原有 `create_release_plan` 用例，验证不传 profile 时 inner command 里不含 `--host-profile`/`--build-profile`。

- [ ] **Step 4: Run script tests**

```bash
uv run python -m unittest scripts.test_pgo_release_wheels
```

Expected: all unittest cases pass.

### Task 9.2: Swap CI `python-api` job to `pgo_release_wheels.py`

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Rework matrix**

把每行的 `preset`/`extra_config` 换成 `variant` + `distribution`：

```yaml
matrix:
  include:
    - { os: ubuntu-latest,  variant: default, distribution: pgo,       profile: conan/profiles/ubuntu-x86_64-gcc,       cc: gcc-13, cxx: g++-13 }
    - { os: ubuntu-latest,  variant: accel,   distribution: pgo-accel, profile: conan/profiles/ubuntu-x86_64-gcc,       cc: gcc-13, cxx: g++-13 }
    - { os: macos-latest,   variant: default, distribution: pgo,       profile: conan/profiles/macos-arm64-apple-clang, cc: cc,    cxx: c++ }
    - { os: macos-latest,   variant: accel,   distribution: pgo-accel, profile: conan/profiles/macos-arm64-apple-clang, cc: cc,    cxx: c++ }
    - { os: windows-latest, variant: default, distribution: pgo,       profile: conan/profiles/windows-x86_64-msvc,     cc: cl,    cxx: cl  }
    - { os: windows-latest, variant: accel,   distribution: pgo-accel, profile: conan/profiles/windows-x86_64-msvc,     cc: cl,    cxx: cl  }
```

job name 改成 `${{ matrix.os }} ${{ matrix.variant }} python`。

- [ ] **Step 2: oneMKL step gating**

保留 `Install oneMKL (Linux/Windows)`，把 `contains(matrix.preset, 'accel')` 换成 `matrix.variant == 'accel'`。

- [ ] **Step 3: 用一次性 `pgo_release_wheels.py` 调用替换 configure + build**

删掉原有 `Get nanobind CMake directory`、`Configure Python package preset`、`Build wheel` 三步，换成单步：

```yaml
- name: Build release wheel
  env:
    CC: ${{ matrix.cc }}
    CXX: ${{ matrix.cxx }}
  run: >
    uv run python scripts/pgo_release_wheels.py
    --variant ${{ matrix.variant }}
    --profile ${{ matrix.profile }}
    --clear
```

`pgo_release_wheels.py` 会复制源树到临时目录、patch pyproject 的 distribution name、再调 `pgo_build_wheel.py`（内部跑 conan install + cmake configure + uv build）。

- [ ] **Step 4: Install + test**

替换 `Install wheel` 与 `Test Python API` 两步：

```yaml
- name: Install wheel
  shell: bash
  run: uv pip install "dist/${{ matrix.distribution }}"/*.whl

- name: Test Python API
  run: uv run pytest tests/python -q
```

`pgo` 和 `pgo-accel` wheel 内部的 import package 都是 `pgo`，pytest 不需要改。

- [ ] **Step 5: Verify locally with `--dry-run`**

跑：

```bash
uv run python scripts/pgo_release_wheels.py --variant accel --profile conan/profiles/macos-arm64-apple-clang --dry-run
```

确认输出的 inner command 含 `--host-profile`/`--build-profile`，并且最终 cmake.define 不再出现 `PGO_EIGEN_ACCELERATION_BACKEND=MKL`（应当只来自 preset 的 `AUTO`）。

## Deferred Work

- Add `size/version` fields to C ABI public structs before the first stable C SDK release.
- Consider standalone C/Python OBJ inspection helpers only if bunny demos need automatic pin-by-height behavior.
- Consider exposing typed Python exception classes if downstream users need to distinguish `INVALID_ARGUMENT` from `IO_ERROR` without parsing messages.
- 若后续 release artifact 需要在 Linux/Windows 强制 MKL，再考虑给 `pgo_release_wheels.py` 加 `--cmake-define` passthrough 或单独的 release-only preset，本期 CI 不引入。

## Completion Criteria

- C API tests cover happy path and all listed negative path categories.
- Python tests cover happy path, error path, OBJ input, frame output, and solver options.
- C cloth example builds and runs in CTest smoke.
- C bunny example builds and documents OBJ/pinned usage.
- Python cloth example runs as a self-contained smoke test.
- Python bunny example documents OBJ/pinned usage.
- Docs and `plan/c_py_api.md` reference the new examples and expanded tests.
- No C ABI `size/version` fields are added in this phase.
- `.github/workflows/ci.yml` 的 `python-api` job 只通过 `scripts/pgo_release_wheels.py` 构建 wheel；matrix 不再包含 `preset` 或 `extra_config`，所有平台一致使用 `PGO_EIGEN_ACCELERATION_BACKEND=AUTO`。
- `scripts/pgo_release_wheels.py` 支持 `--profile`/`--host-profile`/`--build-profile`，`scripts/test_pgo_release_wheels.py` 覆盖该 passthrough。
