# C API

`pgo_c` exposes a C99 ABI over the C++ simulation core.

The public ABI contains only:

- `extern "C"` functions.
- Opaque handles such as `pgo_world_t`.
- POD descriptors such as `pgo_mesh_view_t`.
- Caller-owned output buffers.
- Explicit destroy functions.
- `pgo_status_t` plus `pgo_error_t` error reporting.

It does not expose STL, Eigen, C++ templates, C++ exceptions, namespaces, classes, or allocators.

## Minimal Example

```c
#include "pgo_c/pgo.h"

#include <stddef.h>
#include <stdint.h>

int main(void) {
    const double positions[] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    };
    const uint64_t triangles[] = {0, 1, 2};
    const pgo_mesh_view_t mesh = {positions, 3, triangles, 1, NULL, 0};
    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);

    pgo_error_t error;
    pgo_error_clear(&error);

    pgo_world_t* world = NULL;
    if (pgo_world_create_mass_spring(&mesh, &params, &world, &error) != PGO_STATUS_OK) {
        return 1;
    }

    pgo_step_result_t result;
    pgo_world_step(world, NULL, &result, &error);
    pgo_world_destroy(world);
    return 0;
}
```

## ABI Hygiene Checks

```bash
rg "std::|Eigen::|template|class|namespace|#include <vector>|#include <string>" include/pgo_c
cc -std=c99 -Iinclude -x c -c include/pgo_c/pgo.h -o /tmp/pgo_c_header.o
```

On macOS:

```bash
nm -gU build/debug/src/c_api/libpgo_c.dylib | rg "pgo_|std::|Eigen::"
```

On Linux:

```bash
nm -D --defined-only build/debug/src/c_api/libpgo_c.so | rg "pgo_|std::|Eigen::"
```

The exported-symbol checks should show only public `pgo_*` symbols.

## MKL Runtime Policy

The accelerated Linux/Windows package requires MKL. The default package does not.

Rules:

- `release-all` builds with Eigen's internal kernels unless the user explicitly enables acceleration.
- `release-accel-all` requires `PGO_ENABLE_EIGEN_ACCELERATION=ON`.
- On macOS, `release-accel-all` uses Apple Accelerate.
- On Linux and Windows, `release-accel-all` uses MKL and fails configuration if MKL is unavailable.
- Simulation-level parallelism should own the outer thread count. When TBB is enabled, prefer `MKL_NUM_THREADS=1` and `OMP_NUM_THREADS=1` unless a benchmark proves nested BLAS threads help a specific workload.

## Static GNU Runtime Policy

`PGO_STATIC_GNU_RUNTIME` is a packaging option for selected Linux GNU deliverables, not a development default.

Do not apply it blindly to `pgo_c` or Python wheels. Python wheels should be repaired/audited with the platform wheel tooling used by CI, because static C++ runtime choices interact with manylinux/musllinux policy and with dependencies such as MKL.
