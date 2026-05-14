#include "pgo_c/pgo.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

int main(void) {
    const double positions[] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    };
    const uint64_t triangles[] = {0, 1, 2};
    const uint64_t pinned[] = {0};

    const pgo_mesh_view_t mesh = {
        positions,
        3,
        triangles,
        1,
        pinned,
        1,
    };
    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);

    pgo_error_t error;
    pgo_error_clear(&error);

    pgo_world_t* world = NULL;
    assert(pgo_world_create_mass_spring(&mesh, &params, &world, &error) == PGO_STATUS_OK);
    assert(world != NULL);

    uint64_t vertex_count = 0;
    assert(pgo_world_vertex_count(world, &vertex_count, &error) == PGO_STATUS_OK);
    assert(vertex_count == 3);

    double out_positions[9] = {0.0};
    assert(pgo_world_copy_positions(world, out_positions, 9, &error) == PGO_STATUS_OK);
    assert(out_positions[3] == 1.0);

    pgo_solver_options_t solver_options;
    pgo_solver_options_default(&solver_options);

    pgo_step_result_t step_result;
    assert(pgo_world_step(world, &solver_options, &step_result, &error) == PGO_STATUS_OK);

    pgo_world_destroy(world);
    pgo_world_destroy(NULL);
    return 0;
}
