#pragma once

#include "pgo_c/export.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum pgo_status_t {
    PGO_STATUS_OK = 0,
    PGO_STATUS_INVALID_ARGUMENT = 1,
    PGO_STATUS_IO_ERROR = 2,
    PGO_STATUS_INTERNAL_ERROR = 3
} pgo_status_t;

typedef enum pgo_solver_status_t {
    PGO_SOLVER_CONVERGED = 0,
    PGO_SOLVER_MAX_ITERATIONS = 1,
    PGO_SOLVER_REGULARIZATION_FAILED = 2,
    PGO_SOLVER_LINE_SEARCH_FAILED = 3
} pgo_solver_status_t;

typedef struct pgo_error_t {
    pgo_status_t status;
    char message[512];
} pgo_error_t;

typedef struct pgo_world_t pgo_world_t;

typedef struct pgo_mesh_view_t {
    const double* positions_xyz;
    uint64_t vertex_count;
    const uint64_t* triangles;
    uint64_t triangle_count;
    const uint64_t* pinned_vertices;
    uint64_t pinned_vertex_count;
} pgo_mesh_view_t;

typedef struct pgo_obj_mesh_t {
    double* positions_xyz;
    uint64_t vertex_count;
    uint64_t* triangles;
    uint64_t triangle_count;
} pgo_obj_mesh_t;

typedef struct pgo_mass_spring_params_t {
    double stiffness;
    double gravity;
    double dt;
} pgo_mass_spring_params_t;

typedef struct pgo_solver_options_t {
    uint64_t max_iterations;
    double gradient_tolerance;
    double initial_regularization;
    int commit_on_failure;
} pgo_solver_options_t;

typedef struct pgo_step_result_t {
    pgo_solver_status_t solver_status;
    uint64_t iterations;
    double final_value;
    double final_gradient_norm;
} pgo_step_result_t;

PGO_C_API const char* pgo_version(void);

PGO_C_API void pgo_error_clear(pgo_error_t* error);

PGO_C_API void pgo_mass_spring_params_default(pgo_mass_spring_params_t* params);

PGO_C_API void pgo_solver_options_default(pgo_solver_options_t* options);

PGO_C_API pgo_status_t pgo_world_create_mass_spring(
    const pgo_mesh_view_t* mesh,
    const pgo_mass_spring_params_t* params,
    pgo_world_t** out_world,
    pgo_error_t* error);

PGO_C_API pgo_status_t pgo_world_create_mass_spring_from_obj(
    const char* path,
    const pgo_mass_spring_params_t* params,
    const uint64_t* pinned_vertices,
    uint64_t pinned_vertex_count,
    pgo_world_t** out_world,
    pgo_error_t* error);

PGO_C_API void pgo_world_destroy(pgo_world_t* world);

PGO_C_API pgo_status_t pgo_world_vertex_count(
    const pgo_world_t* world,
    uint64_t* out_vertex_count,
    pgo_error_t* error);

PGO_C_API pgo_status_t pgo_world_copy_positions(
    const pgo_world_t* world,
    double* out_positions_xyz,
    uint64_t out_position_scalar_count,
    pgo_error_t* error);

PGO_C_API pgo_status_t pgo_world_step(
    pgo_world_t* world,
    const pgo_solver_options_t* options,
    pgo_step_result_t* out_result,
    pgo_error_t* error);

PGO_C_API pgo_status_t pgo_world_write_obj_frame(
    const pgo_world_t* world,
    const char* output_dir,
    pgo_error_t* error);

PGO_C_API pgo_status_t pgo_world_write_abc_frame(
    pgo_world_t* world,
    const char* output_path,
    double fps,
    pgo_error_t* error);

PGO_C_API pgo_status_t pgo_read_obj_mesh(
    const char* path,
    pgo_obj_mesh_t* out_mesh,
    pgo_error_t* error);

PGO_C_API void pgo_obj_mesh_free(pgo_obj_mesh_t* mesh);

#ifdef __cplusplus
}
#endif
