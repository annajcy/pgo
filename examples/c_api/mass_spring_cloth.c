#include "pgo_c/pgo.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t resolution;
    uint64_t frames;
    char output[512];
    char abc_output[512];
    double abc_fps;
    double stiffness;
    double gravity;
    double dt;
    uint64_t max_iterations;
} cloth_options_t;

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s [OPTIONS]\n"
            "  --resolution <uint>   Cloth grid resolution (default 8, min 2)\n"
            "  --frames <uint>       Number of frames (default 10, min 1)\n"
            "  --output <path>       Output directory (default output/example/c_api/mass_spring/cloth)\n"
            "  --export-abc <path>   Alembic .abc output file\n"
            "  --abc-fps <double>    FPS for Alembic export (default 24)\n"
            "  --stiffness <double>  Spring stiffness (default from pgo_mass_spring_params_default)\n"
            "  --gravity <double>    Gravity magnitude (default from pgo_mass_spring_params_default)\n"
            "  --dt <double>         Timestep size (default from pgo_mass_spring_params_default)\n"
            "  --max-iterations <uint> Max Newton iterations (default from pgo_solver_options_default)\n"
            "  --help                Show this message\n",
            prog);
}

static int parse_uint64(const char* s, uint64_t* out) {
    char* end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (end == s || *end != '\0') return -1;
    *out = (uint64_t)v;
    return 0;
}

static int parse_double(const char* s, double* out) {
    char* end = NULL;
    double v = strtod(s, &end);
    if (end == s || *end != '\0') return -1;
    *out = v;
    return 0;
}

static void cloth_options_default(cloth_options_t* opts) {
    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);
    pgo_solver_options_t solver;
    pgo_solver_options_default(&solver);

    opts->resolution = 8;
    opts->frames = 10;
    snprintf(opts->output, sizeof(opts->output), "output/example/c_api/mass_spring/cloth");
    opts->abc_output[0] = '\0';
    opts->abc_fps = 24.0;
    opts->stiffness = params.stiffness;
    opts->gravity = params.gravity;
    opts->dt = params.dt;
    opts->max_iterations = solver.max_iterations;
}

static int parse_options(int argc, char** argv, cloth_options_t* opts) {
    cloth_options_default(opts);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "--resolution") == 0 && i + 1 < argc) {
            if (parse_uint64(argv[++i], &opts->resolution) != 0 || opts->resolution < 2) {
                fprintf(stderr, "error: --resolution must be >= 2\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            if (parse_uint64(argv[++i], &opts->frames) != 0 || opts->frames < 1) {
                fprintf(stderr, "error: --frames must be >= 1\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            size_t len = strlen(argv[++i]);
            if (len >= sizeof(opts->output)) {
                fprintf(stderr, "error: --output path too long\n");
                return -1;
            }
            memcpy(opts->output, argv[i], len + 1);
        } else if (strcmp(argv[i], "--export-abc") == 0 && i + 1 < argc) {
            size_t len = strlen(argv[++i]);
            if (len >= sizeof(opts->abc_output)) {
                fprintf(stderr, "error: --export-abc path too long\n");
                return -1;
            }
            memcpy(opts->abc_output, argv[i], len + 1);
        } else if (strcmp(argv[i], "--abc-fps") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &opts->abc_fps) != 0 || opts->abc_fps <= 0.0) {
                fprintf(stderr, "error: --abc-fps must be positive\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--stiffness") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &opts->stiffness) != 0) {
                fprintf(stderr, "error: invalid --stiffness\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--gravity") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &opts->gravity) != 0) {
                fprintf(stderr, "error: invalid --gravity\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--dt") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &opts->dt) != 0) {
                fprintf(stderr, "error: invalid --dt\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--max-iterations") == 0 && i + 1 < argc) {
            if (parse_uint64(argv[++i], &opts->max_iterations) != 0 || opts->max_iterations < 1) {
                fprintf(stderr, "error: --max-iterations must be >= 1\n");
                return -1;
            }
        } else {
            fprintf(stderr, "error: unknown flag '%s'\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }
    return 0;
}

typedef struct {
    double* positions;
    uint64_t* triangles;
    uint64_t* pinned;
    uint64_t vertex_count;
    uint64_t triangle_count;
    uint64_t pinned_count;
} cloth_grid_t;

static void cloth_grid_free(cloth_grid_t* grid) {
    free(grid->positions);
    free(grid->triangles);
    free(grid->pinned);
}

static int make_cloth_grid(uint64_t resolution, cloth_grid_t* out) {
    const uint64_t n = resolution;
    const uint64_t vertex_count = n * n;
    const uint64_t quad_count = (n - 1) * (n - 1);
    const uint64_t triangle_count = quad_count * 2;
    const uint64_t pinned_count = n;

    double* positions = (double*)malloc(vertex_count * 3 * sizeof(double));
    uint64_t* triangles = (uint64_t*)malloc(triangle_count * 3 * sizeof(uint64_t));
    uint64_t* pinned = (uint64_t*)malloc(pinned_count * sizeof(uint64_t));
    if (positions == NULL || triangles == NULL || pinned == NULL) {
        free(positions);
        free(triangles);
        free(pinned);
        return -1;
    }

    const double scale = 1.0 / (double)(n - 1);
    uint64_t vidx = 0;
    for (uint64_t row = 0; row < n; ++row) {
        for (uint64_t col = 0; col < n; ++col) {
            positions[vidx * 3 + 0] = (double)col * scale;
            positions[vidx * 3 + 1] = (double)row * scale;
            positions[vidx * 3 + 2] = 0.0;
            ++vidx;
        }
    }

    uint64_t tidx = 0;
    for (uint64_t row = 0; row < n - 1; ++row) {
        for (uint64_t col = 0; col < n - 1; ++col) {
            const uint64_t v00 = row * n + col;
            const uint64_t v10 = (row + 1) * n + col;
            const uint64_t v11 = (row + 1) * n + col + 1;
            const uint64_t v01 = row * n + col + 1;
            triangles[tidx * 3 + 0] = v00;
            triangles[tidx * 3 + 1] = v10;
            triangles[tidx * 3 + 2] = v11;
            ++tidx;
            triangles[tidx * 3 + 0] = v00;
            triangles[tidx * 3 + 1] = v11;
            triangles[tidx * 3 + 2] = v01;
            ++tidx;
        }
    }

    for (uint64_t col = 0; col < n; ++col) {
        pinned[col] = (n - 1) * n + col;
    }

    out->positions = positions;
    out->triangles = triangles;
    out->pinned = pinned;
    out->vertex_count = vertex_count;
    out->triangle_count = triangle_count;
    out->pinned_count = pinned_count;
    return 0;
}

int main(int argc, char** argv) {
    cloth_options_t opts;
    if (parse_options(argc, argv, &opts) != 0) {
        return 1;
    }

    cloth_grid_t grid;
    if (make_cloth_grid(opts.resolution, &grid) != 0) {
        fprintf(stderr, "error: failed to allocate cloth grid\n");
        return 1;
    }

    pgo_mesh_view_t mesh;
    mesh.positions_xyz = grid.positions;
    mesh.vertex_count = grid.vertex_count;
    mesh.triangles = grid.triangles;
    mesh.triangle_count = grid.triangle_count;
    mesh.pinned_vertices = grid.pinned;
    mesh.pinned_vertex_count = grid.pinned_count;

    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);
    params.stiffness = opts.stiffness;
    params.gravity = opts.gravity;
    params.dt = opts.dt;

    pgo_error_t error;
    pgo_world_t* world = NULL;
    const pgo_status_t status = pgo_world_create_mass_spring(
        &mesh, &params, &world, &error);
    if (status != PGO_STATUS_OK) {
        fprintf(stderr, "error: %s\n", error.message);
        cloth_grid_free(&grid);
        return 1;
    }

    pgo_solver_options_t solver_options;
    pgo_solver_options_default(&solver_options);
    solver_options.max_iterations = opts.max_iterations;

    int exit_code = 0;
    for (uint64_t frame = 0; frame < opts.frames; ++frame) {
        pgo_step_result_t result;
        const pgo_status_t step_status = pgo_world_step(
            world, &solver_options, &result, &error);
        if (step_status != PGO_STATUS_OK) {
            fprintf(stderr, "step error: %s\n", error.message);
            exit_code = 1;
            break;
        }

        const pgo_status_t write_status = pgo_world_write_obj_frame(
            world, opts.output, &error);
        if (write_status != PGO_STATUS_OK) {
            fprintf(stderr, "write error: %s\n", error.message);
            exit_code = 1;
            break;
        }

        if (opts.abc_output[0] != '\0') {
            const pgo_status_t abc_status = pgo_world_write_abc_frame(
                world, opts.abc_output, opts.abc_fps, &error);
            if (abc_status != PGO_STATUS_OK) {
                fprintf(stderr, "abc error: %s\n", error.message);
                exit_code = 1;
                break;
            }
        }

        printf("frame=%llu status=%d iterations=%llu value=%g grad=%g\n",
               (unsigned long long)frame,
               (int)result.solver_status,
               (unsigned long long)result.iterations,
               result.final_value,
               result.final_gradient_norm);
    }

    pgo_world_destroy(world);
    cloth_grid_free(&grid);
    return exit_code;
}
