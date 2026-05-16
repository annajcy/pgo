#include "pgo_c/pgo.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- helpers ---- */

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

static void write_text_file(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    assert(file != NULL);
    const size_t written = fwrite(text, 1, strlen(text), file);
    assert(written == strlen(text));
    assert(fclose(file) == 0);
}

static const char* test_tmpdir(void) {
    const char* value = getenv("PGO_TEST_TMPDIR");
    return value == NULL || value[0] == '\0' ? "." : value;
}

/* ---- happy path ---- */

static void test_create_step_destroy(void) {
    const double positions[] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    };
    const uint64_t triangles[] = {0, 1, 2};
    const uint64_t pinned[] = {0};

    const pgo_mesh_view_t mesh = make_triangle_mesh(positions, 3, triangles, 1, pinned, 1);
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
}

/* ---- invalid mesh / buffer ---- */

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

/* ---- null arguments ---- */

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
    /* pgo_world_step accepts NULL options (uses defaults) */
    expect_status(pgo_world_step(world, NULL, &result, &error), PGO_STATUS_OK, &error);
    expect_status(pgo_world_step(world, &options, NULL, &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_write_obj_frame(NULL, "unused", &error), PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_world_write_obj_frame(world, NULL, &error), PGO_STATUS_INVALID_ARGUMENT, &error);

    pgo_world_destroy(world);
}

/* ---- invalid parameters ---- */

static void test_invalid_params_return_invalid_argument(void) {
    const double positions[] = {0,0,0, 1,0,0, 0,1,0};
    const uint64_t triangles[] = {0, 1, 2};
    pgo_mesh_view_t mesh = make_triangle_mesh(positions, 3, triangles, 1, NULL, 0);

    pgo_mass_spring_params_t params;
    pgo_error_t error;
    pgo_world_t* world = NULL;

    pgo_mass_spring_params_default(&params);
    params.stiffness = 0.0;
    pgo_error_clear(&error);
    expect_status(
        pgo_world_create_mass_spring(&mesh, &params, &world, &error),
        PGO_STATUS_INVALID_ARGUMENT,
        &error);
    assert(world == NULL);

    pgo_mass_spring_params_default(&params);
    params.dt = 0.0;
    pgo_error_clear(&error);
    expect_status(
        pgo_world_create_mass_spring(&mesh, &params, &world, &error),
        PGO_STATUS_INVALID_ARGUMENT,
        &error);
    assert(world == NULL);

    pgo_mass_spring_params_default(&params);
    params.gravity = -1.0;
    pgo_error_clear(&error);
    expect_status(
        pgo_world_create_mass_spring(&mesh, &params, &world, &error),
        PGO_STATUS_INVALID_ARGUMENT,
        &error);
    assert(world == NULL);
}

/* ---- invalid solver options ---- */

static void test_invalid_solver_options_return_invalid_argument(void) {
    const double positions[] = {0,0,0, 1,0,0, 0,1,0};
    const uint64_t triangles[] = {0, 1, 2};
    pgo_mesh_view_t mesh = make_triangle_mesh(positions, 3, triangles, 1, NULL, 0);

    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);

    pgo_error_t error;
    pgo_world_t* world = NULL;
    pgo_error_clear(&error);
    expect_status(
        pgo_world_create_mass_spring(&mesh, &params, &world, &error),
        PGO_STATUS_OK,
        &error);

    pgo_solver_options_t options;
    pgo_step_result_t result;

    pgo_solver_options_default(&options);
    options.max_iterations = 0;
    pgo_error_clear(&error);
    expect_status(pgo_world_step(world, &options, &result, &error),
                  PGO_STATUS_INVALID_ARGUMENT, &error);

    pgo_solver_options_default(&options);
    options.gradient_tolerance = 0.0;
    pgo_error_clear(&error);
    expect_status(pgo_world_step(world, &options, &result, &error),
                  PGO_STATUS_INVALID_ARGUMENT, &error);

    pgo_solver_options_default(&options);
    options.initial_regularization = 0.0;
    pgo_error_clear(&error);
    expect_status(pgo_world_step(world, &options, &result, &error),
                  PGO_STATUS_INVALID_ARGUMENT, &error);

    pgo_world_destroy(world);
}

/* ---- OBJ IO: from_obj ---- */

static void test_from_obj_success(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/success.obj", test_tmpdir());
    remove(path);
    write_text_file(path,
                    "v 0 0 0\n"
                    "v 1 0 0\n"
                    "v 0 1 0\n"
                    "f 1 2 3\n");

    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);

    pgo_error_t error;
    pgo_world_t* world = NULL;
    pgo_error_clear(&error);
    pgo_status_t status = pgo_world_create_mass_spring_from_obj(
        path, &params, NULL, 0, &world, &error);
    expect_status(status, PGO_STATUS_OK, &error);
    assert(world != NULL);

    pgo_world_destroy(world);
}

static void test_from_obj_missing_file_returns_io_error(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/nonexistent.obj", test_tmpdir());

    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);

    pgo_error_t error;
    pgo_world_t* world = NULL;
    pgo_error_clear(&error);
    expect_status(
        pgo_world_create_mass_spring_from_obj(
            path, &params, NULL, 0, &world, &error),
        PGO_STATUS_IO_ERROR,
        &error);
    assert(world == NULL);
}

/* ---- OBJ IO: write_obj_frame ---- */

static void test_write_obj_frame_success(void) {
    const double positions[] = {0,0,0, 1,0,0, 0,1,0};
    const uint64_t triangles[] = {0, 1, 2};
    pgo_mesh_view_t mesh = make_triangle_mesh(positions, 3, triangles, 1, NULL, 0);

    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);

    pgo_error_t error;
    pgo_world_t* world = NULL;
    pgo_error_clear(&error);
    expect_status(
        pgo_world_create_mass_spring(&mesh, &params, &world, &error),
        PGO_STATUS_OK,
        &error);

    char output_dir[512];
    snprintf(output_dir, sizeof(output_dir), "%s/frames", test_tmpdir());

    pgo_error_clear(&error);
    expect_status(
        pgo_world_write_obj_frame(world, output_dir, &error),
        PGO_STATUS_OK,
        &error);

    /* check at least one .obj file was written */
    char expected_file[512];
    snprintf(expected_file, sizeof(expected_file), "%s/frame_0000.obj", output_dir);
    FILE* f = fopen(expected_file, "rb");
    assert(f != NULL);
    fclose(f);

    pgo_world_destroy(world);
}

static void test_write_obj_frame_io_error(void) {
    const double positions[] = {0,0,0, 1,0,0, 0,1,0};
    const uint64_t triangles[] = {0, 1, 2};
    pgo_mesh_view_t mesh = make_triangle_mesh(positions, 3, triangles, 1, NULL, 0);

    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);

    pgo_error_t error;
    pgo_world_t* world = NULL;
    pgo_error_clear(&error);
    expect_status(
        pgo_world_create_mass_spring(&mesh, &params, &world, &error),
        PGO_STATUS_OK,
        &error);

    /* create a regular file that will block create_directories */
    char blocked[512];
    snprintf(blocked, sizeof(blocked), "%s/blocked_dir", test_tmpdir());
    remove(blocked);
    write_text_file(blocked, "");

    pgo_error_clear(&error);
#if !defined(_WIN32)
    expect_status(
        pgo_world_write_obj_frame(world, blocked, &error),
        PGO_STATUS_IO_ERROR,
        &error);
#else
    /* On Windows, create_directories over an existing regular file has
       historically inconsistent behavior. Skip and test empty string instead. */
    (void)blocked;
    pgo_error_clear(&error);
    expect_status(
        pgo_world_write_obj_frame(world, "", &error),
        PGO_STATUS_IO_ERROR,
        &error);
#endif

    pgo_world_destroy(world);
}

static void test_read_obj_mesh_success(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/read_success.obj", test_tmpdir());
    remove(path);
    write_text_file(path,
                    "v 0 0 0\n"
                    "v 1 0 0\n"
                    "v 0 1 0\n"
                    "f 1 2 3\n");

    pgo_error_t error;
    pgo_obj_mesh_t mesh;
    pgo_error_clear(&error);
    expect_status(pgo_read_obj_mesh(path, &mesh, &error), PGO_STATUS_OK, &error);
    assert(mesh.vertex_count == 3);
    assert(mesh.triangle_count == 1);
    assert(mesh.positions_xyz[3] == 1.0);   /* vertex 1, x */
    assert(mesh.positions_xyz[7] == 1.0);   /* vertex 2, y */
    assert(mesh.triangles[0] == 0);
    assert(mesh.triangles[1] == 1);
    assert(mesh.triangles[2] == 2);
    pgo_obj_mesh_free(&mesh);
}

static void test_read_obj_mesh_missing_file(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/nonexistent_read.obj", test_tmpdir());

    pgo_error_t error;
    pgo_obj_mesh_t mesh;
    pgo_error_clear(&error);
    expect_status(
        pgo_read_obj_mesh(path, &mesh, &error), PGO_STATUS_IO_ERROR, &error);
}

static void test_read_obj_mesh_null_args(void) {
    pgo_error_t error;
    pgo_obj_mesh_t mesh;
    pgo_error_clear(&error);
    expect_status(pgo_read_obj_mesh(NULL, &mesh, &error),
                  PGO_STATUS_INVALID_ARGUMENT, &error);
    expect_status(pgo_read_obj_mesh("unused", NULL, &error),
                  PGO_STATUS_INVALID_ARGUMENT, &error);
}

static void test_obj_mesh_free_null_is_safe(void) {
    pgo_obj_mesh_free(NULL);
}

/* ---- driver ---- */

int main(void) {
    test_create_step_destroy();
    test_invalid_triangles_return_invalid_argument();
    test_pinned_and_output_buffer_validation();
    test_null_arguments_return_invalid_argument();
    test_invalid_params_return_invalid_argument();
    test_invalid_solver_options_return_invalid_argument();
    test_from_obj_success();
    test_from_obj_missing_file_returns_io_error();
    test_write_obj_frame_success();
    test_write_obj_frame_io_error();
    test_read_obj_mesh_success();
    test_read_obj_mesh_missing_file();
    test_read_obj_mesh_null_args();
    test_obj_mesh_free_null_is_safe();
    return 0;
}
