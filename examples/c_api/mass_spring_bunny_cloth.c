#include "pgo_c/pgo.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t* values;
    uint64_t count;
    uint64_t capacity;
} pinned_list_t;

typedef struct {
    char input[512];
    char output[512];
    char abc_output[512];
    double abc_fps;
    uint64_t frames;
    double stiffness;
    double gravity;
    double dt;
    uint64_t max_iterations;
    pinned_list_t pinned;
} bunny_options_t;

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s [OPTIONS]\n"
            "  --input <path>         OBJ input file (default assets/model/bunny.obj)\n"
            "  --output <path>        Output directory (default output/example/c_api/mass_spring/bunny)\n"
            "  --export-abc <path>   Alembic .abc output file\n"
            "  --abc-fps <double>    FPS for Alembic export (default 24)\n"
            "  --frames <uint>        Number of frames (default 10)\n"
            "  --stiffness <double>   Spring stiffness (default 200000)\n"
            "  --gravity <double>     Gravity magnitude (default 9.81)\n"
            "  --dt <double>          Timestep size (default 0.001)\n"
            "  --pinned <csv>         Comma-separated uint64 vertex indices\n"
            "  --pinned-file <path>   File with whitespace/newline-separated uint64 vertex indices\n"
            "  --max-iterations <uint> Max Newton iterations (default 500)\n"
            "  --help                 Show this message\n",
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

static void pinned_list_init(pinned_list_t* list) {
    list->values = NULL;
    list->count = 0;
    list->capacity = 0;
}

static void pinned_list_free(pinned_list_t* list) {
    free(list->values);
    pinned_list_init(list);
}

static int pinned_list_push(pinned_list_t* list, uint64_t v) {
    if (list->count == list->capacity) {
        uint64_t new_cap = list->capacity == 0 ? 16 : list->capacity * 2;
        uint64_t* new_values = (uint64_t*)realloc(list->values, new_cap * sizeof(uint64_t));
        if (new_values == NULL) return -1;
        list->values = new_values;
        list->capacity = new_cap;
    }
    list->values[list->count++] = v;
    return 0;
}

static int parse_pinned_csv(const char* s, pinned_list_t* list) {
    const char* p = s;
    while (*p) {
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == '\0') break;
        char* end = NULL;
        unsigned long long v = strtoull(p, &end, 10);
        if (end == p) return -1;
        if (pinned_list_push(list, (uint64_t)v) != 0) return -1;
        p = end;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == ',') ++p;
        else if (*p != '\0') return -1;
    }
    return 0;
}

static int parse_pinned_file(const char* path, pinned_list_t* list) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "error: cannot open pinned file '%s'\n", path);
        return -1;
    }
    uint64_t v;
    int ret = 0;
    while (fscanf(f, "%llu", (unsigned long long*)&v) == 1) {
        if (pinned_list_push(list, v) != 0) {
            ret = -1;
            break;
        }
    }
    fclose(f);
    return ret;
}

static void bunny_options_default(bunny_options_t* opts) {
    memcpy(opts->input, "assets/model/bunny.obj", 22);
    opts->input[21] = '\0';
    memcpy(opts->output, "output/example/c_api/mass_spring/bunny", 39);
    opts->output[38] = '\0';
    opts->abc_output[0] = '\0';
    opts->abc_fps = 24.0;
    opts->frames = 10;
    opts->stiffness = 200000.0;
    opts->gravity = 9.81;
    opts->dt = 0.001;
    opts->max_iterations = 500;
    pinned_list_init(&opts->pinned);
}

static int parse_options(int argc, char** argv, bunny_options_t* opts) {
    bunny_options_default(opts);
    int has_pinned_csv = 0;
    int has_pinned_file = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            size_t len = strlen(argv[++i]);
            if (len >= sizeof(opts->input)) {
                fprintf(stderr, "error: --input path too long\n");
                return -1;
            }
            memcpy(opts->input, argv[i], len + 1);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            size_t len = strlen(argv[++i]);
            if (len >= sizeof(opts->output)) {
                fprintf(stderr, "error: --output path too long\n");
                return -1;
            }
            memcpy(opts->output, argv[i], len + 1);
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            if (parse_uint64(argv[++i], &opts->frames) != 0 || opts->frames < 1) {
                fprintf(stderr, "error: --frames must be >= 1\n");
                return -1;
            }
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
        } else if (strcmp(argv[i], "--pinned") == 0 && i + 1 < argc) {
            if (parse_pinned_csv(argv[++i], &opts->pinned) != 0) {
                fprintf(stderr, "error: invalid --pinned CSV\n");
                return -1;
            }
            has_pinned_csv = 1;
        } else if (strcmp(argv[i], "--pinned-file") == 0 && i + 1 < argc) {
            if (parse_pinned_file(argv[++i], &opts->pinned) != 0) {
                return -1;
            }
            has_pinned_file = 1;
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

    if (has_pinned_csv && has_pinned_file) {
        fprintf(stderr, "error: cannot use both --pinned and --pinned-file\n");
        return -1;
    }
    return 0;
}

int main(int argc, char** argv) {
    bunny_options_t opts;
    if (parse_options(argc, argv, &opts) != 0) {
        pinned_list_free(&opts.pinned);
        return 1;
    }

    pgo_mass_spring_params_t params;
    pgo_mass_spring_params_default(&params);
    params.stiffness = opts.stiffness;
    params.gravity = opts.gravity;
    params.dt = opts.dt;

    pgo_error_t error;
    pgo_world_t* world = NULL;
    const pgo_status_t status = pgo_world_create_mass_spring_from_obj(
        opts.input,
        &params,
        opts.pinned.values,
        opts.pinned.count,
        &world,
        &error);
    if (status != PGO_STATUS_OK) {
        fprintf(stderr, "error: %s\n", error.message);
        pinned_list_free(&opts.pinned);
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
    pinned_list_free(&opts.pinned);
    return exit_code;
}
