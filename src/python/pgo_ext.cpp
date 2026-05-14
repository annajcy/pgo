#include "pgo_c/pgo.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace {

class WorldHandle {
public:
    explicit WorldHandle(pgo_world_t* world)
        : m_world{world} {}

    WorldHandle(const WorldHandle&) = delete;
    WorldHandle& operator=(const WorldHandle&) = delete;

    WorldHandle(WorldHandle&& other) noexcept
        : m_world{other.m_world} {
        other.m_world = nullptr;
    }

    WorldHandle& operator=(WorldHandle&& other) noexcept {
        if (this != &other) {
            pgo_world_destroy(m_world);
            m_world = other.m_world;
            other.m_world = nullptr;
        }
        return *this;
    }

    ~WorldHandle() {
        pgo_world_destroy(m_world);
    }

    pgo_world_t* get() const {
        return m_world;
    }

private:
    pgo_world_t* m_world = nullptr;
};

void throw_on_error(const pgo_status_t status, const pgo_error_t& error) {
    if (status == PGO_STATUS_OK) {
        return;
    }
    throw std::runtime_error(error.message[0] == '\0' ? "pgo C API call failed" : error.message);
}

using FloatArray = nb::ndarray<nb::numpy, const double, nb::shape<-1, 3>, nb::c_contig>;
using UIntArray2 = nb::ndarray<nb::numpy, const std::uint64_t, nb::shape<-1, 3>, nb::c_contig>;
using UIntArray1 = nb::ndarray<nb::numpy, const std::uint64_t, nb::shape<-1>, nb::c_contig>;
using MutableFloatArray = nb::ndarray<nb::numpy, double, nb::shape<-1, 3>, nb::c_contig>;

std::shared_ptr<WorldHandle> create_world_from_arrays(
    FloatArray vertices,
    UIntArray2 triangles,
    UIntArray1 pinned_vertices,
    double stiffness,
    double gravity,
    double dt) {
    const pgo_mesh_view_t mesh{
        vertices.data(),
        static_cast<std::uint64_t>(vertices.shape(0)),
        triangles.data(),
        static_cast<std::uint64_t>(triangles.shape(0)),
        pinned_vertices.data(),
        static_cast<std::uint64_t>(pinned_vertices.shape(0)),
    };
    const pgo_mass_spring_params_t params{stiffness, gravity, dt};

    pgo_error_t error;
    pgo_error_clear(&error);
    pgo_world_t* raw = nullptr;
    throw_on_error(pgo_world_create_mass_spring(&mesh, &params, &raw, &error), error);
    return std::make_shared<WorldHandle>(raw);
}

std::shared_ptr<WorldHandle> create_world_from_obj(
    const std::string& path,
    UIntArray1 pinned_vertices,
    double stiffness,
    double gravity,
    double dt) {
    const pgo_mass_spring_params_t params{stiffness, gravity, dt};

    pgo_error_t error;
    pgo_error_clear(&error);
    pgo_world_t* raw = nullptr;
    throw_on_error(
        pgo_world_create_mass_spring_from_obj(
            path.c_str(),
            &params,
            pinned_vertices.data(),
            static_cast<std::uint64_t>(pinned_vertices.shape(0)),
            &raw,
            &error),
        error);
    return std::make_shared<WorldHandle>(raw);
}

std::uint64_t vertex_count(const std::shared_ptr<WorldHandle>& world) {
    pgo_error_t error;
    pgo_error_clear(&error);
    std::uint64_t count = 0;
    throw_on_error(pgo_world_vertex_count(world->get(), &count, &error), error);
    return count;
}

void copy_positions(const std::shared_ptr<WorldHandle>& world, MutableFloatArray out) {
    pgo_error_t error;
    pgo_error_clear(&error);
    throw_on_error(
        pgo_world_copy_positions(
            world->get(),
            out.data(),
            static_cast<std::uint64_t>(out.shape(0) * 3),
            &error),
        error);
}

nb::tuple step(
    const std::shared_ptr<WorldHandle>& world,
    std::uint64_t max_iterations,
    double gradient_tolerance,
    double initial_regularization,
    bool commit_on_failure) {
    pgo_error_t error;
    pgo_error_clear(&error);
    const pgo_solver_options_t options{
        max_iterations,
        gradient_tolerance,
        initial_regularization,
        commit_on_failure ? 1 : 0,
    };
    pgo_step_result_t result{};
    throw_on_error(pgo_world_step(world->get(), &options, &result, &error), error);

    const char* status = "line_search_failed";
    switch (result.solver_status) {
    case PGO_SOLVER_CONVERGED:
        status = "converged";
        break;
    case PGO_SOLVER_MAX_ITERATIONS:
        status = "max_iterations";
        break;
    case PGO_SOLVER_REGULARIZATION_FAILED:
        status = "regularization_failed";
        break;
    case PGO_SOLVER_LINE_SEARCH_FAILED:
        status = "line_search_failed";
        break;
    }

    return nb::make_tuple(
        status,
        result.iterations,
        result.final_value,
        result.final_gradient_norm);
}

void write_obj_frame(const std::shared_ptr<WorldHandle>& world, const std::string& output_dir) {
    pgo_error_t error;
    pgo_error_clear(&error);
    throw_on_error(pgo_world_write_obj_frame(world->get(), output_dir.c_str(), &error), error);
}

} // namespace

NB_MODULE(_pgo_ext, m) {
    nb::class_<WorldHandle>(m, "WorldHandle");

    m.def("version", &pgo_version);
    m.def("create_world_from_arrays", &create_world_from_arrays);
    m.def("create_world_from_obj", &create_world_from_obj);
    m.def("vertex_count", &vertex_count);
    m.def("copy_positions", &copy_positions);
    m.def("step", &step);
    m.def("write_obj_frame", &write_obj_frame);
}
