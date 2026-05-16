#include "pgo_c/pgo.h"

#include "pgo/core/dof/dirichlet_boundary.hpp"
#include "pgo/core/dof/dof_layout.hpp"
#include "pgo/core/dof/reduced_dof_map.hpp"
#include "pgo/core/energy/assembled_energy.hpp"
#include "pgo/core/energy/constant_force_energy.hpp"
#include "pgo/core/energy/energy_sum.hpp"
#include "pgo/core/energy/mass_spring_local_energy_provider.hpp"
#include "pgo/core/geometry/rest_mesh.hpp"
#include "pgo/core/integrator/backward_euler.hpp"
#include "pgo/core/integrator/dynamic_state.hpp"
#include "pgo/core/solver/solver_options.hpp"
#include "pgo/core/solver/status_name.hpp"
#include "pgo/core/storage/host_buffer.hpp"
#include "pgo/io/obj_reader.hpp"
#include "pgo/io/obj_writer.hpp"

#if defined(PGO_ENABLE_ALEMBIC)
#include "pgo/io/abc_writer.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class IoError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void set_error(pgo_error_t* error, const pgo_status_t status, const std::string_view message) noexcept {
    if (error == nullptr) {
        return;
    }
    error->status = status;
    const std::size_t n = std::min(message.size(), sizeof(error->message) - 1);
    std::copy_n(message.data(), n, error->message);
    error->message[n] = '\0';
}

template <class F>
pgo_status_t call_c_api(pgo_error_t* error, F&& f) noexcept {
    try {
        if (error != nullptr) {
            pgo_error_clear(error);
        }
        return f();
    } catch (const std::bad_alloc&) {
        set_error(error, PGO_STATUS_INTERNAL_ERROR, "allocation failed");
        return PGO_STATUS_INTERNAL_ERROR;
    } catch (const std::invalid_argument& e) {
        set_error(error, PGO_STATUS_INVALID_ARGUMENT, e.what());
        return PGO_STATUS_INVALID_ARGUMENT;
    } catch (const IoError& e) {
        set_error(error, PGO_STATUS_IO_ERROR, e.what());
        return PGO_STATUS_IO_ERROR;
    } catch (const std::filesystem::filesystem_error& e) {
        set_error(error, PGO_STATUS_IO_ERROR, e.what());
        return PGO_STATUS_IO_ERROR;
    } catch (const std::runtime_error& e) {
        set_error(error, PGO_STATUS_INTERNAL_ERROR, e.what());
        return PGO_STATUS_INTERNAL_ERROR;
    } catch (const std::exception& e) {
        set_error(error, PGO_STATUS_INTERNAL_ERROR, e.what());
        return PGO_STATUS_INTERNAL_ERROR;
    } catch (...) {
        set_error(error, PGO_STATUS_INTERNAL_ERROR, "unknown C++ exception");
        return PGO_STATUS_INTERNAL_ERROR;
    }
}

pgo_mass_spring_params_t default_mass_spring_params() {
    return pgo_mass_spring_params_t{100.0, 9.8, 0.016};
}

pgo_solver_options_t default_solver_options() {
    return pgo_solver_options_t{100, 1e-5, 1e-4, 0};
}

void validate_params(const pgo_mass_spring_params_t& params) {
    if (!(params.stiffness > 0.0)) {
        throw std::invalid_argument{"stiffness must be positive"};
    }
    if (!(params.dt > 0.0)) {
        throw std::invalid_argument{"dt must be positive"};
    }
    if (!(params.gravity >= 0.0)) {
        throw std::invalid_argument{"gravity must be non-negative"};
    }
}

void validate_solver_options(const pgo_solver_options_t& options) {
    if (options.max_iterations == 0) {
        throw std::invalid_argument{"max_iterations must be positive"};
    }
    if (!(options.gradient_tolerance > 0.0)) {
        throw std::invalid_argument{"gradient_tolerance must be positive"};
    }
    if (!(options.initial_regularization > 0.0)) {
        throw std::invalid_argument{"initial_regularization must be positive"};
    }
}

pgo::solver::NewtonOptions<double> to_newton_options(const pgo_solver_options_t& options) {
    pgo::solver::NewtonOptions<double> newton_options;
    newton_options.max_iterations = static_cast<std::size_t>(options.max_iterations);
    newton_options.gradient_tolerance = options.gradient_tolerance;
    newton_options.initial_regularization = options.initial_regularization;
    return newton_options;
}

pgo_solver_status_t to_c_solver_status(const pgo::solver::SolverStatus status) {
    switch (status) {
    case pgo::solver::SolverStatus::converged:
        return PGO_SOLVER_CONVERGED;
    case pgo::solver::SolverStatus::max_iterations:
        return PGO_SOLVER_MAX_ITERATIONS;
    case pgo::solver::SolverStatus::regularization_failed:
        return PGO_SOLVER_REGULARIZATION_FAILED;
    case pgo::solver::SolverStatus::line_search_failed:
        return PGO_SOLVER_LINE_SEARCH_FAILED;
    }
    return PGO_SOLVER_LINE_SEARCH_FAILED;
}

std::size_t checked_arity_count(const std::uint64_t count, const std::uint64_t arity) {
    constexpr auto max_size = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (count > max_size / arity) {
        throw std::invalid_argument{"input count is too large for this platform"};
    }
    return static_cast<std::size_t>(count * arity);
}

pgo::geometry::VertexIndex checked_vertex_index(const std::uint64_t vertex,
                                                const std::uint64_t vertex_count) {
    if (vertex >= vertex_count) {
        throw std::invalid_argument{"triangle vertex index is out of range"};
    }
    if (vertex > std::numeric_limits<pgo::geometry::VertexIndex>::max()) {
        throw std::invalid_argument{"triangle vertex index exceeds pgo VertexIndex range"};
    }
    return static_cast<pgo::geometry::VertexIndex>(vertex);
}

std::vector<std::size_t> pinned_vertices_from_array(
    const std::uint64_t* pinned_vertices,
    const std::uint64_t pinned_vertex_count,
    const std::uint64_t vertex_count) {
    const std::size_t pinned_count = checked_arity_count(pinned_vertex_count, 1);
    std::vector<std::size_t> pinned;
    pinned.reserve(pinned_count);
    for (std::uint64_t i = 0; i < pinned_vertex_count; ++i) {
        const auto vertex = pinned_vertices[i];
        if (vertex >= vertex_count) {
            throw std::invalid_argument{"pinned vertex index is out of range"};
        }
        pinned.push_back(static_cast<std::size_t>(vertex));
    }
    return pinned;
}

std::vector<std::size_t> pinned_vertices_from_view(const pgo_mesh_view_t& view) {
    return pinned_vertices_from_array(view.pinned_vertices, view.pinned_vertex_count, view.vertex_count);
}

pgo::geometry::RestMesh<double, 3> make_rest_mesh_from_view(const pgo_mesh_view_t& view) {
    using VertexIndex = pgo::geometry::VertexIndex;

    if (view.vertex_count == 0) {
        throw std::invalid_argument{"vertex_count must be positive"};
    }
    if (view.vertex_count - 1 > std::numeric_limits<VertexIndex>::max()) {
        throw std::invalid_argument{"vertex_count exceeds pgo VertexIndex range"};
    }
    const std::size_t position_scalars = checked_arity_count(view.vertex_count, 3);
    const std::size_t triangle_scalars = checked_arity_count(view.triangle_count, 3);

    pgo::storage::HostBuffer<double> positions;
    positions.reserve(position_scalars);
    for (std::size_t i = 0; i < position_scalars; ++i) {
        positions.push_back(view.positions_xyz[i]);
    }

    pgo::storage::HostBuffer<VertexIndex> faces;
    faces.reserve(triangle_scalars);

    std::set<std::pair<VertexIndex, VertexIndex>> unique_edges;
    auto add_edge = [&](const std::uint64_t a, const std::uint64_t b) {
        const auto va = checked_vertex_index(a, view.vertex_count);
        const auto vb = checked_vertex_index(b, view.vertex_count);
        unique_edges.insert(std::minmax(va, vb));
    };

    for (std::uint64_t tri = 0; tri < view.triangle_count; ++tri) {
        const std::uint64_t a = view.triangles[tri * 3 + 0];
        const std::uint64_t b = view.triangles[tri * 3 + 1];
        const std::uint64_t c = view.triangles[tri * 3 + 2];
        if (a == b || b == c || c == a) {
            throw std::invalid_argument{"triangle must reference three distinct vertices"};
        }
        add_edge(a, b);
        add_edge(b, c);
        add_edge(c, a);
        faces.push_back(checked_vertex_index(a, view.vertex_count));
        faces.push_back(checked_vertex_index(b, view.vertex_count));
        faces.push_back(checked_vertex_index(c, view.vertex_count));
    }

    pgo::storage::HostBuffer<VertexIndex> edges;
    edges.reserve(unique_edges.size() * 2);
    for (const auto& [a, b] : unique_edges) {
        edges.push_back(a);
        edges.push_back(b);
    }

    return pgo::geometry::RestMesh<double, 3>{std::move(positions), std::move(edges), std::move(faces)};
}

pgo::dof::DirichletBoundary<double> make_boundary(const pgo::dof::DofLayout<3>& layout,
                                                   const std::vector<std::size_t>& pinned_vertices) {
    pgo::dof::DirichletBoundary<double> boundary;
    for (const std::size_t vertex : pinned_vertices) {
        if (vertex >= layout.num_vertices()) {
            throw std::invalid_argument{"pinned vertex index is out of range"};
        }
        pgo::dof::fix_vertex(boundary, layout, vertex);
    }
    return boundary;
}

pgo::math::DVec<double> make_gravity_force(const pgo::math::DVec<double>& lumped_mass, const double gravity) {
    pgo::math::DVec<double> force = pgo::math::DVec<double>::Zero(lumped_mass.size());
    for (pgo::math::DenseIndex dof = 0; dof < lumped_mass.size(); ++dof) {
        if (static_cast<std::size_t>(dof % 3) == 1) {
            force[dof] = -gravity * lumped_mass[dof];
        }
    }
    return force;
}

class MassSpringWorld3d {
    using Provider = pgo::energy::MassSpringLocalEnergyProvider<double, 3>;
    using Potential = pgo::energy::AssembledEnergyView<double, Provider>;

public:
    MassSpringWorld3d(pgo::geometry::RestMesh<double, 3> mesh,
                      std::vector<std::size_t> pinned_vertices,
                      const pgo_mass_spring_params_t& params)
        : m_mesh{std::move(mesh)},
          m_layout{m_mesh.num_vertices()},
          m_boundary{make_boundary(m_layout, pinned_vertices)},
          m_dof_map{m_layout.num_dofs(), m_boundary},
          m_lumped_mass{pgo::math::dense_index(m_layout.num_dofs())},
          m_provider{m_mesh, params.stiffness},
          m_potential{m_provider},
          m_gravity{params.gravity},
          m_dt{params.dt} {
        m_lumped_mass.setOnes();
        m_state.u.setZero(pgo::math::dense_index(m_layout.num_dofs()));
        m_state.v.setZero(pgo::math::dense_index(m_layout.num_dofs()));
        m_state.a.setZero(pgo::math::dense_index(m_layout.num_dofs()));
    }

    std::size_t vertex_count() const {
        return m_mesh.num_vertices();
    }

    void copy_positions(double* out_positions_xyz, const std::uint64_t out_scalar_count) const {
        const std::size_t expected = m_layout.num_dofs();
        if (out_scalar_count < expected) {
            throw std::invalid_argument{"output positions buffer is too small"};
        }
        for (std::size_t i = 0; i < expected; ++i) {
            out_positions_xyz[i] = m_mesh.rest_positions()[i] + m_state.u[pgo::math::dense_index(i)];
        }
    }

    pgo_step_result_t step(const pgo_solver_options_t& options) {
        validate_solver_options(options);
        const auto newton_options = to_newton_options(options);
        const auto gravity_force = make_gravity_force(m_lumped_mass, m_gravity);
        const pgo::energy::ConstantForceEnergyView<double> gravity_energy{gravity_force};
        const pgo::energy::EnergySumView<double, Potential, decltype(gravity_energy)> step_energy{
            m_potential, gravity_energy};

        const auto result = m_integrator.step(
            step_energy,
            m_lumped_mass,
            m_dof_map,
            m_state,
            m_dt,
            newton_options,
            options.commit_on_failure != 0);

        return pgo_step_result_t{
            to_c_solver_status(result.status),
            static_cast<std::uint64_t>(result.solver_iterations),
            result.final_value,
            result.final_gradient_norm,
        };
    }

    void write_obj_frame(const std::filesystem::path& output_dir) const {
        try {
            if (!m_writer || m_writer_output_dir != output_dir) {
                m_writer_output_dir = output_dir;
                m_writer = std::make_unique<pgo::io::ObjWriter3d>(m_writer_output_dir);
            }
            static_cast<void>(m_writer->write_frame(m_mesh, m_state.u));
        } catch (const std::exception& e) {
            throw IoError{e.what()};
        }
    }

#if defined(PGO_ENABLE_ALEMBIC)
    void write_abc_frame(const std::filesystem::path& output_path, double fps) const {
        try {
            if (!m_abc_writer || m_abc_writer_path != output_path) {
                const auto nf = m_mesh.num_faces();
                std::vector<std::uint32_t> face_indices;
                face_indices.reserve(nf * 3);
                std::vector<int> face_counts;
                face_counts.reserve(nf);
                for (std::size_t f = 0; f < nf; ++f) {
                    for (std::size_t lv = 0; lv < 3; ++lv) {
                        face_indices.push_back(
                            static_cast<std::uint32_t>(m_mesh.face_vertex(f, lv)));
                    }
                    face_counts.push_back(3);
                }
                m_abc_writer_path = output_path;
                m_abc_writer = std::make_unique<pgo::io::AbcWriter3d>(
                    m_abc_writer_path, fps, face_indices, face_counts);
            }
            m_abc_writer->write_frame(m_mesh, m_state.u);
        } catch (const std::exception& e) {
            throw IoError{e.what()};
        }
    }
#endif

private:
    pgo::geometry::RestMesh<double, 3> m_mesh;
    pgo::dof::DofLayout<3> m_layout;
    pgo::dof::DirichletBoundary<double> m_boundary;
    pgo::dof::ReducedDofMap<double> m_dof_map;
    pgo::math::DVec<double> m_lumped_mass;
    Provider m_provider;
    Potential m_potential;
    double m_gravity;
    double m_dt;
    pgo::integrator::BackwardEuler<double> m_integrator;
    pgo::integrator::DynamicState<double> m_state;
    mutable std::filesystem::path m_writer_output_dir;
    mutable std::unique_ptr<pgo::io::ObjWriter3d> m_writer;
#if defined(PGO_ENABLE_ALEMBIC)
    mutable std::filesystem::path m_abc_writer_path;
    mutable std::unique_ptr<pgo::io::AbcWriter3d> m_abc_writer;
#endif
};

} // namespace

struct pgo_world_t {
    std::unique_ptr<MassSpringWorld3d> impl;
};

extern "C" {

const char* pgo_version(void) {
    return "0.1.0";
}

void pgo_error_clear(pgo_error_t* error) {
    if (error == nullptr) {
        return;
    }
    error->status = PGO_STATUS_OK;
    error->message[0] = '\0';
}

void pgo_mass_spring_params_default(pgo_mass_spring_params_t* params) {
    if (params == nullptr) {
        return;
    }
    *params = default_mass_spring_params();
}

void pgo_solver_options_default(pgo_solver_options_t* options) {
    if (options == nullptr) {
        return;
    }
    *options = default_solver_options();
}

pgo_status_t pgo_world_create_mass_spring(
    const pgo_mesh_view_t* mesh,
    const pgo_mass_spring_params_t* params,
    pgo_world_t** out_world,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (mesh == nullptr || params == nullptr || out_world == nullptr) {
            throw std::invalid_argument{"mesh, params, and out_world must be non-null"};
        }
        *out_world = nullptr;
        validate_params(*params);
        if (mesh->positions_xyz == nullptr || mesh->vertex_count == 0) {
            throw std::invalid_argument{"mesh positions must be non-null and non-empty"};
        }
        if (mesh->triangles == nullptr || mesh->triangle_count == 0) {
            throw std::invalid_argument{"mesh triangles must be non-null and non-empty"};
        }
        if (mesh->pinned_vertex_count > 0 && mesh->pinned_vertices == nullptr) {
            throw std::invalid_argument{"pinned_vertices must be non-null when pinned_vertex_count is positive"};
        }

        auto world = std::make_unique<pgo_world_t>();
        world->impl = std::make_unique<MassSpringWorld3d>(
            make_rest_mesh_from_view(*mesh),
            pinned_vertices_from_view(*mesh),
            *params);

        *out_world = world.release();
        return PGO_STATUS_OK;
    });
}

pgo_status_t pgo_world_create_mass_spring_from_obj(
    const char* path,
    const pgo_mass_spring_params_t* params,
    const std::uint64_t* pinned_vertices,
    std::uint64_t pinned_vertex_count,
    pgo_world_t** out_world,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (path == nullptr || params == nullptr || out_world == nullptr) {
            throw std::invalid_argument{"path, params, and out_world must be non-null"};
        }
        if (pinned_vertex_count > 0 && pinned_vertices == nullptr) {
            throw std::invalid_argument{"pinned_vertices must be non-null when pinned_vertex_count is positive"};
        }
        validate_params(*params);

        *out_world = nullptr;
        pgo::geometry::RestMesh<double, 3> mesh = [&]() {
            try {
                return pgo::io::read_obj_rest_mesh_3d(std::filesystem::path{path});
            } catch (const std::exception& e) {
                throw IoError{e.what()};
            }
        }();
        const auto vertex_count = static_cast<std::uint64_t>(mesh.num_vertices());

        auto world = std::make_unique<pgo_world_t>();
        world->impl = std::make_unique<MassSpringWorld3d>(
            std::move(mesh),
            pinned_vertices_from_array(pinned_vertices, pinned_vertex_count, vertex_count),
            *params);

        *out_world = world.release();
        return PGO_STATUS_OK;
    });
}

void pgo_world_destroy(pgo_world_t* world) {
    delete world;
}

pgo_status_t pgo_world_vertex_count(
    const pgo_world_t* world,
    std::uint64_t* out_vertex_count,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (world == nullptr || out_vertex_count == nullptr) {
            throw std::invalid_argument{"world and out_vertex_count must be non-null"};
        }
        *out_vertex_count = static_cast<std::uint64_t>(world->impl->vertex_count());
        return PGO_STATUS_OK;
    });
}

pgo_status_t pgo_world_copy_positions(
    const pgo_world_t* world,
    double* out_positions_xyz,
    std::uint64_t out_position_scalar_count,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (world == nullptr || out_positions_xyz == nullptr) {
            throw std::invalid_argument{"world and out_positions_xyz must be non-null"};
        }
        world->impl->copy_positions(out_positions_xyz, out_position_scalar_count);
        return PGO_STATUS_OK;
    });
}

pgo_status_t pgo_world_step(
    pgo_world_t* world,
    const pgo_solver_options_t* options,
    pgo_step_result_t* out_result,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (world == nullptr || out_result == nullptr) {
            throw std::invalid_argument{"world and out_result must be non-null"};
        }

        const pgo_solver_options_t default_options = default_solver_options();
        *out_result = world->impl->step(options == nullptr ? default_options : *options);
        return PGO_STATUS_OK;
    });
}

pgo_status_t pgo_world_write_obj_frame(
    const pgo_world_t* world,
    const char* output_dir,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (world == nullptr || output_dir == nullptr) {
            throw std::invalid_argument{"world and output_dir must be non-null"};
        }
        world->impl->write_obj_frame(output_dir);
        return PGO_STATUS_OK;
    });
}

pgo_status_t pgo_world_write_abc_frame(
    pgo_world_t* world,
    const char* output_path,
    double fps,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (world == nullptr || output_path == nullptr) {
            throw std::invalid_argument{"world and output_path must be non-null"};
        }
        if (!(fps > 0.0)) {
            throw std::invalid_argument{"fps must be positive"};
        }
#if defined(PGO_ENABLE_ALEMBIC)
        world->impl->write_abc_frame(output_path, fps);
#else
        (void)world;
        (void)output_path;
        (void)fps;
        throw std::runtime_error{"Alembic export not available in this build"};
#endif
        return PGO_STATUS_OK;
    });
}

pgo_status_t pgo_read_obj_mesh(
    const char* path,
    pgo_obj_mesh_t* out_mesh,
    pgo_error_t* error) {
    return call_c_api(error, [&]() -> pgo_status_t {
        if (path == nullptr || out_mesh == nullptr) {
            throw std::invalid_argument{"path and out_mesh must be non-null"};
        }

        const pgo::geometry::RestMesh<double, 3> mesh = [&]() {
            try {
                return pgo::io::read_obj_rest_mesh_3d(
                    std::filesystem::path{path});
            } catch (const std::exception& e) {
                throw IoError{e.what()};
            }
        }();

        const auto nv = mesh.num_vertices();
        const auto nf = mesh.num_faces();
        const std::size_t pos_count = nv * 3;
        const std::size_t tri_count = nf * 3;

        double* positions =
            static_cast<double*>(std::malloc(pos_count * sizeof(double)));
        std::uint64_t* triangles =
            static_cast<std::uint64_t*>(std::malloc(tri_count * sizeof(std::uint64_t)));
        if (positions == nullptr || triangles == nullptr) {
            std::free(positions);
            std::free(triangles);
            throw std::bad_alloc{};
        }

        for (std::size_t i = 0; i < pos_count; ++i) {
            positions[i] = mesh.rest_positions()[i];
        }
        for (std::size_t i = 0; i < tri_count; ++i) {
            triangles[i] = static_cast<std::uint64_t>(mesh.face_indices()[i]);
        }

        out_mesh->positions_xyz = positions;
        out_mesh->vertex_count = static_cast<std::uint64_t>(nv);
        out_mesh->triangles = triangles;
        out_mesh->triangle_count = static_cast<std::uint64_t>(nf);
        return PGO_STATUS_OK;
    });
}

void pgo_obj_mesh_free(pgo_obj_mesh_t* mesh) {
    if (mesh != nullptr) {
        std::free(mesh->positions_xyz);
        std::free(mesh->triangles);
        mesh->positions_xyz = nullptr;
        mesh->triangles = nullptr;
    }
}

} // extern "C"
