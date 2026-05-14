#include "pgo/dof/dirichlet_boundary.hpp"
#include "pgo/dof/dof_layout.hpp"
#include "pgo/dof/reduced_dof_map.hpp"
#include "pgo/energy/assembled_energy.hpp"
#include "pgo/energy/constant_force_energy.hpp"
#include "pgo/energy/energy_sum.hpp"
#include "pgo/energy/mass_spring_local_energy_provider.hpp"
#include "pgo/geometry/rest_mesh.hpp"
#include "pgo/integrator/backward_euler.hpp"
#include "pgo/integrator/dynamic_state.hpp"
#if defined(PGO_ENABLE_ALEMBIC)
#    include "pgo/io/abc_writer.hpp"
#endif
#include "pgo/io/obj_writer.hpp"
#include "pgo/log/registry.hpp"
#if defined(PGO_ENABLE_SPDLOG)
#    include "pgo/log/spdlog_sink.hpp"
#else
#    include "pgo/log/stdio_sink.hpp"
#endif
#include "pgo/solver/status_name.hpp"

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::filesystem::path output = "output/example/mass_spring/cloth";
    std::filesystem::path abc_output = "output/example/mass_spring/cloth/cloth.abc";
    std::size_t frames = 300;
    std::size_t resolution = 20;
    double stiffness = 1000.0;
    double gravity = 9.8;
    double dt = 0.016;
    int ramp_frames = 0;
    std::string log_file;
};

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options opts;
    CLI::App app{"Mass-spring cloth simulation with Backward Euler"};

    app.add_option("--output", opts.output, "Output directory for frame_*.obj files");
    app.add_option("--export-abc", opts.abc_output, "Export to Alembic .abc file");
    app.add_option("--frames", opts.frames, "Number of frames to output")->check(CLI::Range(2U, 1000000U));
    app.add_option("--resolution", opts.resolution, "Cloth grid resolution (NxN)")->check(CLI::Range(2U, 1000U));
    app.add_option("--stiffness", opts.stiffness, "Uniform spring stiffness")->check(CLI::PositiveNumber);
    app.add_option("--gravity", opts.gravity, "Gravity magnitude (negative Y direction)")->check(CLI::NonNegativeNumber);
    app.add_option("--dt", opts.dt, "Timestep size")->check(CLI::PositiveNumber);
    app.add_option("--ramp-frames", opts.ramp_frames, "Gravity ramp duration (0 = constant force)");
    app.add_option("--log", opts.log_file, "Log output file path (redirects from stderr/stdout)");

    app.parse(argc, argv);
    return opts;
}

[[nodiscard]] pgo::geometry::RestMesh<double, 3> make_cloth_grid(const std::size_t resolution) {
    const auto n = resolution;
    const double cell_size = 1.0 / static_cast<double>(n - 1);

    // positions: grid in XY plane, row * n + col
    pgo::storage::HostBuffer<double> positions;
    positions.reserve(n * n * 3);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            positions.push_back(static_cast<double>(col) * cell_size);  // X
            positions.push_back(static_cast<double>(row) * cell_size);  // Y
            positions.push_back(0.0);                                   // Z = 0
        }
    }

    auto vertex_index = [n](std::size_t row, std::size_t col) -> pgo::geometry::VertexIndex {
        return static_cast<pgo::geometry::VertexIndex>(row * n + col);
    };

    // triangular faces + extract unique undirected edges
    pgo::storage::HostBuffer<pgo::geometry::VertexIndex> face_indices;
    std::set<std::pair<pgo::geometry::VertexIndex, pgo::geometry::VertexIndex>> unique_edges;

    auto add_face = [&](pgo::geometry::VertexIndex a, pgo::geometry::VertexIndex b,
                         pgo::geometry::VertexIndex c) {
        face_indices.push_back(a);
        face_indices.push_back(b);
        face_indices.push_back(c);
        unique_edges.insert(std::minmax(a, b));
        unique_edges.insert(std::minmax(b, c));
        unique_edges.insert(std::minmax(c, a));
    };

    for (std::size_t row = 0; row < n - 1; ++row) {
        for (std::size_t col = 0; col < n - 1; ++col) {
            const auto v00 = vertex_index(row, col);
            const auto v10 = vertex_index(row + 1, col);
            const auto v11 = vertex_index(row + 1, col + 1);
            const auto v01 = vertex_index(row, col + 1);
            add_face(v00, v10, v11);
            add_face(v00, v11, v01);
        }
    }

    pgo::storage::HostBuffer<pgo::geometry::VertexIndex> edge_indices;
    edge_indices.reserve(unique_edges.size() * 2);
    for (const auto& [a, b] : unique_edges) {
        edge_indices.push_back(a);
        edge_indices.push_back(b);
    }

    return pgo::geometry::RestMesh<double, 3>{std::move(positions), std::move(edge_indices), std::move(face_indices)};
}

[[nodiscard]] pgo::math::DVec<double> make_gravity_force(const pgo::math::DVec<double>& lumped_mass,
                                                          const double g_magnitude, const double scale) {
    pgo::math::DVec<double> force = pgo::math::DVec<double>::Zero(lumped_mass.size());
    for (pgo::math::DenseIndex dof = 0; dof < lumped_mass.size(); ++dof) {
        if (static_cast<std::size_t>(dof % 3) == 1) {
            force[dof] = -scale * g_magnitude * lumped_mass[dof];
        }
    }
    return force;
}

} // namespace

int main(int argc, char** argv) {
    const auto opts = parse_options(argc, argv);

    if (!opts.log_file.empty()) {
#if defined(PGO_ENABLE_SPDLOG)
        pgo::log::set_sink(pgo::log::make_spdlog_file_sink(opts.log_file));
#else
        pgo::log::set_sink(std::make_shared<pgo::log::StdIOSink>(opts.log_file));
#endif
    }
#if defined(PGO_ENABLE_SPDLOG)
    else {
        pgo::log::set_sink(pgo::log::make_default_spdlog_sink());
    }
#endif

    auto logger = pgo::log::get("examples.mass_spring.cloth");
    try {

        const auto mesh = make_cloth_grid(opts.resolution);
        const auto num_dofs = mesh.num_vertices() * 3;
        const pgo::dof::DofLayout<3> layout{mesh.num_vertices()};

        // pin top row (max Y = row resolution-1)
        pgo::dof::DirichletBoundary<double> boundary;
        const auto n = opts.resolution;
        for (std::size_t col = 0; col < n; ++col) {
            pgo::dof::fix_vertex(boundary, layout, (n - 1) * n + col);
        }

        const pgo::dof::ReducedDofMap<double> dof_map{num_dofs, boundary};

        pgo::math::DVec<double> lumped_mass{pgo::math::dense_index(num_dofs)};
        lumped_mass.setOnes();

        const pgo::energy::MassSpringLocalEnergyProvider<double, 3> provider{mesh, opts.stiffness};
        const pgo::energy::AssembledEnergyView<double, decltype(provider)> potential{provider};

        const pgo::integrator::BackwardEuler<double> integrator;
        pgo::integrator::DynamicState<double> state;
        state.u.setZero(pgo::math::dense_index(num_dofs));
        state.v.setZero(pgo::math::dense_index(num_dofs));
        state.a.setZero(pgo::math::dense_index(num_dofs));

        pgo::solver::NewtonOptions<double> newton_opts;
        newton_opts.max_iterations = 100;
        newton_opts.gradient_tolerance = 1e-5;
        newton_opts.initial_regularization = 1e-4;

        std::filesystem::create_directories(opts.output);
        pgo::io::ObjWriter3d writer{opts.output};

        std::vector<int> face_counts(mesh.num_faces(), 3);
#if defined(PGO_ENABLE_ALEMBIC)
        std::optional<pgo::io::AbcWriter3d> abc_writer;
        if (!opts.abc_output.empty()) {
            abc_writer.emplace(opts.abc_output, 1.0 / opts.dt,
                               mesh.face_indices(),
                               std::span<const int>{face_counts.data(), face_counts.size()});
        }
#endif

        static_cast<void>(writer.write_frame(mesh, state.u));
#if defined(PGO_ENABLE_ALEMBIC)
        if (abc_writer) abc_writer->write_frame(mesh, state.u);
#endif

        const std::size_t num_steps = opts.frames - 1;
        for (std::size_t step = 0; step < num_steps; ++step) {
            const double ramp_scale = opts.ramp_frames <= 0
                                           ? 1.0
                                           : std::min(1.0, static_cast<double>(step) / static_cast<double>(opts.ramp_frames));
            const auto gravity_force = make_gravity_force(lumped_mass, opts.gravity, ramp_scale);
            const pgo::energy::ConstantForceEnergyView<double> gravity_energy{gravity_force};
            const pgo::energy::EnergySumView<double, decltype(potential), decltype(gravity_energy)> step_energy{
                potential, gravity_energy};

            const auto result = integrator.step(step_energy, lumped_mass, dof_map, state, opts.dt, newton_opts,
                                                /*commit_on_failure=*/true);

            if (result.status != pgo::solver::SolverStatus::converged) {
                logger.warn(std::format("Frame {} warning: status={}, iterations={}, value={}, grad_norm={}\n",
                                         step + 1, pgo::solver::status_name(result.status),
                                         result.solver_iterations, result.final_value, result.final_gradient_norm));
            }

            static_cast<void>(writer.write_frame(mesh, state.u));
#if defined(PGO_ENABLE_ALEMBIC)
            if (abc_writer) abc_writer->write_frame(mesh, state.u);
#endif
        }

        logger.info(std::format("resolution: {}x{}  ({} vertices, {} edges, {} faces)\n",
                                 opts.resolution, opts.resolution,
                                 mesh.num_vertices(), mesh.num_edges(), mesh.num_faces()));
        logger.info(std::format("output: {}\n", opts.output.string()));
        if (!opts.abc_output.empty()) {
            logger.info(std::format("abc: {}\n", opts.abc_output.string()));
        }
        logger.info(std::format("free dofs: {} / {}\n", dof_map.free_dofs(), dof_map.full_dofs()));
        logger.info(std::format("frames: {}  dt: {}  stiffness: {}  gravity: {}\n",
                                 opts.frames, opts.dt, opts.stiffness, opts.gravity));
        return 0;
    } catch (const std::exception& e) {
        logger.error(std::format("error: {}\n", e.what()));
        return 1;
    }
}
