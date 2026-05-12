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
#include "pgo/io/abc_writer.hpp"
#include "pgo/io/obj_frame_writer.hpp"
#include "pgo/io/obj_reader.hpp"
#include "pgo/solver/status_name.hpp"

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <optional>

namespace {

struct Options {
    std::filesystem::path input = "assets/model/bunny.obj";
    std::filesystem::path output = "output/mass_spring/bunny";
    std::filesystem::path abc_output;
    std::size_t frames = 300;
    double stiffness = 200000.0;
    double gravity = 9.81;
    double dt = 0.001;
    int ramp_frames = 10;
    double pin_fraction = 0.05;
};

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options opts;
    CLI::App app{"Mass-spring bunny cloth simulation with Backward Euler"};

    app.add_option("--input", opts.input, "Input OBJ mesh file");
    app.add_option("--output", opts.output, "Output directory for frame_*.obj files");
    app.add_option("--export-abc", opts.abc_output, "Export to Alembic .abc file");
    app.add_option("--frames", opts.frames, "Number of frames to output")->check(CLI::Range(2U, 1000000U));
    app.add_option("--stiffness", opts.stiffness, "Uniform spring stiffness")->check(CLI::PositiveNumber);
    app.add_option("--gravity", opts.gravity, "Gravity magnitude (negative Y direction)")->check(CLI::NonNegativeNumber);
    app.add_option("--dt", opts.dt, "Timestep size")->check(CLI::PositiveNumber);
    app.add_option("--ramp-frames", opts.ramp_frames, "Gravity ramp duration (0 = constant force)");
    app.add_option("--pin-fraction", opts.pin_fraction, "Fraction of top-Y vertices to pin (0-1)")->check(CLI::Range(0.0, 1.0));

    app.parse(argc, argv);
    return opts;
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
    try {
        const auto opts = parse_options(argc, argv);

        const auto mesh = pgo::io::read_obj_rest_mesh_3d(opts.input);
        const auto num_dofs = mesh.num_vertices() * 3;
        const pgo::dof::DofLayout<3> layout{mesh.num_vertices()};

        // Find Y range and pin top vertices by height percentile
        const auto positions = mesh.rest_positions();
        double min_y = std::numeric_limits<double>::max();
        double max_y = std::numeric_limits<double>::lowest();
        for (std::size_t v = 0; v < mesh.num_vertices(); ++v) {
            const double y = positions[static_cast<std::size_t>(v * 3 + 1)];
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }

        const double pin_threshold = max_y - opts.pin_fraction * (max_y - min_y);
        pgo::dof::DirichletBoundary<double> boundary;
        std::size_t pinned_count = 0;
        for (std::size_t v = 0; v < mesh.num_vertices(); ++v) {
            const double y = positions[static_cast<std::size_t>(v * 3 + 1)];
            if (y >= pin_threshold) {
                pgo::dof::fix_vertex(boundary, layout, v);
                ++pinned_count;
            }
        }

        const pgo::dof::ReducedDofMap<double> dof_map{num_dofs, boundary};

        pgo::math::DVec<double> lumped_mass{pgo::math::dense_index(num_dofs)};
        lumped_mass.setConstant(0.1); // uniform mass for simplicity

        const pgo::energy::MassSpringLocalEnergyProvider<double, 3> provider{mesh, opts.stiffness};
        const pgo::energy::AssembledEnergyView<double, decltype(provider)> potential{provider};

        const pgo::integrator::BackwardEuler<double> integrator;
        pgo::integrator::DynamicState<double> state;
        state.u.setZero(pgo::math::dense_index(num_dofs));
        state.v.setZero(pgo::math::dense_index(num_dofs));
        state.a.setZero(pgo::math::dense_index(num_dofs));

        pgo::solver::NewtonOptions<double> newton_opts;
        newton_opts.max_iterations = 500;
        newton_opts.gradient_tolerance = 1e-4;
        newton_opts.initial_regularization = 1e-6;

        std::filesystem::create_directories(opts.output);
        pgo::io::ObjFrameWriter<double, 3> writer{opts.output};

        std::vector<int> face_counts(mesh.num_faces(), 3);
        std::optional<pgo::io::AbcWriter<double, 3>> abc_writer;
        if (!opts.abc_output.empty()) {
            abc_writer.emplace(opts.abc_output, 1.0 / opts.dt,
                               mesh.face_indices(),
                               std::span<const int>{face_counts.data(), face_counts.size()});
        }

        static_cast<void>(writer.write_frame(mesh, state.u));
        if (abc_writer) abc_writer->write_frame(mesh, state.u);

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
                std::cerr << std::format("Frame {} warning: status={}, iterations={}, value={}, grad_norm={}\n",
                                         step + 1, pgo::solver::status_name(result.status),
                                         result.solver_iterations, result.final_value, result.final_gradient_norm);
            }

            static_cast<void>(writer.write_frame(mesh, state.u));
            if (abc_writer) abc_writer->write_frame(mesh, state.u);
        }

        std::cout << std::format("input: {}\n", opts.input.string());
        std::cout << std::format("vertices: {}  edges: {}  faces: {}\n",
                                 mesh.num_vertices(), mesh.num_edges(), mesh.num_faces());
        std::cout << std::format("Y range: [{:.4f}, {:.4f}]  pin threshold: {:.4f}  pinned: {}\n",
                                 min_y, max_y, pin_threshold, pinned_count);
        std::cout << std::format("output: {}\n", opts.output.string());
        if (!opts.abc_output.empty()) {
            std::cout << std::format("abc: {}\n", opts.abc_output.string());
        }
        std::cout << std::format("free dofs: {} / {}\n", dof_map.free_dofs(), dof_map.full_dofs());
        std::cout << std::format("frames: {}  dt: {}  stiffness: {}  gravity: {}\n",
                                 opts.frames, opts.dt, opts.stiffness, opts.gravity);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
