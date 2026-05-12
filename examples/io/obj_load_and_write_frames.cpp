#include "pgo/base/assert.hpp"
#include "pgo/io/obj_frame_writer.hpp"
#include "pgo/io/obj_reader.hpp"

#include <CLI/CLI.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <numbers>

namespace {

struct Options {
    std::filesystem::path input = "assets/model/bunny.obj";
    std::filesystem::path output = "output/example/io/obj_frames";
    std::size_t frames = 20;
    double amplitude = 0.02;
};

[[nodiscard]] Options parse_options(const int argc, char** argv) {
    Options options;

    CLI::App app{"Load an OBJ rest mesh and write a deterministic OBJ frame sequence."};
    app.add_option("--input", options.input, "Input OBJ mesh path");
    app.add_option("--output", options.output, "Output directory for frame_*.obj files");
    app.add_option("--frames", options.frames, "Number of frames to write")->check(CLI::Range(2U, 1000000U));
    app.add_option("--amplitude", options.amplitude, "Displacement amplitude")->check(CLI::NonNegativeNumber);

    app.parse(argc, argv);
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);

        const auto mesh = pgo::io::read_obj_rest_mesh_3d(options.input);
        pgo::io::ObjFrameWriter<double, 3> writer{options.output};

        pgo::math::DVec<double> displacement{pgo::math::dense_index(mesh.num_vertices() * static_cast<std::size_t>(3))};
        std::filesystem::path last_frame;
        for (std::size_t frame = 0; frame < options.frames; ++frame) {
            displacement.setZero();

            const auto phase =
                2.0 * std::numbers::pi_v<double> * static_cast<double>(frame) / static_cast<double>(options.frames - 1);
            const auto y_offset = options.amplitude * std::sin(phase);
            for (std::size_t vertex = 0; vertex < mesh.num_vertices(); ++vertex) {
                displacement[pgo::math::dense_index(vertex * static_cast<std::size_t>(3) + 1)] = y_offset;
            }

            last_frame = writer.write_frame(mesh, displacement);
        }

        std::cout << "input: " << options.input << '\n';
        std::cout << "output: " << options.output << '\n';
        std::cout << "vertices: " << mesh.num_vertices() << '\n';
        std::cout << "edges: " << mesh.num_edges() << '\n';
        std::cout << "faces: " << mesh.num_faces() << '\n';
        std::cout << "frames: " << options.frames << '\n';
        std::cout << "last_frame: " << last_frame << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
