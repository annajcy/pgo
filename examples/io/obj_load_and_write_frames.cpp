#include "pgo/io/obj_reader.hpp"
#include "pgo/io/obj_writer.hpp"
#include "pgo/log/registry.hpp"
#if defined(PGO_ENABLE_SPDLOG)
#    include "pgo/log/spdlog_sink.hpp"
#else
#    include "pgo/log/stdio_sink.hpp"
#endif

#include <CLI/CLI.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <numbers>

namespace {

struct Options {
    std::filesystem::path input = "assets/model/bunny.obj";
    std::filesystem::path output = "output/example/io/obj_frames";
    std::size_t frames = 20;
    double amplitude = 0.02;
    std::string log_file;
};

[[nodiscard]] Options parse_options(const int argc, char** argv) {
    Options options;

    CLI::App app{"Load an OBJ rest mesh and write a deterministic OBJ frame sequence."};
    app.add_option("--input", options.input, "Input OBJ mesh path");
    app.add_option("--output", options.output, "Output directory for frame_*.obj files");
    app.add_option("--frames", options.frames, "Number of frames to write")->check(CLI::Range(2U, 1000000U));
    app.add_option("--amplitude", options.amplitude, "Displacement amplitude")->check(CLI::NonNegativeNumber);
    app.add_option("--log", options.log_file, "Log output file path (redirects from stderr/stdout)");

    app.parse(argc, argv);
    return options;
}

} // namespace

int main(int argc, char** argv) {
    const auto options = parse_options(argc, argv);

    if (!options.log_file.empty()) {
#if defined(PGO_ENABLE_SPDLOG)
        pgo::log::set_sink(pgo::log::make_spdlog_file_sink(options.log_file));
#else
        pgo::log::set_sink(std::make_shared<pgo::log::StdIOSink>(options.log_file));
#endif
    }
#if defined(PGO_ENABLE_SPDLOG)
    else {
        pgo::log::set_sink(pgo::log::make_default_spdlog_sink());
    }
#endif

    auto logger = pgo::log::get("examples.io.obj_frames");
    try {

        const auto mesh = pgo::io::read_obj_rest_mesh_3d(options.input);
        pgo::io::ObjWriter3d writer{options.output};

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

        logger.info(std::format("input: {}\n", options.input.string()));
        logger.info(std::format("output: {}\n", options.output.string()));
        logger.info(std::format("vertices: {}\n", mesh.num_vertices()));
        logger.info(std::format("edges: {}\n", mesh.num_edges()));
        logger.info(std::format("faces: {}\n", mesh.num_faces()));
        logger.info(std::format("frames: {}\n", options.frames));
        logger.info(std::format("last_frame: {}\n", last_frame.string()));
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        logger.error(std::format("error: {}\n", error.what()));
        return EXIT_FAILURE;
    }
}
