#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>
#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[nodiscard]] bool starts_with(const std::string& value, const std::string_view prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

struct Options {
    std::filesystem::path frames_dir;
    std::filesystem::path output;
    double fps = 24.0;
};

struct ObjFrame {
    std::vector<float> positions;
    std::vector<int> face_indices;
    std::vector<int> face_counts;
};

void configure_cli(CLI::App& app, Options& options) {
    app.add_option("--frames-dir", options.frames_dir, "Directory containing frame_*.obj files")->required();
    app.add_option("--output", options.output, "Output Alembic .abc path")->required();
    app.add_option("--fps", options.fps, "Uniform frame rate for Alembic time sampling")->check(CLI::PositiveNumber);
}

[[nodiscard]] std::vector<std::filesystem::path> frame_paths(const std::filesystem::path& frames_dir) {
    if (!std::filesystem::is_directory(frames_dir)) {
        throw std::runtime_error{"frames directory does not exist: " + frames_dir.string()};
    }

    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator{frames_dir}) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const auto filename = entry.path().filename().string();
        if (starts_with(filename, "frame_") && entry.path().extension() == ".obj") {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        throw std::runtime_error{"no frame_*.obj files found in: " + frames_dir.string()};
    }
    return paths;
}

[[nodiscard]] int parse_obj_vertex_index(const std::string_view token, const std::size_t vertex_count,
                                         const std::filesystem::path& source, const std::size_t line_number) {
    const auto slash = token.find('/');
    const auto vertex_token = token.substr(0, slash);
    if (vertex_token.empty()) {
        throw std::runtime_error{source.string() + ":" + std::to_string(line_number) +
                                 ": face token is missing a vertex index"};
    }

    std::size_t parsed_chars = 0;
    long long one_based = 0;
    try {
        one_based = std::stoll(std::string{vertex_token}, &parsed_chars);
    } catch (const std::exception&) {
        throw std::runtime_error{source.string() + ":" + std::to_string(line_number) +
                                 ": invalid OBJ vertex index: " + std::string{vertex_token}};
    }

    if (parsed_chars != vertex_token.size()) {
        throw std::runtime_error{source.string() + ":" + std::to_string(line_number) +
                                 ": invalid OBJ vertex index: " + std::string{vertex_token}};
    }
    if (one_based <= 0) {
        throw std::runtime_error{source.string() + ":" + std::to_string(line_number) +
                                 ": OBJ vertex indices must be positive and one-based"};
    }
    if (static_cast<unsigned long long>(one_based) > vertex_count) {
        throw std::runtime_error{source.string() + ":" + std::to_string(line_number) +
                                 ": OBJ vertex index is out of range"};
    }
    if (one_based - 1 > std::numeric_limits<int>::max()) {
        throw std::runtime_error{source.string() + ":" + std::to_string(line_number) +
                                 ": OBJ vertex index exceeds Alembic int32 range"};
    }

    return static_cast<int>(one_based - 1);
}

[[nodiscard]] ObjFrame read_obj_frame(const std::filesystem::path& path) {
    std::ifstream input{path};
    if (!input.is_open()) {
        throw std::runtime_error{"failed to read OBJ frame: " + path.string()};
    }

    ObjFrame frame;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream stream{line};
        std::string record;
        stream >> record;
        if (record.empty() || starts_with(record, "#")) {
            continue;
        }

        if (record == "v") {
            float x = 0.0F;
            float y = 0.0F;
            float z = 0.0F;
            if (!(stream >> x >> y >> z)) {
                throw std::runtime_error{path.string() + ":" + std::to_string(line_number) +
                                         ": vertex record must contain x y z"};
            }
            frame.positions.push_back(x);
            frame.positions.push_back(y);
            frame.positions.push_back(z);
        } else if (record == "f") {
            std::vector<std::string> tokens;
            std::string token;
            while (stream >> token) {
                tokens.push_back(token);
            }

            if (tokens.size() != 3) {
                throw std::runtime_error{path.string() + ":" + std::to_string(line_number) +
                                         ": only triangular faces are supported"};
            }

            for (const auto& face_token : tokens) {
                frame.face_indices.push_back(
                    parse_obj_vertex_index(face_token, frame.positions.size() / 3, path, line_number));
            }
            frame.face_counts.push_back(3);
        }
    }

    return frame;
}

[[nodiscard]] std::vector<ObjFrame> load_frames(const std::vector<std::filesystem::path>& paths) {
    std::vector<ObjFrame> frames;
    frames.reserve(paths.size());
    for (const auto& path : paths) {
        frames.push_back(read_obj_frame(path));
    }

    const auto& first = frames.front();
    if (first.face_counts.empty()) {
        throw std::runtime_error{"OBJ frames do not contain faces; animated PolyMesh export requires triangular faces"};
    }

    const auto vertex_count = first.positions.size() / 3;
    for (std::size_t frame_id = 1; frame_id < frames.size(); ++frame_id) {
        const auto& frame = frames[frame_id];
        if (frame.positions.size() / 3 != vertex_count) {
            throw std::runtime_error{paths[frame_id].string() + ": vertex count changed from " +
                                     std::to_string(vertex_count) + " to " +
                                     std::to_string(frame.positions.size() / 3)};
        }
        if (frame.face_indices != first.face_indices || frame.face_counts != first.face_counts) {
            throw std::runtime_error{paths[frame_id].string() + ": face topology does not match the first frame"};
        }
    }

    return frames;
}

void write_alembic(const std::filesystem::path& output, const std::vector<ObjFrame>& frames, const double fps) {
    namespace Abc = Alembic::Abc;
    namespace AbcGeom = Alembic::AbcGeom;

    if (output.has_parent_path()) {
        std::filesystem::create_directories(output.parent_path());
    }

    Abc::OArchive archive{Alembic::AbcCoreOgawa::WriteArchive(), output.string()};
    const AbcGeom::TimeSampling time_sampling{1.0 / fps, 0.0};
    const auto time_sampling_index = archive.addTimeSampling(time_sampling);

    AbcGeom::OPolyMesh mesh_object{archive.getTop(), "mesh", time_sampling_index};
    auto mesh_schema = mesh_object.getSchema();
    mesh_schema.setTimeSampling(time_sampling_index);

    Abc::Int32ArraySample face_indices{frames.front().face_indices.data(), frames.front().face_indices.size()};
    Abc::Int32ArraySample face_counts{frames.front().face_counts.data(), frames.front().face_counts.size()};
    for (const auto& frame : frames) {
        std::vector<Imath::V3f> points;
        points.reserve(frame.positions.size() / 3);
        for (std::size_t index = 0; index < frame.positions.size(); index += 3) {
            points.emplace_back(frame.positions[index], frame.positions[index + 1], frame.positions[index + 2]);
        }

        AbcGeom::OPolyMeshSchema::Sample sample{
            AbcGeom::P3fArraySample{points.data(), points.size()},
            face_indices, face_counts};
        mesh_schema.set(sample);
    }
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    CLI::App app{"Convert frame_*.obj vertex animation into an Alembic PolyMesh archive."};
    configure_cli(app, options);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& error) {
        return app.exit(error);
    }

    try {
        const auto paths = frame_paths(options.frames_dir);
        const auto frames = load_frames(paths);
        write_alembic(options.output, frames, options.fps);

        std::cout << "frames: " << frames.size() << '\n';
        std::cout << "vertices: " << frames.front().positions.size() / 3 << '\n';
        std::cout << "faces: " << frames.front().face_counts.size() << '\n';
        std::cout << "fps: " << options.fps << '\n';
        std::cout << "output: " << options.output << '\n';
        std::cout << "first_frame: " << paths.front() << '\n';
        std::cout << "last_frame: " << paths.back() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
