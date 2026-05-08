#pragma once

#include "pgo/base/assert.hpp"
#include "pgo/geometry/rest_mesh.hpp"
#include "pgo/math/types.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace pgo::io {

template <typename T, int Dim>
class ObjFrameWriter {
public:
    explicit ObjFrameWriter(std::filesystem::path output_dir) : m_output_dir{std::move(output_dir)} {
        static_assert(Dim > 0);
        std::filesystem::create_directories(m_output_dir);
    }

    [[nodiscard]] std::filesystem::path write_frame(const pgo::geometry::RestMesh<T, Dim>& mesh,
                                                    const pgo::math::DVec<T>& displacement) {
        const auto expected_dofs = mesh.num_vertices() * static_cast<std::size_t>(Dim);
        pgo::base::require(static_cast<std::size_t>(displacement.size()) == expected_dofs,
                           "displacement vector size does not match mesh");

        const auto path = m_output_dir / frame_name(m_next_frame++);
        std::ofstream output{path};
        pgo::base::require(output.is_open(), "failed to open OBJ frame for writing");
        output << std::setprecision(17);

        for (std::size_t vertex = 0; vertex < mesh.num_vertices(); ++vertex) {
            output << "v";
            for (std::size_t component = 0; component < 3; ++component) {
                T value{};
                if (component < static_cast<std::size_t>(Dim)) {
                    const auto offset = vertex * static_cast<std::size_t>(Dim) + component;
                    value = mesh.rest_positions()[offset] + displacement[pgo::math::dense_index(offset)];
                }
                output << ' ' << value;
            }
            output << '\n';
        }

        if (mesh.num_faces() > 0) {
            for (std::size_t face = 0; face < mesh.num_faces(); ++face) {
                output << "f";
                for (std::size_t local = 0; local < pgo::geometry::kFaceArity; ++local) {
                    output << ' ' << static_cast<std::size_t>(mesh.face_vertex(face, local)) + 1;
                }
                output << '\n';
            }
        } else {
            for (std::size_t edge = 0; edge < mesh.num_edges(); ++edge) {
                output << "l";
                for (std::size_t local = 0; local < pgo::geometry::kEdgeArity; ++local) {
                    output << ' ' << static_cast<std::size_t>(mesh.edge_vertex(edge, local)) + 1;
                }
                output << '\n';
            }
        }

        return path;
    }

private:
    [[nodiscard]] static std::string frame_name(const std::size_t frame) {
        std::ostringstream name;
        name << "frame_" << std::setw(4) << std::setfill('0') << frame << ".obj";
        return name.str();
    }

    std::filesystem::path m_output_dir;
    std::size_t m_next_frame = 0;
};

} // namespace pgo::io
