#include "pgo/io/obj_writer.hpp"
#include "pgo/base/assert.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

namespace pgo::io {

ObjWriter3d::ObjWriter3d(std::filesystem::path output_dir) : m_output_dir{std::move(output_dir)} {
    std::filesystem::create_directories(m_output_dir);
}

std::filesystem::path ObjWriter3d::write_frame(const pgo::geometry::RestMesh<double, 3>& mesh,
                                               const pgo::math::DVec<double>& displacement) {
    const auto expected_dofs = mesh.num_vertices() * 3;
    pgo::base::require(static_cast<std::size_t>(displacement.size()) == expected_dofs,
                       "displacement vector size does not match mesh");

    const auto path = m_output_dir / frame_name(m_next_frame++);
    std::ofstream output{path};
    pgo::base::require(output.is_open(), "failed to open OBJ frame for writing");
    output << std::setprecision(17);

    for (std::size_t vertex = 0; vertex < mesh.num_vertices(); ++vertex) {
        output << "v";
        for (std::size_t component = 0; component < 3; ++component) {
            const auto offset = vertex * 3 + component;
            const auto value = mesh.rest_positions()[offset] + displacement[pgo::math::dense_index(offset)];
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

std::string ObjWriter3d::frame_name(const std::size_t frame) {
    std::ostringstream name;
    name << "frame_" << std::setw(4) << std::setfill('0') << frame << ".obj";
    return name.str();
}

} // namespace pgo::io
