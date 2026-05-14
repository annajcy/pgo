#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "pgo/base/assert.hpp"
#include "pgo/geometry/rest_mesh.hpp"
#include "pgo/geometry/topology.hpp"
#include "pgo/io/obj_reader.hpp"
#include "pgo/storage/host_buffer.hpp"

#include <filesystem>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace pgo::io {

pgo::geometry::RestMesh<double, 3> read_obj_rest_mesh_3d(const std::filesystem::path& path) {
    tinyobj::ObjReader reader;

    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    config.vertex_color = false;

    const auto parse_ok = reader.ParseFromFile(path.string(), config);
    pgo::base::require(parse_ok, std::string{"Failed to parse OBJ: "} + reader.Error());

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    const auto num_vertices = attrib.vertices.size() / 3;
    pgo::base::require(num_vertices > 0, "OBJ file contains no vertices");
    pgo::base::require(num_vertices <= static_cast<std::size_t>(std::numeric_limits<pgo::geometry::VertexIndex>::max()),
                       "OBJ vertex count exceeds supported index range");

    pgo::storage::HostBuffer<double> rest_positions;
    rest_positions.reserve(num_vertices * 3);
    for (std::size_t i = 0; i < attrib.vertices.size(); ++i) {
        rest_positions.push_back(static_cast<double>(attrib.vertices[i]));
    }

    std::set<std::pair<pgo::geometry::VertexIndex, pgo::geometry::VertexIndex>> unique_edges;
    pgo::storage::HostBuffer<pgo::geometry::VertexIndex> face_indices;

    for (const auto& shape : shapes) {
        const auto& mesh = shape.mesh;
        for (std::size_t f = 0; f < mesh.num_face_vertices.size(); ++f) {
            const auto fv = mesh.num_face_vertices[f];
            pgo::base::require(fv == pgo::geometry::kFaceArity,
                               "OBJ face must be triangular after triangulation");

            std::array<pgo::geometry::VertexIndex, pgo::geometry::kFaceArity> face{};
            for (std::size_t v = 0; v < pgo::geometry::kFaceArity; ++v) {
                const auto idx = mesh.indices[f * pgo::geometry::kFaceArity + v];
                pgo::base::require(idx.vertex_index >= 0, "OBJ vertex index must be non-negative");
                pgo::base::require(static_cast<std::size_t>(idx.vertex_index) < num_vertices,
                                   "OBJ vertex index is out of range");
                face[v] = static_cast<pgo::geometry::VertexIndex>(idx.vertex_index);
            }

            pgo::base::require(face[0] != face[1] && face[1] != face[2] && face[2] != face[0],
                               "OBJ face must reference three distinct vertices");

            face_indices.insert(face_indices.end(), face.begin(), face.end());
            unique_edges.insert(std::minmax(face[0], face[1]));
            unique_edges.insert(std::minmax(face[1], face[2]));
            unique_edges.insert(std::minmax(face[2], face[0]));
        }
    }

    pgo::storage::HostBuffer<pgo::geometry::VertexIndex> edge_indices;
    edge_indices.reserve(unique_edges.size() * pgo::geometry::kEdgeArity);
    for (const auto& [a, b] : unique_edges) {
        edge_indices.push_back(a);
        edge_indices.push_back(b);
    }

    return pgo::geometry::RestMesh<double, 3>{std::move(rest_positions), std::move(edge_indices),
                                               std::move(face_indices)};
}

} // namespace pgo::io
