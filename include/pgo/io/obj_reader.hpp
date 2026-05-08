#pragma once

#include "pgo/base/assert.hpp"
#include "pgo/geometry/rest_mesh.hpp"
#include "pgo/math/backend.hpp"
#include "pgo/storage/host_buffer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pgo::io {

namespace detail {

[[nodiscard]] inline pgo::geometry::VertexIndex obj_vertex_index_to_zero_based(const std::string_view token,
                                                                               const std::size_t num_vertices) {
    const auto slash = token.find('/');
    const auto vertex_token = token.substr(0, slash);
    pgo::base::require(!vertex_token.empty(), "OBJ face/line token is missing a vertex index");

    std::size_t parsed_chars = 0;
    const auto one_based = std::stoll(std::string{vertex_token}, &parsed_chars);
    pgo::base::require(parsed_chars == vertex_token.size(), "OBJ vertex index token is invalid");
    pgo::base::require(one_based > 0, "OBJ vertex indices must be positive and one-based");

    const auto zero_based = static_cast<unsigned long long>(one_based - 1);
    pgo::base::require(zero_based < num_vertices, "OBJ vertex index is out of range");
    pgo::base::require(zero_based <= std::numeric_limits<pgo::geometry::VertexIndex>::max(),
                       "OBJ vertex index exceeds supported index range");
    return static_cast<pgo::geometry::VertexIndex>(zero_based);
}

inline void insert_undirected_edge(std::set<std::pair<pgo::geometry::VertexIndex, pgo::geometry::VertexIndex>>& edges,
                                   const pgo::geometry::VertexIndex a, const pgo::geometry::VertexIndex b) {
    pgo::base::require(a != b, "OBJ edge must reference two distinct vertices");
    edges.insert(std::minmax(a, b));
}

} // namespace detail

template <pgo::math::ScalarLike T, int Dim>
[[nodiscard]] pgo::geometry::RestMesh<T, Dim> read_obj_rest_mesh(const std::filesystem::path& path) {
    static_assert(Dim > 0);

    std::ifstream input{path};
    pgo::base::require(input.is_open(), "failed to open OBJ file");

    pgo::storage::HostBuffer<T> rest_positions;
    pgo::storage::HostBuffer<pgo::geometry::VertexIndex> face_indices;
    std::set<std::pair<pgo::geometry::VertexIndex, pgo::geometry::VertexIndex>> unique_edges;

    std::string line;
    while (std::getline(input, line)) {
        std::istringstream stream{line};
        std::string record;
        stream >> record;

        if (record.empty() || record.starts_with('#')) {
            continue;
        }

        if (record == "v") {
            std::array<T, 3> obj_position{};
            const auto parsed_position =
                static_cast<bool>(stream >> obj_position[0] >> obj_position[1] >> obj_position[2]);
            pgo::base::require(parsed_position, "OBJ vertex record must contain x y z coordinates");

            for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
                const auto value = component < obj_position.size() ? obj_position[component] : T{};
                rest_positions.push_back(value);
            }
        } else if (record == "l") {
            std::vector<pgo::geometry::VertexIndex> vertices;
            std::string token;
            while (stream >> token) {
                vertices.push_back(detail::obj_vertex_index_to_zero_based(token, rest_positions.size() /
                                                                                     static_cast<std::size_t>(Dim)));
            }

            pgo::base::require(vertices.size() >= pgo::geometry::kEdgeArity,
                               "OBJ line record must reference at least two vertices");
            for (std::size_t i = 1; i < vertices.size(); ++i) {
                detail::insert_undirected_edge(unique_edges, vertices[i - 1], vertices[i]);
            }
        } else if (record == "f") {
            std::array<pgo::geometry::VertexIndex, pgo::geometry::kFaceArity> face{};
            std::string token;
            for (std::size_t i = 0; i < pgo::geometry::kFaceArity; ++i) {
                const auto parsed_token = static_cast<bool>(stream >> token);
                pgo::base::require(parsed_token, "OBJ face record must be triangular");
                face[i] = detail::obj_vertex_index_to_zero_based(token,
                                                                 rest_positions.size() / static_cast<std::size_t>(Dim));
            }

            std::string extra_token;
            pgo::base::require(!(stream >> extra_token), "OBJ face record must be triangular");

            face_indices.insert(face_indices.end(), face.begin(), face.end());
            detail::insert_undirected_edge(unique_edges, face[0], face[1]);
            detail::insert_undirected_edge(unique_edges, face[1], face[2]);
            detail::insert_undirected_edge(unique_edges, face[2], face[0]);
        }
    }

    pgo::storage::HostBuffer<pgo::geometry::VertexIndex> edge_indices;
    edge_indices.reserve(unique_edges.size() * pgo::geometry::kEdgeArity);
    for (const auto& [a, b] : unique_edges) {
        edge_indices.push_back(a);
        edge_indices.push_back(b);
    }

    return pgo::geometry::RestMesh<T, Dim>{std::move(rest_positions), std::move(edge_indices), std::move(face_indices)};
}

} // namespace pgo::io
