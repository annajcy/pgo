#pragma once

#include "pgo/core/base/assert.hpp"
#include "pgo/core/geometry/topology.hpp"
#include "pgo/core/math/types.hpp"
#include "pgo/core/storage/array_view.hpp"
#include "pgo/core/storage/host_buffer.hpp"

#include <cstddef>
#include <utility>

namespace pgo::geometry {

template <typename T, int Dim>
class RestMesh {
public:
    static_assert(Dim > 0);

    RestMesh(pgo::storage::HostBuffer<T> rest_positions, pgo::storage::HostBuffer<VertexIndex> edge_indices = {},
             pgo::storage::HostBuffer<VertexIndex> face_indices = {})
        : m_rest_positions{std::move(rest_positions)}, m_edge_indices{std::move(edge_indices)},
          m_face_indices{std::move(face_indices)} {
        pgo::base::require(m_rest_positions.size() % static_cast<std::size_t>(Dim) == 0,
                           "rest position buffer length must be a multiple of Dim");
        pgo::base::require(m_edge_indices.size() % kEdgeArity == 0,
                           "edge index buffer length must be a multiple of edge arity");
        pgo::base::require(m_face_indices.size() % kFaceArity == 0,
                           "face index buffer length must be a multiple of face arity");
    }

    [[nodiscard]] std::size_t num_vertices() const {
        return m_rest_positions.size() / static_cast<std::size_t>(Dim);
    }

    [[nodiscard]] std::size_t num_edges() const {
        return m_edge_indices.size() / kEdgeArity;
    }

    [[nodiscard]] std::size_t num_faces() const {
        return m_face_indices.size() / kFaceArity;
    }

    [[nodiscard]] pgo::storage::ConstArrayView<T> rest_positions() const {
        return m_rest_positions;
    }

    [[nodiscard]] pgo::storage::ConstArrayView<VertexIndex> edge_indices() const {
        return m_edge_indices;
    }

    [[nodiscard]] pgo::storage::ConstArrayView<VertexIndex> face_indices() const {
        return m_face_indices;
    }

    [[nodiscard]] pgo::math::Vec<T, Dim> rest_position(const std::size_t vertex) const {
        pgo::base::require(vertex < num_vertices(), "rest position vertex index is out of range");

        pgo::math::Vec<T, Dim> position{};
        const auto offset = vertex * static_cast<std::size_t>(Dim);
        for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
            position[pgo::math::dense_index(component)] = m_rest_positions[offset + component];
        }
        return position;
    }

    [[nodiscard]] VertexIndex edge_vertex(const std::size_t edge_id, const std::size_t local_vertex) const {
        pgo::base::require(edge_id < num_edges(), "edge index is out of range");
        pgo::base::require(local_vertex < kEdgeArity, "edge local vertex is out of range");
        return m_edge_indices[kEdgeArity * edge_id + local_vertex];
    }

    [[nodiscard]] VertexIndex face_vertex(const std::size_t face_id, const std::size_t local_vertex) const {
        pgo::base::require(face_id < num_faces(), "face index is out of range");
        pgo::base::require(local_vertex < kFaceArity, "face local vertex is out of range");
        return m_face_indices[kFaceArity * face_id + local_vertex];
    }

private:
    pgo::storage::HostBuffer<T> m_rest_positions;
    pgo::storage::HostBuffer<VertexIndex> m_edge_indices;
    pgo::storage::HostBuffer<VertexIndex> m_face_indices;
};

} // namespace pgo::geometry
