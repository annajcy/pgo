#pragma once

#include "pgo/core/base/assert.hpp"
#include "pgo/core/dof/dof_layout.hpp"
#include "pgo/core/math/types.hpp"
#include "pgo/core/storage/array_view.hpp"


#include <cstddef>
#include <optional>
#include <unordered_map>

namespace pgo::dof {

template <typename Boundary, typename T>
concept DirichletBoundaryLike = requires(const Boundary& boundary, const std::size_t dof) {
    { boundary.fixed_value(dof) } -> std::same_as<std::optional<T>>;
};

template <typename T>
class DirichletBoundary {
    std::unordered_map<std::size_t, T> m_fixed_values;

public:
    void prescribe_dof(const std::size_t dof, const T value) {
        m_fixed_values[dof] = value;
    }

    void fix_dof(const std::size_t dof) {
        prescribe_dof(dof, T{0});
    }

    [[nodiscard]] bool is_fixed(const std::size_t dof) const {
        return m_fixed_values.contains(dof);
    }

    [[nodiscard]] std::optional<T> fixed_value(const std::size_t dof) const {
        const auto fixed_value = m_fixed_values.find(dof);
        if (fixed_value == m_fixed_values.end()) {
            return std::nullopt;
        }
        return fixed_value->second;
    }

    [[nodiscard]] T value(const std::size_t dof) const {
        const auto fixed = fixed_value(dof);
        pgo::base::require(fixed.has_value(), "DOF is not fixed");
        return *fixed;
    }
};

// --- Free functions for vertex-level boundary helpers ---

template <int Dim, typename Boundary, typename T>
void prescribe_component(Boundary& boundary, const DofLayout<Dim>& layout, const std::size_t vertex,
                         const std::size_t component, const T value) {
    boundary.prescribe_dof(layout.index(vertex, component), value);
}

template <int Dim, typename Boundary>
void fix_component(Boundary& boundary, const DofLayout<Dim>& layout, const std::size_t vertex,
                   const std::size_t component) {
    boundary.fix_dof(layout.index(vertex, component));
}

template <int Dim, typename Boundary, typename T>
void prescribe_vertex(Boundary& boundary, const DofLayout<Dim>& layout, const std::size_t vertex,
                      const pgo::math::Vec<T, Dim>& value) {
    for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
        prescribe_component<Dim>(boundary, layout, vertex, component, value[pgo::math::dense_index(component)]);
    }
}

template <int Dim, typename Boundary>
void fix_vertex(Boundary& boundary, const DofLayout<Dim>& layout, const std::size_t vertex) {
    for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
        fix_component<Dim>(boundary, layout, vertex, component);
    }
}

template <int Dim, typename Boundary, typename T>
void prescribe_vertices(Boundary& boundary, const DofLayout<Dim>& layout,
                        const pgo::storage::ConstArrayView<std::size_t> vertex_indices,
                        const pgo::math::Vec<T, Dim>& value) {
    for (const std::size_t vertex : vertex_indices) {
        prescribe_vertex<Dim>(boundary, layout, vertex, value);
    }
}

template <int Dim, typename Boundary>
void fix_vertices(Boundary& boundary, const DofLayout<Dim>& layout,
                  const pgo::storage::ConstArrayView<std::size_t> vertex_indices) {
    for (const std::size_t vertex : vertex_indices) {
        fix_vertex<Dim>(boundary, layout, vertex);
    }
}

template <int Dim, typename Boundary, typename T>
void prescribe_vertices_by_list(Boundary& boundary, const DofLayout<Dim>& layout,
                                const pgo::storage::ConstArrayView<std::size_t> vertex_indices,
                                const pgo::storage::ConstArrayView<T> values) {
    const auto dimension = static_cast<std::size_t>(Dim);
    pgo::base::require(values.size() == vertex_indices.size() * dimension,
                       "prescribed vertex values length does not match vertex list");

    for (std::size_t local_vertex = 0; local_vertex < vertex_indices.size(); ++local_vertex) {
        for (std::size_t component = 0; component < dimension; ++component) {
            prescribe_component<Dim>(boundary, layout, vertex_indices[local_vertex], component,
                                     values[local_vertex * dimension + component]);
        }
    }
}

} // namespace pgo::dof
