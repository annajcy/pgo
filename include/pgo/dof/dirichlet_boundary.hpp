#pragma once

#include "pgo/base/assert.hpp"
#include "pgo/dof/dof_layout.hpp"
#include "pgo/math/backend.hpp"
#include "pgo/storage/array_view.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <vector>

namespace pgo::dof {

template <class Boundary, class T>
concept DirichletBoundaryLike = pgo::math::ScalarLike<T> && requires(const Boundary& boundary, const std::size_t dof) {
    { boundary.is_fixed(dof) } -> std::same_as<bool>;
    { boundary.value(dof) } -> std::same_as<T>;
};

template <pgo::math::ScalarLike T>
class DirichletBoundary {
    struct FixedValue {
        std::size_t dof;
        T value;
    };

    std::vector<FixedValue> m_fixed_values;

private:
    [[nodiscard]] auto find_fixed_value(const std::size_t dof) {
        return std::lower_bound(
            m_fixed_values.begin(), m_fixed_values.end(), dof,
            [](const FixedValue& fixed_value, const std::size_t target_dof) { return fixed_value.dof < target_dof; });
    }

    [[nodiscard]] auto find_fixed_value(const std::size_t dof) const {
        return std::lower_bound(
            m_fixed_values.begin(), m_fixed_values.end(), dof,
            [](const FixedValue& fixed_value, const std::size_t target_dof) { return fixed_value.dof < target_dof; });
    }

public:
    void prescribe_dof(const std::size_t dof, const T value) {
        const auto fixed_value = find_fixed_value(dof);
        if (fixed_value != m_fixed_values.end() && fixed_value->dof == dof) {
            fixed_value->value = value;
            return;
        }

        m_fixed_values.insert(fixed_value, FixedValue{dof, value});
    }

    void fix_dof(const std::size_t dof) {
        prescribe_dof(dof, T{0});
    }

    [[nodiscard]] bool is_fixed(const std::size_t dof) const {
        const auto fixed_value = find_fixed_value(dof);
        return fixed_value != m_fixed_values.end() && fixed_value->dof == dof;
    }

    [[nodiscard]] T value(const std::size_t dof) const {
        const auto fixed_value = find_fixed_value(dof);
        pgo::base::require(fixed_value != m_fixed_values.end() && fixed_value->dof == dof, "DOF is not fixed");
        return fixed_value->value;
    }
};

// --- Free functions for vertex-level boundary helpers ---

template <int Dim, class Boundary, pgo::math::ScalarLike T>
void prescribe_component(Boundary& boundary, const DofLayout<Dim>& layout, const std::size_t vertex,
                          const std::size_t component, const T value) {
    boundary.prescribe_dof(layout.index(vertex, component), value);
}

template <int Dim, class Boundary>
void fix_component(Boundary& boundary, const DofLayout<Dim>& layout, const std::size_t vertex,
                   const std::size_t component) {
    boundary.fix_dof(layout.index(vertex, component));
}

template <int Dim, class Boundary, pgo::math::ScalarLike T>
void prescribe_vertex(Boundary& boundary, const DofLayout<Dim>& layout, const std::size_t vertex,
                      const pgo::math::Vec<T, Dim>& value) {
    for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
        prescribe_component<Dim>(boundary, layout, vertex, component, value[pgo::math::dense_index(component)]);
    }
}

template <int Dim, class Boundary>
void fix_vertex(Boundary& boundary, const DofLayout<Dim>& layout, const std::size_t vertex) {
    for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
        fix_component<Dim>(boundary, layout, vertex, component);
    }
}

template <int Dim, class Boundary, pgo::math::ScalarLike T>
void prescribe_vertices(Boundary& boundary, const DofLayout<Dim>& layout,
                        const pgo::storage::ConstArrayView<std::size_t> vertex_indices,
                        const pgo::math::Vec<T, Dim>& value) {
    for (const std::size_t vertex : vertex_indices) {
        prescribe_vertex<Dim>(boundary, layout, vertex, value);
    }
}

template <int Dim, class Boundary>
void fix_vertices(Boundary& boundary, const DofLayout<Dim>& layout,
                  const pgo::storage::ConstArrayView<std::size_t> vertex_indices) {
    for (const std::size_t vertex : vertex_indices) {
        fix_vertex<Dim>(boundary, layout, vertex);
    }
}

template <int Dim, class Boundary, pgo::math::ScalarLike T>
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
