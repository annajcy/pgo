#pragma once

#include "pgo/base/assert.hpp"
#include "pgo/dof/dof_layout.hpp"
#include "pgo/math/backend.hpp"
#include "pgo/storage/array_view.hpp"

#include <cstddef>
#include <span>
#include <unordered_map>

namespace pgo::dof {

template <pgo::math::ScalarLike T>
class DirichletBoundary {
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

    [[nodiscard]] T value(const std::size_t dof) const {
        const auto fixed_value = m_fixed_values.find(dof);
        pgo::base::require(fixed_value != m_fixed_values.end(), "DOF is not fixed");
        return fixed_value->second;
    }

    template <int Dim>
    void prescribe_component(const DofLayout<Dim>& layout, const std::size_t vertex, const std::size_t component,
                             const T value) {
        prescribe_dof(layout.index(vertex, component), value);
    }

    template <int Dim>
    void fix_component(const DofLayout<Dim>& layout, const std::size_t vertex, const std::size_t component) {
        prescribe_component(layout, vertex, component, T{0});
    }

    template <int Dim>
    void prescribe_vertex(const DofLayout<Dim>& layout, const std::size_t vertex, const pgo::math::Vec<T, Dim>& value) {
        for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
            prescribe_component(layout, vertex, component, value[static_cast<Eigen::Index>(component)]);
        }
    }

    template <int Dim>
    void fix_vertex(const DofLayout<Dim>& layout, const std::size_t vertex) {
        for (std::size_t component = 0; component < static_cast<std::size_t>(Dim); ++component) {
            fix_component(layout, vertex, component);
        }
    }

    template <int Dim>
    void prescribe_vertices(const DofLayout<Dim>& layout,
                            const pgo::storage::ConstArrayView<std::size_t> vertex_indices,
                            const pgo::math::Vec<T, Dim>& value) {
        for (const std::size_t vertex : vertex_indices) {
            prescribe_vertex(layout, vertex, value);
        }
    }

    template <int Dim>
    void fix_vertices(const DofLayout<Dim>& layout, const pgo::storage::ConstArrayView<std::size_t> vertex_indices) {
        for (const std::size_t vertex : vertex_indices) {
            fix_vertex(layout, vertex);
        }
    }

    template <int Dim>
    void prescribe_vertices_by_list(const DofLayout<Dim>& layout,
                                    const pgo::storage::ConstArrayView<std::size_t> vertex_indices,
                                    const pgo::storage::ConstArrayView<T> values) {
        const auto dimension = static_cast<std::size_t>(Dim);
        pgo::base::require(values.size() == vertex_indices.size() * dimension,
                           "prescribed vertex values length does not match vertex list");

        for (std::size_t local_vertex = 0; local_vertex < vertex_indices.size(); ++local_vertex) {
            for (std::size_t component = 0; component < dimension; ++component) {
                prescribe_component(layout, vertex_indices[local_vertex], component,
                                    values[local_vertex * dimension + component]);
            }
        }
    }

private:
    std::unordered_map<std::size_t, T> m_fixed_values;
};

} // namespace pgo::dof
